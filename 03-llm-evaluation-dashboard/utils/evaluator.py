"""
Core evaluation pipeline: calls the Gemini API, times it, scores the
response, and assembles a fully-populated EvaluationResult.

Uses google-generativeai (the Gemini free-tier-friendly SDK) instead of the
OpenAI SDK. Token counts come straight from the Gemini API's own
count_tokens / usage_metadata, so numbers reported here match what Google
actually billed (or would have billed outside the free tier).
"""

from __future__ import annotations

import time
from typing import Optional

import google.generativeai as genai
from loguru import logger

from models.evaluation import BenchmarkItem, EvaluationResult, PromptVersion
from utils import metrics
from utils.hallucination import detect_hallucination
from utils.pricing import estimate_cost

# Retry policy for transient Gemini API errors (rate limits, brief 5xx blips).
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


class LLMEvaluator:
    """Wraps the Gemini API with retries, timing, and automatic scoring."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        if not api_key:
            raise ValueError("A Google AI Studio API key is required to run evaluations.")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self._model = genai.GenerativeModel(model_name)

    def _generate_with_retries(self, prompt: str) -> tuple[str, int, int, float]:
        """Call Gemini, retrying on transient failures.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens, latency_ms).
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                response = self._model.generate_content(prompt)
                latency_ms = (time.perf_counter() - start) * 1000

                text = (response.text or "").strip()
                usage = getattr(response, "usage_metadata", None)
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0

                return text, input_tokens, output_tokens, latency_ms

            except Exception as exc:  # noqa: BLE001 - Gemini raises several exception types
                last_error = exc
                logger.warning(f"Gemini call failed (attempt {attempt}/{MAX_RETRIES}): {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(f"Gemini API call failed after {MAX_RETRIES} attempts: {last_error}")

    def evaluate(
        self,
        benchmark_item: BenchmarkItem,
        prompt_version: PromptVersion,
    ) -> EvaluationResult:
        """Run one full evaluation: generate, time, score, and package the result.

        Any exception raised while calling the API is caught and turned into
        an EvaluationResult with `error` populated and all scores set to 0,
        so a single failed call never crashes a full benchmark sweep.
        """
        prompt = prompt_version.template.format(question=benchmark_item.question)

        try:
            response_text, input_tokens, output_tokens, latency_ms = self._generate_with_retries(prompt)
            total_tokens = input_tokens + output_tokens
            cost = estimate_cost(self.model_name, input_tokens, output_tokens)

            sim_score = metrics.similarity_score(response_text, benchmark_item.expected_answer)
            kw_score = metrics.keyword_accuracy(response_text, benchmark_item.reference_keywords)
            acc_score = metrics.overall_accuracy(sim_score, kw_score)

            halluc_report = detect_hallucination(
                response_text, benchmark_item.expected_answer, benchmark_item.reference_keywords
            )

            return EvaluationResult(
                benchmark_id=benchmark_item.id,
                question=benchmark_item.question,
                expected_answer=benchmark_item.expected_answer,
                category=benchmark_item.category,
                difficulty=benchmark_item.difficulty,
                prompt_name=prompt_version.name,
                prompt_version=prompt_version.version,
                model_name=self.model_name,
                response=response_text,
                error=None,
                latency_ms=round(latency_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                similarity_score=sim_score,
                keyword_accuracy=kw_score,
                overall_accuracy=acc_score,
                hallucination_score=halluc_report.score,
                hallucination_flags=halluc_report.explanations,
                overall_rating=metrics.rating_from_score(acc_score),
            )

        except Exception as exc:  # noqa: BLE001 - convert any failure into a recorded result
            logger.error(f"Evaluation failed for benchmark_id={benchmark_item.id}: {exc}")
            return EvaluationResult(
                benchmark_id=benchmark_item.id,
                question=benchmark_item.question,
                expected_answer=benchmark_item.expected_answer,
                category=benchmark_item.category,
                difficulty=benchmark_item.difficulty,
                prompt_name=prompt_version.name,
                prompt_version=prompt_version.version,
                model_name=self.model_name,
                response="",
                error=str(exc),
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                similarity_score=0.0,
                keyword_accuracy=0.0,
                overall_accuracy=0.0,
                hallucination_score=100.0,
                hallucination_flags=["Evaluation failed before a response could be scored."],
                overall_rating="Error",
            )
