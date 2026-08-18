"""
Lightweight, dependency-free hallucination heuristic.

This is intentionally NOT a claim-by-claim fact checker (that would require
a second LLM call and its own evaluation problem). Instead it combines three
cheap, explainable signals that correlate reasonably well with hallucinated
or unsupported answers:

1. Missing required keywords -> the response likely omits grounded facts.
2. Low semantic similarity to the expected answer -> the response likely
   drifted from the reference material.
3. Numeric/date claims in the response that do not appear anywhere in the
   expected answer -> a common signature of fabricated specifics
   (invented statistics, invented dates, invented quantities).

The final score is a weighted blend, expressed as 0 (fully grounded) to
100 (fully hallucinated).
"""

from __future__ import annotations

import re
from typing import List

from models.evaluation import HallucinationReport
from utils.metrics import missing_keywords, similarity_score

_NUMBER_PATTERN = re.compile(r"\b\d[\d,.]*%?\b")


def _extract_numbers(text: str) -> List[str]:
    return _NUMBER_PATTERN.findall(text)


def _unsupported_numbers(response: str, expected_answer: str) -> List[str]:
    """Find numeric tokens in the response that don't appear in the expected answer."""
    response_numbers = set(_extract_numbers(response))
    expected_numbers = set(_extract_numbers(expected_answer))
    return sorted(response_numbers - expected_numbers)


def detect_hallucination(
    response: str,
    expected_answer: str,
    reference_keywords: List[str],
) -> HallucinationReport:
    """Estimate a hallucination score and produce human-readable explanations.

    Args:
        response: The model-generated response.
        expected_answer: The reference/expected answer from the benchmark.
        reference_keywords: Keywords the response is expected to mention.

    Returns:
        A HallucinationReport with a 0-100 score and supporting explanations.
    """
    explanations: List[str] = []

    missing = missing_keywords(response, reference_keywords)
    missing_ratio = len(missing) / len(reference_keywords) if reference_keywords else 0.0
    if missing:
        explanations.append(
            f"Response omits {len(missing)}/{len(reference_keywords)} expected keyword(s): "
            f"{', '.join(missing)}."
        )

    sim = similarity_score(response, expected_answer)
    low_similarity_penalty = max(0.0, (60 - sim) / 60) if sim < 60 else 0.0
    if sim < 40:
        explanations.append(f"Response has low semantic overlap with the expected answer (similarity={sim:.1f}).")

    unsupported = _unsupported_numbers(response, expected_answer)
    unsupported_penalty = min(1.0, len(unsupported) * 0.25)
    if unsupported:
        explanations.append(
            f"Response introduces numeric value(s) not present in the expected answer: {', '.join(unsupported)}."
        )

    # Weighted blend: missing keywords and low similarity are the strongest
    # signals; unsupported numbers is a secondary corroborating signal.
    score = (
        missing_ratio * 45
        + low_similarity_penalty * 35
        + unsupported_penalty * 20
    )
    score = round(min(100.0, score), 2)

    if not explanations:
        explanations.append("No hallucination signals detected; response appears well grounded.")

    return HallucinationReport(
        score=score,
        missing_keywords=missing,
        unsupported_statements=unsupported,
        explanations=explanations,
    )
