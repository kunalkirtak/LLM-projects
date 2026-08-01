"""
metrics.py
----------
Token counting and lightweight performance measurement helpers.

Token counting uses tiktoken when an exact encoding is available for the
requested model, and falls back to a conservative whitespace-based estimate
otherwise. This keeps the app functional even for models tiktoken doesn't
recognize yet.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import tiktoken


@dataclass
class TokenUsage:
    """Container for a single request's token accounting."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _get_encoding(model: str) -> "tiktoken.Encoding":
    """
    Resolve the correct tiktoken encoding for a model.

    Falls back to the general-purpose cl100k_base encoding for models
    tiktoken doesn't have a direct mapping for (e.g. very new releases).
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str) -> int:
    """
    Count the number of tokens a string would consume for a given model.

    Args:
        text: The raw text to tokenize.
        model: Model identifier, used to select the correct tokenizer.

    Returns:
        Token count as an integer. Returns 0 for empty input.
    """
    if not text:
        return 0
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


def count_prompt_tokens(system_prompt: str, user_prompt: str, model: str) -> int:
    """
    Count combined tokens for a system + user prompt pair.

    A small fixed overhead (per-message formatting tokens used by chat
    models) is added to keep the estimate close to what the API actually
    bills, mirroring OpenAI's documented chat-completion token counting
    guidance.
    """
    overhead_per_message = 4  # role/name/separator tokens per message
    system_tokens = count_tokens(system_prompt, model) + overhead_per_message
    user_tokens = count_tokens(user_prompt, model) + overhead_per_message
    return system_tokens + user_tokens


@contextmanager
def measure_latency() -> Iterator[dict]:
    """
    Context manager that measures wall-clock latency of the wrapped block.

    Usage:
        with measure_latency() as timer:
            do_work()
        elapsed_seconds = timer["elapsed"]

    The dict is mutated in place so the caller can read the result
    immediately after the `with` block exits.
    """
    timer_result: dict = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield timer_result
    finally:
        timer_result["elapsed"] = round(time.perf_counter() - start, 3)
