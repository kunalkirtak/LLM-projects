"""
llm.py
------
Thin abstraction layer over LLM provider SDKs.

Design intent
-------------
Every provider-specific detail (SDK client construction, request shape,
response parsing) is isolated behind `generate_completion()`. The rest of
the app only ever calls that one function and works with the plain
`CompletionResult` dataclass it returns.

To add a new provider (Gemini, Anthropic, etc.) later:
    1. Write a `_call_<provider>()` function following the same signature
       pattern as `_call_openai()`.
    2. Add a branch for it in `generate_completion()`.
No other file in the project needs to change.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

from utils.metrics import TokenUsage, count_prompt_tokens, measure_latency
from utils.pricing import estimate_cost

load_dotenv()


class LLMRequestError(Exception):
    """Raised when a provider call fails for a reason the UI should surface."""


@dataclass
class CompletionResult:
    """Everything the UI needs to render one generated response."""

    text: str
    model: str
    token_usage: TokenUsage
    estimated_cost: float
    latency_seconds: float


def _get_openai_client() -> OpenAI:
    """
    Build an OpenAI client from the API key in the environment.

    Raises:
        LLMRequestError: if OPENAI_API_KEY is not set, so the caller can
            show a friendly message instead of a raw stack trace.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMRequestError(
            "OPENAI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before generating a response."
        )
    return OpenAI(api_key=api_key)


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[str, int, int]:
    """
    Send a chat completion request to OpenAI.

    Returns:
        Tuple of (response_text, input_tokens, output_tokens) using the
        provider's own reported usage figures, which are authoritative
        over any local token estimate.
    """
    client = _get_openai_client()

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    except AuthenticationError as exc:
        raise LLMRequestError("Authentication failed. Check your OPENAI_API_KEY.") from exc
    except Exception as e:
        print(e)
        raise
    except APIError as exc:
        raise LLMRequestError(f"OpenAI API returned an error: {exc}") from exc

    response_text = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    return response_text, input_tokens, output_tokens


def generate_completion(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    provider: str = "openai",
) -> CompletionResult:
    """
    Generate a completion from the requested provider and package the
    response, token usage, cost, and latency into one result object.

    Args:
        system_prompt: Optional system-level instruction.
        user_prompt: The user's prompt. Must be non-empty.
        model: Provider-specific model identifier.
        temperature: Sampling temperature (0.0 - 2.0).
        top_p: Nucleus sampling parameter (0.0 - 1.0).
        max_tokens: Maximum tokens to generate.
        provider: Which backend to route to. Only "openai" is implemented
            today; the parameter exists so new providers slot in without
            changing the function signature callers depend on.

    Returns:
        CompletionResult with the generated text and full metrics.

    Raises:
        LLMRequestError: on missing credentials, empty input, or a
            provider-side failure.
    """
    if not user_prompt or not user_prompt.strip():
        raise LLMRequestError("Prompt cannot be empty.")

    with measure_latency() as timer:
        if provider == "openai":
            text, input_tokens, output_tokens = _call_openai(
                system_prompt, user_prompt, model, temperature, top_p, max_tokens
            )
        else:
            # Placeholder branch for future providers (Gemini, Anthropic, ...).
            raise LLMRequestError(f"Unsupported provider: {provider}")

    # Fall back to a local tiktoken estimate if a provider ever omits usage.
    if input_tokens == 0:
        input_tokens = count_prompt_tokens(system_prompt, user_prompt, model)

    token_usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost = estimate_cost(model, input_tokens, output_tokens)

    return CompletionResult(
        text=text,
        model=model,
        token_usage=token_usage,
        estimated_cost=cost,
        latency_seconds=timer["elapsed"],
    )
