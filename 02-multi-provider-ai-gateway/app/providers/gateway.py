"""
Gateway orchestration layer.

This is the only place that knows about *all* providers at once. It:
  1. Selects a provider based on the request.
  2. Retries transient failures with exponential backoff (via tenacity).
  3. Falls back to the next provider in the configured chain if the
     selected one exhausts its retries and fallback is allowed.
  4. Normalizes everything into a single ChatResponse regardless of
     which provider ultimately served the request.

Adding a provider is a two-line change: import the adapter, add it to
PROVIDER_REGISTRY. Nothing else in this file needs to know it exists.
"""

import time
import uuid
from typing import AsyncIterator, Optional

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.models.request import ChatRequest
from app.models.response import ChatResponse, UsageMetrics
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import BaseProvider, ProviderError, ProviderResult
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.utils.config import get_settings
from app.utils.logging import log_request_event, logger
from app.utils.metrics import metrics_store
from app.utils.pricing import estimate_cost

settings = get_settings()

# Central provider registry. This is the single line-of-code you touch
# to wire up a new provider once its adapter class exists.
PROVIDER_REGISTRY: dict[str, BaseProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "gemini": GeminiProvider(),
}


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, ProviderError) and exc.retryable


class RetryCounter:
    """Small mutable box so the tenacity `after` callback can report attempt counts."""

    def __init__(self) -> None:
        self.attempts = 0

    def record(self, retry_state: RetryCallState) -> None:
        self.attempts = retry_state.attempt_number


async def _call_with_retry(
    provider: BaseProvider,
    *,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: Optional[dict],
) -> tuple[ProviderResult, int]:
    """Call provider.complete() with exponential-backoff retry. Returns (result, retry_count)."""
    counter = RetryCounter()

    retrying = AsyncRetrying(
        stop=stop_after_attempt(settings.max_retries + 1),
        wait=wait_exponential(
            multiplier=settings.retry_min_wait_seconds,
            max=settings.retry_max_wait_seconds,
        ),
        retry=retry_if_exception(_is_retryable),
        after=counter.record,
        reraise=True,
    )

    async for attempt in retrying:
        with attempt:
            result = await provider.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

    # attempt_number is 1-indexed; retries = attempts beyond the first.
    return result, max(counter.attempts - 1, 0)


class Gateway:
    """Routes chat requests to providers with retry + fallback semantics."""

    def __init__(self, registry: Optional[dict[str, BaseProvider]] = None) -> None:
        self.registry = registry if registry is not None else PROVIDER_REGISTRY

    def get_provider(self, name: str) -> BaseProvider:
        try:
            return self.registry[name]
        except KeyError:
            raise ValueError(f"Unknown provider '{name}'. Available: {list(self.registry)}") from None

    def _resolve_model(self, provider: BaseProvider, requested_model: Optional[str]) -> str:
        return requested_model or provider.default_model

    def _build_fallback_chain(self, requested_provider: str) -> list[str]:
        """Ordered list of provider names to try, starting with the requested one."""
        chain = [requested_provider]
        for name in settings.fallback_chain:
            if name != requested_provider and name not in chain:
                chain.append(name)
        return chain

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Execute a non-streaming chat completion with retry + fallback."""
        request_id = str(uuid.uuid4())
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        response_format = request.response_format.model_dump() if request.response_format else None

        chain = (
            self._build_fallback_chain(request.provider.value)
            if request.allow_fallback and settings.enable_fallback
            else [request.provider.value]
        )

        attempted: list[str] = []
        last_error: Optional[Exception] = None
        start_time = time.perf_counter()

        for provider_name in chain:
            provider = self.registry.get(provider_name)
            if provider is None or not provider.is_configured():
                logger.warning(f"Skipping provider '{provider_name}': not configured")
                continue

            model = self._resolve_model(provider, request.model if provider_name == request.provider.value else None)
            attempted.append(provider_name)
            attempt_start = time.perf_counter()

            try:
                result, retry_count = await _call_with_retry(
                    provider,
                    messages=messages,
                    model=model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format=response_format,
                )
                latency_ms = (time.perf_counter() - attempt_start) * 1000
                cost = estimate_cost(model, result.input_tokens, result.output_tokens)
                fallback_used = len(attempted) > 1

                metrics_store.record(
                    provider=provider_name,
                    success=True,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    fallback_used=fallback_used,
                )
                log_request_event(
                    request_id=request_id,
                    provider=provider_name,
                    model=model,
                    prompt_preview=messages[-1]["content"] if messages else "",
                    status="success",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    total_tokens=result.total_tokens,
                    estimated_cost_usd=cost,
                    latency_ms=latency_ms,
                    retry_count=retry_count,
                    fallback_used=fallback_used,
                )

                return ChatResponse(
                    request_id=request_id,
                    provider=provider_name,
                    model=result.model,
                    content=result.content,
                    finish_reason=result.finish_reason,
                    usage=UsageMetrics(
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        total_tokens=result.total_tokens,
                        estimated_cost_usd=cost,
                        latency_ms=latency_ms,
                        retry_count=retry_count,
                        fallback_used=fallback_used,
                    ),
                    fallback_chain_attempted=attempted[:-1],
                )

            except ProviderError as exc:
                latency_ms = (time.perf_counter() - attempt_start) * 1000
                last_error = exc
                logger.error(f"Provider '{provider_name}' failed: {exc.message}")
                metrics_store.record(
                    provider=provider_name,
                    success=False,
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=latency_ms,
                    retry_count=settings.max_retries,
                    fallback_used=False,
                )
                log_request_event(
                    request_id=request_id,
                    provider=provider_name,
                    model=model,
                    prompt_preview=messages[-1]["content"] if messages else "",
                    status="failure",
                    latency_ms=latency_ms,
                    error=exc.message,
                )
                continue  # try next provider in the chain

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        error_message = str(last_error) if last_error else "No configured provider available"
        logger.error(
            f"All providers exhausted for request {request_id} after {total_latency_ms:.0f}ms: {error_message}"
        )
        raise ProviderError(
            provider=attempted[-1] if attempted else "none",
            message=f"All providers in fallback chain failed. Attempted: {attempted}. Last error: {error_message}",
            retryable=False,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """
        Execute a streaming chat completion.

        Streaming intentionally does not participate in the fallback chain:
        once bytes have started flowing to the client, switching providers
        mid-stream would produce an incoherent response. It still benefits
        from the provider's own transient-error handling at connection time.
        """
        provider = self.get_provider(request.provider.value)
        if not provider.is_configured():
            raise ProviderError(request.provider.value, "Provider is not configured", retryable=False)

        model = self._resolve_model(provider, request.model)
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]

        async for chunk in provider.stream(
            messages=messages,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        ):
            yield chunk


# Process-wide singleton used by the API routes.
gateway = Gateway()
