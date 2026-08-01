"""
API route definitions.

Thin layer: routes validate input via Pydantic (handled automatically by
FastAPI), delegate to the Gateway for actual work, and translate
exceptions into proper HTTP error responses. No business logic lives here.
"""

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.request import ChatRequest
from app.models.response import (
    ChatResponse,
    ErrorDetail,
    HealthResponse,
    HealthStatus,
    ProviderInfo,
    ProvidersResponse,
)
from app.providers.base import ProviderError
from app.providers.gateway import PROVIDER_REGISTRY, gateway
from app.utils.config import get_settings
from app.utils.logging import logger
from app.utils.metrics import metrics_store

router = APIRouter()
settings = get_settings()

APP_VERSION = "1.0.0"


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send a chat completion request to one or more providers",
    responses={502: {"model": ErrorDetail}, 400: {"model": ErrorDetail}},
)
async def chat(request: ChatRequest):
    """
    Route a chat request to the selected provider.

    If `stream=true`, returns a `text/event-stream` response instead of
    the JSON envelope. Streaming responses do not participate in the
    provider fallback chain (see Gateway.chat_stream docstring).
    """
    if request.stream:
        return await _stream_chat(request)

    try:
        return await gateway.chat(request)
    except ProviderError as exc:
        logger.error(f"Chat request failed: {exc.message}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=ErrorDetail(
                error_type=type(exc).__name__,
                message=exc.message,
                provider=exc.provider,
            ).model_dump(),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


async def _stream_chat(request: ChatRequest) -> StreamingResponse:
    async def event_generator():
        try:
            async for chunk in gateway.chat_stream(request):
                payload = json.dumps({"delta": chunk})
                yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except ProviderError as exc:
            error_payload = json.dumps({"error": exc.message, "provider": exc.provider})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/health", response_model=HealthResponse, summary="Service and provider health check")
async def health():
    """Report gateway liveness and whether each provider has valid credentials configured."""
    provider_statuses = [
        HealthStatus(
            provider=name,
            configured=provider.is_configured(),
            status="ready" if provider.is_configured() else "not_configured",
        )
        for name, provider in PROVIDER_REGISTRY.items()
    ]

    any_ready = any(p.configured for p in provider_statuses)

    return HealthResponse(
        status="healthy" if any_ready else "degraded",
        providers=provider_statuses,
        version=APP_VERSION,
    )


@router.get("/providers", response_model=ProvidersResponse, summary="List available providers and their capabilities")
async def list_providers():
    """Describe every registered provider: default model, config status, and capabilities."""
    providers = [
        ProviderInfo(
            name=name,
            default_model=provider.default_model,
            configured=provider.is_configured(),
            supports_streaming=provider.supports_streaming,
            supports_json_mode=provider.supports_json_mode,
        )
        for name, provider in PROVIDER_REGISTRY.items()
    ]

    return ProvidersResponse(
        providers=providers,
        default_provider=settings.default_provider,
        fallback_order=settings.fallback_chain,
    )


@router.get("/metrics", summary="Aggregated usage metrics across all providers")
async def get_metrics():
    """Return running totals: requests, tokens, cost, latency, retries, and fallbacks per provider."""
    return metrics_store.snapshot()
