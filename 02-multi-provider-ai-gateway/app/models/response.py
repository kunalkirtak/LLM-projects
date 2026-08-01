"""
Response schemas.

Every provider adapter normalizes its raw SDK response into these shapes
before it reaches the client, so a caller of POST /chat gets an identical
envelope regardless of whether the request was served by OpenAI,
Anthropic, or Gemini.
"""

from typing import Optional

from pydantic import BaseModel, Field


class UsageMetrics(BaseModel):
    """Token accounting and cost/latency figures for a single request."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    retry_count: int = 0
    fallback_used: bool = False


class ChatResponse(BaseModel):
    """The unified response body returned by POST /chat."""

    request_id: str
    provider: str = Field(..., description="Provider that actually served the request.")
    model: str
    content: str
    finish_reason: Optional[str] = None
    usage: UsageMetrics
    fallback_chain_attempted: list[str] = Field(
        default_factory=list,
        description="Providers attempted, in order, before this one succeeded.",
    )


class HealthStatus(BaseModel):
    """Per-provider readiness for GET /health."""

    provider: str
    configured: bool
    status: str


class HealthResponse(BaseModel):
    status: str
    providers: list[HealthStatus]
    version: str


class ProviderInfo(BaseModel):
    """Static capability description for GET /providers."""

    name: str
    default_model: str
    configured: bool
    supports_streaming: bool
    supports_json_mode: bool


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
    default_provider: str
    fallback_order: list[str]


class ErrorDetail(BaseModel):
    """Structured error payload returned on failure."""

    error_type: str
    message: str
    provider: Optional[str] = None
    retry_count: int = 0
    fallback_chain_attempted: list[str] = Field(default_factory=list)
