"""
Provider abstraction layer.

Every concrete provider (OpenAI, Anthropic, Gemini, ...) implements the
`BaseProvider` interface defined here. The gateway only ever talks to
this interface -- it never imports a provider SDK directly -- which is
what lets `gateway.py` route, retry, and fall back without knowing
anything about the vendor-specific request/response shapes underneath.

Adding a new provider means:
  1. Subclass BaseProvider.
  2. Implement `complete()` and `stream()`.
  3. Register it in `gateway.py`'s PROVIDER_REGISTRY.
No other file needs to change.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class ProviderResult:
    """Normalized result returned by every provider's complete() call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: Optional[str] = None
    raw_response: Optional[dict] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ProviderError(Exception):
    """Base exception for all provider-level failures."""

    def __init__(self, provider: str, message: str, *, retryable: bool = True) -> None:
        self.provider = provider
        self.message = message
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


class ProviderAuthError(ProviderError):
    """Raised when a provider rejects our credentials. Never retryable."""

    def __init__(self, provider: str, message: str = "Authentication failed") -> None:
        super().__init__(provider, message, retryable=False)


class ProviderRateLimitError(ProviderError):
    """Raised on 429 / rate-limit responses. Retryable with backoff."""

    def __init__(self, provider: str, message: str = "Rate limit exceeded") -> None:
        super().__init__(provider, message, retryable=True)


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call exceeds the configured timeout."""

    def __init__(self, provider: str, message: str = "Request timed out") -> None:
        super().__init__(provider, message, retryable=True)


class ProviderNotConfiguredError(ProviderError):
    """Raised when a provider is selected but has no API key configured."""

    def __init__(self, provider: str) -> None:
        super().__init__(provider, "Provider is not configured (missing API key)", retryable=False)


class BaseProvider(ABC):
    """Common interface every LLM provider adapter must implement."""

    name: str
    default_model: str
    supports_streaming: bool = True
    supports_json_mode: bool = True

    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has the credentials it needs to run."""
        raise NotImplementedError

    @abstractmethod
    async def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> ProviderResult:
        """Perform a single non-streaming completion. Must raise ProviderError subclasses on failure."""
        raise NotImplementedError

    @abstractmethod
    async def stream(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield response text chunks as they arrive from the provider."""
        raise NotImplementedError
        yield ""  # pragma: no cover - makes this an async generator for type checkers
