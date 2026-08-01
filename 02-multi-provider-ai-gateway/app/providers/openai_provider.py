"""
OpenAI provider adapter.

Translates the gateway's normalized message format into OpenAI's Chat
Completions API and normalizes the response back into a ProviderResult.
"""

from typing import AsyncIterator, Optional

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

from app.providers.base import (
    BaseProvider,
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderResult,
    ProviderTimeoutError,
)
from app.utils.config import get_settings

settings = get_settings()


class OpenAIProvider(BaseProvider):
    name = "openai"
    default_model = settings.openai_default_model
    supports_streaming = True
    supports_json_mode = True

    def __init__(self) -> None:
        self._client: Optional[AsyncOpenAI] = None
        if settings.openai_api_key:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=settings.request_timeout_seconds,
            )

    def is_configured(self) -> bool:
        return self._client is not None

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ProviderNotConfiguredError(self.name)
        return self._client

    async def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> ProviderResult:
        client = self._require_client()

        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format and response_format.get("type") == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        try:
            completion = await client.chat.completions.create(**kwargs)
        except AuthenticationError as exc:
            raise ProviderAuthError(self.name, str(exc)) from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError(self.name, str(exc)) from exc
        except APITimeoutError as exc:
            raise ProviderTimeoutError(self.name, str(exc)) from exc
        except APIStatusError as exc:
            raise ProviderError(self.name, str(exc), retryable=exc.status_code >= 500) from exc

        choice = completion.choices[0]
        usage = completion.usage

        return ProviderResult(
            content=choice.message.content or "",
            model=completion.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason,
            raw_response=completion.model_dump(),
        )

    async def stream(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        client = self._require_client()

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except AuthenticationError as exc:
            raise ProviderAuthError(self.name, str(exc)) from exc
        except RateLimitError as exc:
            raise ProviderRateLimitError(self.name, str(exc)) from exc
        except APITimeoutError as exc:
            raise ProviderTimeoutError(self.name, str(exc)) from exc
        except APIStatusError as exc:
            raise ProviderError(self.name, str(exc), retryable=exc.status_code >= 500) from exc
