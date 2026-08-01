"""
Anthropic provider adapter.

Anthropic's Messages API splits the system prompt out of the message
list and uses its own error hierarchy -- both are normalized here so
the rest of the gateway never has to know the difference.
"""

from typing import AsyncIterator, Optional

import anthropic
from anthropic import AsyncAnthropic

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


def _split_system_prompt(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    """
    Anthropic wants the system prompt as a separate top-level parameter,
    not as a message with role "system". Pull it out and concatenate if
    there happen to be multiple system messages.
    """
    system_parts: list[str] = []
    remaining: list[dict] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            remaining.append(message)
    system_prompt = "\n".join(system_parts) if system_parts else None
    return system_prompt, remaining


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    default_model = settings.anthropic_default_model
    supports_streaming = True
    supports_json_mode = False  # No native JSON mode; handled via prompting.

    def __init__(self) -> None:
        self._client: Optional[AsyncAnthropic] = None
        if settings.anthropic_api_key:
            self._client = AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.request_timeout_seconds,
            )

    def is_configured(self) -> bool:
        return self._client is not None

    def _require_client(self) -> AsyncAnthropic:
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
        system_prompt, chat_messages = _split_system_prompt(messages)

        # Anthropic has no native JSON mode: emulate it with an instruction.
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            instruction = "Respond with valid JSON only, no prose, no markdown fences."
            system_prompt = f"{system_prompt}\n{instruction}" if system_prompt else instruction

        try:
            response = await client.messages.create(
                model=model,
                system=system_prompt or anthropic.NOT_GIVEN,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(self.name, str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(self.name, str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(self.name, str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(self.name, str(exc), retryable=exc.status_code >= 500) from exc

        text_content = "".join(block.text for block in response.content if block.type == "text")

        return ProviderResult(
            content=text_content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason,
            raw_response=response.model_dump(),
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
        system_prompt, chat_messages = _split_system_prompt(messages)

        try:
            async with client.messages.stream(
                model=model,
                system=system_prompt or anthropic.NOT_GIVEN,
                messages=chat_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except anthropic.AuthenticationError as exc:
            raise ProviderAuthError(self.name, str(exc)) from exc
        except anthropic.RateLimitError as exc:
            raise ProviderRateLimitError(self.name, str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError(self.name, str(exc)) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(self.name, str(exc), retryable=exc.status_code >= 500) from exc
