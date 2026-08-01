"""
Google Gemini provider adapter.

Gemini's SDK uses "contents" instead of "messages" and "model"/"user"
roles instead of "assistant"/"user". This adapter absorbs all of that
translation so the gateway's normalized ChatMessage list works unchanged
across every provider.
"""

from typing import AsyncIterator, Optional

from google import genai
from google.genai import types

from google.api_core import exceptions as google_exceptions

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


def _to_gemini_contents(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    """
    Convert OpenAI-style messages into Gemini's (system_instruction, contents) shape.
    Gemini uses role "model" for assistant turns and "user" for user turns.
    """
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
            continue
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    system_instruction = "\n".join(system_parts) if system_parts else None
    return system_instruction, contents


class GeminiProvider(BaseProvider):
    name = "gemini"
    default_model = settings.gemini_default_model
    supports_streaming = True
    supports_json_mode = True

    def __init__(self) -> None:
        self._configured = bool(settings.gemini_api_key)
        if self._configured:
            self.client = genai.Client(api_key=settings.gemini_api_key)

    def is_configured(self) -> bool:
        return self._configured


    async def complete(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format: Optional[dict] = None,
    ) -> ProviderResult:
        system_instruction, contents = _to_gemini_contents(messages)
        json_mode = bool(response_format and response_format.get("type") in ("json_object", "json_schema"))

        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
            )

            if json_mode:
                config.response_mime_type = "application/json"

            response = await self.client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except google_exceptions.Unauthenticated as exc:
            raise ProviderAuthError(self.name, str(exc)) from exc
        except google_exceptions.ResourceExhausted as exc:
            raise ProviderRateLimitError(self.name, str(exc)) from exc
        except google_exceptions.DeadlineExceeded as exc:
            raise ProviderTimeoutError(self.name, str(exc)) from exc
        except google_exceptions.GoogleAPIError as exc:
            raise ProviderError(self.name, str(exc), retryable=True) from exc

        usage = response.usage_metadata
        finish_reason = None
        if response.candidates:
            finish_reason = str(response.candidates[0].finish_reason)

        return ProviderResult(
            content=response.text,
            model=model,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            finish_reason=finish_reason,
            raw_response=None,  # Gemini SDK objects aren't cleanly JSON-serializable.
        )

    async def stream(
        self,
        *,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        system_instruction, contents = _to_gemini_contents(messages)
        gemini_model = self._build_model(model, system_instruction, json_mode=False)

        try:
            response = await self.client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                ),
            )
            async for chunk in response:
                if getattr(chunk, "text", None):
                    yield chunk.text
        except google_exceptions.Unauthenticated as exc:
            raise ProviderAuthError(self.name, str(exc)) from exc
        except google_exceptions.ResourceExhausted as exc:
            raise ProviderRateLimitError(self.name, str(exc)) from exc
        except google_exceptions.DeadlineExceeded as exc:
            raise ProviderTimeoutError(self.name, str(exc)) from exc
        except google_exceptions.GoogleAPIError as exc:
            raise ProviderError(self.name, str(exc), retryable=True) from exc
