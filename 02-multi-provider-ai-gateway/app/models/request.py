"""
Request schemas.

These are the shapes clients send us. Kept deliberately provider-agnostic:
a client never needs to know that Anthropic wants "system" split out of
the message list, or that Gemini calls it "contents" -- that translation
happens inside each provider adapter, not here.
"""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ProviderName(str, Enum):
    """Supported LLM providers. Add new providers here and in the registry."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single turn in the conversation, OpenAI-style chat format."""

    role: ChatRole
    content: str = Field(..., min_length=1, description="Message text content.")


class ResponseFormat(BaseModel):
    """
    Requests a structured JSON output from the model.

    When `type` is "json_object" or "json_schema", providers that support
    native structured outputs will use them; providers that don't will
    fall back to a strong prompt-level instruction to emit valid JSON.
    """

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[dict] = Field(
        default=None,
        description="Required when type == 'json_schema'. A JSON Schema the output must conform to.",
    )


class ChatRequest(BaseModel):
    """The unified request body accepted by POST /chat."""

    provider: ProviderName = Field(
        default=ProviderName.OPENAI,
        description="Preferred provider to route this request to.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Override the provider's default model. Falls back to config default if omitted.",
    )
    messages: list[ChatMessage] = Field(
        ..., min_length=1, description="Conversation history, oldest first."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=32000)
    stream: bool = Field(default=False, description="If true, response is returned as an SSE stream.")
    response_format: Optional[ResponseFormat] = Field(default=None)
    allow_fallback: bool = Field(
        default=True,
        description="If the selected provider fails, allow the gateway to try the next provider in the fallback chain.",
    )

    @field_validator("messages")
    @classmethod
    def must_contain_user_message(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        if not any(m.role == ChatRole.USER for m in messages):
            raise ValueError("messages must include at least one 'user' role message")
        return messages

    class Config:
        json_schema_extra = {
            "example": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "Explain vector databases in two sentences."},
                ],
                "temperature": 0.7,
                "max_tokens": 300,
                "stream": False,
                "allow_fallback": True,
            }
        }
