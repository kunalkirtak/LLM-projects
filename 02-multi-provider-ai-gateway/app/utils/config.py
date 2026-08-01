"""
Application configuration.

Centralizes all environment-driven settings so the rest of the codebase
never touches os.environ directly. Uses pydantic-settings so values are
validated and typed at startup, failing fast if something required is
missing rather than blowing up mid-request.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider credentials -------------------------------------------------
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    gemini_api_key: Optional[str] = Field(default=None)

    # --- Default models per provider -------------------------------------------
    openai_default_model: str = Field(default="gpt-4o-mini")
    anthropic_default_model: str = Field(default="claude-3-5-sonnet-20241022")
    gemini_default_model: str = Field(default="gemini-flash-latest")
    # --- Gateway behavior --------------------------------------------------
    default_provider: str = Field(default="openai")
    fallback_order: str = Field(
        default="openai,anthropic,gemini",
        description="Comma-separated provider names, in the order to try on fallback.",
    )
    enable_fallback: bool = Field(default=True)

    # --- Retry / resilience --------------------------------------------------
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_min_wait_seconds: float = Field(default=1.0)
    retry_max_wait_seconds: float = Field(default=10.0)
    request_timeout_seconds: float = Field(default=60.0)

    # --- Logging --------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_file_path: str = Field(default="logs/requests.log")
    log_rotation: str = Field(default="10 MB")
    log_retention: str = Field(default="14 days")

    # --- Server -----------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    environment: str = Field(default="development")

    @property
    def fallback_chain(self) -> list[str]:
        """Parsed, order-preserving list of provider names for fallback."""
        return [name.strip() for name in self.fallback_order.split(",") if name.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (avoids re-parsing .env per call)."""
    return Settings()
