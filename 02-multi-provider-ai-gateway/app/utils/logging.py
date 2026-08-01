"""
Logging configuration for the gateway.

Wraps Loguru so the rest of the app can just `from app.utils.logging import logger`
and get a fully configured sink: console output for local dev, a rotating
file sink for durable request logs, and a dedicated JSON-formatted sink
for structured request/response audit records.
"""

import sys
import json
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from app.utils.config import get_settings

settings = get_settings()

_LOG_DIR = Path(settings.log_file_path).parent
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Remove the default handler so we control formatting explicitly.
logger.remove()

# Human-readable console sink.
logger.add(
    sys.stderr,
    level=settings.log_level,
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
)

# Durable rotating file sink (plain text, human-readable).
logger.add(
    settings.log_file_path,
    level=settings.log_level,
    rotation=settings.log_rotation,
    retention=settings.log_retention,
    enqueue=True,
    backtrace=False,
    diagnose=False,
)


def log_request_event(
    *,
    request_id: str,
    provider: str,
    model: str,
    prompt_preview: str,
    status: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int = 0,
    estimated_cost_usd: float = 0.0,
    latency_ms: float = 0.0,
    retry_count: int = 0,
    fallback_used: bool = False,
    error: Optional[str] = None,
) -> None:
    """
    Write a single structured audit record for one gateway request.

    This is intentionally a flat, JSON-serializable dict so downstream
    tooling (dashboards, log shippers, notebooks) can parse it without
    needing to understand Loguru's internal record format.
    """
    record: dict[str, Any] = {
        "request_id": request_id,
        "provider": provider,
        "model": model,
        "prompt_preview": prompt_preview[:200],
        "status": status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "latency_ms": round(latency_ms, 2),
        "retry_count": retry_count,
        "fallback_used": fallback_used,
        "error": error,
    }

    if status == "success":
        logger.bind(audit=True).info(json.dumps(record))
    else:
        logger.bind(audit=True).error(json.dumps(record))
