"""
Cost estimation for Gemini models.

The Gemini API free tier (Google AI Studio) does not charge per token, so
by default this module reports $0.00 for every call. The paid-tier price
table is still included so the dashboard keeps working (and stays useful)
if you later switch `FREE_TIER_MODE=false` and move to a billed project.

Prices are per 1,000,000 tokens and were accurate as of the Gemini pricing
page at the time this project was written. Re-check https://ai.google.dev/pricing
before relying on these numbers for real budgeting.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# (input_price_per_million, output_price_per_million) in USD, paid tier.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.0-pro": (0.50, 1.50),
}

DEFAULT_PRICING = (0.075, 0.30)  # fall back to flash pricing for unknown models


def is_free_tier() -> bool:
    """Whether cost estimation should be forced to zero (Gemini free tier)."""
    return os.getenv("FREE_TIER_MODE", "true").strip().lower() in {"1", "true", "yes"}


def estimate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a single generation call.

    Args:
        model_name: The Gemini model identifier (e.g. "gemini-1.5-flash").
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.

    Returns:
        Estimated cost in USD. Always 0.0 when running in free-tier mode.
    """
    if is_free_tier():
        return 0.0

    input_price, output_price = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return round(cost, 6)
