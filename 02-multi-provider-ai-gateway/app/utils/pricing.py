"""
Cost estimation utilities.

Providers bill per 1,000 or per 1,000,000 tokens depending on the model.
This module keeps a local pricing table (USD per 1M tokens, input/output
split) so the gateway can attach an estimated cost to every response
without calling out to a billing API. Prices are approximate and meant
for relative comparison across providers, not as an invoice source of
truth -- they should be reviewed against each provider's published
pricing page periodically.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """USD price per 1,000,000 tokens, input and output priced separately."""

    input_per_million: float
    output_per_million: float


# Pricing snapshot, USD per 1M tokens. Update as providers change pricing.
PRICING_TABLE: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(input_per_million=2.50, output_per_million=10.00),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.60),
    "gpt-4-turbo": ModelPricing(input_per_million=10.00, output_per_million=30.00),
    "gpt-3.5-turbo": ModelPricing(input_per_million=0.50, output_per_million=1.50),
    # Anthropic
    "claude-3-5-sonnet-20241022": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-3-5-haiku-20241022": ModelPricing(input_per_million=0.80, output_per_million=4.00),
    "claude-3-opus-20240229": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    # Google Gemini
    "gemini-1.5-flash": ModelPricing(input_per_million=0.075, output_per_million=0.30),
    "gemini-1.5-pro": ModelPricing(input_per_million=1.25, output_per_million=5.00),
}

# Conservative fallback used when a model isn't in the table, so the
# gateway degrades gracefully instead of raising on unknown models.
DEFAULT_PRICING = ModelPricing(input_per_million=1.00, output_per_million=3.00)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the USD cost of a request given token counts.

    Falls back to DEFAULT_PRICING for models not present in PRICING_TABLE
    so callers never have to special-case unknown/new models.
    """
    pricing = PRICING_TABLE.get(model, DEFAULT_PRICING)
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return input_cost + output_cost
