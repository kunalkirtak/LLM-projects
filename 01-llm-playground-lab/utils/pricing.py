"""
pricing.py
----------
Central source of truth for model pricing.

Keeping prices in one dedicated module (rather than scattered across the
app) means a price change or a new model release only needs one edit.
Prices are expressed in USD per 1,000 tokens, matching how most LLM
providers publish their rate cards.

NOTE: Prices are illustrative and should be updated to match the provider's
current published rates before relying on this for real budgeting.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-1K-token pricing for a single model."""

    input_price_per_1k: float
    output_price_per_1k: float


# Price table keyed by the exact model identifier used in API calls.
# Add a new row here whenever a new model should be selectable in the UI.
PRICING_TABLE: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(input_price_per_1k=0.0050, output_price_per_1k=0.0150),
    "gpt-4o-mini": ModelPricing(input_price_per_1k=0.00015, output_price_per_1k=0.00060),
    "gpt-4-turbo": ModelPricing(input_price_per_1k=0.0100, output_price_per_1k=0.0300),
    "gpt-3.5-turbo": ModelPricing(input_price_per_1k=0.0005, output_price_per_1k=0.0015),
}

# Fallback used when a model isn't in the table yet, so the app never
# crashes on an unknown model -- it just reports zero cost with a warning
# surfaced by the caller.
_DEFAULT_PRICING = ModelPricing(input_price_per_1k=0.0, output_price_per_1k=0.0)


def get_model_pricing(model: str) -> ModelPricing:
    """Return the pricing entry for a model, falling back to zero-cost."""
    return PRICING_TABLE.get(model, _DEFAULT_PRICING)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the USD cost of a single completion.

    Args:
        model: Model identifier (e.g. "gpt-4o-mini").
        input_tokens: Number of prompt tokens consumed.
        output_tokens: Number of completion tokens generated.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places.
    """
    pricing = get_model_pricing(model)
    input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
    output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
    return round(input_cost + output_cost, 6)


def available_models() -> list[str]:
    """Return the list of model identifiers with known pricing."""
    return list(PRICING_TABLE.keys())
