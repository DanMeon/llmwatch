"""Cost calculation engine."""

from llmwatch.schemas.pricing import ModelPricing
from llmwatch.schemas.usage import TokenUsage

_PER_MTOK = 1_000_000


def calculate_cost(usage: TokenUsage, pricing: ModelPricing) -> float:
    """Calculate USD cost from token usage and pricing info."""
    cost = (
        usage.prompt_tokens * pricing.input_cost_per_mtok
        + usage.completion_tokens * pricing.output_cost_per_mtok
        + usage.cache_creation_input_tokens * pricing.cache_creation_cost_per_mtok
        + usage.cache_read_input_tokens * pricing.cache_read_cost_per_mtok
    ) / _PER_MTOK
    return round(cost, 10)
