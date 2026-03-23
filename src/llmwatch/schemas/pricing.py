"""Per-model pricing schema."""

from pydantic import BaseModel


class ModelPricing(BaseModel):
    """Per-model pricing info (USD per million tokens)."""

    model: str
    provider: str
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    cache_creation_cost_per_mtok: float = 0.0
    cache_read_cost_per_mtok: float = 0.0
