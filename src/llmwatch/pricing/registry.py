"""Per-model pricing mapping registry."""

import json
import logging
from pathlib import Path as PathLibPath

from llmwatch.schemas.pricing import ModelPricing

logger = logging.getLogger("llmwatch")

PRICING_DIR = PathLibPath(__file__).parent.parent / "data"
PRICING_FILE = PRICING_DIR / "pricing.json"


class PricingRegistry:
    """Manages model-to-pricing mappings."""

    def __init__(self, pricing_path: PathLibPath | None = None) -> None:
        self._registry: dict[str, ModelPricing] = {}
        path = pricing_path or PRICING_FILE
        if path.exists():
            self._load_from_file(path)
        # ^ Load supplementary pricing files (pricing_*.json)
        if pricing_path is None:
            for extra in sorted(PRICING_DIR.glob("pricing_*.json")):
                self._load_from_file(extra)

    def _load_from_file(self, path: PathLibPath) -> None:
        with path.open() as f:
            data = json.load(f)
        for entry in data:
            pricing = ModelPricing(**entry)
            key = self._make_key(pricing.model, pricing.provider)
            self._registry[key] = pricing

    def _make_key(self, model: str, provider: str) -> str:
        return f"{provider}:{self._normalize(model)}"

    def _normalize(self, model: str) -> str:
        return model.lower().strip()

    def get_pricing(self, model: str, provider: str = "openai") -> ModelPricing | None:
        # ^ Priority 1: exact match
        key = self._make_key(model, provider)
        if key in self._registry:
            return self._registry[key]
        # ^ Priority 2: requested model starts with a registered model name
        #   e.g. "gpt-4o-2024-08-06" matches "gpt-4o"
        #   Sorted by length desc so "gpt-4" doesn't match "gpt-4o"
        normalized = self._normalize(model)
        candidates = []
        for k, v in self._registry.items():
            registered_model = self._normalize(v.model)
            if v.provider == provider and normalized.startswith(registered_model):
                candidates.append((len(registered_model), v))
        if candidates:
            # ^ Longest match (most specific model) wins
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return None

    def set_pricing(self, model: str, provider: str, pricing: ModelPricing) -> None:
        """Register or override custom pricing."""
        key = self._make_key(model, provider)
        self._registry[key] = pricing

    def list_models(self, provider: str | None = None) -> list[ModelPricing]:
        """Return list of registered models."""
        if provider is None:
            return list(self._registry.values())
        return [p for p in self._registry.values() if p.provider == provider]
