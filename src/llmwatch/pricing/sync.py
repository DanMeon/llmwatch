"""Sync pricing data from pydantic/genai-prices."""

import json
import logging
from datetime import date
from pathlib import Path as PathLibPath
from typing import Any

from llmwatch.pricing.registry import PRICING_FILE

logger = logging.getLogger("llmwatch")

_GENAI_PRICES_URL = (
    "https://raw.githubusercontent.com/pydantic/genai-prices/main/prices/data_slim.json"
)

_SKIP_PROVIDERS = frozenset(
    {
        "avian",
        "huggingface_cerebras",
        "huggingface_fireworks-ai",
        "huggingface_groq",
        "huggingface_hyperbolic",
        "huggingface_nebius",
        "huggingface_novita",
        "huggingface_nscale",
        "huggingface_ovhcloud",
        "huggingface_publicai",
        "huggingface_sambanova",
        "huggingface_together",
    }
)


def _resolve_price_value(field: Any) -> float | None:
    """Extract base price from a simple number or tiered pricing object."""
    if field is None:
        return None
    if isinstance(field, (int, float)):
        return float(field)
    # ^ TieredPrices: {"base": 3, "tiers": [{"start": 200000, "price": 6}]}
    if isinstance(field, dict) and "base" in field:
        return float(field["base"])
    return None


def _resolve_conditional_prices(prices: Any, today: date) -> dict[str, Any] | None:
    """Resolve conditional (date/time-based) pricing to a ModelPrice dict."""
    if isinstance(prices, dict):
        return prices
    # ^ ConditionalPrice[]: list of {constraint?, prices}
    if isinstance(prices, list):
        if not prices:
            return None
        active = prices[0].get("prices")
        for entry in prices:
            constraint = entry.get("constraint")
            if constraint is None:
                active = entry.get("prices")
            elif "start_date" in constraint:
                if today >= date.fromisoformat(constraint["start_date"]):
                    active = entry.get("prices")
        return active  # type: ignore[no-any-return]
    return None


def sync_pricing(
    output_path: PathLibPath | None = None,
    source_url: str = _GENAI_PRICES_URL,
) -> list[dict[str, Any]]:
    """Fetch latest pricing from pydantic/genai-prices and write to pricing.json.

    Returns the list of pricing entries written.
    """
    import urllib.request

    output_path = output_path or PRICING_FILE
    today = date.today()

    # * Fetch remote data
    with urllib.request.urlopen(source_url, timeout=30) as resp:
        raw = json.loads(resp.read().decode())

    entries: list[dict[str, Any]] = []

    for provider_data in raw:
        provider_id = provider_data.get("id", "")
        if provider_id in _SKIP_PROVIDERS:
            continue

        for model_data in provider_data.get("models", []):
            if model_data.get("deprecated", False):
                continue

            model_id = model_data["id"]
            prices_raw = model_data.get("prices")
            if prices_raw is None:
                continue

            prices = _resolve_conditional_prices(prices_raw, today)
            if prices is None:
                continue

            input_price = _resolve_price_value(prices.get("input_mtok"))
            output_price = _resolve_price_value(prices.get("output_mtok"))

            # ^ Skip models with no input/output pricing (e.g. request-only pricing)
            if input_price is None and output_price is None:
                continue

            entry: dict[str, Any] = {
                "model": model_id,
                "provider": provider_id,
                "input_cost_per_mtok": input_price or 0.0,
                "output_cost_per_mtok": output_price or 0.0,
            }

            cache_write = _resolve_price_value(prices.get("cache_write_mtok"))
            if cache_write:
                entry["cache_creation_cost_per_mtok"] = cache_write

            cache_read = _resolve_price_value(prices.get("cache_read_mtok"))
            if cache_read:
                entry["cache_read_cost_per_mtok"] = cache_read

            entries.append(entry)

    # * Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(entries, f, indent=2)
        f.write("\n")

    logger.info("Synced %d models from %s", len(entries), source_url)
    return entries
