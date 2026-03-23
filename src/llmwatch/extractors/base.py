"""Core extraction types and dispatch logic."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llmwatch.schemas.usage import TokenUsage


@dataclass
class ExtractedResponse:
    """Normalized internal representation of provider-specific responses."""

    provider: str = "unknown"
    model: str = "unknown"
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    output_text: str | None = None
    response_id: str | None = None


def _safe_int(obj: Any, attr: str) -> int:
    val = getattr(obj, attr, None)
    return val if isinstance(val, int) else 0


def detect_provider(response: Any) -> str:
    module = type(response).__module__
    if module.startswith("openai"):
        return "openai"
    if module.startswith("anthropic"):
        return "anthropic"
    if module.startswith("google"):
        return "google"
    return "unknown"


def extract(response: Any, provider: str | None = None) -> ExtractedResponse:
    """Extract a normalized ExtractedResponse from a raw response."""
    from llmwatch.extractors.anthropic import extract_anthropic
    from llmwatch.extractors.google import extract_generic, extract_google
    from llmwatch.extractors.openai import extract_openai

    if provider is None:
        provider = detect_provider(response)

    _extractors: dict[str, Callable[[Any, str], ExtractedResponse]] = {
        "openai": extract_openai,
        "anthropic": extract_anthropic,
        "google": extract_google,
    }
    extractor = _extractors.get(provider, extract_generic)
    return extractor(response, provider)
