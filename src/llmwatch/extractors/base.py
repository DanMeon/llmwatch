"""Core extraction types, dispatch logic, and provider registry."""

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

from llmwatch.schemas.usage import TokenUsage

# * Type alias for extractor functions
ExtractorFn = Callable[[Any, str], "ExtractedResponse"]

# * Provider registry — single source of truth for module-prefix detection,
# * extractor dispatch, and instrumentor dispatch.
_EXTRACTOR_REGISTRY: dict[str, ExtractorFn | str] = {}
_MODULE_PREFIXES: dict[str, str] = {}  # module prefix -> provider name


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


def register_extractor(
    provider: str,
    extractor: ExtractorFn | str,
    *,
    module_prefix: str | None = None,
) -> None:
    """Register a provider extractor.

    Args:
        provider: Provider name (e.g. "cohere").
        extractor: Extractor function, or a "module:function" string for lazy import.
        module_prefix: Module prefix for auto-detection (defaults to provider name).
            Set to a custom value when the SDK module differs from the provider name.

    Example — internal (lazy import)::

        register_extractor("cohere", "llmwatch.extractors.cohere:extract_cohere")

    Example — external (user-defined)::

        register_extractor("my_llm", my_extract_fn, module_prefix="my_llm_sdk")
    """
    _EXTRACTOR_REGISTRY[provider] = extractor
    _MODULE_PREFIXES[module_prefix or provider] = provider


def _resolve(ref: ExtractorFn | str) -> ExtractorFn:
    """Resolve an extractor — return as-is if callable, lazy-import if string."""
    if callable(ref):
        return ref
    module_path, func_name = ref.rsplit(":", 1)
    return getattr(import_module(module_path), func_name)  # type: ignore[no-any-return]


def detect_provider(response: Any) -> str:
    """Auto-detect provider from a response object's module prefix."""
    module = type(response).__module__
    for prefix, provider in _MODULE_PREFIXES.items():
        if module.startswith(prefix):
            return provider
    return "unknown"


def extract(response: Any, provider: str | None = None) -> ExtractedResponse:
    """Extract a normalized ExtractedResponse from a raw response."""
    from llmwatch.extractors.google import extract_generic

    if provider is None:
        provider = detect_provider(response)

    ref = _EXTRACTOR_REGISTRY.get(provider)
    extractor = _resolve(ref) if ref else extract_generic
    return extractor(response, provider)


# * Built-in provider registrations (lazy imports to avoid circular dependencies)
register_extractor("openai", "llmwatch.extractors.openai:extract_openai")
register_extractor("anthropic", "llmwatch.extractors.anthropic:extract_anthropic")
register_extractor("google", "llmwatch.extractors.google:extract_google")
register_extractor("cohere", "llmwatch.extractors.cohere:extract_cohere")
register_extractor("voyageai", "llmwatch.extractors.voyageai:extract_voyageai")
