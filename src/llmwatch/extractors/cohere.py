"""Cohere reranker response extraction."""

from typing import Any

from llmwatch.extractors.base import ExtractedResponse, _safe_int
from llmwatch.schemas.usage import TokenUsage


def extract_cohere(response: Any, provider: str) -> ExtractedResponse:
    """Extract usage from a Cohere rerank response (RerankResponse / V2RerankResponse)."""
    model = getattr(response, "model", None) or "unknown"
    response_id = getattr(response, "id", None)

    prompt_tokens = 0
    meta = getattr(response, "meta", None)
    if meta is not None:
        billed = getattr(meta, "billed_units", None)
        if billed is not None:
            # ^ Cohere reports search_units for rerank; also exposes input_tokens
            prompt_tokens = _safe_int(billed, "input_tokens") or _safe_int(billed, "search_units")
        tokens = getattr(meta, "tokens", None)
        if tokens is not None and prompt_tokens == 0:
            prompt_tokens = _safe_int(tokens, "input_tokens")

    return ExtractedResponse(
        provider=provider,
        model=model,
        token_usage=TokenUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
        response_id=str(response_id) if response_id else None,
    )
