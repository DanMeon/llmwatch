"""VoyageAI reranker response extraction."""

from typing import Any

from llmwatch.extractors.base import ExtractedResponse
from llmwatch.schemas.usage import TokenUsage


def extract_voyageai(response: Any, provider: str) -> ExtractedResponse:
    """Extract usage from a VoyageAI rerank response (RerankingObject)."""
    # ^ VoyageAI exposes total_tokens directly on the response object (not under .usage)
    total_tokens = getattr(response, "total_tokens", 0) or 0
    model = getattr(response, "model", None) or "unknown"

    return ExtractedResponse(
        provider=provider,
        model=model,
        token_usage=TokenUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )
