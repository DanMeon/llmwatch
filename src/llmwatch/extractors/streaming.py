"""Streaming response collection — dispatches to provider-specific collectors."""

from collections.abc import AsyncIterator
from typing import Any

from llmwatch.extractors.base import ExtractedResponse


async def collect_stream(
    stream: AsyncIterator[Any],
    *,
    provider: str | None = None,
) -> ExtractedResponse:
    """Collect an async streaming response and return an ExtractedResponse."""
    if provider is None:
        provider = _detect_stream_provider(stream)

    if provider == "openai":
        from llmwatch.extractors.openai import collect_openai_stream

        return await collect_openai_stream(stream, provider=provider)
    if provider == "anthropic":
        from llmwatch.extractors.anthropic import collect_anthropic_stream

        return await collect_anthropic_stream(stream)
    if provider == "google":
        from llmwatch.extractors.google import collect_google_stream

        return await collect_google_stream(stream)
    return await _collect_generic_stream(stream)


def _detect_stream_provider(stream: Any) -> str:
    module = type(stream).__module__
    if "openai" in module:
        return "openai"
    if "anthropic" in module:
        return "anthropic"
    if "google" in module:
        return "google"
    return "unknown"


async def _collect_generic_stream(stream: AsyncIterator[Any]) -> ExtractedResponse:
    chunks: list[str] = []
    async for chunk in stream:
        if isinstance(chunk, str):
            chunks.append(chunk)
        elif hasattr(chunk, "text"):
            chunks.append(chunk.text)
    return ExtractedResponse(
        provider="unknown",
        model="unknown",
        output_text="".join(chunks),
    )
