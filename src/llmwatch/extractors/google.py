"""Google and generic response extraction and streaming collection."""

from collections.abc import AsyncIterator
from typing import Any

from llmwatch.extractors.base import ExtractedResponse, _safe_int
from llmwatch.schemas.usage import TokenUsage


def extract_google(response: Any, provider: str) -> ExtractedResponse:
    model = getattr(response, "model", "unknown")
    usage_meta = getattr(response, "usage_metadata", None)

    token_usage = TokenUsage()
    if usage_meta is not None:
        token_usage = TokenUsage(
            prompt_tokens=_safe_int(usage_meta, "prompt_token_count"),
            completion_tokens=_safe_int(usage_meta, "candidates_token_count"),
            total_tokens=_safe_int(usage_meta, "total_token_count"),
            cache_read_input_tokens=_safe_int(usage_meta, "cached_content_token_count"),
        )

    return ExtractedResponse(
        provider=provider,
        model=model,
        token_usage=token_usage,
        output_text=getattr(response, "text", None),
        response_id=getattr(response, "id", None),
    )


def extract_generic(response: Any, provider: str) -> ExtractedResponse:
    if isinstance(response, dict):
        usage = response.get("usage", {})
        output_text = None
        choices = response.get("choices", [])
        if choices:
            output_text = choices[0].get("message", {}).get("content")

        return ExtractedResponse(
            provider=provider,
            model=response.get("model", "unknown"),
            token_usage=TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                completion_tokens=usage.get("completion_tokens", 0) or 0,
                total_tokens=usage.get("total_tokens", 0) or 0,
            ),
            output_text=output_text,
        )

    # ^ Object form — attempt OpenAI-compatible structure
    model = getattr(response, "model", "unknown")
    usage_obj = getattr(response, "usage", None)

    token_usage = TokenUsage()
    if usage_obj is not None:
        if isinstance(usage_obj, dict):
            token_usage = TokenUsage(
                prompt_tokens=usage_obj.get("prompt_tokens", 0) or 0,
                completion_tokens=usage_obj.get("completion_tokens", 0) or 0,
                total_tokens=usage_obj.get("total_tokens", 0) or 0,
            )
        else:
            token_usage = TokenUsage(
                prompt_tokens=_safe_int(usage_obj, "prompt_tokens"),
                completion_tokens=_safe_int(usage_obj, "completion_tokens"),
                total_tokens=_safe_int(usage_obj, "total_tokens"),
            )

    return ExtractedResponse(
        provider=provider,
        model=model,
        token_usage=token_usage,
        response_id=getattr(response, "id", None),
    )


async def collect_google_stream(
    stream: AsyncIterator[Any],
    *,
    model: str = "unknown",
) -> ExtractedResponse:
    """Collect a Google Gemini async streaming response into an ExtractedResponse.

    Args:
        stream: Async iterator of Google Gen AI response chunks. Each chunk
            exposes a ``.text`` attribute. The final chunk carries
            ``usage_metadata`` with ``prompt_token_count``,
            ``candidates_token_count``, and ``total_token_count``.
        model: Model name to record when the stream itself does not carry it.

    Returns:
        Aggregated ExtractedResponse with combined text and final token usage.
    """
    chunks: list[str] = []
    token_usage = TokenUsage()

    async for chunk in stream:
        # ^ Collect text from each streamed chunk
        text = getattr(chunk, "text", None)
        if text:
            chunks.append(text)

        # ^ Model name may be present on any chunk
        chunk_model = getattr(chunk, "model", None)
        if chunk_model:
            model = chunk_model

        # ^ usage_metadata is typically populated on the final chunk
        usage_meta = getattr(chunk, "usage_metadata", None)
        if usage_meta is not None:
            token_usage = TokenUsage(
                prompt_tokens=_safe_int(usage_meta, "prompt_token_count"),
                completion_tokens=_safe_int(usage_meta, "candidates_token_count"),
                total_tokens=_safe_int(usage_meta, "total_token_count"),
                cache_read_input_tokens=_safe_int(usage_meta, "cached_content_token_count"),
            )

    return ExtractedResponse(
        provider="google",
        model=model,
        token_usage=token_usage,
        output_text="".join(chunks),
    )
