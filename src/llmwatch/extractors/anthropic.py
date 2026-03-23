"""Anthropic response extraction and streaming collection."""

from collections.abc import AsyncIterator
from typing import Any

from llmwatch.extractors.base import ExtractedResponse, _safe_int
from llmwatch.schemas.usage import TokenUsage


def extract_anthropic(response: Any, provider: str) -> ExtractedResponse:
    model = getattr(response, "model", "unknown")
    usage_obj = getattr(response, "usage", None)

    token_usage = TokenUsage()
    if usage_obj is not None:
        input_tokens = _safe_int(usage_obj, "input_tokens")
        output_tokens = _safe_int(usage_obj, "output_tokens")
        token_usage = TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_creation_input_tokens=_safe_int(usage_obj, "cache_creation_input_tokens"),
            cache_read_input_tokens=_safe_int(usage_obj, "cache_read_input_tokens"),
        )

    output_text = None
    content = getattr(response, "content", None)
    if content and len(content) > 0:
        output_text = getattr(content[0], "text", None)

    return ExtractedResponse(
        provider=provider,
        model=model,
        token_usage=token_usage,
        output_text=output_text,
        response_id=getattr(response, "id", None),
    )


async def collect_anthropic_stream(stream: AsyncIterator[Any]) -> ExtractedResponse:
    chunks: list[str] = []
    model = "unknown"
    response_id = None
    input_tokens = 0
    output_tokens = 0
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0

    async for event in stream:
        event_type = getattr(event, "type", None)
        if event_type == "message_start":
            msg = getattr(event, "message", None)
            if msg:
                model = getattr(msg, "model", model)
                response_id = getattr(msg, "id", response_id)
                u = getattr(msg, "usage", None)
                if u:
                    input_tokens = _safe_int(u, "input_tokens")
                    cache_creation_input_tokens = _safe_int(u, "cache_creation_input_tokens")
                    cache_read_input_tokens = _safe_int(u, "cache_read_input_tokens")
        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if delta:
                text = getattr(delta, "text", None)
                if text:
                    chunks.append(text)
        elif event_type == "message_delta":
            u = getattr(event, "usage", None)
            if u:
                output_tokens = _safe_int(u, "output_tokens")

    return ExtractedResponse(
        provider="anthropic",
        model=model,
        token_usage=TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
        output_text="".join(chunks),
        response_id=response_id,
    )
