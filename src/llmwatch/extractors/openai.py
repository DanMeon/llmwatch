"""OpenAI response extraction and streaming collection."""

from collections.abc import AsyncIterator
from typing import Any

from llmwatch.extractors.base import ExtractedResponse, _safe_int
from llmwatch.schemas.usage import TokenUsage


def extract_openai(response: Any, provider: str) -> ExtractedResponse:
    model = getattr(response, "model", "unknown")
    usage_obj = getattr(response, "usage", None)

    token_usage = TokenUsage()
    if usage_obj is not None:
        token_usage = TokenUsage(
            prompt_tokens=_safe_int(usage_obj, "prompt_tokens"),
            completion_tokens=_safe_int(usage_obj, "completion_tokens"),
            total_tokens=_safe_int(usage_obj, "total_tokens"),
        )

    output_text = None
    choices = getattr(response, "choices", None)
    if choices and len(choices) > 0:
        message = getattr(choices[0], "message", None)
        if message:
            output_text = getattr(message, "content", None)

    return ExtractedResponse(
        provider=provider,
        model=model,
        token_usage=token_usage,
        output_text=output_text,
        response_id=getattr(response, "id", None),
    )


async def collect_openai_stream(stream: AsyncIterator[Any]) -> ExtractedResponse:
    chunks: list[str] = []
    model = "unknown"
    response_id = None
    token_usage = TokenUsage()

    async for chunk in stream:
        if hasattr(chunk, "model") and chunk.model:
            model = chunk.model
        if hasattr(chunk, "id") and chunk.id:
            response_id = chunk.id
        if hasattr(chunk, "usage") and chunk.usage is not None:
            token_usage = TokenUsage(
                prompt_tokens=_safe_int(chunk.usage, "prompt_tokens"),
                completion_tokens=_safe_int(chunk.usage, "completion_tokens"),
                total_tokens=_safe_int(chunk.usage, "total_tokens"),
            )
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta:
                content = getattr(delta, "content", None)
                if content:
                    chunks.append(content)

    return ExtractedResponse(
        provider="openai",
        model=model,
        token_usage=token_usage,
        output_text="".join(chunks),
        response_id=response_id,
    )
