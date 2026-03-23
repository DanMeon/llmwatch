"""SDK client instrumentation for automatic LLM call tracking."""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("llmwatch")

if TYPE_CHECKING:
    from llmwatch.tracker import LLMWatch


def detect_client_type(client: Any) -> str:
    """Auto-detect SDK client type from its module path."""
    module = type(client).__module__
    if module.startswith("openai"):
        return "openai"
    if module.startswith("anthropic"):
        return "anthropic"
    if module.startswith("google"):
        return "google"
    raise ValueError(f"Unsupported client type: {type(client).__name__} (module: {module})")


def _is_async_client(client: Any, method_path: str) -> bool:
    """Detect whether a client method is async."""
    obj = client
    for attr in method_path.split("."):
        obj = getattr(obj, attr)
    return asyncio.iscoroutinefunction(obj)


# * Async instrumentors


def instrument_openai_async(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch AsyncOpenAI client to automatically track all chat completion calls."""
    original_create = client.chat.completions.create

    async def patched_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = await original_create(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        await tracker._safe_record(response, provider="openai", latency_ms=elapsed_ms)
        return response

    client.chat.completions.create = patched_create


def instrument_anthropic_async(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch AsyncAnthropic client to automatically track all message calls."""
    original_create = client.messages.create

    async def patched_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = await original_create(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        await tracker._safe_record(response, provider="anthropic", latency_ms=elapsed_ms)
        return response

    client.messages.create = patched_create


# * Sync instrumentors


def instrument_openai_sync(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch sync OpenAI client to automatically track all chat completion calls."""
    original_create = client.chat.completions.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_create(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        tracker._safe_record_sync(response, provider="openai", latency_ms=elapsed_ms)
        return response

    client.chat.completions.create = patched_create


def instrument_anthropic_sync(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch sync Anthropic client to automatically track all message calls."""
    original_create = client.messages.create

    def patched_create(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_create(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        tracker._safe_record_sync(response, provider="anthropic", latency_ms=elapsed_ms)
        return response

    client.messages.create = patched_create


# * Google Gen AI instrumentors


def instrument_google_async(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch async Google Gen AI client to automatically track all generate_content calls."""
    original_generate = client.models.generate_content

    async def patched_generate(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = await original_generate(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        await tracker._safe_record(response, provider="google", latency_ms=elapsed_ms)
        return response

    client.models.generate_content = patched_generate


def instrument_google_sync(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch sync Google Gen AI client to automatically track all generate_content calls."""
    original_generate = client.models.generate_content

    def patched_generate(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_generate(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        tracker._safe_record_sync(response, provider="google", latency_ms=elapsed_ms)
        return response

    client.models.generate_content = patched_generate


# * Dispatcher — picks async or sync instrumentor based on client method


def _instrument_openai(client: Any, *, tracker: "LLMWatch") -> None:
    if _is_async_client(client, "chat.completions.create"):
        instrument_openai_async(client, tracker=tracker)
    else:
        instrument_openai_sync(client, tracker=tracker)


def _instrument_anthropic(client: Any, *, tracker: "LLMWatch") -> None:
    if _is_async_client(client, "messages.create"):
        instrument_anthropic_async(client, tracker=tracker)
    else:
        instrument_anthropic_sync(client, tracker=tracker)


def _instrument_google(client: Any, *, tracker: "LLMWatch") -> None:
    if _is_async_client(client, "models.generate_content"):
        instrument_google_async(client, tracker=tracker)
    else:
        instrument_google_sync(client, tracker=tracker)


_INSTRUMENTORS: dict[str, Callable[..., None]] = {
    "openai": _instrument_openai,
    "anthropic": _instrument_anthropic,
    "google": _instrument_google,
}
