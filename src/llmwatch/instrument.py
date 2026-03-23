"""SDK client instrumentation for automatic LLM call tracking."""

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("llmwatch")

if TYPE_CHECKING:
    from llmwatch.tracker import LLMWatch


class _TrackedAsyncStream:
    """Async stream wrapper that collects chunks and records usage on completion."""

    def __init__(
        self,
        stream: Any,
        *,
        tracker: "LLMWatch",
        provider: str,
        start: float,
    ) -> None:
        self._stream = stream
        self._tracker = tracker
        self._provider = provider
        self._start = start
        self._recorded = False

    def __aiter__(self) -> "_TrackedAsyncStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return await self._stream.__anext__()
        except StopAsyncIteration:
            if not self._recorded:
                self._recorded = True
                await self._record_from_stream()
            raise

    async def _record_from_stream(self) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        # ^ If the stream exposes a final response (OpenAI stream.get_final_completion()),
        # ^ use it. Otherwise the per-chunk usage was already the best we could capture.
        final = getattr(self._stream, "get_final_completion", None)
        if final and callable(final):
            try:
                response = final()
                await self._tracker._safe_record(
                    response, provider=self._provider, latency_ms=elapsed_ms
                )
                return
            except Exception:
                pass
        # ^ Fallback: record with whatever usage the stream accumulated
        final_response = getattr(self._stream, "response", None)
        if final_response is not None:
            await self._tracker._safe_record(
                final_response, provider=self._provider, latency_ms=elapsed_ms
            )

    async def __aenter__(self) -> "_TrackedAsyncStream":
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if not self._recorded:
            self._recorded = True
            await self._record_from_stream()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _TrackedSyncStream:
    """Sync stream wrapper that collects chunks and records usage on completion."""

    def __init__(
        self,
        stream: Any,
        *,
        tracker: "LLMWatch",
        provider: str,
        start: float,
    ) -> None:
        self._stream = stream
        self._tracker = tracker
        self._provider = provider
        self._start = start
        self._recorded = False

    def __iter__(self) -> "_TrackedSyncStream":
        return self

    def __next__(self) -> Any:
        try:
            return next(self._stream)
        except StopIteration:
            if not self._recorded:
                self._recorded = True
                self._record_from_stream()
            raise

    def _record_from_stream(self) -> None:
        elapsed_ms = (time.monotonic() - self._start) * 1000
        final = getattr(self._stream, "get_final_completion", None)
        if final and callable(final):
            try:
                response = final()
                self._tracker._safe_record_sync(
                    response, provider=self._provider, latency_ms=elapsed_ms
                )
                return
            except Exception:
                pass
        final_response = getattr(self._stream, "response", None)
        if final_response is not None:
            self._tracker._safe_record_sync(
                final_response, provider=self._provider, latency_ms=elapsed_ms
            )

    def __enter__(self) -> "_TrackedSyncStream":
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if not self._recorded:
            self._recorded = True
            self._record_from_stream()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_async_stream(
    stream: Any, *, tracker: "LLMWatch", provider: str, start: float
) -> _TrackedAsyncStream:
    return _TrackedAsyncStream(stream, tracker=tracker, provider=provider, start=start)


def _wrap_sync_stream(
    stream: Any, *, tracker: "LLMWatch", provider: str, start: float
) -> _TrackedSyncStream:
    return _TrackedSyncStream(stream, tracker=tracker, provider=provider, start=start)


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
        if kwargs.get("stream"):
            return _wrap_async_stream(response, tracker=tracker, provider="openai", start=start)
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
        if kwargs.get("stream"):
            return _wrap_async_stream(response, tracker=tracker, provider="anthropic", start=start)
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
        if kwargs.get("stream"):
            return _wrap_sync_stream(response, tracker=tracker, provider="openai", start=start)
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
        if kwargs.get("stream"):
            return _wrap_sync_stream(response, tracker=tracker, provider="anthropic", start=start)
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
        if kwargs.get("stream"):
            return _wrap_async_stream(response, tracker=tracker, provider="google", start=start)
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
        if kwargs.get("stream"):
            return _wrap_sync_stream(response, tracker=tracker, provider="google", start=start)
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
