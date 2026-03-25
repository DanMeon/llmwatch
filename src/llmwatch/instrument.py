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
    """Auto-detect SDK client type from its module path.

    Uses the shared module-prefix registry from extractors.base so that adding
    a provider via register_extractor() automatically enables client detection.
    Falls back to _INSTRUMENTORS keys for providers without an extractor.
    """
    from llmwatch.extractors.base import _MODULE_PREFIXES

    module = type(client).__module__
    # ^ Check shared prefix registry first (kept in sync with register_extractor)
    for prefix, provider in _MODULE_PREFIXES.items():
        if module.startswith(prefix) and provider in _INSTRUMENTORS:
            return provider
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


# * Cohere instrumentors


def instrument_cohere_async(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch async Cohere client to track rerank calls."""
    # ^ Cohere v2 client exposes rerank on client.v2 namespace; v1 directly on client
    target = getattr(client, "v2", client)
    original_rerank = target.rerank

    async def patched_rerank(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = await original_rerank(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        await tracker._safe_record(response, provider="cohere", latency_ms=elapsed_ms)
        return response

    target.rerank = patched_rerank


def instrument_cohere_sync(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch sync Cohere client to track rerank calls."""
    target = getattr(client, "v2", client)
    original_rerank = target.rerank

    def patched_rerank(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_rerank(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        tracker._safe_record_sync(response, provider="cohere", latency_ms=elapsed_ms)
        return response

    target.rerank = patched_rerank


def _instrument_cohere(client: Any, *, tracker: "LLMWatch") -> None:
    target = getattr(client, "v2", client)
    if asyncio.iscoroutinefunction(target.rerank):
        instrument_cohere_async(client, tracker=tracker)
    else:
        instrument_cohere_sync(client, tracker=tracker)


# * VoyageAI instrumentors


def instrument_voyageai_async(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch async VoyageAI client to track rerank calls."""
    original_rerank = client.rerank

    async def patched_rerank(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = await original_rerank(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        await tracker._safe_record(response, provider="voyageai", latency_ms=elapsed_ms)
        return response

    client.rerank = patched_rerank


def instrument_voyageai_sync(client: Any, *, tracker: "LLMWatch") -> None:
    """Patch sync VoyageAI client to track rerank calls."""
    original_rerank = client.rerank

    def patched_rerank(*args: Any, **kwargs: Any) -> Any:
        start = time.monotonic()
        response = original_rerank(*args, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000
        tracker._safe_record_sync(response, provider="voyageai", latency_ms=elapsed_ms)
        return response

    client.rerank = patched_rerank


def _instrument_voyageai(client: Any, *, tracker: "LLMWatch") -> None:
    if asyncio.iscoroutinefunction(client.rerank):
        instrument_voyageai_async(client, tracker=tracker)
    else:
        instrument_voyageai_sync(client, tracker=tracker)


_INSTRUMENTORS: dict[str, Callable[..., None]] = {}


def register_instrumentor(provider: str, instrumentor: Callable[..., None]) -> None:
    """Register a provider instrumentor.

    Args:
        provider: Provider name (must match the name used in register_extractor).
        instrumentor: Function(client, *, tracker) that patches the client.

    Example::

        register_instrumentor("cohere", _instrument_cohere)
    """
    _INSTRUMENTORS[provider] = instrumentor


# * Built-in instrumentor registrations
register_instrumentor("openai", _instrument_openai)
register_instrumentor("anthropic", _instrument_anthropic)
register_instrumentor("google", _instrument_google)
register_instrumentor("cohere", _instrument_cohere)
register_instrumentor("voyageai", _instrument_voyageai)
