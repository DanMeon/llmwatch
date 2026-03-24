"""Tests for _TrackedAsyncStream and _TrackedSyncStream wrappers in instrument.py."""

import asyncio
import time

import pytest

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.instrument import _TrackedAsyncStream, _TrackedSyncStream
from llmwatch.tracker import LLMWatch
from tests.conftest import FakeOpenAIResponse

# * Fake stream helpers


class _FakeAsyncStream:
    """Minimal async iterator yielding a fixed sequence of chunks."""

    def __init__(self, chunks, *, final_response=None, has_context_manager=False):
        self._chunks = iter(chunks)
        self._final_response = final_response
        self._has_context_manager = has_context_manager

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    # ^ Optional context-manager protocol
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def response(self):
        return self._final_response


class _FakeAsyncStreamWithFinal(_FakeAsyncStream):
    """Async stream that exposes get_final_completion()."""

    def __init__(self, chunks, *, final_response):
        super().__init__(chunks, final_response=final_response)

    def get_final_completion(self):
        return self._final_response


class _FakeSyncStream:
    """Minimal sync iterator yielding a fixed sequence of chunks."""

    def __init__(self, chunks, *, final_response=None):
        self._chunks = iter(chunks)
        self._final_response = final_response

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise

    # ^ Optional context-manager protocol
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def response(self):
        return self._final_response


class _FakeSyncStreamWithFinal(_FakeSyncStream):
    """Sync stream that exposes get_final_completion()."""

    def __init__(self, chunks, *, final_response):
        super().__init__(chunks, final_response=final_response)

    def get_final_completion(self):
        return self._final_response


# * Shared tracker fixture


@pytest.fixture
async def tracker_with_storage():
    storage = Storage("sqlite+aiosqlite://")
    tracker = LLMWatch(storage=storage)
    yield tracker
    await storage.close()


# * _TrackedAsyncStream tests


class TestTrackedAsyncStreamIteration:
    async def test_yields_all_chunks(self, tracker_with_storage):
        chunks = ["a", "b", "c"]
        raw = _FakeAsyncStream(chunks)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        collected = [chunk async for chunk in wrapped]
        assert collected == ["a", "b", "c"]

    async def test_records_on_exhaustion_via_response_attr(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=10, completion_tokens=5)
        raw = _FakeAsyncStream(["chunk1"], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        async for _ in wrapped:
            pass
        assert await tracker_with_storage._storage.count() == 1

    async def test_records_on_exhaustion_via_get_final_completion(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=20, completion_tokens=10)
        raw = _FakeAsyncStreamWithFinal(["chunk1", "chunk2"], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        async for _ in wrapped:
            pass
        assert await tracker_with_storage._storage.count() == 1

    async def test_does_not_double_record(self, tracker_with_storage):
        response = FakeOpenAIResponse()
        raw = _FakeAsyncStream(["chunk"], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        async for _ in wrapped:
            pass
        # ^ _recorded is True after iteration; __aexit__ must not record again
        assert wrapped._recorded is True
        await wrapped.__aexit__(None, None, None)
        # ^ Still exactly 1 record — aexit skips because _recorded is True
        assert await tracker_with_storage._storage.count() == 1

    async def test_no_recording_when_no_response(self, tracker_with_storage):
        raw = _FakeAsyncStream(["chunk"])  # ^ no final_response
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        async for _ in wrapped:
            pass
        # ^ Nothing to record without a response object
        assert await tracker_with_storage._storage.count() == 0

    async def test_getattr_delegates_to_inner_stream(self, tracker_with_storage):
        raw = _FakeAsyncStream(["x"])
        raw.custom_attr = "hello"
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        assert wrapped.custom_attr == "hello"


class TestTrackedAsyncStreamContextManager:
    async def test_aenter_returns_self(self, tracker_with_storage):
        raw = _FakeAsyncStream([])
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        result = await wrapped.__aenter__()
        assert result is wrapped

    async def test_aexit_records_when_not_already_recorded(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=30, completion_tokens=15)
        raw = _FakeAsyncStream([], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        await wrapped.__aexit__(None, None, None)
        assert await tracker_with_storage._storage.count() == 1

    async def test_aexit_skips_recording_when_already_recorded(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=40, completion_tokens=20)
        raw = _FakeAsyncStream([], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        wrapped._recorded = True
        await wrapped.__aexit__(None, None, None)
        assert await tracker_with_storage._storage.count() == 0

    async def test_context_manager_protocol_full(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=50, completion_tokens=25)
        raw = _FakeAsyncStream(["c1", "c2"], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        async with wrapped as stream:
            chunks = [c async for c in stream]
        assert chunks == ["c1", "c2"]
        # ^ One record: from iteration (not aexit, because _recorded is already True)
        assert await tracker_with_storage._storage.count() == 1

    async def test_aenter_without_inner_context_manager(self, tracker_with_storage):
        """aenter must not crash when inner stream lacks __aenter__."""

        class _PlainAsyncStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        raw = _PlainAsyncStream()
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        result = await wrapped.__aenter__()
        assert result is wrapped

    async def test_aexit_without_inner_context_manager(self, tracker_with_storage):
        """aexit must not crash when inner stream lacks __aexit__."""

        class _PlainAsyncStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        raw = _PlainAsyncStream()
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        wrapped._recorded = True  # ^ prevent actual record call
        await wrapped.__aexit__(None, None, None)  # ^ must not raise


class TestTrackedAsyncStreamGetFinalCompletionFailure:
    async def test_get_final_completion_exception_falls_back_to_response(
        self, tracker_with_storage
    ):
        response = FakeOpenAIResponse(prompt_tokens=60, completion_tokens=30)

        class _BrokenFinal(_FakeAsyncStream):
            def get_final_completion(self):
                raise RuntimeError("broken")

        raw = _BrokenFinal([], final_response=response)
        wrapped = _TrackedAsyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        await wrapped._record_from_stream()
        assert await tracker_with_storage._storage.count() == 1


# * _TrackedSyncStream tests


class TestTrackedSyncStreamIteration:
    async def test_yields_all_chunks(self, tracker_with_storage):
        chunks = [1, 2, 3]
        raw = _FakeSyncStream(chunks)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        collected = list(wrapped)
        assert collected == [1, 2, 3]

    async def test_records_on_exhaustion_via_response_attr(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=10, completion_tokens=5)
        raw = _FakeSyncStream(["chunk"], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        list(wrapped)
        # ^ _safe_record_sync posts to event loop; allow it to run
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 1

    async def test_records_on_exhaustion_via_get_final_completion(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=20, completion_tokens=10)
        raw = _FakeSyncStreamWithFinal(["c1", "c2"], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        list(wrapped)
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 1

    async def test_does_not_double_record(self, tracker_with_storage):
        response = FakeOpenAIResponse()
        raw = _FakeSyncStream(["chunk"], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        list(wrapped)
        await asyncio.sleep(0)
        count_after_iteration = await tracker_with_storage._storage.count()

        # ^ _recorded is True after iteration; __exit__ must not record again
        assert wrapped._recorded is True
        wrapped.__exit__(None, None, None)
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == count_after_iteration

    async def test_no_recording_when_no_response(self, tracker_with_storage):
        raw = _FakeSyncStream(["chunk"])  # ^ no final_response
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        list(wrapped)
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 0

    async def test_getattr_delegates_to_inner_stream(self, tracker_with_storage):
        raw = _FakeSyncStream([])
        raw.custom_flag = True
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        assert wrapped.custom_flag is True


class TestTrackedSyncStreamContextManager:
    async def test_enter_returns_self(self, tracker_with_storage):
        raw = _FakeSyncStream([])
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        result = wrapped.__enter__()
        assert result is wrapped

    async def test_exit_records_when_not_already_recorded(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=30, completion_tokens=15)
        raw = _FakeSyncStream([], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        wrapped.__exit__(None, None, None)
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 1

    async def test_exit_skips_recording_when_already_recorded(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=40, completion_tokens=20)
        raw = _FakeSyncStream([], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        wrapped._recorded = True
        wrapped.__exit__(None, None, None)
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 0

    async def test_context_manager_protocol_full(self, tracker_with_storage):
        response = FakeOpenAIResponse(prompt_tokens=50, completion_tokens=25)
        raw = _FakeSyncStream([10, 20], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        with wrapped as stream:
            chunks = list(stream)
        assert chunks == [10, 20]
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 1

    async def test_enter_without_inner_context_manager(self, tracker_with_storage):
        """__enter__ must not crash when inner stream lacks __enter__."""

        class _PlainSyncStream:
            def __iter__(self):
                return iter([])

        raw = _PlainSyncStream()
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        result = wrapped.__enter__()
        assert result is wrapped

    async def test_exit_without_inner_context_manager(self, tracker_with_storage):
        """__exit__ must not crash when inner stream lacks __exit__."""

        class _PlainSyncStream:
            def __iter__(self):
                return iter([])

        raw = _PlainSyncStream()
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        wrapped._recorded = True
        wrapped.__exit__(None, None, None)  # ^ must not raise


class TestTrackedSyncStreamGetFinalCompletionFailure:
    async def test_get_final_completion_exception_falls_back_to_response(
        self, tracker_with_storage
    ):
        response = FakeOpenAIResponse(prompt_tokens=60, completion_tokens=30)

        class _BrokenFinal(_FakeSyncStream):
            def get_final_completion(self):
                raise RuntimeError("broken")

        raw = _BrokenFinal([], final_response=response)
        wrapped = _TrackedSyncStream(
            raw, tracker=tracker_with_storage, provider="openai", start=time.monotonic()
        )
        wrapped._record_from_stream()
        await asyncio.sleep(0)
        assert await tracker_with_storage._storage.count() == 1
