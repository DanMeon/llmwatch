"""Tests for SDK instrumentation."""

import pytest

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.tracker import LLMWatch
from tests.conftest import FakeAsyncAnthropic, FakeAsyncGoogle, FakeAsyncOpenAI, FakeSyncGoogle


class TestInstrumentOpenAI:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeAsyncOpenAI()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    async def test_basic_tracking(self):
        await self.client.chat.completions.create(model="gpt-4o")
        assert await self.storage.count() == 1
        records = await self.storage.query()
        assert records[0].model == "gpt-4o"
        assert records[0].provider == "openai"
        assert records[0].cost_usd > 0

    async def test_tracks_without_decorator(self):
        """Instrument alone should track calls even without @tracked()."""
        await self.client.chat.completions.create(model="gpt-4o")
        await self.client.chat.completions.create(model="gpt-4o-mini")
        assert await self.storage.count() == 2

    async def test_response_unchanged(self):
        """Instrumentation must not alter the response object."""
        response = await self.client.chat.completions.create(model="gpt-4o")
        assert response.model == "gpt-4o"
        assert hasattr(response, "choices")

    async def test_latency_recorded(self):
        await self.client.chat.completions.create(model="gpt-4o")
        records = await self.storage.query()
        assert records[0].latency_ms is not None
        assert records[0].latency_ms >= 0


class TestInstrumentAnthropic:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeAsyncAnthropic()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    async def test_basic_tracking(self):
        await self.client.messages.create(model="claude-sonnet-4-20250514")
        assert await self.storage.count() == 1
        records = await self.storage.query()
        assert records[0].model == "claude-sonnet-4-20250514"
        assert records[0].provider == "anthropic"


class TestInstrumentWithTracked:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeAsyncOpenAI()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    async def test_no_double_record(self):
        """When both instrument and @tracked are active, record only once."""

        @self.watcher.tracked(feature="chat")
        async def chat():
            return await self.client.chat.completions.create(model="gpt-4o")

        await chat()
        assert await self.storage.count() == 1

    async def test_tags_applied_via_tracked(self):
        """Tags from @tracked() should be picked up by the instrument layer."""

        @self.watcher.tracked(feature="summarize", user_id="bob")
        async def summarize():
            return await self.client.chat.completions.create(model="gpt-4o")

        await summarize()
        records = await self.storage.query()
        assert records[0].tags.feature == "summarize"
        assert records[0].tags.user_id == "bob"

    async def test_processed_return_value(self):
        """Instrument tracks usage even when return value is processed."""

        @self.watcher.tracked(feature="chat")
        async def chat():
            response = await self.client.chat.completions.create(model="gpt-4o")
            return response.choices[0].message.content

        result = await chat()
        assert isinstance(result, str)
        assert await self.storage.count() == 1
        records = await self.storage.query()
        assert records[0].tags.feature == "chat"
        assert records[0].cost_usd > 0


class TestInstrumentGoogle:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeAsyncGoogle()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    async def test_basic_tracking(self):
        await self.client.models.generate_content(model="gemini-2.0-flash")
        assert await self.storage.count() == 1
        records = await self.storage.query()
        assert records[0].model == "gemini-2.0-flash"
        assert records[0].provider == "google"

    async def test_response_unchanged(self):
        """Instrumentation must not alter the response object."""
        response = await self.client.models.generate_content(model="gemini-2.0-flash")
        assert response.model == "gemini-2.0-flash"
        assert hasattr(response, "text")

    async def test_latency_recorded(self):
        await self.client.models.generate_content(model="gemini-2.0-flash")
        records = await self.storage.query()
        assert records[0].latency_ms is not None
        assert records[0].latency_ms >= 0

    async def test_token_usage_recorded(self):
        await self.client.models.generate_content(model="gemini-2.0-flash")
        records = await self.storage.query()
        assert records[0].token_usage.prompt_tokens == 150
        assert records[0].token_usage.completion_tokens == 75


class TestInstrumentGoogleSync:
    async def test_sync_google_tracking(self):
        storage = Storage("sqlite+aiosqlite://")
        client = FakeSyncGoogle()
        _watcher = LLMWatch(storage=storage, client=client)  # noqa: F841

        client.models.generate_content(model="gemini-2.0-flash")
        assert await storage.count() == 1
        records = await storage.query()
        assert records[0].provider == "google"
        await storage.close()


class TestInstrumentDetection:
    async def test_unsupported_client_raises(self):
        storage = Storage("sqlite+aiosqlite://")
        watcher = LLMWatch(storage=storage)

        with pytest.raises(ValueError, match="Unsupported client type"):
            watcher.instrument(object())

        await storage.close()

    async def test_explicit_provider(self):
        """Allow explicit provider override."""
        storage = Storage("sqlite+aiosqlite://")
        client = FakeAsyncOpenAI()
        watcher = LLMWatch(storage=storage)

        watcher.instrument(client, provider="openai")
        await client.chat.completions.create(model="gpt-4o")
        assert await storage.count() == 1
        await storage.close()
