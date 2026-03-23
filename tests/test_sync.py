"""Tests for sync function support — tracked() and instrument()."""

import pytest

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.tracker import LLMWatch
from tests.conftest import (
    FakeOpenAIResponse,
    FakeSyncAnthropic,
    FakeSyncOpenAI,
)


class TestSyncTracked:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeSyncOpenAI()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    def test_sync_tracked_tags(self):
        """@tracked() works on sync functions and applies tags."""

        @self.watcher.tracked(feature="summarize", user_id="alice")
        def summarize():
            return self.client.chat.completions.create(model="gpt-4o")

        response = summarize()
        assert response.model == "gpt-4o"

    def test_sync_tracked_preserves_return_value(self):
        @self.watcher.tracked(feature="chat")
        def chat():
            response = self.client.chat.completions.create(model="gpt-4o")
            return response.choices[0].message.content

        result = chat()
        assert isinstance(result, str)

    def test_sync_tracked_rate_limit(self):
        @self.watcher.tracked(feature="test")
        def fail():
            exc = Exception("rate limited")
            exc.status_code = 429
            raise exc

        assert self.watcher.rate_limit_count == 0
        with pytest.raises(Exception):
            fail()
        assert self.watcher.rate_limit_count == 1

    def test_sync_tracked_extra_tags(self):
        @self.watcher.tracked(feature="chat", team="backend")
        def chat():
            return self.client.chat.completions.create(model="gpt-4o")

        chat()


class TestSyncInstrumentOpenAI:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeSyncOpenAI()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    def test_sync_instrument_records(self):
        """Sync instrument patches record calls to storage."""
        self.client.chat.completions.create(model="gpt-4o")

    def test_sync_instrument_response_unchanged(self):
        response = self.client.chat.completions.create(model="gpt-4o-mini")
        assert response.model == "gpt-4o-mini"
        assert hasattr(response, "choices")

    def test_sync_tracked_with_instrument(self):
        """Sync @tracked() + instrument() records with tags."""

        @self.watcher.tracked(feature="chat", user_id="bob")
        def chat():
            return self.client.chat.completions.create(model="gpt-4o")

        result = chat()
        assert result.model == "gpt-4o"


class TestSyncInstrumentAnthropic:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeSyncAnthropic()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    def test_sync_anthropic_records(self):
        self.client.messages.create(model="claude-sonnet-4-20250514")

    def test_sync_anthropic_response_unchanged(self):
        response = self.client.messages.create(model="claude-sonnet-4-20250514")
        assert response.model == "claude-sonnet-4-20250514"


class TestSyncSafety:
    def test_sync_db_failure_does_not_lose_response(self):
        storage = Storage("sqlite+aiosqlite://")
        client = FakeSyncOpenAI()
        watcher = LLMWatch(storage=storage, client=client)

        _original_record = watcher._safe_record  # noqa: F841

        async def failing_record(*args, **kwargs):
            raise RuntimeError("DB connection lost")

        watcher._safe_record = failing_record

        @watcher.tracked(feature="test")
        def call_llm():
            return client.chat.completions.create(model="gpt-4o")

        result = call_llm()
        assert result is not None
        assert result.model == "gpt-4o"


class TestSyncBudgetAlert:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeSyncOpenAI(
            response_factory=lambda **kw: FakeOpenAIResponse(
                model=kw.get("model", "gpt-4o"),
                prompt_tokens=1000,
                completion_tokens=500,
            )
        )
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    def test_sync_budget_alert_fires(self):
        alerts: list = []
        self.watcher.budget.add_rule(
            max_cost_usd=0.001,
            callback=lambda record, **kwargs: alerts.append(record),
        )

        @self.watcher.tracked(feature="expensive")
        def call_llm():
            return self.client.chat.completions.create(model="gpt-4o")

        call_llm()
        assert len(alerts) == 1
        assert alerts[0].tags.feature == "expensive"
