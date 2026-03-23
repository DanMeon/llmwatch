"""Tests for the tracker module."""

import pytest

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.tracker import LLMWatch
from tests.conftest import FakeAsyncOpenAI, FakeOpenAIResponse


class TestLLMWatch:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.client = FakeAsyncOpenAI()
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    async def test_tracked_with_instrument(self):
        @self.watcher.tracked(feature="summarize")
        async def summarize():
            return await self.client.chat.completions.create(model="gpt-4o")

        response = await summarize()
        assert response.model == "gpt-4o"
        assert await self.storage.count() == 1
        records = await self.storage.query(feature="summarize")
        assert len(records) == 1
        assert records[0].tags.feature == "summarize"
        assert records[0].cost_usd > 0

    async def test_tracked_with_user_id(self):
        @self.watcher.tracked(feature="chat", user_id="alice")
        async def chat():
            return await self.client.chat.completions.create(model="gpt-4o")

        await chat()
        records = await self.storage.query(user_id="alice")
        assert len(records) == 1
        assert records[0].tags.feature == "chat"

    async def test_auto_detect_provider(self):
        @self.watcher.tracked(feature="test")
        async def call_llm():
            return await self.client.chat.completions.create(model="gpt-4o")

        await call_llm()
        records = await self.storage.query()
        assert records[0].provider == "openai"

    async def test_extra_tags(self):
        @self.watcher.tracked(feature="chat", team="backend", version="v2")
        async def call_llm():
            return await self.client.chat.completions.create(model="gpt-4o")

        await call_llm()
        records = await self.storage.query()
        assert records[0].tags.extra["team"] == "backend"
        assert records[0].tags.extra["version"] == "v2"

    async def test_preserves_return_value(self):
        @self.watcher.tracked(feature="test")
        async def call_llm():
            return await self.client.chat.completions.create(model="gpt-4o-mini")

        result = await call_llm()
        assert result.model == "gpt-4o-mini"

    async def test_processed_return_value(self):
        """tracked() works even when return value is not an LLM response."""

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

    async def test_no_double_record(self):
        """instrument + tracked should record only once."""

        @self.watcher.tracked(feature="chat")
        async def chat():
            return await self.client.chat.completions.create(model="gpt-4o")

        await chat()
        assert await self.storage.count() == 1

    async def test_multiple_calls(self):
        @self.watcher.tracked(feature="chat")
        async def chat():
            return await self.client.chat.completions.create(model="gpt-4o")

        @self.watcher.tracked(feature="summarize")
        async def summarize():
            return await self.client.chat.completions.create(model="gpt-4o")

        await chat()
        await chat()
        await summarize()

        assert await self.storage.count() == 3
        summary = await self.watcher.report.by_feature()
        assert summary.total_requests == 3
        assert len(summary.breakdowns) == 2

    async def test_report_by_model(self):
        @self.watcher.tracked(feature="test")
        async def call_4o():
            return await self.client.chat.completions.create(model="gpt-4o")

        @self.watcher.tracked(feature="test")
        async def call_mini():
            return await self.client.chat.completions.create(model="gpt-4o-mini")

        await call_4o()
        await call_mini()
        summary = await self.watcher.report.by_model()
        assert len(summary.breakdowns) == 2

    async def test_latency_recorded(self):
        @self.watcher.tracked(feature="test")
        async def call_llm():
            return await self.client.chat.completions.create(model="gpt-4o")

        await call_llm()
        records = await self.storage.query()
        assert records[0].latency_ms is not None
        assert records[0].latency_ms >= 0

    def test_sync_function_accepted(self):
        """tracked() now accepts sync functions."""

        @self.watcher.tracked(feature="test")
        def sync_func():
            return "ok"

        assert sync_func() == "ok"


class TestClientConstructor:
    async def test_client_param(self):
        """LLMWatch(client=client) auto-instruments."""
        storage = Storage("sqlite+aiosqlite://")
        client = FakeAsyncOpenAI()
        _watcher = LLMWatch(storage=storage, client=client)  # noqa: F841

        await client.chat.completions.create(model="gpt-4o")
        assert await storage.count() == 1
        await storage.close()

    async def test_clients_param(self):
        """LLMWatch(clients=[...]) auto-instruments multiple clients."""
        storage = Storage("sqlite+aiosqlite://")
        openai_client = FakeAsyncOpenAI()
        from tests.conftest import FakeAsyncAnthropic

        anthropic_client = FakeAsyncAnthropic()
        _watcher = LLMWatch(storage=storage, clients=[openai_client, anthropic_client])  # noqa: F841

        await openai_client.chat.completions.create(model="gpt-4o")
        await anthropic_client.messages.create(model="claude-sonnet-4-20250514")
        assert await storage.count() == 2
        await storage.close()


class TestBudgetAlert:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        # ^ High token counts to exceed budget thresholds
        self.client = FakeAsyncOpenAI(
            response_factory=lambda **kw: FakeOpenAIResponse(
                model=kw.get("model", "gpt-4o"),
                prompt_tokens=1000,
                completion_tokens=500,
            )
        )
        self.watcher = LLMWatch(storage=self.storage, client=self.client)
        yield
        await self.storage.close()

    async def test_budget_alert_fires(self):
        alerts: list = []
        self.watcher.budget.add_rule(
            max_cost_usd=0.001,
            callback=lambda record, **kwargs: alerts.append(record),
        )

        @self.watcher.tracked(feature="expensive")
        async def call_llm():
            return await self.client.chat.completions.create(model="gpt-4o")

        await call_llm()
        assert len(alerts) == 1
        assert alerts[0].tags.feature == "expensive"

    async def test_budget_alert_feature_filter(self):
        alerts: list = []
        self.watcher.budget.add_rule(
            max_cost_usd=0.001,
            callback=lambda record, **kwargs: alerts.append(record),
            feature="expensive",
        )

        @self.watcher.tracked(feature="cheap")
        async def cheap():
            return await self.client.chat.completions.create(model="gpt-4o")

        await cheap()
        assert len(alerts) == 0

    async def test_budget_alert_async_callback(self):
        alerts: list = []

        async def on_alert(record, **kwargs):
            alerts.append(record)

        self.watcher.budget.add_rule(max_cost_usd=0.001, callback=on_alert)

        @self.watcher.tracked(feature="test")
        async def call_llm():
            return await self.client.chat.completions.create(model="gpt-4o")

        await call_llm()
        assert len(alerts) == 1


class TestRateLimit:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.watcher = LLMWatch(storage=self.storage)
        yield
        await self.storage.close()

    async def test_rate_limit_by_status_code(self):
        @self.watcher.tracked(feature="test")
        async def fail():
            exc = Exception("rate limited")
            exc.status_code = 429
            raise exc

        assert self.watcher.rate_limit_count == 0
        with pytest.raises(Exception):
            await fail()
        assert self.watcher.rate_limit_count == 1

    async def test_rate_limit_by_class_name(self):
        @self.watcher.tracked(feature="test")
        async def fail():
            raise type("RateLimitError", (Exception,), {})("too many requests")

        with pytest.raises(Exception):
            await fail()
        assert self.watcher.rate_limit_count == 1


class TestSafety:
    async def test_db_failure_does_not_lose_response(self):
        storage = Storage("sqlite+aiosqlite://")
        client = FakeAsyncOpenAI()
        watcher = LLMWatch(storage=storage, client=client)

        async def failing_save(record):
            raise RuntimeError("DB connection lost")

        storage.save = failing_save

        @watcher.tracked(feature="test")
        async def call_llm():
            return await client.chat.completions.create(model="gpt-4o")

        result = await call_llm()
        assert result is not None
        assert result.model == "gpt-4o"
        await storage.close()

    async def test_budget_callback_failure_does_not_lose_response(self):
        storage = Storage("sqlite+aiosqlite://")
        client = FakeAsyncOpenAI()
        watcher = LLMWatch(storage=storage, client=client)

        def bad_callback(record, **kwargs):
            raise RuntimeError("Slack webhook failed")

        watcher.budget.add_rule(max_cost_usd=0.0, callback=bad_callback)

        @watcher.tracked(feature="test")
        async def call_llm():
            return await client.chat.completions.create(model="gpt-4o")

        result = await call_llm()
        assert result is not None
        await storage.close()
