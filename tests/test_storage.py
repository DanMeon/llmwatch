"""Tests for the Storage module."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.schemas.tags import Tags
from llmwatch.schemas.usage import TokenUsage, UsageRecord


def _make_record(
    feature: str = "chat",
    user_id: str = "alice",
    model: str = "gpt-4o",
    provider: str = "openai",
    cost: float = 0.01,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> UsageRecord:
    return UsageRecord(
        tags=Tags(feature=feature, user_id=user_id),
        model=model,
        provider=provider,
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        cost_usd=cost,
    )


class TestStorage:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        yield
        await self.storage.close()

    async def test_save_and_count(self):
        await self.storage.save(_make_record())
        assert await self.storage.count() == 1

    async def test_save_batch(self):
        records = [_make_record(feature=f"feat_{i}") for i in range(5)]
        await self.storage.save_batch(records)
        assert await self.storage.count() == 5

    async def test_query_by_feature(self):
        await self.storage.save(_make_record(feature="chat"))
        await self.storage.save(_make_record(feature="summarize"))
        results = await self.storage.query(feature="chat")
        assert len(results) == 1
        assert results[0].tags.feature == "chat"

    async def test_query_by_user_id(self):
        await self.storage.save(_make_record(user_id="alice"))
        await self.storage.save(_make_record(user_id="bob"))
        results = await self.storage.query(user_id="bob")
        assert len(results) == 1
        assert results[0].tags.user_id == "bob"

    async def test_aggregate_by_feature(self):
        await self.storage.save(_make_record(feature="chat", cost=0.01))
        await self.storage.save(_make_record(feature="chat", cost=0.02))
        await self.storage.save(_make_record(feature="summarize", cost=0.05))

        reports = await self.storage.aggregate(group_by="feature")
        assert len(reports) == 2
        assert reports[0].group_value == "summarize"
        assert abs(reports[0].total_cost_usd - 0.05) < 1e-9
        assert reports[1].group_value == "chat"
        assert reports[1].total_requests == 2

    async def test_aggregate_by_model(self):
        await self.storage.save(_make_record(model="gpt-4o", cost=0.01))
        await self.storage.save(_make_record(model="claude-sonnet-4-20250514", cost=0.02))
        reports = await self.storage.aggregate(group_by="model")
        assert len(reports) == 2

    async def test_delete_by_feature(self):
        await self.storage.save(_make_record(feature="chat"))
        await self.storage.save(_make_record(feature="summarize"))
        deleted = await self.storage.delete(feature="chat")
        assert deleted == 1
        assert await self.storage.count() == 1

    async def test_delete_by_time(self):
        old_record = _make_record()
        old_record.timestamp = datetime.now(UTC) - timedelta(days=100)
        await self.storage.save(old_record)
        await self.storage.save(_make_record())

        cutoff = datetime.now(UTC) - timedelta(days=90)
        deleted = await self.storage.delete(before=cutoff)
        assert deleted == 1
        assert await self.storage.count() == 1

    async def test_query_with_time_range(self):
        old = _make_record(feature="old")
        old.timestamp = datetime.now(UTC) - timedelta(days=10)
        await self.storage.save(old)
        await self.storage.save(_make_record(feature="new"))

        start = datetime.now(UTC) - timedelta(days=1)
        results = await self.storage.query(start_time=start)
        assert len(results) == 1
        assert results[0].tags.feature == "new"

    async def test_roundtrip_preserves_data(self):
        record = _make_record(
            feature="translate",
            user_id="bob",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            cost=0.123,
            prompt_tokens=500,
            completion_tokens=250,
        )
        await self.storage.save(record)
        results = await self.storage.query()
        assert len(results) == 1
        r = results[0]
        assert r.tags.feature == "translate"
        assert r.tags.user_id == "bob"
        assert r.model == "claude-sonnet-4-20250514"
        assert abs(r.cost_usd - 0.123) < 1e-9

    async def test_io_roundtrip(self):
        record = _make_record()
        record.input_data = {"prompt": "hello", "temperature": 0.7}
        record.output_data = "world"
        await self.storage.save(record)
        results = await self.storage.query()
        assert results[0].input_data == {"prompt": "hello", "temperature": 0.7}
        assert results[0].output_data == "world"


_OLD_SCHEMA = """
CREATE TABLE llmwatch_usage (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    feature TEXT,
    user_tag TEXT,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    environment TEXT,
    extra_tags TEXT,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL,
    latency_ms REAL,
    raw_response_id TEXT
)
"""


class TestAutoMigration:
    async def test_missing_column_added(self):
        """Verify that missing columns are auto-added to existing tables."""
        import tempfile
        from pathlib import Path as PathLibPath

        tmp = PathLibPath(tempfile.mktemp(suffix=".db"))
        try:
            # ^ Create table with old schema (no input_data, output_data columns)
            url = f"sqlite+aiosqlite:///{tmp}"
            engine = create_async_engine(url)
            async with engine.begin() as conn:
                await conn.execute(text(_OLD_SCHEMA))
            await engine.dispose()

            # ^ Connect Storage — auto_migrate adds missing columns
            storage = Storage(url)
            record = _make_record()
            record.input_data = {"prompt": "test"}
            record.output_data = "response"
            await storage.save(record)

            results = await storage.query()
            assert len(results) == 1
            assert results[0].input_data == {"prompt": "test"}
            assert results[0].output_data == "response"
            await storage.close()
        finally:
            tmp.unlink(missing_ok=True)

    async def test_existing_data_preserved(self):
        """Verify that existing data is preserved after migration."""
        import tempfile
        from pathlib import Path as PathLibPath

        tmp = PathLibPath(tempfile.mktemp(suffix=".db"))
        try:
            url = f"sqlite+aiosqlite:///{tmp}"
            engine = create_async_engine(url)
            async with engine.begin() as conn:
                await conn.execute(text(_OLD_SCHEMA))
                await conn.execute(
                    text(
                        "INSERT INTO llmwatch_usage "
                        "(id, timestamp, model, provider, cost_usd, prompt_tokens, completion_tokens, total_tokens) "
                        "VALUES ('00000000-0000-0000-0000-000000000001', '2026-01-01T00:00:00', 'gpt-4o', 'openai', 0.01, 100, 50, 150)"
                    )
                )
            await engine.dispose()

            storage = Storage(url)
            results = await storage.query()
            assert len(results) == 1
            assert results[0].model == "gpt-4o"
            # ^ New columns are NULL
            assert results[0].input_data is None
            assert results[0].output_data is None
            await storage.close()
        finally:
            tmp.unlink(missing_ok=True)
