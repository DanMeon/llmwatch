"""Tests for the reporting module."""

import csv
import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path as PathLibPath

import pytest

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.reporting import Reporter, _period_to_start
from llmwatch.schemas.tags import Tags
from llmwatch.schemas.usage import TokenUsage, UsageRecord

# * Shared helper to build UsageRecord instances


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


# * _period_to_start unit tests


class TestPeriodToStart:
    def test_hours(self):
        before = datetime.now(UTC)
        result = _period_to_start("24h")
        after = datetime.now(UTC)
        assert (
            before - timedelta(hours=24)
            <= result
            <= after - timedelta(hours=24) + timedelta(seconds=1)
        )

    def test_days(self):
        before = datetime.now(UTC)
        result = _period_to_start("7d")
        assert (
            before - timedelta(days=7)
            <= result
            <= before - timedelta(days=7) + timedelta(seconds=1)
        )

    def test_weeks(self):
        before = datetime.now(UTC)
        result = _period_to_start("2w")
        assert (
            before - timedelta(weeks=2)
            <= result
            <= before - timedelta(weeks=2) + timedelta(seconds=1)
        )

    def test_months(self):
        before = datetime.now(UTC)
        result = _period_to_start("3m")
        # ^ 3m = 90 days
        assert (
            before - timedelta(days=90)
            <= result
            <= before - timedelta(days=90) + timedelta(seconds=1)
        )

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid period string"):
            _period_to_start("")

    def test_single_char_raises(self):
        with pytest.raises(ValueError, match="Invalid period string"):
            _period_to_start("d")

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="Period value must be positive"):
            _period_to_start("-7d")

    def test_zero_value_raises(self):
        with pytest.raises(ValueError, match="Period value must be positive"):
            _period_to_start("0d")

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown period unit"):
            _period_to_start("7x")


# * Reporter integration tests


class TestReporter:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.reporter = Reporter(self.storage)
        yield
        await self.storage.close()

    async def test_summary_empty_storage(self):
        summary = await self.reporter.summary()
        assert summary.total_requests == 0
        assert summary.total_cost_usd == 0.0
        assert summary.breakdowns == []

    async def test_summary_with_period(self):
        await self.storage.save(_make_record(feature="chat", cost=0.05))
        summary = await self.reporter.summary(group_by="feature", period="7d")
        assert summary.total_requests == 1
        assert abs(summary.total_cost_usd - 0.05) < 1e-9

    async def test_by_feature(self):
        await self.storage.save(_make_record(feature="chat"))
        await self.storage.save(_make_record(feature="summarize"))
        summary = await self.reporter.by_feature()
        features = {b.group_value for b in summary.breakdowns}
        assert "chat" in features
        assert "summarize" in features

    async def test_by_user_id(self):
        await self.storage.save(_make_record(user_id="alice"))
        await self.storage.save(_make_record(user_id="bob"))
        summary = await self.reporter.by_user_id()
        users = {b.group_value for b in summary.breakdowns}
        assert "alice" in users
        assert "bob" in users

    async def test_by_model(self):
        await self.storage.save(_make_record(model="gpt-4o"))
        await self.storage.save(_make_record(model="gpt-4o-mini"))
        summary = await self.reporter.by_model()
        models = {b.group_value for b in summary.breakdowns}
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    async def test_by_provider(self):
        await self.storage.save(_make_record(provider="openai"))
        await self.storage.save(_make_record(provider="anthropic"))
        summary = await self.reporter.by_provider()
        providers = {b.group_value for b in summary.breakdowns}
        assert "openai" in providers
        assert "anthropic" in providers


class TestExportCSV:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.reporter = Reporter(self.storage)
        yield
        await self.storage.close()

    async def test_export_csv_creates_file(self):
        await self.storage.save(_make_record(feature="chat", cost=0.01))
        await self.storage.save(_make_record(feature="summarize", cost=0.02))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = PathLibPath(tmp.name)

        try:
            await self.reporter.export_csv(path, group_by="feature")

            assert path.exists()
            content = path.read_text()
            assert "feature" in content
            assert "requests" in content
            assert "cost_usd" in content
        finally:
            path.unlink(missing_ok=True)

    async def test_export_csv_rows_match_data(self):
        await self.storage.save(
            _make_record(feature="chat", cost=0.01, prompt_tokens=100, completion_tokens=50)
        )
        await self.storage.save(
            _make_record(feature="chat", cost=0.02, prompt_tokens=200, completion_tokens=100)
        )

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
            path = PathLibPath(tmp.name)

        try:
            await self.reporter.export_csv(path, group_by="feature")

            with path.open(newline="") as f:
                rows = list(csv.reader(f))

            # ^ header + 1 data row for "chat"
            assert len(rows) == 2
            header = rows[0]
            assert header == [
                "feature",
                "requests",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
            ]
            data_row = rows[1]
            assert data_row[0] == "chat"
            assert data_row[1] == "2"  # ^ total_requests
        finally:
            path.unlink(missing_ok=True)

    async def test_export_csv_accepts_string_path(self):
        await self.storage.save(_make_record(feature="chat"))

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path_str = tmp.name

        path = PathLibPath(path_str)
        try:
            # ^ Pass path as a plain string (not PathLibPath)
            await self.reporter.export_csv(path_str, group_by="feature")
            assert path.exists()
        finally:
            path.unlink(missing_ok=True)

    async def test_export_csv_empty_storage(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            path = PathLibPath(tmp.name)

        try:
            await self.reporter.export_csv(path, group_by="feature")
            with path.open(newline="") as f:
                rows = list(csv.reader(f))
            # ^ Only header row when no data
            assert len(rows) == 1
        finally:
            path.unlink(missing_ok=True)


class TestExportJSON:
    @pytest.fixture(autouse=True)
    async def setup(self):
        self.storage = Storage("sqlite+aiosqlite://")
        self.reporter = Reporter(self.storage)
        yield
        await self.storage.close()

    async def test_export_json_creates_file(self):
        await self.storage.save(_make_record(feature="chat", cost=0.05))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = PathLibPath(tmp.name)

        try:
            await self.reporter.export_json(path, group_by="feature")
            assert path.exists()
            data = json.loads(path.read_text())
            assert "total_cost_usd" in data
            assert "breakdowns" in data
        finally:
            path.unlink(missing_ok=True)

    async def test_export_json_values_match_storage(self):
        await self.storage.save(_make_record(feature="chat", cost=0.03))
        await self.storage.save(_make_record(feature="summarize", cost=0.07))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = PathLibPath(tmp.name)

        try:
            await self.reporter.export_json(path, group_by="feature")
            data = json.loads(path.read_text())

            assert data["total_requests"] == 2
            assert abs(data["total_cost_usd"] - 0.10) < 1e-9
            group_values = {b["group_value"] for b in data["breakdowns"]}
            assert "chat" in group_values
            assert "summarize" in group_values
        finally:
            path.unlink(missing_ok=True)

    async def test_export_json_accepts_string_path(self):
        await self.storage.save(_make_record(feature="chat"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path_str = tmp.name

        path = PathLibPath(path_str)
        try:
            await self.reporter.export_json(path_str, group_by="feature")
            assert path.exists()
            data = json.loads(path.read_text())
            assert isinstance(data, dict)
        finally:
            path.unlink(missing_ok=True)

    async def test_export_json_empty_storage(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = PathLibPath(tmp.name)

        try:
            await self.reporter.export_json(path, group_by="feature")
            data = json.loads(path.read_text())
            assert data["total_requests"] == 0
            assert data["breakdowns"] == []
        finally:
            path.unlink(missing_ok=True)
