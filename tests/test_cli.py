"""Tests for the CLI module (llmwatch.cli)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from llmwatch.cli import app
from llmwatch.schemas.pricing import ModelPricing
from llmwatch.schemas.reporting import CostReport, CostSummary

runner = CliRunner()

# * Shared helpers


def _make_summary(
    breakdowns: list[CostReport] | None = None,
    total_requests: int = 0,
    total_cost_usd: float = 0.0,
    total_prompt_tokens: int = 0,
    total_completion_tokens: int = 0,
) -> CostSummary:
    """Build a CostSummary for use in mocked reporter returns."""
    return CostSummary(
        breakdowns=breakdowns or [],
        total_requests=total_requests,
        total_cost_usd=total_cost_usd,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
    )


def _make_breakdown(
    group_value: str = "chat",
    total_requests: int = 10,
    total_cost_usd: float = 0.05,
    total_prompt_tokens: int = 500,
    total_completion_tokens: int = 250,
) -> CostReport:
    """Build a CostReport breakdown entry."""
    return CostReport(
        group_key="feature",
        group_value=group_value,
        total_requests=total_requests,
        total_cost_usd=total_cost_usd,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
    )


def _make_pricing(
    provider: str = "openai",
    model: str = "gpt-4o",
    input_cost: float = 2.50,
    output_cost: float = 10.00,
) -> ModelPricing:
    return ModelPricing(
        provider=provider,
        model=model,
        input_cost_per_mtok=input_cost,
        output_cost_per_mtok=output_cost,
    )


# * Patch target helpers


def _patch_watch(mock_watch: MagicMock):
    """Return a context manager that replaces _make_watch in cli."""
    return patch("llmwatch.cli._make_watch", return_value=mock_watch)


def _build_mock_watch(summary: CostSummary | None = None) -> MagicMock:
    """Build a fully-wired mock LLMWatch with async stubs."""
    mock = MagicMock()
    mock.close = AsyncMock()
    mock.storage.count = AsyncMock(return_value=0)
    mock.storage.delete = AsyncMock(return_value=0)
    mock.report.summary = AsyncMock(return_value=summary or _make_summary())
    mock.report.export_csv = AsyncMock()
    mock.report.export_json = AsyncMock()
    mock.pricing.list_models = MagicMock(return_value=[])
    return mock


# * report command tests


class TestReportCommand:
    def test_report_default_options_exits_zero(self):
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["report"])
        assert result.exit_code == 0

    def test_report_shows_totals_row(self):
        summary = _make_summary(
            breakdowns=[_make_breakdown("chat", total_requests=5, total_cost_usd=0.01)],
            total_requests=5,
            total_cost_usd=0.01,
        )
        mock_watch = _build_mock_watch(summary)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["report"])
        assert result.exit_code == 0
        assert "TOTAL" in result.output
        assert "chat" in result.output

    def test_report_custom_group_by(self):
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["report", "--group-by", "user_id"])
        assert result.exit_code == 0
        # ^ summary() should have been called with group_by="user_id"
        mock_watch.report.summary.assert_called_once()
        call_kwargs = mock_watch.report.summary.call_args.kwargs
        assert call_kwargs["group_by"] == "user_id"

    def test_report_custom_period(self):
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["report", "--period", "7d"])
        assert result.exit_code == 0
        call_kwargs = mock_watch.report.summary.call_args.kwargs
        assert call_kwargs["period"] == "7d"

    def test_report_with_db_option(self):
        mock_watch = _build_mock_watch()
        with patch("llmwatch.cli._make_watch", return_value=mock_watch) as mock_factory:
            result = runner.invoke(app, ["report", "--db", "sqlite+aiosqlite:///test.db"])
        assert result.exit_code == 0
        mock_factory.assert_called_once_with("sqlite+aiosqlite:///test.db")

    def test_report_empty_database_shows_total_zero(self):
        summary = _make_summary(total_requests=0, total_cost_usd=0.0)
        mock_watch = _build_mock_watch(summary)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["report"])
        assert result.exit_code == 0
        assert "$0.0000" in result.output

    def test_report_multiple_breakdowns(self):
        summary = _make_summary(
            breakdowns=[
                _make_breakdown("chat", total_requests=3, total_cost_usd=0.02),
                _make_breakdown("summarize", total_requests=7, total_cost_usd=0.08),
            ],
            total_requests=10,
            total_cost_usd=0.10,
        )
        mock_watch = _build_mock_watch(summary)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["report"])
        assert result.exit_code == 0
        assert "chat" in result.output
        assert "summarize" in result.output

    def test_report_calls_close_on_watch(self):
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            runner.invoke(app, ["report"])
        mock_watch.close.assert_called_once()


# * export command tests


class TestExportCommand:
    def test_export_csv_default_format(self, tmp_path):
        output_path = str(tmp_path / "out.csv")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["export", output_path])
        assert result.exit_code == 0
        mock_watch.report.export_csv.assert_called_once()

    def test_export_json_format(self, tmp_path):
        output_path = str(tmp_path / "out.json")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["export", output_path, "--format", "json"])
        assert result.exit_code == 0
        mock_watch.report.export_json.assert_called_once()

    def test_export_unknown_format_exits_nonzero(self, tmp_path):
        output_path = str(tmp_path / "out.xml")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["export", output_path, "--format", "xml"])
        assert result.exit_code != 0

    def test_export_csv_passes_output_path(self, tmp_path):
        output_path = str(tmp_path / "report.csv")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            runner.invoke(app, ["export", output_path, "--format", "csv"])
        call_args = mock_watch.report.export_csv.call_args
        # ^ first positional arg is the output path
        assert call_args.args[0] == output_path

    def test_export_csv_passes_group_by_and_period(self, tmp_path):
        output_path = str(tmp_path / "report.csv")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            runner.invoke(
                app,
                ["export", output_path, "--group-by", "model", "--period", "7d"],
            )
        call_kwargs = mock_watch.report.export_csv.call_args.kwargs
        assert call_kwargs["group_by"] == "model"
        assert call_kwargs["period"] == "7d"

    def test_export_prints_success_message(self, tmp_path):
        output_path = str(tmp_path / "out.csv")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["export", output_path])
        assert "Exported" in result.output

    def test_export_calls_close_on_watch(self, tmp_path):
        output_path = str(tmp_path / "out.csv")
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            runner.invoke(app, ["export", output_path])
        mock_watch.close.assert_called_once()

    def test_export_with_db_option(self, tmp_path):
        output_path = str(tmp_path / "out.csv")
        mock_watch = _build_mock_watch()
        with patch("llmwatch.cli._make_watch", return_value=mock_watch) as mock_factory:
            runner.invoke(app, ["export", output_path, "--db", "sqlite+aiosqlite:///x.db"])
        mock_factory.assert_called_once_with("sqlite+aiosqlite:///x.db")


# * prune command tests


class TestPruneCommand:
    def test_prune_with_yes_flag_exits_zero(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=42)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["prune", "--yes"])
        assert result.exit_code == 0
        assert "42" in result.output

    def test_prune_default_days_is_90(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            runner.invoke(app, ["prune", "--yes"])
        call_kwargs = mock_watch.storage.delete.call_args.kwargs
        # ^ cutoff should be approximately 90 days ago
        cutoff: datetime = call_kwargs["before"]
        diff = datetime.now(UTC) - cutoff
        assert 89 < diff.days < 91

    def test_prune_custom_days(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            runner.invoke(app, ["prune", "--days", "30", "--yes"])
        call_kwargs = mock_watch.storage.delete.call_args.kwargs
        cutoff: datetime = call_kwargs["before"]
        diff = datetime.now(UTC) - cutoff
        assert 29 < diff.days < 31

    def test_prune_without_yes_prompts_user(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            # ^ Provide "n" to abort the confirmation prompt
            runner.invoke(app, ["prune"], input="n\n")
        # ^ Should abort without deleting
        mock_watch.storage.delete.assert_not_called()

    def test_prune_confirms_and_deletes_when_yes_entered(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=5)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["prune"], input="y\n")
        assert result.exit_code == 0
        mock_watch.storage.delete.assert_called_once()

    def test_prune_calls_close_on_watch(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            runner.invoke(app, ["prune", "--yes"])
        mock_watch.close.assert_called_once()

    def test_prune_with_db_option(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.delete = AsyncMock(return_value=0)
        with patch("llmwatch.cli._make_watch", return_value=mock_watch) as mock_factory:
            runner.invoke(app, ["prune", "--yes", "--db", "sqlite+aiosqlite:///p.db"])
        mock_factory.assert_called_once_with("sqlite+aiosqlite:///p.db")


# * stats command tests


class TestStatsCommand:
    def test_stats_exits_zero(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.count = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0

    def test_stats_shows_total_count(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.count = AsyncMock(return_value=1234)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["stats"])
        assert "1,234" in result.output

    def test_stats_zero_records(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.count = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["stats"])
        assert "0" in result.output

    def test_stats_calls_close_on_watch(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.count = AsyncMock(return_value=0)
        with _patch_watch(mock_watch):
            runner.invoke(app, ["stats"])
        mock_watch.close.assert_called_once()

    def test_stats_with_db_option(self):
        mock_watch = _build_mock_watch()
        mock_watch.storage.count = AsyncMock(return_value=0)
        with patch("llmwatch.cli._make_watch", return_value=mock_watch) as mock_factory:
            runner.invoke(app, ["stats", "--db", "sqlite+aiosqlite:///s.db"])
        mock_factory.assert_called_once_with("sqlite+aiosqlite:///s.db")


# * pricing sync command tests


class TestPricingSyncCommand:
    def test_pricing_sync_success(self):
        fake_entries = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "anthropic", "model": "claude-3"},
            {"provider": "openai", "model": "gpt-4o-mini"},
        ]
        with patch("llmwatch.cli.sync_pricing", return_value=fake_entries):
            result = runner.invoke(app, ["pricing", "sync"])
        assert result.exit_code == 0
        assert "3" in result.output
        assert "2" in result.output  # ^ 2 unique providers

    def test_pricing_sync_failure_exits_nonzero(self):
        with patch("llmwatch.cli.sync_pricing", side_effect=Exception("network error")):
            result = runner.invoke(app, ["pricing", "sync"])
        assert result.exit_code != 0
        assert "Sync failed" in result.output

    def test_pricing_sync_shows_provider_count(self):
        fake_entries = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "google", "model": "gemini-1.5-pro"},
        ]
        with patch("llmwatch.cli.sync_pricing", return_value=fake_entries):
            result = runner.invoke(app, ["pricing", "sync"])
        # ^ 2 providers: google, openai
        assert "2" in result.output


# * pricing list command tests


class TestPricingListCommand:
    def test_pricing_list_empty_shows_warning(self):
        mock_watch = _build_mock_watch()
        mock_watch.pricing.list_models = MagicMock(return_value=[])
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["pricing", "list"])
        assert result.exit_code == 0
        assert "No pricing data" in result.output

    def test_pricing_list_shows_models(self):
        mock_watch = _build_mock_watch()
        mock_watch.pricing.list_models = MagicMock(
            return_value=[
                _make_pricing("openai", "gpt-4o", 2.50, 10.00),
                _make_pricing("anthropic", "claude-3-5-sonnet", 3.00, 15.00),
            ]
        )
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["pricing", "list"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output
        assert "claude-3-5-sonnet" in result.output

    def test_pricing_list_with_provider_filter(self):
        mock_watch = _build_mock_watch()
        mock_watch.pricing.list_models = MagicMock(return_value=[_make_pricing("openai", "gpt-4o")])
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["pricing", "list", "--provider", "openai"])
        assert result.exit_code == 0
        mock_watch.pricing.list_models.assert_called_once_with("openai")

    def test_pricing_list_no_provider_filter_passes_none(self):
        mock_watch = _build_mock_watch()
        with _patch_watch(mock_watch):
            runner.invoke(app, ["pricing", "list"])
        mock_watch.pricing.list_models.assert_called_once_with(None)

    def test_pricing_list_shows_cost_columns(self):
        mock_watch = _build_mock_watch()
        mock_watch.pricing.list_models = MagicMock(
            return_value=[_make_pricing("openai", "gpt-4o", 2.50, 10.00)]
        )
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["pricing", "list"])
        assert "$2.50" in result.output
        assert "$10.00" in result.output

    def test_pricing_list_shows_cache_costs_when_nonzero(self):
        mock_watch = _build_mock_watch()
        pricing = ModelPricing(
            provider="anthropic",
            model="claude-3-5-sonnet",
            input_cost_per_mtok=3.00,
            output_cost_per_mtok=15.00,
            cache_creation_cost_per_mtok=3.75,
            cache_read_cost_per_mtok=0.30,
        )
        mock_watch.pricing.list_models = MagicMock(return_value=[pricing])
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["pricing", "list"])
        assert "$3.75" in result.output
        assert "$0.30" in result.output

    def test_pricing_list_shows_dash_for_zero_cache_costs(self):
        # ^ ModelPricing uses float defaults (0.0) — CLI renders "-" when value is falsy (0.0)
        mock_watch = _build_mock_watch()
        pricing = ModelPricing(
            provider="openai",
            model="gpt-4o",
            input_cost_per_mtok=2.50,
            output_cost_per_mtok=10.00,
            cache_creation_cost_per_mtok=0.0,
            cache_read_cost_per_mtok=0.0,
        )
        mock_watch.pricing.list_models = MagicMock(return_value=[pricing])
        with _patch_watch(mock_watch):
            result = runner.invoke(app, ["pricing", "list"])
        assert "-" in result.output


# * dashboard command tests


def _make_fake_dashboard_module(fake_dash_app: MagicMock) -> MagicMock:
    """Build a fake llmwatch.dashboard module with create_dashboard_app."""
    fake_module = MagicMock()
    fake_module.create_dashboard_app = MagicMock(return_value=fake_dash_app)
    return fake_module


class TestDashboardCommand:
    def test_dashboard_exits_nonzero_when_uvicorn_missing(self):
        with patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(app, ["dashboard"])
        assert result.exit_code != 0

    def test_dashboard_exits_nonzero_when_llmwatch_dashboard_missing(self):
        fake_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn, "llmwatch.dashboard": None}):
            result = runner.invoke(app, ["dashboard"])
        assert result.exit_code != 0

    def test_dashboard_runs_with_deps_available(self):
        fake_uvicorn = MagicMock()
        fake_dash_app = MagicMock()
        fake_dashboard_mod = _make_fake_dashboard_module(fake_dash_app)
        with patch.dict(
            "sys.modules",
            {"uvicorn": fake_uvicorn, "llmwatch.dashboard": fake_dashboard_mod},
        ):
            result = runner.invoke(app, ["dashboard"])
        assert result.exit_code == 0
        fake_uvicorn.run.assert_called_once_with(fake_dash_app, host="127.0.0.1", port=8000)

    def test_dashboard_custom_host_and_port(self):
        fake_uvicorn = MagicMock()
        fake_dash_app = MagicMock()
        fake_dashboard_mod = _make_fake_dashboard_module(fake_dash_app)
        with patch.dict(
            "sys.modules",
            {"uvicorn": fake_uvicorn, "llmwatch.dashboard": fake_dashboard_mod},
        ):
            result = runner.invoke(app, ["dashboard", "--host", "0.0.0.0", "--port", "9000"])
        assert result.exit_code == 0
        fake_uvicorn.run.assert_called_once_with(fake_dash_app, host="0.0.0.0", port=9000)

    def test_dashboard_prints_url(self):
        fake_uvicorn = MagicMock()
        fake_dash_app = MagicMock()
        fake_dashboard_mod = _make_fake_dashboard_module(fake_dash_app)
        with patch.dict(
            "sys.modules",
            {"uvicorn": fake_uvicorn, "llmwatch.dashboard": fake_dashboard_mod},
        ):
            result = runner.invoke(app, ["dashboard"])
        assert "127.0.0.1:8000" in result.output

    def test_dashboard_passes_db_to_create_dashboard_app(self):
        fake_uvicorn = MagicMock()
        fake_dash_app = MagicMock()
        fake_dashboard_mod = _make_fake_dashboard_module(fake_dash_app)
        with patch.dict(
            "sys.modules",
            {"uvicorn": fake_uvicorn, "llmwatch.dashboard": fake_dashboard_mod},
        ):
            runner.invoke(app, ["dashboard", "--db", "sqlite+aiosqlite:///dash.db"])
        fake_dashboard_mod.create_dashboard_app.assert_called_once_with(
            storage_url="sqlite+aiosqlite:///dash.db"
        )


# * _make_watch helper tests


class TestMakeWatch:
    def test_make_watch_without_db_uses_default_storage(self):
        """_make_watch(None) should create LLMWatch with default storage."""
        from llmwatch.cli import _make_watch

        watch = _make_watch(None)
        try:
            assert watch is not None
        finally:
            # ^ Synchronously close to avoid resource leak in tests
            import asyncio

            asyncio.run(watch.close())

    def test_make_watch_with_db_url_creates_storage(self):
        """_make_watch with a URL should create Storage with that URL."""
        from llmwatch.cli import _make_watch
        from llmwatch.databases.sqlalchemy import Storage

        watch = _make_watch("sqlite+aiosqlite://")
        try:
            assert isinstance(watch._storage, Storage)
        finally:
            import asyncio

            asyncio.run(watch.close())
