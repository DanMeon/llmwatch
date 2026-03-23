"""Smoke tests for the llmwatch web dashboard (Starlette app).

starlette is an optional dependency — all tests are skipped if it is not installed.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

starlette = pytest.importorskip("starlette")

# ^ Import after importorskip so the skip propagates before any ImportError
from starlette.testclient import TestClient  # noqa: E402

from llmwatch.dashboard import create_dashboard_app  # noqa: E402
from llmwatch.dashboard_html import DASHBOARD_HTML  # noqa: E402
from llmwatch.schemas.reporting import CostReport, CostSummary  # noqa: E402
from llmwatch.schemas.tags import Tags  # noqa: E402
from llmwatch.schemas.usage import TokenUsage, UsageRecord  # noqa: E402

# * Helpers


def _make_cost_report(
    group_key: str = "feature",
    group_value: str = "chat",
    cost: float = 0.05,
    requests: int = 10,
    prompt_tokens: int = 1000,
    completion_tokens: int = 500,
) -> CostReport:
    return CostReport(
        group_key=group_key,
        group_value=group_value,
        total_cost_usd=cost,
        total_requests=requests,
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
    )


def _make_usage_record(
    model: str = "gpt-4o",
    provider: str = "openai",
    feature: str = "chat",
    user_id: str = "alice",
    cost: float = 0.01,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    latency_ms: float = 200.0,
) -> UsageRecord:
    return UsageRecord(
        id=uuid4(),
        timestamp=datetime.now(UTC),
        model=model,
        provider=provider,
        tags=Tags(feature=feature, user_id=user_id),
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        cost_usd=cost,
        latency_ms=latency_ms,
    )


def _make_cost_summary(breakdowns: list[CostReport] | None = None) -> CostSummary:
    reports = breakdowns or []
    return CostSummary(
        total_cost_usd=sum(r.total_cost_usd for r in reports),
        total_requests=sum(r.total_requests for r in reports),
        total_prompt_tokens=sum(r.total_prompt_tokens for r in reports),
        total_completion_tokens=sum(r.total_completion_tokens for r in reports),
        breakdowns=reports,
    )


# * Fixtures


@pytest.fixture
def mock_storage() -> MagicMock:
    """In-memory mock storage that satisfies the BaseStorage protocol."""
    storage = MagicMock()
    storage.close = AsyncMock()
    storage.query = AsyncMock(return_value=[])
    storage.aggregate = AsyncMock(return_value=[])
    storage.count = AsyncMock(return_value=0)
    return storage


@pytest.fixture
def app_with_mock(mock_storage: MagicMock) -> TestClient:
    """Dashboard app backed by mock_storage via module-level patches."""
    with (
        patch("llmwatch.dashboard.Storage", return_value=mock_storage),
        patch("llmwatch.dashboard._default_storage_url", return_value="sqlite+aiosqlite://"),
    ):
        app = create_dashboard_app(storage_url="sqlite+aiosqlite://")
    return TestClient(app)


# * Test: app creation


class TestAppCreation:
    def test_create_dashboard_app_returns_starlette(self, mock_storage: MagicMock):
        """create_dashboard_app() should return a Starlette application."""
        from starlette.applications import Starlette

        with patch("llmwatch.dashboard.Storage", return_value=mock_storage):
            app = create_dashboard_app(storage_url="sqlite+aiosqlite://")

        assert isinstance(app, Starlette)

    def test_app_has_expected_routes(self, mock_storage: MagicMock):
        """The app should expose /, /api/overview, /api/summary, /api/records."""
        with patch("llmwatch.dashboard.Storage", return_value=mock_storage):
            app = create_dashboard_app(storage_url="sqlite+aiosqlite://")

        route_paths = {route.path for route in app.routes}
        assert "/" in route_paths
        assert "/api/overview" in route_paths
        assert "/api/summary" in route_paths
        assert "/api/records" in route_paths


# * Test: HTML endpoint


class TestIndexEndpoint:
    def test_index_returns_200(self, app_with_mock: TestClient):
        response = app_with_mock.get("/")
        assert response.status_code == 200

    def test_index_content_type_is_html(self, app_with_mock: TestClient):
        response = app_with_mock.get("/")
        assert "text/html" in response.headers["content-type"]

    def test_index_body_matches_dashboard_html(self, app_with_mock: TestClient):
        response = app_with_mock.get("/")
        assert response.text == DASHBOARD_HTML

    def test_index_contains_llmwatch_brand(self, app_with_mock: TestClient):
        response = app_with_mock.get("/")
        assert "llmwatch" in response.text


# * Test: /api/overview endpoint


class TestOverviewEndpoint:
    def test_overview_returns_200(self, app_with_mock: TestClient, mock_storage: MagicMock):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/overview")
        assert response.status_code == 200

    def test_overview_default_period_is_30d(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/overview")
        data = response.json()
        assert data["period"] == "30d"

    def test_overview_period_param_is_reflected(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/overview?period=7d")
        data = response.json()
        assert data["period"] == "7d"

    def test_overview_invalid_period_returns_400(self, app_with_mock: TestClient):
        response = app_with_mock.get("/api/overview?period=99x")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_overview_empty_storage_returns_zeros(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/overview")
        data = response.json()
        assert data["total_cost_usd"] == 0.0
        assert data["total_requests"] == 0
        assert data["top_model"] is None
        assert data["top_feature"] is None

    def test_overview_with_records_returns_top_model_and_feature(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        feature_report = _make_cost_report(group_key="feature", group_value="summarize", cost=0.10)
        model_report = _make_cost_report(group_key="model", group_value="gpt-4o", cost=0.10)

        # ^ aggregate is called twice: once for feature, once for model
        mock_storage.aggregate = AsyncMock(
            side_effect=[
                [feature_report],
                [model_report],
            ]
        )

        response = app_with_mock.get("/api/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["top_feature"] == "summarize"
        assert data["top_model"] == "gpt-4o"
        assert abs(data["total_cost_usd"] - 0.10) < 1e-9

    def test_overview_response_has_required_keys(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/overview")
        data = response.json()
        required = {
            "total_cost_usd",
            "total_requests",
            "total_prompt_tokens",
            "total_completion_tokens",
            "top_model",
            "top_model_cost",
            "top_feature",
            "top_feature_cost",
            "period",
        }
        assert required.issubset(data.keys())


# * Test: /api/summary endpoint


class TestSummaryEndpoint:
    def test_summary_returns_200(self, app_with_mock: TestClient, mock_storage: MagicMock):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/summary")
        assert response.status_code == 200

    def test_summary_default_group_by_is_feature(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/summary")
        data = response.json()
        # ^ CostSummary.model_dump() returns a dict with breakdowns list
        assert "breakdowns" in data
        assert "total_cost_usd" in data

    @pytest.mark.parametrize("group_by", ["feature", "user_id", "model", "provider"])
    def test_summary_accepts_valid_group_by(
        self, app_with_mock: TestClient, mock_storage: MagicMock, group_by: str
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get(f"/api/summary?group_by={group_by}")
        assert response.status_code == 200

    def test_summary_rejects_invalid_group_by(self, app_with_mock: TestClient):
        response = app_with_mock.get("/api/summary?group_by=unknown")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_summary_invalid_period_returns_400(self, app_with_mock: TestClient):
        response = app_with_mock.get("/api/summary?period=99x")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_summary_empty_returns_zero_totals(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.aggregate = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/summary")
        data = response.json()
        assert data["total_cost_usd"] == 0.0
        assert data["total_requests"] == 0
        assert data["breakdowns"] == []

    def test_summary_with_data_returns_breakdowns(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        reports = [
            _make_cost_report(group_value="chat", cost=0.05, requests=10),
            _make_cost_report(group_value="translate", cost=0.02, requests=4),
        ]
        mock_storage.aggregate = AsyncMock(return_value=reports)

        response = app_with_mock.get("/api/summary?group_by=feature")
        assert response.status_code == 200
        data = response.json()
        assert len(data["breakdowns"]) == 2
        assert abs(data["total_cost_usd"] - 0.07) < 1e-9
        assert data["total_requests"] == 14

    def test_summary_breakdown_has_required_fields(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        report = _make_cost_report(group_value="chat")
        mock_storage.aggregate = AsyncMock(return_value=[report])

        response = app_with_mock.get("/api/summary")
        breakdown = response.json()["breakdowns"][0]
        required = {
            "group_key",
            "group_value",
            "total_cost_usd",
            "total_requests",
            "total_prompt_tokens",
            "total_completion_tokens",
        }
        assert required.issubset(breakdown.keys())


# * Test: /api/records endpoint


class TestRecordsEndpoint:
    def test_records_returns_200(self, app_with_mock: TestClient):
        response = app_with_mock.get("/api/records")
        assert response.status_code == 200

    def test_records_returns_json_list(self, app_with_mock: TestClient):
        response = app_with_mock.get("/api/records")
        assert isinstance(response.json(), list)

    def test_records_empty_storage_returns_empty_list(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        response = app_with_mock.get("/api/records")
        assert response.json() == []

    def test_records_invalid_limit_returns_400(self, app_with_mock: TestClient):
        response = app_with_mock.get("/api/records?limit=notanumber")
        assert response.status_code == 400
        assert "error" in response.json()

    def test_records_with_data_returns_list(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        records = [
            _make_usage_record(feature="chat", cost=0.01),
            _make_usage_record(feature="summarize", cost=0.02),
        ]
        mock_storage.query = AsyncMock(return_value=records)

        response = app_with_mock.get("/api/records")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_records_item_has_required_fields(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        record = _make_usage_record()
        mock_storage.query = AsyncMock(return_value=[record])

        response = app_with_mock.get("/api/records")
        item = response.json()[0]
        required = {"id", "model", "provider", "cost_usd", "token_usage", "tags", "timestamp"}
        assert required.issubset(item.keys())

    def test_records_limit_param_is_forwarded(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?limit=25")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["limit"] == 25

    def test_records_limit_clamped_to_500(self, app_with_mock: TestClient, mock_storage: MagicMock):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?limit=9999")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["limit"] == 500

    def test_records_limit_clamped_to_minimum_1(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?limit=0")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["limit"] == 1

    def test_records_feature_filter_forwarded(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?feature=chat")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["feature"] == "chat"

    def test_records_model_filter_forwarded(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?model=gpt-4o")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_records_provider_filter_forwarded(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?provider=anthropic")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["provider"] == "anthropic"

    def test_records_user_id_filter_forwarded(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?user_id=alice")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["user_id"] == "alice"

    def test_records_empty_filter_params_become_none(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        """Empty string query params should be treated as None (not passed as empty string)."""
        mock_storage.query = AsyncMock(return_value=[])
        app_with_mock.get("/api/records?feature=&user_id=")
        call_kwargs = mock_storage.query.call_args.kwargs
        assert call_kwargs["feature"] is None
        assert call_kwargs["user_id"] is None

    def test_records_storage_exception_returns_500(
        self, app_with_mock: TestClient, mock_storage: MagicMock
    ):
        mock_storage.query = AsyncMock(side_effect=RuntimeError("DB is down"))
        response = app_with_mock.get("/api/records")
        assert response.status_code == 500
        assert "error" in response.json()


# * Test: integration with in-memory SQLite


class TestIntegrationWithRealStorage:
    """Smoke tests using real in-memory SQLite — no mocking."""

    @pytest.fixture
    def real_app(self) -> TestClient:
        app = create_dashboard_app(storage_url="sqlite+aiosqlite://")
        return TestClient(app)

    def test_index_200(self, real_app: TestClient):
        assert real_app.get("/").status_code == 200

    def test_overview_200_empty_db(self, real_app: TestClient):
        response = real_app.get("/api/overview")
        assert response.status_code == 200
        data = response.json()
        assert data["total_cost_usd"] == 0.0
        assert data["total_requests"] == 0

    def test_summary_200_empty_db(self, real_app: TestClient):
        response = real_app.get("/api/summary")
        assert response.status_code == 200
        assert response.json()["breakdowns"] == []

    def test_records_200_empty_db(self, real_app: TestClient):
        response = real_app.get("/api/records")
        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.parametrize("group_by", ["feature", "user_id", "model", "provider"])
    def test_summary_all_group_by_values(self, real_app: TestClient, group_by: str):
        response = real_app.get(f"/api/summary?group_by={group_by}")
        assert response.status_code == 200

    @pytest.mark.parametrize("period", ["24h", "7d", "30d", "90d"])
    def test_overview_all_standard_periods(self, real_app: TestClient, period: str):
        response = real_app.get(f"/api/overview?period={period}")
        assert response.status_code == 200
