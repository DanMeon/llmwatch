"""Web dashboard for llmwatch — Starlette-based single-page app.

Starlette is an optional dependency. Install with:
    pip install llmwatch[dashboard]
or:
    uv add "llmwatch[dashboard]"
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

try:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route
except ImportError as exc:
    raise ImportError(
        "Dashboard requires starlette. Install it with: pip install llmwatch[dashboard]"
    ) from exc

from llmwatch.dashboard_html import DASHBOARD_HTML
from llmwatch.databases.sqlalchemy import Storage
from llmwatch.reporting import Reporter, _period_to_start

logger = logging.getLogger("llmwatch")


def _default_storage_url() -> str:
    from pathlib import Path as PathLibPath

    home = PathLibPath.home()
    db_dir = home / ".llmwatch"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_dir / 'usage.db'}"


def create_dashboard_app(storage_url: str | None = None) -> Any:
    """Create and return the Starlette dashboard application.

    Args:
        storage_url: SQLAlchemy async connection URL. Defaults to the
            standard llmwatch SQLite database at ~/.llmwatch/usage.db.
    """
    url = storage_url or _default_storage_url()
    storage = Storage(url)
    reporter = Reporter(storage)

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncGenerator[None, None]:
        yield
        await storage.close()

    # * HTML page
    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(DASHBOARD_HTML)

    # * GET /api/overview?period=30d
    async def api_overview(request: Request) -> JSONResponse:
        period = request.query_params.get("period", "30d")
        try:
            _period_to_start(period)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        # ^ Fetch feature and model breakdowns in parallel
        import asyncio

        feature_summary, model_summary = await asyncio.gather(
            reporter.summary(group_by="feature", period=period),
            reporter.summary(group_by="model", period=period),
        )

        top_model = None
        top_model_cost = None
        if model_summary.breakdowns:
            top = model_summary.breakdowns[0]
            top_model = top.group_value
            top_model_cost = top.total_cost_usd

        top_feature = None
        top_feature_cost = None
        if feature_summary.breakdowns:
            top = feature_summary.breakdowns[0]
            top_feature = top.group_value
            top_feature_cost = top.total_cost_usd

        return JSONResponse(
            {
                "total_cost_usd": feature_summary.total_cost_usd,
                "total_requests": feature_summary.total_requests,
                "total_prompt_tokens": feature_summary.total_prompt_tokens,
                "total_completion_tokens": feature_summary.total_completion_tokens,
                "top_model": top_model,
                "top_model_cost": top_model_cost,
                "top_feature": top_feature,
                "top_feature_cost": top_feature_cost,
                "period": period,
            }
        )

    # * GET /api/summary?group_by=feature&period=30d
    async def api_summary(request: Request) -> JSONResponse:
        group_by = request.query_params.get("group_by", "feature")
        period = request.query_params.get("period", "30d")

        valid_groups = {"feature", "user_id", "model", "provider"}
        if group_by not in valid_groups:
            return JSONResponse(
                {"error": f"group_by must be one of: {', '.join(sorted(valid_groups))}"},
                status_code=400,
            )

        try:
            summary = await reporter.summary(group_by=group_by, period=period)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        return JSONResponse(summary.model_dump(mode="json"))

    # * GET /api/records?limit=50&feature=...&user_id=...
    async def api_records(request: Request) -> JSONResponse:
        params = request.query_params

        try:
            limit = int(params.get("limit", 50))
        except ValueError:
            return JSONResponse({"error": "limit must be an integer"}, status_code=400)

        limit = max(1, min(limit, 500))

        feature = params.get("feature") or None
        user_id = params.get("user_id") or None
        model = params.get("model") or None
        provider = params.get("provider") or None

        try:
            records = await storage.query(
                feature=feature,
                user_id=user_id,
                model=model,
                provider=provider,
                limit=limit,
            )
        except Exception:
            logger.exception("Failed to query records")
            return JSONResponse({"error": "Internal server error"}, status_code=500)

        return JSONResponse([r.model_dump(mode="json") for r in records])

    routes = [
        Route("/", endpoint=index),
        Route("/api/overview", endpoint=api_overview),
        Route("/api/summary", endpoint=api_summary),
        Route("/api/records", endpoint=api_records),
    ]

    return Starlette(routes=routes, lifespan=lifespan)
