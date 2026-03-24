"""Cost report aggregation module."""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path as PathLibPath
from typing import Any

from llmwatch.databases.base import BaseStorage
from llmwatch.schemas.reporting import AggregationPeriod, CostSummary


def _period_to_start(period: str) -> datetime:
    """Convert a period string to a start datetime. e.g. '7d', '24h', '30d'."""
    if len(period) < 2:
        msg = f"Invalid period string: {period!r}. Expected format like '7d', '24h'."
        raise ValueError(msg)
    now = datetime.now(UTC)
    unit = period[-1]
    value = int(period[:-1])
    if value <= 0:
        msg = f"Period value must be positive, got: {value}"
        raise ValueError(msg)
    if unit == "h":
        return now - timedelta(hours=value)
    if unit == "d":
        return now - timedelta(days=value)
    if unit == "w":
        return now - timedelta(weeks=value)
    if unit == "m":
        return now - timedelta(days=value * 30)
    msg = f"Unknown period unit: {unit}. Use h/d/w/m."
    raise ValueError(msg)


class Reporter:
    """Cost report generator."""

    def __init__(self, storage: BaseStorage) -> None:
        self._storage = storage

    async def summary(
        self,
        *,
        group_by: str = "feature",
        period: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        aggregation_period: AggregationPeriod | None = None,
    ) -> CostSummary:
        """Generate an aggregated summary report."""
        if period is not None:
            start_time = _period_to_start(period)

        breakdowns = await self._storage.aggregate(
            group_by=group_by,
            start_time=start_time,
            end_time=end_time,
            period=aggregation_period,
        )

        total_cost = sum(b.total_cost_usd for b in breakdowns)
        total_requests = sum(b.total_requests for b in breakdowns)
        total_prompt = sum(b.total_prompt_tokens for b in breakdowns)
        total_completion = sum(b.total_completion_tokens for b in breakdowns)

        return CostSummary(
            total_cost_usd=total_cost,
            total_requests=total_requests,
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            period_start=start_time,
            period_end=end_time,
            breakdowns=breakdowns,
        )

    async def by_feature(self, *, period: str = "30d", **kwargs: Any) -> CostSummary:
        return await self.summary(group_by="feature", period=period, **kwargs)

    async def by_user_id(self, *, period: str = "30d", **kwargs: Any) -> CostSummary:
        return await self.summary(group_by="user_id", period=period, **kwargs)

    async def by_model(self, *, period: str = "30d", **kwargs: Any) -> CostSummary:
        return await self.summary(group_by="model", period=period, **kwargs)

    async def by_provider(self, *, period: str = "30d", **kwargs: Any) -> CostSummary:
        return await self.summary(group_by="provider", period=period, **kwargs)

    async def export_csv(
        self, path: PathLibPath | str, *, group_by: str = "feature", period: str = "30d"
    ) -> None:
        report = await self.summary(group_by=group_by, period=period)
        filepath = PathLibPath(path)
        with filepath.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    group_by,
                    "requests",
                    "prompt_tokens",
                    "completion_tokens",
                    "cost_usd",
                ]
            )
            for b in report.breakdowns:
                writer.writerow(
                    [
                        b.group_value,
                        b.total_requests,
                        b.total_prompt_tokens,
                        b.total_completion_tokens,
                        f"{b.total_cost_usd:.6f}",
                    ]
                )

    async def export_json(
        self, path: PathLibPath | str, *, group_by: str = "feature", period: str = "30d"
    ) -> None:
        report = await self.summary(group_by=group_by, period=period)
        filepath = PathLibPath(path)
        with filepath.open("w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
