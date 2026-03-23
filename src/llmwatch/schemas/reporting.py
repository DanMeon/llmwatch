"""Cost report and summary schemas."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class AggregationPeriod(StrEnum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class CostReport(BaseModel):
    """Aggregated cost report entry."""

    group_key: str
    group_value: str
    total_cost_usd: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_requests: int
    period_start: datetime | None = None
    period_end: datetime | None = None


class CostSummary(BaseModel):
    """Overall cost summary report."""

    total_cost_usd: float
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    period_start: datetime | None = None
    period_end: datetime | None = None
    breakdowns: list[CostReport] = Field(default_factory=list)
