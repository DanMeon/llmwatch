"""Budget rule schema."""

from pydantic import BaseModel


class BudgetRule(BaseModel):
    """Budget alert rule."""

    max_cost_usd: float
    feature: str | None = None
    user_id: str | None = None
