"""Budget alert engine."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from llmwatch.schemas.budget import BudgetRule
from llmwatch.schemas.usage import UsageRecord

logger = logging.getLogger("llmwatch")


class BudgetAlert:
    """Invokes callbacks when cost exceeds a threshold."""

    def __init__(self) -> None:
        self._rules: list[tuple[BudgetRule, Callable[..., Any]]] = []

    def add_rule(
        self,
        *,
        max_cost_usd: float,
        callback: Callable[..., Any],
        feature: str | None = None,
        user_id: str | None = None,
    ) -> None:
        rule = BudgetRule(max_cost_usd=max_cost_usd, feature=feature, user_id=user_id)
        self._rules.append((rule, callback))

    async def check(self, record: UsageRecord) -> None:
        for rule, callback in self._rules:
            if rule.feature and record.tags.feature != rule.feature:
                continue
            if rule.user_id and record.tags.user_id != rule.user_id:
                continue
            if record.cost_usd > rule.max_cost_usd:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(record)
                    else:
                        callback(record)
                except Exception:
                    logger.exception("Budget alert callback failed for rule %s", rule)
