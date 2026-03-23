"""Budget alert engine."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from llmwatch.schemas.budget import BudgetRule
from llmwatch.schemas.usage import UsageRecord

logger = logging.getLogger("llmwatch")


class BudgetAlert:
    """Invokes callbacks when cumulative cost exceeds a budget threshold."""

    def __init__(self) -> None:
        self._rules: list[tuple[BudgetRule, Callable[..., Any]]] = []
        self._cumulative: dict[int, float] = defaultdict(float)
        self._fired: set[int] = set()

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

    def reset(self) -> None:
        """Reset cumulative spend counters and fired state for all rules."""
        self._cumulative.clear()
        self._fired.clear()

    def get_cumulative_cost(self, rule_index: int) -> float:
        """Get the current cumulative cost for a rule by its index."""
        return self._cumulative.get(rule_index, 0.0)

    async def check(self, record: UsageRecord) -> None:
        for idx, (rule, callback) in enumerate(self._rules):
            if rule.feature and record.tags.feature != rule.feature:
                continue
            if rule.user_id and record.tags.user_id != rule.user_id:
                continue
            self._cumulative[idx] += record.cost_usd
            if idx not in self._fired and self._cumulative[idx] > rule.max_cost_usd:
                self._fired.add(idx)
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(record, cumulative_cost=self._cumulative[idx])
                    else:
                        callback(record, cumulative_cost=self._cumulative[idx])
                except Exception:
                    logger.exception("Budget alert callback failed for rule %s", rule)
