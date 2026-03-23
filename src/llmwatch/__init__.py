"""llmwatch — LLM Cost Attribution library.

Track, tag, and report LLM API costs by feature, user, and model.

Example usage::

    from llmwatch import LLMWatch, Tags, UsageRecord

    watcher = LLMWatch()

    @watcher.tracked(feature="search", user_id="alice")
    async def call_llm():
        ...
"""

from llmwatch.budget import BudgetAlert
from llmwatch.databases.base import BaseStorage
from llmwatch.databases.sqlalchemy import Storage
from llmwatch.pricing.registry import PricingRegistry
from llmwatch.reporting import Reporter
from llmwatch.schemas.budget import BudgetRule
from llmwatch.schemas.pricing import ModelPricing
from llmwatch.schemas.reporting import CostReport, CostSummary
from llmwatch.schemas.tags import Tags
from llmwatch.schemas.usage import TokenUsage, UsageRecord
from llmwatch.tracker import LLMWatch

__all__ = [
    "BudgetAlert",
    "BudgetRule",
    "BaseStorage",
    "CostReport",
    "CostSummary",
    "LLMWatch",
    "ModelPricing",
    "PricingRegistry",
    "Reporter",
    "Storage",
    "Tags",
    "TokenUsage",
    "UsageRecord",
]
