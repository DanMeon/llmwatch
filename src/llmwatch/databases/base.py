"""Storage protocol — structural interface for persistence backends."""

from datetime import datetime
from typing import Protocol, runtime_checkable

from llmwatch.schemas.reporting import AggregationPeriod, CostReport
from llmwatch.schemas.usage import UsageRecord


@runtime_checkable
class BaseStorage(Protocol):
    """Structural interface for storage backends.

    Any class implementing these methods satisfies the protocol —
    no inheritance required.
    """

    async def save(self, record: UsageRecord) -> None:
        """Save a single record."""
        ...

    async def save_batch(self, records: list[UsageRecord]) -> None:
        """Save a batch of records."""
        ...

    async def query(
        self,
        *,
        feature: str | None = None,
        user_id: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        environment: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[UsageRecord]:
        """Query records by conditions."""
        ...

    async def aggregate(
        self,
        *,
        group_by: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        period: AggregationPeriod | None = None,
    ) -> list[CostReport]:
        """Aggregation query."""
        ...

    async def delete(
        self,
        *,
        before: datetime | None = None,
        feature: str | None = None,
    ) -> int:
        """Delete records. Returns the number of deleted rows."""
        ...

    async def count(self) -> int:
        """Total record count."""
        ...

    async def close(self) -> None:
        """Clean up resources."""
        ...
