"""SQLAlchemy-based storage."""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    select,
    text,
)
from sqlalchemy import (
    inspect as sa_inspect,
)
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from llmwatch.schemas.reporting import AggregationPeriod, CostReport
from llmwatch.schemas.tags import Tags
from llmwatch.schemas.usage import TokenUsage, UsageRecord

logger = logging.getLogger("llmwatch")

_GROUPABLE_COLUMNS = {"feature", "user_id", "model", "provider", "environment"}


class Storage:
    """SQLAlchemy async storage backend."""

    def __init__(
        self,
        connection_url: str,
        *,
        table_name: str = "llmwatch_usage",
        schema: str | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._engine: AsyncEngine = create_async_engine(
            connection_url,
            **(engine_kwargs or {}),
        )
        self._metadata = MetaData(schema=schema)
        self._table = self._define_table(table_name)
        self._initialized = False

    async def _ensure_tables(self) -> None:
        if self._initialized:
            return
        async with self._engine.begin() as conn:
            await conn.run_sync(self._metadata.create_all)
            await conn.run_sync(self._auto_migrate)
        self._initialized = True

    def _auto_migrate(self, conn: Any) -> None:
        """Auto-add missing columns to existing tables."""
        inspector = sa_inspect(conn)
        table_name = self._table.name
        schema = self._table.schema

        if not inspector.has_table(table_name, schema=schema):
            return

        existing = {col["name"] for col in inspector.get_columns(table_name, schema=schema)}
        dialect = conn.engine.dialect
        for column in self._table.columns:
            if column.name not in existing:
                col_type = column.type.compile(dialect)
                # ^ Use dialect-specific identifier quoting to prevent SQL injection
                preparer = dialect.identifier_preparer
                if schema:
                    quoted_table = f"{preparer.quote_identifier(schema)}.{preparer.quote_identifier(table_name)}"
                else:
                    quoted_table = preparer.quote_identifier(table_name)
                quoted_col = preparer.quote_identifier(column.name)
                sql = f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_col} {col_type}"
                conn.execute(text(sql))
                logger.info("Auto-migrated: added column %s.%s", table_name, column.name)

    def _define_table(self, table_name: str) -> Table:
        table = Table(
            table_name,
            self._metadata,
            Column("id", String(36), primary_key=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("feature", String(255), nullable=True),
            Column("user_id", String(255), nullable=True),
            Column("model", String(255), nullable=False),
            Column("provider", String(100), nullable=False),
            Column("environment", String(100), nullable=True),
            Column("extra_tags", Text, nullable=True),
            Column("prompt_tokens", Integer, nullable=False, default=0),
            Column("completion_tokens", Integer, nullable=False, default=0),
            Column("total_tokens", Integer, nullable=False, default=0),
            Column("cache_creation_input_tokens", Integer, nullable=False, default=0),
            Column("cache_read_input_tokens", Integer, nullable=False, default=0),
            Column("cost_usd", Float, nullable=False),
            Column("latency_ms", Float, nullable=True),
            Column("raw_response_id", String(255), nullable=True),
            Column("input_data", Text, nullable=True),
            Column("output_data", Text, nullable=True),
        )
        Index(f"idx_{table_name}_timestamp", table.c.timestamp)
        Index(f"idx_{table_name}_feature", table.c.feature)
        Index(f"idx_{table_name}_user_id", table.c.user_id)
        Index(f"idx_{table_name}_model", table.c.model)
        return table

    @staticmethod
    def _serialize_input(input_data: Any) -> str | None:
        if input_data is None:
            return None
        if isinstance(input_data, str):
            return input_data
        return json.dumps(input_data, ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize_input(raw: str | None) -> Any:
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def _record_to_dict(self, record: UsageRecord) -> dict[str, Any]:
        return {
            "id": str(record.id),
            "timestamp": record.timestamp,
            "feature": record.tags.feature,
            "user_id": record.tags.user_id,
            "model": record.model,
            "provider": record.provider,
            "environment": record.tags.environment,
            "extra_tags": json.dumps(record.tags.extra) if record.tags.extra else None,
            "prompt_tokens": record.token_usage.prompt_tokens,
            "completion_tokens": record.token_usage.completion_tokens,
            "total_tokens": record.token_usage.total_tokens,
            "cache_creation_input_tokens": record.token_usage.cache_creation_input_tokens,
            "cache_read_input_tokens": record.token_usage.cache_read_input_tokens,
            "cost_usd": record.cost_usd,
            "latency_ms": record.latency_ms,
            "raw_response_id": record.raw_response_id,
            "input_data": self._serialize_input(record.input_data),
            "output_data": record.output_data,
        }

    def _row_to_record(self, row: Any) -> UsageRecord:
        extra = json.loads(row.extra_tags) if row.extra_tags else {}
        return UsageRecord(
            id=UUID(row.id),
            timestamp=row.timestamp,
            tags=Tags(
                feature=row.feature,
                user_id=row.user_id,
                environment=row.environment,
                extra=extra,
            ),
            model=row.model,
            provider=row.provider,
            token_usage=TokenUsage(
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                total_tokens=row.total_tokens,
                cache_creation_input_tokens=row.cache_creation_input_tokens,
                cache_read_input_tokens=row.cache_read_input_tokens,
            ),
            cost_usd=row.cost_usd,
            latency_ms=row.latency_ms,
            raw_response_id=row.raw_response_id,
            input_data=self._deserialize_input(row.input_data),
            output_data=row.output_data,
        )

    async def save(self, record: UsageRecord) -> None:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(self._table.insert().values(self._record_to_dict(record)))

    async def save_batch(self, records: list[UsageRecord]) -> None:
        await self._ensure_tables()
        async with self._engine.begin() as conn:
            await conn.execute(
                self._table.insert(),
                [self._record_to_dict(r) for r in records],
            )

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
        await self._ensure_tables()
        t = self._table
        stmt = select(t).order_by(t.c.timestamp.desc())

        if feature is not None:
            stmt = stmt.where(t.c.feature == feature)
        if user_id is not None:
            stmt = stmt.where(t.c.user_id == user_id)
        if model is not None:
            stmt = stmt.where(t.c.model == model)
        if provider is not None:
            stmt = stmt.where(t.c.provider == provider)
        if environment is not None:
            stmt = stmt.where(t.c.environment == environment)
        if start_time is not None:
            stmt = stmt.where(t.c.timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(t.c.timestamp <= end_time)

        stmt = stmt.limit(limit).offset(offset)

        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            return [self._row_to_record(row) for row in result.fetchall()]

    async def aggregate(
        self,
        *,
        group_by: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        period: AggregationPeriod | None = None,
    ) -> list[CostReport]:
        await self._ensure_tables()
        if group_by not in _GROUPABLE_COLUMNS:
            msg = f"group_by must be one of {_GROUPABLE_COLUMNS}"
            raise ValueError(msg)

        t = self._table
        group_col = getattr(t.c, group_by)

        # * Build SELECT columns list
        select_cols = [
            func.coalesce(group_col, "(untagged)").label("group_value"),
            func.coalesce(func.sum(t.c.cost_usd), 0).label("total_cost_usd"),
            func.coalesce(func.sum(t.c.prompt_tokens), 0).label("total_prompt_tokens"),
            func.coalesce(func.sum(t.c.completion_tokens), 0).label("total_completion_tokens"),
            func.count().label("total_requests"),
            func.min(t.c.timestamp).label("period_start"),
            func.max(t.c.timestamp).label("period_end"),
        ]

        # * Add time-bucket column when AggregationPeriod is specified
        group_by_cols = [group_col]
        if period is not None:
            from llmwatch.databases.functions import date_trunc as dt_func

            bucket_col = dt_func(period.value, t.c.timestamp).label("time_bucket")
            select_cols.append(bucket_col)
            group_by_cols.append(bucket_col)

        stmt = select(*select_cols).group_by(*group_by_cols).order_by(func.sum(t.c.cost_usd).desc())

        if start_time is not None:
            stmt = stmt.where(t.c.timestamp >= start_time)
        if end_time is not None:
            stmt = stmt.where(t.c.timestamp <= end_time)

        async with self._engine.connect() as conn:
            result = await conn.execute(stmt)
            rows = result.fetchall()

        return [
            CostReport(
                group_key=group_by,
                group_value=row.group_value,
                total_cost_usd=row.total_cost_usd,
                total_prompt_tokens=row.total_prompt_tokens,
                total_completion_tokens=row.total_completion_tokens,
                total_requests=row.total_requests,
                # ^ Use time_bucket as period_start when period bucketing is active
                period_start=row.time_bucket if period is not None else row.period_start,
                period_end=row.period_end,
            )
            for row in rows
        ]

    async def delete(
        self,
        *,
        before: datetime | None = None,
        feature: str | None = None,
    ) -> int:
        await self._ensure_tables()
        t = self._table
        stmt = delete(t)

        has_filter = False
        if before is not None:
            stmt = stmt.where(t.c.timestamp < before)
            has_filter = True
        if feature is not None:
            stmt = stmt.where(t.c.feature == feature)
            has_filter = True

        if not has_filter:
            msg = "At least one filter required for delete"
            raise ValueError(msg)

        async with self._engine.begin() as conn:
            result = await conn.execute(stmt)
            return result.rowcount

    async def count(self) -> int:
        await self._ensure_tables()
        async with self._engine.connect() as conn:
            result = await conn.execute(select(func.count()).select_from(self._table))
            return result.scalar_one()

    async def close(self) -> None:
        await self._engine.dispose()
