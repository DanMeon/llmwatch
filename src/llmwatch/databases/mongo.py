"""MongoDB storage backend using Beanie ODM.

Beanie is an optional dependency. Install with:
    pip install llmwatch[mongo]
or:
    uv add "llmwatch[mongo]"
"""

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

try:
    import pymongo
    from beanie import Document, init_beanie
    from pymongo import AsyncMongoClient
except ImportError as exc:
    raise ImportError(
        "MongoDB backend requires beanie. Install it with: pip install llmwatch[mongo]"
    ) from exc

from pydantic import Field

from llmwatch.schemas.reporting import AggregationPeriod, CostReport
from llmwatch.schemas.tags import Tags
from llmwatch.schemas.usage import TokenUsage, UsageRecord

logger = logging.getLogger("llmwatch")

_GROUPABLE_COLUMNS = {"feature", "user_id", "model", "provider", "environment"}


class UsageDocument(Document):
    """MongoDB document for LLM usage records."""

    doc_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime
    feature: str | None = None
    user_id: str | None = None
    model: str
    provider: str
    environment: str | None = None
    extra_tags: dict[str, str] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    raw_response_id: str | None = None
    input_data: str | None = None
    output_data: str | None = None

    class Settings:
        name = "llmwatch_usage"
        indexes = [
            pymongo.IndexModel([("timestamp", pymongo.DESCENDING)]),
            pymongo.IndexModel([("feature", pymongo.ASCENDING)]),
            pymongo.IndexModel([("user_id", pymongo.ASCENDING)]),
            pymongo.IndexModel([("model", pymongo.ASCENDING)]),
        ]


class MongoStorage:
    """MongoDB async storage backend using Beanie ODM."""

    def __init__(
        self,
        connection_url: str,
        *,
        database: str = "llmwatch",
        collection_name: str = "llmwatch_usage",
    ) -> None:
        self._connection_url = connection_url
        self._database = database
        self._collection_name = collection_name
        self._client: AsyncMongoClient[dict[str, Any]] | None = None
        self._initialized = False

        # ^ Apply custom collection name
        UsageDocument.Settings.name = collection_name

    async def _ensure_init(self) -> None:
        if self._initialized:
            return
        self._client = AsyncMongoClient(self._connection_url)
        db = self._client[self._database]
        await init_beanie(database=db, document_models=[UsageDocument])
        self._initialized = True

    @staticmethod
    def _record_to_doc(record: UsageRecord) -> UsageDocument:
        return UsageDocument(
            doc_id=record.id,
            timestamp=record.timestamp,
            feature=record.tags.feature,
            user_id=record.tags.user_id,
            model=record.model,
            provider=record.provider,
            environment=record.tags.environment,
            extra_tags=record.tags.extra if record.tags.extra else None,
            prompt_tokens=record.token_usage.prompt_tokens,
            completion_tokens=record.token_usage.completion_tokens,
            total_tokens=record.token_usage.total_tokens,
            cache_creation_input_tokens=record.token_usage.cache_creation_input_tokens,
            cache_read_input_tokens=record.token_usage.cache_read_input_tokens,
            cost_usd=record.cost_usd,
            latency_ms=record.latency_ms,
            raw_response_id=record.raw_response_id,
            input_data=(
                json.dumps(record.input_data, ensure_ascii=False, default=str)
                if record.input_data is not None
                else None
            ),
            output_data=record.output_data,
        )

    @staticmethod
    def _doc_to_record(doc: UsageDocument) -> UsageRecord:
        input_data = None
        if doc.input_data is not None:
            try:
                input_data = json.loads(doc.input_data)
            except (json.JSONDecodeError, TypeError):
                input_data = doc.input_data

        return UsageRecord(
            id=doc.doc_id,
            timestamp=doc.timestamp,
            tags=Tags(
                feature=doc.feature,
                user_id=doc.user_id,
                environment=doc.environment,
                extra=doc.extra_tags or {},
            ),
            model=doc.model,
            provider=doc.provider,
            token_usage=TokenUsage(
                prompt_tokens=doc.prompt_tokens,
                completion_tokens=doc.completion_tokens,
                total_tokens=doc.total_tokens,
                cache_creation_input_tokens=doc.cache_creation_input_tokens,
                cache_read_input_tokens=doc.cache_read_input_tokens,
            ),
            cost_usd=doc.cost_usd,
            latency_ms=doc.latency_ms,
            raw_response_id=doc.raw_response_id,
            input_data=input_data,
            output_data=doc.output_data,
        )

    async def save(self, record: UsageRecord) -> None:
        await self._ensure_init()
        doc = self._record_to_doc(record)
        await doc.insert()

    async def save_batch(self, records: list[UsageRecord]) -> None:
        await self._ensure_init()
        docs = [self._record_to_doc(r) for r in records]
        await UsageDocument.insert_many(docs)

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
        await self._ensure_init()

        filters: dict[str, Any] = {}
        if feature is not None:
            filters["feature"] = feature
        if user_id is not None:
            filters["user_id"] = user_id
        if model is not None:
            filters["model"] = model
        if provider is not None:
            filters["provider"] = provider
        if environment is not None:
            filters["environment"] = environment
        if start_time is not None or end_time is not None:
            ts_filter: dict[str, Any] = {}
            if start_time is not None:
                ts_filter["$gte"] = start_time
            if end_time is not None:
                ts_filter["$lte"] = end_time
            filters["timestamp"] = ts_filter

        docs = (
            await UsageDocument.find(filters)
            .sort([("timestamp", pymongo.DESCENDING)])  # type: ignore[list-item]
            .skip(offset)
            .limit(limit)
            .to_list()
        )
        return [self._doc_to_record(doc) for doc in docs]

    async def aggregate(
        self,
        *,
        group_by: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        period: AggregationPeriod | None = None,
    ) -> list[CostReport]:
        await self._ensure_init()

        if group_by not in _GROUPABLE_COLUMNS:
            msg = f"group_by must be one of {_GROUPABLE_COLUMNS}"
            raise ValueError(msg)

        pipeline: list[dict[str, Any]] = []

        # * Match stage (time filter)
        match_filter: dict[str, Any] = {}
        if start_time is not None:
            match_filter.setdefault("timestamp", {})["$gte"] = start_time
        if end_time is not None:
            match_filter.setdefault("timestamp", {})["$lte"] = end_time
        if match_filter:
            pipeline.append({"$match": match_filter})

        # * Group stage
        pipeline.append(
            {
                "$group": {
                    "_id": {"$ifNull": [f"${group_by}", "(untagged)"]},
                    "total_cost_usd": {"$sum": "$cost_usd"},
                    "total_prompt_tokens": {"$sum": "$prompt_tokens"},
                    "total_completion_tokens": {"$sum": "$completion_tokens"},
                    "total_requests": {"$sum": 1},
                    "period_start": {"$min": "$timestamp"},
                    "period_end": {"$max": "$timestamp"},
                },
            }
        )

        # * Sort by total cost descending
        pipeline.append({"$sort": {"total_cost_usd": -1}})

        results = await UsageDocument.aggregate(pipeline).to_list()

        return [
            CostReport(
                group_key=group_by,
                group_value=str(r["_id"]),
                total_cost_usd=r["total_cost_usd"],
                total_prompt_tokens=r["total_prompt_tokens"],
                total_completion_tokens=r["total_completion_tokens"],
                total_requests=r["total_requests"],
                period_start=r.get("period_start"),
                period_end=r.get("period_end"),
            )
            for r in results
        ]

    async def delete(
        self,
        *,
        before: datetime | None = None,
        feature: str | None = None,
    ) -> int:
        await self._ensure_init()

        filters: dict[str, Any] = {}
        if before is not None:
            filters["timestamp"] = {"$lt": before}
        if feature is not None:
            filters["feature"] = feature

        if not filters:
            msg = "At least one filter required for delete"
            raise ValueError(msg)

        result = await UsageDocument.find(filters).delete()
        return result.deleted_count if result else 0

    async def count(self) -> int:
        await self._ensure_init()
        return int(await UsageDocument.count())

    async def close(self) -> None:
        if self._client:
            await self._client.close()
