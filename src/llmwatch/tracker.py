"""Main entry point for llmwatch — decorator-based LLM cost tracking."""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from functools import wraps
from pathlib import Path as PathLibPath
from typing import Any

from llmwatch.budget import BudgetAlert
from llmwatch.context import get_current_tags, pop_tags, push_tags
from llmwatch.cost import calculate_cost
from llmwatch.databases.base import BaseStorage
from llmwatch.databases.sqlalchemy import Storage
from llmwatch.extractors.base import ExtractedResponse, extract
from llmwatch.pricing.registry import PricingRegistry
from llmwatch.reporting import Reporter
from llmwatch.schemas.tags import Tags
from llmwatch.schemas.usage import UsageRecord

logger = logging.getLogger("llmwatch")


def _run_async_safe(coro: Coroutine[Any, Any, Any]) -> None:
    """Run an async coroutine from sync context safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # ^ No event loop running — safe to use asyncio.run()
        asyncio.run(coro)
        return
    # ^ Already inside an event loop — run in a separate thread
    # ^ Copy context so ContextVars (tag stack) propagate to the new thread
    import concurrent.futures
    import contextvars

    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(ctx.run, asyncio.run, coro).result()


class LLMWatch:
    """Main interface for LLM cost tracking."""

    def __init__(
        self,
        storage: BaseStorage | None = None,
        pricing_path: PathLibPath | None = None,
        save_io: bool = True,
        batch_size: int = 1,
        client: Any | None = None,
        clients: list[Any] | None = None,
    ) -> None:
        if storage is None:
            db_dir = PathLibPath.home() / ".llmwatch"
            db_dir.mkdir(parents=True, exist_ok=True)
            storage = Storage(f"sqlite+aiosqlite:///{db_dir / 'usage.db'}")
        self._storage = storage
        self._pricing = PricingRegistry(pricing_path)
        self._reporter = Reporter(self._storage)
        self._save_io = save_io
        self._budget = BudgetAlert()
        self._rate_limit_count: int = 0
        self._batch_size = batch_size
        self._buffer: list[UsageRecord] = []
        self._buffer_lock = asyncio.Lock()

        # * Auto-instrument clients
        for c in (clients or []) + ([client] if client else []):
            self.instrument(c)

    @property
    def storage(self) -> BaseStorage:
        return self._storage

    @property
    def pricing(self) -> PricingRegistry:
        return self._pricing

    @property
    def report(self) -> Reporter:
        return self._reporter

    @property
    def budget(self) -> BudgetAlert:
        return self._budget

    @property
    def rate_limit_count(self) -> int:
        return self._rate_limit_count

    def instrument(self, client: Any, *, provider: str | None = None) -> None:
        """Instrument an SDK client for automatic cost tracking.

        Patches the client's API methods so every call is recorded automatically.
        Currently supports AsyncOpenAI and AsyncAnthropic.
        """
        from llmwatch.instrument import _INSTRUMENTORS, detect_client_type

        detected = provider or detect_client_type(client)
        instrumentor = _INSTRUMENTORS.get(detected)
        if instrumentor is None:
            raise ValueError(f"No instrumentor for provider: {detected}")
        instrumentor(client, tracker=self)

    def tracked(
        self,
        *,
        feature: str | None = None,
        user_id: str | None = None,
        environment: str | None = None,
        **extra: str,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Tag context decorator. Use with instrument() for cost tracking."""
        tags = Tags(
            feature=feature,
            user_id=user_id,
            environment=environment,
            extra=extra,
        )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    push_tags(tags)
                    try:
                        try:
                            return await func(*args, **kwargs)
                        except Exception as e:
                            self._handle_rate_limit(e)
                            raise
                    finally:
                        pop_tags()

                return async_wrapper

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                push_tags(tags)
                try:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        self._handle_rate_limit(e)
                        raise
                finally:
                    pop_tags()

            return sync_wrapper

        return decorator

    def _handle_rate_limit(self, exc: Exception) -> None:
        # ^ Check status_code attribute (supported by both OpenAI and Anthropic SDKs)
        status = getattr(exc, "status_code", None)
        if status == 429:
            self._rate_limit_count += 1
            return
        # ^ Fallback to class name when HTTP status attribute is absent
        if "RateLimitError" in type(exc).__name__:
            self._rate_limit_count += 1

    def _safe_record_sync(self, response: Any, **kwargs: Any) -> None:
        """Sync wrapper for _safe_record. Used by sync instrument patches. Never raises."""
        try:
            _run_async_safe(self._safe_record(response, **kwargs))
        except Exception:
            logger.exception("Failed to record usage — LLM response is unaffected")

    async def _safe_record(self, response: Any, **kwargs: Any) -> None:
        """Record usage from an LLM response. Never raises."""
        try:
            await self._record(response, **kwargs)
        except Exception:
            logger.exception("Failed to record usage — LLM response is unaffected")

    async def _record(
        self,
        response: Any,
        *,
        provider: str | None = None,
        latency_ms: float | None = None,
    ) -> UsageRecord:
        if isinstance(response, ExtractedResponse):
            extracted = response
        else:
            extracted = extract(response, provider)

        merged_tags = get_current_tags()
        final_provider = provider or extracted.provider

        pricing = self._pricing.get_pricing(extracted.model, final_provider)
        cost_usd = calculate_cost(extracted.token_usage, pricing) if pricing else 0.0

        record = UsageRecord(
            tags=merged_tags,
            model=extracted.model,
            provider=final_provider,
            token_usage=extracted.token_usage,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            raw_response_id=extracted.response_id,
            output_data=extracted.output_text if self._save_io else None,
        )

        if self._batch_size <= 1:
            await self._storage.save(record)
        else:
            async with self._buffer_lock:
                self._buffer.append(record)
                if len(self._buffer) >= self._batch_size:
                    await self._flush()

        await self._budget.check(record)
        return record

    async def _flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer.copy()
        self._buffer.clear()
        await self._storage.save_batch(batch)

    async def close(self) -> None:
        if self._buffer:
            try:
                async with self._buffer_lock:
                    await self._flush()
            except Exception:
                logger.exception("Failed to flush remaining buffer on close")
        await self._storage.close()
