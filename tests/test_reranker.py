"""Tests for reranker SDK instrumentation and extraction (Cohere, VoyageAI, manual)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llmwatch.extractors.base import ExtractedResponse, detect_provider, extract
from llmwatch.extractors.cohere import extract_cohere
from llmwatch.extractors.voyageai import extract_voyageai

# * Fake response builders


class _FakeCohereResponse:
    """Fake Cohere RerankResponse with correct __module__ for detect_provider."""

    __module__ = "cohere.types"

    def __init__(
        self,
        *,
        model: str = "rerank-v3.5",
        search_units: int = 1,
        input_tokens: int = 150,
        response_id: str = "resp-cohere-1",
    ) -> None:
        self.model = model
        self.id = response_id
        self.meta = MagicMock()
        self.meta.billed_units = MagicMock()
        self.meta.billed_units.search_units = search_units
        self.meta.billed_units.input_tokens = input_tokens
        self.meta.billed_units.output_tokens = 0
        self.meta.tokens = None


class _FakeVoyageAIResponse:
    """Fake VoyageAI RerankingObject with correct __module__ for detect_provider."""

    __module__ = "voyageai.object.reranking"

    def __init__(self, *, model: str = "rerank-2.5", total_tokens: int = 500) -> None:
        self.model = model
        self.total_tokens = total_tokens
        self.results: list[object] = []


def _fake_cohere_response(**kwargs: object) -> _FakeCohereResponse:
    return _FakeCohereResponse(**kwargs)  # type: ignore[arg-type]


def _fake_voyageai_response(**kwargs: object) -> _FakeVoyageAIResponse:
    return _FakeVoyageAIResponse(**kwargs)  # type: ignore[arg-type]


# * Extractor tests


class TestCohereExtractor:
    def test_extracts_input_tokens_from_billed_units(self):
        resp = _fake_cohere_response(input_tokens=200, search_units=1)
        result = extract_cohere(resp, "cohere")

        assert isinstance(result, ExtractedResponse)
        assert result.provider == "cohere"
        assert result.model == "rerank-v3.5"
        assert result.token_usage.prompt_tokens == 200
        assert result.response_id == "resp-cohere-1"

    def test_falls_back_to_search_units_when_no_input_tokens(self):
        resp = _fake_cohere_response(input_tokens=0, search_units=5)
        result = extract_cohere(resp, "cohere")

        assert result.token_usage.prompt_tokens == 5

    def test_handles_missing_meta_gracefully(self):
        resp = MagicMock()
        resp.__class__ = type("RerankResponse", (), {"__module__": "cohere.types"})
        resp.model = "rerank-v3.5"
        resp.id = None
        resp.meta = None

        result = extract_cohere(resp, "cohere")
        assert result.token_usage.prompt_tokens == 0
        assert result.model == "rerank-v3.5"


class TestVoyageAIExtractor:
    def test_extracts_total_tokens(self):
        resp = _fake_voyageai_response(total_tokens=800)
        result = extract_voyageai(resp, "voyageai")

        assert isinstance(result, ExtractedResponse)
        assert result.provider == "voyageai"
        assert result.model == "rerank-2.5"
        assert result.token_usage.prompt_tokens == 800
        assert result.token_usage.total_tokens == 800

    def test_handles_zero_tokens(self):
        resp = _fake_voyageai_response(total_tokens=0)
        result = extract_voyageai(resp, "voyageai")
        assert result.token_usage.prompt_tokens == 0


# * Provider detection tests


class TestDetectProvider:
    def test_detects_cohere(self):
        resp = _fake_cohere_response()
        assert detect_provider(resp) == "cohere"

    def test_detects_voyageai(self):
        resp = _fake_voyageai_response()
        assert detect_provider(resp) == "voyageai"


class TestExtractDispatch:
    def test_dispatches_to_cohere_extractor(self):
        resp = _fake_cohere_response(input_tokens=100)
        result = extract(resp, "cohere")
        assert result.provider == "cohere"
        assert result.token_usage.prompt_tokens == 100

    def test_dispatches_to_voyageai_extractor(self):
        resp = _fake_voyageai_response(total_tokens=300)
        result = extract(resp, "voyageai")
        assert result.provider == "voyageai"
        assert result.token_usage.prompt_tokens == 300


# * Instrumentation tests


class TestCohereInstrumentation:
    def test_instrument_sync_cohere_v1_client(self):
        """Cohere v1 client: client.rerank is patched directly."""
        from llmwatch.instrument import instrument_cohere_sync

        client = MagicMock()
        # ^ v1 client has no .v2 attribute
        del client.v2
        client.rerank = MagicMock(return_value=_fake_cohere_response())

        tracker = MagicMock()
        tracker._safe_record_sync = MagicMock()

        instrument_cohere_sync(client, tracker=tracker)

        result = client.rerank(query="test", documents=["a", "b"])
        tracker._safe_record_sync.assert_called_once()
        assert result.model == "rerank-v3.5"

    def test_instrument_sync_cohere_v2_client(self):
        """Cohere v2 client: client.v2.rerank is patched."""
        from llmwatch.instrument import instrument_cohere_sync

        client = MagicMock()
        client.v2 = MagicMock()
        client.v2.rerank = MagicMock(return_value=_fake_cohere_response())

        tracker = MagicMock()
        tracker._safe_record_sync = MagicMock()

        instrument_cohere_sync(client, tracker=tracker)

        result = client.v2.rerank(query="test", documents=["a", "b"])
        tracker._safe_record_sync.assert_called_once()
        assert result.model == "rerank-v3.5"

    @pytest.mark.asyncio
    async def test_instrument_async_cohere(self):
        from llmwatch.instrument import instrument_cohere_async

        client = MagicMock()
        del client.v2
        client.rerank = AsyncMock(return_value=_fake_cohere_response())

        tracker = MagicMock()
        tracker._safe_record = AsyncMock()

        instrument_cohere_async(client, tracker=tracker)

        result = await client.rerank(query="test", documents=["a", "b"])
        tracker._safe_record.assert_called_once()
        assert result.model == "rerank-v3.5"


class TestVoyageAIInstrumentation:
    def test_instrument_sync_voyageai(self):
        from llmwatch.instrument import instrument_voyageai_sync

        client = MagicMock()
        client.rerank = MagicMock(return_value=_fake_voyageai_response())

        tracker = MagicMock()
        tracker._safe_record_sync = MagicMock()

        instrument_voyageai_sync(client, tracker=tracker)

        result = client.rerank(query="test", documents=["a"], model="rerank-2.5")
        tracker._safe_record_sync.assert_called_once()
        assert result.model == "rerank-2.5"

    @pytest.mark.asyncio
    async def test_instrument_async_voyageai(self):
        from llmwatch.instrument import instrument_voyageai_async

        client = MagicMock()
        client.rerank = AsyncMock(return_value=_fake_voyageai_response())

        tracker = MagicMock()
        tracker._safe_record = AsyncMock()

        instrument_voyageai_async(client, tracker=tracker)

        result = await client.rerank(query="test", documents=["a"], model="rerank-2.5")
        tracker._safe_record.assert_called_once()
        assert result.model == "rerank-2.5"


# * Manual record_usage() tests


class TestRecordUsage:
    @pytest.mark.asyncio
    async def test_record_usage_basic(self):
        from llmwatch.tracker import LLMWatch

        watcher = LLMWatch()
        record = await watcher.record_usage(
            model="jina-reranker-v3",
            provider="jina",
            input_tokens=500,
            feature="search",
        )

        assert record.model == "jina-reranker-v3"
        assert record.provider == "jina"
        assert record.token_usage.prompt_tokens == 500
        assert record.token_usage.total_tokens == 500
        assert record.tags.feature == "search"
        await watcher.close()

    @pytest.mark.asyncio
    async def test_record_usage_explicit_cost(self):
        from llmwatch.tracker import LLMWatch

        watcher = LLMWatch()
        record = await watcher.record_usage(
            model="rerank-2.5",
            provider="voyageai",
            input_tokens=1000,
            cost_usd=0.00005,
        )

        assert record.cost_usd == 0.00005
        await watcher.close()

    @pytest.mark.asyncio
    async def test_record_usage_auto_cost_from_pricing(self):
        from llmwatch.tracker import LLMWatch

        watcher = LLMWatch()
        record = await watcher.record_usage(
            model="rerank-2.5",
            provider="voyageai",
            input_tokens=1_000_000,
        )

        # ^ pricing_reranker.json has rerank-2.5 at 0.05 per Mtok
        assert record.cost_usd == pytest.approx(0.05, rel=1e-6)
        await watcher.close()

    @pytest.mark.asyncio
    async def test_record_usage_merges_context_tags(self):
        from llmwatch.context import pop_tags, push_tags
        from llmwatch.schemas.tags import Tags
        from llmwatch.tracker import LLMWatch

        watcher = LLMWatch()
        push_tags(Tags(environment="prod"))
        try:
            record = await watcher.record_usage(
                model="jina-reranker-v3",
                provider="jina",
                input_tokens=100,
                feature="search",
                user_id="u-1",
            )
            assert record.tags.feature == "search"
            assert record.tags.user_id == "u-1"
            assert record.tags.environment == "prod"
        finally:
            pop_tags()
            await watcher.close()

    @pytest.mark.asyncio
    async def test_record_usage_computes_total_tokens(self):
        from llmwatch.tracker import LLMWatch

        watcher = LLMWatch()
        record = await watcher.record_usage(
            model="test-model",
            provider="test",
            input_tokens=100,
            output_tokens=50,
        )
        assert record.token_usage.total_tokens == 150
        await watcher.close()
