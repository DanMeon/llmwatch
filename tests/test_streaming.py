"""Tests for the streaming module."""

from llmwatch.extractors.base import ExtractedResponse
from llmwatch.extractors.google import collect_google_stream
from llmwatch.extractors.streaming import collect_stream


class _FakeUsage:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeOpenAIChunk:
    __module__ = "openai.types.chat"

    def __init__(self, content=None, model=None, chunk_id=None, usage=None):
        self.model = model
        self.id = chunk_id
        self.usage = usage
        if content is not None:
            delta = type("Delta", (), {"content": content})()
            self.choices = [type("Choice", (), {"delta": delta})()]
        else:
            self.choices = []


async def _fake_openai_stream():
    yield _FakeOpenAIChunk(model="gpt-4o", chunk_id="chatcmpl-123", content="Hello")
    yield _FakeOpenAIChunk(content=" world")
    yield _FakeOpenAIChunk(content="!")
    yield _FakeOpenAIChunk(usage=_FakeUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13))


class _FakeAnthropicEvent:
    __module__ = "anthropic.types"

    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for k, v in kwargs.items():
            setattr(self, k, v)


async def _fake_anthropic_stream():
    msg = type(
        "Msg",
        (),
        {
            "model": "claude-sonnet-4-20250514",
            "id": "msg-123",
            "usage": _FakeUsage(input_tokens=20),
        },
    )()
    yield _FakeAnthropicEvent("message_start", message=msg)
    delta1 = type("D", (), {"text": "Hello"})()
    yield _FakeAnthropicEvent("content_block_delta", delta=delta1)
    delta2 = type("D", (), {"text": " Claude"})()
    yield _FakeAnthropicEvent("content_block_delta", delta=delta2)
    yield _FakeAnthropicEvent("message_delta", usage=_FakeUsage(output_tokens=5))


# * Google fake stream helpers


class _FakeGoogleChunk:
    __module__ = "google.genai.types"

    def __init__(self, text=None, model=None, usage_metadata=None):
        self.text = text
        self.model = model
        self.usage_metadata = usage_metadata


async def _fake_google_stream():
    yield _FakeGoogleChunk(text="Hello", model="gemini-2.0-flash")
    yield _FakeGoogleChunk(text=" Gemini")
    yield _FakeGoogleChunk(text="!")
    # ^ Final chunk carries usage metadata
    yield _FakeGoogleChunk(
        usage_metadata=_FakeUsage(
            prompt_token_count=15,
            candidates_token_count=3,
            total_token_count=18,
        )
    )


async def _fake_google_stream_no_model():
    """Stream where model appears only in the first chunk."""
    yield _FakeGoogleChunk(text="Hi")
    yield _FakeGoogleChunk(
        usage_metadata=_FakeUsage(
            prompt_token_count=5,
            candidates_token_count=1,
            total_token_count=6,
        )
    )


async def _fake_google_stream_empty():
    """Stream with no text chunks."""
    yield _FakeGoogleChunk(
        usage_metadata=_FakeUsage(
            prompt_token_count=10,
            candidates_token_count=0,
            total_token_count=10,
        )
    )


class TestCollectStream:
    async def test_openai_stream(self):
        result = await collect_stream(_fake_openai_stream(), provider="openai")
        assert isinstance(result, ExtractedResponse)
        assert result.output_text == "Hello world!"
        assert result.model == "gpt-4o"
        assert result.response_id == "chatcmpl-123"
        assert result.token_usage.prompt_tokens == 10
        assert result.token_usage.completion_tokens == 3

    async def test_anthropic_stream(self):
        result = await collect_stream(_fake_anthropic_stream(), provider="anthropic")
        assert result.output_text == "Hello Claude"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.token_usage.prompt_tokens == 20
        assert result.token_usage.completion_tokens == 5

    async def test_google_stream_via_collect_stream(self):
        result = await collect_stream(_fake_google_stream(), provider="google")
        assert isinstance(result, ExtractedResponse)
        assert result.output_text == "Hello Gemini!"
        assert result.model == "gemini-2.0-flash"
        assert result.provider == "google"
        assert result.token_usage.prompt_tokens == 15
        assert result.token_usage.completion_tokens == 3
        assert result.token_usage.total_tokens == 18

    async def test_google_stream_auto_detect(self):
        """Provider detection from stream module path."""

        class _GoogleStream:
            __module__ = "google.genai.types"

            def __aiter__(self):
                return _fake_google_stream().__aiter__()

        result = await collect_stream(_GoogleStream())
        assert result.provider == "google"


class TestCollectGoogleStream:
    async def test_full_stream(self):
        result = await collect_google_stream(_fake_google_stream())
        assert result.output_text == "Hello Gemini!"
        assert result.model == "gemini-2.0-flash"
        assert result.token_usage.prompt_tokens == 15
        assert result.token_usage.completion_tokens == 3
        assert result.token_usage.total_tokens == 18

    async def test_model_override_from_argument(self):
        """When stream chunks carry no model, the default argument applies."""
        result = await collect_google_stream(_fake_google_stream_no_model(), model="gemini-1.5-pro")
        assert result.model == "gemini-1.5-pro"
        assert result.output_text == "Hi"

    async def test_empty_text_stream(self):
        result = await collect_google_stream(_fake_google_stream_empty())
        assert result.output_text == ""
        assert result.token_usage.prompt_tokens == 10
        assert result.token_usage.completion_tokens == 0

    async def test_default_model_when_no_chunks_carry_model(self):
        result = await collect_google_stream(_fake_google_stream_no_model())
        # ^ Falls back to default "unknown" when no model arg is provided
        assert result.model == "unknown"

    async def test_provider_is_always_google(self):
        result = await collect_google_stream(_fake_google_stream())
        assert result.provider == "google"
