"""Tests for the extractors module."""

from llmwatch.extractors.base import detect_provider, extract
from llmwatch.extractors.google import extract_generic, extract_google
from tests.conftest import FakeAnthropicResponse, FakeGoogleResponse, FakeOpenAIResponse


class TestDetectProvider:
    def test_openai(self):
        assert detect_provider(FakeOpenAIResponse()) == "openai"

    def test_anthropic(self):
        assert detect_provider(FakeAnthropicResponse()) == "anthropic"

    def test_google(self):
        assert detect_provider(FakeGoogleResponse()) == "google"

    def test_unknown(self):
        assert detect_provider("hello") == "unknown"


class TestExtract:
    def test_openai_response(self):
        response = FakeOpenAIResponse(prompt_tokens=100, completion_tokens=50)
        result = extract(response)
        assert result.provider == "openai"
        assert result.model == "gpt-4o"
        assert result.token_usage.prompt_tokens == 100
        assert result.token_usage.completion_tokens == 50
        assert result.output_text == "Hello, this is a test response."

    def test_anthropic_response(self):
        response = FakeAnthropicResponse(
            input_tokens=200,
            output_tokens=100,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=10,
        )
        result = extract(response)
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.token_usage.prompt_tokens == 200
        assert result.token_usage.completion_tokens == 100
        assert result.token_usage.cache_creation_input_tokens == 50
        assert result.output_text == "Hello from Claude."

    def test_dict_response(self):
        response = {
            "model": "some-model",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        result = extract(response, provider="custom")
        assert result.provider == "custom"
        assert result.model == "some-model"
        assert result.token_usage.total_tokens == 30

    def test_google_response(self):
        response = FakeGoogleResponse(
            prompt_token_count=150,
            candidates_token_count=75,
        )
        result = extract(response)
        assert result.provider == "google"
        assert result.model == "gemini-2.0-flash"
        assert result.token_usage.prompt_tokens == 150
        assert result.token_usage.completion_tokens == 75
        assert result.token_usage.total_tokens == 225
        assert result.output_text == "Hello from Gemini."

    def test_explicit_provider(self):
        response = FakeOpenAIResponse()
        result = extract(response, provider="openai")
        assert result.provider == "openai"
        assert result.token_usage.prompt_tokens == 100


class TestExtractGoogle:
    def test_basic_extraction(self):
        response = FakeGoogleResponse(
            model="gemini-2.0-pro",
            prompt_token_count=200,
            candidates_token_count=100,
            text="Test output.",
        )
        result = extract_google(response, "google")
        assert result.provider == "google"
        assert result.model == "gemini-2.0-pro"
        assert result.token_usage.prompt_tokens == 200
        assert result.token_usage.completion_tokens == 100
        assert result.token_usage.total_tokens == 300
        assert result.output_text == "Test output."

    def test_missing_usage_metadata(self):
        # ^ usage_metadata absent — should return zero token usage
        response = FakeGoogleResponse()
        response.usage_metadata = None
        result = extract_google(response, "google")
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.total_tokens == 0

    def test_missing_model(self):
        response = FakeGoogleResponse()
        del response.model
        result = extract_google(response, "google")
        assert result.model == "unknown"

    def test_no_text(self):
        response = FakeGoogleResponse()
        response.text = None
        result = extract_google(response, "google")
        assert result.output_text is None

    def test_zero_token_counts(self):
        response = FakeGoogleResponse(prompt_token_count=0, candidates_token_count=0)
        result = extract_google(response, "google")
        assert result.token_usage.total_tokens == 0

    def test_generic_fallback_dict(self):
        response = {
            "model": "custom-model",
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "choices": [{"message": {"content": "hi"}}],
        }
        result = extract_generic(response, "custom")
        assert result.model == "custom-model"
        assert result.token_usage.total_tokens == 15
        assert result.output_text == "hi"

    def test_generic_fallback_object(self):
        class _Obj:
            model = "obj-model"
            usage = None

        result = extract_generic(_Obj(), "custom")
        assert result.model == "obj-model"
        assert result.token_usage.prompt_tokens == 0

    def test_generic_fallback_dict_no_usage(self):
        response = {"model": "m", "usage": {}}
        result = extract_generic(response, "fallback")
        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
