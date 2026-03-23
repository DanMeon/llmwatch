"""Shared test fixtures and fake response/client objects."""


class _FakeUsage:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class FakeOpenAIResponse:
    __module__ = "openai.types.chat"

    def __init__(
        self,
        model: str = "gpt-4o",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        response_id: str = "chatcmpl-test123",
        content: str = "Hello, this is a test response.",
    ):
        self.id = response_id
        self.model = model
        self.usage = _FakeUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.choices = [_FakeChoice(content)]


class FakeAnthropicResponse:
    __module__ = "anthropic.types"

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        input_tokens: int = 200,
        output_tokens: int = 100,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        content: str = "Hello from Claude.",
    ):
        self.id = "msg-test123"
        self.model = model
        self.usage = _FakeUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        self.content = [_FakeContentBlock(content)]


class FakeGoogleResponse:
    __module__ = "google.genai.types"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        prompt_token_count: int = 150,
        candidates_token_count: int = 75,
        text: str = "Hello from Gemini.",
    ):
        self.model = model
        self.text = text
        self.usage_metadata = _FakeUsage(
            prompt_token_count=prompt_token_count,
            candidates_token_count=candidates_token_count,
            total_token_count=prompt_token_count + candidates_token_count,
        )


# * Fake SDK clients for instrument() testing


class _FakeCompletions:
    def __init__(self, response_factory):
        self._factory = response_factory

    async def create(self, **kwargs):
        return self._factory(**kwargs)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeAsyncOpenAI:
    """Mock AsyncOpenAI client."""

    __module__ = "openai._client"

    def __init__(self, response_factory=None):
        factory = response_factory or (
            lambda **kw: FakeOpenAIResponse(model=kw.get("model", "gpt-4o"))
        )
        self.chat = _FakeChat(_FakeCompletions(factory))


class _FakeMessages:
    def __init__(self, response_factory):
        self._factory = response_factory

    async def create(self, **kwargs):
        return self._factory(**kwargs)


class FakeAsyncAnthropic:
    """Mock AsyncAnthropic client."""

    __module__ = "anthropic._client"

    def __init__(self, response_factory=None):
        factory = response_factory or (
            lambda **kw: FakeAnthropicResponse(model=kw.get("model", "claude-sonnet-4-20250514"))
        )
        self.messages = _FakeMessages(factory)


# * Sync fake clients


class _FakeSyncCompletions:
    def __init__(self, response_factory):
        self._factory = response_factory

    def create(self, **kwargs):
        return self._factory(**kwargs)


class _FakeSyncChat:
    def __init__(self, completions):
        self.completions = completions


class FakeSyncOpenAI:
    """Mock sync OpenAI client."""

    __module__ = "openai._client"

    def __init__(self, response_factory=None):
        factory = response_factory or (
            lambda **kw: FakeOpenAIResponse(model=kw.get("model", "gpt-4o"))
        )
        self.chat = _FakeSyncChat(_FakeSyncCompletions(factory))


class _FakeSyncMessages:
    def __init__(self, response_factory):
        self._factory = response_factory

    def create(self, **kwargs):
        return self._factory(**kwargs)


class FakeSyncAnthropic:
    """Mock sync Anthropic client."""

    __module__ = "anthropic._client"

    def __init__(self, response_factory=None):
        factory = response_factory or (
            lambda **kw: FakeAnthropicResponse(model=kw.get("model", "claude-sonnet-4-20250514"))
        )
        self.messages = _FakeSyncMessages(factory)


# * Google Gen AI fake clients


class _FakeGoogleModels:
    def __init__(self, response_factory):
        self._factory = response_factory

    async def generate_content(self, **kwargs):
        return self._factory(**kwargs)


class FakeAsyncGoogle:
    """Mock async Google Gen AI client."""

    __module__ = "google.genai._client"

    def __init__(self, response_factory=None):
        factory = response_factory or (
            lambda **kw: FakeGoogleResponse(model=kw.get("model", "gemini-2.0-flash"))
        )
        self.models = _FakeGoogleModels(factory)


class _FakeGoogleModelsSync:
    def __init__(self, response_factory):
        self._factory = response_factory

    def generate_content(self, **kwargs):
        return self._factory(**kwargs)


class FakeSyncGoogle:
    """Mock sync Google Gen AI client."""

    __module__ = "google.genai._client"

    def __init__(self, response_factory=None):
        factory = response_factory or (
            lambda **kw: FakeGoogleResponse(model=kw.get("model", "gemini-2.0-flash"))
        )
        self.models = _FakeGoogleModelsSync(factory)
