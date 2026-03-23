"""Token usage and usage record schemas."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from llmwatch.schemas.tags import Tags


class TokenUsage(BaseModel):
    """Token usage extracted from an LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # ^ Anthropic cache tokens
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class UsageRecord(BaseModel):
    """Usage record for a single LLM call."""

    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: Tags = Field(default_factory=Tags)
    model: str
    provider: str
    token_usage: TokenUsage
    cost_usd: float
    latency_ms: float | None = None
    raw_response_id: str | None = None
    # ^ LLM input/output storage
    input_data: dict[str, object] | list[object] | str | None = None
    output_data: str | None = None
