"""Metadata tags for cost attribution."""

from pydantic import BaseModel, Field, field_validator


class Tags(BaseModel):
    """Metadata tags for cost attribution."""

    feature: str | None = None
    user_id: str | None = None
    environment: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)

    # ^ Treat blank strings as "not set" — tags must be meaningful identifiers
    @field_validator("feature", "user_id", "environment", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: str | None) -> str | None:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    def merge(self, other: "Tags") -> "Tags":
        merged_extra = {**self.extra, **other.extra}
        return Tags(
            feature=other.feature or self.feature,
            user_id=other.user_id or self.user_id,
            environment=other.environment or self.environment,
            extra=merged_extra,
        )
