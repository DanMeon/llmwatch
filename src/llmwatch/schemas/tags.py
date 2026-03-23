"""Metadata tags for cost attribution."""

from pydantic import BaseModel, Field


class Tags(BaseModel):
    """Metadata tags for cost attribution."""

    feature: str | None = None
    user_id: str | None = None
    environment: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)

    def merge(self, other: "Tags") -> "Tags":
        merged_extra = {**self.extra, **other.extra}
        return Tags(
            feature=other.feature or self.feature,
            user_id=other.user_id or self.user_id,
            environment=other.environment or self.environment,
            extra=merged_extra,
        )
