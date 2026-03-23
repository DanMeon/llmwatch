"""Tests for the schema modules."""

from llmwatch.schemas.tags import Tags


class TestTags:
    def test_merge_overrides(self):
        base = Tags(feature="chat", user_id="alice")
        override = Tags(feature="summarize")
        merged = base.merge(override)
        assert merged.feature == "summarize"
        assert merged.user_id == "alice"

    def test_merge_extra(self):
        base = Tags(extra={"team": "backend"})
        override = Tags(extra={"env": "prod"})
        merged = base.merge(override)
        assert merged.extra == {"team": "backend", "env": "prod"}

    def test_merge_none_does_not_override(self):
        base = Tags(feature="chat", user_id="alice")
        override = Tags(user_id=None)
        merged = base.merge(override)
        assert merged.user_id == "alice"
