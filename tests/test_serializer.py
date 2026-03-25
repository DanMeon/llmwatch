"""Tests for extractors.serializer module."""

from llmwatch.extractors.serializer import serialize_input


def _sample_func(model: str, messages: list, api_key: str = "sk-xxx") -> None:
    pass


class TestSerializeInput:
    def test_basic_serialization(self):
        result = serialize_input(
            _sample_func, (), {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result is not None
        assert result["model"] == "gpt-4"
        assert result["messages"] == [{"role": "user", "content": "hi"}]

    def test_redacts_sensitive_keys(self):
        result = serialize_input(
            _sample_func, (), {"model": "gpt-4", "messages": [], "api_key": "sk-secret"}
        )
        assert result is not None
        assert result["api_key"] == "***REDACTED***"

    def test_custom_redact_keys(self):
        def func(model: str, custom_secret: str = ""):
            pass

        result = serialize_input(
            func, (), {"model": "gpt-4", "custom_secret": "val"}, redact_keys={"custom_secret"}
        )
        assert result is not None
        assert result["custom_secret"] == "***REDACTED***"

    def test_truncates_long_values(self):
        def func(data: str):
            pass

        long_str = "x" * 20_000
        result = serialize_input(func, (), {"data": long_str})
        assert result is not None
        assert "truncated" in result["data"]

    def test_non_serializable_value_uses_repr(self):
        def func(obj: object):
            pass

        class Unserializable:
            def __repr__(self):
                return "<Unserializable>"

            def __str__(self):
                raise TypeError("cannot serialize")

        # json.dumps with default=str will call str(), which raises TypeError,
        # so it falls back to repr
        result = serialize_input(func, (), {"obj": Unserializable()})
        assert result is not None

    def test_returns_none_on_bind_failure(self):
        def func(a: int, b: int):
            pass

        # Too many args -> bind fails
        result = serialize_input(func, (1, 2, 3), {})
        assert result is None

    def test_positional_args(self):
        result = serialize_input(_sample_func, ("gpt-4", [{"role": "user", "content": "hi"}]), {})
        assert result is not None
        assert result["model"] == "gpt-4"

    def test_applies_defaults(self):
        def func(model: str, temperature: float = 0.7):
            pass

        result = serialize_input(func, (), {"model": "gpt-4"})
        assert result is not None
        assert result["temperature"] == 0.7

    def test_redacts_nested_dict_sensitive_keys(self):
        def func(config: dict):
            pass

        result = serialize_input(
            func,
            (),
            {"config": {"api_key": "sk-secret", "model": "gpt-4"}},
        )
        assert result is not None
        assert result["config"]["api_key"] == "***REDACTED***"
        assert result["config"]["model"] == "gpt-4"

    def test_redacts_deeply_nested_sensitive_keys(self):
        def func(payload: dict):
            pass

        result = serialize_input(
            func,
            (),
            {"payload": {"outer": {"password": "hunter2", "data": "ok"}}},
        )
        assert result is not None
        assert result["payload"]["outer"]["password"] == "***REDACTED***"
        assert result["payload"]["outer"]["data"] == "ok"

    def test_redacts_sensitive_keys_in_list_of_dicts(self):
        def func(items: list):
            pass

        result = serialize_input(
            func,
            (),
            {"items": [{"token": "abc123", "name": "foo"}, {"name": "bar"}]},
        )
        assert result is not None
        assert result["items"][0]["token"] == "***REDACTED***"
        assert result["items"][0]["name"] == "foo"
        assert result["items"][1]["name"] == "bar"

    def test_redacts_case_insensitive_nested_keys(self):
        def func(headers: dict):
            pass

        result = serialize_input(
            func,
            (),
            {"headers": {"Authorization": "Bearer tok", "Content-Type": "application/json"}},
        )
        assert result is not None
        assert result["headers"]["Authorization"] == "***REDACTED***"
        assert result["headers"]["Content-Type"] == "application/json"
