"""Input serialization with sensitive key redaction."""

import inspect
import json
import logging
from typing import Any

logger = logging.getLogger("llmwatch")

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "credential",
    "authorization",
    "auth",
    "private_key",
    "secret_key",
    "access_token",
}

_MAX_VALUE_LENGTH = 10_000


def _redact_value(value: Any, sensitive: set[str]) -> Any:
    """Recursively redact sensitive keys in nested dicts and lists."""
    if isinstance(value, dict):
        return {
            # ^ Case-insensitive match against sensitive set
            k: "***REDACTED***" if k.lower() in sensitive else _redact_value(v, sensitive)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, sensitive) for item in value]
    return value


def serialize_input(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    redact_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    """Convert function call arguments to a JSON-serializable dict."""
    sensitive = _SENSITIVE_KEYS | (redact_keys or set())
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        result: dict[str, Any] = {}
        for key, value in bound.arguments.items():
            if key.lower() in sensitive:
                result[key] = "***REDACTED***"
                continue
            # * Recursively redact sensitive keys in nested structures
            redacted = _redact_value(value, sensitive)
            try:
                serialized = json.dumps(redacted, ensure_ascii=False, default=str)
                if len(serialized) > _MAX_VALUE_LENGTH:
                    result[key] = serialized[:_MAX_VALUE_LENGTH] + "...(truncated)"
                else:
                    result[key] = redacted
            except (TypeError, ValueError):
                result[key] = repr(value)[:_MAX_VALUE_LENGTH]
        return result
    except Exception:
        logger.debug("Failed to serialize input for %s", getattr(func, "__name__", func))
        return None
