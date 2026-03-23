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
            try:
                serialized = json.dumps(value, ensure_ascii=False, default=str)
                if len(serialized) > _MAX_VALUE_LENGTH:
                    result[key] = serialized[:_MAX_VALUE_LENGTH] + "...(truncated)"
                else:
                    result[key] = value
            except (TypeError, ValueError):
                result[key] = repr(value)[:_MAX_VALUE_LENGTH]
        return result
    except Exception:
        logger.debug("Failed to serialize input for %s", getattr(func, "__name__", func))
        return None
