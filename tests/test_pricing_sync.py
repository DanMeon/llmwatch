"""Tests for pricing sync module (llmwatch.pricing.sync)."""

import json
import urllib.error
from datetime import date
from io import BytesIO
from pathlib import Path as PathLibPath
from unittest.mock import MagicMock, patch

import pytest

from llmwatch.pricing.sync import (
    _resolve_conditional_prices,
    _resolve_price_value,
    sync_pricing,
)

# * Helpers


def _make_urlopen_mock(payload: list | dict) -> MagicMock:
    """Return a context-manager mock that yields a readable bytes response."""
    raw_bytes = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=BytesIO(raw_bytes))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _make_provider(provider_id: str, models: list[dict]) -> dict:
    return {"id": provider_id, "models": models}


def _make_model(model_id: str, prices: dict | list | None, deprecated: bool = False) -> dict:
    entry: dict = {"id": model_id, "deprecated": deprecated}
    if prices is not None:
        entry["prices"] = prices
    return entry


# * Unit tests for _resolve_price_value


class TestResolvePriceValue:
    def test_none_returns_none(self):
        assert _resolve_price_value(None) is None

    def test_integer_returns_float(self):
        assert _resolve_price_value(5) == 5.0
        assert isinstance(_resolve_price_value(5), float)

    def test_float_passthrough(self):
        assert _resolve_price_value(2.5) == 2.5

    def test_tiered_dict_uses_base(self):
        tiered = {"base": 3, "tiers": [{"start": 200_000, "price": 6}]}
        assert _resolve_price_value(tiered) == 3.0

    def test_dict_without_base_returns_none(self):
        assert _resolve_price_value({"tiers": [{"start": 100, "price": 2}]}) is None

    def test_zero_returns_zero(self):
        assert _resolve_price_value(0) == 0.0


# * Unit tests for _resolve_conditional_prices


class TestResolveConditionalPrices:
    def test_dict_returned_as_is(self):
        prices = {"input_mtok": 1.0, "output_mtok": 2.0}
        result = _resolve_conditional_prices(prices, date.today())
        assert result == prices

    def test_empty_list_returns_none(self):
        assert _resolve_conditional_prices([], date.today()) is None

    def test_list_with_no_constraint_returns_prices(self):
        prices = {"input_mtok": 3.0, "output_mtok": 6.0}
        conditional = [{"prices": prices}]
        result = _resolve_conditional_prices(conditional, date.today())
        assert result == prices

    def test_list_active_start_date_in_past(self):
        """Entry with start_date in the past should be selected."""
        old_prices = {"input_mtok": 1.0, "output_mtok": 2.0}
        new_prices = {"input_mtok": 3.0, "output_mtok": 5.0}
        conditional = [
            {"prices": old_prices},
            {"constraint": {"start_date": "2020-01-01"}, "prices": new_prices},
        ]
        result = _resolve_conditional_prices(conditional, date(2024, 6, 1))
        assert result == new_prices

    def test_list_future_start_date_not_selected(self):
        """Entry with start_date in the future should be skipped."""
        old_prices = {"input_mtok": 1.0, "output_mtok": 2.0}
        future_prices = {"input_mtok": 9.0, "output_mtok": 18.0}
        conditional = [
            {"prices": old_prices},
            {"constraint": {"start_date": "2099-01-01"}, "prices": future_prices},
        ]
        result = _resolve_conditional_prices(conditional, date(2024, 6, 1))
        assert result == old_prices

    def test_non_list_non_dict_returns_none(self):
        assert _resolve_conditional_prices("invalid", date.today()) is None

    def test_first_entry_used_as_initial_active(self):
        """When list has one entry without constraint, its prices are returned."""
        prices = {"input_mtok": 5.0}
        result = _resolve_conditional_prices([{"prices": prices}], date.today())
        assert result == prices


# * Integration tests for sync_pricing()


class TestSyncPricingSuccess:
    def test_basic_model_written(self, tmp_path: PathLibPath):
        """Successful sync writes a valid pricing.json file."""
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 1
        entry = result[0]
        assert entry["model"] == "gpt-4o"
        assert entry["provider"] == "openai"
        assert entry["input_cost_per_mtok"] == 2.5
        assert entry["output_cost_per_mtok"] == 10.0

    def test_output_file_written(self, tmp_path: PathLibPath):
        """sync_pricing writes a JSON file that can be loaded back."""
        output = tmp_path / "pricing.json"
        payload = [
            _make_provider(
                "anthropic",
                [
                    _make_model("claude-opus-4", {"input_mtok": 15.0, "output_mtok": 75.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            sync_pricing(output_path=output)

        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["model"] == "claude-opus-4"

    def test_multiple_providers_and_models(self, tmp_path: PathLibPath):
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                    _make_model("gpt-4o-mini", {"input_mtok": 0.15, "output_mtok": 0.6}),
                ],
            ),
            _make_provider(
                "anthropic",
                [
                    _make_model("claude-3-5-sonnet", {"input_mtok": 3.0, "output_mtok": 15.0}),
                ],
            ),
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 3
        models = {e["model"] for e in result}
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "claude-3-5-sonnet" in models

    def test_returns_list_of_dicts(self, tmp_path: PathLibPath):
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert isinstance(result, list)
        assert all(isinstance(e, dict) for e in result)


class TestSyncPricingFiltering:
    def test_deprecated_models_skipped(self, tmp_path: PathLibPath):
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                    _make_model("gpt-3", {"input_mtok": 1.0, "output_mtok": 2.0}, deprecated=True),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        models = [e["model"] for e in result]
        assert "gpt-4o" in models
        assert "gpt-3" not in models

    def test_models_without_prices_skipped(self, tmp_path: PathLibPath):
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                    _make_model("no-price-model", None),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        models = [e["model"] for e in result]
        assert "gpt-4o" in models
        assert "no-price-model" not in models

    def test_models_with_only_request_pricing_skipped(self, tmp_path: PathLibPath):
        """Models that have only request-level pricing (no input/output tok) are dropped."""
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("request-only", {"request": 0.01}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 0

    def test_skip_providers_excluded(self, tmp_path: PathLibPath):
        """Providers in _SKIP_PROVIDERS list are not included in output."""
        payload = [
            _make_provider(
                "avian",
                [
                    _make_model("avian-model", {"input_mtok": 1.0, "output_mtok": 2.0}),
                ],
            ),
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            ),
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        providers = {e["provider"] for e in result}
        assert "avian" not in providers
        assert "openai" in providers

    def test_all_huggingface_providers_skipped(self, tmp_path: PathLibPath):
        """All huggingface_* providers in _SKIP_PROVIDERS are excluded."""
        hf_providers = [
            "huggingface_cerebras",
            "huggingface_fireworks-ai",
            "huggingface_groq",
            "huggingface_hyperbolic",
        ]
        payload = [
            _make_provider(
                pid, [_make_model("some-model", {"input_mtok": 1.0, "output_mtok": 2.0})]
            )
            for pid in hf_providers
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 0

    def test_empty_provider_list_handled(self, tmp_path: PathLibPath):
        payload: list = []
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert result == []


class TestSyncPricingTieredAndConditional:
    def test_tiered_input_price_uses_base(self, tmp_path: PathLibPath):
        tiered_prices = {
            "input_mtok": {"base": 3.0, "tiers": [{"start": 200_000, "price": 6.0}]},
            "output_mtok": 15.0,
        }
        payload = [
            _make_provider(
                "anthropic",
                [
                    _make_model("claude-opus-4", tiered_prices),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 1
        assert result[0]["input_cost_per_mtok"] == 3.0
        assert result[0]["output_cost_per_mtok"] == 15.0

    def test_tiered_output_price_uses_base(self, tmp_path: PathLibPath):
        prices = {
            "input_mtok": 2.5,
            "output_mtok": {"base": 10.0, "tiers": [{"start": 100_000, "price": 20.0}]},
        }
        payload = [_make_provider("openai", [_make_model("gpt-4o", prices)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert result[0]["output_cost_per_mtok"] == 10.0

    def test_conditional_pricing_past_date_selected(self, tmp_path: PathLibPath):
        """When a conditional list has a constraint with past start_date, it is used."""
        old_prices = {"input_mtok": 1.0, "output_mtok": 2.0}
        new_prices = {"input_mtok": 3.0, "output_mtok": 6.0}
        conditional = [
            {"prices": old_prices},
            {"constraint": {"start_date": "2020-01-01"}, "prices": new_prices},
        ]
        payload = [_make_provider("google", [_make_model("gemini-1.5-pro", conditional)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 1
        assert result[0]["input_cost_per_mtok"] == 3.0

    def test_conditional_pricing_future_date_not_selected(self, tmp_path: PathLibPath):
        """When a conditional list has a future start_date constraint, the fallback is used."""
        current_prices = {"input_mtok": 1.0, "output_mtok": 2.0}
        future_prices = {"input_mtok": 9.0, "output_mtok": 18.0}
        conditional = [
            {"prices": current_prices},
            {"constraint": {"start_date": "2099-01-01"}, "prices": future_prices},
        ]
        payload = [_make_provider("google", [_make_model("gemini-future", conditional)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert len(result) == 1
        assert result[0]["input_cost_per_mtok"] == 1.0

    def test_cache_write_price_included(self, tmp_path: PathLibPath):
        prices = {
            "input_mtok": 3.0,
            "output_mtok": 15.0,
            "cache_write_mtok": 3.75,
        }
        payload = [_make_provider("anthropic", [_make_model("claude-3-5-sonnet", prices)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert result[0]["cache_creation_cost_per_mtok"] == 3.75
        assert "cache_read_cost_per_mtok" not in result[0]

    def test_cache_read_price_included(self, tmp_path: PathLibPath):
        prices = {
            "input_mtok": 3.0,
            "output_mtok": 15.0,
            "cache_read_mtok": 0.3,
        }
        payload = [_make_provider("anthropic", [_make_model("claude-3-5-sonnet", prices)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert result[0]["cache_read_cost_per_mtok"] == 0.3
        assert "cache_creation_cost_per_mtok" not in result[0]

    def test_both_cache_prices_included(self, tmp_path: PathLibPath):
        prices = {
            "input_mtok": 3.0,
            "output_mtok": 15.0,
            "cache_write_mtok": 3.75,
            "cache_read_mtok": 0.3,
        }
        payload = [_make_provider("anthropic", [_make_model("claude-3-5-sonnet", prices)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        entry = result[0]
        assert entry["cache_creation_cost_per_mtok"] == 3.75
        assert entry["cache_read_cost_per_mtok"] == 0.3


class TestSyncPricingErrorHandling:
    def test_network_error_raises(self, tmp_path: PathLibPath):
        """URLError (e.g. no network) propagates to caller."""
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Network unreachable"),
        ):
            with pytest.raises(urllib.error.URLError):
                sync_pricing(output_path=tmp_path / "pricing.json")

    def test_http_error_raises(self, tmp_path: PathLibPath):
        """HTTP 404/500 from the source raises HTTPError."""
        http_err = urllib.error.HTTPError(
            url="http://example.com",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,  # type: ignore[arg-type]
        )
        with patch("urllib.request.urlopen", side_effect=http_err):
            with pytest.raises(urllib.error.HTTPError):
                sync_pricing(output_path=tmp_path / "pricing.json")

    def test_invalid_json_raises(self, tmp_path: PathLibPath):
        """Malformed JSON from the remote source raises JSONDecodeError."""
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=BytesIO(b"not valid json!!!"))
        cm.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=cm):
            with pytest.raises(json.JSONDecodeError):
                sync_pricing(output_path=tmp_path / "pricing.json")

    def test_output_file_not_created_on_network_error(self, tmp_path: PathLibPath):
        """If fetch fails, no file is written."""
        output = tmp_path / "pricing.json"
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("timeout"),
        ):
            with pytest.raises(urllib.error.URLError):
                sync_pricing(output_path=output)

        assert not output.exists()

    def test_missing_model_id_raises(self, tmp_path: PathLibPath):
        """A model dict missing the 'id' key should raise KeyError."""
        payload = [
            {"id": "openai", "models": [{"prices": {"input_mtok": 1.0, "output_mtok": 2.0}}]}
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            with pytest.raises(KeyError):
                sync_pricing(output_path=tmp_path / "pricing.json")


class TestSyncPricingOutputFormat:
    def test_output_json_is_valid_list(self, tmp_path: PathLibPath):
        output = tmp_path / "pricing.json"
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            sync_pricing(output_path=output)

        data = json.loads(output.read_text())
        assert isinstance(data, list)

    def test_output_entry_has_required_fields(self, tmp_path: PathLibPath):
        output = tmp_path / "pricing.json"
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            sync_pricing(output_path=output)

        data = json.loads(output.read_text())
        entry = data[0]
        assert "model" in entry
        assert "provider" in entry
        assert "input_cost_per_mtok" in entry
        assert "output_cost_per_mtok" in entry

    def test_output_file_ends_with_newline(self, tmp_path: PathLibPath):
        """JSON output file ends with a trailing newline (consistent formatting)."""
        output = tmp_path / "pricing.json"
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            sync_pricing(output_path=output)

        content = output.read_text()
        assert content.endswith("\n")

    def test_output_file_parent_created_if_missing(self, tmp_path: PathLibPath):
        """sync_pricing creates the parent directory if it does not exist."""
        output = tmp_path / "nested" / "dir" / "pricing.json"
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            sync_pricing(output_path=output)

        assert output.exists()

    def test_custom_source_url_passed_to_urlopen(self, tmp_path: PathLibPath):
        """sync_pricing uses the provided source_url, not the default."""
        custom_url = "https://example.com/custom/pricing.json"
        payload = [
            _make_provider(
                "openai",
                [
                    _make_model("gpt-4o", {"input_mtok": 2.5, "output_mtok": 10.0}),
                ],
            )
        ]
        mock_cm = _make_urlopen_mock(payload)
        with patch("urllib.request.urlopen", return_value=mock_cm) as mock_open:
            sync_pricing(output_path=tmp_path / "pricing.json", source_url=custom_url)

        mock_open.assert_called_once_with(custom_url, timeout=30)

    def test_null_input_price_defaults_to_zero(self, tmp_path: PathLibPath):
        """When only output_mtok is set, input defaults to 0.0."""
        prices = {"output_mtok": 5.0}
        payload = [_make_provider("openai", [_make_model("gpt-output-only", prices)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert result[0]["input_cost_per_mtok"] == 0.0
        assert result[0]["output_cost_per_mtok"] == 5.0

    def test_null_output_price_defaults_to_zero(self, tmp_path: PathLibPath):
        """When only input_mtok is set, output defaults to 0.0."""
        prices = {"input_mtok": 2.0}
        payload = [_make_provider("openai", [_make_model("gpt-input-only", prices)])]
        with patch("urllib.request.urlopen", return_value=_make_urlopen_mock(payload)):
            result = sync_pricing(output_path=tmp_path / "pricing.json")

        assert result[0]["input_cost_per_mtok"] == 2.0
        assert result[0]["output_cost_per_mtok"] == 0.0
