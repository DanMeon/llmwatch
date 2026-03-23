"""Tests for the cost module."""

from llmwatch.cost import calculate_cost
from llmwatch.schemas.pricing import ModelPricing
from llmwatch.schemas.usage import TokenUsage


class TestCalculateCost:
    def test_basic(self):
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        pricing = ModelPricing(
            model="gpt-4o",
            provider="openai",
            input_cost_per_mtok=2.5,
            output_cost_per_mtok=10.0,
        )
        cost = calculate_cost(usage, pricing)
        # ^ (1000 * 2.5 + 500 * 10.0) / 1_000_000 = 7500 / 1_000_000 = 0.0075
        assert abs(cost - 0.0075) < 1e-9

    def test_with_cache_tokens(self):
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_creation_input_tokens=200,
            cache_read_input_tokens=100,
        )
        pricing = ModelPricing(
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            input_cost_per_mtok=3.0,
            output_cost_per_mtok=15.0,
            cache_creation_cost_per_mtok=3.75,
            cache_read_cost_per_mtok=0.3,
        )
        cost = calculate_cost(usage, pricing)
        expected = (1000 * 3.0 + 500 * 15.0 + 200 * 3.75 + 100 * 0.3) / 1_000_000
        assert abs(cost - expected) < 1e-9

    def test_zero_usage(self):
        usage = TokenUsage()
        pricing = ModelPricing(
            model="gpt-4o",
            provider="openai",
            input_cost_per_mtok=2.5,
            output_cost_per_mtok=10.0,
        )
        assert calculate_cost(usage, pricing) == 0.0
