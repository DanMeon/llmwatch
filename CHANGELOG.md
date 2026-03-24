# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-24

### Added

- Time-bucketed aggregation via `AggregationPeriod` (hour/day/week/month) with cross-dialect `date_trunc` support (PostgreSQL, SQLite, MySQL, Oracle, MSSQL)
- `aggregation_period` parameter in `Reporter.summary()` for time-series cost breakdowns
- `BudgetAlert.reset()` and `get_cumulative_cost()` methods
- `cumulative_cost` keyword argument passed to budget alert callbacks
- OpenAI prompt caching token extraction (`prompt_tokens_details.cached_tokens`)
- Anthropic streaming cache token extraction (`cache_creation_input_tokens`, `cache_read_input_tokens`)
- Google Gemini cached content token extraction (`cached_content_token_count`)
- Streaming response auto-instrumentation (`_TrackedAsyncStream` / `_TrackedSyncStream`)
- "Environment" group-by option in web dashboard

### Fixed

- Streaming responses (`stream=True`) were silently producing zero-usage records when using auto-instrumentation
- Budget alert compared per-request cost instead of cumulative spend — now tracks cumulative cost per rule
- SQL injection path in `_auto_migrate` via f-string interpolation — now uses `dialect.identifier_preparer`
- Cache tokens not extracted for OpenAI/Anthropic/Google, causing inflated cost calculations
- Dashboard API rejected `group_by=environment` despite storage backends supporting it
- `_period_to_start()` accepted negative values and empty strings without error

## [0.1.0] - 2026-03-23

### Added

- Initial public release
- Core tracking via `LLMWatch` class with `instrument()` and `@tracked()` decorator
- Provider support: OpenAI, Anthropic, Google Gemini
- SQLAlchemy async storage (SQLite, PostgreSQL, MySQL, Oracle, MSSQL)
- MongoDB storage via Beanie ODM
- Bundled pricing data with automatic sync from pydantic/genai-prices
- Budget alert system
- CLI with `report`, `export`, `prune`, `stats`, `pricing`, `dashboard` commands
- Web dashboard with Chart.js visualizations
