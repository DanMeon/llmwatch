<p align="center">
  <img src="assets/banner.svg" alt="llmwatch banner" width="100%">
</p>

# llmwatch

**A lightweight Python library for LLM cost attribution — track, tag, and report LLM API costs by feature, user, and model.**

[![CI](https://github.com/DanMeon/llmwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/DanMeon/llmwatch/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/llmwatch)](https://pypi.org/project/llmwatch/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/llmwatch)](https://pypi.org/project/llmwatch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why llmwatch?

llmwatch is **not** an observability platform or proxy server. It's a lightweight Python library that integrates directly into your existing codebase.

Unlike solutions like Langfuse, LangSmith, or LiteLLM, llmwatch requires no external infrastructure, no API gateway, and no proxy setup. Just `pip install llmwatch` and add 3 lines of code to start tracking LLM costs.

Key differentiators:

- **No proxy or gateway needed** — Unlike LiteLLM and Helicone, which sit between your code and LLM APIs
- **No external platform** — Unlike Langfuse and LangSmith, which require cloud infrastructure
- **Works with your existing SDK** — Patch your OpenAI, Anthropic, Google, Cohere, or VoyageAI clients with `instrument(client)`
- **Feature-level cost attribution** — Tag LLM calls by feature, user, environment, and any custom dimension
- **Minimal setup** — 3 lines of code to get started
- **1000+ models** — Bundled pricing data covering OpenAI, Anthropic, Google, and more

## Quick Start

### Async

```python
from openai import AsyncOpenAI
from llmwatch.tracker import LLMWatch

client = AsyncOpenAI()
watcher = LLMWatch(client=client)

@watcher.tracked(feature="summarize", user_id="alice")
async def summarize(text: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.choices[0].message.content

result = await summarize("Long document text...")
```

### Sync

```python
from openai import OpenAI
from llmwatch.tracker import LLMWatch

client = OpenAI()
watcher = LLMWatch(client=client)

@watcher.tracked(feature="summarize", user_id="alice")
def summarize(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.choices[0].message.content

result = summarize("Long document text...")
```

## Features

- **Automatic cost tracking** — Instrument SDK clients to capture token usage and calculate costs without modifying your LLM calls
- **Flexible tagging** — Attach metadata to tracked calls with `@watcher.tracked(feature=..., user_id=..., environment=...)`
- **Multi-provider support** — OpenAI, Anthropic, Google, Cohere, VoyageAI (sync, async, and streaming)
- **Reranker support** — Auto-instrument Cohere and VoyageAI reranker SDKs, or use `record_usage()` for any HTTP-based API
- **Bundled pricing** — 1000+ models with up-to-date pricing data synced from pydantic/genai-prices
- **Multiple database backends** — SQLite (default), PostgreSQL, MySQL, MongoDB (Beanie ODM), Oracle, MSSQL
- **Budget alerts** — Set thresholds and trigger callbacks when spending exceeds limits
- **Reporting and export** — Generate cost summaries by feature, user, model, or provider (CSV/JSON)
- **CLI tools** — View reports, manage data, sync pricing
- **Web dashboard** — Optional interactive dashboard for cost visualization (`llmwatch dashboard`)
- **Streaming support** — Track costs for streaming responses (SSE, async streams)

## Supported Providers

| Provider | Sync | Async | Streaming | Models |
|----------|:----:|:-----:|:---------:|--------|
| OpenAI | O | O | O | GPT-5.4, o4-mini, o3, o1, GPT-4o, etc. |
| Anthropic | O | O | O | Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5, etc. |
| Google | O | O | O | Gemini 3.1, Gemini 2.5, Gemini 2.0, etc. |
| Cohere | O | O | - | Rerank v3.5, Rerank v4.0, etc. |
| VoyageAI | O | O | - | Rerank 2.5, Rerank 2, etc. |

## Installation

```bash
pip install llmwatch
# or
uv add llmwatch
```

### Optional Database Backends

```bash
pip install llmwatch[pg]         # PostgreSQL
pip install llmwatch[mysql]      # MySQL
pip install llmwatch[mongo]      # MongoDB (Beanie ODM)
pip install llmwatch[dashboard]  # Web dashboard (Starlette + Uvicorn)
```

## Usage

### Basic Tracking

```python
from openai import AsyncOpenAI
from llmwatch.tracker import LLMWatch

client = AsyncOpenAI()
watcher = LLMWatch(client=client)

@watcher.tracked(feature="chat", user_id="user123", environment="production")
async def chat_response(prompt: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# Costs are tracked automatically
result = await chat_response("Hello, how are you?")
```

### Budget Alerts

```python
watcher = LLMWatch(client=client)

async def on_budget_exceeded(record):
    print(f"Budget exceeded: ${record.cost_usd:.4f} on feature={record.tags.feature}")

watcher.budget.add_rule(
    max_cost_usd=0.50,
    callback=on_budget_exceeded,
    feature="summarize",
)
```

### Reporting

#### Programmatic

```python
summary = await watcher.report.by_feature(period="7d")
print(f"Total cost: ${summary.total_cost_usd:.4f}")
for b in summary.breakdowns:
    print(f"  {b.group_value}: ${b.total_cost_usd:.4f} ({b.total_requests} calls)")

# Also available: by_user_id(), by_model(), by_provider()
await watcher.report.export_csv("costs.csv", group_by="feature", period="30d")
await watcher.report.export_json("costs.json", group_by="model", period="7d")
```

#### CLI

```bash
llmwatch report --group-by feature --period 7d
llmwatch export costs.csv --format csv
llmwatch pricing list --provider openai
llmwatch pricing sync
```

### Web Dashboard

```bash
pip install llmwatch[dashboard]
llmwatch dashboard
# Opens at http://localhost:8000
```

### Multiple Database Backends

By default, llmwatch uses SQLite (`~/.llmwatch/usage.db`). Switch to other backends by passing a storage instance:

```python
from llmwatch.tracker import LLMWatch
from llmwatch.databases.sqlalchemy import Storage

# PostgreSQL
watcher = LLMWatch(
    client=client,
    storage=Storage("postgresql+asyncpg://user:password@localhost/llmwatch"),
)

# MySQL
watcher = LLMWatch(
    client=client,
    storage=Storage("mysql+aiomysql://user:password@localhost/llmwatch"),
)
```

#### MongoDB

```python
from llmwatch.tracker import LLMWatch
from llmwatch.databases.mongo import MongoStorage

watcher = LLMWatch(
    client=client,
    storage=MongoStorage("mongodb://localhost:27017", database="llmwatch"),
)
```

### Manual Recording (for HTTP-based APIs)

For providers without a Python SDK (e.g., Jina reranker via httpx), use `record_usage()`:

```python
import httpx

response = await httpx.AsyncClient().post(
    "https://api.jina.ai/v1/rerank",
    headers={"Authorization": f"Bearer {JINA_API_KEY}"},
    json={"model": "jina-reranker-v3", "query": query, "documents": docs},
)
data = response.json()

await watcher.record_usage(
    model="jina-reranker-v3",
    provider="jina",
    input_tokens=data["usage"]["total_tokens"],
    feature="search",
)
```

### Custom Provider Registration

Register your own provider extractor and instrumentor:

```python
from llmwatch.extractors.base import register_extractor
from llmwatch.instrument import register_instrumentor

register_extractor("my_llm", my_extract_fn, module_prefix="my_llm_sdk")
register_instrumentor("my_llm", my_instrumentor_fn)
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `llmwatch report` | Generate cost report (`--group-by`, `--period`) |
| `llmwatch export` | Export usage records to CSV or JSON |
| `llmwatch prune` | Delete old records by date |
| `llmwatch stats` | Show database statistics |
| `llmwatch pricing list` | List pricing data by provider |
| `llmwatch pricing sync` | Sync pricing data from upstream |
| `llmwatch dashboard` | Start interactive web dashboard |

## How It Works

1. **Instrument** — `LLMWatch(client=client)` patches the SDK client's methods
2. **Extract** — On each LLM call, extractors normalize the response (handles OpenAI, Anthropic, Google, Cohere, VoyageAI, streaming)
3. **Calculate** — `calculate_cost()` computes USD cost using bundled pricing data
4. **Store** — `Storage.save()` persists the `UsageRecord` to your database
5. **Tag** — `@watcher.tracked()` provides tag context (feature, user_id, environment)
6. **Alert** — Optional `BudgetAlert` callbacks trigger when thresholds are exceeded
7. **Report** — `Reporter` generates cost summaries grouped by feature, user, model, or provider

```
LLM Call
  | (via instrumented SDK client)
Extract Response -> Calculate Cost -> Save Record + Tags
  |
Database
  | (queried by Reporter)
Reports, Dashboards, Exports
```

## Development

```bash
uv sync --group dev
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/llmwatch/
```

## License

MIT

---

**Pricing data sourced from [pydantic/genai-prices](https://github.com/pydantic/genai-prices).**
