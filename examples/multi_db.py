"""Various DB connection examples."""

import asyncio

from openai import AsyncOpenAI

from llmwatch import LLMWatch, Storage

client = AsyncOpenAI()

# * SQLite (default — ~/.llmwatch/usage.db)
watcher_default = LLMWatch()

# * SQLite (custom path)
watcher_sqlite = LLMWatch(
    storage=Storage("sqlite+aiosqlite:///~/my_project/llm_costs.db")
)

# * PostgreSQL
watcher_pg = LLMWatch(
    storage=Storage("postgresql+asyncpg://user:password@localhost:5432/mydb")
)

# * MySQL
watcher_mysql = LLMWatch(
    storage=Storage("mysql+aiomysql://user:password@localhost:3306/mydb")
)

# * Oracle
watcher_oracle = LLMWatch(
    storage=Storage("oracle+oracledb://user:password@localhost:1521/?service_name=xe")
)

# * Custom table name + schema
watcher_custom = LLMWatch(
    storage=Storage(
        "postgresql+asyncpg://user:pw@host/db",
        table_name="llm_usage_prod",
        schema="analytics",
    )
)


async def main():
    # ^ Usage is the same regardless of DB
    @watcher_default.tracked(feature="test")
    async def call_llm():
        return await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )

    await call_llm()

    summary = await watcher_default.report.by_feature(period="1h")
    print(f"Total: ${summary.total_cost_usd:.4f}")

    await watcher_default.close()


if __name__ == "__main__":
    asyncio.run(main())
