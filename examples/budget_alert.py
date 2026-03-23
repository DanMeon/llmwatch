"""Budget alert — notify when cost exceeds a threshold."""

import asyncio

from openai import AsyncOpenAI

from llmwatch import LLMWatch, UsageRecord

watcher = LLMWatch()
client = AsyncOpenAI()


# * Alert callback (can be replaced with Slack, email, etc.)
async def on_expensive_call(record: UsageRecord):
    print(
        f"[ALERT] {record.tags.feature} cost exceeded! "
        f"${record.cost_usd:.4f} (model: {record.model})"
    )


# * Alert when a single call exceeds $0.01
watcher.budget.add_rule(max_cost_usd=0.01, callback=on_expensive_call)

# * Watch specific feature only (alert when exceeding $0.005)
watcher.budget.add_rule(
    max_cost_usd=0.005,
    callback=on_expensive_call,
    feature="rag_search",
)


@watcher.tracked(feature="rag_search")
async def rag_search(query: str):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer based on RAG search results."},
            {"role": "user", "content": query},
        ],
    )


async def main():
    await rag_search("What is the sales trend over the last 3 months?")
    await watcher.close()


if __name__ == "__main__":
    asyncio.run(main())
