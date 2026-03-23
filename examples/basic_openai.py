"""Basic usage — OpenAI API cost tracking."""

import asyncio

from openai import AsyncOpenAI

from llmwatch import LLMWatch

watcher = LLMWatch()
client = AsyncOpenAI()


# * Just add the decorator for automatic cost tracking
@watcher.tracked(feature="chat", user_id="alice")
async def chat(prompt: str):
    return await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )


@watcher.tracked(feature="summarize", user_id="alice")
async def summarize(text: str):
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the given text in 3 lines."},
            {"role": "user", "content": text},
        ],
    )


async def main():
    # * Normal usage — just call the function
    await chat("Recommend places to visit in Seoul")
    await chat("Suggest good restaurants in Busan")
    await summarize("Long text...")

    # * View report
    summary = await watcher.report.by_feature(period="1h")
    for b in summary.breakdowns:
        print(f"{b.group_value}: ${b.total_cost_usd:.4f} ({b.total_requests} calls)")

    print(f"\nTotal: ${summary.total_cost_usd:.4f}")

    await watcher.close()


if __name__ == "__main__":
    asyncio.run(main())
