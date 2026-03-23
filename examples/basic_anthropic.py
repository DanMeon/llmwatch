"""Basic usage — Anthropic API cost tracking."""

import asyncio

from anthropic import AsyncAnthropic

from llmwatch import LLMWatch

watcher = LLMWatch()
client = AsyncAnthropic()


@watcher.tracked(feature="chat", user_id="bob")
async def chat(prompt: str):
    return await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )


async def main():
    await chat("Explain Python's GIL")
    await chat("How does asyncio work?")

    summary = await watcher.report.by_feature(period="1h")
    for b in summary.breakdowns:
        print(f"{b.group_value}: ${b.total_cost_usd:.4f}")

    await watcher.close()


if __name__ == "__main__":
    asyncio.run(main())
