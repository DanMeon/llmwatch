"""Streaming response cost tracking."""

import asyncio

from openai import AsyncOpenAI

from llmwatch import LLMWatch
from llmwatch.extractors.streaming import collect_stream

watcher = LLMWatch()
client = AsyncOpenAI()


@watcher.tracked(feature="stream_chat")
async def stream_chat(prompt: str):
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        # ^ Required to receive token usage in stream
        stream_options={"include_usage": True},
    )
    # ^ collect_stream aggregates chunks and extracts token usage + text
    return await collect_stream(stream)


async def main():
    result = await stream_chat("What are the key changes in Python 3.12?")
    print(f"Response: {result.text[:100]}...")

    summary = await watcher.report.by_feature(period="1h")
    for b in summary.breakdowns:
        print(f"{b.group_value}: ${b.total_cost_usd:.4f}")

    await watcher.close()


if __name__ == "__main__":
    asyncio.run(main())
