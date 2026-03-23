"""Report viewing and export."""

import asyncio

from llmwatch import LLMWatch

watcher = LLMWatch()


async def main():
    # * Report by feature (last 7 days)
    by_feature = await watcher.report.by_feature(period="7d")
    print("=== Cost by Feature ===")
    for b in by_feature.breakdowns:
        print(f"  {b.group_value}: ${b.total_cost_usd:.4f} / {b.total_requests:,} calls")

    # * Report by user (last 30 days)
    by_user = await watcher.report.by_user_id(period="30d")
    print("\n=== Cost by User ===")
    for b in by_user.breakdowns:
        print(f"  {b.group_value}: ${b.total_cost_usd:.4f}")

    # * Report by model
    by_model = await watcher.report.by_model(period="30d")
    print("\n=== Cost by Model ===")
    for b in by_model.breakdowns:
        print(f"  {b.group_value}: ${b.total_cost_usd:.4f}")

    # * Report by provider
    by_provider = await watcher.report.by_provider(period="30d")
    print("\n=== Cost by Provider ===")
    for b in by_provider.breakdowns:
        print(f"  {b.group_value}: ${b.total_cost_usd:.4f}")

    # * CSV/JSON export
    await watcher.report.export_csv("cost_report.csv", group_by="feature", period="30d")
    await watcher.report.export_json("cost_report.json", group_by="user", period="30d")
    print("\nExported: cost_report.csv, cost_report.json")

    await watcher.close()


if __name__ == "__main__":
    asyncio.run(main())
