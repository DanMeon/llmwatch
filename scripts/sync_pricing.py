"""Sync pricing data from pydantic/genai-prices.

Run manually or via GitHub Actions:
    uv run python scripts/sync_pricing.py
"""

import sys
from pathlib import Path as PathLibPath

# ^ Ensure src/ is importable when running as a standalone script
sys.path.insert(0, str(PathLibPath(__file__).parent.parent / "src"))

from llmwatch.pricing import sync_pricing


def main() -> None:
    entries = sync_pricing()
    providers = sorted({e["provider"] for e in entries})
    print(f"Synced {len(entries)} models across {len(providers)} providers")
    print(f"Providers: {', '.join(providers)}")


if __name__ == "__main__":
    main()
