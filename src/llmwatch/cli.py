"""llmwatch CLI interface."""

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from llmwatch.databases.sqlalchemy import Storage
from llmwatch.pricing.sync import sync_pricing
from llmwatch.tracker import LLMWatch

app = typer.Typer(name="llmwatch", help="LLM Cost Attribution CLI")
pricing_app = typer.Typer(name="pricing", help="Pricing data management")
app.add_typer(pricing_app)
console = Console()


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    return asyncio.run(coro)


def _make_watch(db: str | None) -> LLMWatch:
    if db is None:
        return LLMWatch()
    return LLMWatch(storage=Storage(db))


@app.command()
def report(
    group_by: str = typer.Option("feature", help="Group by: feature, user_id, model, provider"),
    period: str = typer.Option("30d", help="Look back period: e.g. 7d, 24h, 30d"),
    db: str | None = typer.Option(None, help="DB connection URL"),
) -> None:
    """Print cost report."""

    async def _run_report() -> None:
        watcher = _make_watch(db)
        summary = await watcher.report.summary(group_by=group_by, period=period)

        table = Table(title=f"LLM Cost Report (by {group_by}, last {period})")
        table.add_column(group_by, style="cyan")
        table.add_column("Requests", justify="right")
        table.add_column("Prompt Tokens", justify="right")
        table.add_column("Completion Tokens", justify="right")
        table.add_column("Cost (USD)", justify="right", style="green")

        for b in summary.breakdowns:
            table.add_row(
                b.group_value,
                f"{b.total_requests:,}",
                f"{b.total_prompt_tokens:,}",
                f"{b.total_completion_tokens:,}",
                f"${b.total_cost_usd:.4f}",
            )

        table.add_section()
        table.add_row(
            "TOTAL",
            f"{summary.total_requests:,}",
            f"{summary.total_prompt_tokens:,}",
            f"{summary.total_completion_tokens:,}",
            f"${summary.total_cost_usd:.4f}",
            style="bold",
        )
        console.print(table)
        await watcher.close()

    _run(_run_report())


@app.command()
def export(
    output: str = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("csv", help="Export format: csv, json"),
    group_by: str = typer.Option("feature", help="Group by: feature, user_id, model"),
    period: str = typer.Option("30d", help="Look back period"),
    db: str | None = typer.Option(None, help="DB connection URL"),
) -> None:
    """Export usage data."""

    async def _run_export() -> None:
        watcher = _make_watch(db)
        if format == "csv":
            await watcher.report.export_csv(output, group_by=group_by, period=period)
        elif format == "json":
            await watcher.report.export_json(output, group_by=group_by, period=period)
        else:
            console.print(f"[red]Unknown format: {format}[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Exported to {output}[/green]")
        await watcher.close()

    _run(_run_export())


@app.command()
def prune(
    days: int = typer.Option(90, help="Delete records older than N days"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    db: str | None = typer.Option(None, help="DB connection URL"),
) -> None:
    """Delete old records."""

    async def _run_prune() -> None:
        watcher = _make_watch(db)
        cutoff = datetime.now(UTC) - timedelta(days=days)
        if not yes:
            typer.confirm(f"Delete all records before {cutoff.date()}?", abort=True)
        deleted = await watcher.storage.delete(before=cutoff)
        console.print(f"[green]Deleted {deleted} records.[/green]")
        await watcher.close()

    _run(_run_prune())


@app.command()
def stats(
    db: str | None = typer.Option(None, help="DB connection URL"),
) -> None:
    """Print basic statistics."""

    async def _run_stats() -> None:
        watcher = _make_watch(db)
        total = await watcher.storage.count()
        console.print(f"Total records: {total:,}")
        await watcher.close()

    _run(_run_stats())


@pricing_app.command("sync")
def pricing_sync() -> None:
    """Sync latest pricing from pydantic/genai-prices."""
    try:
        entries = sync_pricing()
        providers = sorted({e["provider"] for e in entries})
        console.print(
            f"[green]Synced {len(entries)} models across {len(providers)} providers[/green]"
        )
    except Exception as e:
        console.print(f"[red]Sync failed: {e}[/red]")
        raise typer.Exit(1)


@pricing_app.command("list")
def pricing_list(
    provider: str | None = typer.Option(None, help="Filter by provider"),
) -> None:
    """List available model pricing."""
    watcher = _make_watch(None)
    models = watcher.pricing.list_models(provider)
    if not models:
        console.print("[yellow]No pricing data found. Run 'llmwatch pricing sync' first.[/yellow]")
        return

    table = Table(title="Model Pricing (USD per 1M tokens)")
    table.add_column("Provider", style="cyan")
    table.add_column("Model")
    table.add_column("Input", justify="right", style="green")
    table.add_column("Output", justify="right", style="green")
    table.add_column("Cache Write", justify="right")
    table.add_column("Cache Read", justify="right")

    for m in sorted(models, key=lambda x: (x.provider, x.model)):
        table.add_row(
            m.provider,
            m.model,
            f"${m.input_cost_per_mtok:.2f}",
            f"${m.output_cost_per_mtok:.2f}",
            f"${m.cache_creation_cost_per_mtok:.2f}" if m.cache_creation_cost_per_mtok else "-",
            f"${m.cache_read_cost_per_mtok:.2f}" if m.cache_read_cost_per_mtok else "-",
        )

    console.print(table)


@app.command()
def dashboard(
    port: int = typer.Option(8000, help="Server port"),
    host: str = typer.Option("127.0.0.1", help="Server host"),
    db: str | None = typer.Option(None, help="DB connection URL"),
) -> None:
    """Launch the web dashboard."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn is not installed. Install the dashboard extras:[/red]\n"
            "  pip install llmwatch[dashboard]\n"
            "  uv add 'llmwatch[dashboard]'"
        )
        raise typer.Exit(1)

    try:
        from llmwatch.dashboard import create_dashboard_app
    except ImportError:
        console.print(
            "[red]starlette is not installed. Install the dashboard extras:[/red]\n"
            "  pip install llmwatch[dashboard]\n"
            "  uv add 'llmwatch[dashboard]'"
        )
        raise typer.Exit(1)

    dash_app = create_dashboard_app(storage_url=db)
    console.print(f"[green]Dashboard running at http://{host}:{port}[/green]")
    uvicorn.run(dash_app, host=host, port=port)


if __name__ == "__main__":
    app()
