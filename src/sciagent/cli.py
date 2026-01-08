"""
CLI interface for SciAgent

Interactive command-line interface with rich formatting.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from sciagent.api import SciAgent
from sciagent.utils.config import Config, load_config
from sciagent.utils.logging import setup_logging
from sciagent.utils.models import HumanApprovalRequest

console = Console()


@click.group()
@click.option("--config", type=click.Path(), help="Path to config file")
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def cli(ctx, config: Optional[str], log_level: str):
    """SciAgent: Advanced multi-agent scientific experimentation"""

    # Setup logging
    setup_logging(level=log_level, rich_console=True)

    # Load configuration
    if config:
        ctx.obj = load_config(Path(config))
    else:
        ctx.obj = load_config()


@cli.command()
@click.argument("query")
@click.option("--interactive/--auto", default=True, help="Interactive mode with approvals")
@click.option("--budget", type=float, default=10000, help="Compute budget in USD")
@click.pass_obj
def run(config: Config, query: str, interactive: bool, budget: float):
    """Run a scientific experiment with human-in-the-loop"""

    console.print(
        Panel(
            f"[bold blue]Starting Experiment[/bold blue]\n\n"
            f"Query: {query}\n"
            f"Budget: ${budget:,.2f}\n"
            f"Mode: {'Interactive' if interactive else 'Automated'}",
            title="SciAgent",
            border_style="blue",
        )
    )

    # Create agent
    agent = SciAgent(config)

    # Set up user callback for interactive mode
    if interactive:
        user_callback = create_cli_callback(console)
    else:
        user_callback = None

    # Run experiment
    try:
        result = asyncio.run(
            agent.run(query=query, interactive=interactive, on_approval=user_callback)
        )

        # Display results
        display_results(console, result)

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Experiment interrupted. Use 'sciagent resume' to continue.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


@cli.command()
@click.argument("experiment_id")
@click.pass_obj
def resume(config: Config, experiment_id: str):
    """Resume a paused experiment"""

    console.print(f"[blue]Resuming experiment {experiment_id}...[/blue]")

    agent = SciAgent(config)

    try:
        result = asyncio.run(agent.resume(experiment_id))
        display_results(console, result)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.argument("experiment_id")
@click.pass_obj
def pause(config: Config, experiment_id: str):
    """Pause a running experiment"""

    console.print(f"[yellow]Pausing experiment {experiment_id}...[/yellow]")

    agent = SciAgent(config)

    try:
        asyncio.run(agent.pause(experiment_id))
        console.print(f"[green]Experiment {experiment_id} paused successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.pass_obj
def list(config: Config):
    """List all experiments"""

    agent = SciAgent(config)
    experiments = agent.list_experiments()

    if not experiments:
        console.print("[yellow]No experiments found[/yellow]")
        return

    table = Table(title="Experiments")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Query", style="green")

    for exp_id in experiments:
        state = agent.get_experiment_state(exp_id)
        if state:
            table.add_row(exp_id, state.status.value, state.query[:50] + "...")

    console.print(table)


@cli.command()
@click.option("--output", type=click.Path(), help="Output path for config file")
@click.pass_obj
def init_config(config: Config, output: Optional[str]):
    """Initialize configuration file"""

    output_path = Path(output) if output else Path.home() / ".sciagent" / "config.yaml"

    console.print(f"[blue]Creating configuration file at {output_path}[/blue]")

    config.to_yaml(output_path)

    console.print(f"[green]Configuration created successfully![/green]")
    console.print(f"\nEdit {output_path} to customize settings.")


def create_cli_callback(console: Console):
    """Create callback for CLI user interaction"""

    async def callback(request: HumanApprovalRequest):
        """Handle user approval requests"""

        if request.request_type == "approval":
            # Yes/no question
            message = request.context.get("message", "Approve?")

            console.print(
                Panel(
                    message,
                    title="[yellow]Approval Required[/yellow]",
                    border_style="yellow",
                )
            )

            response = Confirm.ask("Proceed?")

            # Set response
            request.response.set_result("yes" if response else "no")

        elif request.request_type == "choice":
            # Multiple choice
            message = request.context["message"]

            console.print(
                Panel(message, title="[yellow]Decision Required[/yellow]", border_style="yellow")
            )

            # Display options
            for i, option in enumerate(request.options):
                console.print(f"  [{i+1}] {option}")

            if request.context.get("allow_multiple"):
                choice = Prompt.ask(
                    "Select options (comma-separated)", choices=[str(i + 1) for i in range(len(request.options))]
                )
                response = [opt.strip() for opt in choice.split(",")]
            else:
                choice = Prompt.ask(
                    "Select option", choices=[str(i + 1) for i in range(len(request.options))]
                )
                response = request.options[int(choice) - 1]

            # Set response
            request.response.set_result(response)

    return callback


def display_results(console: Console, result):
    """Display experiment results"""

    console.print("\n")
    console.print(
        Panel(
            f"[bold green]Experiment Complete![/bold green]\n\n"
            f"ID: {result.experiment_id}\n"
            f"Status: {result.state.status.value}",
            title="Results",
            border_style="green",
        )
    )

    # Analysis summary
    if result.analysis:
        console.print("\n[bold]Analysis Summary:[/bold]")
        console.print(result.analysis.summary)

        console.print(
            f"\nSupports Hypothesis: {'✓ Yes' if result.analysis.supports_hypothesis else '✗ No'}"
        )
        console.print(f"Confidence: {result.analysis.confidence:.2%}")

    # Validation
    if result.validation:
        console.print("\n[bold]Validation:[/bold]")
        console.print(
            f"Approved: {'✓ Yes' if result.validation.approved else '✗ No'}"
        )
        console.print(f"Confidence: {result.validation.confidence:.2%}")

        if result.validation.recommendations:
            console.print("\n[bold]Recommendations:[/bold]")
            for rec in result.validation.recommendations[:5]:
                console.print(f"  • {rec}")

    # Refinements
    if result.refinements:
        console.print("\n[bold]Proposed Refinements:[/bold]")
        for i, ref in enumerate(result.refinements, 1):
            console.print(
                f"  {i}. {ref.description} (Info Gain: {ref.expected_information_gain:.2f})"
            )


if __name__ == "__main__":
    cli()
