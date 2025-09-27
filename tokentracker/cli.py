"""Command-line interface for TokenTracker."""

import click
import sys
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

from .tracker import TokenTracker
from .models import SUPPORTED_MODELS

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """TokenTracker - Track context lengths for LLMs so you know when you're close to the limit."""
    pass


@main.command()
@click.argument('text', required=False)
@click.option('--model', '-m', default='gpt-3.5-turbo', help='Model to use for token counting')
@click.option('--file', '-f', type=click.File('r'), help='Read text from file')
def count(text: str, model: str, file) -> None:
    """Count tokens in text or file."""
    if file:
        text = file.read()
    elif not text:
        # Read from stdin if no text provided
        text = sys.stdin.read()
    
    if not text.strip():
        console.print("[red]Error: No text provided[/red]")
        sys.exit(1)
    
    tracker = TokenTracker(model)
    
    # Show approximation notice if not using tiktoken
    if not tracker.use_tiktoken:
        console.print("[yellow]Note: Using approximate token counting (tiktoken unavailable)[/yellow]\n")
    
    token_count = tracker.count_tokens(text)
    usage = tracker.track_prompt(text)
    
    # Create a nice display
    table = Table(title=f"Token Count for {model}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Text Length", f"{len(text)} characters")
    table.add_row("Token Count", f"{token_count}")
    table.add_row("Model Max Tokens", f"{usage.max_tokens}")
    table.add_row("Percentage Used", f"{usage.percentage_used:.2f}%")
    table.add_row("Remaining Tokens", f"{tracker.get_remaining_tokens()}")
    
    if usage.cost_estimate:
        table.add_row("Estimated Cost", f"${usage.cost_estimate:.4f}")
    
    console.print(table)
    
    # Show warning if near limit
    warning = tracker.get_warning_message()
    if warning:
        console.print(Panel(warning, style="bold red"))


@main.command()
@click.option('--model', '-m', default='gpt-3.5-turbo', help='Model to use for token counting')
def interactive(model: str) -> None:
    """Interactive token tracking session."""
    tracker = TokenTracker(model)
    console.print(f"[green]Started interactive session with {model}[/green]")
    console.print("Commands: /count <text>, /reset, /status, /quit")
    console.print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input == "/quit":
                break
            elif user_input == "/reset":
                tracker.reset()
                console.print("[green]Session reset[/green]")
            elif user_input == "/status":
                usage = tracker._get_current_usage()
                _display_usage_status(usage, tracker)
            elif user_input.startswith("/count "):
                text = user_input[7:]  # Remove "/count "
                if text:
                    usage = tracker.track_prompt(text)
                    _display_usage_status(usage, tracker)
                else:
                    console.print("[red]Error: No text provided[/red]")
            elif user_input:
                usage = tracker.track_prompt(user_input)
                _display_usage_status(usage, tracker)
                
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    console.print("\n[green]Session ended[/green]")


@main.command()
def models() -> None:
    """List supported models and their configurations."""
    table = Table(title="Supported Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Max Tokens", style="green")
    table.add_column("Cost per 1K tokens", style="magenta")
    
    for model_name, config in SUPPORTED_MODELS.items():
        cost = f"${config.cost_per_1k_tokens:.4f}" if config.cost_per_1k_tokens else "N/A"
        table.add_row(
            model_name,
            config.provider,
            f"{config.max_tokens:,}",
            cost
        )
    
    console.print(table)


@main.command()
@click.argument('file', type=click.File('r'))
@click.option('--model', '-m', default='gpt-3.5-turbo', help='Model to use for token counting')
def analyze(file, model: str) -> None:
    """Analyze token usage of a conversation file (JSON format)."""
    import json
    
    try:
        data = json.load(file)
        tracker = TokenTracker(model)
        
        if isinstance(data, list) and all(isinstance(msg, dict) for msg in data):
            # Assume it's a list of messages
            usage = tracker.track_conversation(data)
            
            console.print(f"[green]Analyzed conversation with {len(data)} messages[/green]")
            _display_usage_status(usage, tracker)
            
        else:
            console.print("[red]Error: File should contain a JSON array of message objects[/red]")
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        console.print(f"[red]Error parsing JSON: {e}[/red]")
        sys.exit(1)


def _display_usage_status(usage, tracker) -> None:
    """Display current usage status with progress bar."""
    # Progress bar
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(
            f"Token Usage ({usage.model_name})",
            total=usage.max_tokens,
            completed=usage.total_tokens
        )
        progress.update(task, advance=0)  # Just to display it
    
    # Detailed info
    table = Table(show_header=False, box=None)
    table.add_column("Label", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Prompt Tokens:", f"{usage.prompt_tokens:,}")
    table.add_row("Completion Tokens:", f"{usage.completion_tokens:,}")
    table.add_row("Total Tokens:", f"{usage.total_tokens:,}")
    table.add_row("Max Tokens:", f"{usage.max_tokens:,}")
    table.add_row("Remaining:", f"{tracker.get_remaining_tokens():,}")
    
    if usage.cost_estimate:
        table.add_row("Est. Cost:", f"${usage.cost_estimate:.4f}")
    
    console.print(table)
    
    # Warning if near limit
    warning = tracker.get_warning_message()
    if warning:
        console.print(Panel(warning, style="bold red"))


if __name__ == "__main__":
    main()