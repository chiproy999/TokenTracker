#!/usr/bin/env python3
"""Demonstration of TokenTracker key features."""

from tokentracker import TokenTracker
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def main():
    console.print(Panel.fit("🎯 TokenTracker Demo", style="bold blue"))
    
    # 1. Basic token counting
    console.print("\n[bold]1. Basic Token Counting[/bold]")
    tracker = TokenTracker("gpt-3.5-turbo")
    
    texts = [
        "Hello world!",
        "This is a longer piece of text to demonstrate token counting.",
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test text processing systems."
    ]
    
    table = Table(title="Token Counts")
    table.add_column("Text", style="cyan", width=50)
    table.add_column("Tokens", style="magenta", justify="right")
    
    for text in texts:
        count = tracker.count_tokens(text)
        display_text = text if len(text) <= 47 else text[:44] + "..."
        table.add_row(display_text, str(count))
    
    console.print(table)
    
    # 2. Different models comparison
    console.print("\n[bold]2. Model Comparison[/bold]")
    models_to_compare = ["gpt-3.5-turbo", "gpt-4", "claude-3-haiku"]
    
    table = Table(title="Model Limits")
    table.add_column("Model", style="yellow")
    table.add_column("Max Tokens", style="green", justify="right")
    table.add_column("Cost per 1K", style="magenta", justify="right")
    
    for model_name in models_to_compare:
        tracker = TokenTracker(model_name)
        cost = f"${tracker.model_config.cost_per_1k_tokens:.4f}" if tracker.model_config.cost_per_1k_tokens else "N/A"
        table.add_row(model_name, f"{tracker.model_config.max_tokens:,}", cost)
    
    console.print(table)
    
    # 3. Usage tracking
    console.print("\n[bold]3. Usage Tracking & Warnings[/bold]")
    tracker = TokenTracker("gpt-4")
    
    # Simulate some usage
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Explain quantum computing to a 10-year-old.",
        "Create a recipe for chocolate chip cookies with detailed instructions.",
        "Discuss the environmental impact of renewable energy sources."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        usage = tracker.track_prompt(prompt)
        console.print(f"Prompt {i}: {usage.prompt_tokens} tokens (Total: {usage.total_tokens})")
    
    # Show final usage
    remaining = tracker.get_remaining_tokens()
    console.print(f"\n[green]✓ Used {tracker.session_usage.total_tokens} tokens ({tracker.session_usage.percentage_used:.2f}%)[/green]")
    console.print(f"[blue]📊 Remaining: {remaining:,} tokens[/blue]")
    
    if tracker.session_usage.cost_estimate:
        console.print(f"[yellow]💰 Estimated cost: ${tracker.session_usage.cost_estimate:.4f}[/yellow]")
    
    # 4. Warning demonstration
    console.print("\n[bold]4. Limit Warning System[/bold]")
    
    # Create a tracker with artificially high usage
    high_usage_tracker = TokenTracker("gpt-3.5-turbo")  # Smaller context for demo
    high_usage_tracker.session_usage.prompt_tokens = 3500  # 85% of 4096
    high_usage_tracker._update_usage_stats()
    
    warning = high_usage_tracker.get_warning_message(threshold=0.8)
    if warning:
        console.print(Panel(warning, style="bold red"))
    
    console.print("\n[bold green]🎉 Demo complete! TokenTracker is ready to help you manage your LLM token usage.[/bold green]")

if __name__ == "__main__":
    main()