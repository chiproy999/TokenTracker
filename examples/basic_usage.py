#!/usr/bin/env python3
"""Basic usage example for TokenTracker."""

from tokentracker import TokenTracker

def main():
    # Initialize tracker for GPT-3.5-turbo
    tracker = TokenTracker("gpt-3.5-turbo")
    
    print(f"Tracking tokens for: {tracker.model_name}")
    print(f"Max tokens: {tracker.model_config.max_tokens}")
    print()
    
    # Track a simple prompt
    prompt = "Explain the concept of machine learning in simple terms."
    usage = tracker.track_prompt(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Tokens used: {usage.prompt_tokens}")
    print(f"Percentage of limit: {usage.percentage_used:.2f}%")
    print()
    
    # Track a response
    response = """Machine learning is a subset of artificial intelligence that enables 
    computers to learn and make decisions from data without being explicitly programmed 
    for every scenario. Instead of following pre-written rules, ML algorithms identify 
    patterns in data and use these patterns to make predictions or decisions about new, 
    unseen data."""
    
    usage = tracker.track_completion(response)
    
    print(f"Response tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    print(f"Remaining tokens: {tracker.get_remaining_tokens()}")
    
    if usage.cost_estimate:
        print(f"Estimated cost: ${usage.cost_estimate:.4f}")
    
    # Check if we're near the limit
    warning = tracker.get_warning_message()
    if warning:
        print(f"\n{warning}")
    else:
        print("\n✅ Token usage is within safe limits")


if __name__ == "__main__":
    main()