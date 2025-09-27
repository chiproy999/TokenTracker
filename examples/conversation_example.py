#!/usr/bin/env python3
"""Example of tracking a conversation with TokenTracker."""

from tokentracker import TokenTracker

def main():
    # Initialize tracker for GPT-4
    tracker = TokenTracker("gpt-4")
    
    print(f"Tracking conversation for: {tracker.model_name}")
    print(f"Max tokens: {tracker.model_config.max_tokens:,}")
    print()
    
    # Example conversation
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that explains complex topics simply."},
        {"role": "user", "content": "What is quantum computing?"},
        {"role": "assistant", "content": "Quantum computing is a revolutionary approach to computation that harnesses the principles of quantum mechanics. Unlike classical computers that use bits (0s and 1s), quantum computers use quantum bits or 'qubits' that can exist in multiple states simultaneously through a property called superposition."},
        {"role": "user", "content": "How is it different from regular computers?"},
        {"role": "assistant", "content": "The key differences are: 1) Qubits can be in superposition (multiple states at once), 2) Quantum entanglement allows qubits to be correlated in ways classical bits cannot, 3) This enables quantum computers to explore many possible solutions simultaneously, potentially solving certain problems exponentially faster than classical computers."},
        {"role": "user", "content": "What are some practical applications?"},
        {"role": "assistant", "content": "Practical applications include: cryptography and security, drug discovery and molecular modeling, financial portfolio optimization, weather prediction, artificial intelligence and machine learning, logistics and supply chain optimization. However, we're still in the early stages - most quantum computers today are experimental and limited in scope."}
    ]
    
    # Track the entire conversation
    usage = tracker.track_conversation(conversation)
    
    print(f"Conversation messages: {len(conversation)}")
    print(f"Total tokens used: {usage.total_tokens:,}")
    print(f"Percentage of limit: {usage.percentage_used:.2f}%")
    print(f"Remaining tokens: {tracker.get_remaining_tokens():,}")
    
    if usage.cost_estimate:
        print(f"Estimated cost: ${usage.cost_estimate:.4f}")
    
    print()
    
    # Check if we're approaching limits
    warning = tracker.get_warning_message(threshold=0.7)  # Warn at 70%
    if warning:
        print(warning)
    else:
        print("✅ Conversation is within safe token limits")
    
    # Show how much more conversation we could have
    avg_tokens_per_message = usage.total_tokens / len(conversation)
    estimated_remaining_messages = tracker.get_remaining_tokens() / avg_tokens_per_message
    
    print(f"\nAverage tokens per message: {avg_tokens_per_message:.1f}")
    print(f"Estimated remaining messages: {estimated_remaining_messages:.0f}")


if __name__ == "__main__":
    main()