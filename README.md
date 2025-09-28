# TokenTracker

**Track context lengths for LLMs so you know when you're close to the limit**

TokenTracker is a Python library and CLI tool that helps you monitor token usage when working with Large Language Models (LLMs). It provides accurate token counting, usage tracking, cost estimation, and warnings when approaching context limits.

## Features

- 🔢 **Accurate Token Counting**: Uses tiktoken for precise token counting across different model encodings
- 📊 **Usage Tracking**: Track tokens for prompts, completions, and entire conversations
- ⚠️ **Limit Warnings**: Get alerts when approaching model context limits
- 💰 **Cost Estimation**: Calculate estimated costs for API calls (where pricing is available)
- 🎯 **Multi-Model Support**: Support for OpenAI GPT models and Anthropic Claude models
- 🖥️ **CLI Interface**: Easy-to-use command-line tools
- 📚 **Rich Output**: Beautiful terminal output with progress bars and tables

## Installation

```bash
pip install tokentracker
```

Or install from source:

```bash
git clone https://github.com/chiproy999/TokenTracker.git
cd TokenTracker
pip install -e .
```

## Quick Start

### Python API

```python
from tokentracker import TokenTracker

# Initialize tracker for your model
tracker = TokenTracker("gpt-3.5-turbo")

# Track a prompt
usage = tracker.track_prompt("Explain quantum computing")
print(f"Tokens used: {usage.total_tokens}")
print(f"Percentage of limit: {usage.percentage_used:.1f}%")

# Track a completion
usage = tracker.track_completion("Quantum computing is...")
print(f"Total tokens: {usage.total_tokens}")

# Check if near limit
if tracker.is_near_limit():
    print("⚠️ Approaching token limit!")
```

### CLI Usage

Count tokens in text:
```bash
tokentracker count "Your text here"
tokentracker count --model gpt-4 --file document.txt
```

Interactive session:
```bash
tokentracker interactive --model gpt-3.5-turbo
```

Analyze conversation files:
```bash
tokentracker analyze conversation.json --model gpt-4
```

List supported models:
```bash
tokentracker models
```

## Supported Models

| Model | Provider | Max Tokens | Cost per 1K tokens |
|-------|----------|------------|-------------------|
| gpt-3.5-turbo | OpenAI | 4,096 | $0.002 |
| gpt-3.5-turbo-16k | OpenAI | 16,384 | $0.004 |
| gpt-4 | OpenAI | 8,192 | $0.03 |
| gpt-4-32k | OpenAI | 32,768 | $0.06 |
| gpt-4-turbo | OpenAI | 128,000 | $0.01 |
| claude-3-haiku | Anthropic | 200,000 | $0.00025 |
| claude-3-sonnet | Anthropic | 200,000 | $0.003 |
| claude-3-opus | Anthropic | 200,000 | $0.015 |

## Usage Examples

### Basic Token Counting

```python
from tokentracker import TokenTracker

tracker = TokenTracker("gpt-4")

# Count tokens in a string
count = tracker.count_tokens("Hello, world!")
print(f"Token count: {count}")

# Track a conversation
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."}
]
usage = tracker.track_conversation(messages)
print(f"Conversation uses {usage.total_tokens} tokens")
```

### Warning System

```python
tracker = TokenTracker("gpt-3.5-turbo")

# ... add some content ...

# Check for warnings
warning = tracker.get_warning_message(threshold=0.8)  # Warn at 80%
if warning:
    print(warning)

# Check remaining capacity
remaining = tracker.get_remaining_tokens()
print(f"Tokens remaining: {remaining}")
```

### Cost Estimation

```python
tracker = TokenTracker("gpt-4")
usage = tracker.track_prompt("Long prompt here...")

if usage.cost_estimate:
    print(f"Estimated cost: ${usage.cost_estimate:.4f}")
```

## CLI Examples

### Count tokens in a file
```bash
tokentracker count --file my_document.txt --model gpt-4
```

### Interactive session
```bash
tokentracker interactive --model claude-3-sonnet
>>> Hello, how are you?
>>> /status
>>> /reset
>>> /quit
```

### Analyze a conversation
```bash
# conversation.json should contain an array of message objects
tokentracker analyze conversation.json --model gpt-4-turbo
```

## Development

Clone the repository and install development dependencies:

```bash
git clone https://github.com/chiproy999/TokenTracker.git
cd TokenTracker
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

Format code:
```bash
black tokentracker tests
isort tokentracker tests
```

Lint code:
```bash
flake8 tokentracker tests
mypy tokentracker
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0
- Initial release
- Support for OpenAI and Anthropic models
- CLI interface with multiple commands
- Token counting, usage tracking, and cost estimation
- Warning system for approaching limits
- Interactive mode and conversation analysis
