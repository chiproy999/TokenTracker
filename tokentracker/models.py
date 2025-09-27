"""Data models for TokenTracker."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pydantic import BaseModel, model_validator


class ModelConfig(BaseModel):
    """Configuration for an LLM model."""
    
    name: str
    max_tokens: int
    encoding_name: str
    provider: str
    cost_per_1k_tokens: Optional[float] = None
    
    
class TokenUsage(BaseModel):
    """Token usage information."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: str
    max_tokens: int
    percentage_used: float = 0.0
    cost_estimate: Optional[float] = None
    
    @model_validator(mode='after')
    def calculate_derived_fields(self):
        """Calculate derived fields."""
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        if self.max_tokens > 0:
            self.percentage_used = (self.total_tokens / self.max_tokens) * 100
        return self


# Pre-defined model configurations
SUPPORTED_MODELS = {
    # OpenAI GPT Models
    "gpt-4": ModelConfig(
        name="gpt-4",
        max_tokens=8192,
        encoding_name="cl100k_base",
        provider="openai",
        cost_per_1k_tokens=0.03
    ),
    "gpt-4-32k": ModelConfig(
        name="gpt-4-32k",
        max_tokens=32768,
        encoding_name="cl100k_base",
        provider="openai",
        cost_per_1k_tokens=0.06
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        max_tokens=128000,
        encoding_name="cl100k_base",
        provider="openai",
        cost_per_1k_tokens=0.01
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        max_tokens=4096,
        encoding_name="cl100k_base",
        provider="openai",
        cost_per_1k_tokens=0.002
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        name="gpt-3.5-turbo-16k",
        max_tokens=16384,
        encoding_name="cl100k_base",
        provider="openai",
        cost_per_1k_tokens=0.004
    ),
    
    # Anthropic Claude Models
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        max_tokens=200000,
        encoding_name="cl100k_base",  # Approximation
        provider="anthropic",
        cost_per_1k_tokens=0.015
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet",
        max_tokens=200000,
        encoding_name="cl100k_base",  # Approximation
        provider="anthropic",
        cost_per_1k_tokens=0.003
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku",
        max_tokens=200000,
        encoding_name="cl100k_base",  # Approximation
        provider="anthropic",
        cost_per_1k_tokens=0.00025
    ),
}