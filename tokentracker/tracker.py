"""Core TokenTracker implementation."""

import tiktoken
import re
from typing import List, Optional, Dict, Any, Union
from .models import TokenUsage, ModelConfig, SUPPORTED_MODELS


class TokenTracker:
    """Track token usage for LLM interactions."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize TokenTracker with a specific model.
        
        Args:
            model_name: Name of the LLM model to track tokens for
        """
        self.model_name = model_name
        self.model_config = self._get_model_config(model_name)
        
        # Try to get the tiktoken encoding, fall back to approximation if no network
        try:
            self.encoding = tiktoken.get_encoding(self.model_config.encoding_name)
            self.use_tiktoken = True
        except Exception:
            # Network or other error - use approximation
            self.encoding = None
            self.use_tiktoken = False
            
        self.session_usage = TokenUsage(
            model_name=model_name,
            max_tokens=self.model_config.max_tokens
        )
        
    def _get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_name in SUPPORTED_MODELS:
            return SUPPORTED_MODELS[model_name]
        else:
            # Default fallback for unknown models
            return ModelConfig(
                name=model_name,
                max_tokens=4096,
                encoding_name="cl100k_base",
                provider="unknown"
            )
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Number of tokens in the text
        """
        if not text:
            return 0
            
        if self.use_tiktoken:
            try:
                return len(self.encoding.encode(text))
            except Exception:
                # Fall back to approximation if tiktoken fails
                pass
        
        # Approximation method when tiktoken is not available
        return self._approximate_token_count(text)
    
    def _approximate_token_count(self, text: str) -> int:
        """Approximate token count using simple heuristics.
        
        This is a fallback when tiktoken is not available.
        Based on the general rule that 1 token ≈ 4 characters for English text.
        """
        if not text:
            return 0
        
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b', text)
        
        # Count words, punctuation, and spaces
        word_count = len(words)
        
        # Account for punctuation and special characters
        punctuation_count = len(re.findall(r'[^\w\s]', text))
        
        # Rough approximation: most words are 1 token, some are 2+
        # Add punctuation as separate tokens
        estimated_tokens = word_count + punctuation_count
        
        # Adjust for longer words (rough heuristic)
        for word in words:
            if len(word) > 6:  # Longer words might be multiple tokens
                estimated_tokens += len(word) // 6
        
        return max(1, estimated_tokens)  # At least 1 token for non-empty text
    
    def count_tokens_from_messages(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages (ChatML format).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Total number of tokens including message formatting overhead
        """
        total_tokens = 0
        
        for message in messages:
            # Count tokens in content
            content_tokens = self.count_tokens(message.get('content', ''))
            
            # Add overhead for message formatting
            # This is an approximation based on OpenAI's token counting
            if self.model_config.provider == "openai":
                total_tokens += content_tokens + 4  # message overhead
                if message.get('name'):
                    total_tokens += self.count_tokens(message['name']) - 1
        
        # Add conversation overhead
        if self.model_config.provider == "openai":
            total_tokens += 2  # conversation overhead
            
        return total_tokens
    
    def track_prompt(self, text: str) -> TokenUsage:
        """Track tokens for a prompt.
        
        Args:
            text: The prompt text
            
        Returns:
            TokenUsage object with updated statistics
        """
        tokens = self.count_tokens(text)
        self.session_usage.prompt_tokens += tokens
        self._update_usage_stats()
        return self._get_current_usage()
    
    def track_completion(self, text: str) -> TokenUsage:
        """Track tokens for a completion/response.
        
        Args:
            text: The completion text
            
        Returns:
            TokenUsage object with updated statistics
        """
        tokens = self.count_tokens(text)
        self.session_usage.completion_tokens += tokens
        self._update_usage_stats()
        return self._get_current_usage()
    
    def track_conversation(self, messages: List[Dict[str, str]]) -> TokenUsage:
        """Track tokens for an entire conversation.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            TokenUsage object with conversation statistics
        """
        tokens = self.count_tokens_from_messages(messages)
        # For conversations, we count everything as prompt tokens
        self.session_usage.prompt_tokens += tokens
        self._update_usage_stats()
        return self._get_current_usage()
    
    def _update_usage_stats(self) -> None:
        """Update calculated statistics."""
        self.session_usage.total_tokens = (
            self.session_usage.prompt_tokens + self.session_usage.completion_tokens
        )
        self.session_usage.percentage_used = (
            (self.session_usage.total_tokens / self.model_config.max_tokens) * 100
        )
        
        # Calculate cost estimate if pricing is available
        if self.model_config.cost_per_1k_tokens:
            self.session_usage.cost_estimate = (
                (self.session_usage.total_tokens / 1000) * self.model_config.cost_per_1k_tokens
            )
    
    def _get_current_usage(self) -> TokenUsage:
        """Get current usage statistics."""
        return TokenUsage(
            prompt_tokens=self.session_usage.prompt_tokens,
            completion_tokens=self.session_usage.completion_tokens,
            total_tokens=self.session_usage.total_tokens,
            model_name=self.model_name,
            max_tokens=self.model_config.max_tokens,
            percentage_used=self.session_usage.percentage_used,
            cost_estimate=self.session_usage.cost_estimate
        )
    
    def reset(self) -> None:
        """Reset the session token usage."""
        self.session_usage = TokenUsage(
            model_name=self.model_name,
            max_tokens=self.model_config.max_tokens
        )
    
    def get_remaining_tokens(self) -> int:
        """Get the number of tokens remaining before hitting the limit."""
        return max(0, self.model_config.max_tokens - self.session_usage.total_tokens)
    
    def is_near_limit(self, threshold: float = 0.8) -> bool:
        """Check if token usage is near the model's limit.
        
        Args:
            threshold: Percentage threshold (0.0 to 1.0) to consider "near limit"
            
        Returns:
            True if usage is above the threshold
        """
        return (self.session_usage.total_tokens / self.model_config.max_tokens) >= threshold
    
    def get_warning_message(self, threshold: float = 0.8) -> Optional[str]:
        """Get a warning message if near token limit.
        
        Args:
            threshold: Percentage threshold to trigger warning
            
        Returns:
            Warning message if near limit, None otherwise
        """
        if self.is_near_limit(threshold):
            remaining = self.get_remaining_tokens()
            percentage = self.session_usage.percentage_used
            return (
                f"⚠️  Token usage warning: {percentage:.1f}% of limit reached. "
                f"Only {remaining} tokens remaining for {self.model_name}."
            )
        return None
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get list of supported model names."""
        return list(SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelConfig]:
        """Get information about a specific model."""
        return SUPPORTED_MODELS.get(model_name)