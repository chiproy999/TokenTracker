"""Tests for TokenTracker core functionality."""

import pytest
from tokentracker.tracker import TokenTracker
from tokentracker.models import SUPPORTED_MODELS


class TestTokenTracker:
    """Test cases for TokenTracker."""
    
    def test_initialization(self):
        """Test TokenTracker initialization."""
        tracker = TokenTracker("gpt-3.5-turbo")
        assert tracker.model_name == "gpt-3.5-turbo"
        assert tracker.model_config.name == "gpt-3.5-turbo"
        assert tracker.session_usage.total_tokens == 0
    
    def test_unknown_model_fallback(self):
        """Test fallback for unknown models."""
        tracker = TokenTracker("unknown-model")
        assert tracker.model_name == "unknown-model"
        assert tracker.model_config.max_tokens == 4096  # Default fallback
    
    def test_count_tokens_basic(self):
        """Test basic token counting."""
        tracker = TokenTracker()
        
        # Test empty string
        assert tracker.count_tokens("") == 0
        
        # Test simple text
        count = tracker.count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_tokens_from_messages(self):
        """Test counting tokens from messages."""
        tracker = TokenTracker()
        
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        count = tracker.count_tokens_from_messages(messages)
        assert count > 0
        
        # Should be more than just the content due to formatting overhead
        content_only = tracker.count_tokens("Hello!") + tracker.count_tokens("Hi there!")
        assert count > content_only
    
    def test_track_prompt(self):
        """Test prompt tracking."""
        tracker = TokenTracker()
        
        usage = tracker.track_prompt("This is a test prompt")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == usage.prompt_tokens
        assert usage.percentage_used > 0
    
    def test_track_completion(self):
        """Test completion tracking."""
        tracker = TokenTracker()
        
        # Track a prompt first
        tracker.track_prompt("Test prompt")
        
        # Then track completion
        usage = tracker.track_completion("This is a response")
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    
    def test_track_conversation(self):
        """Test conversation tracking."""
        tracker = TokenTracker()
        
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
        ]
        
        usage = tracker.track_conversation(messages)
        assert usage.prompt_tokens > 0
        assert usage.total_tokens > 0
    
    def test_reset(self):
        """Test session reset."""
        tracker = TokenTracker()
        
        # Track some tokens
        tracker.track_prompt("Test")
        assert tracker.session_usage.total_tokens > 0
        
        # Reset
        tracker.reset()
        assert tracker.session_usage.total_tokens == 0
        assert tracker.session_usage.prompt_tokens == 0
        assert tracker.session_usage.completion_tokens == 0
    
    def test_remaining_tokens(self):
        """Test remaining tokens calculation."""
        tracker = TokenTracker()
        
        initial_remaining = tracker.get_remaining_tokens()
        assert initial_remaining == tracker.model_config.max_tokens
        
        # Use some tokens
        tracker.track_prompt("Test")
        remaining = tracker.get_remaining_tokens()
        assert remaining < initial_remaining
        assert remaining >= 0
    
    def test_near_limit_detection(self):
        """Test near limit detection."""
        tracker = TokenTracker()
        
        # Initially should not be near limit
        assert not tracker.is_near_limit()
        
        # Artificially set high usage
        tracker.session_usage.prompt_tokens = int(tracker.model_config.max_tokens * 0.9)
        tracker._update_usage_stats()
        
        assert tracker.is_near_limit(threshold=0.8)
        
        warning = tracker.get_warning_message()
        assert warning is not None
        assert "warning" in warning.lower()
    
    def test_cost_estimation(self):
        """Test cost estimation for models with pricing."""
        tracker = TokenTracker("gpt-4")  # Has pricing info
        
        usage = tracker.track_prompt("This is a test for cost estimation")
        assert usage.cost_estimate is not None
        assert usage.cost_estimate > 0
    
    def test_list_supported_models(self):
        """Test listing supported models."""
        models = TokenTracker.list_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "gpt-3.5-turbo" in models
        assert "gpt-4" in models
    
    def test_get_model_info(self):
        """Test getting model information."""
        info = TokenTracker.get_model_info("gpt-4")
        assert info is not None
        assert info.name == "gpt-4"
        assert info.max_tokens > 0
        
        # Test unknown model
        unknown_info = TokenTracker.get_model_info("nonexistent-model")
        assert unknown_info is None
    
    def test_percentage_calculation(self):
        """Test percentage usage calculation."""
        tracker = TokenTracker()
        
        # Track some tokens
        usage = tracker.track_prompt("Test prompt for percentage calculation")
        
        expected_percentage = (usage.total_tokens / tracker.model_config.max_tokens) * 100
        assert abs(usage.percentage_used - expected_percentage) < 0.01


class TestModelConfigurations:
    """Test model configuration data."""
    
    def test_all_models_have_required_fields(self):
        """Test that all supported models have required configuration fields."""
        for model_name, config in SUPPORTED_MODELS.items():
            assert config.name == model_name
            assert config.max_tokens > 0
            assert config.encoding_name
            assert config.provider
            # cost_per_1k_tokens is optional
    
    def test_model_providers(self):
        """Test that models have expected providers."""
        openai_models = [name for name, config in SUPPORTED_MODELS.items() 
                        if config.provider == "openai"]
        anthropic_models = [name for name, config in SUPPORTED_MODELS.items() 
                           if config.provider == "anthropic"]
        
        assert len(openai_models) > 0
        assert len(anthropic_models) > 0
        assert "gpt-3.5-turbo" in openai_models
        assert "claude-3-haiku" in anthropic_models