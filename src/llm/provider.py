"""
LLM Provider factory for multi-model support.
Supports OpenAI, Anthropic, Google, and Ollama.
"""

from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..config import settings


# ═══════════════════════════════════════════════════════════════
#  Model Usage Tracking
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelUsageStats:
    """Statistics for model usage."""
    model_id: str
    provider: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    total_tokens: int = 0
    last_used: Optional[datetime] = None
    avg_latency_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.invocations == 0:
            return 0.0
        return (self.successes / self.invocations) * 100


class ModelUsageTracker:
    """Global tracker for model usage statistics."""
    
    def __init__(self):
        self._stats: dict[str, ModelUsageStats] = {}
        self._session_start = datetime.now()
    
    def record_invocation(self, model_id: str, provider: str, success: bool, 
                         latency_ms: float = 0, tokens: int = 0):
        """Record a model invocation."""
        if model_id not in self._stats:
            self._stats[model_id] = ModelUsageStats(
                model_id=model_id,
                provider=provider
            )
        
        stats = self._stats[model_id]
        stats.invocations += 1
        stats.last_used = datetime.now()
        
        if success:
            stats.successes += 1
            stats.total_tokens += tokens
            # Running average of latency
            n = stats.successes
            stats.avg_latency_ms = ((stats.avg_latency_ms * (n - 1)) + latency_ms) / n
        else:
            stats.failures += 1
    
    def get_stats(self, model_id: Optional[str] = None) -> dict:
        """Get usage statistics."""
        if model_id:
            return self._stats.get(model_id)
        return dict(self._stats)
    
    def get_summary(self) -> str:
        """Get a formatted summary of all usage."""
        if not self._stats:
            return "No model usage recorded."
        
        lines = [
            "═" * 70,
            "📊 Model Usage Summary",
            "═" * 70,
            f"Session started: {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total models used: {len(self._stats)}",
            "",
        ]
        
        for model_id, stats in sorted(self._stats.items(), 
                                      key=lambda x: x[1].invocations, 
                                      reverse=True):
            lines.extend([
                f"🔹 {model_id} ({stats.provider})",
                f"   Invocations: {stats.invocations} | Success: {stats.successes} | Failed: {stats.failures}",
                f"   Success Rate: {stats.success_rate:.1f}%",
                f"   Total Tokens: {stats.total_tokens:,}" if stats.total_tokens > 0 else "",
                f"   Avg Latency: {stats.avg_latency_ms:.0f}ms" if stats.avg_latency_ms > 0 else "",
                f"   Last Used: {stats.last_used.strftime('%H:%M:%S')}" if stats.last_used else "",
                "",
            ])
        
        lines.append("═" * 70)
        return "\n".join(line for line in lines if line is not None)
    
    def reset(self):
        """Reset all statistics."""
        self._stats.clear()
        self._session_start = datetime.now()


# Global usage tracker instance
_usage_tracker = ModelUsageTracker()


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_chat_model(self, model_id: str) -> BaseChatModel:
        """Get a chat model instance."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        pass
    
    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """
        Perform a health check on the provider.
        
        Returns:
            (is_healthy, message) tuple
        """
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def is_available(self) -> bool:
        return bool(settings.openai_api_key)
    
    def health_check(self) -> tuple[bool, str]:
        """Check if OpenAI API is accessible."""
        if not self.is_available():
            return False, "OpenAI API key not configured"
        
        try:
            # Quick validation by creating client
            from langchain_openai import ChatOpenAI
            model = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=settings.openai_api_key,
                max_tokens=10,
            )
            # Don't actually call it - just check instantiation
            return True, "OpenAI provider configured"
        except Exception as e:
            return False, f"OpenAI configuration error: {str(e)}"

    def get_chat_model(self, model_id: str) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_id,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=4096,
        )


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider."""

    def is_available(self) -> bool:
        return bool(settings.anthropic_api_key)
    
    def health_check(self) -> tuple[bool, str]:
        """Check if Anthropic API is accessible."""
        if not self.is_available():
            return False, "Anthropic API key not configured"
        
        try:
            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(
                model="claude-3-haiku-20240307",
                api_key=settings.anthropic_api_key,
                max_tokens=10,
            )
            return True, "Anthropic provider configured"
        except Exception as e:
            return False, f"Anthropic configuration error: {str(e)}"

    def get_chat_model(self, model_id: str) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model_id,
            api_key=settings.anthropic_api_key,
            temperature=0.1,
            max_tokens=4096,
        )


class GoogleProvider(LLMProvider):
    """Google AI (Gemini) LLM provider."""

    def is_available(self) -> bool:
        return bool(settings.google_api_key)
    
    def health_check(self) -> tuple[bool, str]:
        """Check if Google AI API is accessible."""
        if not self.is_available():
            return False, "Google API key not configured"
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=settings.google_api_key,
                max_output_tokens=10,
            )
            return True, "Google AI provider configured"
        except Exception as e:
            return False, f"Google AI configuration error: {str(e)}"

    def get_chat_model(self, model_id: str) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=settings.google_api_key,
            temperature=0.1,
            max_output_tokens=4096,
        )


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def is_available(self) -> bool:
        # Ollama is always "available" - actual availability checked at runtime
        return True
    
    def health_check(self) -> tuple[bool, str]:
        """Check if Ollama server is accessible."""
        try:
            import requests
            response = requests.get(f"{settings.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return True, f"Ollama server running ({len(models)} models available)"
            return False, f"Ollama server returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Ollama at {settings.ollama_url}"
        except Exception as e:
            return False, f"Ollama health check failed: {str(e)}"

    def get_chat_model(self, model_id: str) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_id,
            base_url=settings.ollama_url,
            temperature=0.1,
        )


# Model to provider mapping
MODEL_PROVIDERS = {
    # Google Gemini
    "gemini-2.5-flash": "google",
    "gemini-2.5-pro": "google",
    "gemini-2.0-flash": "google",
    "gemini-3-flash": "google",
    "gemini-3-pro": "google",
}

# Provider instances
_providers: dict[str, LLMProvider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "google": GoogleProvider(),
    "ollama": OllamaProvider(),
}


def get_provider(provider_name: str) -> Optional[LLMProvider]:
    """Get a provider instance by name."""
    return _providers.get(provider_name)


def get_model_provider(model_id: str) -> Optional[str]:
    """Get the provider name for a model ID."""
    return MODEL_PROVIDERS.get(model_id)


def get_chat_model(model_id: str) -> BaseChatModel:
    """
    Get a chat model instance for the given model ID.
    
    Args:
        model_id: The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        
    Returns:
        A LangChain chat model instance
        
    Raises:
        ValueError: If the model is unknown or provider is not configured
    """
    provider_name = get_model_provider(model_id)
    
    if not provider_name:
        raise ValueError(f"Unknown model: {model_id}")
    
    provider = get_provider(provider_name)
    
    if not provider:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    if not provider.is_available():
        raise ValueError(
            f"Provider {provider_name} is not configured. "
            f"Please set the appropriate API key in environment variables."
        )
    
    return provider.get_chat_model(model_id)


def list_available_models() -> list[str]:
    """List all models that are currently available (provider configured)."""
    available = []
    
    for model_id, provider_name in MODEL_PROVIDERS.items():
        provider = get_provider(provider_name)
        if provider and provider.is_available():
            available.append(model_id)
    
    return available


def check_providers_health() -> dict[str, tuple[bool, str]]:
    """
    Check health of all providers.
    
    Returns:
        Dict mapping provider name to (is_healthy, message) tuple
    """
    health_status = {}
    
    for provider_name, provider in _providers.items():
        is_healthy, message = provider.health_check()
        health_status[provider_name] = (is_healthy, message)
        
        # Log the status
        status_emoji = "✅" if is_healthy else "❌"
        print(f"[Provider Health] {status_emoji} {provider_name}: {message}")
    
    return health_status


def get_default_model() -> str:
    """Get the default model based on what's available."""
    # Priority order - Gemini only
    priority = [
        "gemini-2.5-flash",    # Fast and capable
        "gemini-2.0-flash",    # Fallback flash
        "gemini-2.5-pro",      # Pro fallback
        "gemini-3-flash",      # Latest gen
        "gemini-3-pro",        # Most powerful
    ]
    
    available = list_available_models()
    
    print(f"[Model Selection] Available models: {available}")
    
    for model in priority:
        if model in available:
            print(f"[Model Selection] ✅ Selected: {model}")
            return model
    
    # Default fallback
    print(f"[Model Selection] ⚠️ Using fallback: gemini-2.5-flash (may fail if API key not set)")
    return "gemini-2.5-flash"


def validate_model_before_use(model_id: str) -> tuple[bool, str]:
    """
    Validate that a model can be used before attempting to use it.
    
    Returns:
        (is_valid, message) tuple
    """
    provider_name = get_model_provider(model_id)
    
    if not provider_name:
        return False, f"Unknown model: {model_id}"
    
    provider = get_provider(provider_name)
    
    if not provider:
        return False, f"Unknown provider: {provider_name}"
    
    if not provider.is_available():
        return False, f"Provider {provider_name} is not configured (missing API key)"
    
    # Run health check
    is_healthy, health_msg = provider.health_check()
    
    if not is_healthy:
        return False, f"Provider {provider_name} health check failed: {health_msg}"
    
    return True, f"Model {model_id} is ready to use"


class MultiModelChat:
    """
    A wrapper that allows switching between models dynamically.
    Useful for the LangGraph agent to use different models for different tasks.
    
    Features:
    - Model caching for performance
    - Usage tracking and statistics
    - Automatic retry with fallback models
    - Detailed logging of model selection
    """

    def __init__(self, default_model: str = None):
        self.default_model = default_model or get_default_model()
        self._model_cache: dict[str, BaseChatModel] = {}
        self._current_task: Optional[str] = None
        
        # Validate default model on initialization
        is_valid, msg = validate_model_before_use(self.default_model)
        if is_valid:
            print(f"[MultiModelChat] ✅ Initialized with model: {self.default_model}")
        else:
            print(f"[MultiModelChat] ⚠️ Warning: {msg}")
            print(f"[MultiModelChat]    Will attempt to use anyway (may fail at runtime)")

    def get_model(self, model_id: str = None) -> BaseChatModel:
        """Get a model, using cache for repeated access."""
        model_id = model_id or self.default_model
        
        if model_id not in self._model_cache:
            print(f"[MultiModelChat] 🔨 Creating new model instance: {model_id}")
            self._model_cache[model_id] = get_chat_model(model_id)
            print(f"[MultiModelChat] ✅ Model cached: {model_id}")
        else:
            print(f"[MultiModelChat] ♻️  Using cached model: {model_id}")
        
        return self._model_cache[model_id]
    
    def set_task_context(self, task_name: str):
        """Set the current task context for logging purposes."""
        self._current_task = task_name
        print(f"[MultiModelChat] 📋 Task: {task_name}")

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        model_id: str = None,
        retry_on_failure: bool = True,
    ) -> BaseMessage:
        """
        Invoke the model asynchronously with automatic retry.
        
        Args:
            messages: List of messages to send
            model_id: Optional model override
            retry_on_failure: Whether to retry with fallback model on failure
        """
        import time
        
        model_id = model_id or self.default_model
        provider_name = get_model_provider(model_id) or "unknown"
        
        task_info = f" [{self._current_task}]" if self._current_task else ""
        print(f"[MultiModelChat]{task_info} 🚀 Invoking: {model_id}")
        
        start_time = time.time()
        
        try:
            model = self.get_model(model_id)
            response = await model.ainvoke(messages)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens (rough approximation)
            prompt_tokens = sum(len(m.content.split()) for m in messages) * 1.3
            completion_tokens = len(response.content.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Record success
            _usage_tracker.record_invocation(
                model_id=model_id,
                provider=provider_name,
                success=True,
                latency_ms=latency_ms,
                tokens=total_tokens
            )
            
            print(f"[MultiModelChat]{task_info} ✅ Success: {model_id} ({latency_ms:.0f}ms, ~{total_tokens} tokens)")
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Record failure
            _usage_tracker.record_invocation(
                model_id=model_id,
                provider=provider_name,
                success=False,
                latency_ms=latency_ms
            )
            
            print(f"[MultiModelChat]{task_info} ❌ Failed: {model_id} - {str(e)}")
            
            # Retry with fallback if enabled
            if retry_on_failure and model_id != "gemini-2.0-flash":
                print(f"[MultiModelChat]{task_info} 🔄 Retrying with fallback: gemini-2.0-flash")
                return await self.ainvoke(messages, model_id="gemini-2.0-flash", retry_on_failure=False)
            
            raise

    def invoke(
        self,
        messages: list[BaseMessage],
        model_id: str = None,
        retry_on_failure: bool = True,
    ) -> BaseMessage:
        """
        Invoke the model synchronously with automatic retry.
        
        Args:
            messages: List of messages to send
            model_id: Optional model override
            retry_on_failure: Whether to retry with fallback model on failure
        """
        import time
        
        model_id = model_id or self.default_model
        provider_name = get_model_provider(model_id) or "unknown"
        
        task_info = f" [{self._current_task}]" if self._current_task else ""
        print(f"[MultiModelChat]{task_info} 🚀 Invoking: {model_id}")
        
        start_time = time.time()
        
        try:
            model = self.get_model(model_id)
            response = model.invoke(messages)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate tokens
            prompt_tokens = sum(len(m.content.split()) for m in messages) * 1.3
            completion_tokens = len(response.content.split()) * 1.3
            total_tokens = int(prompt_tokens + completion_tokens)
            
            # Record success
            _usage_tracker.record_invocation(
                model_id=model_id,
                provider=provider_name,
                success=True,
                latency_ms=latency_ms,
                tokens=total_tokens
            )
            
            print(f"[MultiModelChat]{task_info} ✅ Success: {model_id} ({latency_ms:.0f}ms, ~{total_tokens} tokens)")
            return response
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Record failure
            _usage_tracker.record_invocation(
                model_id=model_id,
                provider=provider_name,
                success=False,
                latency_ms=latency_ms
            )
            
            print(f"[MultiModelChat]{task_info} ❌ Failed: {model_id} - {str(e)}")
            
            # Retry with fallback if enabled
            if retry_on_failure and model_id != "gemini-2.0-flash":
                print(f"[MultiModelChat]{task_info} 🔄 Retrying with fallback: gemini-2.0-flash")
                return self.invoke(messages, model_id="gemini-2.0-flash", retry_on_failure=False)
            
            raise
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics for all models."""
        return _usage_tracker.get_stats()
    
    def print_usage_summary(self):
        """Print a formatted usage summary."""
        print(_usage_tracker.get_summary())

    async def analyze_code(
        self,
        code: str,
        prompt: str,
        model_id: str = None
    ) -> str:
        """
        Analyze code with a given prompt.
        
        Args:
            code: The code to analyze
            prompt: The analysis prompt
            model_id: Optional model to use
            
        Returns:
            The model's response as a string
        """
        messages = [
            SystemMessage(content=(
                "You are an expert code analyst. Analyze the provided code "
                "carefully and provide detailed, actionable insights."
            )),
            HumanMessage(content=f"{prompt}\n\n```\n{code}\n```"),
        ]
        
        response = await self.ainvoke(messages, model_id)
        return response.content
