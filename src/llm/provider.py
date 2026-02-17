"""
LLM Provider factory for multi-model support.
Supports OpenAI, Anthropic, Google, and Ollama.
"""

from typing import Optional, Union
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from ..config import settings


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


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def is_available(self) -> bool:
        return bool(settings.openai_api_key)

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
    
    for model in priority:
        if model in available:
            return model
    
    # Default fallback
    return "gemini-2.5-flash"


class MultiModelChat:
    """
    A wrapper that allows switching between models dynamically.
    Useful for the LangGraph agent to use different models for different tasks.
    """

    def __init__(self, default_model: str = None):
        self.default_model = default_model or get_default_model()
        self._model_cache: dict[str, BaseChatModel] = {}

    def get_model(self, model_id: str = None) -> BaseChatModel:
        """Get a model, using cache for repeated access."""
        model_id = model_id or self.default_model
        
        if model_id not in self._model_cache:
            self._model_cache[model_id] = get_chat_model(model_id)
        
        return self._model_cache[model_id]

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        model_id: str = None
    ) -> BaseMessage:
        """Invoke the model asynchronously."""
        model = self.get_model(model_id)
        return await model.ainvoke(messages)

    def invoke(
        self,
        messages: list[BaseMessage],
        model_id: str = None
    ) -> BaseMessage:
        """Invoke the model synchronously."""
        model = self.get_model(model_id)
        return model.invoke(messages)

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
