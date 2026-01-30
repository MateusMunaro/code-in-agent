"""
Configuration management for the Code Indexer Agent.
Loads environment variables and provides typed configuration.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379", alias="REDIS_URL")

    # Supabase Configuration
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_service_key: str = Field(default="", alias="SUPABASE_SERVICE_KEY")

    # LLM API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")

    # Agent Configuration
    max_files_to_analyze: int = Field(default=100, alias="MAX_FILES_TO_ANALYZE")
    max_file_size_kb: int = Field(default=500, alias="MAX_FILE_SIZE_KB")
    confidence_threshold: float = Field(default=0.8, alias="CONFIDENCE_THRESHOLD")
    max_reasoning_iterations: int = Field(default=5, alias="MAX_REASONING_ITERATIONS")

    # Repository Storage
    repos_base_path: str = Field(default="/tmp/code-indexer/repos", alias="REPOS_BASE_PATH")

    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=1536, alias="EMBEDDING_DIMENSION")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()


def get_available_providers() -> list[str]:
    """Return list of configured LLM providers."""
    providers = []
    
    if settings.openai_api_key:
        providers.append("openai")
    if settings.anthropic_api_key:
        providers.append("anthropic")
    if settings.google_api_key:
        providers.append("google")
    
    # Always include Ollama as it might be running locally
    providers.append("ollama")
    
    return providers


def validate_model_availability(model_id: str) -> tuple[bool, str]:
    """Check if a model can be used based on configuration."""
    
    model_providers = {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "gpt-4-turbo": "openai",
        "claude-3-5-sonnet-20241022": "anthropic",
        "claude-3-opus-20240229": "anthropic",
        "claude-3-haiku-20240307": "anthropic",
        "gemini-2.0-flash": "google",
        "gemini-1.5-pro": "google",
        "llama3.2": "ollama",
        "codellama": "ollama",
        "deepseek-coder-v2": "ollama",
        "qwen2.5-coder": "ollama",
    }
    
    provider = model_providers.get(model_id)
    
    if not provider:
        return False, f"Unknown model: {model_id}"
    
    if provider == "openai" and not settings.openai_api_key:
        return False, "OPENAI_API_KEY not configured"
    
    if provider == "anthropic" and not settings.anthropic_api_key:
        return False, "ANTHROPIC_API_KEY not configured"
    
    if provider == "google" and not settings.google_api_key:
        return False, "GOOGLE_API_KEY not configured"
    
    return True, "OK"
