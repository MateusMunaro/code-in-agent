"""
LLM package for the Code Indexer Agent.
"""

from .provider import (
    get_chat_model,
    list_available_models,
    get_default_model,
    check_providers_health,
    validate_model_before_use,
    MultiModelChat,
    ModelUsageStats,
    ModelUsageTracker,
)
from .embeddings import (
    EmbeddingService,
    chunk_code,
    chunk_by_structure,
)

__all__ = [
    "get_chat_model",
    "list_available_models",
    "get_default_model",
    "check_providers_health",
    "validate_model_before_use",
    "MultiModelChat",
    "ModelUsageStats",
    "ModelUsageTracker",
    "EmbeddingService",
    "chunk_code",
    "chunk_by_structure",
]
