"""
LLM package for the Code Indexer Agent.
"""

from .provider import (
    get_chat_model,
    list_available_models,
    get_default_model,
    MultiModelChat,
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
    "MultiModelChat",
    "EmbeddingService",
    "chunk_code",
    "chunk_by_structure",
]
