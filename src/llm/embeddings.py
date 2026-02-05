"""
Embeddings service for generating vector embeddings.
Supports OpenAI and Ollama embedding models.
"""

from typing import Optional
import asyncio

from langchain_core.embeddings import Embeddings

from ..config import settings


class EmbeddingService:
    """Service for generating text embeddings for semantic search."""

    def __init__(self, model: str = None, provider: str = None):
        """
        Initialize the embedding service.
        
        Args:
            model: The embedding model to use
            provider: The provider ("openai" or "ollama")
        """
        self.model = model or settings.embedding_model
        self.provider = provider or self._detect_provider()
        self._embeddings: Optional[Embeddings] = None

    def _detect_provider(self) -> str:
        """Detect the best available embedding provider."""
        if settings.google_api_key:
            return "google"
        if settings.openai_api_key:
            return "openai"
        return "ollama"

    def _get_embeddings(self) -> Embeddings:
        """Get or create the embeddings instance."""
        if self._embeddings is not None:
            return self._embeddings

        if self.provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=self.model,
                api_key=settings.openai_api_key,
            )
        elif self.provider == "google":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",  # Google's text embedding model
                google_api_key=settings.google_api_key,
            )
        elif self.provider == "ollama":
            from langchain_ollama import OllamaEmbeddings

            self._embeddings = OllamaEmbeddings(
                model="nomic-embed-text",  # Good embedding model for Ollama
                base_url=settings.ollama_url,
            )
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

        return self._embeddings

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embeddings = self._get_embeddings()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: embeddings.embed_query(text)
        )
        
        return result

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self._get_embeddings()
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: embeddings.embed_documents(texts)
        )
        
        return results

    async def embed_code_chunks(
        self,
        chunks: list[dict],
        batch_size: int = 50
    ) -> list[dict]:
        """
        Embed code chunks with metadata.
        
        Args:
            chunks: List of dicts with 'content' and 'metadata' keys
            batch_size: Number of chunks to process at once
            
        Returns:
            List of dicts with 'embedding' added
        """
        results = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["content"] for c in batch]
            
            embeddings = await self.embed_texts(texts)
            
            for chunk, embedding in zip(batch, embeddings):
                results.append({
                    **chunk,
                    "embedding": embedding,
                })

        return results


def chunk_code(
    content: str,
    file_path: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    language: str = None
) -> list[dict]:
    """
    Split code into chunks for embedding.
    
    Uses intelligent splitting that respects code structure
    (functions, classes, etc.) when possible.
    
    Args:
        content: The code content
        file_path: Path to the file (for metadata)
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        language: Programming language
        
    Returns:
        List of chunk dicts with content and metadata
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks = []
    lines = content.split("\n")
    
    current_chunk = []
    current_size = 0
    chunk_index = 0
    start_line = 0

    for i, line in enumerate(lines):
        line_size = len(line) + 1  # +1 for newline
        
        # Check if adding this line exceeds chunk size
        if current_size + line_size > chunk_size and current_chunk:
            # Save current chunk
            chunk_content = "\n".join(current_chunk)
            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "file_path": file_path,
                    "language": language,
                    "chunk_index": chunk_index,
                    "start_line": start_line + 1,
                    "end_line": i,
                    "char_count": len(chunk_content),
                },
            })
            
            # Start new chunk with overlap
            overlap_lines = int(chunk_overlap / 50)  # Approximate lines for overlap
            current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
            current_size = sum(len(l) + 1 for l in current_chunk)
            start_line = max(0, i - len(current_chunk))
            chunk_index += 1

        current_chunk.append(line)
        current_size += line_size

    # Don't forget the last chunk
    if current_chunk:
        chunk_content = "\n".join(current_chunk)
        chunks.append({
            "content": chunk_content,
            "metadata": {
                "file_path": file_path,
                "language": language,
                "chunk_index": chunk_index,
                "start_line": start_line + 1,
                "end_line": len(lines),
                "char_count": len(chunk_content),
            },
        })

    return chunks


def chunk_by_structure(
    content: str,
    file_path: str,
    language: str
) -> list[dict]:
    """
    Split code by structural elements (functions, classes).
    Falls back to line-based chunking if structure detection fails.
    
    Args:
        content: The code content
        file_path: Path to the file
        language: Programming language
        
    Returns:
        List of chunk dicts
    """
    import re

    chunks = []
    
    # Language-specific patterns for splitting
    patterns = {
        "python": [
            r"^(class\s+\w+.*?:.*?)(?=^class\s|\Z)",
            r"^(def\s+\w+.*?:.*?)(?=^def\s|^class\s|\Z)",
        ],
        "javascript": [
            r"((?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\)\s*\{[^}]*\})",
            r"(class\s+\w+\s*(?:extends\s+\w+)?\s*\{[^}]*\})",
        ],
        "typescript": [
            r"((?:export\s+)?(?:async\s+)?function\s+\w+\s*(?:<[^>]+>)?\s*\([^)]*\)(?:\s*:\s*\w+)?\s*\{[^}]*\})",
            r"((?:export\s+)?class\s+\w+\s*(?:<[^>]+>)?(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{[^}]*\})",
            r"((?:export\s+)?interface\s+\w+\s*(?:<[^>]+>)?\s*\{[^}]*\})",
        ],
    }

    lang_patterns = patterns.get(language, [])
    
    if not lang_patterns:
        # Fallback to line-based chunking
        return chunk_code(content, file_path, language=language)

    # Try structure-based splitting
    remaining = content
    chunk_index = 0

    for pattern in lang_patterns:
        matches = list(re.finditer(pattern, remaining, re.MULTILINE | re.DOTALL))
        
        for match in matches:
            chunk_content = match.group(1).strip()
            if len(chunk_content) > 50:  # Skip tiny chunks
                # Find line numbers
                start_pos = match.start()
                start_line = content[:start_pos].count("\n") + 1
                end_line = start_line + chunk_content.count("\n")

                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "file_path": file_path,
                        "language": language,
                        "chunk_index": chunk_index,
                        "start_line": start_line,
                        "end_line": end_line,
                        "char_count": len(chunk_content),
                        "type": "structure",
                    },
                })
                chunk_index += 1

    # If no structural chunks found, fall back to line-based
    if not chunks:
        return chunk_code(content, file_path, language=language)

    return chunks
