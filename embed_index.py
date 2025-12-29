#!/usr/bin/env python3
"""
Embedding Index Module for MarketResearchAI.

Provides embeddings-based retrieval with multi-backend support:
1. sentence-transformers (preferred, configurable via SENTENCE_TRANSFORMER_MODEL env var)
2. OpenAI embeddings (fallback if OPENAI_API_KEY set)
3. Deterministic hash-based embeddings (offline/test fallback)

CLI Usage:
    python embed_index.py --build --input data/documents.json
    python embed_index.py --query "market trends" --top_k 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embed_index")

# Constants
DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2
DEFAULT_INDEX_PATH = Path("data/embeddings.json")
DEFAULT_TOP_K = 5


def get_sentence_transformer_model() -> str:
    """Get the sentence transformer model name from environment or default."""
    return os.environ.get("SENTENCE_TRANSFORMER_MODEL", DEFAULT_MODEL)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector
        b: Second vector

    Returns:
        Cosine similarity score between -1 and 1
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def hash_based_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Generate a deterministic hash-based embedding for offline/test use.

    This creates a reproducible pseudo-embedding based on the text hash.
    NOT suitable for semantic similarity in production.

    Args:
        text: Input text to embed
        dim: Embedding dimension

    Returns:
        List of floats representing the embedding
    """
    # Create deterministic hash
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Use hash to seed random generator for reproducibility
    seed = int(text_hash[:8], 16)
    rng = np.random.default_rng(seed)

    # Generate unit vector
    embedding = rng.standard_normal(dim)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding.tolist()


class EmbeddingBackend:
    """Abstract base for embedding backends."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""
        raise NotImplementedError

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]


class SentenceTransformerBackend(EmbeddingBackend):
    """Sentence Transformers embedding backend."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize the sentence transformer backend.

        Args:
            model_name: Model name or path (defaults to env var or all-MiniLM-L6-v2)
        """
        self.model_name = model_name or get_sentence_transformer_model()
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                raise
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using sentence transformers.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class OpenAIBackend(EmbeddingBackend):
    """OpenAI embeddings backend."""

    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI backend.

        Args:
            model: OpenAI embedding model name
        """
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()  # Uses OPENAI_API_KEY env var
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class HashBackend(EmbeddingBackend):
    """Deterministic hash-based embedding backend for offline use."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        """
        Initialize hash backend.

        Args:
            dim: Embedding dimension
        """
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate hash-based embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [hash_based_embedding(text, self.dim) for text in texts]


def get_embedding_backend() -> EmbeddingBackend:
    """
    Get the appropriate embedding backend based on available resources.

    Priority:
    1. SentenceTransformer (if available)
    2. OpenAI (if OPENAI_API_KEY set)
    3. Hash-based fallback

    Returns:
        An EmbeddingBackend instance
    """
    # Check for force hash mode (useful for testing)
    if os.environ.get("EMBEDDING_BACKEND") == "hash":
        logger.info("Using hash-based embedding backend (forced via env)")
        return HashBackend()

    # Try SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        backend = SentenceTransformerBackend()
        # Test loading
        _ = backend.model
        logger.info("Using SentenceTransformer backend")
        return backend
    except Exception as e:
        logger.warning(f"SentenceTransformer not available: {e}")

    # Try OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI  # noqa: F401
            logger.info("Using OpenAI embedding backend")
            return OpenAIBackend()
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")

    # Fall back to hash
    logger.info("Using hash-based embedding backend (fallback)")
    return HashBackend()


class EmbeddingIndex:
    """
    Local embedding index with JSON storage and cosine similarity search.

    Attributes:
        index_path: Path to the JSON index file
        backend: Embedding backend to use
        data: List of indexed documents with embeddings
    """

    def __init__(
        self,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        backend: EmbeddingBackend | None = None,
    ):
        """
        Initialize the embedding index.

        Args:
            index_path: Path to store/load the index
            backend: Embedding backend (auto-detected if None)
        """
        self.index_path = Path(index_path)
        self.backend = backend or get_embedding_backend()
        self.data: list[dict[str, Any]] = []

    def build(self, documents: list[dict[str, Any]], text_field: str = "text") -> None:
        """
        Build the embedding index from documents.

        Args:
            documents: List of documents with text to embed
            text_field: Field name containing the text to embed
        """
        logger.info(f"Building index from {len(documents)} documents")

        texts = [doc.get(text_field, "") for doc in documents]
        embeddings = self.backend.embed(texts)

        self.data = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings, strict=False)):
            self.data.append({
                "id": doc.get("id", f"doc_{i}"),
                "text": doc.get(text_field, ""),
                "metadata": {k: v for k, v in doc.items() if k not in ["id", text_field]},
                "embedding": embedding,
            })

        logger.info(f"Index built with {len(self.data)} entries")

    def save(self) -> None:
        """Save the index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)

        logger.info(f"Index saved to {self.index_path}")

    def load(self) -> None:
        """Load the index from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {self.index_path}")

        with open(self.index_path, encoding="utf-8") as f:
            self.data = json.load(f)

        logger.info(f"Index loaded with {len(self.data)} entries")

    def query(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """
        Query the index for similar documents.

        Args:
            text: Query text
            top_k: Number of results to return

        Returns:
            List of results with id, score, and snippet
        """
        if not self.data:
            logger.warning("Index is empty")
            return []

        query_embedding = np.array(self.backend.embed_single(text))

        results = []
        for item in self.data:
            doc_embedding = np.array(item["embedding"])
            score = cosine_similarity(query_embedding, doc_embedding)
            results.append({
                "id": item["id"],
                "score": score,
                "snippet": item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"],
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]


def build_index(
    input_path: str,
    output_path: str = str(DEFAULT_INDEX_PATH),
    text_field: str = "text",
) -> None:
    """
    Build an embedding index from a JSON file.

    Args:
        input_path: Path to input JSON file with documents
        output_path: Path to output index file
        text_field: Field name containing text to embed
    """
    logger.info(f"Building index from {input_path}")

    with open(input_path, encoding="utf-8") as f:
        documents = json.load(f)

    index = EmbeddingIndex(index_path=output_path)
    index.build(documents, text_field=text_field)
    index.save()


def query_index(
    query_text: str,
    index_path: str = str(DEFAULT_INDEX_PATH),
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """
    Query an existing embedding index.

    Args:
        query_text: Query string
        index_path: Path to the index file
        top_k: Number of results

    Returns:
        List of search results
    """
    index = EmbeddingIndex(index_path=index_path)
    index.load()
    return index.query(query_text, top_k=top_k)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build and query embedding index for MarketResearchAI"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build index from input documents",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query text to search for",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/documents.json",
        help="Input JSON file for building index",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help="Output path for index file",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=str(DEFAULT_INDEX_PATH),
        help="Index file path for querying",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Field name containing text in input documents",
    )

    args = parser.parse_args()

    if args.build:
        build_index(args.input, args.output, args.text_field)
        print(f"Index built and saved to {args.output}")
        return 0

    if args.query:
        results = query_index(args.query, args.index, args.top_k)
        print(json.dumps(results, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
