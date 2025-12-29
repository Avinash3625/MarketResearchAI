"""
Tests for embed_index module.

These tests are designed to run OFFLINE without HuggingFace downloads.
Uses hash-based deterministic embeddings for reproducibility.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

# Force hash backend for offline testing
os.environ["EMBEDDING_BACKEND"] = "hash"

from embed_index import (
    EmbeddingIndex,
    HashBackend,
    build_index,
    cosine_similarity,
    hash_based_embedding,
    query_index,
)


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {"id": "doc1", "text": "Apple is a technology company based in California"},
        {"id": "doc2", "text": "Microsoft develops software and cloud services"},
        {"id": "doc3", "text": "Tesla manufactures electric vehicles and solar panels"},
        {"id": "doc4", "text": "Amazon is an e-commerce and cloud computing giant"},
        {"id": "doc5", "text": "Google specializes in internet services and AI research"},
    ]


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestHashBasedEmbedding:
    """Tests for hash-based embedding fallback."""

    def test_deterministic_output(self):
        """Same input should produce same embedding."""
        text = "Hello, world!"
        emb1 = hash_based_embedding(text)
        emb2 = hash_based_embedding(text)
        assert emb1 == emb2

    def test_correct_dimension(self):
        """Embedding should have correct dimension."""
        text = "Test text"
        emb = hash_based_embedding(text, dim=384)
        assert len(emb) == 384

    def test_different_texts_different_embeddings(self):
        """Different texts should produce different embeddings."""
        emb1 = hash_based_embedding("Hello")
        emb2 = hash_based_embedding("World")
        assert emb1 != emb2


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1."""
        import numpy as np
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0."""
        import numpy as np
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1."""
        import numpy as np
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert abs(cosine_similarity(v1, v2) + 1.0) < 1e-6


class TestHashBackend:
    """Tests for HashBackend class."""

    def test_embed_single(self):
        """Test embedding a single text."""
        backend = HashBackend(dim=128)
        embedding = backend.embed_single("test")
        assert len(embedding) == 128

    def test_embed_batch(self):
        """Test embedding multiple texts."""
        backend = HashBackend(dim=128)
        texts = ["text1", "text2", "text3"]
        embeddings = backend.embed(texts)
        assert len(embeddings) == 3
        assert all(len(e) == 128 for e in embeddings)


class TestEmbeddingIndex:
    """Tests for EmbeddingIndex class."""

    def test_build_index(self, sample_documents, temp_dir):
        """Test building an index from documents."""
        index_path = temp_dir / "embeddings.json"

        index = EmbeddingIndex(index_path=index_path, backend=HashBackend())
        index.build(sample_documents)
        index.save()

        # Verify file was created
        assert index_path.exists()

        # Verify content
        with open(index_path) as f:
            data = json.load(f)
        assert len(data) == 5
        assert all("embedding" in item for item in data)
        assert all("id" in item for item in data)

    def test_load_index(self, sample_documents, temp_dir):
        """Test loading an existing index."""
        index_path = temp_dir / "embeddings.json"

        # Build and save
        index1 = EmbeddingIndex(index_path=index_path, backend=HashBackend())
        index1.build(sample_documents)
        index1.save()

        # Load in new instance
        index2 = EmbeddingIndex(index_path=index_path, backend=HashBackend())
        index2.load()

        assert len(index2.data) == 5

    def test_query_returns_correct_shape(self, sample_documents, temp_dir):
        """Test that query returns correctly shaped results."""
        index_path = temp_dir / "embeddings.json"

        index = EmbeddingIndex(index_path=index_path, backend=HashBackend())
        index.build(sample_documents)

        results = index.query("technology company", top_k=3)

        assert len(results) == 3
        assert all("id" in r for r in results)
        assert all("score" in r for r in results)
        assert all("snippet" in r for r in results)

    def test_query_scores_are_sorted(self, sample_documents, temp_dir):
        """Test that results are sorted by score descending."""
        index_path = temp_dir / "embeddings.json"

        index = EmbeddingIndex(index_path=index_path, backend=HashBackend())
        index.build(sample_documents)

        results = index.query("cloud services", top_k=5)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestBuildAndQueryIndex:
    """Integration tests for CLI-like usage."""

    def test_build_and_query(self, sample_documents, temp_dir):
        """Test full build and query workflow."""
        input_path = temp_dir / "documents.json"
        index_path = temp_dir / "embeddings.json"

        # Write input documents
        with open(input_path, "w") as f:
            json.dump(sample_documents, f)

        # Build index
        build_index(str(input_path), str(index_path))

        # Verify index file exists
        assert index_path.exists()

        # Query index
        results = query_index("electric vehicles", str(index_path), top_k=2)

        assert len(results) == 2
        assert all(isinstance(r["score"], float) for r in results)

    def test_query_nonexistent_index_raises(self, temp_dir):
        """Test that querying nonexistent index raises error."""
        with pytest.raises(FileNotFoundError):
            query_index("test", str(temp_dir / "nonexistent.json"))
