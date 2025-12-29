"""
Tests for inference pipeline.

Uses monkeypatching to return deterministic results for offline testing.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

# Force offline backends
os.environ["EMBEDDING_BACKEND"] = "hash"
os.environ["NER_BACKEND"] = "heuristic"

from infer_pipeline import (
    aggregate_entities,
    generate_grounded_summary,
    heuristic_ner,
    run_pipeline,
)


@pytest.fixture
def mock_hits():
    """Mock search hits for testing."""
    return [
        {"id": "doc1", "score": 0.95, "snippet": "Apple Inc is a technology company..."},
        {"id": "doc2", "score": 0.85, "snippet": "Microsoft develops cloud services..."},
        {"id": "doc3", "score": 0.75, "snippet": "Google specializes in AI and search..."},
    ]


@pytest.fixture
def mock_entities():
    """Mock entity lists for testing."""
    return [
        [
            {"text": "Apple", "label": "ORG", "start": 0, "end": 5},
            {"text": "California", "label": "LOC", "start": 20, "end": 30},
        ],
        [
            {"text": "Microsoft", "label": "ORG", "start": 0, "end": 9},
        ],
        [
            {"text": "Google", "label": "ORG", "start": 0, "end": 6},
        ],
    ]


@pytest.fixture
def temp_index(tmp_path):
    """Create a temporary embedding index."""
    index_path = tmp_path / "embeddings.json"

    # Create minimal index
    data = [
        {
            "id": "doc1",
            "text": "Apple Inc is a technology company based in California.",
            "embedding": [0.1] * 384,
            "metadata": {}
        },
        {
            "id": "doc2",
            "text": "Microsoft develops cloud services and software.",
            "embedding": [0.2] * 384,
            "metadata": {}
        },
    ]

    with open(index_path, "w") as f:
        json.dump(data, f)

    return str(index_path)


class TestHeuristicNER:
    """Tests for heuristic NER function."""

    def test_extracts_entities(self):
        """Test basic entity extraction."""
        texts = ["Apple Inc is in California."]
        results = heuristic_ner(texts)

        assert len(results) == 1
        entity_texts = [e["text"] for e in results[0]]
        # Heuristic NER may extract multi-word phrases like "Apple Inc"
        assert any("Apple" in e for e in entity_texts)

    def test_handles_empty_text(self):
        """Test handling of empty text."""
        texts = [""]
        results = heuristic_ner(texts)

        assert len(results) == 1
        assert results[0] == []


class TestAggregateEntities:
    """Tests for entity aggregation."""

    def test_aggregates_by_type(self, mock_entities):
        """Test that entities are grouped by type."""
        result = aggregate_entities(mock_entities)

        assert "by_type" in result
        assert "ORG" in result["by_type"]
        assert "LOC" in result["by_type"]

    def test_counts_entities(self, mock_entities):
        """Test that entities are counted correctly."""
        result = aggregate_entities(mock_entities)

        assert "top_entities" in result
        assert all("count" in e for e in result["top_entities"])

    def test_limits_top_entities(self):
        """Test that top_entities is limited to 10."""
        # Create many entities
        entities = [[{"text": f"Entity{i}", "label": "ORG"} for i in range(20)]]
        result = aggregate_entities(entities)

        assert len(result["top_entities"]) <= 10


class TestGenerateGroundedSummary:
    """Tests for summary generation."""

    def test_generates_summary(self, mock_hits):
        """Test that summary is generated."""
        entities = {"top_entities": [{"text": "Apple", "label": "ORG"}], "by_type": {"ORG": ["Apple"]}}
        summary = generate_grounded_summary("tech companies", mock_hits, entities)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_mentions_hits(self, mock_hits):
        """Test that summary mentions number of results."""
        entities = {"top_entities": [], "by_type": {}}
        summary = generate_grounded_summary("query", mock_hits, entities)

        assert "3" in summary or "three" in summary.lower()

    def test_summary_handles_no_hits(self):
        """Test summary with no hits."""
        entities = {"top_entities": [], "by_type": {}}
        summary = generate_grounded_summary("query", [], entities)

        assert "no" in summary.lower() or "0" in summary


class TestRunPipeline:
    """Tests for the full pipeline."""

    def test_pipeline_output_structure(self, temp_index):
        """Test that pipeline returns correct structure."""
        result = run_pipeline(
            query="technology companies",
            top_k=2,
            index_path=temp_index,
        )

        assert "query" in result
        assert "hits" in result
        assert "entities" in result
        assert "summary" in result

    def test_pipeline_query_preserved(self, temp_index):
        """Test that query is preserved in output."""
        query = "cloud services"
        result = run_pipeline(query=query, top_k=2, index_path=temp_index)

        assert result["query"] == query

    def test_pipeline_hits_are_list(self, temp_index):
        """Test that hits is a list."""
        result = run_pipeline(query="test", top_k=2, index_path=temp_index)

        assert isinstance(result["hits"], list)

    def test_pipeline_entities_has_required_keys(self, temp_index):
        """Test that entities dict has required keys."""
        result = run_pipeline(query="test", top_k=2, index_path=temp_index)

        assert "top_entities" in result["entities"]
        assert "by_type" in result["entities"]

    def test_pipeline_with_monkeypatched_query(self, temp_index, mock_hits):
        """Test pipeline with monkeypatched embed_index.query."""
        with patch("embed_index.EmbeddingIndex") as MockIndex:
            # Setup mock
            mock_instance = MockIndex.return_value
            mock_instance.load.return_value = None
            mock_instance.query.return_value = mock_hits

            result = run_pipeline(
                query="technology",
                top_k=3,
                index_path=temp_index,
            )

            # Verify structure
            assert result["query"] == "technology"
            assert len(result["hits"]) == 3
            assert "entities" in result
            assert "summary" in result

    def test_pipeline_handles_missing_index(self, tmp_path):
        """Test that pipeline handles missing index gracefully."""
        nonexistent = str(tmp_path / "nonexistent.json")

        result = run_pipeline(query="test", top_k=2, index_path=nonexistent)

        # Should return empty hits, not crash
        assert result["hits"] == []
        assert "summary" in result


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    def test_end_to_end_with_hash_backend(self, temp_index):
        """Test full pipeline with hash backend."""
        result = run_pipeline(
            query="Apple technology California",
            top_k=2,
            index_path=temp_index,
        )

        # Verify all fields present
        assert all(key in result for key in ["query", "hits", "entities", "summary"])

        # Verify types
        assert isinstance(result["hits"], list)
        assert isinstance(result["entities"], dict)
        assert isinstance(result["summary"], str)
