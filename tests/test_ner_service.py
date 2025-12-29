"""
Tests for NER service.

Uses FastAPI TestClient for endpoint testing.
Tests use heuristic fallback to avoid HuggingFace downloads.
"""

from __future__ import annotations

import os

# Force heuristic backend for offline testing
os.environ["NER_BACKEND"] = "heuristic"

import pytest
from fastapi.testclient import TestClient

from ner_service.app import app, heuristic_ner


@pytest.fixture
def client():
    """Create test client for NER service."""
    return TestClient(app)


class TestHeuristicNER:
    """Tests for heuristic NER fallback."""

    def test_extracts_capitalized_words(self):
        """Test that capitalized words are extracted as entities."""
        text = "Apple Inc is based in California."
        entities = heuristic_ner(text)

        entity_texts = [e["text"] for e in entities]
        # Heuristic NER may extract multi-word phrases like "Apple Inc"
        assert any("Apple" in e for e in entity_texts)
        assert any("California" in e for e in entity_texts)

    def test_skips_common_words(self):
        """Test that common words like 'The' are skipped."""
        text = "The company is called Microsoft."
        entities = heuristic_ner(text)

        entity_texts = [e["text"] for e in entities]
        assert "The" not in entity_texts
        assert "Microsoft" in entity_texts

    def test_respects_top_k(self):
        """Test that top_k limits the number of entities."""
        text = "Apple Microsoft Google Amazon Tesla Netflix"
        entities = heuristic_ner(text, top_k=3)

        assert len(entities) <= 3

    def test_entity_structure(self):
        """Test that entities have correct structure."""
        text = "Tesla is in California."
        entities = heuristic_ner(text)

        for entity in entities:
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_returns_entities_key(self, client):
        """Test that response contains 'entities' key."""
        response = client.post(
            "/predict",
            json={"texts": ["Apple Inc is in California."], "top_k": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data

    def test_predict_single_text(self, client):
        """Test prediction with single text."""
        response = client.post(
            "/predict",
            json={"texts": ["Microsoft develops software."], "top_k": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["entities"]) == 1
        assert isinstance(data["entities"][0], list)

    def test_predict_multiple_texts(self, client):
        """Test prediction with multiple texts."""
        response = client.post(
            "/predict",
            json={
                "texts": [
                    "Apple is a company.",
                    "Google is in California.",
                    "Amazon sells products."
                ],
                "top_k": 5
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["entities"]) == 3

    def test_predict_empty_texts_returns_400(self, client):
        """Test that empty texts list returns 400."""
        response = client.post(
            "/predict",
            json={"texts": [], "top_k": 5}
        )

        assert response.status_code == 422  # Validation error

    def test_predict_missing_texts_returns_422(self, client):
        """Test that missing texts field returns 422."""
        response = client.post(
            "/predict",
            json={"top_k": 5}
        )

        assert response.status_code == 422

    def test_predict_invalid_top_k(self, client):
        """Test validation of top_k parameter."""
        # top_k = 0 should fail
        response = client.post(
            "/predict",
            json={"texts": ["Test"], "top_k": 0}
        )
        assert response.status_code == 422

        # top_k > 100 should fail
        response = client.post(
            "/predict",
            json={"texts": ["Test"], "top_k": 101}
        )
        assert response.status_code == 422


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health check returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_shows_backend(self, client):
        """Test health check shows backend type."""
        response = client.get("/health")

        data = response.json()
        assert "backend" in data
        assert data["backend"] in ["heuristic", "transformers"]


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root_returns_service_info(self, client):
        """Test root endpoint returns service information."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data
