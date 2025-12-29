#!/usr/bin/env python3
"""
NER Service for MarketResearchAI.

FastAPI service providing Named Entity Recognition via POST /predict endpoint.
Uses HuggingFace token-classification pipeline or heuristic fallback.

Environment Variables:
    MODEL_PATH: Path to local HF model (optional, uses default if not set)
    LOG_LEVEL: Logging level (default: INFO)

Run:
    uvicorn ner_service.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ner_service")

# FastAPI app
app = FastAPI(
    title="MarketResearchAI NER Service",
    description="Named Entity Recognition service with HF model or heuristic fallback",
    version="0.1.0",
)

# Global NER pipeline (lazy loaded)
_ner_pipeline = None
_use_heuristic = False


class PredictRequest(BaseModel):
    """Request model for /predict endpoint."""

    texts: list[str] = Field(..., min_items=1, max_items=100, description="List of texts to analyze")
    top_k: int = Field(default=10, ge=1, le=100, description="Maximum entities per text")

    @validator("texts", each_item=True)
    def validate_text_length(cls, v):
        """Validate individual text length."""
        if len(v) > 10000:
            raise ValueError("Text exceeds maximum length of 10000 characters")
        return v


class Entity(BaseModel):
    """Entity model."""

    text: str
    label: str
    start: int
    end: int
    score: float = 1.0


class PredictResponse(BaseModel):
    """Response model for /predict endpoint."""

    entities: list[list[dict[str, Any]]]


def heuristic_ner(text: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Heuristic NER using capitalized phrase extraction.

    Args:
        text: Input text
        top_k: Maximum entities to return

    Returns:
        List of entity dictionaries
    """
    entities = []

    # Match capitalized words/phrases
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
    matches = list(re.finditer(pattern, text))

    # Skip common words
    skip_words = {"The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who", "Why", "How", "And", "But", "For", "With"}

    for match in matches[:top_k * 2]:  # Get extra to filter
        entity_text = match.group(1)
        if entity_text not in skip_words:
            entities.append({
                "text": entity_text,
                "label": "ENTITY",
                "start": match.start(),
                "end": match.end(),
                "score": 0.5,  # Low confidence for heuristic
            })

    return entities[:top_k]


def load_ner_pipeline():
    """Load HuggingFace NER pipeline or set heuristic fallback."""
    global _ner_pipeline, _use_heuristic

    model_path = os.environ.get("MODEL_PATH", "dslim/bert-base-NER")

    try:
        from transformers import pipeline
        logger.info(f"Loading NER model from: {model_path}")
        _ner_pipeline = pipeline(
            "token-classification",
            model=model_path,
            aggregation_strategy="simple",
        )
        logger.info("NER model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load HF model, using heuristic fallback: {e}")
        _use_heuristic = True


def run_ner(texts: list[str], top_k: int = 10) -> list[list[dict[str, Any]]]:
    """
    Run NER on a list of texts.

    Args:
        texts: List of input texts
        top_k: Maximum entities per text

    Returns:
        List of entity lists
    """
    global _ner_pipeline, _use_heuristic

    if _use_heuristic:
        return [heuristic_ner(text, top_k) for text in texts]

    if _ner_pipeline is None:
        load_ner_pipeline()
        if _use_heuristic:
            return [heuristic_ner(text, top_k) for text in texts]

    results = []
    for text in texts:
        try:
            entities = _ner_pipeline(text)
            formatted = [
                {
                    "text": e["word"],
                    "label": e["entity_group"],
                    "start": e["start"],
                    "end": e["end"],
                    "score": float(e["score"]),
                }
                for e in entities[:top_k]
            ]
            results.append(formatted)
        except Exception as e:
            logger.error(f"NER failed for text, using heuristic: {e}")
            results.append(heuristic_ner(text, top_k))

    return results


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("NER Service starting up...")
    load_ner_pipeline()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "backend": "heuristic" if _use_heuristic else "transformers"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Extract named entities from texts.

    Args:
        request: PredictRequest with texts and top_k

    Returns:
        PredictResponse with entities for each text
    """
    logger.info(f"Processing {len(request.texts)} texts")

    try:
        entities = run_ner(request.texts, request.top_k)
        return PredictResponse(entities=entities)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "MarketResearchAI NER Service",
        "version": "0.1.0",
        "endpoints": ["/predict", "/health"],
    }
