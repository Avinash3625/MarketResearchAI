#!/usr/bin/env python3
"""
Inference Pipeline for MarketResearchAI.

Implements: Retrieval → NER → Aggregation → Grounded Summary.

CLI Usage:
    python infer_pipeline.py --query "What are the trends in AI?" --top_k 5
    python infer_pipeline.py --query "Tesla stock analysis" --ner_service_url http://localhost:8000

Output: JSON with query, hits, entities, and summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from typing import Any

import requests

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("infer_pipeline")


def heuristic_ner(texts: list[str]) -> list[list[dict[str, Any]]]:
    """
    Heuristic NER fallback using capitalized phrase extraction.

    Extracts capitalized words/phrases as potential named entities.
    NOT suitable for production - use HF model or NER service instead.

    Args:
        texts: List of text strings to extract entities from

    Returns:
        List of entity lists, one per input text
    """
    results = []

    for text in texts:
        entities = []
        # Match capitalized words (potential proper nouns)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        matches = re.finditer(pattern, text)

        for match in matches:
            entity_text = match.group(1)
            # Skip common words that happen to be capitalized
            skip_words = {"The", "This", "That", "These", "Those", "What", "When", "Where", "Which", "Who", "Why", "How"}
            if entity_text not in skip_words:
                entities.append({
                    "text": entity_text,
                    "label": "ENTITY",  # Generic label for heuristic
                    "start": match.start(),
                    "end": match.end(),
                })

        results.append(entities)

    return results


def call_ner_service(texts: list[str], service_url: str, top_k: int = 10) -> list[list[dict[str, Any]]]:
    """
    Call external NER service for entity extraction.

    Args:
        texts: List of texts to process
        service_url: URL of the NER service
        top_k: Maximum entities per text

    Returns:
        List of entity lists from the service

    Raises:
        requests.RequestException: If service call fails
    """
    endpoint = f"{service_url.rstrip('/')}/predict"

    try:
        response = requests.post(
            endpoint,
            json={"texts": texts, "top_k": top_k},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("entities", [])
    except requests.RequestException as e:
        logger.error(f"NER service call failed: {e}")
        raise


def local_hf_ner(texts: list[str], model_name: str | None = None) -> list[list[dict[str, Any]]]:
    """
    Run NER using local HuggingFace pipeline.

    Args:
        texts: List of texts to process
        model_name: Optional model name (defaults to dslim/bert-base-NER)

    Returns:
        List of entity lists
    """
    try:
        from transformers import pipeline

        model = model_name or os.environ.get("NER_MODEL", "dslim/bert-base-NER")
        logger.info(f"Loading HF NER model: {model}")

        ner_pipeline = pipeline("token-classification", model=model, aggregation_strategy="simple")

        results = []
        for text in texts:
            entities = ner_pipeline(text)
            results.append([
                {
                    "text": e["word"],
                    "label": e["entity_group"],
                    "start": e["start"],
                    "end": e["end"],
                    "score": float(e["score"]),
                }
                for e in entities
            ])

        return results
    except Exception as e:
        logger.warning(f"HF NER failed, falling back to heuristic: {e}")
        return heuristic_ner(texts)


def get_ner_extractor(service_url: str | None = None, model_name: str | None = None):
    """
    Get the appropriate NER extractor function.

    Priority:
    1. External service (if service_url provided)
    2. Local HF model (if available)
    3. Heuristic fallback

    Args:
        service_url: Optional URL for NER service
        model_name: Optional HF model name

    Returns:
        Callable that extracts entities from texts
    """
    if service_url:
        logger.info(f"Using NER service at {service_url}")
        return lambda texts: call_ner_service(texts, service_url)

    # Check for force heuristic mode (for testing)
    if os.environ.get("NER_BACKEND") == "heuristic":
        logger.info("Using heuristic NER (forced via env)")
        return heuristic_ner

    # Try local HF
    try:
        from transformers import pipeline  # noqa: F401
        logger.info("Using local HuggingFace NER")
        return lambda texts: local_hf_ner(texts, model_name)
    except ImportError:
        logger.warning("transformers not available, using heuristic NER")
        return heuristic_ner


def aggregate_entities(entity_lists: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """
    Aggregate entities across multiple texts.

    Args:
        entity_lists: List of entity lists from NER

    Returns:
        Aggregated entity information with top entities and by-type breakdown
    """
    all_entities = []
    by_type: dict[str, list[str]] = {}

    for entities in entity_lists:
        for entity in entities:
            text = entity.get("text", "")
            label = entity.get("label", "UNKNOWN")

            all_entities.append((text, label))

            if label not in by_type:
                by_type[label] = []
            if text not in by_type[label]:
                by_type[label].append(text)

    # Count entity occurrences
    entity_counts = Counter(all_entities)
    top_entities = [
        {"text": text, "label": label, "count": count}
        for (text, label), count in entity_counts.most_common(10)
    ]

    return {
        "top_entities": top_entities,
        "by_type": {k: v[:10] for k, v in by_type.items()},  # Limit each type
    }


def generate_grounded_summary(
    query: str,
    hits: list[dict[str, Any]],
    entities: dict[str, Any],
) -> str:
    """
    Generate a grounded summary based on query, hits, and entities.

    This is a simple template-based summary. For production RAG,
    integrate with an LLM (e.g., GPT-4, Claude) for better summaries.

    Args:
        query: Original query
        hits: Search results
        entities: Aggregated entities

    Returns:
        Grounded summary text
    """
    # Extract key information
    top_entities = entities.get("top_entities", [])[:5]
    entity_names = [e["text"] for e in top_entities]

    num_hits = len(hits)
    top_score = hits[0]["score"] if hits else 0

    # Build summary
    parts = []

    if num_hits > 0:
        parts.append(f"Found {num_hits} relevant documents (top relevance: {top_score:.2f}).")
    else:
        parts.append("No relevant documents found.")

    if entity_names:
        parts.append(f"Key entities identified: {', '.join(entity_names)}.")

    if hits:
        parts.append(f"Top result: \"{hits[0]['snippet'][:100]}...\"")

    # Entity type breakdown
    by_type = entities.get("by_type", {})
    if by_type:
        type_summary = "; ".join([f"{k}: {len(v)} unique" for k, v in list(by_type.items())[:3]])
        parts.append(f"Entity types: {type_summary}.")

    return " ".join(parts)


def run_pipeline(
    query: str,
    top_k: int = 5,
    ner_model: str | None = None,
    ner_service_url: str | None = None,
    index_path: str = "data/embeddings.json",
) -> dict[str, Any]:
    """
    Run the full inference pipeline.

    Pipeline: Retrieval → NER → Aggregation → Summary

    Args:
        query: User query
        top_k: Number of results to retrieve
        ner_model: Optional HF NER model name
        ner_service_url: Optional NER service URL
        index_path: Path to embedding index

    Returns:
        Pipeline result with query, hits, entities, and summary
    """
    logger.info(f"Running pipeline for query: {query[:50]}...")

    # Step 1: Retrieval
    from embed_index import EmbeddingIndex

    index = EmbeddingIndex(index_path=index_path)
    try:
        index.load()
        hits = index.query(query, top_k=top_k)
    except FileNotFoundError:
        logger.warning(f"Index not found at {index_path}, returning empty results")
        hits = []

    logger.info(f"Retrieved {len(hits)} hits")

    # Step 2: NER on query and top snippets
    texts_to_analyze = [query] + [hit["snippet"] for hit in hits]

    ner_extractor = get_ner_extractor(ner_service_url, ner_model)
    entity_lists = ner_extractor(texts_to_analyze)

    logger.info(f"Extracted entities from {len(texts_to_analyze)} texts")

    # Step 3: Aggregate entities
    entities = aggregate_entities(entity_lists)

    # Step 4: Generate summary
    summary = generate_grounded_summary(query, hits, entities)

    return {
        "query": query,
        "hits": hits,
        "entities": entities,
        "summary": summary,
    }


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run MarketResearchAI inference pipeline"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query text to process",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--ner_model",
        type=str,
        help="HuggingFace NER model name",
    )
    parser.add_argument(
        "--ner_service_url",
        type=str,
        help="URL of external NER service",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="data/embeddings.json",
        help="Path to embedding index file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (stdout if not specified)",
    )

    args = parser.parse_args()

    result = run_pipeline(
        query=args.query,
        top_k=args.top_k,
        ner_model=args.ner_model,
        ner_service_url=args.ner_service_url,
        index_path=args.index,
    )

    output = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        logger.info(f"Output written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
