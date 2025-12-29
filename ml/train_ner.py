#!/usr/bin/env python3
"""
NER Model Training Script for MarketResearchAI.

Skeleton for training a custom NER model using HuggingFace Transformers Trainer.
Implements tokenization, label alignment, and model training workflow.

Usage:
    python ml/train_ner.py --data ml/data/ner_sample.jsonl --output models/custom-ner

Requirements:
    - GPU recommended for training (CPU will be very slow)
    - datasets, transformers, torch packages
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_ner")


def load_jsonl_data(filepath: str) -> list[dict[str, Any]]:
    """
    Load NER training data from JSONL file.

    Expected format per line:
    {"tokens": ["word1", "word2", ...], "labels": ["O", "B-PER", ...]}

    Args:
        filepath: Path to JSONL file

    Returns:
        List of training examples
    """
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} examples from {filepath}")
    return data


def get_label_list(data: list[dict[str, Any]]) -> list[str]:
    """
    Extract unique labels from training data.

    Args:
        data: Training data with 'labels' field

    Returns:
        Sorted list of unique labels
    """
    labels = set()
    for example in data:
        labels.update(example.get("labels", []))
    return sorted(labels)


def tokenize_and_align_labels(
    examples: dict[str, list],
    tokenizer,
    label_to_id: dict[str, int],
    max_length: int = 128,
) -> dict[str, list]:
    """
    Tokenize examples and align labels with tokenized output.

    Handles subword tokenization by propagating labels to first subword.

    Args:
        examples: Batch of examples with 'tokens' and 'labels'
        tokenizer: HuggingFace tokenizer
        label_to_id: Mapping from label strings to IDs
        max_length: Maximum sequence length

    Returns:
        Tokenized batch with aligned labels
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
    )

    labels_batch = []
    for i, labels in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word
                label_ids.append(label_to_id.get(labels[word_idx], 0))
            else:
                # Subsequent tokens of same word: use -100 or same label
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels_batch.append(label_ids)

    tokenized_inputs["labels"] = labels_batch
    return tokenized_inputs


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Compute evaluation metrics for NER.

    Args:
        eval_pred: Tuple of (predictions, labels)

    Returns:
        Dictionary of metrics
    """
    import numpy as np

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Flatten and filter -100 labels
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels, strict=False):
        for pred, label in zip(pred_seq, label_seq, strict=False):
            if label != -100:
                true_labels.append(label)
                pred_labels.append(pred)

    # Calculate accuracy
    correct = sum(1 for t, p in zip(true_labels, pred_labels, strict=False) if t == p)
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0.0

    return {"accuracy": accuracy}


def train_ner_model(
    data_path: str,
    output_dir: str,
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
) -> None:
    """
    Train a NER model using HuggingFace Transformers.

    Args:
        data_path: Path to JSONL training data
        output_dir: Directory to save trained model
        model_name: Base model name or path
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        max_length: Maximum sequence length
    """
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error(f"Required packages not installed: {e}")
        logger.error("Install: pip install transformers datasets torch")
        sys.exit(1)

    # Load data
    raw_data = load_jsonl_data(data_path)
    label_list = get_label_list(raw_data)
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    logger.info(f"Labels: {label_list}")
    logger.info(f"Number of labels: {len(label_list)}")

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # Create dataset
    dataset = Dataset.from_list(raw_data)

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(
            {"tokens": [x["tokens"]], "labels": [x["labels"]]},
            tokenizer,
            label_to_id,
            max_length,
        ),
        remove_columns=dataset.column_names,
    )

    # Split train/eval
    split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_map_path = Path(output_dir) / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)

    logger.info("Training complete!")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train NER model using HuggingFace Transformers"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to JSONL training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/custom-ner",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("BASE_MODEL", "bert-base-uncased"),
        help="Base model name or path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    train_ner_model(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
