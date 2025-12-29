#!/usr/bin/env python3
"""
Push trained model to S3 bucket.

Uses AWS CLI's `aws s3 sync` for efficient upload with incremental updates.

Usage:
    python scripts/push_model_to_s3.py --model-path models/ner-v1.0.0
    python scripts/push_model_to_s3.py --model-path models/ner-v1.0.0 --bucket my-bucket

Environment Variables:
    MODEL_BUCKET: Default S3 bucket name
    AWS_ACCESS_KEY_ID: AWS credentials
    AWS_SECRET_ACCESS_KEY: AWS credentials
    AWS_DEFAULT_REGION: AWS region (default: us-east-1)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("push_model_to_s3")


def validate_aws_credentials() -> bool:
    """
    Validate that AWS credentials are available.

    Returns:
        True if credentials are configured
    """
    # Check environment variables
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True

    # Check for AWS CLI configuration
    try:
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def push_model_to_s3(
    model_path: str,
    bucket: str,
    s3_prefix: str = "models",
    delete: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Push model directory to S3 using aws s3 sync.

    Args:
        model_path: Local path to model directory
        bucket: S3 bucket name
        s3_prefix: Prefix path in S3 bucket
        delete: Whether to delete files not in source
        dry_run: If True, show what would be uploaded without uploading

    Returns:
        True if successful, False otherwise
    """
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        return False

    if not model_path.is_dir():
        logger.error(f"Model path is not a directory: {model_path}")
        return False

    # Construct S3 URI
    model_name = model_path.name
    s3_uri = f"s3://{bucket}/{s3_prefix}/{model_name}/"

    logger.info(f"Syncing {model_path} to {s3_uri}")

    # Build aws s3 sync command
    cmd = ["aws", "s3", "sync", str(model_path), s3_uri]

    if delete:
        cmd.append("--delete")

    if dry_run:
        cmd.append("--dryrun")

    # Add common options
    cmd.extend([
        "--exclude", "*.pyc",
        "--exclude", "__pycache__/*",
        "--exclude", ".git/*",
    ])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large models
        )

        if result.stdout:
            logger.info(result.stdout)

        if result.returncode != 0:
            logger.error(f"S3 sync failed: {result.stderr}")
            return False

        logger.info(f"Successfully synced model to {s3_uri}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("S3 sync timed out after 1 hour")
        return False
    except FileNotFoundError:
        logger.error("AWS CLI not found. Install with: pip install awscli")
        return False


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Push trained model to S3 bucket"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model directory to upload",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("MODEL_BUCKET"),
        help="S3 bucket name (or set MODEL_BUCKET env var)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="models",
        help="S3 prefix path (default: models)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete files in S3 that don't exist locally",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )

    args = parser.parse_args()

    if not args.bucket:
        logger.error("Bucket not specified. Use --bucket or set MODEL_BUCKET env var")
        return 1

    if not validate_aws_credentials():
        logger.error("AWS credentials not configured")
        logger.error("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or run 'aws configure'")
        return 1

    success = push_model_to_s3(
        model_path=args.model_path,
        bucket=args.bucket,
        s3_prefix=args.prefix,
        delete=args.delete,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
