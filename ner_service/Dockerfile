# NER Service Dockerfile
# Production-grade container for NER inference

FROM python:3.11-slim

# Labels
LABEL maintainer="MarketResearchAI Team"
LABEL version="0.1.0"
LABEL description="Named Entity Recognition service"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
# NOTE: For GPU support, replace torch with torch+cu118 or similar
# pip install torch --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Create non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
# MODEL_PATH can be overridden to point to a local model
# ENV MODEL_PATH=/models/ner-model

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# GPU NOTES:
# ---------
# For GPU inference, use the nvidia/cuda base image or install CUDA:
#
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
# ... install python, etc ...
#
# Then in your Kubernetes deployment, request GPU resources:
# resources:
#   limits:
#     nvidia.com/gpu: 1
#
# See infra/gpu/ner-deployment-gpu.yaml for example.
