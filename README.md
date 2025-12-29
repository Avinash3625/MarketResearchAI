# MarketResearchAI

Production-grade embeddings-based retrieval, NER, aggregation, and RAG-ready summarization system.

[![CI](https://github.com/Avinash3625/MarketResearchAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash3625/MarketResearchAI/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

MarketResearchAI provides a complete pipeline for:
- **Embeddings-based retrieval** using sentence-transformers (with OpenAI and hash fallbacks)
- **Named Entity Recognition (NER)** via HuggingFace models or heuristic fallback
- **Entity aggregation** and statistics
- **RAG-ready grounded summarization**

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MarketResearchAI                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│  │   Query     │───▶│  Embedding  │───▶│  Retrieval  │               │
│  │   Input     │    │   Index     │    │   (Top-K)   │               │
│  └─────────────┘    └─────────────┘    └──────┬──────┘               │
│                                               │                       │
│                                               ▼                       │
│  ┌─────────────────────────────────────────────────────┐             │
│  │                    NER Service                       │             │
│  │   HuggingFace Model ──or── Heuristic Fallback       │             │
│  └───────────────────────────┬─────────────────────────┘             │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               │
│  │  Aggregate  │───▶│  Summarize  │───▶│   Output    │               │
│  │  Entities   │    │  (Grounded) │    │   (JSON)    │               │
│  └─────────────┘    └─────────────┘    └─────────────┘               │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.11+
- pip or conda

### Installation

```bash
# Clone repository
git clone https://github.com/Avinash3625/MarketResearchAI.git
cd MarketResearchAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests (offline-safe)
pytest -q

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_embed_index.py
```

### Lint Check

```bash
ruff check .
ruff format --check .
```

## Usage

### Build Embedding Index

```bash
# Build from sample documents
python embed_index.py --build --input data/documents.json

# With custom output path
python embed_index.py --build --input data/documents.json --output data/my_embeddings.json
```

### Query the Index

```bash
# Query for similar documents
python embed_index.py --query "technology companies" --top_k 5
```

### Run NER Service

```bash
# Start the NER service
cd ner_service
uvicorn app:app --host 0.0.0.0 --port 8000

# Or from project root
uvicorn ner_service.app:app --host 0.0.0.0 --port 8000
```

### Run Inference Pipeline

```bash
# Run full pipeline
python infer_pipeline.py --query "What are the major tech companies?" --top_k 5

# With external NER service
python infer_pipeline.py --query "Electric vehicles" --ner_service_url http://localhost:8000

# Save output to file
python infer_pipeline.py --query "Cloud computing" --output results.json
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SENTENCE_TRANSFORMER_MODEL` | Sentence transformer model name | `all-MiniLM-L6-v2` |
| `EMBEDDING_BACKEND` | Force embedding backend (`hash` for testing) | Auto-detect |
| `NER_BACKEND` | Force NER backend (`heuristic` for testing) | Auto-detect |
| `OPENAI_API_KEY` | OpenAI API key for fallback embeddings | None |
| `MODEL_PATH` | Path to local NER model | `dslim/bert-base-NER` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Embedding Backend Priority

1. **SentenceTransformers** (preferred) - Uses local model
2. **OpenAI** (fallback) - If `OPENAI_API_KEY` is set
3. **Hash-based** (offline) - Deterministic fallback for testing

## Docker

### Build Images

```bash
# Build embedder image
docker build -f docker/Dockerfile.embed -t marketresearch/embedder:latest .

# Build NER service image
docker build -f ner_service/Dockerfile -t marketresearch/ner-service:latest ner_service/
```

### Run Containers

```bash
# Run NER service
docker run -p 8000:8000 marketresearch/ner-service:latest

# Run embedder job
docker run -v $(pwd)/data:/data marketresearch/embedder:latest --build
```

## Kubernetes Deployment

### Using Helm

```bash
# Deploy embedder CronJob
helm upgrade --install embedder infra/helm/embedder -n marketresearch

# Deploy inference API
helm upgrade --install inference-api infra/helm/inference-api -n marketresearch
```

### GPU Deployment

See `infra/gpu/ner-deployment-gpu.yaml` for GPU-enabled deployment example.

## CI/CD

GitHub Actions workflows included:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR | Lint + Tests |
| `build_and_push_images.yml` | Tags | Build Docker images |
| `deploy_model.yml` | Manual | Deploy model to S3 |
| `smoke_test_deploy.yml` | Manual | Test deployed services |

### Required Secrets

Set these in GitHub repository settings:

- `AWS_ACCESS_KEY_ID` - AWS credentials for S3
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `MODEL_BUCKET` - S3 bucket for models
- `KUBECONFIG` - Base64-encoded kubeconfig (for cluster deploy)

## Project Structure

```
MarketResearchAI/
├── embed_index.py          # Embedding index builder/query
├── infer_pipeline.py       # Full inference pipeline
├── ner_service/            # NER FastAPI service
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── ml/                     # ML training scripts
│   ├── train_ner.py
│   ├── data/
│   └── labeling_strategy.md
├── docker/                 # Docker configurations
├── infra/                  # Infrastructure configs
│   ├── helm/
│   │   ├── embedder/
│   │   └── inference-api/
│   ├── gpu/
│   └── deployment_architecture.md
├── scripts/                # Utility scripts
├── tests/                  # Unit tests
├── run_reports/            # Sample outputs
├── .github/workflows/      # CI/CD workflows
├── .devcontainer/          # Codespaces config
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project configuration
└── README.md
```

## Acceptance Criteria

- [x] Embeddings-based retrieval with multi-backend support
- [x] NER with HuggingFace model and heuristic fallback
- [x] Entity aggregation and grounded summarization
- [x] Dockerfiles for all services
- [x] Helm charts for Kubernetes
- [x] GitHub Actions CI/CD
- [x] DevContainer for Codespaces
- [x] Offline-safe unit tests
- [x] No credentials in repository

## Limitations

1. **CI Environment**: No GPU, limited network - tests use fallbacks
2. **Training**: Full HF model training requires GPU and longer runtime
3. **Cold Start**: First request may be slow due to model loading
4. **RAG Summary**: Uses template-based summary; integrate LLM for production

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run `pytest` and `ruff check .`
5. Submit a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file.
