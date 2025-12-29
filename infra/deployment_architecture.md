# Deployment Architecture

## Overview

This document describes the production deployment architecture for MarketResearchAI, including infrastructure components, model management, observability, and operational procedures.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Kubernetes Cluster                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   Ingress       │───▶│  Inference API  │───▶│   NER Service   │             │
│  │   Controller    │    │   (Deployment)  │    │   (Deployment)  │             │
│  └─────────────────┘    └────────┬────────┘    └────────┬────────┘             │
│                                  │                      │                       │
│                                  ▼                      ▼                       │
│                         ┌─────────────────┐    ┌─────────────────┐             │
│                         │  Embeddings     │    │  Model Storage  │             │
│                         │  Index (PVC)    │    │   (PVC/S3)      │             │
│                         └─────────────────┘    └─────────────────┘             │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐           │
│  │                        CronJobs                                  │           │
│  │  ┌───────────────────┐    ┌───────────────────┐                 │           │
│  │  │  Embedder         │    │  Model Sync       │                 │           │
│  │  │  (Daily rebuild)  │    │  (S3 → PVC)       │                 │           │
│  │  └───────────────────┘    └───────────────────┘                 │           │
│  └─────────────────────────────────────────────────────────────────┘           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          External Services                                       │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐             │
│  │   S3 / GCS      │    │   Prometheus    │    │   Grafana       │             │
│  │ (Model Storage) │    │  (Metrics)      │    │  (Dashboards)   │             │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Inference API
- **Purpose**: Main entry point for inference requests
- **Helm Chart**: `infra/helm/inference-api`
- **Scaling**: Horizontal Pod Autoscaler (HPA) based on CPU/memory
- **Health Checks**: `/health` endpoint for liveness/readiness

### 2. NER Service
- **Purpose**: Named Entity Recognition microservice
- **Helm Chart**: Deploy using `ner_service/Dockerfile`
- **GPU Support**: See `infra/gpu/ner-deployment-gpu.yaml`

### 3. Embedder CronJob
- **Purpose**: Rebuild embedding index on schedule
- **Helm Chart**: `infra/helm/embedder`
- **Schedule**: Configurable (default: daily at 2 AM)

## Model Registry

### Storage Options

| Option | Pros | Cons |
|--------|------|------|
| S3/GCS | Versioned, durable, scalable | Network latency |
| Local PVC | Fast access, simple | Limited to single cluster |
| HuggingFace Hub | Easy sharing, versioning | Internet dependency |

### Recommended Flow

```
Training → S3 Upload → scripts/push_model_to_s3.py
                           ↓
                    Model Sync Job
                           ↓
                   Local PVC Mount → Services
```

### Model Versioning

1. Use semantic versioning: `model-v1.0.0`
2. Store metadata in `model_config.json`
3. Keep last 3 versions for rollback

## Secrets Management

### Required Secrets

| Secret Key | Description | Used By |
|------------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (optional fallback) | Embedder, Inference API |
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 | Model sync jobs |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 | Model sync jobs |

### Configuration

```yaml
# Example Kubernetes Secret
apiVersion: v1
kind: Secret
metadata:
  name: marketresearch-secrets
type: Opaque
stringData:
  openai-api-key: ${OPENAI_API_KEY}  # Set via CI/CD
  aws-access-key-id: ${AWS_ACCESS_KEY_ID}
  aws-secret-access-key: ${AWS_SECRET_ACCESS_KEY}
```

**IMPORTANT**: Never commit actual secrets to the repository. Use:
- Kubernetes Secrets (encrypted at rest)
- AWS Secrets Manager / HashiCorp Vault
- CI/CD secret injection (GitHub Secrets, GitLab CI Variables)

## Observability

### Metrics (Prometheus)

Recommended metrics to expose:

| Metric | Type | Description |
|--------|------|-------------|
| `inference_requests_total` | Counter | Total inference requests |
| `inference_latency_seconds` | Histogram | Request latency |
| `ner_entities_extracted` | Counter | Entities extracted |
| `embedding_index_size` | Gauge | Number of indexed documents |

### Logging

- **Format**: JSON structured logs
- **Fields**: timestamp, service, level, message, request_id
- **Aggregation**: Fluentd/Fluent Bit → Elasticsearch/Loki

### Alerting

```yaml
# Example Prometheus Alert
groups:
  - name: marketresearch
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, inference_latency_seconds) > 2
        for: 5m
        labels:
          severity: warning
```

## Deployment Procedures

### Initial Deployment

```bash
# 1. Create namespace
kubectl create namespace marketresearch

# 2. Deploy secrets (pre-configured in CI/CD)
kubectl apply -f secrets.yaml -n marketresearch

# 3. Deploy embedder
helm upgrade --install embedder infra/helm/embedder -n marketresearch

# 4. Deploy inference API
helm upgrade --install inference-api infra/helm/inference-api -n marketresearch
```

### Model Updates

```bash
# 1. Upload new model to S3
python scripts/push_model_to_s3.py --model-path models/ner-v1.1.0 --bucket my-bucket

# 2. Trigger redeploy (rolling update)
./scripts/redeploy_after_model_upload.sh marketresearch
```

### Rollback

```bash
# Helm rollback
helm rollback inference-api 1 -n marketresearch

# Or revert model version
kubectl set env deployment/ner-service MODEL_PATH=/models/ner-v1.0.0 -n marketresearch
```

## Resource Sizing

### CPU-Only (Development/Staging)

| Component | CPU Request | Memory Request |
|-----------|-------------|----------------|
| Inference API | 500m | 1Gi |
| NER Service | 1 | 2Gi |
| Embedder Job | 1 | 2Gi |

### GPU (Production)

| Component | GPU | CPU | Memory |
|-----------|-----|-----|--------|
| NER Service | 1x nvidia.com/gpu | 4 | 16Gi |
| Training Job | 1-4x nvidia.com/gpu | 8 | 32Gi |

## Limitations

1. **No GPU in CI**: Tests use heuristic/hash fallbacks
2. **Model Size**: Large models require pre-download or model registry
3. **Cold Start**: First request may be slow due to model loading
4. **Single Cluster**: Multi-region requires additional configuration

## Security Considerations

1. **Network Policies**: Restrict pod-to-pod communication
2. **RBAC**: Minimal service account permissions
3. **Image Scanning**: Scan container images for vulnerabilities
4. **TLS**: Enable TLS for all ingress traffic

## Disaster Recovery

1. **Model Backups**: S3 versioning enabled
2. **Index Backups**: Daily PVC snapshots
3. **Configuration**: GitOps with version-controlled Helm values
4. **RTO/RPO**: Target 4h RTO, 1h RPO for model data
