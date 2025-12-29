#!/bin/bash
# Redeploy services after model upload
# Triggers rolling restart of deployments to pick up new models
#
# Usage:
#   ./scripts/redeploy_after_model_upload.sh <namespace>
#   ./scripts/redeploy_after_model_upload.sh marketresearch
#
# Prerequisites:
#   - kubectl configured with cluster access
#   - Appropriate RBAC permissions

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <namespace> [deployment1] [deployment2] ..."
    echo ""
    echo "Examples:"
    echo "  $0 marketresearch                    # Restart all deployments"
    echo "  $0 marketresearch ner-service       # Restart specific deployment"
    echo ""
    exit 1
fi

NAMESPACE="$1"
shift

# Verify kubectl is available
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl not found. Please install kubectl."
    exit 1
fi

# Verify namespace exists
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    log_error "Namespace '$NAMESPACE' not found"
    exit 1
fi

log_info "Starting redeploy in namespace: $NAMESPACE"

# Get deployments to restart
if [ $# -gt 0 ]; then
    # Specific deployments provided
    DEPLOYMENTS=("$@")
else
    # Get all deployments in namespace
    DEPLOYMENTS=$(kubectl get deployments -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')
fi

# Restart each deployment
for DEPLOYMENT in $DEPLOYMENTS; do
    log_info "Restarting deployment: $DEPLOYMENT"
    
    if kubectl rollout restart deployment/"$DEPLOYMENT" -n "$NAMESPACE"; then
        log_info "Successfully triggered restart for $DEPLOYMENT"
    else
        log_warn "Failed to restart $DEPLOYMENT (may not exist)"
    fi
done

# Wait for rollouts to complete
log_info "Waiting for rollouts to complete..."

for DEPLOYMENT in $DEPLOYMENTS; do
    if kubectl get deployment "$DEPLOYMENT" -n "$NAMESPACE" &> /dev/null; then
        log_info "Waiting for $DEPLOYMENT..."
        if kubectl rollout status deployment/"$DEPLOYMENT" -n "$NAMESPACE" --timeout=300s; then
            log_info "$DEPLOYMENT rolled out successfully"
        else
            log_error "$DEPLOYMENT rollout failed or timed out"
        fi
    fi
done

log_info "Redeploy complete!"

# Show current status
echo ""
log_info "Current deployment status:"
kubectl get deployments -n "$NAMESPACE" -o wide
