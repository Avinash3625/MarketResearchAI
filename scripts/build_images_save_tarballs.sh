#!/bin/bash
# Build Docker images and save as tarballs for offline deployment
#
# Usage:
#   ./scripts/build_images_save_tarballs.sh
#   ./scripts/build_images_save_tarballs.sh --tag v0.1.0
#
# Output:
#   - docker/embedder-<tag>.tar
#   - docker/ner-service-<tag>.tar

set -euo pipefail

# Configuration
TAG="${1:-latest}"
OUTPUT_DIR="docker"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

log_info "Building images with tag: $TAG"

# Build embedder image
log_info "Building embedder image..."
docker build \
    -f docker/Dockerfile.embed \
    -t marketresearch/embedder:"$TAG" \
    .

# Build NER service image
log_info "Building NER service image..."
docker build \
    -f ner_service/Dockerfile \
    -t marketresearch/ner-service:"$TAG" \
    ner_service/

# Save as tarballs
log_info "Saving embedder image to tarball..."
docker save marketresearch/embedder:"$TAG" | gzip > "$OUTPUT_DIR/embedder-$TAG.tar.gz"

log_info "Saving NER service image to tarball..."
docker save marketresearch/ner-service:"$TAG" | gzip > "$OUTPUT_DIR/ner-service-$TAG.tar.gz"

# Show output
log_info "Build complete! Tarballs saved to:"
ls -lh "$OUTPUT_DIR"/*.tar.gz

echo ""
log_info "To load on another machine:"
echo "  gunzip -c $OUTPUT_DIR/embedder-$TAG.tar.gz | docker load"
echo "  gunzip -c $OUTPUT_DIR/ner-service-$TAG.tar.gz | docker load"
