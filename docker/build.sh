#!/bin/bash -e

# Always run from the repo root so COPY . in the Dockerfiles works correctly
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

VERSIONED=false
if [[ "$1" == "--versioned" ]]; then
    VERSIONED=true
    VERSION=$(cat VERSION)
fi

BASE_TAGS="--tag bscwdc/dislib:latest"
TORCH_TAGS="--tag bscwdc/dislib:torch"
if $VERSIONED; then
    BASE_TAGS="$BASE_TAGS --tag bscwdc/dislib:${VERSION}"
    TORCH_TAGS="$TORCH_TAGS --tag bscwdc/dislib:${VERSION}-torch"
fi

echo "Building base image..."
docker build $BASE_TAGS -f docker/Dockerfile.base .

echo "Building torch image..."
docker build $TORCH_TAGS -f docker/Dockerfile.torch .

echo "Building CI image (bscwdc/dislib:ci)..."
docker build --tag bscwdc/dislib:ci -f docker/Dockerfile.ci .

echo "Done."
