#!/bin/bash -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

VERSIONED=false
if [[ "$1" == "--versioned" ]]; then
    VERSIONED=true
    VERSION=$(cat "$ROOT_DIR/VERSION")
fi

docker push bscwdc/dislib:latest
docker push bscwdc/dislib:torch
if $VERSIONED; then
    docker tag bscwdc/dislib:latest bscwdc/dislib:v${VERSION}
    docker tag bscwdc/dislib:torch bscwdc/dislib:v${VERSION}-torch
    docker push bscwdc/dislib:v${VERSION}
    docker push bscwdc/dislib:v${VERSION}-torch
fi
