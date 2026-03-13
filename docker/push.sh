#!/bin/bash -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

VERSIONED=false
if [[ "$1" == "--versioned" ]]; then
    VERSIONED=true
    VERSION=$(cat "$ROOT_DIR/VERSION")
fi

docker login -u "${dh_username}" -p "${dh_password}"
docker push bscwdc/dislib:latest
docker push bscwdc/dislib:torch
if $VERSIONED; then
    docker push bscwdc/dislib:${VERSION}
    docker push bscwdc/dislib:${VERSION}-torch
fi
