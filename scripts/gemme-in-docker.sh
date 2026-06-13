#!/usr/bin/env bash
# Run an arbitrary command inside a "GEMME runtime" container so we don't
# have to install Python 2.7 + R + Java + JET2 on the host.
#
# The container is built from `scripts/Dockerfile.gemme`, which combines
# the upstream `elodielaine/gemme:gemme` image (GEMME source + JET2 source
# + R + Java + python2.7) with a Python 3.12 base so we can run the
# evedesign Python 3 code that drives the wrapper. The build is cached
# (image tag `evedesign-gemme:latest`) so it only happens once.
#
# This script bind-mounts the repo root at /work, exports GEMME_PATH /
# JET_PATH so the wrapper picks them up, makes the local source importable
# via PYTHONPATH=/work/src, and then runs whatever you pass as arguments.
# Examples:
#
#     scripts/gemme-in-docker.sh pytest tests/test_gemme_conformance.py -v
#     scripts/gemme-in-docker.sh python examples/gemme_protein_scoring/test_gemme.py
#     scripts/gemme-in-docker.sh jupyter nbconvert --to html --execute \
#         examples/gemme_protein_scoring/gemme_protein_scoring.ipynb
#
# Without arguments, drops you into a bash shell.

set -euo pipefail

IMAGE_TAG="${EVEDESIGN_GEMME_IMAGE:-evedesign-gemme:latest}"
PLATFORM="${GEMME_DOCKER_PLATFORM:-linux/amd64}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"

# Build the image if it does not already exist locally. The build is fairly
# heavy (~2 GB once cached) but only runs once.
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
    echo "Building $IMAGE_TAG (one-time, ~5-10 min) ..."
    docker build \
        --platform "$PLATFORM" \
        -f "$REPO_ROOT/scripts/Dockerfile.gemme" \
        -t "$IMAGE_TAG" \
        "$REPO_ROOT/scripts"
fi

if [[ $# -eq 0 ]]; then
    USER_CMD=(bash)
else
    USER_CMD=("$@")
fi

# Allocate -it only when both stdin and stdout are TTYs (interactive use);
# otherwise (CI, piped commands, etc.) plain `--rm` is enough.
TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_FLAGS+=(-it)
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
    --platform "$PLATFORM" \
    -v "$REPO_ROOT":/work \
    -w /work \
    -e GEMME_PATH=/opt/GEMME/ \
    -e JET_PATH=/opt/JET2/ \
    -e PYTHONPATH=/work \
    "$IMAGE_TAG" \
    "${USER_CMD[@]}"
