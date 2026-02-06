#!/bin/bash
# build.sh — Build the ONNX Vision Service module into a single executable.
#
# Usage (from the repo root):
#   bash src/onnx_vision_service/build.sh
#
# Output:
#   dist/onnx-vision-service           (executable)
#   dist/onnx-vision-service.tar.gz    (tarball for upload to Viam registry)

set -e

# Ensure we're at the repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

echo "=== Building ONNX Vision Service module ==="
echo "  Repo root: $REPO_ROOT"

# Activate virtual environment
source .venv/bin/activate

# Build single executable with PyInstaller
python3 -m PyInstaller \
    --onefile \
    --hidden-import="googleapiclient" \
    --name onnx-vision-service \
    src/onnx_vision_service/main.py

# Create tarball
tar -czvf dist/onnx-vision-service.tar.gz -C dist onnx-vision-service

echo ""
echo "=== Build complete ==="
echo "  Executable: dist/onnx-vision-service"
echo "  Tarball:    dist/onnx-vision-service.tar.gz"
