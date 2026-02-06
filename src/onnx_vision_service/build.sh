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

# Create/reuse a dedicated venv for the vision service build
VENV_DIR=".venv-vision-service"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install vision-service dependencies from pyproject.toml
echo "Installing vision-service dependencies..."
pip install -q ".[vision-service]"

# Build single executable with PyInstaller
python3 -m PyInstaller \
    --onefile \
    --hidden-import="googleapiclient" \
    --name onnx-vision-service \
    src/onnx_vision_service/main.py

# Create tarball
tar -czvf dist/onnx-vision-service.tar.gz -C dist onnx-vision-service

# Clean up PyInstaller artifacts
rm -rf build onnx-vision-service.spec

echo ""
echo "=== Build complete ==="
echo "  Executable: dist/onnx-vision-service"
echo "  Tarball:    dist/onnx-vision-service.tar.gz"
