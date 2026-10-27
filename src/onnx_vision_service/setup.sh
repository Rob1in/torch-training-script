#!/usr/bin/env bash
set -exuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SUDO=sudo
if ! command -v $SUDO &>/dev/null; then
    echo "no sudo on this system, proceeding as current user"
    SUDO=""
fi

# python3-venv is only needed on Debian/Ubuntu images that ship python3
# without the venv module. apt failures (stale security mirrors on old images)
# must not abort the build when venv already works.
if ! python3 -m venv --help >/dev/null 2>&1 && command -v apt-get &>/dev/null; then
    $SUDO apt-get update -qq || true
    $SUDO apt-get install -y python3-venv || echo "apt-get install python3-venv failed; continuing"
fi

VENV_DIR="$REPO_ROOT/.venv-vision-service"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source "$VENV_DIR/Scripts/activate"
else
    source "$VENV_DIR/bin/activate"
fi

pip install --prefer-binary ".[vision-service]"

touch "$SCRIPT_DIR/.setup"
