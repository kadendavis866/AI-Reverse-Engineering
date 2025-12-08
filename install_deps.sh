#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies listed in boundary/requirements.txt into the current environment.
# Usage: ./install_deps.sh

PYTHON="${PYTHON:-python3}"

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r boundary/requirements.txt
