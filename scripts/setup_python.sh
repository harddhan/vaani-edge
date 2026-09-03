#!/usr/bin/env bash
# Creates a virtualenv and installs Python dependencies.
# Run from the repository root: ./scripts/setup_python.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
