#!/usr/bin/env bash
# Starts the Python audio/ASR server. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m server.main "$@"
