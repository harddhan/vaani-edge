#!/usr/bin/env bash
# Runs the desktop end-to-end simulation against a WAV file.
# Usage: ./scripts/run_desktop_demo.sh path/to/clip.wav [--model ml/artifacts/small_cnn_best.keras]
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
  echo "Usage: $0 <wav_file> [extra args passed to simulate_pipeline.py]"
  exit 2
fi

WAV_FILE="$1"
shift
python3 -m desktop.simulate_pipeline --wav "$WAV_FILE" "$@"
