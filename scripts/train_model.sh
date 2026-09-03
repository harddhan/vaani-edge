#!/usr/bin/env bash
# Runs the full ML pipeline: dataset prep -> split -> train -> evaluate.
# Requires real recordings under data/raw/ (see data/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHITECTURE="${1:-small_cnn}"

python3 -m ml.dataset.validate_wavs --input data/raw
python3 -m ml.dataset.prepare_dataset --input data/raw --output data/processed
python3 -m ml.dataset.split_dataset --manifest data/processed/manifest.csv
python3 -m ml.models.model_summary --architecture "$ARCHITECTURE"
python3 -m ml.training.train --architecture "$ARCHITECTURE"
python3 -m ml.training.evaluate --model "ml/artifacts/${ARCHITECTURE}_best.keras"
python3 -m ml.training.threshold_sweep --model "ml/artifacts/${ARCHITECTURE}_best.keras"
