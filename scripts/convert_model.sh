#!/usr/bin/env bash
# Quantizes a trained model to INT8, verifies it, and generates the C
# array for firmware embedding.
# Usage: ./scripts/convert_model.sh [architecture]  (default: small_cnn)
set -euo pipefail
cd "$(dirname "$0")/.."

ARCHITECTURE="${1:-small_cnn}"
MODEL_PATH="ml/artifacts/${ARCHITECTURE}_best.keras"

python3 -m ml.quantization.convert_int8 --model "$MODEL_PATH"
python3 -m ml.quantization.evaluate_quantized \
    --float32-model "ml/artifacts/${ARCHITECTURE}_best_float32.tflite" \
    --int8-model "ml/artifacts/${ARCHITECTURE}_best_int8.tflite"
python3 -m ml.quantization.quantization_report --name "${ARCHITECTURE}_best"
python3 -m ml.quantization.generate_model_cc --int8-model "ml/artifacts/${ARCHITECTURE}_best_int8.tflite"
