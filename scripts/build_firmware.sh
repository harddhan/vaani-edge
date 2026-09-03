#!/usr/bin/env bash
# Builds the ESP32-S3 firmware. Requires ESP-IDF v5.x installed and
# sourced (`. $IDF_PATH/export.sh`).
set -euo pipefail
cd "$(dirname "$0")/../firmware/esp32s3"

if ! command -v idf.py >/dev/null 2>&1; then
  echo "idf.py not found. Install ESP-IDF v5.x and run: . \$IDF_PATH/export.sh"
  exit 2
fi

idf.py set-target esp32s3
idf.py build
