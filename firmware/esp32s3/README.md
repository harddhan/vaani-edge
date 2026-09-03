# ESP32-S3 Firmware

ESP-IDF project implementing the edge side of the pipeline: continuous
audio capture -> ring buffer -> feature extraction -> INT8 TFLite Micro
KWS inference -> trigger state machine -> LAN streaming.

## Status / honesty notes

- **Not yet built or flashed on real hardware as part of generating this
  repository.** The code is written to compile against ESP-IDF v5.x and
  a TensorFlow Lite Micro / LiteRT Micro component, but it has not been
  verified on a physical ESP32-S3 board here. Build and flash it
  yourself (`scripts/build_firmware.sh`) and report any compile issues.
- The KWS model embedded in `components/kws_inference/model/` is a
  **placeholder** (empty/zero-length) until you run
  `ml/quantization/generate_model_cc.py` with a real trained model.
- `components/audio_capture` ships a **mock** implementation
  (`mock_audio_source.cpp`) that generates silence/synthetic tones. The
  hardware teammate replaces this with the real I2S microphone driver -
  see `docs/hardware_teammate_integration.md`.
- Memory (RAM/Flash) and CPU numbers in the docs are **placeholders**
  ("to be measured") until measured on real hardware via
  `runtime_metrics`.

## Prerequisites

- ESP-IDF v5.x installed and `. $IDF_PATH/export.sh` sourced (see
  https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/).
- A TensorFlow Lite Micro / LiteRT Micro ESP-IDF component. Add it via
  `idf_component.yml` (see `main/idf_component.yml`) - the Espressif
  component registry hosts `espressif/esp-tflite-micro`.

## Build

```bash
cd firmware/esp32s3
idf.py set-target esp32s3
idf.py build
```

Or from the repo root: `./scripts/build_firmware.sh`

## Flash

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

## Components

| Component | Responsibility |
|---|---|
| `audio_capture` | `audio_read()` interface + mock source; hardware teammate replaces with real I2S mic driver |
| `ring_buffer` | Fixed-capacity PCM ring buffer with pre-roll retrieval |
| `feature_extractor` | On-device log-Mel feature extraction (fixed-point-friendly) |
| `kws_inference` | TFLite Micro interpreter wrapper + embedded INT8 model |
| `trigger_state_machine` | Consecutive-window smoothing, cooldown, LISTENING/KEYWORD_DETECTED/STREAMING/SESSION_COMPLETE/ERROR_RECOVERY |
| `lan_transport` | Wi-Fi + WebSocket-over-TCP client implementing the binary protocol |
| `runtime_metrics` | Heap/stack/timing measurement and logging |

## Task architecture

```
audio_capture_task  (highest priority, never blocks on network)
kws_task             (feature extraction + inference)
streaming_task       (network I/O, only active while STREAMING)
metrics_task         (low priority, periodic logging)
```

See `docs/state_machine.md` and `docs/architecture.md` for details.
