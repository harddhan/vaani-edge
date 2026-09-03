# Problem Requirements Mapping

## Scope

This document maps the project requirements to the current repository implementation and distinguishes completed development work from hardware validation that is still pending.

| # | Requirement | Repository area | Current status |
|---:|---|---|---|
| 1 | Ultra-lightweight custom KWS | `ml/models/` | DS-CNN architecture trained and evaluated |
| 2 | Local execution on a low-power edge device | `firmware/esp32s3/components/kws_inference/` | Firmware integration pending |
| 3 | Custom keyword detection | `configs/model.yaml`, dataset, training pipeline | Current keyword class is `vaani`; model trained on current development dataset |
| 4 | No dependency on pretrained global wake words | `ml/training/`, `ml/models/` | No Alexa/Google-style wake-word dependency |
| 5 | Stream audio to remote ASR after detection | `trigger_state_machine` -> `lan_transport` -> `server` -> `server/asr` | Desktop/server path implemented; hardware path pending |
| 6 | Low latency and reduced data transmission | Triggered streaming + bounded binary protocol | Implemented in software path; hardware latency pending |
| 7 | Open-source implementation | ESP-IDF, TensorFlow/TFLite Micro, NumPy, SciPy, WebSockets/faster-whisper option | Yes |
| 8 | <256 KB RAM | `docs/memory_budget.md` | Hardware measurement pending |
| 9 | <10% idle CPU | `runtime_metrics` | Hardware measurement pending |
| 10 | High true-positive rate | `ml/training/evaluate.py` | Current held-out test accuracy is 98.73%; broader robustness still needs testing |
| 11 | Near-zero false activations | `ml/training/evaluate.py`, threshold tooling | Requires extended real-audio soak testing |
| 12 | ESP32/Raspberry-Pi-class edge evaluation | `firmware/esp32s3/` | ESP32-S3 target exists; physical validation pending |
| 13 | No heavy pretrained transformer on-device | `ml/models/`, INT8 conversion | Current KWS model is lightweight and INT8; ASR remains remote |

## Current ML evidence

The current DS-CNN test result is 98.73% accuracy on a 315-sample held-out split. The current `vaani` test subset is 15/15 correct.

The INT8 TFLite model also reports 98.73% on the same test set with 100% prediction agreement with the Keras model.

These figures should be described as development-dataset results, not as hardware or real-world performance guarantees.

## Hardware validation still required

Before claiming full compliance, validate:

- Real microphone capture
- Feature numerical parity
- On-device INT8 inference
- RAM usage
- Idle CPU
- End-to-end latency
- Extended false-activation rate
