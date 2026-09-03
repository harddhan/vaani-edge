# Troubleshooting & Known Limitations

## Firmware integration

### KWS inference is not hardware-validated

`kws_inference.cpp` contains the structure for INT8 TFLite Micro integration, but the embedded interpreter path still requires completion and testing against the selected ESP-IDF/TFLite Micro component version.

Do not describe the firmware KWS path as production-ready until `Init()` and `Run()` have been exercised on the target board.

### WebSocket client integration is incomplete

The firmware LAN transport still requires a complete WebSocket client handshake/frame implementation compatible with the Python `websockets` server.

The Python/server side is the current reference path.

### Feature extraction needs numerical parity testing

The firmware and Python feature extractors must receive identical PCM input and produce sufficiently close tensors before the Python-trained model is trusted on-device.

Compare:

1. Raw PCM input
2. Pre-emphasized samples
3. Log-Mel output
4. MFCC output
5. Final normalized `50 x 13 x 1` tensor

### FFT performance

The firmware feature extractor currently uses a direct DFT implementation. It is useful for initial correctness work but is not the preferred production implementation.

After numerical correctness is established, an optimized ESP-DSP FFT should be evaluated against the CPU budget.

### Idle CPU measurement

Idle CPU compliance is not currently measured. The runtime metrics component needs to be completed and exercised on the target board before reporting the `<10%` requirement as satisfied.

## Hardware capture

The repository currently provides a mock audio source. Real microphone integration must preserve the `audio_capture` contract:

- 16 kHz
- mono
- signed 16-bit PCM
- little-endian

See `docs/hardware_teammate_integration.md`.

## RAM

Do not infer total RAM usage from model size or tensor arena size alone.

Measure the actual minimum free heap during sustained operation, including Wi-Fi, TCP/IP, task stacks, ring buffers, feature buffers, and model runtime.

See `docs/memory_budget.md`.

## False activations

A finite test-set estimate is useful for comparing thresholds, but it is not a substitute for a long-duration real-microphone test.

For a deployment claim, log every trigger over an extended listening period and report the test duration and acoustic conditions.

## Python/server

The server supports a mock ASR backend for development. The optional local ASR backend requires its additional dependency and should be treated separately from the core KWS pipeline.

## Recommended debugging order

1. Validate the Python feature/training pipeline.
2. Validate the desktop KWS simulation.
3. Validate the Python server and protocol tests.
4. Validate firmware audio capture.
5. Compare firmware features against Python.
6. Validate on-device KWS inference.
7. Validate WebSocket transport.
8. Measure RAM, CPU, and latency.
9. Run extended false-activation tests.

The firmware directory is intentionally retained as an integration target while these steps are completed.
