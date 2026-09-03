# Architecture

## Overview

VAANI is an edge-first voice activation pipeline. Audio is continuously captured locally and evaluated by a lightweight keyword-spotting model. Only after the custom keyword is detected does the system open a LAN session and stream the following audio to the server for ASR.

```mermaid
flowchart LR
    Mic["Microphone"] --> Cap["audio_capture\n16 kHz mono PCM"]
    Cap --> Ring["ring_buffer\npre-roll retention"]
    Ring --> Feat["feature_extractor\nMFCC"]
    Feat --> KWS["kws_inference\nINT8 TFLite Micro"]
    KWS --> Trig["trigger_state_machine"]
    Trig -- keyword detected --> Stream["lan_transport\nWebSocket/TCP"]
    Stream --> Server["Python server"]
    Server --> ASR["ASR backend"]
    ASR --> Server
    Server -- ASR_RESULT --> Stream
```

## Component responsibilities

| Layer | Component | Responsibility | Status |
|---|---|---|---|
| Firmware | `audio_capture` | Provide mono int16 PCM through `audio_read()` | Hardware integration pending |
| Firmware | `ring_buffer` | Maintain rolling audio and pre-roll | Implemented |
| Firmware | `feature_extractor` | Generate model input features | Implementation pending numerical parity validation |
| Firmware | `kws_inference` | Run embedded INT8 model | Integration pending |
| Firmware | `trigger_state_machine` | Threshold, consecutive positives, cooldown | Implemented |
| Firmware | `lan_transport` | Wi-Fi and WebSocket transport | Integration pending |
| Firmware | `runtime_metrics` | Runtime measurement instrumentation | Partial; hardware validation pending |
| Server | `server/main.py` | Session lifecycle, validation, ASR orchestration | Implemented |
| Server | `server/asr/*` | Pluggable ASR backend | Implemented |
| ML | `ml/*` | Dataset preparation, training, evaluation, quantization | Implemented |

## Data flow

1. `audio_capture_task` continuously fills the ring buffer.
2. `kws_task` evaluates the latest approximately one-second audio window.
3. The feature extractor produces the fixed `50 x 13 x 1` MFCC tensor used by the current model.
4. The INT8 KWS model produces four class scores: `speech`, `noise`, `silence`, and `vaani`.
5. `TriggerStateMachine` requires the configured confidence threshold for consecutive windows.
6. After a trigger, the streaming path sends a session start, retained pre-roll, live audio, and a session end.
7. The Python server reconstructs the received PCM and invokes the configured ASR backend.
8. The server returns an `ASR_RESULT` or `ERROR`.
9. The firmware returns to listening after the session/cooldown path completes.

## Design principles

- KWS runs locally; continuous raw audio is not sent to the server.
- The network path is separated from audio capture.
- Audio buffering is bounded.
- The model is quantized to INT8 before firmware embedding.
- The Python pipeline is the reference for model training.
- Hardware resource claims are made only after measurement on the target board.

## Current implementation boundary

The desktop simulation and Python server provide the current end-to-end development path. The ESP32-S3 firmware directory is retained as the hardware integration target, but it is not presented as fully validated firmware.

The main remaining hardware-side work is real microphone capture, TFLite Micro inference integration, WebSocket interoperability, feature numerical parity, and RAM/CPU/latency measurement.
