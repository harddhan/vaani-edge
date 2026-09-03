# Interview Defense: Anticipated Questions

## Architecture

**Q: Why perform KWS locally and ASR remotely?**

A: The device only needs a tiny local model to decide when the user is addressing the system. Keeping KWS local avoids continuously transmitting raw audio and reduces unnecessary server work. Only the triggered utterance is sent for ASR.

**Q: Why not stream all audio continuously?**

A: Continuous streaming increases bandwidth, server workload, and exposure of ambient audio. The project specifically targets an edge-first activation path where transmission begins after local keyword detection.

**Q: Why WebSocket over TCP?**

A: The current protocol needs reliable ordered delivery of short audio sessions. WebSocket provides message framing over TCP and is straightforward to integrate with the Python server. UDP is intentionally outside the core implementation.

## Model

**Q: Why MFCC?**

A: The current model uses MFCCs as a compact representation derived from a 40-band log-Mel spectrogram. The DCT reduces the representation to 13 coefficients per frame, producing a `50 x 13 x 1` model input.

**Q: Why DS-CNN?**

A: Depthwise-separable convolutions reduce parameter and computation requirements compared with standard convolutions, making DS-CNN a suitable architecture to evaluate for a constrained edge device.

**Q: How do you know INT8 quantization preserved accuracy?**

A: The quantized evaluation script runs the Keras and INT8 TFLite models on the same held-out test set. The current results are 98.73% accuracy for both models with 100% prediction agreement.

**Q: How do you reduce false activations?**

A: The system uses a dedicated non-keyword class structure, a configurable probability threshold, consecutive-positive-window smoothing, and a cooldown period. The threshold can be evaluated with the threshold sweep tooling.

## Systems

**Q: What happens if the network fails?**

A: The streaming path uses bounded connection/session behavior and returns to an error-recovery path rather than waiting indefinitely.

**Q: How is audio capture protected from network work?**

A: Audio capture and network streaming are separated into different firmware tasks. Capture writes to the local buffer, while streaming owns network I/O after a trigger.

**Q: Do you already meet the <256 KB RAM and <10% CPU requirements?**

A: Not yet claimed. Those values require measurement on the target hardware. The repository includes runtime instrumentation and documents the measurement procedure.

## Current limitations

The ESP32-S3 integration still requires:

- Real microphone capture
- Python/firmware feature numerical parity validation
- TFLite Micro inference wiring
- WebSocket interoperability
- Optimized FFT implementation
- Hardware RAM/CPU measurements
- Hardware latency measurements
- Extended real-microphone false-activation testing
