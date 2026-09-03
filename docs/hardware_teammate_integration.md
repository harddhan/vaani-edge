# Hardware Integration Guide

This guide defines the boundary between the software repository and the hardware implementation.

## Hardware responsibility

The hardware integration covers:

- Microphone selection and wiring
- Electrical design
- PCB and GPIO details
- I2S/PDM configuration
- Power and peripheral configuration
- Real microphone capture

The software pipeline consumes audio through the existing `audio_capture` interface.

## Audio interface contract

```cpp
int audio_capture_init();
int audio_read(int16_t *buffer, size_t samples);
void audio_capture_deinit();
```

`audio_read()` must provide exactly:

- Mono
- Signed 16-bit PCM
- Little-endian
- 16000 Hz
- Requested sample count

If the microphone natively provides stereo, 24-bit, PDM, or another format, conversion should happen inside the audio-capture implementation so the rest of the firmware sees the agreed PCM format.

## Current repository implementation

The firmware currently contains a mock audio source for development. It is not a replacement for real microphone hardware.

Replace the mock source in:

`firmware/esp32s3/components/audio_capture/src/mock_audio_source.cpp`

with the board-specific implementation while preserving the public interface.

No other firmware component should directly depend on microphone GPIO or I2S details.

## Suggested integration sequence

1. Configure the selected microphone using the appropriate ESP-IDF I2S/PDM API.
2. Implement `audio_capture_init()`.
3. Implement blocking `audio_read()` with reliable handling of partial driver reads.
4. Implement `audio_capture_deinit()`.
5. Verify 16 kHz mono int16 samples independently.
6. Verify the ring buffer receives real microphone audio.
7. Compare firmware feature tensors with the Python reference.
8. Validate KWS inference.
9. Validate LAN streaming.
10. Measure RAM, CPU, and latency on the target board.

## Current status

The hardware boundary is intentionally isolated. Real microphone integration is pending, and the firmware directory should therefore be treated as an integration target rather than a completed hardware deliverable.

Resource and latency targets must be updated only with measurements from the actual board.
