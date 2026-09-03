// Audio capture abstraction.
//
// OWNERSHIP BOUNDARY: this header defines the contract the hardware
// teammate implements for the real microphone (I2S wiring, GPIO,
// electrical/PCB details). This repository ships ONLY a mock
// implementation (`mock_audio_source.cpp`) so the rest of the firmware
// (feature extraction, KWS inference, trigger logic, streaming) can be
// developed and tested without physical hardware. See
// docs/hardware_teammate_integration.md.
//
// The rest of the firmware must depend ONLY on this interface, never on
// I2S/GPIO details directly.
#ifndef AUDIO_CAPTURE_H_
#define AUDIO_CAPTURE_H_

#include <cstddef>
#include <cstdint>

// Initializes the audio capture backend. Must be called once before
// audio_read(). Returns 0 on success, negative on failure.
//
// The REAL implementation (to be written by the hardware teammate) is
// expected to configure the I2S peripheral here, matching the format
// below. The mock implementation ignores hardware setup entirely.
int audio_capture_init();

// Reads exactly `samples` mono, signed 16-bit, little-endian PCM samples
// at 16000 Hz into `buffer`. Blocks until `samples` are available (a
// real I2S driver blocks on i2s_channel_read(); the mock generates data
// synchronously). Returns the number of samples actually written (equal
// to `samples` on success, less on error/EOF for file-backed test
// sources).
//
//     int audio_read(int16_t *buffer, size_t samples);
//
// This exact signature is the "clean interface" requested by the
// project brief so the hardware teammate can drop in a real
// implementation without touching any other firmware module.
int audio_read(int16_t* buffer, size_t samples);

// Releases any resources held by the capture backend. Optional to call
// before shutdown; mainly useful for host-side unit tests.
void audio_capture_deinit();

#endif  // AUDIO_CAPTURE_H_
