// Mock audio source: generates synthetic PCM (silence + a low-amplitude
// tone) so the rest of the pipeline can be built, flashed, and smoke
// tested without a physical microphone connected.
//
// HARDWARE TEAMMATE: replace this file's implementation of
// audio_capture_init()/audio_read()/audio_capture_deinit() with the real
// I2S microphone driver. Do not change audio_capture.h's function
// signatures unless you also update every caller (feature_extractor via
// kws_task, and any test code) - see docs/hardware_teammate_integration.md.
#include "audio_capture.h"

#include <cmath>

#include "app_config.h"

namespace {
constexpr float kToneFrequencyHz = 440.0f;
constexpr float kToneAmplitude = 500.0f;  // small amplitude int16 tone
uint32_t g_sample_index = 0;
bool g_initialized = false;
}  // namespace

int audio_capture_init() {
    g_sample_index = 0;
    g_initialized = true;
    return 0;
}

int audio_read(int16_t* buffer, size_t samples) {
    if (!g_initialized || buffer == nullptr) {
        return -1;
    }
    // Generates mostly-silence with a faint tone every ~5 seconds so a
    // logic analyzer / serial log can distinguish "capture is running"
    // from "capture is stuck", without ever falsely triggering the KWS
    // model (a pure tone should score far below the keyword class for
    // any reasonably trained model).
    const bool tone_active = (g_sample_index / app_config::kSampleRateHz) % 5 == 0;
    for (size_t i = 0; i < samples; ++i) {
        float t = static_cast<float>(g_sample_index) / app_config::kSampleRateHz;
        float value = tone_active ? kToneAmplitude * sinf(2.0f * static_cast<float>(M_PI) * kToneFrequencyHz * t) : 0.0f;
        buffer[i] = static_cast<int16_t>(value);
        ++g_sample_index;
    }
    return static_cast<int>(samples);
}

void audio_capture_deinit() {
    g_initialized = false;
}
