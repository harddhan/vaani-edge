#include "feature_extractor.h"

#include <cmath>
#include <cstring>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kFMinHz = 300.0f;
constexpr float kFMaxHz = 8000.0f;
constexpr float kLogFloor = 1e-10f;

float HzToMel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float MelToHz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

}

FeatureExtractor::FeatureExtractor() {
    for (int i = 0; i < app_config::kWindowLengthSamples; ++i) {
        window_[i] =
            0.5f -
            0.5f * cosf(
                2.0f * kPi * static_cast<float>(i) /
                static_cast<float>(app_config::kWindowLengthSamples - 1));
    }

    BuildMelFilterbank();
}

void FeatureExtractor::BuildMelFilterbank() {
    const int n_bins = app_config::kFftSize / 2 + 1;
    const int n_mels = app_config::kNumMelBins;

    const float mel_min = HzToMel(kFMinHz);
    const float mel_max = HzToMel(kFMaxHz);

    float mel_points[n_mels + 2];
    int bin_points[n_mels + 2];

    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] =
            mel_min +
            (mel_max - mel_min) *
                static_cast<float>(i) /
                static_cast<float>(n_mels + 1);

        const float hz = MelToHz(mel_points[i]);

        int bin = static_cast<int>(
            floorf(
                (static_cast<float>(app_config::kFftSize) + 1.0f) *
                hz /
                static_cast<float>(app_config::kSampleRateHz)));

        if (bin < 0) {
            bin = 0;
        }

        if (bin >= n_bins) {
            bin = n_bins - 1;
        }

        bin_points[i] = bin;
    }

    std::memset(mel_filterbank_, 0, sizeof(mel_filterbank_));

    for (int m = 1; m <= n_mels; ++m) {
        int left = bin_points[m - 1];
        int center = bin_points[m];
        int right = bin_points[m + 1];

        if (center <= left) {
            center = left + 1;
        }

        if (right <= center) {
            right = center + 1;
        }

        if (right >= n_bins) {
            right = n_bins - 1;
        }

        for (int k = left; k < center && k < n_bins; ++k) {
            mel_filterbank_[m - 1][k] =
                static_cast<float>(k - left) /
                static_cast<float>(center - left);
        }

        for (int k = center; k <= right && k < n_bins; ++k) {
            mel_filterbank_[m - 1][k] =
                static_cast<float>(right - k) /
                static_cast<float>(right - center);
        }
    }
}

void FeatureExtractor::ComputeFrame(const int16_t* frame_samples,
                                    float* log_mel_out) {
    const int n_fft = app_config::kFftSize;
    const int n_bins = n_fft / 2 + 1;
    const int frame_len = app_config::kWindowLengthSamples;

    float windowed[app_config::kWindowLengthSamples];

    float previous = 0.0f;

    for (int i = 0; i < frame_len; ++i) {
        const float sample =
            static_cast<float>(frame_samples[i]) / 32768.0f;

        float emphasized;

        if (i == 0) {
            emphasized = sample;
        } else {
            emphasized =
                sample - app_config::kPreEmphasis * previous;
        }

        previous = sample;
        windowed[i] = emphasized * window_[i];
    }

    float power[n_bins];

    for (int k = 0; k < n_bins; ++k) {
        float real = 0.0f;
        float imag = 0.0f;

        for (int n = 0; n < frame_len && n < n_fft; ++n) {
            const float angle =
                -2.0f * kPi *
                static_cast<float>(k * n) /
                static_cast<float>(n_fft);

            real += windowed[n] * cosf(angle);
            imag += windowed[n] * sinf(angle);
        }

        power[k] =
            (real * real + imag * imag) /
            static_cast<float>(n_fft);
    }

    for (int m = 0; m < app_config::kNumMelBins; ++m) {
        float energy = 0.0f;

        for (int k = 0; k < n_bins; ++k) {
            energy += power[k] * mel_filterbank_[m][k];
        }

        if (energy < kLogFloor) {
            energy = kLogFloor;
        }

        log_mel_out[m] = logf(energy);
    }
}

void FeatureExtractor::ComputeDct(const float* log_mel,
                                  float* mfcc_out) {
    const int n_mels = app_config::kNumMelBins;
    const int n_mfcc = app_config::kNumMfcc;

    const float scale =
        sqrtf(2.0f / static_cast<float>(n_mels));

    for (int k = 0; k < n_mfcc; ++k) {
        float sum = 0.0f;

        for (int n = 0; n < n_mels; ++n) {
            const float angle =
                kPi *
                static_cast<float>(k) *
                (static_cast<float>(n) + 0.5f) /
                static_cast<float>(n_mels);

            sum += log_mel[n] * cosf(angle);
        }

        mfcc_out[k] = sum * scale;
    }
}

bool FeatureExtractor::Compute(const int16_t* pcm_samples,
                               size_t num_samples,
                               float* out_features) {
    if (pcm_samples == nullptr || out_features == nullptr) {
        return false;
    }

    const int frame_len = app_config::kWindowLengthSamples;
    const int hop_len = app_config::kHopLengthSamples;
    const int num_frames = app_config::kNumFrames;

    const size_t required_samples =
        frame_len +
        static_cast<size_t>(num_frames - 1) * hop_len;

    if (num_samples < required_samples) {
        return false;
    }

    float log_mel[app_config::kNumMelBins];
    float mfcc[app_config::kNumMfcc];

    for (int t = 0; t < num_frames; ++t) {
        const int16_t* frame_start =
            pcm_samples + t * hop_len;

        ComputeFrame(frame_start, log_mel);
        ComputeDct(log_mel, mfcc);

        std::memcpy(
            out_features +
                t * app_config::kNumMfcc,
            mfcc,
            sizeof(mfcc));
    }

    return true;
}