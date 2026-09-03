#ifndef FEATURE_EXTRACTOR_H_
#define FEATURE_EXTRACTOR_H_

#include <cstddef>
#include <cstdint>

#include "app_config.h"

class FeatureExtractor {
   public:
    FeatureExtractor();

    bool Compute(const int16_t* pcm_samples,
                 size_t num_samples,
                 float* out_features);

   private:
    void BuildMelFilterbank();
    void ComputeFrame(const int16_t* frame_samples, float* log_mel_out);
    void ComputeDct(const float* log_mel, float* mfcc_out);

    float mel_filterbank_[app_config::kNumMelBins][app_config::kFftSize / 2 + 1];
    float window_[app_config::kWindowLengthSamples];
};

#endif