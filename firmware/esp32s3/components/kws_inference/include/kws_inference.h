// TensorFlow Lite Micro (LiteRT Micro) interpreter wrapper for the
// keyword-spotting model.
//
// Owns the tensor arena (statically allocated - see app_config.h's
// kTensorArenaBytes) and exposes a minimal Run() API so the rest of the
// firmware does not need to touch TFLM internals.
#ifndef KWS_INFERENCE_H_
#define KWS_INFERENCE_H_

#include <cstddef>

#include "app_config.h"

struct KwsResult {
    float probabilities[app_config::kNumClasses];  // order: keyword, unknown, silence
    uint32_t inference_duration_us;
};

class KwsInference {
   public:
    KwsInference();

    // Loads the embedded INT8 model and allocates tensors. Returns 0 on
    // success. Aborts with an explicit log message (does not silently
    // continue) if the embedded model is the zero-length placeholder -
    // see model/kws_model_data.cc.
    int Init();

    // Runs inference on a (kNumFrames x kNumMelBins) float feature
    // tensor, quantizing it to the model's INT8 input on the fly using
    // the model's own input scale/zero-point (read from the model at
    // Init() time, not hardcoded). Returns 0 on success.
    int Run(const float* features, KwsResult* result);

   private:
    bool initialized_ = false;
    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;
    float output_scale_ = 1.0f;
    int output_zero_point_ = 0;

    // Statically allocated tensor arena - avoids heap fragmentation and
    // makes the RAM cost visible and measurable (see
    // docs/memory_budget.md). Size MUST be tuned per-model; the default
    // in app_config.h is a starting point only.
    alignas(16) uint8_t tensor_arena_[app_config::kTensorArenaBytes];
};

#endif  // KWS_INFERENCE_H_
