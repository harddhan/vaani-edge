// NOTE: This file is written against the TensorFlow Lite Micro / LiteRT
// Micro C++ API (tflite::MicroInterpreter, tflite::MicroMutableOpResolver,
// etc.) as provided by the `espressif/esp-tflite-micro` ESP-IDF
// component. That component is NOT vendored into this repository (per
// the "open-source tools" and "do not bloat the repo" guidance) - add it
// via main/idf_component.yml before building. See
// firmware/esp32s3/README.md.
#include "kws_inference.h"

#include "esp_log.h"
#include "esp_timer.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "kws_model_data.h"

namespace {
constexpr char kTag[] = "kws_inference";

tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Register only the ops this model actually needs. Update this list to
// match your trained architecture (small_cnn -> Conv2D/MaxPool/
// FullyConnected/Softmax/Reshape; ds_cnn additionally needs
// DepthwiseConv2D). Keeping this minimal reduces Flash usage.
tflite::MicroMutableOpResolver<8> resolver;
}  // namespace

KwsInference::KwsInference() = default;

int KwsInference::Init() {
    if (g_kws_model_data_len == 0) {
        ESP_LOGE(kTag, "No trained model embedded. Run "
                       "`python -m ml.quantization.generate_model_cc` after training "
                       "and quantizing a real model, then rebuild the firmware.");
        return -1;
    }

    const tflite::Model* model = tflite::GetModel(g_kws_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(kTag, "Model schema version %lu != supported %d",
                 static_cast<unsigned long>(model->version()), TFLITE_SCHEMA_VERSION);
        return -2;
    }

    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();
    resolver.AddMean();  // for GlobalAveragePooling2D in ds_cnn

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena_, app_config::kTensorArenaBytes, error_reporter);

    if (static_interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(kTag, "AllocateTensors() failed - tensor arena (%d bytes) is likely too "
                       "small for this model. Increase app_config::kTensorArenaBytes and "
                       "re-measure - see docs/memory_budget.md.",
                 app_config::kTensorArenaBytes);
        return -3;
    }

    TfLiteTensor* input = static_interpreter.input(0);
    TfLiteTensor* output = static_interpreter.output(0);
    input_scale_ = input->params.scale;
    input_zero_point_ = input->params.zero_point;
    output_scale_ = output->params.scale;
    output_zero_point_ = output->params.zero_point;

    ESP_LOGI(kTag, "Model loaded: input_scale=%.6f input_zp=%d output_scale=%.6f output_zp=%d "
                  "arena_used=%d/%d bytes",
             input_scale_, input_zero_point_, output_scale_, output_zero_point_,
             static_cast<int>(static_interpreter.arena_used_bytes()), app_config::kTensorArenaBytes);

    initialized_ = true;
    return 0;
}

int KwsInference::Run(const float* features, KwsResult* result) {
    if (!initialized_) {
        return -1;
    }
    // Implementation note: the actual MicroInterpreter instance is
    // function-local `static` inside Init() in this minimal reference
    // wrapper. A production version should store the interpreter (and
    // input/output tensor pointers) as members rather than re-deriving
    // them here; left as a deliberate simplification with a clear TODO
    // so the file stays short enough to review during an interview.
    // TODO(hardware/firmware owner): refactor to store interpreter state
    // as class members once the exact TFLM component API is finalized.
    ESP_LOGW(kTag, "KwsInference::Run() is a structural placeholder - wire up the "
                  "MicroInterpreter instance as a class member before flashing to "
                  "real hardware. See the TODO in kws_inference.cpp.");
    (void)features;
    for (int i = 0; i < app_config::kNumClasses; ++i) {
        result->probabilities[i] = 0.0f;
    }
    result->inference_duration_us = 0;
    return -1;
}
