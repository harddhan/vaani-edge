#include "runtime_metrics.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace {
constexpr char kTag[] = "runtime_metrics";
}

RuntimeMetrics& RuntimeMetrics::Instance() {
    static RuntimeMetrics instance;
    return instance;
}

void RuntimeMetrics::SetStaticSizes(uint32_t tensor_arena_bytes, uint32_t model_size_bytes,
                                      uint32_t ring_buffer_bytes, uint32_t feature_buffer_bytes,
                                      uint32_t network_buffer_bytes) {
    tensor_arena_bytes_ = tensor_arena_bytes;
    model_size_bytes_ = model_size_bytes;
    ring_buffer_bytes_ = ring_buffer_bytes;
    feature_buffer_bytes_ = feature_buffer_bytes;
    network_buffer_bytes_ = network_buffer_bytes;
}

void RuntimeMetrics::RecordTensorArenaUsed(uint32_t bytes_used) { tensor_arena_used_bytes_ = bytes_used; }
void RuntimeMetrics::RecordKwsInferenceDuration(uint32_t duration_us) {
    last_kws_inference_duration_us_ = duration_us;
}
void RuntimeMetrics::RecordFeatureExtractionDuration(uint32_t duration_us) {
    last_feature_extraction_duration_us_ = duration_us;
}
void RuntimeMetrics::RecordLoopPeriod(uint32_t duration_us) { last_loop_period_us_ = duration_us; }

float RuntimeMetrics::EstimateIdleCpuPercent() {
#if configGENERATE_RUN_TIME_STATS
    // Requires CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS=y in sdkconfig.
    // Left as a documented extension point: computing this accurately
    // needs vTaskGetRunTimeStats() and identifying the IDLE task's
    // counter, which is verbose boilerplate not central to the KWS
    // logic. See docs/memory_budget.md / docs/troubleshooting.md for the
    // exact steps to wire this up and measure idle CPU on real hardware.
    return -1.0f;  // TODO: implement via vTaskGetRunTimeStats()
#else
    return -1.0f;  // Not enabled; enable CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS to measure.
#endif
}

RuntimeMetricsSnapshot RuntimeMetrics::Snapshot() {
    RuntimeMetricsSnapshot snapshot{};
    snapshot.free_heap_bytes = static_cast<uint32_t>(esp_get_free_heap_size());
    snapshot.min_free_heap_bytes = static_cast<uint32_t>(esp_get_minimum_free_heap_size());
    snapshot.tensor_arena_bytes = tensor_arena_bytes_;
    snapshot.tensor_arena_used_bytes = tensor_arena_used_bytes_;
    snapshot.model_size_bytes = model_size_bytes_;
    snapshot.ring_buffer_bytes = ring_buffer_bytes_;
    snapshot.feature_buffer_bytes = feature_buffer_bytes_;
    snapshot.network_buffer_bytes = network_buffer_bytes_;
    snapshot.last_kws_inference_duration_us = last_kws_inference_duration_us_;
    snapshot.last_feature_extraction_duration_us = last_feature_extraction_duration_us_;
    snapshot.last_loop_period_us = last_loop_period_us_;
    snapshot.estimated_idle_cpu_percent = EstimateIdleCpuPercent();
    return snapshot;
}

void RuntimeMetrics::LogSnapshot() {
    RuntimeMetricsSnapshot s = Snapshot();
    ESP_LOGI(kTag,
             "free_heap=%u min_free_heap=%u arena=%u/%u model=%u ring=%u feat=%u net=%u "
             "kws_us=%u feat_us=%u loop_us=%u idle_cpu_pct=%.1f",
             s.free_heap_bytes, s.min_free_heap_bytes, s.tensor_arena_used_bytes, s.tensor_arena_bytes,
             s.model_size_bytes, s.ring_buffer_bytes, s.feature_buffer_bytes, s.network_buffer_bytes,
             s.last_kws_inference_duration_us, s.last_feature_extraction_duration_us, s.last_loop_period_us,
             s.estimated_idle_cpu_percent);
}
