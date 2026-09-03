// Runtime metrics collection: free heap, tensor arena usage, per-stage
// timing, and a coarse idle-CPU estimate. Logged periodically by
// metrics_task and available for the team to read via serial console
// while validating the RAM/CPU requirements in docs/memory_budget.md.
//
// IMPORTANT: this module MEASURES; it does not claim compliance. Reading
// these numbers on real hardware and writing them into
// docs/memory_budget.md / docs/latency_measurement.md is a required step
// before claiming the 256KB RAM / <10% idle CPU targets are met.
#ifndef RUNTIME_METRICS_H_
#define RUNTIME_METRICS_H_

#include <cstdint>

struct RuntimeMetricsSnapshot {
    uint32_t free_heap_bytes;
    uint32_t min_free_heap_bytes;
    uint32_t tensor_arena_bytes;      // configured size, see app_config.h
    uint32_t tensor_arena_used_bytes; // actual usage, set by kws_inference after Init()
    uint32_t model_size_bytes;
    uint32_t ring_buffer_bytes;
    uint32_t feature_buffer_bytes;
    uint32_t network_buffer_bytes;
    uint32_t last_kws_inference_duration_us;
    uint32_t last_feature_extraction_duration_us;
    uint32_t last_loop_period_us;
    float estimated_idle_cpu_percent;  // see EstimateIdleCpuPercent()
};

class RuntimeMetrics {
   public:
    static RuntimeMetrics& Instance();

    void SetStaticSizes(uint32_t tensor_arena_bytes, uint32_t model_size_bytes,
                          uint32_t ring_buffer_bytes, uint32_t feature_buffer_bytes,
                          uint32_t network_buffer_bytes);
    void RecordTensorArenaUsed(uint32_t bytes_used);
    void RecordKwsInferenceDuration(uint32_t duration_us);
    void RecordFeatureExtractionDuration(uint32_t duration_us);
    void RecordLoopPeriod(uint32_t duration_us);

    // Coarse idle-CPU estimate: fraction of time the idle task actually
    // ran over a measurement window, derived from FreeRTOS's per-task
    // runtime stats (requires CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS).
    // Returns -1.0f if runtime stats are not enabled in sdkconfig.
    float EstimateIdleCpuPercent();

    RuntimeMetricsSnapshot Snapshot();
    void LogSnapshot();

   private:
    RuntimeMetrics() = default;

    uint32_t tensor_arena_bytes_ = 0;
    uint32_t tensor_arena_used_bytes_ = 0;
    uint32_t model_size_bytes_ = 0;
    uint32_t ring_buffer_bytes_ = 0;
    uint32_t feature_buffer_bytes_ = 0;
    uint32_t network_buffer_bytes_ = 0;
    uint32_t last_kws_inference_duration_us_ = 0;
    uint32_t last_feature_extraction_duration_us_ = 0;
    uint32_t last_loop_period_us_ = 0;
};

#endif  // RUNTIME_METRICS_H_
