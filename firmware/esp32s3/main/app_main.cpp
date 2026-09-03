#include <cstring>

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "app_config.h"
#include "audio_capture.h"
#include "feature_extractor.h"
#include "kws_inference.h"
#include "lan_transport.h"
#include "ring_buffer.h"
#include "runtime_metrics.h"
#include "trigger_state_machine.h"

namespace {

constexpr char kTag[] = "app_main";
constexpr size_t kKeywordIndex = 3;

int16_t g_ring_buffer_storage[
    app_config::kRingBufferSamples
];

RingBuffer g_ring_buffer(
    g_ring_buffer_storage,
    app_config::kRingBufferSamples
);

float g_feature_buffer[
    app_config::kNumFrames *
    app_config::kNumMelBins
];

FeatureExtractor g_feature_extractor;
KwsInference g_kws_inference;

TriggerStateMachine g_trigger_sm(
    app_config::kDetectionThreshold,
    app_config::kConsecutivePositiveWindows,
    app_config::kCooldownMs /
        (app_config::kHopLengthSamples * 1000 /
         app_config::kSampleRateHz)
);

QueueHandle_t g_stream_request_queue;

struct StreamRequest {
    uint8_t session_id[16];
};

void WifiInit() {
    esp_err_t nvs_result = nvs_flash_init();

    if (nvs_result == ESP_ERR_NVS_NO_FREE_PAGES ||
        nvs_result == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        nvs_result = nvs_flash_init();
    }

    ESP_ERROR_CHECK(nvs_result);
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_create_default_wifi_sta());

    wifi_init_config_t cfg =
        WIFI_INIT_CONFIG_DEFAULT();

    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t wifi_config = {};

    std::strncpy(
        reinterpret_cast<char*>(
            wifi_config.sta.ssid
        ),
        app_config::kWifiSsid,
        sizeof(wifi_config.sta.ssid) - 1
    );

    std::strncpy(
        reinterpret_cast<char*>(
            wifi_config.sta.password
        ),
        app_config::kWifiPassword,
        sizeof(wifi_config.sta.password) - 1
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_mode(WIFI_MODE_STA)
    );

    ESP_ERROR_CHECK(
        esp_wifi_set_config(
            WIFI_IF_STA,
            &wifi_config
        )
    );

    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_connect());
}

void AudioCaptureTask(void*) {
    if (audio_capture_init() != 0) {
        ESP_LOGE(
            kTag,
            "audio_capture_init() failed"
        );

        vTaskDelete(nullptr);
        return;
    }

    int16_t frame[
        app_config::kFrameSizeSamples
    ];

    while (true) {
        int64_t t0 = esp_timer_get_time();

        int read = audio_read(
            frame,
            app_config::kFrameSizeSamples
        );

        if (read ==
            static_cast<int>(
                app_config::kFrameSizeSamples
            )) {
            if (!g_ring_buffer.Push(
                    frame,
                    app_config::kFrameSizeSamples
                )) {
                ESP_LOGW(
                    kTag,
                    "ring buffer push failed"
                );
            }
        } else {
            ESP_LOGW(
                kTag,
                "audio_read short read: %d",
                read
            );
        }

        int64_t elapsed_us =
            esp_timer_get_time() - t0;

        RuntimeMetrics::Instance()
            .RecordLoopPeriod(
                static_cast<uint32_t>(
                    elapsed_us
                )
            );
    }
}

void KwsTask(void*) {
    if (g_kws_inference.Init() != 0) {
        ESP_LOGE(
            kTag,
            "KWS model initialization failed"
        );

        while (true) {
            vTaskDelay(
                pdMS_TO_TICKS(5000)
            );

            ESP_LOGE(
                kTag,
                "KWS model not initialized"
            );
        }
    }

    const size_t required_samples =
        app_config::kWindowLengthSamples +
        static_cast<size_t>(
            app_config::kNumFrames - 1
        ) *
        app_config::kHopLengthSamples;

    int16_t window_buffer[
        required_samples
    ];

    RuntimeMetrics::Instance()
        .SetStaticSizes(
            app_config::kTensorArenaBytes,
            0,
            app_config::kRingBufferSamples *
                sizeof(int16_t),
            sizeof(g_feature_buffer),
            app_config::kTcpSendBufferBytes +
                app_config::kTcpRecvBufferBytes
        );

    while (true) {
        if (g_trigger_sm.state() !=
            AppState::kListening) {
            vTaskDelay(
                pdMS_TO_TICKS(20)
            );
            continue;
        }

        size_t got =
            g_ring_buffer.GetLast(
                window_buffer,
                required_samples
            );

        if (got < required_samples) {
            vTaskDelay(
                pdMS_TO_TICKS(
                    app_config::kFrameDurationMs
                )
            );
            continue;
        }

        int64_t feat_t0 =
            esp_timer_get_time();

        bool feature_ok =
            g_feature_extractor.Compute(
                window_buffer,
                required_samples,
                g_feature_buffer
            );

        int64_t feat_elapsed =
            esp_timer_get_time() - feat_t0;

        RuntimeMetrics::Instance()
            .RecordFeatureExtractionDuration(
                static_cast<uint32_t>(
                    feat_elapsed
                )
            );

        if (!feature_ok) {
            vTaskDelay(
                pdMS_TO_TICKS(
                    app_config::kFrameDurationMs
                )
            );
            continue;
        }

        KwsResult result;

        int64_t inf_t0 =
            esp_timer_get_time();

        int inference_status =
            g_kws_inference.Run(
                g_feature_buffer,
                &result
            );

        int64_t inf_elapsed =
            esp_timer_get_time() - inf_t0;

        RuntimeMetrics::Instance()
            .RecordKwsInferenceDuration(
                static_cast<uint32_t>(
                    inf_elapsed
                )
            );

        if (inference_status != 0) {
            vTaskDelay(
                pdMS_TO_TICKS(
                    app_config::kFrameDurationMs
                )
            );
            continue;
        }

        const float keyword_probability =
            result.probabilities[
                kKeywordIndex
            ];

        AppState new_state =
            g_trigger_sm.Update(
                keyword_probability
            );

        if (new_state ==
            AppState::kKeywordDetected) {
            ESP_LOGI(
                kTag,
                "VAANI detected! probability=%.3f",
                keyword_probability
            );

            StreamRequest request{};

            const uint32_t r0 = esp_random();
            const uint32_t r1 = esp_random();
            const uint32_t r2 = esp_random();
            const uint32_t r3 = esp_random();

            std::memcpy(
                request.session_id + 0,
                &r0,
                sizeof(r0)
            );

            std::memcpy(
                request.session_id + 4,
                &r1,
                sizeof(r1)
            );

            std::memcpy(
                request.session_id + 8,
                &r2,
                sizeof(r2)
            );

            std::memcpy(
                request.session_id + 12,
                &r3,
                sizeof(r3)
            );

            if (xQueueSend(
                    g_stream_request_queue,
                    &request,
                    0
                ) != pdTRUE) {
                ESP_LOGW(
                    kTag,
                    "stream request queue full"
                );

                g_trigger_sm.ForceState(
                    AppState::kListening
                );
            } else {
                g_trigger_sm.ForceState(
                    AppState::kStreaming
                );
            }
        }

        vTaskDelay(
            pdMS_TO_TICKS(
                app_config::kFrameDurationMs
            )
        );
    }
}

void StreamingTask(void*) {
    LanTransport transport(
        app_config::kServerHost,
        app_config::kServerPort,
        app_config::kConnectTimeoutMs
    );

    int16_t pre_roll_buffer[
        app_config::kPreRollSamples
    ];

    int16_t frame[
        app_config::kFrameSizeSamples
    ];

    while (true) {
        StreamRequest request{};

        if (xQueueReceive(
                g_stream_request_queue,
                &request,
                portMAX_DELAY
            ) != pdTRUE) {
            continue;
        }

        TransportStatus status =
            transport.Connect();

        if (status !=
            TransportStatus::kOk) {
            ESP_LOGE(
                kTag,
                "stream connection failed"
            );

            g_trigger_sm.ForceState(
                AppState::kErrorRecovery
            );

            vTaskDelay(
                pdMS_TO_TICKS(500)
            );

            g_trigger_sm.ForceState(
                AppState::kListening
            );

            continue;
        }

        status =
            transport.SendStartSession(
                request.session_id,
                app_config::kSampleRateHz,
                app_config::kChannels
            );

        if (status !=
            TransportStatus::kOk) {
            ESP_LOGE(
                kTag,
                "SendStartSession failed"
            );

            transport.Disconnect();

            g_trigger_sm.ForceState(
                AppState::kErrorRecovery
            );

            vTaskDelay(
                pdMS_TO_TICKS(500)
            );

            g_trigger_sm.ForceState(
                AppState::kListening
            );

            continue;
        }

        size_t pre_roll_count =
            g_ring_buffer.GetLast(
                pre_roll_buffer,
                app_config::kPreRollSamples
            );

        if (pre_roll_count > 0) {
            status =
                transport.SendAudioChunk(
                    request.session_id,
                    0,
                    pre_roll_buffer,
                    pre_roll_count,
                    app_config::kSampleRateHz,
                    app_config::kChannels
                );

            if (status !=
                TransportStatus::kOk) {
                ESP_LOGW(
                    kTag,
                    "pre-roll send failed"
                );

                transport.Disconnect();

                g_trigger_sm.ForceState(
                    AppState::kErrorRecovery
                );

                vTaskDelay(
                    pdMS_TO_TICKS(500)
                );

                g_trigger_sm.ForceState(
                    AppState::kListening
                );

                continue;
            }
        }

        g_ring_buffer.StartStreaming();

        const uint32_t max_extra_frames =
            app_config::kMaxStreamDurationMs /
            app_config::kFrameDurationMs;

        uint32_t sequence_number = 1;

        for (uint32_t i = 0;
             i < max_extra_frames;
             ++i) {

            size_t got =
                g_ring_buffer.ReadStreaming(
                    frame,
                    app_config::kFrameSizeSamples
                );

            if (got == 0) {
                vTaskDelay(
                    pdMS_TO_TICKS(
                        app_config::kFrameDurationMs
                    )
                );
                continue;
            }

            TransportStatus send_status =
                transport.SendAudioChunk(
                    request.session_id,
                    sequence_number++,
                    frame,
                    got,
                    app_config::kSampleRateHz,
                    app_config::kChannels
                );

            if (send_status !=
                TransportStatus::kOk) {
                ESP_LOGW(
                    kTag,
                    "audio streaming failed"
                );
                break;
            }

            vTaskDelay(
                pdMS_TO_TICKS(
                    app_config::kFrameDurationMs
                )
            );
        }

        transport.SendEndSession(
            request.session_id,
            sequence_number
        );

        char transcript[512] = {};

        TransportStatus result_status =
            transport.ReceiveResult(
                transcript,
                sizeof(transcript),
                app_config::kSessionTimeoutMs
            );

        if (result_status ==
            TransportStatus::kOk) {
            ESP_LOGI(
                kTag,
                "Transcript: %s",
                transcript
            );
        } else {
            ESP_LOGW(
                kTag,
                "No valid ASR result: %d",
                static_cast<int>(
                    result_status
                )
            );
        }

        transport.Disconnect();

        g_trigger_sm.ForceState(
            AppState::kSessionComplete
        );

        g_trigger_sm.ForceState(
            AppState::kListening
        );
    }
}

void MetricsTask(void*) {
    while (true) {
        RuntimeMetrics::Instance()
            .LogSnapshot();

        vTaskDelay(
            pdMS_TO_TICKS(5000)
        );
    }
}

}

extern "C" void app_main() {
    ESP_LOGI(
        kTag,
        "VAANI firmware starting"
    );

    g_stream_request_queue =
        xQueueCreate(
            2,
            sizeof(StreamRequest)
        );

    if (g_stream_request_queue == nullptr) {
        ESP_LOGE(
            kTag,
            "failed to create stream queue"
        );
        return;
    }

    WifiInit();

    xTaskCreate(
        AudioCaptureTask,
        "audio_capture_task",
        4096,
        nullptr,
        10,
        nullptr
    );

    xTaskCreate(
        KwsTask,
        "kws_task",
        8192,
        nullptr,
        8,
        nullptr
    );

    xTaskCreate(
        StreamingTask,
        "streaming_task",
        6144,
        nullptr,
        5,
        nullptr
    );

    xTaskCreate(
        MetricsTask,
        "metrics_task",
        3072,
        nullptr,
        2,
        nullptr
    );
}