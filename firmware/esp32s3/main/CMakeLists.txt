idf_component_register(
    SRCS "app_main.cpp"
    INCLUDE_DIRS "."
    REQUIRES audio_capture ring_buffer feature_extractor kws_inference
             trigger_state_machine lan_transport runtime_metrics
             nvs_flash esp_wifi esp_event
)
