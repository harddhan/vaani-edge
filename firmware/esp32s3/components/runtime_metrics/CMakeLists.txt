idf_component_register(
    SRCS "src/runtime_metrics.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_timer freertos
)
