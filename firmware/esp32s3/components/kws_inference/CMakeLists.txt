idf_component_register(
    SRCS "src/kws_inference.cpp" "model/kws_model_data.cc"
    INCLUDE_DIRS "include" "model"
    REQUIRES feature_extractor
    # Requires a TensorFlow Lite Micro / LiteRT Micro ESP-IDF component,
    # e.g. `espressif/esp-tflite-micro` added via main/idf_component.yml.
    # PRIV_REQUIRES esp-tflite-micro
)
