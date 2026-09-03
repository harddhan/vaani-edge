idf_component_register(
    SRCS "src/lan_transport.cpp" "src/protocol_codec.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp_wifi esp_event nvs_flash lwip
)
