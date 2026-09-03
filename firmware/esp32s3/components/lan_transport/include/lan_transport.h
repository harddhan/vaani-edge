// LAN transport: Wi-Fi connection management + WebSocket-over-TCP client
// for streaming audio to the Python server.
//
// Design goal: a blocked/failed network connection must NEVER stall
// audio capture or KWS inference (see docs/state_machine.md). This is
// achieved by running all networking on its own FreeRTOS task
// (streaming_task in app_main.cpp) that only becomes active once the
// trigger state machine enters kStreaming; audio_capture_task and
// kws_task never call into this module directly.
#ifndef LAN_TRANSPORT_H_
#define LAN_TRANSPORT_H_

#include <cstddef>
#include <cstdint>

#include "protocol_codec.h"

enum class TransportStatus {
    kOk,
    kConnectFailed,
    kSendFailed,
    kTimeout,
    kProtocolError,
};

class LanTransport {
   public:
    LanTransport(const char* server_host, uint16_t server_port, uint32_t connect_timeout_ms);

    // Establishes the TCP connection + WebSocket handshake. Retries
    // internally using the configured backoff (see app_config.h) up to a
    // small bounded number of attempts, then gives up and returns
    // kConnectFailed - callers must handle this by transitioning to
    // kErrorRecovery rather than blocking forever.
    TransportStatus Connect();

    // Sends a START_SESSION / AUDIO_CHUNK / END_SESSION message. Returns
    // kSendFailed on any write error (e.g. connection dropped
    // mid-stream) so the caller can transition to kErrorRecovery and
    // return to LISTENING without crashing.
    TransportStatus SendStartSession(const uint8_t session_id[16], uint32_t sample_rate_hz,
                                       uint8_t channels);
    TransportStatus SendAudioChunk(const uint8_t session_id[16], uint32_t sequence_number,
                                     const int16_t* samples, size_t num_samples,
                                     uint32_t sample_rate_hz, uint8_t channels);
    TransportStatus SendEndSession(const uint8_t session_id[16], uint32_t sequence_number);

    // Blocks (with `timeout_ms`) for the ASR_RESULT/ERROR response.
    // Writes the decoded text payload into `out_text` (caller-provided
    // buffer of size `out_text_capacity`) and returns kOk /
    // kProtocolError / kTimeout.
    TransportStatus ReceiveResult(char* out_text, size_t out_text_capacity, uint32_t timeout_ms);

    void Disconnect();

   private:
    TransportStatus SendMessage(protocol::MessageType type, const uint8_t session_id[16],
                                  uint32_t sequence_number, const uint8_t* payload,
                                  size_t payload_len, uint32_t sample_rate_hz, uint8_t channels);

    const char* server_host_;
    uint16_t server_port_;
    uint32_t connect_timeout_ms_;
    int socket_fd_ = -1;
    bool connected_ = false;
};

#endif  // LAN_TRANSPORT_H_
