// C++ mirror of server/protocol.py's binary wire format.
// Keep the header layout byte-for-byte identical to that file /
// docs/network_protocol.md.
#ifndef PROTOCOL_CODEC_H_
#define PROTOCOL_CODEC_H_

#include <cstddef>
#include <cstdint>

namespace protocol {

constexpr uint8_t kProtocolVersion = 1;
constexpr char kProtocolMagic[4] = {'E', 'V', 'A', '1'};
constexpr size_t kHeaderSize = 44;

enum class MessageType : uint8_t {
    kStartSession = 1,
    kAudioChunk = 2,
    kEndSession = 3,
    kError = 4,
    kAsrResult = 5,
};

enum class SampleFormat : uint8_t {
    kPcmS16Le = 0,
};

#pragma pack(push, 1)
struct MessageHeader {
    char magic[4];
    uint8_t version;
    uint8_t msg_type;
    uint32_t sample_rate_hz;      // big-endian on the wire, see EncodeHeader/DecodeHeader
    uint8_t sample_format;
    uint8_t channels;
    uint32_t sequence_number;     // big-endian on the wire
    uint64_t timestamp_ms;        // big-endian on the wire
    uint8_t session_id[16];
    uint32_t payload_length;      // big-endian on the wire
};
#pragma pack(pop)

// Serializes a header into `out` (must have space for kHeaderSize
// bytes), converting multi-byte integer fields to network byte order
// (big-endian) to match struct.pack("!...", ...) on the Python side.
void EncodeHeader(MessageType type, const uint8_t session_id[16], uint32_t sequence_number,
                   uint64_t timestamp_ms, uint32_t payload_length, uint32_t sample_rate_hz,
                   SampleFormat sample_format, uint8_t channels, uint8_t* out);

// Parses a header from `data` (must have at least kHeaderSize bytes).
// Returns true on success (valid magic/version), false otherwise.
bool DecodeHeader(const uint8_t* data, size_t len, MessageHeader* out);

}  // namespace protocol

#endif  // PROTOCOL_CODEC_H_
