#include "protocol_codec.h"

#include <cstring>

namespace protocol {
namespace {

void WriteU32BE(uint8_t* out, uint32_t value) {
    out[0] = static_cast<uint8_t>((value >> 24) & 0xFF);
    out[1] = static_cast<uint8_t>((value >> 16) & 0xFF);
    out[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
    out[3] = static_cast<uint8_t>(value & 0xFF);
}

uint32_t ReadU32BE(const uint8_t* in) {
    return (static_cast<uint32_t>(in[0]) << 24) | (static_cast<uint32_t>(in[1]) << 16) |
           (static_cast<uint32_t>(in[2]) << 8) | static_cast<uint32_t>(in[3]);
}

void WriteU64BE(uint8_t* out, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out[i] = static_cast<uint8_t>((value >> (56 - i * 8)) & 0xFF);
    }
}

uint64_t ReadU64BE(const uint8_t* in) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value = (value << 8) | in[i];
    }
    return value;
}

}  // namespace

void EncodeHeader(MessageType type, const uint8_t session_id[16], uint32_t sequence_number,
                   uint64_t timestamp_ms, uint32_t payload_length, uint32_t sample_rate_hz,
                   SampleFormat sample_format, uint8_t channels, uint8_t* out) {
    size_t offset = 0;
    std::memcpy(out + offset, kProtocolMagic, 4);
    offset += 4;
    out[offset++] = kProtocolVersion;
    out[offset++] = static_cast<uint8_t>(type);
    WriteU32BE(out + offset, sample_rate_hz);
    offset += 4;
    out[offset++] = static_cast<uint8_t>(sample_format);
    out[offset++] = channels;
    WriteU32BE(out + offset, sequence_number);
    offset += 4;
    WriteU64BE(out + offset, timestamp_ms);
    offset += 8;
    std::memcpy(out + offset, session_id, 16);
    offset += 16;
    WriteU32BE(out + offset, payload_length);
    offset += 4;
    // offset should now equal kHeaderSize (44); enforced by the caller's
    // buffer sizing and by the host-side unit test.
}

bool DecodeHeader(const uint8_t* data, size_t len, MessageHeader* out) {
    if (len < kHeaderSize) {
        return false;
    }
    size_t offset = 0;
    std::memcpy(out->magic, data + offset, 4);
    offset += 4;
    if (std::memcmp(out->magic, kProtocolMagic, 4) != 0) {
        return false;
    }
    out->version = data[offset++];
    if (out->version != kProtocolVersion) {
        return false;
    }
    out->msg_type = data[offset++];
    out->sample_rate_hz = ReadU32BE(data + offset);
    offset += 4;
    out->sample_format = data[offset++];
    out->channels = data[offset++];
    out->sequence_number = ReadU32BE(data + offset);
    offset += 4;
    out->timestamp_ms = ReadU64BE(data + offset);
    offset += 8;
    std::memcpy(out->session_id, data + offset, 16);
    offset += 16;
    out->payload_length = ReadU32BE(data + offset);
    offset += 4;
    return true;
}

}  // namespace protocol
