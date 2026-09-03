# LAN Network Protocol

## Transport

VAANI uses **WebSocket over TCP** for the core LAN transport. Audio is carried in binary protocol messages.

UDP is outside the core implementation.

## Header

Every message has a 44-byte big-endian header.

| Offset | Size | Field | Type | Notes |
|---|---:|---|---|---|
| 0 | 4 | magic | ASCII | `EVA1` |
| 4 | 1 | version | uint8 | `1` |
| 5 | 1 | message type | uint8 | See message types |
| 6 | 4 | sample rate | uint32 | `16000` |
| 10 | 1 | sample format | uint8 | `0 = PCM_S16_LE` |
| 11 | 1 | channels | uint8 | `1` |
| 12 | 4 | sequence number | uint32 | Monotonic per session |
| 16 | 8 | timestamp | uint64 | Sender timestamp in ms |
| 24 | 16 | session ID | bytes | Per-session identifier |
| 40 | 4 | payload length | uint32 | Bytes after header |

## Message types

| Value | Name | Payload |
|---:|---|---|
| 1 | `START_SESSION` | Empty |
| 2 | `AUDIO_CHUNK` | Raw PCM bytes |
| 3 | `END_SESSION` | Empty |
| 4 | `ERROR` | UTF-8 error message |
| 5 | `ASR_RESULT` | UTF-8 transcript |

## Session flow

```text
Client                         Server
  |-- START_SESSION ----------->|
  |-- AUDIO_CHUNK pre-roll ---->|
  |-- AUDIO_CHUNK live -------->|
  |-- AUDIO_CHUNK live -------->|
  |-- END_SESSION ------------->|
  |<-- ASR_RESULT / ERROR ------|
```

## Reliability

- Sequence numbers are monotonic within a session.
- Each session has a 16-byte session identifier.
- The server enforces message and session size limits.
- Sessions time out after the configured inactivity period.
- Headers are validated before processing.
- Connection failures are handled without unbounded waiting.
- The server can reconstruct received audio on a best-effort basis if a session ends prematurely.

## Canonical implementations

Python:

`server/protocol.py`

Firmware:

`firmware/esp32s3/components/lan_transport/`

Protocol tests:

`tests/python/test_protocol.py`

## Current hardware status

The Python/server protocol path is implemented and testable. The ESP32-S3 WebSocket client still requires hardware-side interoperability work before the complete firmware-to-server transport can be claimed as validated.

## UDP

UDP is not part of the core deliverable. Any future UDP experiment should be implemented as a separate transport and evaluated independently.
