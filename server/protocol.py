from __future__ import annotations

import dataclasses
import struct
import uuid
from enum import IntEnum


PROTOCOL_MAGIC = b"EVA1"
PROTOCOL_VERSION = 1

_HEADER_FORMAT = "!4sBBIBBIQ16sI"
HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)

if HEADER_SIZE != 44:
    raise RuntimeError(f"Invalid protocol header size: {HEADER_SIZE}")

MAX_MESSAGE_BYTES = 65536
MAX_PAYLOAD_BYTES = MAX_MESSAGE_BYTES - HEADER_SIZE


class MessageType(IntEnum):
    START_SESSION = 1
    AUDIO_CHUNK = 2
    END_SESSION = 3
    ERROR = 4
    ASR_RESULT = 5


class SampleFormat(IntEnum):
    PCM_S16_LE = 0


class ProtocolError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class MessageHeader:
    magic: bytes
    version: int
    msg_type: MessageType
    sample_rate_hz: int
    sample_format: SampleFormat
    channels: int
    sequence_number: int
    timestamp_ms: int
    session_id: bytes
    payload_length: int

    @property
    def session_id_str(self) -> str:
        return str(uuid.UUID(bytes=self.session_id))


def encode_message(
    msg_type: MessageType,
    session_id: bytes,
    sequence_number: int,
    timestamp_ms: int,
    payload: bytes = b"",
    sample_rate_hz: int = 16000,
    sample_format: SampleFormat = SampleFormat.PCM_S16_LE,
    channels: int = 1,
) -> bytes:
    if len(session_id) != 16:
        raise ProtocolError(
            f"session_id must be 16 bytes, got {len(session_id)}"
        )

    if not 0 <= sequence_number <= 0xFFFFFFFF:
        raise ProtocolError("sequence_number out of range")

    if not 0 <= sample_rate_hz <= 0xFFFFFFFF:
        raise ProtocolError("sample_rate_hz out of range")

    if not 0 <= timestamp_ms <= 0xFFFFFFFFFFFFFFFF:
        raise ProtocolError("timestamp_ms out of range")

    if not 1 <= channels <= 255:
        raise ProtocolError("channels must be between 1 and 255")

    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload too large: {len(payload)} > {MAX_PAYLOAD_BYTES}"
        )

    try:
        msg_type = MessageType(msg_type)
    except ValueError as exc:
        raise ProtocolError(f"unknown message type: {msg_type}") from exc

    try:
        sample_format = SampleFormat(sample_format)
    except ValueError as exc:
        raise ProtocolError(
            f"unknown sample format: {sample_format}"
        ) from exc

    header = struct.pack(
        _HEADER_FORMAT,
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        int(msg_type),
        sample_rate_hz,
        int(sample_format),
        channels,
        sequence_number,
        timestamp_ms,
        session_id,
        len(payload),
    )

    return header + payload


def decode_header(data: bytes) -> MessageHeader:
    if len(data) < HEADER_SIZE:
        raise ProtocolError(
            f"message too short for header: {len(data)} < {HEADER_SIZE}"
        )

    (
        magic,
        version,
        msg_type_raw,
        sample_rate_hz,
        sample_format_raw,
        channels,
        sequence_number,
        timestamp_ms,
        session_id,
        payload_length,
    ) = struct.unpack(_HEADER_FORMAT, data[:HEADER_SIZE])

    if magic != PROTOCOL_MAGIC:
        raise ProtocolError(
            f"bad magic: {magic!r} (expected {PROTOCOL_MAGIC!r})"
        )

    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported protocol version: {version}"
        )

    try:
        msg_type = MessageType(msg_type_raw)
    except ValueError as exc:
        raise ProtocolError(
            f"unknown message type: {msg_type_raw}"
        ) from exc

    try:
        sample_format = SampleFormat(sample_format_raw)
    except ValueError as exc:
        raise ProtocolError(
            f"unknown sample format: {sample_format_raw}"
        ) from exc

    if sample_rate_hz <= 0:
        raise ProtocolError("sample_rate_hz must be positive")

    if channels <= 0:
        raise ProtocolError("channels must be positive")

    if payload_length > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"declared payload too large: {payload_length}"
        )

    return MessageHeader(
        magic=magic,
        version=version,
        msg_type=msg_type,
        sample_rate_hz=sample_rate_hz,
        sample_format=sample_format,
        channels=channels,
        sequence_number=sequence_number,
        timestamp_ms=timestamp_ms,
        session_id=session_id,
        payload_length=payload_length,
    )


def decode_message(data: bytes) -> tuple[MessageHeader, bytes]:
    header = decode_header(data)
    expected_size = HEADER_SIZE + header.payload_length

    if len(data) != expected_size:
        raise ProtocolError(
            f"message length mismatch: expected {expected_size}, got {len(data)}"
        )

    return header, data[HEADER_SIZE:expected_size]


def new_session_id() -> bytes:
    return uuid.uuid4().bytes