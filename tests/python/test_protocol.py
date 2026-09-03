"""Tests for server/protocol.py: header encoding/decoding, validation."""
from __future__ import annotations

import pytest

from server.protocol import (
    HEADER_SIZE,
    MAX_PAYLOAD_BYTES,
    MessageType,
    ProtocolError,
    SampleFormat,
    decode_header,
    decode_message,
    encode_message,
    new_session_id,
)


def test_header_size_is_44_bytes():
    assert HEADER_SIZE == 44


def test_encode_decode_roundtrip_audio_chunk():
    session_id = new_session_id()
    payload = b"\x01\x02" * 100
    message = encode_message(
        msg_type=MessageType.AUDIO_CHUNK,
        session_id=session_id,
        sequence_number=42,
        timestamp_ms=1234567890,
        payload=payload,
        sample_rate_hz=16000,
        channels=1,
    )
    header, decoded_payload = decode_message(message)

    assert header.msg_type == MessageType.AUDIO_CHUNK
    assert header.session_id == session_id
    assert header.sequence_number == 42
    assert header.timestamp_ms == 1234567890
    assert header.sample_rate_hz == 16000
    assert header.channels == 1
    assert header.sample_format == SampleFormat.PCM_S16_LE
    assert decoded_payload == payload


def test_encode_decode_roundtrip_empty_payload():
    session_id = new_session_id()
    message = encode_message(
        msg_type=MessageType.START_SESSION,
        session_id=session_id,
        sequence_number=0,
        timestamp_ms=0,
    )
    header, payload = decode_message(message)
    assert header.msg_type == MessageType.START_SESSION
    assert payload == b""


def test_decode_rejects_bad_magic():
    session_id = new_session_id()
    message = bytearray(
        encode_message(
            msg_type=MessageType.START_SESSION,
            session_id=session_id,
            sequence_number=0,
            timestamp_ms=0,
        )
    )
    message[0:4] = b"XXXX"
    with pytest.raises(ProtocolError):
        decode_header(bytes(message))


def test_decode_rejects_short_message():
    with pytest.raises(ProtocolError):
        decode_header(b"too short")


def test_decode_rejects_unknown_message_type():
    session_id = new_session_id()
    message = bytearray(
        encode_message(
            msg_type=MessageType.START_SESSION,
            session_id=session_id,
            sequence_number=0,
            timestamp_ms=0,
        )
    )
    message[5] = 99  # corrupt msg_type byte
    with pytest.raises(ProtocolError):
        decode_header(bytes(message))


def test_encode_rejects_oversized_payload():
    session_id = new_session_id()
    with pytest.raises(ProtocolError):
        encode_message(
            msg_type=MessageType.AUDIO_CHUNK,
            session_id=session_id,
            sequence_number=0,
            timestamp_ms=0,
            payload=b"\x00" * (MAX_PAYLOAD_BYTES + 1),
        )


def test_encode_rejects_bad_session_id_length():
    with pytest.raises(ProtocolError):
        encode_message(
            msg_type=MessageType.START_SESSION,
            session_id=b"\x00" * 10,
            sequence_number=0,
            timestamp_ms=0,
        )


def test_session_id_str_is_valid_uuid():
    session_id = new_session_id()
    header, _ = decode_message(
        encode_message(
            msg_type=MessageType.START_SESSION,
            session_id=session_id,
            sequence_number=0,
            timestamp_ms=0,
        )
    )
    assert len(header.session_id_str) == 36  # canonical UUID string length
