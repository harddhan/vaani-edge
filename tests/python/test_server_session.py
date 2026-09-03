"""Tests for server/session.py's ServerSession and asyncio server integration."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from server.audio_buffer import AudioBufferOverflowError
from server.session import ServerSession


def test_session_add_chunk_and_reconstruct():
    session = ServerSession(session_id=b"\x00" * 16, sample_rate_hz=16000, channels=1, max_buffer_bytes=1000)
    session.add_chunk(0, b"AAAA")
    session.add_chunk(1, b"BBBB")
    assert session.reconstruct_audio() == b"AAAABBBB"


def test_session_close_sets_end_timestamp():
    session = ServerSession(session_id=b"\x00" * 16, sample_rate_hz=16000, channels=1, max_buffer_bytes=1000)
    assert session.timestamps.session_end_monotonic is None
    session.close()
    assert session.closed is True
    assert session.timestamps.session_end_monotonic is not None


def test_session_latency_report_has_expected_keys():
    session = ServerSession(session_id=b"\x00" * 16, sample_rate_hz=16000, channels=1, max_buffer_bytes=1000)
    session.add_chunk(0, b"AAAA")
    session.mark_asr_start()
    session.mark_asr_end()
    session.close()
    report = session.latency_report()
    for key in (
        "buffering_delay_ms",
        "network_delay_ms",
        "server_processing_delay_ms",
        "asr_duration_ms",
        "total_session_duration_ms",
        "sequence_stats",
    ):
        assert key in report


def test_session_overflow_propagates():
    session = ServerSession(session_id=b"\x00" * 16, sample_rate_hz=16000, channels=1, max_buffer_bytes=4)
    session.add_chunk(0, b"AAAA")
    with pytest.raises(AudioBufferOverflowError):
        session.add_chunk(1, b"BBBB")


@pytest.mark.asyncio
async def test_mock_asr_backend_transcribes_without_crashing():
    from server.asr.mock_backend import MockASRBackend

    backend = MockASRBackend()
    audio = np.zeros(16000, dtype=np.float32)
    transcript = await backend.transcribe(audio, 16000)
    assert isinstance(transcript, str)
    assert "MOCK_TRANSCRIPT" in transcript


def test_session_id_str_roundtrip():
    import uuid

    raw = uuid.uuid4().bytes
    session = ServerSession(session_id=raw, sample_rate_hz=16000, channels=1, max_buffer_bytes=1000)
    assert session.session_id_str == str(uuid.UUID(bytes=raw))
