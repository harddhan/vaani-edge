"""Tests for connection-loss / bounded-memory behavior in the server layer."""
from __future__ import annotations

import pytest

from server.audio_buffer import AudioBufferOverflowError, BoundedAudioBuffer
from server.session import ServerSession


def test_session_never_exceeds_configured_memory_bound():
    """Never allocate unlimited memory: appending beyond max_buffer_bytes raises."""
    max_bytes = 16
    session = ServerSession(session_id=b"\x00" * 16, sample_rate_hz=16000, channels=1, max_buffer_bytes=max_bytes)
    session.add_chunk(0, b"A" * 16)  # exactly at the limit - OK
    assert session.buffer.total_bytes == 16
    with pytest.raises(AudioBufferOverflowError):
        session.add_chunk(1, b"B")  # any more must be rejected


def test_partial_stream_reconstructs_available_audio_on_disconnect():
    """Simulates a connection dropping mid-stream: only received chunks
    are reconstructed, no exception, no hang."""
    session = ServerSession(session_id=b"\x00" * 16, sample_rate_hz=16000, channels=1, max_buffer_bytes=1000)
    session.add_chunk(0, b"AAAA")
    session.add_chunk(1, b"BBBB")
    # Simulate disconnect before END_SESSION / chunk 2 ever arrives.
    session.close()
    assert session.reconstruct_audio() == b"AAAABBBB"
    assert session.closed is True


def test_buffer_reports_dropped_chunks_without_raising():
    buf = BoundedAudioBuffer(max_bytes=1000)
    buf.append(0, b"AAAA")
    buf.append(5, b"FFFF")  # large gap simulating dropped chunks 1-4
    assert buf.stats.dropped_estimate == 4
    # Reconstruction still succeeds with the data that IS present.
    assert buf.reconstruct() == b"AAAAFFFF"
