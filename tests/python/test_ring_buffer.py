"""Tests for the desktop RingBuffer (pre-roll recovery), mirroring the
firmware's ring_buffer component behavior (see
firmware/esp32s3/components/ring_buffer)."""
from __future__ import annotations

import numpy as np

from desktop.desktop_kws_client import RingBuffer


def test_push_and_get_last_returns_most_recent():
    rb = RingBuffer(capacity_samples=100)
    rb.push(np.arange(50, dtype="<i2"))
    last_10 = rb.get_last(10)
    assert list(last_10) == list(range(40, 50))


def test_wraparound_preserves_recent_data():
    rb = RingBuffer(capacity_samples=10)
    rb.push(np.arange(0, 8, dtype="<i2"))
    rb.push(np.arange(8, 16, dtype="<i2"))  # wraps past capacity
    last_5 = rb.get_last(5)
    assert list(last_5) == list(range(11, 16))


def test_get_last_capped_at_filled_amount():
    rb = RingBuffer(capacity_samples=100)
    rb.push(np.arange(5, dtype="<i2"))
    result = rb.get_last(50)  # asking for more than pushed
    assert len(result) == 5


def test_preroll_recovery_matches_expected_window():
    # Simulates the pre-roll requirement: after streaming many frames,
    # the last `pre_roll_samples` should be recoverable exactly.
    rb = RingBuffer(capacity_samples=32000)
    frame_size = 480
    total_frames = 100
    all_samples = []
    for i in range(total_frames):
        frame = np.full(frame_size, i % 128, dtype="<i2")
        rb.push(frame)
        all_samples.append(frame)
    full_stream = np.concatenate(all_samples)

    pre_roll_samples = 12800
    recovered = rb.get_last(pre_roll_samples)
    expected = full_stream[-pre_roll_samples:]
    assert np.array_equal(recovered, expected)
