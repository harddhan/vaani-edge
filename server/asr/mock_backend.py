"""Mock ASR backend for tests and development without a real ASR engine.

Returns a deterministic placeholder transcript that encodes basic
information about the received audio (duration, RMS) so integration
tests can assert on it without needing a real speech engine installed.
"""
from __future__ import annotations

import numpy as np

from server.asr.base import ASRBackend


class MockASRBackend(ASRBackend):
    """Deterministic fake ASR backend - never claims to transcribe real speech."""

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        duration_s = len(audio) / max(sample_rate, 1)
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12)) if len(audio) else 0.0
        return f"[MOCK_TRANSCRIPT duration={duration_s:.2f}s rms={rms:.4f}]"
