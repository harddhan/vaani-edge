"""ASR backend interface.

Any ASR engine (mock, local open-source, or a future replacement) must
implement this interface so ``server/main.py`` can remain agnostic to the
specific backend in use.
"""
from __future__ import annotations

import abc

import numpy as np


class ASRBackend(abc.ABC):
    """Abstract base class for speech-to-text backends."""

    @abc.abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe mono float32 PCM audio in [-1, 1] and return text.

        Implementations should raise ``ASRUnavailableError`` if the
        backend's dependencies are not installed/loaded rather than
        silently returning a fake transcript.
        """
        raise NotImplementedError


class ASRUnavailableError(Exception):
    """Raised when an ASR backend cannot run (missing dependency, etc.)."""
