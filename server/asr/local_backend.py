"""Local, open-source ASR backend adapter.

Wraps an open-source local ASR engine (recommended: `faster-whisper`,
which uses CTranslate2 and runs fully offline/locally, satisfying the
"no cloud dependency for the core demonstration" requirement).

This backend is OPTIONAL. If `faster-whisper` is not installed, this
module raises ``ASRUnavailableError`` from ``create()`` rather than
pretending to work - see server/asr/README.md for installation
instructions and server/main.py's fallback-to-WAV-only behavior.

Any other local, open-source ASR engine (e.g. Vosk, whisper.cpp via a
Python binding) can be swapped in here by implementing the same
``ASRBackend`` interface; the rest of the server does not need to change.
"""
from __future__ import annotations

import asyncio

import numpy as np

from server.asr.base import ASRBackend, ASRUnavailableError


class LocalWhisperASRBackend(ASRBackend):
    """Adapter around faster-whisper (optional dependency)."""

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dep
            raise ASRUnavailableError(
                "faster-whisper is not installed. Install it with: "
                "pip install faster-whisper  (see server/asr/README.md)"
            ) from exc

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio, sample_rate)

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError(f"LocalWhisperASRBackend expects 16kHz audio, got {sample_rate}Hz")
        segments, _info = self._model.transcribe(audio.astype(np.float32), language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()


def create_local_backend(model_size: str = "tiny") -> ASRBackend:
    """Factory used by server/main.py; isolates the optional-dependency check."""
    return LocalWhisperASRBackend(model_size=model_size)
