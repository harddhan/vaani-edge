"""Write reconstructed session audio to a debug WAV file."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_debug_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> None:
    """Write raw little-endian int16 PCM bytes to a WAV file at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert raw little-endian int16 PCM bytes to float32 audio in [-1, 1]."""
    if len(pcm_bytes) % 2 != 0:
        pcm_bytes = pcm_bytes[:-1]  # drop a stray trailing byte defensively
    samples = np.frombuffer(pcm_bytes, dtype="<i2")
    return (samples.astype(np.float32) / 32768.0)
