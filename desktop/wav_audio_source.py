from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


class WavAudioSource:
    def __init__(self, wav_path: Path, expected_sample_rate: int = 16000) -> None:
        self._wav_path = Path(wav_path)

        with wave.open(str(self._wav_path), "rb") as wf:
            if wf.getframerate() != expected_sample_rate:
                raise ValueError(
                    f"{wav_path}: expected {expected_sample_rate} Hz, "
                    f"got {wf.getframerate()} Hz"
                )

            if wf.getsampwidth() != 2:
                raise ValueError(f"{wav_path}: expected 16-bit PCM")

            channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())

        data = np.frombuffer(raw, dtype="<i2")

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1).astype("<i2")

        self._samples = data
        self._position = 0
        self._exhausted = False

    @property
    def total_samples(self) -> int:
        return len(self._samples)

    def read(self, frame_size: int) -> np.ndarray:
        if self._exhausted:
            raise StopIteration

        end = self._position + frame_size
        chunk = self._samples[self._position:end]

        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
            self._exhausted = True

        self._position = min(end, len(self._samples))

        return chunk.astype("<i2")