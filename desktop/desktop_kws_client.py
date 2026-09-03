from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from desktop.protocol_client import StreamingSessionClient
from desktop.wav_audio_source import WavAudioSource
from ml.features.audio_features import extract_features_from_pcm16
from ml.features.feature_spec import FeatureSpec
from ml.training.dataset_loader import (
    load_normalization_stats,
    normalize_features,
)

logger = logging.getLogger("vaani.desktop.kws")


LABEL_ORDER = (
    "speech",
    "noise",
    "silence",
    "vaani",
)

KEYWORD_INDEX = LABEL_ORDER.index("vaani")


class TriggerStateMachine:
    def __init__(
        self,
        threshold: float,
        consecutive_positive_windows: int,
        cooldown_windows: int,
    ) -> None:
        self.threshold = threshold
        self.consecutive_positive_windows = max(
            1,
            consecutive_positive_windows,
        )
        self.cooldown_windows = max(
            0,
            cooldown_windows,
        )
        self.consecutive_count = 0
        self.cooldown_remaining = 0

    def update(self, probability: float) -> bool:
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.consecutive_count = 0
            return False

        if probability >= self.threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0

        if self.consecutive_count >= self.consecutive_positive_windows:
            self.consecutive_count = 0
            self.cooldown_remaining = self.cooldown_windows
            return True

        return False

    def reset(self) -> None:
        self.consecutive_count = 0
        self.cooldown_remaining = 0


class RingBuffer:
    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")

        self._capacity = capacity_samples
        self._buffer = np.zeros(
            capacity_samples,
            dtype="<i2",
        )
        self._write_pos = 0
        self._filled = 0

    def push(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype="<i2")

        if len(frame) > self._capacity:
            frame = frame[-self._capacity:]

        n = len(frame)
        end = self._write_pos + n

        if end <= self._capacity:
            self._buffer[self._write_pos:end] = frame
        else:
            first = self._capacity - self._write_pos
            self._buffer[self._write_pos:] = frame[:first]
            self._buffer[:end - self._capacity] = frame[first:]

        self._write_pos = end % self._capacity
        self._filled = min(
            self._capacity,
            self._filled + n,
        )

    def get_last(self, samples: int) -> np.ndarray:
        samples = min(samples, self._filled)

        start = (self._write_pos - samples) % self._capacity

        if start + samples <= self._capacity:
            return self._buffer[
                start:start + samples
            ].copy()

        first = self._capacity - start

        return np.concatenate(
            [
                self._buffer[start:],
                self._buffer[:samples - first],
            ]
        )


async def run_simulation(
    wav_path: Path,
    server_uri: str,
    force_trigger: bool,
    trigger_frame_index: int | None,
    model_path: Path | None = None,
) -> None:
    spec = FeatureSpec()

    sample_rate = spec.sample_rate_hz
    frame_size = int(sample_rate * 30 / 1000)
    pre_roll_samples = int(sample_rate * 800 / 1000)
    ring_buffer_samples = int(sample_rate * 2000 / 1000)

    normalization_path = Path(
        "ml/artifacts/normalization.json"
    )

    model = None
    mean = None
    std = None

    if model_path is not None and not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not force_trigger:
        if model_path is None:
            raise ValueError(
                "Provide --model or use --force-trigger."
            )

        if not normalization_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {normalization_path}"
            )

        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)

        mean, std = load_normalization_stats(
            normalization_path
        )

    source = WavAudioSource(
        wav_path,
        expected_sample_rate=sample_rate,
    )

    ring_buffer = RingBuffer(ring_buffer_samples)

    trigger_sm = TriggerStateMachine(
        threshold=0.80,
        consecutive_positive_windows=3,
        cooldown_windows=75,
    )

    window_samples = spec.expected_audio_samples

    accumulated = np.zeros(
        0,
        dtype="<i2",
    )

    frame_index = 0
    triggered = False

    while True:
        try:
            frame = source.read(frame_size)
        except StopIteration:
            break

        ring_buffer.push(frame)

        accumulated = np.concatenate(
            [accumulated, frame]
        )

        frame_index += 1

        if len(accumulated) < window_samples:
            continue

        window = accumulated[-window_samples:]

        if (
            force_trigger
            and (
                trigger_frame_index is None
                or frame_index >= trigger_frame_index
            )
        ):
            probability = 0.99

        elif model is not None:
            features = extract_features_from_pcm16(
                window,
                spec,
            )

            features = normalize_features(
                features,
                mean,
                std,
            )

            probabilities = model.predict(
                features[np.newaxis, ...],
                verbose=0,
            )[0]

            probability = float(
                probabilities[KEYWORD_INDEX]
            )

        else:
            probability = 0.0

        if trigger_sm.update(probability):
            triggered = True

            logger.info(
                "VAANI detected at frame %d: %.3f",
                frame_index,
                probability,
            )

            break

    if not triggered:
        logger.info("No VAANI trigger detected.")
        return

    pre_roll = ring_buffer.get_last(
        pre_roll_samples
    )

    remaining = source_remaining(
        source,
        frame_size,
    )

    utterance = np.concatenate(
        [pre_roll, remaining]
    )

    client = StreamingSessionClient(
        server_uri,
        sample_rate_hz=sample_rate,
    )

    transcript = await client.stream_utterance(
        utterance,
        frame_size_samples=frame_size,
    )

    print(f"TRANSCRIPT: {transcript}")


def source_remaining(
    source: WavAudioSource,
    frame_size: int,
) -> np.ndarray:
    chunks = []

    while True:
        try:
            chunks.append(
                source.read(frame_size)
            )
        except StopIteration:
            break

    if not chunks:
        return np.zeros(
            0,
            dtype="<i2",
        )

    return np.concatenate(chunks)