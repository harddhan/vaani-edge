from __future__ import annotations

import csv
import json
import wave
from pathlib import Path

import numpy as np

from ml.features.audio_features import extract_features
from ml.features.feature_spec import FeatureSpec


LABEL_ORDER = ("speech", "noise", "silence", "vaani")


def read_wav_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if channels != 1:
        raise ValueError(f"{path}: expected mono WAV, got {channels} channels")

    if sample_width != 2:
        raise ValueError(
            f"{path}: expected 16-bit PCM WAV, got {sample_width * 8}-bit"
        )

    audio = (
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    )

    return audio, sample_rate


def load_raw_split(
    manifest_path: Path,
    spec: FeatureSpec,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = []
    y = []
    paths = []

    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            path_value = row.get("path")
            label = row.get("label")

            if not path_value or not label:
                continue

            if label not in LABEL_ORDER:
                continue

            path = Path(path_value)

            if not path.exists():
                continue

            audio, sample_rate = read_wav_float(path)

            if sample_rate != spec.sample_rate_hz:
                raise ValueError(
                    f"{path}: expected {spec.sample_rate_hz} Hz, "
                    f"got {sample_rate} Hz"
                )

            features = extract_features(audio, spec)

            X.append(features)
            y.append(LABEL_ORDER.index(label))
            paths.append(str(path))

    if not X:
        raise ValueError(f"No valid samples found in {manifest_path}")

    X_array = np.stack(X).astype(np.float32)
    y_array = np.asarray(y, dtype=np.int64)

    X_array = np.nan_to_num(
        X_array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return X_array, y_array, paths


def calculate_normalization_stats(
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if X.ndim != 4:
        raise ValueError(
            "Expected feature array with shape "
            f"(samples, frames, mfcc, channels), got {X.shape}"
        )

    if X.shape[-1] != 1:
        raise ValueError(
            "Expected a single feature channel, "
            f"got shape {X.shape}"
        )

    mean = np.mean(
        X,
        axis=(0, 1, 3),
    ).astype(np.float32)

    std = np.std(
        X,
        axis=(0, 1, 3),
    ).astype(np.float32)

    std = np.maximum(std, 1e-6)

    return mean, std


def normalize_features(
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if X.ndim != 4:
        raise ValueError(
            "Expected feature array with shape "
            f"(samples, frames, mfcc, channels), got {X.shape}"
        )

    expected_shape = (X.shape[2],)

    if mean.shape != expected_shape:
        raise ValueError(
            f"Expected mean shape {expected_shape}, "
            f"got {mean.shape}"
        )

    if std.shape != expected_shape:
        raise ValueError(
            f"Expected std shape {expected_shape}, "
            f"got {std.shape}"
        )

    if np.any(std < 1e-6):
        raise ValueError("Feature standard deviation is too small.")

    mean = mean.reshape(1, 1, -1, 1)
    std = std.reshape(1, 1, -1, 1)

    return ((X - mean) / std).astype(np.float32)


def save_normalization_stats(
    path: Path,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    if mean.shape != std.shape:
        raise ValueError(
            f"Mean shape {mean.shape} does not match "
            f"std shape {std.shape}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def load_normalization_stats(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if "mean" not in payload or "std" not in payload:
        raise ValueError(
            f"Invalid normalization file: {path}"
        )

    mean = np.asarray(
        payload["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        payload["std"],
        dtype=np.float32,
    )

    if mean.shape != std.shape:
        raise ValueError(
            f"Mean shape {mean.shape} does not match "
            f"std shape {std.shape}"
        )

    if mean.shape != (13,):
        raise ValueError(
            f"Expected normalization shape (13,), "
            f"got {mean.shape}"
        )

    if np.any(std < 1e-6):
        raise ValueError(
            "Normalization standard deviation is too small."
        )

    return mean, std


def representative_dataset_generator(
    manifest_path: Path,
    spec: FeatureSpec,
    mean: np.ndarray,
    std: np.ndarray,
    max_samples: int = 200,
):
    X, _, _ = load_raw_split(
        manifest_path,
        spec,
    )

    X = normalize_features(
        X,
        mean,
        std,
    )

    n = min(
        max_samples,
        len(X),
    )

    for i in range(n):
        yield [
            X[i : i + 1].astype(np.float32)
        ]