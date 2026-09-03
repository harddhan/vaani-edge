from __future__ import annotations

import numpy as np

from ml.features.feature_spec import FeatureSpec


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(spec: FeatureSpec) -> np.ndarray:
    n_bins = spec.n_fft // 2 + 1

    mel_min = hz_to_mel(np.array(spec.fmin_hz, dtype=np.float64))
    mel_max = hz_to_mel(np.array(spec.fmax_hz, dtype=np.float64))

    mel_points = np.linspace(
        mel_min,
        mel_max,
        spec.n_mels + 2,
    )

    hz_points = mel_to_hz(mel_points)

    bins = np.floor(
        (spec.n_fft + 1) * hz_points / spec.sample_rate_hz
    ).astype(int)

    bins = np.clip(bins, 0, n_bins - 1)

    filterbank = np.zeros(
        (spec.n_mels, n_bins),
        dtype=np.float32,
    )

    for m in range(1, spec.n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = left + 1

        if right <= center:
            right = center + 1

        for k in range(left, min(center, n_bins)):
            filterbank[m - 1, k] = (
                (k - left) / (center - left)
            )

        for k in range(center, min(right, n_bins)):
            filterbank[m - 1, k] = (
                (right - k) / (right - center)
            )

    return filterbank


def pre_emphasis(
    audio: np.ndarray,
    coefficient: float,
) -> np.ndarray:
    if len(audio) == 0:
        return audio.astype(np.float32)

    emphasized = np.empty_like(audio, dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = (
        audio[1:] - coefficient * audio[:-1]
    )

    return emphasized


def frame_signal(
    audio: np.ndarray,
    frame_length: int,
    hop_length: int,
    num_frames: int,
) -> np.ndarray:
    required_samples = (
        frame_length
        + (num_frames - 1) * hop_length
    )

    if len(audio) < required_samples:
        audio = np.pad(
            audio,
            (0, required_samples - len(audio)),
        )
    else:
        audio = audio[:required_samples]

    indices = (
        np.arange(frame_length)[None, :]
        + hop_length * np.arange(num_frames)[:, None]
    )

    return audio[indices]


def log_mel_spectrogram(
    audio: np.ndarray,
    spec: FeatureSpec,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)

    audio = pre_emphasis(
        audio,
        spec.pre_emphasis,
    )

    frames = frame_signal(
        audio,
        spec.window_length_samples,
        spec.hop_length_samples,
        spec.num_frames,
    )

    window = np.hanning(
        spec.window_length_samples
    ).astype(np.float32)

    frames = frames * window

    spectrum = np.fft.rfft(
        frames,
        n=spec.n_fft,
        axis=1,
    )

    power = (
        np.abs(spectrum) ** 2
    ) / spec.n_fft

    filterbank = build_mel_filterbank(spec)

    mel_energy = power @ filterbank.T

    mel_energy = np.maximum(
        mel_energy,
        1e-10,
    )

    return np.log(mel_energy).astype(np.float32)


def fixed_length_features(
    features: np.ndarray,
    num_frames: int,
) -> np.ndarray:
    if features.shape[0] == num_frames:
        return features

    if features.shape[0] > num_frames:
        return features[:num_frames]

    padding = num_frames - features.shape[0]

    return np.pad(
        features,
        ((0, padding), (0, 0)),
        mode="constant",
    )


def normalize_features(
    features: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return features

    if mode == "peak":
        peak = np.max(
            np.abs(features)
        ) + 1e-8
        return features / peak

    if mode == "per_feature_mean_std":
        mean = features.mean(
            axis=0,
            keepdims=True,
        )

        std = features.std(
            axis=0,
            keepdims=True,
        ) + 1e-8

        return (features - mean) / std

    raise ValueError(
        f"Unknown normalization mode: {mode}"
    )