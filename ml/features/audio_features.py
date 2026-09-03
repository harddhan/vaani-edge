from __future__ import annotations

import numpy as np
from scipy.fftpack import dct

from ml.features.feature_spec import FeatureSpec
from ml.features.mel_spectrogram import log_mel_spectrogram


def pcm16_to_float(pcm: np.ndarray) -> np.ndarray:
    return np.asarray(pcm, dtype=np.float32) / 32768.0


def extract_features(
    audio_float: np.ndarray,
    spec: FeatureSpec,
) -> np.ndarray:
    audio = np.asarray(audio_float, dtype=np.float32)

    log_mel = log_mel_spectrogram(audio, spec)

    mfcc = dct(
        log_mel,
        type=2,
        axis=1,
        norm="ortho",
    )

    mfcc = mfcc[:, :spec.n_mfcc]

    if mfcc.shape[0] != spec.num_frames:
        raise ValueError(
            f"Expected {spec.num_frames} frames, got {mfcc.shape[0]}"
        )

    if mfcc.shape[1] != spec.n_mfcc:
        raise ValueError(
            f"Expected {spec.n_mfcc} MFCC coefficients, got {mfcc.shape[1]}"
        )

    return mfcc[..., np.newaxis].astype(np.float32)


def extract_features_from_pcm16(
    pcm: np.ndarray,
    spec: FeatureSpec,
) -> np.ndarray:
    return extract_features(
        pcm16_to_float(pcm),
        spec,
    )