"""Tests for feature extraction shape/behavior (ml/features)."""
from __future__ import annotations

import numpy as np

from ml.features.audio_features import extract_features, pcm16_to_float
from ml.features.feature_spec import FeatureSpec
from ml.features.mel_spectrogram import (
    build_mel_filterbank,
    fixed_length_features,
    log_mel_spectrogram,
    normalize_features,
)


def test_feature_tensor_shape_matches_spec():
    spec = FeatureSpec(num_frames=49, n_mels=40)
    audio = np.random.uniform(-0.1, 0.1, spec.expected_audio_samples()).astype(np.float32)
    features = extract_features(audio, spec)
    assert features.shape == spec.input_shape
    assert features.shape == (49, 40, 1)


def test_feature_tensor_shape_with_short_audio_is_still_correct():
    spec = FeatureSpec(num_frames=49, n_mels=40)
    short_audio = np.random.uniform(-0.1, 0.1, 4000).astype(np.float32)  # shorter than needed
    features = extract_features(short_audio, spec)
    assert features.shape == spec.input_shape


def test_mel_filterbank_shape():
    spec = FeatureSpec(n_fft=512, n_mels=40)
    filterbank = build_mel_filterbank(spec)
    assert filterbank.shape == (40, 512 // 2 + 1)


def test_fixed_length_features_pads_short_input():
    features = np.ones((10, 5), dtype=np.float32)
    result = fixed_length_features(features, num_frames=20)
    assert result.shape == (20, 5)


def test_fixed_length_features_crops_long_input():
    features = np.ones((30, 5), dtype=np.float32)
    result = fixed_length_features(features, num_frames=20)
    assert result.shape == (20, 5)


def test_normalize_per_feature_mean_std():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    normalized = normalize_features(features, "per_feature_mean_std")
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-5)


def test_pcm16_to_float_range():
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)
    floats = pcm16_to_float(pcm)
    assert floats.min() >= -1.0
    assert floats.max() <= 1.0


def test_log_mel_spectrogram_no_nan_or_inf():
    spec = FeatureSpec()
    audio = np.zeros(spec.expected_audio_samples(), dtype=np.float32)  # pure silence
    result = log_mel_spectrogram(audio, spec)
    assert np.all(np.isfinite(result))
