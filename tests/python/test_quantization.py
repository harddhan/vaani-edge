"""Tests for ml/quantization/convert_int8.py's verification logic.

TensorFlow is a heavy optional dependency for running the full test
suite in lightweight environments; these tests are skipped automatically
if `tensorflow` is not importable (they still run in the full dev
environment described in requirements.txt).
"""
from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

import numpy as np  # noqa: E402

from ml.quantization.convert_int8 import (  # noqa: E402
    QuantizationError,
    convert_float32,
    convert_int8,
    verify_full_int8,
)


def _build_tiny_model():
    inputs = tf.keras.layers.Input(shape=(8, 8, 1))
    x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def _representative_dataset():
    for _ in range(20):
        yield [np.random.uniform(-1, 1, (1, 8, 8, 1)).astype(np.float32)]


def test_int8_conversion_produces_int8_io():
    model = _build_tiny_model()
    int8_bytes = convert_int8(model, _representative_dataset)
    metadata = verify_full_int8(int8_bytes)
    assert metadata["input_dtype"] == "<class 'numpy.int8'>" or "int8" in metadata["input_dtype"]
    assert metadata["num_float32_tensors_remaining"] == 0


def test_float32_conversion_produces_valid_model():
    model = _build_tiny_model()
    float32_bytes = convert_float32(model)
    assert len(float32_bytes) > 0
    interpreter = tf.lite.Interpreter(model_content=float32_bytes)
    interpreter.allocate_tensors()  # should not raise


def test_verify_rejects_float_model_as_not_int8():
    model = _build_tiny_model()
    float32_bytes = convert_float32(model)
    with pytest.raises(QuantizationError):
        verify_full_int8(float32_bytes)
