"""A small CNN baseline suitable for microcontroller keyword spotting.

Architecture (documented, intentionally simple):

    Input (num_frames, n_mels, 1)
    -> Conv2D(8, 3x3, relu) -> MaxPool(2x2)
    -> Conv2D(16, 3x3, relu) -> MaxPool(2x2)
    -> Flatten
    -> Dense(32, relu) -> Dropout
    -> Dense(num_classes, softmax)

This is the baseline to compare the DS-CNN candidate against. Parameter
count is small enough for INT8 deployment on an ESP32-S3 (order of tens
of thousands of parameters at default config); always confirm the actual
count with ``model_summary.py`` after changing ``configs/model.yaml``.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_small_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.2,
) -> tf.keras.Model:
    """Build and return the small CNN baseline model (uncompiled)."""
    inputs = layers.Input(shape=input_shape, name="mel_features")

    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="small_cnn_kws")
