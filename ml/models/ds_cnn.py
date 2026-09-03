"""Depthwise-Separable CNN (DS-CNN) keyword-spotting candidate.

Based on the well-known DS-CNN architecture family used in the TensorFlow
"speech commands" microcontroller examples (Zhang et al., "Hello Edge:
Keyword Spotting on Microcontrollers"). Depthwise-separable convolutions
give a good accuracy/compute trade-off versus the small_cnn baseline at
roughly similar or lower parameter count.

Architecture (documented):

    Input (num_frames, n_mels, 1)
    -> Conv2D(64, 10x4, stride 2x2, relu) + BatchNorm
    -> [DepthwiseConv2D(3x3, relu) + BatchNorm + Conv2D(64,1x1,relu) + BatchNorm] x 3
    -> GlobalAveragePooling2D
    -> Dense(num_classes, softmax)

Use ``model_summary.py`` to compare parameter/operation counts against
small_cnn before committing to one architecture.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _depthwise_separable_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.DepthwiseConv2D((3, 3), padding="same", activation=None, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)

    x = layers.Conv2D(filters, (1, 1), padding="same", activation=None, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def build_ds_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.2,
    base_filters: int = 64,
    num_ds_blocks: int = 3,
) -> tf.keras.Model:
    """Build and return the DS-CNN candidate model (uncompiled)."""
    inputs = layers.Input(shape=input_shape, name="mel_features")

    x = layers.Conv2D(base_filters, (10, 4), strides=(2, 2), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="conv1_bn")(x)
    x = layers.ReLU(name="conv1_relu")(x)

    for i in range(num_ds_blocks):
        x = _depthwise_separable_block(x, base_filters, name=f"ds{i + 1}")

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="ds_cnn_kws")
