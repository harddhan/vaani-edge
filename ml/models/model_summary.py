"""Print model summary, parameter count, and an approximate MAC/op count.

Usage:
    python -m ml.models.model_summary --architecture small_cnn
    python -m ml.models.model_summary --architecture ds_cnn
"""
from __future__ import annotations

import argparse

import tensorflow as tf

from ml.features.feature_spec import feature_spec_from_config, load_config
from ml.models.ds_cnn import build_ds_cnn
from ml.models.small_cnn import build_small_cnn


def estimate_flops(model: tf.keras.Model) -> int:
    """Rough MAC-based op estimate for Conv2D/Dense/DepthwiseConv2D layers.

    This is an approximation useful for comparing architectures, NOT a
    cycle-accurate estimate for the ESP32-S3. Always benchmark actual
    inference time on-device (see runtime_metrics component) before
    making performance claims.
    """
    total_macs = 0
    for layer in model.layers:
        config = layer.get_config()
        if isinstance(layer, tf.keras.layers.Conv2D):
            out_shape = layer.output_shape
            kernel_h, kernel_w = config["kernel_size"]
            in_channels = layer.input_shape[-1]
            out_channels = out_shape[-1]
            out_h, out_w = out_shape[1], out_shape[2]
            groups = config.get("groups", 1)
            total_macs += (kernel_h * kernel_w * in_channels * out_channels * out_h * out_w) // max(groups, 1)
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            out_shape = layer.output_shape
            kernel_h, kernel_w = config["kernel_size"]
            in_channels = layer.input_shape[-1]
            out_h, out_w = out_shape[1], out_shape[2]
            total_macs += kernel_h * kernel_w * in_channels * out_h * out_w
        elif isinstance(layer, tf.keras.layers.Dense):
            in_units = layer.input_shape[-1]
            out_units = layer.output_shape[-1]
            total_macs += in_units * out_units
    return total_macs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["small_cnn", "ds_cnn"], default="small_cnn")
    args = parser.parse_args(argv)

    config = load_config()
    spec = feature_spec_from_config(config)
    model_cfg = config.get("model", {})
    num_classes = model_cfg.get("num_classes", 3)
    dropout = model_cfg.get("dropout", 0.2)

    if args.architecture == "small_cnn":
        model = build_small_cnn(spec.input_shape, num_classes, dropout)
    else:
        model = build_ds_cnn(spec.input_shape, num_classes, dropout)

    model.summary()

    total_params = model.count_params()
    macs = estimate_flops(model)
    print(f"\nArchitecture: {args.architecture}")
    print(f"Input shape: {spec.input_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Approximate MACs per inference: {macs:,}")
    print(
        "NOTE: This is a parameter/MAC estimate only. Actual RAM (tensor "
        "arena), Flash size, and inference latency MUST be measured after "
        "INT8 conversion and on-device deployment (see ml/quantization/ "
        "and docs/memory_budget.md)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
