"""Convert a trained Keras model to float32 and full-INT8 TFLite models.

Usage:
    python -m ml.quantization.convert_int8 --model ml/artifacts/ds_cnn_final.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.features.feature_spec import feature_spec_from_config, load_config
from ml.training.dataset_loader import (
    load_normalization_stats,
    representative_dataset_generator,
)


class QuantizationError(Exception):
    """Raised when full integer quantization could not be achieved."""


def convert_float32(model: tf.keras.Model) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_int8(
    model: tf.keras.Model,
    representative_dataset,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def verify_full_int8(tflite_model_bytes: bytes) -> dict:
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model_bytes
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details["dtype"] != np.int8:
        raise QuantizationError(
            f"Input dtype is {input_details['dtype']}, expected int8."
        )

    if output_details["dtype"] != np.int8:
        raise QuantizationError(
            f"Output dtype is {output_details['dtype']}, expected int8."
        )

    float_tensors = []
    op_names = set()

    for tensor_detail in interpreter.get_tensor_details():
        if tensor_detail["dtype"] == np.float32:
            float_tensors.append(tensor_detail["name"])

    try:
        from tensorflow.lite.python import schema_py_generated as schema_fb

        model_obj = schema_fb.Model.GetRootAsModel(
            tflite_model_bytes,
            0,
        )
        subgraph = model_obj.Subgraphs(0)

        for i in range(subgraph.OperatorsLength()):
            op = subgraph.Operators(i)
            opcode_index = op.OpcodeIndex()
            opcode = model_obj.OperatorCodes(opcode_index)
            builtin_code = opcode.DeprecatedBuiltinCode()
            op_names.add(str(builtin_code))

    except Exception:
        op_names = {"unavailable_in_this_tf_version"}

    if float_tensors:
        raise QuantizationError(
            f"Full INT8 quantization NOT achieved: "
            f"{len(float_tensors)} float32 tensors remain "
            f"(e.g. {float_tensors[:5]})."
        )

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    return {
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "num_float32_tensors_remaining": len(float_tensors),
        "operator_codes_seen": sorted(op_names),
        "model_size_bytes": len(tflite_model_bytes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts"),
    )

    parser.add_argument(
        "--normalization",
        type=Path,
        default=Path("ml/artifacts/normalization.json"),
    )

    args = parser.parse_args(argv)

    if not args.model.exists():
        print(
            f"[ERROR] Model not found: {args.model}. "
            "Train a model first."
        )
        return 2

    calibration_manifest = args.data_dir / "train.csv"

    if not calibration_manifest.exists():
        print(
            f"[ERROR] {calibration_manifest} not found. "
            "Run dataset preparation and splitting first."
        )
        return 2

    if not args.normalization.exists():
        print(
            f"[ERROR] Normalization file not found: "
            f"{args.normalization}. "
            "Train the model first so train-only normalization "
            "statistics are generated."
        )
        return 2

    if args.calibration_samples <= 0:
        print("[ERROR] --calibration-samples must be greater than zero.")
        return 2

    config = load_config()
    spec = feature_spec_from_config(config)

    mean, std = load_normalization_stats(
        args.normalization
    )

    if mean.shape != (spec.n_mfcc,):
        print(
            f"[ERROR] Normalization mean shape {mean.shape} "
            f"does not match expected feature shape "
            f"({spec.n_mfcc},)."
        )
        return 2

    if std.shape != (spec.n_mfcc,):
        print(
            f"[ERROR] Normalization std shape {std.shape} "
            f"does not match expected feature shape "
            f"({spec.n_mfcc},)."
        )
        return 2

    model = tf.keras.models.load_model(args.model)
    name = args.model.stem

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Converting float32 TFLite model...")

    float32_bytes = convert_float32(model)

    float32_path = (
        args.output_dir /
        f"{name}_float32.tflite"
    )

    float32_path.write_bytes(float32_bytes)

    print(
        f"  -> {float32_path} "
        f"({len(float32_bytes):,} bytes)"
    )

    print("Converting full-INT8 TFLite model...")

    def representative_dataset():
        yield from representative_dataset_generator(
            calibration_manifest,
            spec,
            mean,
            std,
            args.calibration_samples,
        )

    int8_bytes = convert_int8(
        model,
        representative_dataset,
    )

    int8_path = (
        args.output_dir /
        f"{name}_int8.tflite"
    )

    int8_path.write_bytes(int8_bytes)

    print(
        f"  -> {int8_path} "
        f"({len(int8_bytes):,} bytes)"
    )

    print("Verifying full integer quantization...")

    try:
        metadata = verify_full_int8(
            int8_bytes
        )
    except QuantizationError as exc:
        print(f"[FAILED] {exc}")
        return 1

    metadata["source_model"] = str(args.model)
    metadata["normalization_file"] = str(
        args.normalization
    )
    metadata["calibration_samples"] = min(
        args.calibration_samples,
        len(
            list(
                open(
                    calibration_manifest,
                    encoding="utf-8",
                )
            )
        ) - 1,
    )
    metadata["float32_model_bytes"] = len(
        float32_bytes
    )
    metadata["int8_model_bytes"] = len(
        int8_bytes
    )
    metadata["compression_ratio"] = round(
        len(float32_bytes) /
        max(len(int8_bytes), 1),
        2,
    )

    metadata_path = (
        args.output_dir /
        f"{name}_quant_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            metadata,
            fh,
            indent=2,
        )

    print(json.dumps(metadata, indent=2))

    print(
        f"\n[OK] Full INT8 quantization verified. "
        f"Metadata saved to {metadata_path}"
    )

    print(
        "\nNOTE: TFLite model size is FLASH usage. "
        "Tensor arena RAM usage must still be measured "
        "on the ESP32-S3."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())