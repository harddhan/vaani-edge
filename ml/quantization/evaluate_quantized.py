from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

from ml.features.feature_spec import FeatureSpec
from ml.training.dataset_loader import (
    LABEL_ORDER,
    load_normalization_stats,
    load_raw_split,
    normalize_features,
)


DEFAULT_MODEL = Path("ml/artifacts/ds_cnn_final.keras")
DEFAULT_TFLITE = Path("ml/artifacts/ds_cnn_final_int8.tflite")
DEFAULT_MANIFEST = Path("data/processed/test.csv")
DEFAULT_NORMALIZATION = Path("ml/artifacts/normalization.json")
DEFAULT_OUTPUT = Path("reports/quantized_evaluation.json")


def evaluate_keras(
    model_path: Path,
    X: np.ndarray,
) -> np.ndarray:
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    probabilities = model.predict(
        X,
        verbose=0,
    )

    return np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int64)


def quantize_input(
    X: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(
            f"Invalid input quantization scale: {scale}"
        )

    quantized = np.round(
        X / scale + zero_point
    )

    return np.clip(
        quantized,
        -128,
        127,
    ).astype(np.int8)


def dequantize_output(
    output: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    return (
        output.astype(np.float32) - zero_point
    ) * scale


def evaluate_tflite(
    model_path: Path,
    X: np.ndarray,
) -> tuple[np.ndarray, dict]:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise ValueError(
            f"Expected one input tensor, got {len(input_details)}"
        )

    if len(output_details) != 1:
        raise ValueError(
            f"Expected one output tensor, got {len(output_details)}"
        )

    input_detail = input_details[0]
    output_detail = output_details[0]

    input_scale, input_zero_point = (
        input_detail["quantization"]
    )

    output_scale, output_zero_point = (
        output_detail["quantization"]
    )

    if input_detail["dtype"] != np.int8:
        raise ValueError(
            f"Expected int8 input, got {input_detail['dtype']}"
        )

    if output_detail["dtype"] != np.int8:
        raise ValueError(
            f"Expected int8 output, got {output_detail['dtype']}"
        )

    X_quantized = quantize_input(
        X,
        float(input_scale),
        int(input_zero_point),
    )

    predictions = []

    for sample in X_quantized:
        interpreter.set_tensor(
            input_detail["index"],
            sample[np.newaxis, ...],
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_detail["index"]
        )

        output_float = dequantize_output(
            output,
            float(output_scale),
            int(output_zero_point),
        )

        predictions.append(
            int(np.argmax(output_float[0]))
        )

    metadata = {
        "input_shape": [
            int(value)
            for value in input_detail["shape"]
        ],
        "input_dtype": str(input_detail["dtype"]),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_shape": [
            int(value)
            for value in output_detail["shape"]
        ],
        "output_dtype": str(output_detail["dtype"]),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    }

    return np.asarray(
        predictions,
        dtype=np.int64,
    ), metadata


def main() -> int:
    model_path = DEFAULT_MODEL
    tflite_path = DEFAULT_TFLITE
    manifest_path = DEFAULT_MANIFEST
    normalization_path = DEFAULT_NORMALIZATION
    output_path = DEFAULT_OUTPUT

    spec = FeatureSpec()

    X_raw, y_true, _ = load_raw_split(
        manifest_path,
        spec,
    )

    mean, std = load_normalization_stats(
        normalization_path,
    )

    X = normalize_features(
        X_raw,
        mean,
        std,
    )

    keras_predictions = evaluate_keras(
        model_path,
        X,
    )

    tflite_predictions, tflite_metadata = evaluate_tflite(
        tflite_path,
        X,
    )

    keras_accuracy = accuracy_score(
        y_true,
        keras_predictions,
    )

    int8_accuracy = accuracy_score(
        y_true,
        tflite_predictions,
    )

    agreement = float(
        np.mean(
            keras_predictions == tflite_predictions
        )
    )

    keras_matrix = confusion_matrix(
        y_true,
        keras_predictions,
        labels=list(range(len(LABEL_ORDER))),
    )

    int8_matrix = confusion_matrix(
        y_true,
        tflite_predictions,
        labels=list(range(len(LABEL_ORDER))),
    )

    report = {
        "model": str(model_path),
        "tflite_model": str(tflite_path),
        "manifest": str(manifest_path),
        "normalization": str(normalization_path),
        "num_samples": int(len(y_true)),
        "labels": list(LABEL_ORDER),
        "keras_accuracy": float(keras_accuracy),
        "int8_accuracy": float(int8_accuracy),
        "accuracy_difference": float(
            int8_accuracy - keras_accuracy
        ),
        "keras_int8_prediction_agreement": agreement,
        "keras_confusion_matrix": [
            [int(value) for value in row]
            for row in keras_matrix
        ],
        "int8_confusion_matrix": [
            [int(value) for value in row]
            for row in int8_matrix
        ],
        "tflite": tflite_metadata,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            report,
            fh,
            indent=2,
        )
        fh.write("\n")

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print(
        f"\n[OK] Quantized evaluation saved to {output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())