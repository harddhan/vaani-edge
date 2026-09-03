from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.features.audio_features import extract_features
from ml.features.feature_spec import FeatureSpec
from ml.training.dataset_loader import read_wav_float


LABELS = ("speech", "noise", "silence", "vaani")


def predict(
    model_path: Path,
    audio_path: Path,
) -> None:
    spec = FeatureSpec()

    audio, sample_rate = read_wav_float(audio_path)

    if sample_rate != spec.sample_rate_hz:
        raise ValueError(
            f"Expected {spec.sample_rate_hz} Hz, got {sample_rate} Hz"
        )

    features = extract_features(
        audio,
        spec,
    )

    features = np.nan_to_num(
        features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    mean = features.mean()
    std = features.std()

    if std < 1e-6:
        raise ValueError("Feature standard deviation is too small.")

    features = (features - mean) / std

    x = features[np.newaxis, ...].astype(np.float32)

    model = tf.keras.models.load_model(model_path)

    probabilities = model.predict(
        x,
        verbose=0,
    )[0]

    prediction = int(np.argmax(probabilities))

    print()
    print(f"Prediction: {LABELS[prediction]}")
    print(f"Confidence: {probabilities[prediction] * 100:.2f}%")
    print()
    print("Scores:")

    for label, probability in zip(LABELS, probabilities):
        print(f"  {label}: {probability * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "audio",
        type=Path,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ml/artifacts/ds_cnn_final.keras"),
    )

    args = parser.parse_args()

    predict(
        args.model,
        args.audio,
    )


if __name__ == "__main__":
    main()