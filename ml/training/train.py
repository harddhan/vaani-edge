from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from ml.features.feature_spec import FeatureSpec
from ml.models.ds_cnn import build_ds_cnn
from ml.models.small_cnn import build_small_cnn
from ml.training.dataset_loader import (
    LABEL_ORDER,
    calculate_normalization_stats,
    load_raw_split,
    normalize_features,
    save_normalization_stats,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model(
    architecture: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float,
) -> tf.keras.Model:
    if architecture == "small_cnn":
        return build_small_cnn(
            input_shape,
            num_classes,
            dropout,
        )

    if architecture == "ds_cnn":
        return build_ds_cnn(
            input_shape,
            num_classes,
            dropout,
        )

    raise ValueError(f"Unknown architecture: {architecture}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        choices=["small_cnn", "ds_cnn"],
        default="ds_cnn",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("ml/artifacts"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    args = parser.parse_args(argv)

    set_seed(args.seed)

    spec = FeatureSpec()

    train_manifest = args.data_dir / "train.csv"
    val_manifest = args.data_dir / "val.csv"
    test_manifest = args.data_dir / "test.csv"

    for manifest in (
        train_manifest,
        val_manifest,
        test_manifest,
    ):
        if not manifest.exists():
            raise FileNotFoundError(
                f"Missing manifest: {manifest}"
            )

    print("Loading dataset...")

    X_train, y_train, _ = load_raw_split(
        train_manifest,
        spec,
    )

    X_val, y_val, _ = load_raw_split(
        val_manifest,
        spec,
    )

    X_test, y_test, _ = load_raw_split(
        test_manifest,
        spec,
    )

    train_mean, train_std = calculate_normalization_stats(
        X_train
    )

    X_train = normalize_features(
        X_train,
        train_mean,
        train_std,
    )

    X_val = normalize_features(
        X_val,
        train_mean,
        train_std,
    )

    X_test = normalize_features(
        X_test,
        train_mean,
        train_std,
    )

    print(
        f"train={len(X_train)} "
        f"val={len(X_val)} "
        f"test={len(X_test)}"
    )

    print(
        "Normalization mean:",
        np.array2string(
            train_mean,
            precision=4,
        ),
    )

    print(
        "Normalization std:",
        np.array2string(
            train_std,
            precision=4,
        ),
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABEL_ORDER)),
        y=y_train,
    )

    class_weight = {
        i: float(weight)
        for i, weight in enumerate(class_weights_array)
    }

    print(
        "Class weights:",
        class_weight,
    )

    model = build_model(
        args.architecture,
        spec.input_shape,
        len(LABEL_ORDER),
        0.2,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=args.learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(
        f"\nArchitecture: {args.architecture}"
    )

    print(
        f"Input shape: {spec.input_shape}"
    )

    print(
        f"Parameters: {model.count_params():,}"
    )

    args.artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalization_path = (
        args.artifacts_dir
        / "normalization.json"
    )

    save_normalization_stats(
        normalization_path,
        train_mean,
        train_std,
    )

    print(
        f"Saved normalization: {normalization_path}"
    )

    checkpoint_path = (
        args.artifacts_dir
        / f"{args.architecture}_best.keras"
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    history_path = (
        args.artifacts_dir
        / f"{args.architecture}_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as fh:
        json.dump(
            history.history,
            fh,
            indent=2,
        )

    print(
        f"Saved history: {history_path}"
    )

    print("\nEvaluating test set...")

    y_pred_probs = model.predict(
        X_test,
        verbose=0,
    )

    y_pred = np.argmax(
        y_pred_probs,
        axis=1,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(LABEL_ORDER)),
        target_names=LABEL_ORDER,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(LABEL_ORDER)),
    )

    print(report)

    print("Confusion matrix:")
    print(cm)

    report_path = (
        args.artifacts_dir
        / f"{args.architecture}_test_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write(report)
        fh.write("\n\nConfusion matrix:\n")
        fh.write(np.array2string(cm))

    final_model_path = (
        args.artifacts_dir
        / f"{args.architecture}_final.keras"
    )

    model.save(final_model_path)

    print(
        f"Saved model: {final_model_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())