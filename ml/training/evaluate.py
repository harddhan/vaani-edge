"""Evaluate a trained keyword-spotting model with KWS-specific metrics.

Reports accuracy metrics AND keyword-spotting-specific measures required
by the problem statement: false activations per hour, detection latency
(from temporal smoothing simulation), per-speaker and per-noise-condition
breakdowns when metadata is available.

Usage:
    python -m ml.training.evaluate --model ml/artifacts/small_cnn_best.keras
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from ml.features.feature_spec import feature_spec_from_config, load_config
from ml.training.dataset_loader import LABEL_ORDER, load_split

KEYWORD_INDEX = LABEL_ORDER.index("keyword")


def compute_metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Binary keyword-vs-rest metrics at a given probability threshold."""
    y_true_binary = (y_true == KEYWORD_INDEX).astype(int)
    y_pred_binary = (probs[:, KEYWORD_INDEX] >= threshold).astype(int)

    tp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 1)))
    fp = int(np.sum((y_pred_binary == 1) & (y_true_binary == 0)))
    fn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 1)))
    tn = int(np.sum((y_pred_binary == 0) & (y_true_binary == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def estimate_false_activations_per_hour(
    fp_count: int, num_negative_windows: int, window_hop_ms: float
) -> float:
    """Estimate false activations/hour assuming continuous streaming.

    This extrapolates from the evaluation set's negative (non-keyword)
    windows to a full hour of continuous audio at the configured hop
    length. It is an ESTIMATE from the test set, not a substitute for a
    long-duration live soak test on the actual ESP32 hardware - see
    docs/evaluation_methodology.md.
    """
    if num_negative_windows == 0:
        return float("nan")
    windows_per_hour = 3600.0 / (window_hop_ms / 1000.0)
    fp_rate_per_window = fp_count / num_negative_windows
    return fp_rate_per_window * windows_per_hour


def per_group_metrics(
    y_true: np.ndarray, probs: np.ndarray, groups: list[str], threshold: float
) -> dict:
    result = {}
    unique_groups = sorted(set(g for g in groups if g))
    for group in unique_groups:
        idx = [i for i, g in enumerate(groups) if g == group]
        if not idx:
            continue
        result[group] = compute_metrics_at_threshold(y_true[idx], probs[idx], threshold)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"[ERROR] Model not found: {args.model}. Run ml/training/train.py first.")
        return 2

    config = load_config()
    spec = feature_spec_from_config(config)
    detection_cfg = config.get("detection", {})
    threshold = args.threshold or detection_cfg.get("default_threshold", 0.8)

    manifest_path = args.data_dir / f"{args.split}.csv"
    if not manifest_path.exists():
        print(f"[ERROR] {manifest_path} not found. Run dataset prep + split first.")
        return 2

    X, y, paths = load_split(manifest_path, spec)

    with open(manifest_path, "r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    path_to_row = {row["path"]: row for row in rows}
    speakers = [path_to_row.get(p, {}).get("speaker_id", "") for p in paths]
    noise_conditions = [path_to_row.get(p, {}).get("noise_condition", "") for p in paths]

    model = tf.keras.models.load_model(args.model)
    probs = model.predict(X)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y, y_pred, labels=list(range(len(LABEL_ORDER))))
    overall = compute_metrics_at_threshold(y, probs, threshold)

    num_negative_windows = int(np.sum(y != KEYWORD_INDEX))
    hop_ms = config.get("features", {}).get("hop_length_ms", 20)
    fa_per_hour = estimate_false_activations_per_hour(overall["false_positives"], num_negative_windows, hop_ms)

    sweep = [
        compute_metrics_at_threshold(y, probs, t) for t in detection_cfg.get("threshold_sweep", [0.5, 0.6, 0.7, 0.8, 0.9])
    ]

    per_speaker = per_group_metrics(y, probs, speakers, threshold) if any(speakers) else {}
    per_noise = per_group_metrics(y, probs, noise_conditions, threshold) if any(noise_conditions) else {}

    report = {
        "model_path": str(args.model),
        "split": args.split,
        "num_samples": len(y),
        "confusion_matrix": cm.tolist(),
        "label_order": list(LABEL_ORDER),
        "overall_at_default_threshold": overall,
        "threshold_sweep": sweep,
        "estimated_false_activations_per_hour": fa_per_hour,
        "false_activations_per_hour_note": (
            "Extrapolated from held-out negative windows at the configured "
            "hop length. MUST be re-validated with a long-duration live "
            "soak test on real ESP32 hardware before claiming compliance "
            "with the 'near zero false activations' requirement."
        ),
        "per_speaker": per_speaker,
        "per_noise_condition": per_noise,
        "detection_latency_note": (
            "This script measures per-window classification only. "
            "End-to-end detection latency (including temporal smoothing "
            "and cooldown) must be measured via desktop/latency_probe.py "
            "or on-device runtime_metrics."
        ),
    }

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.reports_dir / f"evaluation_{args.model.stem}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved evaluation report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
