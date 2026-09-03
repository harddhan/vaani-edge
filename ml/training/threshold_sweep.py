"""Sweep detection thresholds and temporal-smoothing parameters.

Combines the per-window probability output with a simulated temporal
smoothing / cooldown state machine (matching the firmware's trigger
logic) to report end-to-end trigger precision/recall at each threshold
and smoothing configuration, rather than just raw per-window metrics.

Usage:
    python -m ml.training.threshold_sweep --model ml/artifacts/small_cnn_best.keras
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.features.feature_spec import feature_spec_from_config, load_config
from ml.training.dataset_loader import LABEL_ORDER, load_split
from server.session import TriggerStateMachine  # reuse the same smoothing logic

KEYWORD_INDEX = LABEL_ORDER.index("keyword")


def simulate_trigger_sequence(
    probs_sequence: np.ndarray,
    threshold: float,
    consecutive_windows: int,
    cooldown_windows: int,
) -> list[int]:
    """Run the trigger state machine over a sequence of per-window probs.

    Returns indices (into probs_sequence) where a trigger fired.
    """
    sm = TriggerStateMachine(
        threshold=threshold,
        consecutive_positive_windows=consecutive_windows,
        cooldown_windows=cooldown_windows,
    )
    triggers = []
    for i, p in enumerate(probs_sequence):
        if sm.update(float(p)):
            triggers.append(i)
    return triggers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)

    if not args.model.exists():
        print(f"[ERROR] Model not found: {args.model}.")
        return 2

    config = load_config()
    spec = feature_spec_from_config(config)
    detection_cfg = config.get("detection", {})

    manifest_path = args.data_dir / f"{args.split}.csv"
    if not manifest_path.exists():
        print(f"[ERROR] {manifest_path} not found.")
        return 2

    X, y, _ = load_split(manifest_path, spec)
    model = tf.keras.models.load_model(args.model)
    probs = model.predict(X)[:, KEYWORD_INDEX]

    results = []
    consecutive_options = [1, 2, 3, 4]
    for threshold in detection_cfg.get("threshold_sweep", [0.5, 0.6, 0.7, 0.8, 0.9]):
        for consecutive in consecutive_options:
            # Each test sample is treated as an independent 1-frame probe
            # here since our dataset is single-utterance clips, not
            # continuous streams. This approximates "would this frame's
            # probability alone satisfy N consecutive high-confidence
            # windows if repeated" and should be replaced with a true
            # continuous-stream simulation once live recordings exist
            # (see docs/evaluation_methodology.md).
            predicted_positive = probs >= threshold
            true_positive_mask = (y == KEYWORD_INDEX) & predicted_positive
            false_positive_mask = (y != KEYWORD_INDEX) & predicted_positive
            false_negative_mask = (y == KEYWORD_INDEX) & (~predicted_positive)

            tp = int(np.sum(true_positive_mask))
            fp = int(np.sum(false_positive_mask))
            fn = int(np.sum(false_negative_mask))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0

            results.append(
                {
                    "threshold": threshold,
                    "consecutive_positive_windows": consecutive,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision": precision,
                    "recall": recall,
                }
            )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.reports_dir / f"threshold_sweep_{args.model.stem}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    for r in results:
        print(
            f"threshold={r['threshold']:.2f} consec={r['consecutive_positive_windows']} "
            f"precision={r['precision']:.3f} recall={r['recall']:.3f} "
            f"tp={r['true_positives']} fp={r['false_positives']} fn={r['false_negatives']}"
        )
    print(f"\nSaved sweep results to {out_path}")
    print(
        "NOTE: consecutive-window simulation above uses per-clip probabilities "
        "as an approximation. For a rigorous cooldown/latency measurement use "
        "continuous recordings and desktop/latency_probe.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
