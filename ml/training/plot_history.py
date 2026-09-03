"""Plot training history (loss/accuracy curves) saved by train.py.

Usage:
    python -m ml.training.plot_history --history ml/artifacts/small_cnn_history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.history.exists():
        print(f"[ERROR] History file not found: {args.history}. Run train.py first.")
        return 2

    with open(args.history, "r", encoding="utf-8") as fh:
        history = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.get("loss", []), label="train_loss")
    axes[0].plot(history.get("val_loss", []), label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train_accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    output = args.output or args.history.with_suffix(".png")
    fig.savefig(output)
    print(f"Saved plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
