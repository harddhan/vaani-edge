from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


LABELS = ("speech", "noise", "silence", "vaani")


def load_manifest(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def split_dataset(
    rows: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)

    by_label = defaultdict(list)

    for row in rows:
        by_label[row["label"]].append(row)

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for label in LABELS:
        label_rows = by_label[label]
        rng.shuffle(label_rows)

        total = len(label_rows)
        test_count = int(total * test_ratio)
        val_count = int(total * val_ratio)

        splits["test"].extend(label_rows[:test_count])
        splits["val"].extend(
            label_rows[test_count:test_count + val_count]
        )
        splits["train"].extend(
            label_rows[test_count + val_count:]
        )

    for split in splits.values():
        rng.shuffle(split)

    return splits


def write_manifests(
    splits: dict[str, list[dict]],
    output_dir: Path,
    fieldnames: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        path = output_dir / f"{split_name}.csv"

        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"{split_name}: {len(rows)} files")


def print_class_counts(
    splits: dict[str, list[dict]],
) -> None:
    for split_name, rows in splits.items():
        counts = {label: 0 for label in LABELS}

        for row in rows:
            counts[row["label"]] += 1

        print(
            f"{split_name}: "
            + ", ".join(
                f"{label}={counts[label]}"
                for label in LABELS
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        return 2

    if args.val_split + args.test_split >= 1.0:
        print("[ERROR] Validation and test splits must total less than 1.0")
        return 2

    rows = load_manifest(args.manifest)

    if not rows:
        print("[ERROR] Manifest is empty.")
        return 1

    splits = split_dataset(
        rows,
        args.val_split,
        args.test_split,
        args.seed,
    )

    write_manifests(
        splits,
        args.output,
        list(rows[0].keys()),
    )

    print()
    print_class_counts(splits)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())