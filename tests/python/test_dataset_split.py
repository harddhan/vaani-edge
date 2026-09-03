"""Tests for ml/dataset/split_dataset.py correctness."""
from __future__ import annotations

from ml.dataset.split_dataset import LABELS, split_dataset


def _row(path: str, label: str) -> dict:
    return {
        "path": path,
        "label": label,
    }


def test_all_rows_accounted_for():
    rows = [
        _row(f"{label}_{i}.wav", label)
        for label in LABELS
        for i in range(20)
    ]

    splits = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    total = sum(len(rows) for rows in splits.values())

    assert total == len(rows)


def test_all_labels_are_preserved():
    rows = [
        _row(f"{label}_{i}.wav", label)
        for label in LABELS
        for i in range(20)
    ]

    splits = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    for split_rows in splits.values():
        assert all(row["label"] in LABELS for row in split_rows)


def test_deterministic_with_fixed_seed():
    rows = [
        _row(f"file_{i}.wav", LABELS[i % len(LABELS)])
        for i in range(100)
    ]

    splits_a = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=1337,
    )

    splits_b = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=1337,
    )

    for split_name in splits_a:
        paths_a = [row["path"] for row in splits_a[split_name]]
        paths_b = [row["path"] for row in splits_b[split_name]]
        assert paths_a == paths_b


def test_different_seed_changes_split():
    rows = [
        _row(f"file_{i}.wav", LABELS[i % len(LABELS)])
        for i in range(100)
    ]

    splits_a = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=1,
    )

    splits_b = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=2,
    )

    test_a = {row["path"] for row in splits_a["test"]}
    test_b = {row["path"] for row in splits_b["test"]}

    assert test_a != test_b


def test_split_ratios_are_applied_per_class():
    rows = [
        _row(f"{label}_{i}.wav", label)
        for label in LABELS
        for i in range(100)
    ]

    splits = split_dataset(
        rows,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    for label in LABELS:
        train_count = sum(
            row["label"] == label
            for row in splits["train"]
        )
        val_count = sum(
            row["label"] == label
            for row in splits["val"]
        )
        test_count = sum(
            row["label"] == label
            for row in splits["test"]
        )

        assert train_count == 70
        assert val_count == 15
        assert test_count == 15