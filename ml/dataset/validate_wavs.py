"""Validate WAV files in the raw dataset directory before processing.

Usage:
    python -m ml.dataset.validate_wavs --input data/raw

Checks:
    - File is a readable WAV.
    - Duration is within a sane range (0.1s - 10s by default).
    - Not silent (RMS above a very low floor).
    - Reports sample rate / channel mismatches so they can be fixed by
      ``prepare_dataset.py`` (which resamples/mono-converts automatically).

This script does not modify files. It only reports problems.
"""
from __future__ import annotations

import argparse
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ValidationIssue:
    path: Path
    message: str


def _read_wav_as_float(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    if sample_width != 2:
        raise ValueError(f"Unsupported sample width {sample_width * 8}-bit (expected 16-bit)")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sample_rate, n_channels


def validate_file(
    path: Path,
    min_duration_s: float,
    max_duration_s: float,
    silence_rms_floor: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        audio, sample_rate, n_channels = _read_wav_as_float(path)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return [ValidationIssue(path, f"Failed to read WAV: {exc}")]

    duration_s = len(audio) / max(sample_rate, 1)
    if duration_s < min_duration_s:
        issues.append(ValidationIssue(path, f"Too short: {duration_s:.3f}s"))
    if duration_s > max_duration_s:
        issues.append(ValidationIssue(path, f"Too long: {duration_s:.3f}s"))

    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if rms < silence_rms_floor:
        issues.append(ValidationIssue(path, f"Likely silent, RMS={rms:.5f}"))

    if sample_rate != 16000:
        issues.append(ValidationIssue(path, f"Sample rate {sample_rate}Hz != 16000Hz (will be resampled)"))
    if n_channels != 1:
        issues.append(ValidationIssue(path, f"{n_channels} channels (will be downmixed to mono)"))

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--min-duration-s", type=float, default=0.1)
    parser.add_argument("--max-duration-s", type=float, default=10.0)
    parser.add_argument("--silence-rms-floor", type=float, default=0.001)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input directory does not exist: {args.input}", file=sys.stderr)
        return 2

    wav_files = sorted(args.input.rglob("*.wav"))
    if not wav_files:
        print(f"No .wav files found under {args.input}. Nothing to validate.")
        return 0

    total_issues = 0
    for wav_path in wav_files:
        issues = validate_file(
            wav_path, args.min_duration_s, args.max_duration_s, args.silence_rms_floor
        )
        for issue in issues:
            print(f"[ISSUE] {issue.path}: {issue.message}")
            total_issues += 1

    print(f"\nValidated {len(wav_files)} files, found {total_issues} issue(s).")
    return 1 if total_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
