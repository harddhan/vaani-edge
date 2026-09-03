from __future__ import annotations

import argparse
import csv
import wave
from pathlib import Path

import numpy as np

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


LABELS = ("speech", "noise", "silence", "vaani")
TARGET_SAMPLE_RATE = 16000
TARGET_SAMPLES = 16000


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit WAV")

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def resample(audio: np.ndarray, original_sr: int) -> np.ndarray:
    if original_sr == TARGET_SAMPLE_RATE:
        return audio

    if resample_poly is None:
        raise RuntimeError("scipy is required for resampling")

    from math import gcd

    divisor = gcd(original_sr, TARGET_SAMPLE_RATE)
    up = TARGET_SAMPLE_RATE // divisor
    down = original_sr // divisor

    return resample_poly(audio, up, down).astype(np.float32)


def fix_length(audio: np.ndarray) -> np.ndarray:
    if len(audio) == TARGET_SAMPLES:
        return audio

    if len(audio) > TARGET_SAMPLES:
        start = (len(audio) - TARGET_SAMPLES) // 2
        return audio[start:start + TARGET_SAMPLES]

    padding = TARGET_SAMPLES - len(audio)
    before = padding // 2
    after = padding - before

    return np.pad(audio, (before, after))


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())


def process_dataset(
    input_dir: Path,
    output_dir: Path,
) -> list[dict]:
    manifest = []

    for label in LABELS:
        label_dir = input_dir / label

        if not label_dir.exists():
            print(f"[WARN] Missing: {label_dir}")
            continue

        wav_files = sorted(label_dir.rglob("*.wav"))

        for wav_path in wav_files:
            try:
                audio, sample_rate = read_wav(wav_path)
                audio = resample(audio, sample_rate)
                audio = fix_length(audio)

                output_path = output_dir / "all" / label / wav_path.name
                write_wav(output_path, audio)

                manifest.append(
                    {
                        "path": str(output_path),
                        "label": label,
                        "source": str(wav_path),
                        "sample_rate": TARGET_SAMPLE_RATE,
                        "duration_s": 1.0,
                    }
                )

            except Exception as exc:
                print(f"[ERROR] {wav_path}: {exc}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input directory not found: {args.input}")
        return 2

    manifest = process_dataset(args.input, args.output)

    if not manifest:
        print("[ERROR] No WAV files processed.")
        return 1

    manifest_path = args.output / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest[0].keys(),
        )
        writer.writeheader()
        writer.writerows(manifest)

    counts = {}

    for row in manifest:
        counts[row["label"]] = counts.get(row["label"], 0) + 1

    print(f"\nProcessed {len(manifest)} files.")

    for label in LABELS:
        print(f"{label}: {counts.get(label, 0)}")

    print(f"\nManifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())