"""Basic offline data augmentation for the keyword dataset.

Implements a small set of well-understood, cheap augmentations:
    - Background noise mixing at a target SNR.
    - Random time shift.
    - Random gain (volume) perturbation.

This is intentionally simple. Do not add heavy augmentation pipelines
(e.g. SpecAugment, pitch-shifting libraries with native deps) unless the
team has a specific accuracy problem that calls for it - keep the system
easy to reason about for the interview defense.

Usage:
    python -m ml.dataset.augment_audio \
        --input data/processed/all/keyword \
        --noise data/processed/all/noise \
        --output data/processed/augmented/keyword \
        --num-augmentations 2
"""
from __future__ import annotations

import argparse
import random
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def mix_with_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / max(len(noise), 1)))
        noise = np.tile(noise, repeats)
    start = rng.randint(0, max(len(noise) - len(audio), 0))
    noise_segment = noise[start:start + len(audio)]

    audio_power = np.mean(audio ** 2) + 1e-12
    noise_power = np.mean(noise_segment ** 2) + 1e-12
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return audio + noise_segment * scale


def random_time_shift(audio: np.ndarray, max_shift_samples: int, rng: random.Random) -> np.ndarray:
    shift = rng.randint(-max_shift_samples, max_shift_samples)
    return np.roll(audio, shift)


def random_gain(audio: np.ndarray, min_db: float, max_db: float, rng: random.Random) -> np.ndarray:
    gain_db = rng.uniform(min_db, max_db)
    gain = 10 ** (gain_db / 20)
    return audio * gain


def augment_file(
    audio: np.ndarray,
    sample_rate: int,
    noise_files: list[Path],
    rng: random.Random,
) -> np.ndarray:
    out = audio.copy()
    out = random_time_shift(out, max_shift_samples=int(0.05 * sample_rate), rng=rng)
    out = random_gain(out, -6.0, 6.0, rng)
    if noise_files:
        noise_path = rng.choice(noise_files)
        noise, noise_sr = read_wav(noise_path)
        if noise_sr == sample_rate:
            snr_db = rng.uniform(5.0, 20.0)
            out = mix_with_noise(out, noise, snr_db, rng)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory of WAV files to augment")
    parser.add_argument("--noise", type=Path, default=None, help="Directory of noise WAV files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-augmentations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"[ERROR] Input directory not found: {args.input}")
        return 2

    rng = random.Random(args.seed)
    noise_files = sorted(args.noise.rglob("*.wav")) if args.noise and args.noise.exists() else []
    if args.noise and not noise_files:
        print(f"[WARN] No noise files found in {args.noise}; noise mixing will be skipped.")

    wav_files = sorted(args.input.rglob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No WAV files found in {args.input}")
        return 1

    count = 0
    for wav_path in wav_files:
        audio, sample_rate = read_wav(wav_path)
        for i in range(args.num_augmentations):
            augmented = augment_file(audio, sample_rate, noise_files, rng)
            out_path = args.output / f"{wav_path.stem}_aug{i}.wav"
            write_wav(out_path, augmented, sample_rate)
            count += 1

    print(f"Wrote {count} augmented files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
