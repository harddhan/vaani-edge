from __future__ import annotations

import argparse
import asyncio
import logging
import time
import wave
from pathlib import Path

import numpy as np

from desktop.protocol_client import StreamingSessionClient


def load_pcm(
    wav_path: Path,
) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()

        if wf.getsampwidth() != 2:
            raise ValueError("Expected 16-bit PCM WAV")

        if wf.getnchannels() != 1:
            raise ValueError("Expected mono WAV")

        raw = wf.readframes(
            wf.getnframes()
        )

    return (
        np.frombuffer(
            raw,
            dtype="<i2",
        ),
        sample_rate,
    )


async def probe(
    wav_path: Path,
    server_uri: str,
    frame_size_samples: int,
) -> None:
    pcm, sample_rate = load_pcm(wav_path)

    client = StreamingSessionClient(
        server_uri,
        sample_rate_hz=sample_rate,
    )

    start = time.monotonic()

    transcript = await client.stream_utterance(
        pcm,
        frame_size_samples,
    )

    elapsed_ms = (
        time.monotonic() - start
    ) * 1000

    print(
        f"Round trip: {elapsed_ms:.1f} ms"
    )
    print(
        f"Transcript: {transcript}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--wav",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--server-uri",
        default="ws://127.0.0.1:8765",
    )

    parser.add_argument(
        "--frame-size-samples",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.wav.exists():
        logging.error(
            "WAV file not found: %s",
            args.wav,
        )
        return 2

    asyncio.run(
        probe(
            args.wav,
            args.server_uri,
            args.frame_size_samples,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())