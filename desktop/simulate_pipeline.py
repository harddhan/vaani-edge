from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from desktop.desktop_kws_client import run_simulation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the VAANI desktop pipeline."
    )

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
        "--model",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--force-trigger",
        action="store_true",
    )

    parser.add_argument(
        "--trigger-frame-index",
        type=int,
        default=None,
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
        run_simulation(
            wav_path=args.wav,
            server_uri=args.server_uri,
            force_trigger=args.force_trigger,
            trigger_frame_index=args.trigger_frame_index,
            model_path=args.model,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())