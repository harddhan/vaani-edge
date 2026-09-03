from __future__ import annotations

import asyncio
import logging
import time

import numpy as np
import websockets

from server.protocol import (
    HEADER_SIZE,
    MAX_PAYLOAD_BYTES,
    MessageType,
    decode_header,
    encode_message,
    new_session_id,
)

logger = logging.getLogger("vaani.desktop.protocol")


class StreamingSessionClient:
    def __init__(
        self,
        server_uri: str,
        sample_rate_hz: int = 16000,
        channels: int = 1,
        connect_timeout_s: float = 5.0,
        reconnect_backoff_s: list[float] | None = None,
    ) -> None:
        self.server_uri = server_uri
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels
        self.connect_timeout_s = connect_timeout_s
        self.reconnect_backoff_s = reconnect_backoff_s or [1, 2, 5]

    async def _connect_with_retry(self):
        last_error: Exception | None = None

        for backoff in [0.0] + self.reconnect_backoff_s:
            if backoff:
                await asyncio.sleep(backoff)

            try:
                return await asyncio.wait_for(
                    websockets.connect(
                        self.server_uri,
                        max_size=None,
                    ),
                    timeout=self.connect_timeout_s,
                )
            except Exception as exc:
                last_error = exc

        raise ConnectionError(
            f"Could not connect to {self.server_uri}: {last_error}"
        )

    async def stream_utterance(
        self,
        pcm_int16: np.ndarray,
        frame_size_samples: int,
    ) -> str:
        session_id = new_session_id()
        websocket = await self._connect_with_retry()

        try:
            await websocket.send(
                encode_message(
                    msg_type=MessageType.START_SESSION,
                    session_id=session_id,
                    sequence_number=0,
                    timestamp_ms=int(time.time() * 1000),
                    sample_rate_hz=self.sample_rate_hz,
                    channels=self.channels,
                )
            )

            max_samples = min(
                frame_size_samples,
                MAX_PAYLOAD_BYTES // 2,
            )

            pcm_bytes = np.asarray(
                pcm_int16,
                dtype="<i2",
            ).tobytes()

            sequence_number = 0
            offset = 0

            while offset < len(pcm_bytes):
                chunk = pcm_bytes[
                    offset:offset + max_samples * 2
                ]

                await websocket.send(
                    encode_message(
                        msg_type=MessageType.AUDIO_CHUNK,
                        session_id=session_id,
                        sequence_number=sequence_number,
                        timestamp_ms=int(time.time() * 1000),
                        payload=chunk,
                        sample_rate_hz=self.sample_rate_hz,
                        channels=self.channels,
                    )
                )

                sequence_number += 1
                offset += len(chunk)

            await websocket.send(
                encode_message(
                    msg_type=MessageType.END_SESSION,
                    session_id=session_id,
                    sequence_number=sequence_number,
                    timestamp_ms=int(time.time() * 1000),
                )
            )

            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.connect_timeout_s * 4,
            )

            header = decode_header(response)

            payload = response[
                HEADER_SIZE:HEADER_SIZE + header.payload_length
            ]

            text = payload.decode("utf-8", errors="replace")

            if header.msg_type == MessageType.ASR_RESULT:
                return text

            if header.msg_type == MessageType.ERROR:
                return f"[ERROR] {text}"

            return f"[UNEXPECTED_RESPONSE] {text}"

        finally:
            await websocket.close()