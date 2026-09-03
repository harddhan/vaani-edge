```python
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import websockets
from websockets.server import WebSocketServerProtocol

from server.asr.base import ASRBackend, ASRUnavailableError
from server.asr.local_backend import create_local_backend
from server.asr.mock_backend import MockASRBackend
from server.config import ServerConfig, load_server_config
from server.metrics import MetricsCollector
from server.protocol import (
    MAX_PAYLOAD_BYTES,
    MessageType,
    ProtocolError,
    SampleFormat,
    decode_message,
    encode_message,
)
from server.session import ServerSession, SessionError
from server.storage.wav_writer import pcm_bytes_to_float32, write_debug_wav

logger = logging.getLogger("vaani.server")


class UnavailableASRBackend(ASRBackend):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def transcribe(self, audio, sample_rate: int) -> str:
        raise ASRUnavailableError(str(self.error))


def build_asr_backend(config: ServerConfig) -> ASRBackend:
    if config.asr_backend == "mock":
        return MockASRBackend()

    if config.asr_backend == "local_whisper":
        try:
            return create_local_backend(model_size=config.asr_model_size)
        except ASRUnavailableError as exc:
            logger.warning("Local ASR unavailable: %s", exc)
            return UnavailableASRBackend(exc)

    raise ValueError(f"Unknown asr backend: {config.asr_backend}")


class AudioStreamServer:
    def __init__(
        self,
        config: ServerConfig,
        asr_backend: ASRBackend,
    ) -> None:
        self.config = config
        self.asr_backend = asr_backend
        self.metrics = MetricsCollector(config.reports_dir)
        self._sessions: dict[str, ServerSession] = {}

    async def handle_connection(
        self,
        websocket: WebSocketServerProtocol,
    ) -> None:
        peer = websocket.remote_address
        logger.info("Client connected: %s", peer)
        session: ServerSession | None = None

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=self.config.session_timeout_s,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Session timed out for %s", peer)
                    if session is not None:
                        await self._finalize_session(websocket, session)
                        session = None
                    break

                if isinstance(message, str):
                    await self._send_error(
                        websocket,
                        "Text frames are not supported; use binary frames.",
                    )
                    continue

                if len(message) > self.config.max_message_size_bytes:
                    await self._send_error(
                        websocket,
                        "Message exceeds max_message_size_bytes.",
                    )
                    continue

                try:
                    header, payload = decode_message(message)
                except ProtocolError as exc:
                    await self._send_error(
                        websocket,
                        f"Protocol error: {exc}",
                    )
                    continue

                if len(payload) > MAX_PAYLOAD_BYTES:
                    await self._send_error(
                        websocket,
                        "Payload exceeds protocol limit.",
                    )
                    continue

                if header.msg_type == MessageType.START_SESSION:
                    if session is not None:
                        await self._send_error(
                            websocket,
                            "A session is already active.",
                        )
                        continue

                    if payload:
                        await self._send_error(
                            websocket,
                            "START_SESSION payload must be empty.",
                        )
                        continue

                    if (
                        header.sample_format != SampleFormat.PCM_S16_LE
                        or header.channels != 1
                        or header.sample_rate_hz != 16000
                    ):
                        await self._send_error(
                            websocket,
                            "Audio format must be 16 kHz mono PCM S16 LE.",
                        )
                        continue

                    try:
                        session = ServerSession(
                            session_id=header.session_id,
                            sample_rate_hz=header.sample_rate_hz,
                            channels=header.channels,
                            max_buffer_bytes=self.config.max_session_buffer_bytes,
                        )
                    except SessionError as exc:
                        await self._send_error(
                            websocket,
                            f"Invalid session: {exc}",
                        )
                        continue

                    self._sessions[session.session_id_str] = session

                    logger.info(
                        "Session started: %s",
                        session.session_id_str,
                    )

                elif header.msg_type == MessageType.AUDIO_CHUNK:
                    if session is None:
                        await self._send_error(
                            websocket,
                            "AUDIO_CHUNK received with no active session.",
                        )
                        continue

                    if header.session_id != session.session_id:
                        await self._send_error(
                            websocket,
                            "AUDIO_CHUNK session ID does not match active session.",
                        )
                        continue

                    if (
                        header.sample_rate_hz != session.sample_rate_hz
                        or header.channels != session.channels
                        or header.sample_format != SampleFormat.PCM_S16_LE
                    ):
                        await self._send_error(
                            websocket,
                            "AUDIO_CHUNK audio format does not match the session.",
                        )
                        continue

                    try:
                        session.add_chunk(
                            header.sequence_number,
                            payload,
                        )
                    except SessionError as exc:
                        await self._send_error(
                            websocket,
                            f"Failed to buffer audio: {exc}",
                        )
                        await self._finalize_session(websocket, session)
                        session = None
                    except Exception as exc:
                        await self._send_error(
                            websocket,
                            f"Failed to buffer audio: {exc}",
                        )
                        await self._finalize_session(websocket, session)
                        session = None

                elif header.msg_type == MessageType.END_SESSION:
                    if session is None:
                        await self._send_error(
                            websocket,
                            "END_SESSION received with no active session.",
                        )
                        continue

                    if header.session_id != session.session_id:
                        await self._send_error(
                            websocket,
                            "END_SESSION session ID does not match active session.",
                        )
                        continue

                    if payload:
                        await self._send_error(
                            websocket,
                            "END_SESSION payload must be empty.",
                        )
                        continue

                    await self._finalize_session(
                        websocket,
                        session,
                    )
                    session = None

                else:
                    await self._send_error(
                        websocket,
                        f"Unexpected message type from client: {header.msg_type.name}",
                    )

        except websockets.ConnectionClosed:
            logger.info("Connection closed: %s", peer)

            if session is not None and not session.closed:
                session.close()
                self._sessions.pop(
                    session.session_id_str,
                    None,
                )
                logger.warning(
                    "Client disconnected mid-session: %s",
                    session.session_id_str,
                )

        finally:
            logger.info("Client disconnected: %s", peer)

    async def _send_error(
        self,
        websocket: WebSocketServerProtocol,
        message: str,
        session_id: bytes = b"\x00" * 16,
    ) -> None:
        logger.warning("Sending ERROR to client: %s", message)

        frame = encode_message(
            msg_type=MessageType.ERROR,
            session_id=session_id,
            sequence_number=0,
            timestamp_ms=int(time.time() * 1000),
            payload=message.encode("utf-8"),
        )

        await websocket.send(frame)

    async def _finalize_session(
        self,
        websocket: WebSocketServerProtocol,
        session: ServerSession,
    ) -> None:
        if session.closed:
            return

        pcm_bytes = session.reconstruct_audio()

        logger.info(
            "Session %s complete: %d bytes, stats=%s",
            session.session_id_str,
            len(pcm_bytes),
            session.buffer.stats,
        )

        if self.config.debug_save_wav and pcm_bytes:
            debug_path = (
                self.config.debug_wav_dir
                / f"{session.session_id_str}.wav"
            )

            write_debug_wav(
                debug_path,
                pcm_bytes,
                session.sample_rate_hz,
                session.channels,
            )

            logger.info(
                "Saved debug WAV: %s",
                debug_path,
            )

        transcript = ""
        error_message = ""

        session.mark_asr_start()

        try:
            audio_float = pcm_bytes_to_float32(pcm_bytes)
            
        finally:
            session.mark_asr_end()
        session.close()

latency_report = session.latency_report()

            transcript = await self.asr_backend.transcribe(
                audio_float,
                session.sample_rate_hz,
            )

        except ASRUnavailableError as exc:
            error_message = f"ASR unavailable: {exc}"

        except Exception as exc:
            error_message = f"ASR failed: {exc}"

        finally:
            session.mark_asr_end()

        session.close()

        latency_report = session.latency_report()
        self.metrics.record_session(latency_report)

        if error_message:
            await self._send_error(
                websocket,
                error_message,
                session.session_id,
            )
        else:
            frame = encode_message(
                msg_type=MessageType.ASR_RESULT,
                session_id=session.session_id,
                sequence_number=0,
                timestamp_ms=int(time.time() * 1000),
                payload=transcript.encode("utf-8"),
                sample_rate_hz=session.sample_rate_hz,
                channels=session.channels,
            )

            await websocket.send(frame)

        self._sessions.pop(
            session.session_id_str,
            None,
        )

        self.metrics.write_report()


async def run_server(config: ServerConfig) -> None:
    asr_backend = build_asr_backend(config)

    server = AudioStreamServer(
        config,
        asr_backend,
    )

    logger.info(
        "Starting server on %s:%d (max_size=%d)",
        config.host,
        config.port,
        config.max_message_size_bytes,
    )

    async with websockets.serve(
        server.handle_connection,
        config.host,
        config.port,
        max_size=config.max_message_size_bytes,
    ):
        await asyncio.Future()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
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

    config = load_server_config(args.config_dir)

    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
