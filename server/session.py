from __future__ import annotations

import dataclasses
import time
import uuid

from server.audio_buffer import AudioBuffer


class SessionError(Exception):
    pass


@dataclasses.dataclass
class SessionTimestamps:
    first_chunk_monotonic: float | None = None
    last_chunk_monotonic: float | None = None
    asr_start_monotonic: float | None = None
    asr_end_monotonic: float | None = None
    session_end_monotonic: float | None = None


@dataclasses.dataclass
class ServerSession:
    session_id: bytes
    sample_rate_hz: int
    channels: int
    max_buffer_bytes: int

    buffer: AudioBuffer = dataclasses.field(init=False)
    timestamps: SessionTimestamps = dataclasses.field(
        default_factory=SessionTimestamps
    )
    closed: bool = False

    def __post_init__(self) -> None:
        if len(self.session_id) != 16:
            raise SessionError("invalid session id")

        if self.sample_rate_hz <= 0:
            raise SessionError("invalid sample rate")

        if self.channels <= 0:
            raise SessionError("invalid channel count")

        if self.max_buffer_bytes <= 0:
            raise SessionError("invalid maximum buffer size")

        self.buffer = AudioBuffer(self.max_buffer_bytes)

    @property
    def session_id_str(self) -> str:
        return str(uuid.UUID(bytes=self.session_id))

    def add_chunk(self, sequence: int, payload: bytes) -> None:
        if self.closed:
            raise SessionError("session is closed")

        if not payload:
            raise SessionError("empty audio chunk")

        self.buffer.append(sequence, payload)

        now = time.monotonic()

        if self.timestamps.first_chunk_monotonic is None:
            self.timestamps.first_chunk_monotonic = now

        self.timestamps.last_chunk_monotonic = now

    def reconstruct_audio(self) -> bytes:
        return self.buffer.reconstruct()

    def mark_asr_start(self) -> None:
        if self.timestamps.asr_start_monotonic is None:
            self.timestamps.asr_start_monotonic = time.monotonic()

    def mark_asr_end(self) -> None:
        if self.timestamps.asr_end_monotonic is None:
            self.timestamps.asr_end_monotonic = time.monotonic()

    def close(self) -> None:
        if self.closed:
            return

        self.closed = True
        self.timestamps.session_end_monotonic = time.monotonic()

    def latency_report(self) -> dict[str, float | int | dict]:
        report: dict[str, float | int | dict] = {
            "buffering_delay_ms": 0.0,
            "network_delay_ms": 0.0,
            "audio_stream_duration_ms": 0.0,
            "server_processing_delay_ms": 0.0,
            "asr_duration_ms": 0.0,
            "total_session_duration_ms": 0.0,
            "sequence_stats": dataclasses.asdict(self.buffer.stats),
        }

        first_chunk = self.timestamps.first_chunk_monotonic
        last_chunk = self.timestamps.last_chunk_monotonic
        asr_start = self.timestamps.asr_start_monotonic
        asr_end = self.timestamps.asr_end_monotonic
        session_end = self.timestamps.session_end_monotonic

        if first_chunk is not None and last_chunk is not None:
            report["audio_stream_duration_ms"] = (
                last_chunk - first_chunk
            ) * 1000.0

        if first_chunk is not None and asr_start is not None:
            report["server_processing_delay_ms"] = (
                asr_start - first_chunk
            ) * 1000.0

        if asr_start is not None and asr_end is not None:
            report["asr_duration_ms"] = (
                asr_end - asr_start
            ) * 1000.0

        if first_chunk is not None and session_end is not None:
            report["total_session_duration_ms"] = (
                session_end - first_chunk
            ) * 1000.0

        return report