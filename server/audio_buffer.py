from __future__ import annotations

import dataclasses


class AudioBufferOverflowError(Exception):
    pass


@dataclasses.dataclass
class SequenceStats:
    expected_next: int = 0
    received: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    dropped_estimate: int = 0


class AudioBuffer:
    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")

        self._max_bytes = max_bytes
        self._chunks: dict[int, bytes] = {}
        self._stats = SequenceStats()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def stats(self) -> SequenceStats:
        return self._stats

    def append(self, sequence_number: int, payload: bytes) -> None:
        if not payload:
            raise ValueError("payload must not be empty")

        if sequence_number < 0:
            raise ValueError("sequence_number must not be negative")

        if sequence_number in self._chunks:
            self._stats.duplicates += 1
            return

        if self._total_bytes + len(payload) > self._max_bytes:
            raise AudioBufferOverflowError(
                f"buffer would exceed max_bytes={self._max_bytes} "
                f"(current={self._total_bytes}, incoming={len(payload)})"
            )

        if sequence_number < self._stats.expected_next:
            self._stats.out_of_order += 1
        elif sequence_number > self._stats.expected_next:
            self._stats.dropped_estimate += (
                sequence_number - self._stats.expected_next
            )

        self._chunks[sequence_number] = bytes(payload)
        self._total_bytes += len(payload)
        self._stats.received += 1
        self._stats.expected_next = max(
            self._stats.expected_next,
            sequence_number + 1,
        )

    def reconstruct(self) -> bytes:
        return b"".join(
            self._chunks[key]
            for key in sorted(self._chunks)
        )

    def clear(self) -> None:
        self._chunks.clear()
        self._total_bytes = 0
        self._stats = SequenceStats()

BoundedAudioBuffer = AudioBuffer
