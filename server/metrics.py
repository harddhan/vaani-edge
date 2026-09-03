```python
from __future__ import annotations

import json
import time
from pathlib import Path


class MetricsCollector:
    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: list[dict] = []

    def record_session(self, latency_report: dict) -> None:
        self._sessions.append(dict(latency_report))

    def write_report(self, filename: str | None = None) -> Path:
        filename = filename or f"server_metrics_{int(time.time())}.json"

        path = self.reports_dir / filename

        with path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "sessions": self._sessions,
                    "summary": self.summary(),
                },
                fh,
                indent=2,
            )

        return path

    def summary(self) -> dict:
        if not self._sessions:
            return {"num_sessions": 0}

        def avg(key: str) -> float | None:
            values = [
                session[key]
                for session in self._sessions
                if session.get(key) is not None
            ]

            if not values:
                return None

            return round(sum(values) / len(values), 2)

        return {
            "num_sessions": len(self._sessions),
            "avg_buffering_delay_ms": avg(
                "buffering_delay_ms"
            ),
            "avg_audio_stream_duration_ms": avg(
                "audio_stream_duration_ms"
            ),
            "avg_server_processing_delay_ms": avg(
                "server_processing_delay_ms"
            ),
            "avg_asr_duration_ms": avg(
                "asr_duration_ms"
            ),
            "avg_total_session_duration_ms": avg(
                "total_session_duration_ms"
            ),
        }
```
