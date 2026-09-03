from __future__ import annotations

import dataclasses
from pathlib import Path

from ml.features.feature_spec import load_config as load_yaml_config


class ConfigError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    max_message_size_bytes: int
    max_session_buffer_bytes: int
    session_timeout_s: float
    debug_save_wav: bool
    debug_wav_dir: Path
    reports_dir: Path
    asr_backend: str
    asr_model_size: str


def load_server_config(config_dir: Path | None = None) -> ServerConfig:
    raw = load_yaml_config(config_dir)

    network = raw.get("network", {})
    paths = raw.get("paths", {})
    asr = raw.get("asr", {})

    host = network.get("server_host")
    port = network.get("server_port")

    if not isinstance(host, str) or not host.strip():
        raise ConfigError("server_host must be a non-empty string")

    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("server_port must be between 1 and 65535")

    max_message_size = int(
        network.get("max_message_size_bytes", 65536)
    )
    max_session_buffer = int(
        network.get("max_session_buffer_bytes", 8 * 1024 * 1024)
    )
    session_timeout = float(
        network.get("session_timeout_s", 15)
    )

    if max_message_size < 44:
        raise ConfigError("max_message_size_bytes must be at least 44")

    if max_session_buffer <= 0:
        raise ConfigError("max_session_buffer_bytes must be positive")

    if session_timeout <= 0:
        raise ConfigError("session_timeout_s must be positive")

    backend = str(asr.get("backend", "mock")).lower()
    model_size = str(asr.get("model_size", "tiny"))

    if backend not in {"mock", "local_whisper"}:
        raise ConfigError(f"unsupported ASR backend: {backend}")

    reports_dir = Path(paths.get("reports_dir", "reports"))
    debug_wav_dir = reports_dir / "debug_wav"

    return ServerConfig(
        host=host,
        port=port,
        max_message_size_bytes=max_message_size,
        max_session_buffer_bytes=max_session_buffer,
        session_timeout_s=session_timeout,
        debug_save_wav=True,
        debug_wav_dir=debug_wav_dir,
        reports_dir=reports_dir,
        asr_backend=backend,
        asr_model_size=model_size,
    )