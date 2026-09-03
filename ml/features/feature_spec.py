from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    sample_rate_hz: int = 16000
    n_fft: int = 256
    window_length_ms: int = 32
    hop_length_ms: int = 20
    n_mels: int = 40
    n_mfcc: int = 13
    fmin_hz: int = 300
    fmax_hz: int = 8000
    num_frames: int = 50
    pre_emphasis: float = 0.98

    @property
    def window_length_samples(self) -> int:
        return int(self.sample_rate_hz * self.window_length_ms / 1000)

    @property
    def hop_length_samples(self) -> int:
        return int(self.sample_rate_hz * self.hop_length_ms / 1000)

    @property
    def feature_dim(self) -> int:
        return self.n_mfcc

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.num_frames, self.feature_dim, 1)

    @property
    def expected_audio_samples(self) -> int:
        return (
            self.window_length_samples
            + (self.num_frames - 1) * self.hop_length_samples
        )

    @property
    def input_size(self) -> int:
        return self.num_frames * self.feature_dim

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_config(config_dir: Path | None = None) -> dict:
    if config_dir is None:
        config_dir = Path("configs")

    default_path = config_dir / "default.yaml"

    if not default_path.exists():
        raise FileNotFoundError(f"Config file not found: {default_path}")

    with default_path.open("r", encoding="utf-8") as fh:
        default_config = yaml.safe_load(fh) or {}

    merged = dict(default_config)

    for include_name in default_config.get("includes", []):
        include_path = config_dir / include_name

        if not include_path.exists():
            raise FileNotFoundError(f"Included config file not found: {include_path}")

        with include_path.open("r", encoding="utf-8") as fh:
            included = yaml.safe_load(fh) or {}

        merged = _deep_merge(merged, included)

    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def feature_spec_from_config(config: dict) -> FeatureSpec:
    features = config.get("features", {})

    return FeatureSpec(
        sample_rate_hz=int(features.get("sample_rate_hz", 16000)),
        n_fft=int(features.get("n_fft", 256)),
        window_length_ms=int(features.get("window_length_ms", 32)),
        hop_length_ms=int(features.get("hop_length_ms", 20)),
        n_mels=int(features.get("n_mels", 40)),
        n_mfcc=int(features.get("n_mfcc", 13)),
        fmin_hz=int(features.get("fmin_hz", 300)),
        fmax_hz=int(features.get("fmax_hz", 8000)),
        num_frames=int(features.get("num_frames", 50)),
        pre_emphasis=float(features.get("pre_emphasis", 0.98)),
    )