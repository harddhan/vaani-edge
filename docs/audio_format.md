# Audio Format & Feature Extraction Spec

This document defines the audio and feature configuration used by the current Python training pipeline. Firmware must reproduce the same preprocessing before the embedded model can be trusted on-device.

## Raw audio

| Parameter | Value | Configuration |
|---|---|---|
| Sample rate | 16000 Hz | `configs/audio.yaml` |
| Channels | 1 (mono) | `configs/audio.yaml` |
| Sample format | signed 16-bit PCM | `configs/audio.yaml` |
| Endianness | little-endian | `configs/audio.yaml` |
| Streaming frame | 30 ms / 480 samples | `configs/audio.yaml` |
| Pre-roll | 800 ms / 12800 samples | `configs/audio.yaml` |

## Model features

The current model uses MFCC features derived from a log-Mel spectrogram.

| Parameter | Value | Configuration |
|---|---|---|
| Window length | 32 ms / 512 samples | `configs/model.yaml` |
| Hop length | 20 ms / 320 samples | `configs/model.yaml` |
| FFT size | 256 | `configs/model.yaml` |
| Mel bins | 40 | `configs/model.yaml` |
| Frequency range | 300 Hz - 8000 Hz | `configs/model.yaml` |
| Window | Hann | `ml/features/mel_spectrogram.py` |
| Pre-emphasis | 0.98 | `configs/model.yaml` |
| Log floor | `1e-10` | `ml/features/mel_spectrogram.py` |
| MFCC coefficients | 13 | `configs/model.yaml` |
| Fixed frame count | 50 | `configs/model.yaml` |
| Model input | `50 x 13 x 1` | `configs/model.yaml` |
| Normalization | train-set per-feature mean/std | `configs/model.yaml` |

The current feature pipeline is:

```text
PCM16
  -> float32
  -> pre-emphasis
  -> framing + Hann window
  -> FFT power spectrum
  -> 40-band log-Mel spectrogram
  -> DCT-II
  -> first 13 MFCC coefficients
  -> train-set mean/std normalization
  -> 50 x 13 x 1 tensor
```

## Reference implementation

Python reference:

`ml/features/mel_spectrogram.py`
`ml/features/audio_features.py`

Training normalization:

`ml/training/dataset_loader.py`

Firmware implementation:

`firmware/esp32s3/components/feature_extractor/`

## Numerical parity requirement

The firmware feature extractor must be compared against the Python implementation using identical PCM input before the model is considered hardware-validated.

Small floating-point differences are acceptable. Large or systematic differences are not.

A direct firmware-vs-Python tensor comparison should be added as part of hardware integration.
