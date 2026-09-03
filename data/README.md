# Dataset

The dataset is not included in the repository. Keep raw recordings and generated processed audio outside version control.

## Structure

```text
data/
├── raw/
│   ├── speech/
│   ├── noise/
│   ├── silence/
│   └── vaani/
└── processed/
```

### Classes

| Class     | Purpose                            |
| --------- | ---------------------------------- |
| `speech`  | Normal speech                      |
| `noise`   | Environmental and background noise |
| `silence` | Silent or near silent audio        |
| `vaani`   | Custom keyword recordings          |

The `funny` class is intentionally not part of the initial model.

## Audio Format

Raw recordings can use different sample rates and channel formats.

The dataset preparation pipeline converts them to:

* 16 kHz
* Mono
* 16 bit PCM
* 1 second per sample

## Dataset Preparation

Run:

```powershell
python -m ml.dataset.prepare_dataset
```

This creates the processed dataset and manifest under `data/processed`.

## Dataset Splitting

Create deterministic train, validation, and test splits:

```powershell
python -m ml.dataset.split_dataset
```

The current split uses:

```text
70% training
15% validation
15% testing
```

## Validation

Check the audio files with:

```powershell
python -m ml.dataset.validate_wavs
```

## Current Dataset

The current development dataset contains:

```text
Speech:  1000
Noise:    500
Silence:  500
Vaani:    100
```

The Vaani class will be expanded with more speakers, recording conditions, distances, speaking styles, and background environments.

## Data Collection

For reliable keyword detection, Vaani recordings should vary across:

* Different speakers
* Microphone distances
* Speaking volume
* Speaking speed
* Background noise
* Room environments

Additional negative samples should include normal speech, environmental sounds, and silence to reduce false activations.

## Version Control

Do not commit:

* Raw audio recordings
* Processed audio
* Generated dataset manifests containing local paths
* Large model artifacts

The repository contains the scripts required to recreate the dataset pipeline.
