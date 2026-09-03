# VAANI

**Low Latency and Efficient Voice Activator for Edge Devices**

VAANI is a TinyML based voice activation system designed for low power edge devices. It detects a custom keyword locally and, after activation, streams the subsequent audio to a remote Automatic Speech Recognition server over a local network.

The system is designed around the core requirements of the problem statement: low latency, low memory usage, low idle CPU utilization, custom keyword spotting, and efficient communication between the edge device and the ASR server.

## Problem

Cloud based voice activation can introduce unnecessary latency, network overhead, and privacy concerns. VAANI moves the initial voice activation step to the edge.

The edge device continuously listens for a custom keyword. Once the keyword is detected, the device starts sending the following audio to a remote ASR server for further processing.

The target runtime is a resource constrained microcontroller such as an ESP32, with the edge application designed around:

* Less than 256 KB RAM usage
* Less than 10% CPU utilization during idle listening
* Low latency keyword detection
* High keyword detection accuracy
* Very low false activation rate
* A custom trained keyword rather than a generic assistant keyword
* Open source ML and TinyML technologies

## System Architecture

```text
Microphone
    |
    v
Audio Capture
    |
    v
16 kHz Audio
    |
    v
MFCC Feature Extraction
    |
    v
TinyML Keyword Spotting Model
    |
    +---- No Keyword ----> Continue Listening
    |
    +---- VAANI Detected
              |
              v
        Trigger State
              |
              v
        Audio Streaming
              |
              v
        LAN / WebSocket
              |
              v
        ASR Server
              |
              v
       Speech Recognition
```

## Current ML Pipeline

The current development pipeline uses:

```text
WAV Audio
   |
   v
16 kHz Mono
   |
   v
Preprocessing
   |
   v
40-band Log Mel Spectrogram
   |
   v
13 MFCC Features
   |
   v
Normalization
   |
   v
Lightweight CNN
   |
   v
Keyword Classification
```

### Audio configuration

| Parameter         |       Value |
| ----------------- | ----------: |
| Sample rate       |      16 kHz |
| Channels          |        Mono |
| Audio length      |    1 second |
| Window length     |       32 ms |
| Frame stride      |       20 ms |
| FFT size          |         256 |
| Mel filters       |          40 |
| MFCC coefficients |          13 |
| Frequency range   | 300–8000 Hz |
| Pre emphasis      |        0.98 |
| Model input       | 50 × 13 × 1 |

The resulting feature representation contains **650 values per audio sample**.

## Keyword Classes

The initial model uses four classes:

| Class   | Purpose                            |
| ------- | ---------------------------------- |
| Speech  | Normal speech                      |
| Noise   | Environmental and non speech audio |
| Silence | Silent or near silent audio        |
| Vaani   | Custom activation keyword          |

The initial development dataset contains:

* Speech: 1000 samples
* Noise: 500 samples
* Silence: 500 samples
* Vaani: 100 samples

The Vaani dataset will be expanded as development continues.

## Model

The current primary model is a lightweight depthwise separable CNN designed for eventual TinyML deployment.

Current development model:

* Input: `50 × 13 × 1`
* Classes: 4
* Parameters: approximately 19K
* Architecture: convolution + depthwise separable convolution blocks
* Output: class probabilities

The model is intentionally small so that it can be optimized for microcontroller deployment.

## Current Baseline

The current PC development baseline achieved approximately **99.05% test accuracy** on the available 315 sample test split.

The result is a development baseline rather than a final real world performance claim. In particular, the current Vaani test set is small, so additional recordings and testing under different speakers, environments, microphones, and background conditions are required.

## Dataset Pipeline

Raw audio is organized as:

```text
data/
└── raw/
    ├── speech/
    ├── noise/
    ├── silence/
    └── vaani/
```

The dataset preparation pipeline:

1. Converts audio to mono
2. Resamples audio to 16 kHz
3. Normalizes each recording to the required length
4. Generates processed WAV files
5. Creates a dataset manifest
6. Creates deterministic train, validation, and test splits

The current split is:

```text
Train:      1470
Validation:  315
Test:        315
```

## Repository Structure

```text
VAANI/
├── configs/
├── data/
├── desktop/
├── docs/
├── firmware/
├── ml/
│   ├── dataset/
│   ├── features/
│   ├── inference/
│   ├── models/
│   ├── quantization/
│   └── training/
├── reports/
├── scripts/
├── server/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

Create the Python environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Dataset Preparation

Prepare the processed dataset:

```powershell
python -m ml.dataset.prepare_dataset
```

Create train, validation, and test splits:

```powershell
python -m ml.dataset.split_dataset
```

Validate the processed WAV files:

```powershell
python -m ml.dataset.validate_wavs
```

## Model Training

Train the current lightweight CNN:

```powershell
python -m ml.training.train
```

The training pipeline:

* Loads the dataset splits
* Extracts MFCC features
* Calculates normalization statistics from the training data
* Applies the same statistics to validation and test data
* Trains the keyword spotting model
* Evaluates the test set
* Saves the trained model and preprocessing information locally

Generated ML artifacts are kept outside version control.

## PC Inference

A trained model can be tested against an individual WAV file:

```powershell
python -m ml.inference.predict data\raw\vaani\vani_0000.wav
```

The inference pipeline uses the same feature extraction and normalization process as training.

## TinyML Deployment

The final deployment target is a resource constrained edge device such as an ESP32.

The planned runtime pipeline is:

```text
Microphone
    |
    v
Audio Buffer
    |
    v
Feature Extraction
    |
    v
INT8 TinyML Model
    |
    v
Keyword Detection
    |
    v
Trigger State Machine
    |
    v
LAN Audio Transport
```

The model will be quantized to an integer representation suitable for microcontroller inference and integrated into the firmware without relying on proprietary voice activation SDKs.

## LAN Communication

After the keyword is detected, the edge device should transition from continuous keyword listening to audio streaming.

The communication path is:

```text
ESP32
  |
  | audio stream
  v
LAN
  |
  v
Python ASR Server
  |
  v
Speech Recognition
```

The server side is intended to handle the computationally heavier speech recognition stage while the edge device performs the lightweight activation task.

## Development Status

### Completed

* Dataset preparation pipeline
* Audio preprocessing
* 16 kHz mono conversion
* MFCC feature extraction
* Dataset splitting
* Lightweight CNN models
* Training pipeline
* Train based normalization
* PC model inference
* Initial performance evaluation

### In Progress

* Larger Vaani dataset
* Robustness testing
* INT8 quantization
* Quantized model validation
* ESP32 audio capture
* ESP32 TinyML inference
* Trigger state machine
* LAN audio transport
* End to end latency measurement

### Future

* More speaker diversity
* More environmental noise conditions
* False activation testing
* RAM and Flash profiling
* Idle CPU profiling
* End to end latency profiling
* Hardware validation

## Design Goals

VAANI is ultimately evaluated against four practical goals:

**Accuracy**
Detect the custom keyword reliably while minimizing false activations.

**Efficiency**
Keep model memory, Flash usage, and idle computation suitable for a low power microcontroller.

**Latency**
Minimize the time between keyword detection and the ASR server receiving the subsequent audio.

**Deployability**
Keep the complete pipeline reproducible and suitable for deployment on resource constrained edge hardware.

## License

MIT License
