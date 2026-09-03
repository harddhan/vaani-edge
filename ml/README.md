# ML Pipeline

VAANI's machine-learning pipeline covers dataset preparation, feature extraction, model training, evaluation, INT8 quantization, and generation of the model array used by the ESP32-S3 firmware.

## Pipeline

```text
Raw WAV files
    ↓
Dataset validation
    ↓
Dataset preparation
    ↓
Train / validation / test split
    ↓
MFCC feature extraction
    ↓
Train DS-CNN
    ↓
Evaluate
    ↓
INT8 quantization
    ↓
Quantized-model evaluation
    ↓
Generate C model array
```

## Dataset

The current development dataset contains four classes:

```text
speech
noise
silence
vaani
```

Current counts:

```text
speech   1000
noise     500
silence   500
vaani     100
----------------
total    2100
```

The dataset is split 70/15/15:

```text
train    1470
validation 315
test      315
```

Audio is prepared as 16 kHz, mono, signed 16-bit PCM.

The dataset itself is not committed to the repository.

## Dataset commands

Validate raw WAV files:

```bash
python -m ml.dataset.validate_wavs
```

Prepare the dataset:

```bash
python -m ml.dataset.prepare_dataset
```

Create train/validation/test splits:

```bash
python -m ml.dataset.split_dataset
```

Optional offline augmentation:

```bash
python -m ml.dataset.augment_audio \
    --input data/processed/vaani \
    --noise data/processed/noise \
    --output data/processed/augmented/vaani \
    --num-augmentations 2
```

Augmentation is optional and should only be used when additional robustness is required.

## Feature extraction

The current model uses MFCC features.

```text
16 kHz mono PCM
    ↓
Pre-emphasis
    ↓
512-sample Hann windows
    ↓
256-point FFT
    ↓
40-band log-Mel spectrogram
    ↓
DCT-II
    ↓
13 MFCC coefficients
    ↓
Train-set mean/std normalization
    ↓
50 × 13 × 1 model input
```

Configuration is defined in:

```text
configs/model.yaml
configs/audio.yaml
```

Python reference implementation:

```text
ml/features/mel_spectrogram.py
ml/features/audio_features.py
ml/features/feature_spec.py
```

## Model

The primary model is a depthwise-separable CNN:

```text
DS-CNN
Input: 50 × 13 × 1
Output: 4 classes
```

The classes are ordered:

```text
0 = speech
1 = noise
2 = silence
3 = vaani
```

Model implementation:

```text
ml/models/ds_cnn.py
```

A smaller conventional CNN is retained as a baseline for architecture comparison:

```text
ml/models/small_cnn.py
```

Inspect a model before training:

```bash
python -m ml.models.model_summary --architecture ds_cnn
```

## Training

Train the current DS-CNN:

```bash
python -m ml.training.train --architecture ds_cnn
```

The training pipeline:

* Uses train-set feature normalization statistics.
* Applies the same normalization to validation and test data.
* Uses class-weighted training.
* Saves the best checkpoint.
* Uses early stopping.
* Evaluates the final restored model on the held-out test set.

Generated artifacts are stored under:

```text
ml/artifacts/
```

## Current development result

The current DS-CNN model achieved:

```text
Test accuracy: 98.73%
Test samples: 315
Vaani samples: 15/15 correct
```

These are held-out development-dataset results. They are not a guarantee of real-world or hardware performance.

## Evaluation

Run the standard evaluation:

```bash
python -m ml.training.evaluate \
    --model ml/artifacts/ds_cnn_final.keras
```

Run the threshold sweep:

```bash
python -m ml.training.threshold_sweep \
    --model ml/artifacts/ds_cnn_final.keras
```

The current trigger configuration is:

```text
threshold: 0.80
consecutive positive windows: 3
cooldown: 1500 ms
```

False-activation estimates should be treated as development metrics. A real long-duration microphone test is required before claiming near-zero false activations.

## INT8 quantization

Convert the trained Keras model to a fully INT8 TFLite model:

```bash
python -m ml.quantization.convert_int8 \
    --model ml/artifacts/ds_cnn_final.keras
```

The conversion uses the training normalization statistics and a representative dataset.

Evaluate the quantized model:

```bash
python -m ml.quantization.evaluate_quantized \
    --float32-model ml/artifacts/ds_cnn_final_float32.tflite \
    --int8-model ml/artifacts/ds_cnn_final_int8.tflite
```

Current INT8 result:

```text
Float32 accuracy: 98.73%
INT8 accuracy:    98.73%
Prediction agreement: 100%
Accuracy difference: 0
```

Current INT8 model size:

```text
38,712 bytes
```

The model file size is a Flash/storage measurement. It is not the total runtime RAM requirement.

## Generate firmware model array

Generate the C++ model array used by the firmware:

```bash
python -m ml.quantization.generate_model_cc \
    --int8-model ml/artifacts/ds_cnn_final_int8.tflite
```

Output:

```text
firmware/esp32s3/components/kws_inference/model/kws_model_data.h
firmware/esp32s3/components/kws_inference/model/kws_model_data.cc
```

The generated array contains the INT8 TFLite model.

## Artifacts

Typical generated artifacts include:

```text
ml/artifacts/
├── ds_cnn_best.keras
├── ds_cnn_final.keras
├── ds_cnn_final_float32.tflite
├── ds_cnn_final_int8.tflite
├── ds_cnn_final_quant_metadata.json
├── ds_cnn_history.json
├── ds_cnn_test_report.txt
└── normalization.json
```

These generated artifacts are ignored by Git by default.

## Reproducibility

The main configuration lives in:

```text
configs/model.yaml
configs/audio.yaml
```

The training seed is configured in `configs/model.yaml`.

For a clean reproduction, use the same dataset, configuration, and software environment. Dataset contents are intentionally excluded from version control.

## Hardware boundary

The Python ML pipeline is the current development and validation environment.

The ESP32-S3 firmware remains the target deployment environment, but the complete embedded inference path still requires hardware integration and validation.

In particular, the following must be validated before claiming deployment readiness:

* Firmware feature numerical parity with Python
* INT8 inference on the target board
* Tensor arena usage
* Total RAM usage
* Idle CPU
* End-to-end latency
* Extended false-activation behavior
  "" end="0"
  ::
