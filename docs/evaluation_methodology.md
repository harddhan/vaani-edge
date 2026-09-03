# Evaluation Methodology

Accuracy is necessary but not sufficient for a continuously listening keyword-spotting system. Evaluation therefore covers classification quality, false activations, threshold behavior, and system latency.

## Classification metrics

Report:

- Overall accuracy
- Per-class precision, recall, and F1
- Confusion matrix
- Keyword true positives and false negatives
- Keyword-vs-rest precision and recall
- False-positive behavior at the selected operating threshold

The current dataset uses four classes:

`speech`, `noise`, `silence`, `vaani`

## Current development result

The current trained DS-CNN model was evaluated on a held-out test split of 315 samples.

- Test accuracy: **98.73%**
- `vaani`: **15/15** correct
- INT8 test accuracy: **98.73%**
- Keras/INT8 prediction agreement: **100%**

These are dataset results, not hardware deployment measurements.

## Threshold evaluation

`ml/training/threshold_sweep.py` evaluates combinations of:

- Threshold: `0.50`, `0.60`, `0.70`, `0.80`, `0.90`
- Consecutive positive windows: `1`, `2`, `3`, `4`

The current configured operating point is:

- Threshold: `0.80`
- Consecutive positive windows: `3`
- Cooldown: `1500 ms`

The final operating point should be selected using the false-activation and recall trade-off on representative validation data.

## False activations

False activations per hour may be estimated from a finite held-out negative dataset, but that estimate is not equivalent to a real continuous listening test.

Before claiming near-zero false activations, run the actual firmware or a real-microphone desktop simulation continuously for an extended period in representative acoustic conditions and log every trigger.

## Data quality

The evaluation set should contain variation in:

- Speakers
- Distance from microphone
- Speaking volume
- Speaking speed
- Rooms
- Background noise
- Non-keyword speech

Avoid allowing near-duplicate recordings of the same utterance to cross train/validation/test boundaries.

## Quantized model validation

`ml/quantization/evaluate_quantized.py` compares the Keras model and INT8 TFLite model on the same held-out test data.

The current INT8 artifact:

- Input: `int8`
- Output: `int8`
- Model size: `38,712` bytes
- Keras/INT8 accuracy difference: `0`
- Prediction agreement: `100%`

## System-level evaluation

Hardware validation must additionally measure:

- Peak/minimum free RAM
- Tensor arena usage
- Idle CPU
- Feature extraction time
- KWS inference time
- Keyword detection delay
- Network latency
- End-to-end keyword-end to server audio receipt latency
- Real false activations over an extended listening run
