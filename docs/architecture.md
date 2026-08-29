# Vaani Architecture

## system overview

Vaani is designed as a local voice activation pipeline where audio is captured, processed and classified directly on the edge device. The system should keep the path from microphone input to wake word detection as short and efficient as possible.

The overall flow is:

```text
Microphone
    ↓
Audio Capture
    ↓
Audio Buffer
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
TinyML Inference
    ↓
Decision
    ↓
Trigger
```

The exact hardware and model will be finalized after testing different options.

## 1. audio acquisition

The microphone continuously provides audio to the edge device. The audio acquisition layer is responsible for configuring the microphone interface, collecting samples at the required sampling rate and maintaining the buffers needed by the processing pipeline.

The choice of microphone and interface will depend on factors such as audio quality, hardware compatibility, power consumption and implementation complexity.

## 2. audio buffering

Incoming audio needs to be stored temporarily before it can be processed. Vaani will use a small rolling audio buffer so that the system can examine recent audio without storing unnecessary amounts of data.

The buffer size and processing window will be selected based on the requirements of the final ML model.

## 3. preprocessing

The captured audio will be prepared before feature extraction and inference. Depending on the final approach, this stage may include operations such as normalization, filtering, framing and windowing.

The preprocessing pipeline should remain lightweight because it will run continuously on the edge device.

## 4. feature extraction

Raw audio is usually not the most efficient input representation for a small ML model. Vaani will therefore extract useful characteristics from the audio before inference.

The final feature representation will be selected after comparing suitable approaches for accuracy, computational cost, memory usage and latency.

## 5. TinyML inference

The extracted features are passed to a lightweight machine learning model running directly on the edge device.

The model will classify the incoming audio and estimate whether the target wake word is present. The model should be small enough to fit within the available memory and computational limits of the selected hardware.

## 6. decision layer

The model output should not necessarily result in an immediate trigger from a single prediction. Vaani can use a decision layer to interpret model confidence and reduce accidental activations.

The exact decision logic will be determined through experimentation with different thresholds and detection strategies.

## 7. trigger

When the system determines that the target wake word has been detected, it generates a trigger event.

The trigger could be used to start another process or control an external function depending on the final prototype design.

## data flow

The system can be viewed as three main sections:

```text
INPUT
Microphone
    ↓
Audio samples

PROCESSING
Audio buffer
    ↓
Preprocessing
    ↓
Feature extraction
    ↓
TinyML inference

OUTPUT
Wake word detected
    ↓
Trigger action
```

The main design goal is to keep this entire path local to the device.

## performance considerations

Every stage contributes to the final system performance. A model that is accurate but slow may not be suitable, while a very small model that produces too many false activations is also not useful.

Vaani will therefore evaluate the complete pipeline rather than optimizing only the ML model.

Important measurements include:

- End to end detection latency
- Inference time
- RAM usage
- Model size
- Audio processing cost
- Power consumption
- Detection accuracy
- False activations
- Performance under background noise

## architecture decisions

The architecture is intentionally not completely fixed at this stage. Hardware, audio processing parameters and model design will be selected based on experiments.

Any major changes to the architecture and the reasons behind them will be documented as development continues.

## current architecture status

The current architecture is a proposed baseline and will evolve as the first software and hardware experiments are completed.
