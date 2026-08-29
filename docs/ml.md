# Vaani Machine Learning

## overview

The ML part of Vaani is responsible for detecting the predefined wake word from audio while running within the limitations of an edge device.

The goal is not to use the biggest or most complicated model possible. We want a model that gives a useful balance between detection performance, latency, memory usage, model size and power consumption.

The final model and framework will be selected after experiments.

## ML pipeline

The planned workflow is:

```text
Audio Dataset
      ↓
Data Preparation
      ↓
Audio Preprocessing
      ↓
Feature Extraction
      ↓
Model Training
      ↓
Evaluation
      ↓
Optimization
      ↓
Embedded Deployment
```

Each stage will be evaluated as part of the complete system rather than treating the ML model separately from the hardware.

## dataset

The dataset will contain examples of the target wake word as well as audio that should not activate the system.

The dataset may include:

- Target wake word recordings
- Different speakers
- Different speaking styles
- Negative examples
- Background noise
- Silence
- Similar sounding words or phrases
- Different recording conditions

The exact dataset sources, size and composition will be documented once the dataset strategy is finalized.

If an external dataset is used, its source and license will be documented before it is included in the project.

## data preparation

Raw audio may need to be cleaned, organized and converted into a consistent format before training.

The preparation process may include:

- Removing unusable recordings
- Checking audio format
- Resampling when required
- Splitting audio into suitable segments
- Creating training, validation and test sets
- Adding suitable background noise
- Maintaining speaker separation where appropriate

The exact preprocessing pipeline will depend on the selected model and hardware.

## audio features

The ML model will not necessarily operate directly on raw audio. Audio features can be extracted to provide a more compact representation of the information needed for wake word detection.

Possible approaches will be compared based on:

- Accuracy
- Computational cost
- Memory requirements
- Feature size
- Processing latency
- Suitability for embedded inference

The final feature representation will be chosen after experimentation.

## model development

A baseline model will be developed first so that we have something measurable to compare against.

The model will be evaluated not only on its accuracy but also on whether it can realistically run on the selected edge hardware.

Different lightweight model architectures may be tested if the baseline does not provide a good balance between performance and resource usage.

## model evaluation

The model will be evaluated using data that was not used during training.

Important metrics include:

- Accuracy
- Precision
- Recall
- False acceptance rate
- False rejection rate
- Confusion matrix

For a voice activation system, overall accuracy alone is not enough. A model that frequently activates when nobody said the wake word can be annoying even if its dataset accuracy looks good.

## optimization

Once a baseline model works, optimization will focus on making it suitable for the target hardware.

Possible areas include:

- Model size
- Quantization
- Feature size
- Inference time
- Memory usage
- Audio processing cost
- Number of parameters
- Power consumption

Optimization will be guided by measurements rather than assumptions.

## edge deployment

The final model will be converted into a format suitable for the selected embedded platform and integrated into the Vaani firmware.

The deployed pipeline should be able to process incoming audio, generate the required features, run inference and return a wake word decision without relying on a remote server.

The actual deployment method will depend on the selected hardware and ML framework.

## robustness

A wake word model needs to work outside of ideal recording conditions.

Testing will therefore consider:

- Different speakers
- Different distances
- Quiet speech
- Background conversation
- Fan noise
- Music
- Other environmental noise
- Similar sounding speech

The goal is to understand where the model starts failing and use those results to improve the system.

## model and hardware tradeoff

The ML model cannot be optimized independently from the hardware.

A larger model may improve detection performance but increase memory usage and latency. A smaller model may be much faster but could produce more false detections.

Vaani will therefore compare these tradeoffs and select the configuration that makes the most sense for the complete device.

## experiment tracking

Important ML experiments will be recorded in the `experiments` directory.

Each experiment should ideally record:

- Model or configuration used
- Dataset version
- Feature configuration
- Training settings
- Evaluation metrics
- Model size
- Inference performance
- Notes about what changed

This will make it easier to understand why a particular model was selected.

## current status

The ML pipeline is currently at the planning stage. The dataset, feature extraction method, model architecture and deployment framework have not been finalized.

The first goal is to establish a simple baseline and then improve it based on actual measurements.
