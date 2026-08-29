# Vaani

## what is Vaani

Vaani is a low latency voice activation system designed to run directly on an edge device. The basic idea is to let the device listen for a predefined wake word, process the audio locally and trigger an action when it detects it. Instead of depending on a cloud server just to figure out whether the wake word was spoken, the main voice activation process should happen directly on the device.

## the problem

Voice activation sounds pretty simple until you try running it on a small embedded device. The hardware has limited processing power, memory, storage and battery, while the system still needs to keep listening to audio and respond quickly when the wake word is spoken. Real environments make it harder too, because the device has to deal with background noise, different speakers, quiet or unclear speech and the possibility of false activations.

So the problem isn't just making a model that can recognize a word. The actual challenge is making the complete audio processing and ML pipeline work efficiently on real hardware.

## what we are trying to build

Vaani will capture audio through a microphone and process it locally on the edge device. The audio will go through preprocessing and feature extraction before being passed to a lightweight ML model. If the model detects the target wake word with sufficient confidence, Vaani will generate the required trigger.

```text
Microphone
    ↓
Audio Capture
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
TinyML Model
    ↓
Wake Word Detection
    ↓
Trigger
```

The exact hardware, audio processing approach and ML model will be decided through experimentation rather than being fixed from the beginning.

## objectives

The main goal is to build a voice activation system that is actually practical on resource constrained hardware. This means we care about low detection latency, reliable wake word detection, low memory usage, low computational requirements and low power consumption. The system should also work without continuous internet connectivity and maintain reasonable performance when there is background noise.

We also want the final prototype to be compact enough to feel like an actual device rather than just a development board connected to a microphone.

## requirements

Vaani should be able to capture audio through a microphone, process that audio locally, extract useful features and run a lightweight ML model for wake word detection. When the target wake word is detected, the system should generate the required trigger while ignoring unrelated speech as much as possible.

Along with basic functionality, we will measure detection accuracy, false acceptance rate, false rejection rate, detection latency, RAM usage, model size and power consumption. We will also test how the system behaves with different speakers, distances and background noise.

## constraints

The biggest constraint is the edge hardware itself. Unlike a desktop or cloud server, the selected device will have limited processing capability and memory, so the model and audio pipeline need to be designed around those limitations. Power is another important consideration because the device may need to listen continuously, which means inefficient processing can quickly become a problem.

Latency also matters because a voice activator that takes too long to respond won't feel responsive even if the detection accuracy is good. The system therefore needs a reasonable balance between accuracy, speed, memory usage and power consumption.

## scope

The initial scope of Vaani is focused on local voice activation and wake word detection. This includes microphone based audio capture, audio preprocessing, feature extraction, lightweight ML inference, embedded deployment, hardware integration and optimization of latency, memory and power.

Full speech recognition, conversational AI, cloud based processing and large language model inference are outside the initial scope. The idea is to solve the voice activation problem properly first instead of trying to turn Vaani into a complete voice assistant.

## how we will evaluate it

We don't want to stop at saying that Vaani works. A model can perform well on a dataset and still behave badly when it is running on an actual device, so the final system will be evaluated using real measurements.

The important things we'll look at are accuracy, latency, memory usage, model size, power consumption, false activations and robustness to noise. Different models and configurations can also be compared to understand the tradeoffs. For example, if a smaller model loses a little accuracy but reduces latency and memory usage significantly, that may actually be the better choice for an edge device.

## development approach

The project will start with research and system planning rather than immediately buying hardware. We'll first understand the audio pipeline, compare suitable hardware and explore possible ML approaches. After that, a software prototype and baseline model can be developed before moving the system onto the selected embedded hardware.

Once the first embedded version works, the focus will shift towards measurement and optimization. We'll test latency, memory, power and detection performance, then improve the parts that are actually limiting the system. Finally, everything will be integrated into the Vaani prototype and tested under realistic conditions.

## success criteria

Vaani should eventually be able to listen continuously, detect the target wake word locally and respond with low enough latency to feel real time. It should operate within the memory and computational limits of the selected hardware, avoid unnecessary cloud dependency and maintain useful detection performance under reasonable background noise.

More importantly, the final claims about Vaani should be supported by actual measurements rather than just saying that the system is fast, efficient or accurate.

## current status

Vaani is currently in the early research and system planning stage. We are working on understanding the requirements, audio pipeline, hardware options and possible TinyML approaches. The architecture and hardware decisions will be updated as experiments give us better information about what actually works.
