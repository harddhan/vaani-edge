# Desktop Pipeline

The desktop pipeline simulates the intended VAANI runtime before ESP32 hardware is available.

It can test:

* Audio input
* MFCC extraction
* Keyword inference
* Trigger detection
* Pre roll handling
* LAN audio streaming
* Server communication
* Latency

## Run the server

Terminal 1:

```powershell
python -m server.main
```

## Run the pipeline

Terminal 2:

```powershell
python -m desktop.simulate_pipeline --wav data\raw\vaani\vani_0000.wav --model ml\artifacts\ds_cnn_best.keras
```

To test the streaming path without a trained model:

```powershell
python -m desktop.simulate_pipeline --wav data\raw\vaani\vani_0000.wav --force-trigger
```

## Latency Test

```powershell
python -m desktop.latency_probe --wav data\raw\vaani\vani_0000.wav
```

## Files

* `wav_audio_source.py` — WAV based microphone simulation.
* `desktop_kws_client.py` — KWS inference, trigger logic, and pre roll.
* `protocol_client.py` — WebSocket audio streaming.
* `simulate_pipeline.py` — desktop pipeline command.
* `latency_probe.py` — streaming latency measurement.
