# ASR Backends

The server uses the `ASRBackend` interface so the audio-processing and session layers remain independent of the speech-recognition engine.

## Interface

```python
class ASRBackend:
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        ...
```

The backend receives mono `float32` PCM audio in the range `[-1, 1]` and returns recognized text.

## Included backends

### Mock ASR

`MockASRBackend`

Used for development, automated tests, and integration testing without requiring an ASR engine.

It does not perform speech recognition. It returns a deterministic result containing the received audio duration and RMS level.

### Local Whisper

`LocalWhisperASRBackend`

Optional local speech recognition through `faster-whisper`.

The model runs locally on the server and does not require sending audio to a cloud speech service.

## Configuration

The default configuration uses the mock backend:

```yaml
asr:
  backend: mock
```

To use local Whisper:

```bash
pip install faster-whisper
```

Then configure:

```yaml
asr:
  backend: local_whisper
  model_size: tiny
```

The `tiny` model is the default starting point for local CPU testing.

## Missing optional dependency

If `faster-whisper` is not installed, the local backend raises `ASRUnavailableError`.

The server can continue in development mode without a real ASR engine. Received audio can also be retained as debug WAV files when configured.

## Adding another backend

Implement the `ASRBackend` interface in a new module under this directory.

The rest of the server should not need to change as long as the backend follows the interface.

Possible future local engines include Vosk or whisper.cpp bindings.
