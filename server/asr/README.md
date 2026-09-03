````markdown
# ASR Backends

The server uses the `ASRBackend` interface so the audio server does not depend on a specific speech recognition engine.

## Interface

```python
class ASRBackend:
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        ...
````

The backend receives mono float32 PCM audio in the range `[-1, 1]` and returns the recognized text.

## Included Backends

### Mock

`MockASRBackend`

Used for development, testing and integration checks.

It does not perform speech recognition. It returns a deterministic result containing the received audio duration and RMS level.

### Local Whisper

`LocalWhisperASRBackend`

Optional local speech recognition using `faster-whisper`.

The model runs locally and does not require sending audio to a cloud service.

## Installation

The default server configuration uses the mock backend, so no additional ASR package is required.

To use the local backend:

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

## Missing Dependency

If `faster-whisper` is not installed, the local backend raises `ASRUnavailableError`.

The server can then continue operating in WAV-only mode rather than returning a fake transcription.

Received audio can be inspected under:

```text
reports/debug_wav/
```

## Adding Another Backend

Implement the `ASRBackend` interface in a new file under this directory.

Possible local engines include Vosk or whisper.cpp bindings.

Only the backend selection needs to change. The audio server and WebSocket protocol remain unchanged.

```

### One important server issue I caught

Before we start testing, there is **one cross-file bug** between the `session.py` and `main.py` versions we already fixed.

`main.py` currently closes the session **before** starting ASR, while `ServerSession.mark_asr_start()` rejects operations on a closed session.

So don't test the server yet.

We should fix that when we do the final `server/` cross-check.

**ASR folder itself is done after replacing that README.**

Next: **`server/storage/` files.**
```
