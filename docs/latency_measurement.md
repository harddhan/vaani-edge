# Latency Measurement

## Primary metric

The main system metric is:

```text
keyword-end -> server receives the last audio byte required for ASR
```

The exact timestamps used in the implementation should be recorded rather than inferred from model accuracy.

## Measurement stages

| Stage | Definition | Source |
|---|---|---|
| Keyword detection delay | Keyword end to trigger after consecutive-positive windows | Firmware timestamps/logs |
| Buffering delay | Session start/client connection to first audio byte received | Server session metrics |
| Network/streaming delay | First received audio byte to last required audio byte | Server session metrics |
| Server processing delay | Last audio byte received to ASR invocation start | Server session metrics |
| ASR duration | ASR invocation duration | Server ASR metrics |

ASR duration is reported separately because it depends strongly on the selected backend and model.

## Procedure

1. Start the server with `python -m server.main`.
2. Run the desktop simulation or real firmware.
3. Generate a session and inspect the resulting server metrics JSON.
4. For hardware detection delay, correlate the trigger event with firmware runtime metrics.
5. Repeat measurements under the same network and audio conditions.
6. Report averages and, where useful, percentile values.

## Desktop latency probe

The standalone probe can isolate network/server behavior without relying on the KWS trigger:

```bash
python -m desktop.latency_probe --wav samples/test_clip.wav
```

## Reporting rule

Do not publish a latency number as a measured result unless it came from an actual run and recorded timestamp data.

Desktop measurements should be labeled as desktop measurements. Hardware measurements should be labeled with the board and test conditions.
