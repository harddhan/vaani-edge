# Memory Budget

## Status

The ESP32-S3 RAM requirement has **not yet been validated on physical hardware**.

The values below are configuration values or known allocation sizes, not a claim that the complete application fits within the 256 KB target.

## Known allocations

| Component | Current configuration | Notes |
|---|---:|---|
| Tensor arena | 60 KB configured | Actual required arena usage must be measured |
| Audio ring buffer | 32,000 int16 samples = 64 KB | `configs/audio.yaml` |
| Network send/receive buffers | 4 KB + 2 KB = 6 KB | Does not include all TCP/lwIP/Wi-Fi overhead |
| Task stacks | 4096 + 8192 + 6144 + 3072 bytes | Configured stack sizes |
| Feature buffer | Depends on current firmware buffer definition | Must be verified against the current `50 x 13` model input |
| ESP-IDF/FreeRTOS/Wi-Fi/lwIP | Not estimated here | Must be measured |

These values cannot simply be added to obtain total application RAM because framework allocations, alignment, heap fragmentation, task runtime behavior, and driver buffers also contribute.

## Measurement procedure

1. Build and flash the firmware on the target ESP32-S3.
2. Run the system under sustained listening and representative network conditions.
3. Record free heap and minimum free heap from `runtime_metrics`.
4. Measure actual tensor arena usage.
5. Record task stack high-water marks.
6. Repeat while streaming a representative utterance.
7. Compare the observed peak memory requirement with the project's 256 KB target.
8. Replace this document's pending fields with measured values.

## Important distinction

The INT8 model file size is a Flash/storage measurement. It is not equivalent to total runtime RAM usage.

The current INT8 model artifact is `38,712` bytes, but that number alone does not demonstrate RAM compliance.

## Reporting rule

Until physical measurements exist, reports should say:

> RAM compliance: pending hardware measurement.

Do not claim the 256 KB requirement is satisfied from source-code estimates alone.
