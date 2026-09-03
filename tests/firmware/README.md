# Firmware Tests

## Host-side unit tests

Pure-logic firmware components with no ESP-IDF/hardware dependency can,
in principle, be compiled and unit-tested on the host (e.g. with a
lightweight GoogleTest/Catch2 setup and a host-side FreeRTOS stub, or by
extracting the logic into hardware-independent headers). This repository
does not currently ship a host-side C++ test harness (out of scope to
stand up a full ESP-IDF host test environment here), but two components
are specifically written to be portable and worth prioritizing if you
add one:

- `trigger_state_machine` (`components/trigger_state_machine/`): pure
  C++ with no ESP-IDF dependencies beyond standard headers. Its behavior
  is fully covered by the Python reference implementation's tests
  (`tests/python/test_trigger_logic.py`, testing
  `server/session.py::TriggerStateMachine`, which mirrors the C++
  version exactly) - use those tests as the specification if you port
  them to a C++ test framework.
- `protocol_codec` (`components/lan_transport/`): pure C++ encode/decode
  functions with no I/O. Recommended to unit test byte-for-byte against
  `tests/python/test_protocol.py`'s fixtures (encode a message in
  Python, decode it in C++, and vice versa) to catch endianness/layout
  bugs early.

## Hardware-in-the-loop tests (documented separately, not automated here)

The following require a physical ESP32-S3 + microphone and cannot be
exercised in CI:

- `audio_capture`'s real (non-mock) implementation.
- End-to-end latency/RAM/CPU measurement (`docs/memory_budget.md`,
  `docs/latency_measurement.md`).
- `lan_transport`'s real Wi-Fi connection and WebSocket handshake.
- `kws_inference`'s on-device INT8 inference correctness versus the
  Python TFLite interpreter (`ml/quantization/evaluate_quantized.py`
  covers the desktop-side equivalent).

Recommended manual test procedure once hardware is available:
1. Flash firmware, open serial monitor.
2. Confirm `metrics_task` logs show stable, non-zero free heap.
3. Speak the configured keyword and confirm a trigger + successful
   round-trip to the server (check server logs / `reports/debug_wav/`).
4. Speak non-keyword phrases for several minutes and confirm no
   triggers (false-activation spot check).
