# E84 HIL runtime validation

## Verdict

`E84_RAW_1KHZ_HIL_FAIL`

The full raw-IMU deployment path runs on actual E84/U55 hardware and meets the
board's 1 ms processing budget, but the complete 1 kHz HIL gate fails closed:
at 1 Mbps the host loses or rejects result frames. The highest reliably repeated
clean raw rate in the final sweep was 900 Hz; 925 Hz was intermittent. Strict
target-output numerical parity against the uncompiled host INT8 prototype also
fails, while every compared threshold, persistence, reflex, and onset decision
is exact.

This report does not change formal M3, the scientific verdict, or M4 authority.

## Setup and modes

- Board: `KIT_PSE84_AI`, `PSE846GPS2DBZC4A`, probe `13070E98012D2400`
- Transport: KitProg3 virtual COM, 1,000,000 baud, 8N1, full duplex
- Protocol: binary v1, sequence and endpoint IDs, payload type/length, CRC32
- Replay: 140 raw/features or 121 complete windows from the frozen non-protected
  runtime chain; the first 19 feature/raw responses report window-not-ready
- Deadline: 1,000 µs board processing time

All runs start with an acknowledged state reset. The evaluator fails on missing
results, host or board sequence gaps, out-of-order frames, sender misses,
unexpected status, board drops/overruns/deadline misses, or strict numerical
parity.

## Preprocessing parity

Both the Python streaming implementation and the firmware C compiled as a host
shared library were checked against every golden layer. Base/causal maximum
absolute error is at most `9.54e-7`; normalized/window error is at most
`1.49e-6`. The rolling C INT8 window is element-for-element equal to quantizing
the frozen golden model windows. This gate is `PASS`.

## Board timing

For complete raw inference windows at the final code state:

| Stage | Mean | p95 | Maximum |
|---|---:|---:|---:|
| Feature extraction | about 28.2 µs | 29 µs | 29 µs |
| Normalize + quantize/push/copy | about 47.2 µs | 52 µs | 52 µs |
| Member 1 | about 248.0 µs | 248 µs | 249 µs |
| Member 2 | about 248.5 µs | 249 µs | 249 µs |
| Member 3 | about 248.0 µs | 249 µs | 249 µs |
| Three-member inference | about 744.5 µs | 745 µs | 746 µs |
| Decision | 1 µs | 1 µs | 1 µs |
| Complete raw inference processing | about 830.5 µs | 831 µs | 832 µs |

The raw optimization maintains an INT8 ring, avoiding 1,600-value
requantization on every sample. The direct-window diagnostic path still
quantizes the complete float window and measures about 1,076 µs, so all 121
window packets exceed the 1 ms processing deadline even at a 10 Hz replay rate.
That does not affect the optimized raw path.

## Replay results

| Mode/rate | Results | Board drops/deadlines | Transport | Strict numerical | Discrete decisions |
|---|---:|---:|---|---|---|
| Window 10 Hz | 122/122 | 0 / 121 | complete | FAIL | exact |
| Feature 100 Hz | 141/141 | 0 / 0 | complete | FAIL | exact |
| Raw 900 Hz A | 141/141 | 0 / 0 | complete | FAIL | exact |
| Raw 900 Hz B | 141/141 | 0 / 0 | complete | FAIL | exact |
| Raw 925 Hz | intermittent | 0 / 0 | later run lost 2 results | FAIL | exact where compared |
| Raw 1,000 Hz | 106/141 | 0 / 0 | 35 CRC/missing results | FAIL | exact where compared |

Both clean 900 Hz runs produced deployment digest
`255cd14ee7d89ba0048b49b2e888b6ad4949063b5217afe6c93d7427f975b7be`.
This digest excludes timing/counters but includes endpoints, member/ensemble
outputs, decision flags, persistence, and status. The repeated identity proves
determinism for the complete clean trace.

At 1 kHz the board's final counters still report all 140 inputs received and
processed, zero dropped samples, zero queue overruns, zero board sequence gaps,
and zero deadline misses. The host decoder reports 35 corrupt/missing result
frames. Thus input processing is not the failing resource; the fail-closed gate
is the synchronous result channel. A result is 94 wire bytes, consuming 940 µs
at 1 Mbps before USB/KitProg scheduling. The host response latency for the
received 1 kHz results was mean 5.96 ms, p95 8.61 ms, maximum 12.79 ms. It
reflects UART/USB queueing as well as compute and is not a board processing-time
measurement.

Two higher-baud experiments were rejected: 2 Mbps improved raw delivery but
failed the larger feature/window payloads, and 3 Mbps did not acknowledge the
reset. The canonical reproducible transport therefore remains 1 Mbps rather
than selectively reporting a raw-only setting.

## Deployment parity

Across complete window, feature, and raw traces, target-Vela E84 output differs
from the original frozen host INT8 TFLite prototype by at most `0.04296875` per
member and `0.0143229167` for the ensemble. This exceeds the evaluator's one
output-quantum (`0.00390625`) diagnostic tolerance and is reported as strict
numerical parity `FAIL`; it is not converted into a prototype numerical pass.

Nevertheless, all 121 comparable complete traces have zero mismatches for
threshold crossing, persistence count, `REFLEX_REQUIRED`, and reflex onset. No
threshold, persistence, quantization range, model weight, seed, or output
calibration was tuned from these observations.

## Remaining blockers

The 1 kHz HIL milestone requires an asynchronous/buffered result transport or a
different verified link, followed by all-mode regression. Target-Vela numerical
drift must also be localized if strict host/target equality is required. Neither
blocker authorizes a model change or relaxes the formal numerical contract.
