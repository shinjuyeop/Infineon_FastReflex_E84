# Research → Deployment Model Contract

## Accepted reference identity

This is the reviewed handoff plus the completed M3 characterization.

| Field | Frozen value |
|---|---|
| Candidate | `model_v2_anchor_refined_gru20_20260902` |
| Deployment role | `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL` |
| Status | `NON_RELEASE_ENGINEERING_REFERENCE` |
| Research repository | `https://github.com/shinjuyeop/Infineon_FastReflex` |
| Candidate source commit | `9bef402e900523e4b5477bf47cd91c0adddf9b2a` |
| Candidate record commit | `8a4970cad8778e100751d4f6a8ae4f15b5eb4c03` |
| Scientific verdict commit | `7fdb61940a6fc60edbd0b2ad5e0726b5eb07d3b6` |
| Packaging/contract source commit | `ff9873da5342091783bcd0b96307d4f217f083a0` |
| Reviewed Research release commit | `314ded6aa3fdfe8d661d1cc670f1b9f3936084fb` |
| Release manifest SHA-256 | `d5d4e7225a35d7547e373b0ac62dbaf552d45c1a3290f214882a032355589dc7` |
| Float contract SHA-256 | `c3e90b4615d52804bee20ac6ef3728fc715d5629c8fc23e39f87640e908f423c` |
| INT8 calibration SHA-256 | `cd82304d34b2cdc60a0feb3de3e84ca7bc7f45e73223e2df389bd695cddbab5f` |

The accepted bundle is [`model/source/model_v2_anchor_refined_gru20_20260902`](../model/source/model_v2_anchor_refined_gru20_20260902). [`configs/deployment/reference_model.yaml`](../configs/deployment/reference_model.yaml) pins its identity independently of the bundle.

## Bundle and integrity

`release_manifest.json` fixes the complete 18-file payload. Acceptance fails on a missing file, extra file, path traversal, manifest mismatch, or content checksum mismatch before a checkpoint is opened.

```text
model_manifest.json
sensor_schema.json
preprocessing.json
label_map.json
metrics.json
normalizer.json
models/member_seed20260828.pt
models/member_seed20260829.pt
models/member_seed20260830.pt
golden_manifest.json
float_numerical_contract.json
calibration_manifest.json
calibration_inputs/int8_representative.npz
golden_inputs/runtime_chain.npz
golden_outputs/runtime_chain.npz
golden_outputs/deployment_runtime_chain.npz
golden_inputs/decision_probe.npz
golden_outputs/decision_probe.npz
release_manifest.json       # deployment config pins this outer manifest
```

## INT8 calibration handoff

The Research-owned calibration artifact contains 2,597 finite `[20,80] float32` tensors from the exact effective TRAIN: 152 Unified `train` runs and 290 valid Model-V2 `V2_TRAIN` runs. Five runtime-uniform endpoints per run are combined with valid physical precursor, Slip, and Support anchors and deduplicated. Selection uses no model output or quantization result. Every source run/file hash and endpoint is recorded. Validation/HOLDOUT/evaluation payloads are excluded, and the artifact is not scientific evidence.

The raw artifact retains normalized TRAIN tails (`min=-40.3635`, `max=314.6293`). Deployment's selected calibration range clips representative values symmetrically to the TRAIN absolute-value p99 `±4.132843623161313`; runtime preprocessing remains unchanged and INT8 input saturation is the ordinary quantizer clamp.

The frozen model inputs have these checksums:

| Artifact | SHA-256 |
|---|---|
| Normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` |
| Seed 20260828 | `7094a2dca40e8d3c84554619652d69c17920c8e82765460ff8621c13ef494cb9` |
| Seed 20260829 | `3ad298eea4c35eca896afd31f860fd6b44ce35d7d9978e2546bb40b693e62c39` |
| Seed 20260830 | `fe96dfeb8461871044de0f8672190680ce46164a1c28efbe6738d22f9d439bbd` |

## Sensor contract

Runtime input is Pelvis IMU6 at exactly 1 kHz, stored as finite `float32 [samples,6]` in pelvis-local axes:

```text
0 accel_x  m/s^2
1 accel_y  m/s^2
2 accel_z  m/s^2
3 gyro_x   rad/s
4 gyro_y   rad/s
5 gyro_z   rad/s
```

The Research reference is the MuJoCo accelerometer/gyro site at the pelvis-frame origin with identity orientation relative to the pelvis body. No software filter precedes feature extraction. Hardware axis, sign, scale, bandwidth, timestamp alignment, and acceleration/gravity convention are not yet validated; target input must reproduce this contract before HIL evidence is meaningful.

## Exact causal preprocessing

The six raw channels are extended in this order to ten base signals:

```text
accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,
accel_norm, gyro_norm, horizontal_accel_norm, horizontal_gyro_norm
```

The complete 80D tensor is block-major in this transform order:

```text
base
delta_1ms
delta_5ms
delta_10ms
causal_mean_5ms
causal_mean_10ms
causal_variance_5ms
causal_variance_10ms
```

Delta is current minus lagged input, with an exact-zero unavailable prefix. Rolling statistics trail and include the current sample. At startup they use every available sample from index zero rather than padding. Mean and population variance (`ddof=0`) use `float64` cumulative sums; variance is clamped to at least zero and cast to `float32`. The output is finite `float32 [samples,80]`.

The exact 80 names are stored in `preprocessing.json` and `normalizer.json`. Their canonical compact-JSON SHA-256 is `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb`.

Normalization is per-channel z-score:

```text
normalized = (float32_feature - float32_mean) / float32_std
```

The stored standard deviation is used exactly; there is no runtime epsilon or clamp. The normalizer records its TRAIN-only fit membership and sample count.

## Window, model, and ensemble

Every inference uses the 20 features ending at the current endpoint, oldest to current, with model input shape `float32 [1,20,80]`. The first valid endpoint is index 19 and replay stride is one sample.

Each of the three members is:

```text
GRU(input_size=80, hidden_size=32, layers=1,
    unidirectional, dropout=0, batch_first=true)
Linear(32 -> 2)
```

Each member has 11,010 parameters and produces `float32 [2]` logits. Every 20-sample window is an independent call with zero-initialized GRU hidden state; hidden state is never carried between endpoints. This is not a stateful streaming-GRU contract.

Class mapping is exact:

```text
0 NORMAL
1 HAZARD_REFLEX_REQUIRED
```

For each member, softmax is computed in `float32` and class index 1 is selected. Member probabilities are promoted to `float64`, and the three values in seed order `20260828, 20260829, 20260830` are averaged in `float64`. Logits are never averaged.

## Decision semantics

The ensemble probability comparison is inclusive: `probability >= 0.99`. A counter increments for each consecutive passing 1 ms sample and resets to zero on the first failure. `REFLEX_REQUIRED` becomes true on the fifth passing sample, remains true only while the current streak is at least five, and deasserts immediately on a failed sample. Terrain is not an input or gate.

## Golden evidence and parity

The runtime-chain golden input is samples `[2950,3090)` from non-protected `V2_VALIDATION` run `m2v2_dss_v_c_0250_s10`. Its source run SHA-256 is `54bf650db12a7ce1ecea2085864bccdd02ecc9d920922d2216cd61650bb84bd3`. The slice is parity plumbing, not scientific evidence, and no Generalization HOLDOUT payload was accessed to create it. The original `runtime_chain.npz` was produced with one `[121,20,80]` PyTorch call and remains byte-identical historical M1 evidence. Canonical deployment output is `deployment_runtime_chain.npz`, generated by an independent `[1,20,80]` call at every endpoint.

Golden outputs preserve:

1. raw Pelvis IMU6 and 1 kHz timestamps;
2. 10D base signals;
3. causal 80D features;
4. normalized features;
5. `[20,80]` windows and endpoints;
6. each member's logits;
7. each member's Hazard probability;
8. ensemble mean;
9. inclusive threshold crossing and consecutive count;
10. 5 ms persistence/onset;
11. final `REFLEX_REQUIRED` trace.

A separate synthetic probability probe covers equality at 0.99, one-below-threshold reset, fifth-sample assertion, continued assertion, and immediate deassertion. Shape and dtype are exact for every layer. Preprocessing/model windows, member probabilities, and ensemble use the retained `atol=rtol=1e-6`. Member logits use `atol=4e-6, rtol=0`, justified by a Research batch-size sweep whose maximum was `2.9802322387695312e-6`; relative error is not meaningful near zero logits. Threshold crossing, consecutive count, persistence, onset, and final decision are exact. The current accepted Host batch-1 run is bit-identical at every numeric layer (`max_absolute_error = 0.0`).

The closest canonical ensemble value to `0.99` is `0.9909300009409586`, with margin `0.0009300009409586307`. This is more than 467 times the continuous ensemble error permitted at that point. This margin applies only to the frozen non-protected golden slice; M3 must evaluate INT8 boundary sensitivity independently.

## Scientific and release status

The immutable scientific verdict is `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`; whole-simulation status is `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`.

| Evidence | Hazard | Slip | Support | Primary no-hazard | Premature |
|---|---:|---:|---:|---:|---:|
| V2 internal validation | 59/64 | 30/35 | 30/30 | 26/26 | 5/64 |
| Generalization validation | 25/26 | 11/12 | 14/14 | 10/10 | 1/26 |
| Generalization HOLDOUT | 25/28 | 11/14 | 14/14 | 5/8 | 2/28 |

M3 proves deterministic conversion, exact decision parity on the canonical golden, and formal-model generic Vela mapping. It does not accept continuous INT8 probability parity: observed recurrent excursions violate the Deployment-owned contract, so M4 is not authorized. The handoff still does not establish scientific support, real-robot support, production readiness, safety certification, final sensor architecture, firmware execution, board timing/RAM/Flash, or HIL behavior.
