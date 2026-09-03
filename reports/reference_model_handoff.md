# Reference Model Handoff and Host Float Parity

## Verdict

`REFERENCE_MODEL_HANDOFF_AND_HOST_FLOAT_PARITY_PASS`

The Deployment repository received the exact frozen Research V2 engineering reference and independently reproduces its Float runtime semantics. This is an engineering handoff PASS, not a scientific or deployment-readiness verdict.

## Provenance and acceptance

- Deployment starting commit: `2689c0dd18166f1ebd1ddc2da87d21194062e9cd`
- Research starting commit: `d7ee4aa37e4d5319d63b993c5fcb2bf846d916ab`
- Research reviewed release commit: `df277458a49ff5d391deac3ef0bd51ff7e18a84b`
- Candidate: `model_v2_anchor_refined_gru20_20260902`
- Role: `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`
- Research candidate source commit: `9bef402e900523e4b5477bf47cd91c0adddf9b2a`
- Accepted release manifest: `9cbd42c95e42e90ef05f4a2ba77306a18dc3cbfa4e814c6e8432e29281a2b642`
- Payload files verified: 14/14

Acceptance is config-pinned outside the bundle and fails on manifest, file-set, checksum, provenance, contract, feature-order, architecture, ensemble, decision, or scientific-status drift.

## Golden methodology

Research packaged an immutable 140-sample slice from non-protected `V2_VALIDATION` run `m2v2_dss_v_c_0250_s10`; Generalization HOLDOUT was not accessed. The E84 implementation imports no Research package. It starts from raw `float32 [140,6]` Pelvis IMU and recomputes the complete chain.

| Layer | Shape | Dtype | Result | Max absolute error |
|---|---:|---|---|---:|
| Raw Pelvis IMU6 | `[140,6]` | float32 | PASS | input/schema valid |
| 10D base | `[140,10]` | float32 | PASS | 0 |
| Causal 80D | `[140,80]` | float32 | PASS | 0 |
| Normalized 80D | `[140,80]` | float32 | PASS | 0 |
| Model windows | `[121,20,80]` | float32 | PASS | 0 |
| Three member logits | `[3,121,2]` | float32 | PASS | 0 |
| Three member Hazard probabilities | `[3,121]` | float64 | PASS | 0 |
| Ensemble mean | `[121]` | float64 | PASS | 0 |
| Endpoints / threshold / count | `[121]` | integer/bool | exact PASS | exact |
| Persistence / onset / final decision | `[121]` | bool | exact PASS | exact |

The golden chain asserts `REFLEX_REQUIRED` and exercises deasserted regions; onset endpoints are 65, 90, and 107 in the slice. A separate 18-sample probability probe verifies inclusive equality at 0.99, reset below threshold, assertion on the fifth consecutive pass, continued assertion, and immediate deassertion.

The declared numeric acceptance tolerance is `atol=1e-6`, `rtol=1e-6` to permit justified host-library/backend variation. Current results are bit-identical for all stored numeric layers. Shapes/dtypes and all discrete outputs must match exactly.

## Automated verification

```text
python tools/deployment.py verify-reference
  REFERENCE_MODEL_HANDOFF_AND_HOST_FLOAT_PARITY_PASS

PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
  5 passed

python -m compileall -q src tools tests
  PASS
```

Tests also prove that checksum corruption is rejected before artifact loading and that feature-order drift is rejected even after internally consistent manifest re-signing.

## Scientific status and unresolved risk

The immutable scientific verdict remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, with `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`. The bundle is not a release model, real-robot-supported model, final sensor architecture, or safety-certified model.

Unresolved deployment risks are:

- real Pelvis IMU axes, sign, scale, bandwidth, timestamp and acceleration/gravity convention are not mapped to the simulator contract;
- the source format is a PyTorch checkpoint, not yet a target-compatible graph;
- GRU, zero-init-per-window execution, softmax and three-model mean feasibility for the E84/Ethos-U55 path is unknown;
- INT8 error, RAM, Flash, latency, firmware, board execution and HIL have not been evaluated;
- three ensemble members may be a material resource cost.

## Next milestone

Proceed with exactly `EXPORT_AND_TARGET_OPERATOR_FEASIBILITY`. Do not begin INT8 quantization, Vela compilation, or firmware integration until that gate identifies a faithful target representation and operator/resource blockers.
