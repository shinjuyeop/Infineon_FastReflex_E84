# Deployment Pipeline

The deployment stages are deliberately gated:

```text
Reviewed Frozen Float Model + Contract
  -> M1 Contract/checksum validation + independent Host Float parity  [PASS]
  -> Export and target-operator feasibility                            [NEXT]
  -> INT8 quantization and parity                                      [NOT STARTED]
  -> Vela / Ethos-U55 compilation                                     [NOT STARTED]
  -> Firmware integration                                             [NOT STARTED]
  -> KIT_PSE84_AI execution                                           [NOT STARTED]
  -> HIL and runtime latency/RAM/Flash validation                      [NOT STARTED]
```

## Completed gate: M1

`REFERENCE_MODEL_HANDOFF_AND_HOST_FLOAT_PARITY` accepts the exact non-release engineering reference through [`configs/deployment/reference_model.yaml`](../configs/deployment/reference_model.yaml). The validator checks the outer manifest pin, all 14 payload hashes, provenance, scientific/non-release status, sensor schema, feature order, normalization, architecture, ensemble, and decision contract before loading model weights.

The E84 host implementation is independent of the Research Python package. It reproduces every numeric golden layer within `atol=rtol=1e-6` and requires exact endpoint/decision parity. Run it with:

```bash
python tools/deployment.py verify-reference
```

## Next gate

The exact next milestone is `EXPORT_AND_TARGET_OPERATOR_FEASIBILITY`. It should select a target-compatible export representation and determine whether the GRU, per-window zero hidden state, softmax, three-member mean, and surrounding preprocessing/decision operations map cleanly to the supported E84/Ethos-U55 toolchain.

No M1 result authorizes quantization or firmware work. The next gate must preserve source/conversion/build/runtime provenance separately and must not reinterpret target parity as scientific validation.
