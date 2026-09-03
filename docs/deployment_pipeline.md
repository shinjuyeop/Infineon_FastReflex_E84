# Deployment Pipeline

The deployment stages are deliberately gated:

```text
Reviewed Frozen Float Model + Contract
  -> M1 Contract/checksum validation + independent Host Float parity  [PASS]
  -> M2 Float export parity                                            [FAIL]
  -> M2 INT8 Ethos-U55 operator probe                                  [PASS]
  -> Float numerical-contract resolution                               [NEXT]
  -> INT8 quantization and parity                                      [NOT STARTED]
  -> Production Vela / Ethos-U55 compilation                          [NOT STARTED]
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

## M2 result

`FLOAT_EXPORT_PARITY_FAIL_INT8_U55_OPERATOR_MAPPING_PASS`

The selected representation is static-batch Float32 TFLite with the exact PyTorch reset-after GRU equation unrolled for 20 timesteps. All three members export, but two logits from seed `20260829` exceed the frozen combined `atol=rtol=1e-6` limit. Downstream Float probability and every discrete decision remain within/exact, but the numeric gate is intentionally not waived.

Vela cannot place any Float operator on the U55. A minimum INT8+softmax probe establishes that the same primitive topology can be represented entirely on Ethos-U55-128 with no CPU graph fallback. It does not authorize or validate quantization.

## Next gate

Resolve the Float deployment numerical contract with Research ownership before formal INT8 work. The resolution must explicitly address the fact that the frozen golden logits were produced as a 121-window batch while the runtime contract requires independent batch-1 calls. Do not change weights, architecture, preprocessing, decision semantics, or scientific status. After a reviewed contract resolution, proceed to formal INT8 parity with an E84-specific Vela memory configuration, then firmware timing.
