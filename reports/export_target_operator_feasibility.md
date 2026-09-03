# Export and Target Operator Feasibility

## Verdict

`FLOAT_EXPORT_PARITY_FAIL_INT8_U55_OPERATOR_MAPPING_PASS`

The exact frozen GRU20 architecture has no observed Ethos-U55 operator-topology blocker after integer lowering, but it cannot pass the current deployment gate unchanged: two converted Float logits exceed the frozen numerical tolerance. Do not proceed directly to formal INT8 sign-off. Resolve the Float numerical contract first; no architecture redesign is justified by M2 operator evidence.

This is not a scientific, quantization, performance, firmware, or release-readiness verdict. The immutable scientific verdict remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`.

## 1. Starting state and boundary

- Deployment started clean at `4454cdb66110e3c7fba2e97b2447382c6dbc2b34` (`origin/main` matched).
- Research reference was `df277458a49ff5d391deac3ef0bd51ff7e18a84b` (`origin/main` matched) with unrelated working-tree experiments already present. Research was read-only and none of those changes were included here.
- Accepted candidate: `model_v2_anchor_refined_gru20_20260902`.
- Accepted outer manifest SHA-256: `9cbd42c95e42e90ef05f4a2ba77306a18dc3cbfa4e814c6e8432e29281a2b642`.
- M1 verification runs before every M2 export. Weights, 80D preprocessing, `[20,80]`, zero hidden state per window, member order, threshold `>=0.99`, and 5-sample persistence were not changed. No HOLDOUT access occurred.

## 2. Local environment and target format

Discovered host environment:

| Component | Local result |
|---|---|
| OS / Python | Linux 6.8 x86-64 / Python 3.10.12 |
| PyTorch | 2.13.0+cpu |
| TensorFlow | 2.19.0 |
| LiteRT | 2.1.0 |
| Vela | 4.2.0 at `/home/alswoghd/.local/bin/vela` |
| ModusToolbox | 3.8.0.18384 |
| Programming Tools | 1.8.1.2061 |
| Infineon ML Pack / DEEPCRAFT Model Converter | Not found in PATH, `/opt/Tools`, or the user Infineon tool directory |
| Connected hardware | Three `04b4:f155` KitProg3 CMSIS-DAP devices, read-only enumerated; no connect/reset/flash |

Infineon's ML flow accepts TFLite/H5 and uses TFLite Micro; the PSOC Edge target exposes Vela options. The E84 integrates a 400 MHz Ethos-U55 with 128 MACs/cycle. Therefore static built-in TFLite is the locally executable and vendor-compatible interchange path, and Vela is invoked with `ethos-u55-128`. Sources: [Infineon Machine Learning user guide](https://www.infineon.com/assets/row/public/documents/30/44/infineon-infineon-modustoolbox-machine-learning-user-guide-usermanual-v02-00-en-usermanual-en.pdf), [ML Configurator user guide](https://www.infineon.com/assets/row/public/documents/30/44/infineon-modustoolbox-machine-learning-configurator-user-guide-usermanual-en-09018a908087681f.pdf), and [PSOC Edge E84 datasheet](https://documentation.infineon.com/psocedge/docs/hvr1750409588574).

The installed `Arm/vela.ini` contains generic reference systems, not an E84 memory model. Arm explicitly warns that platform-specific configuration and target measurement are required; Vela estimates are compiler guidance rather than performance evidence. See [Arm Vela](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela).

## 3. Conversion approaches evaluated

1. PyTorch legacy ONNX export was attempted at opset 18 and failed because the local `onnx` module is absent. The dynamo path likewise lacks `onnxscript`. Even with those packages, ONNX is not the local Infineon target format and would add an ONNX-to-TFLite recurrent lowering stage.
2. A native Keras GRU with gate-order and reset-after weight remapping was tested. It produced larger Float deviations (maximum `5.2452087e-6`) and an `UNPACK`-bearing graph, so it was rejected.
3. The selected path directly lowers the frozen PyTorch reset-after GRU equations to TensorFlow primitives, statically unrolls exactly 20 steps, and converts only TFLite built-ins. Input projection is performed across all 20 timesteps before the recurrent loop, matching PyTorch's execution structure and reducing both error and Vela work. This is an equivalent deployment representation, not a stateful GRU or architecture change.

Generated artifacts are reproducible but ignored by Git:

| Member | Float TFLite bytes | SHA-256 |
|---:|---:|---|
| 20260828 | 88,936 | `30824d3767d549adde4f98f392284460d4b1d5e92e276c0ae103567efb2c6bc0` |
| 20260829 | 88,936 | `90da2ce606ad1b81f89a0a28772f8c2d8d593790a3c2ff8adb8643204d4f5772` |
| 20260830 | 88,936 | `283e7f9355c0ee2feae0977483ce73922caa6adca56b723ba56e32520a46aed2` |

## 4. Float parity

The test executes each TFLite with LiteRT built-in kernels and no host delegate over all 121 non-protected golden windows. TFLite input/output are static `[1,20,80] float32` and `[1,2] float32 logits`.

| Layer | Result | Maximum absolute error | Tolerance violations |
|---|---|---:|---:|
| Seed 20260828 logits | PASS | `9.536743e-7` | 0 |
| Seed 20260829 logits | **FAIL** | `2.682209e-6` | 2 |
| Seed 20260830 logits | PASS | `2.622604e-6` | 0 (relative term covers the worst value) |
| All member probabilities | PASS | `1.132488e-6` | 0 |
| Float64 ensemble mean | PASS | `3.838601e-7` | 0 |
| Threshold/count/reflex/onset | PASS | exact | 0 |

The criterion is the frozen combined `atol=1e-6, rtol=1e-6`, not maximum absolute error alone. Seed 20260829 has two values outside that combined bound. Final decisions matching does not override the logit failure.

A diagnostic PyTorch run also showed that changing the frozen evidence execution from its 121-window batch to contract-style independent batch-1 calls can itself exceed the same tolerance for one seed. This explains why a reviewed numerical-contract resolution is needed; it is not permission to widen tolerance locally.

## 5. Actual graph

The converted model does not contain a GRU operator, loop, dynamic dimension, or carried hidden-state input. One Float member contains 301 static built-in operations:

| Operator | Count |
|---|---:|
| ADD | 80 |
| FULLY_CONNECTED | 21 |
| LOGISTIC | 40 |
| MUL | 40 |
| SPLIT | 20 |
| STRIDED_SLICE | 60 |
| SUB | 20 |
| TANH | 20 |

The 21 fully-connected operations are one batched 20-timestep input projection, 19 non-constant hidden projections, and one classifier. The initial hidden state is a graph constant of zeros. Softmax is intentionally outside the Float member graph so that the frozen per-member Float32 softmax, Float64 promotion, and Float64 three-member mean remain explicit.

## 6. Vela / Ethos-U55 mapping

Vela 4.2.0 was run against the actual generated graph with `--accelerator-config ethos-u55-128`.

| Graph / memory assumption | Raw TFLite ops | Vela CPU | Vela NPU | Result |
|---|---:|---:|---:|---|
| Float32 member / Shared SRAM | 301 | 301 (100%) | 0 | Float is entirely CPU fallback |
| INT8+softmax probe / Shared SRAM | 302 | 0 | 192 (100%) | Full NPU placement |
| INT8+softmax probe / SRAM-only | 302 | 0 | 192 (100%) | Full NPU placement |

The quantized probe adds one TFLite SOFTMAX; Vela decomposes/fuses the static graph, hence 192 scheduled NPU operations rather than the raw 302. No operator remains on the CPU. The probe uses only M1 `V2_VALIDATION` golden windows as representative data and is explicitly not formal quantization parity.

| Generated operator artifact | Bytes | SHA-256 |
|---|---:|---|
| INT8+softmax input probe | 103,760 | `dc14a0d9a7a51cd5204abb7b8328b5f6a33c48a5113e212cfc48ff8198757036` |
| Vela Shared-SRAM output | 83,040 | `607404e4ac958394f8d2ecf83d22d1a2a8aedbf669e9583e1608927ad70e875a` |
| Vela SRAM-only output | 82,912 | `03cc2bf6393ba25ad690b786558a1e5ed7565cc9daa1e8eb4e2579af264d6e80` |

Exact runtime postprocessing still requires a small Cortex-M55 section for Float32 member softmax/dequantization as selected, Float64 ensemble mean, threshold, persistence, and decision. That is an explicit contract boundary, not an unidentified Vela graph fallback. The causal 80D preprocessing is also outside this model graph.

## 7. Memory and compute implications

Tool-reported for one INT8+softmax probe:

| Metric | Shared SRAM generic config | SRAM-only generic config |
|---|---:|---:|
| MACs / member | 212,036 | 212,036 |
| NPU cycles | 42,207 | 19,748 |
| Total cycles | 108,587 | 19,799 |
| Tool inference time at generic 500 MHz | 0.22 ms | 0.04 ms |
| SRAM | 13.03 KiB | 3.44 KiB |
| Constant storage | 79.75 KiB off-chip | 79.64 KiB on-chip |
| Maximum subgraph IO | 14.596 KiB | 5.002 KiB |

Three members imply 636,108 MACs per 1 ms endpoint and about 238.9 KiB of compiled constants. Scratch may be reusable for sequential invocations, but the actual TFLM/interpreter integration policy is unknown.

Analytical only: scaling the SRAM-only 19,799-cycle estimate to the E84's 400 MHz clock gives `49.50 us/member` and `148.49 us` for three sequential members. Scaling the generic Shared-SRAM total gives about `814.40 us` for three, before preprocessing, input quantization, TFLM calls, postprocessing, interrupts, and application load. Neither number is measured, and the generic memory bandwidth/latency is not E84-specific. Float CPU latency is unknown because Vela does not model CPU execution; its printed zero cycles/time for an all-CPU graph is not a performance result.

Thus 1 kHz is plausible only for a fully mapped integer graph with favorable model placement. It is not yet established.

## 8. Reproduction and tests

```bash
python tools/verify_environment.py
python tools/deployment.py verify-reference
python tools/deployment.py evaluate-export
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src tools tests
```

`evaluate-export` writes its complete evidence JSON and then returns exit code 2 for the current expected fail-closed Float verdict. Current suite: `6 passed`. The M2 test regenerates all Float models and the INT8 operator probe in an isolated temporary directory, checks exact graph inventories, reproduces the current Float gate failure, verifies downstream decisions, and requires Vela's Float CPU-only / INT8 NPU-only placement.

Generated evidence locations (ignored by Git):

- `model/converted/`
- `model/quantized/`
- `model/vela/export_feasibility/`
- `reports/generated/export_target_feasibility.json`

## 9. Unresolved risks and next milestone

- The frozen golden logit tolerance is not met by the best selected target-format path.
- Formal INT8 logit/probability/decision parity has not started; the 0.99 boundary may be sensitive to output quantization.
- The Infineon ML Pack and its target-specific E84 Vela/middleware configuration are not installed locally.
- Vela 4.2.0 generic memory assumptions are not an E84 memory/linker contract.
- Float CPU fallback timing, 80D preprocessing, quantization, TFLM invocation overhead, three-model arena policy, and interrupt/application contention are unknown.
- No firmware was created, no board was connected through a debugger, and no board timing was measured.
- Hardware IMU mapping and the scientific HOLDOUT failure remain unchanged.

Next milestone: `FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLUTION`. Research should review batch-121 golden generation versus the batch-1 runtime contract and either provide deployment-appropriate frozen evidence/tolerance or identify another faithful export reference. E84 must not change this contract unilaterally. Only after that review passes should Deployment perform formal `INT8_QUANTIZATION_AND_PARITY` with the supported Infineon ML Pack/Vela configuration, followed by board profiling.
