# M3 INT8 Quantization and Parity

## Verdict

`INT8_DECISION_PARITY_PASS_NUMERICAL_CONTRACT_FAIL`

The frozen three-member GRU20 can be converted reproducibly to fully mapped E84/U55-compatible INT8 graphs, and its complete deployment decision trace is exact on the canonical golden. It cannot yet be accepted as the formal M4 representation because large member-level recurrent probability excursions violate the independently stated INT8 numerical contract. M4 is not authorized.

This is deployment evidence only. The role remains `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`; `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED` and `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED` are unchanged.

## 1. Starting state and scope

- Deployment started clean at `78226d8fcbe680a40908f106bf9ade5446bf2a8c`, matching `origin/main`.
- Research started at `cb1b83371c3ff8be26b44e85fb709f6449c7e1f7`, matching `origin/main`, with unrelated Sand work in progress. Only the M3 calibration exporter/config/bundle/docs/tests were staged.
- No protected HOLDOUT, retraining, model selection, threshold tuning, firmware, debugger connection, reset, or board flash occurred.
- The fixed seeds are `20260828`, `20260829`, `20260830`; execution remains independent batch-1 `[1,20,80]` with a fresh zero hidden state.

## 2. Reviewed calibration provenance

Research added `INT8_CALIBRATION_HANDOFF_EXPORTED`. The exact candidate effective TRAIN contributes 152 `unified_hazard_reflex_20260829/train` runs and 290 valid `model_v2_hazard_reflex_20260901/V2_TRAIN` runs. Effective run identity SHA-256 is `0466ea84871d178856ffb10d8b1c0cec730286b35b589c73ff1b5fb1065aa5ab`.

The model-blind selection takes five evenly spaced runtime-valid endpoints from every run, adds valid physical precursor/Slip/Support endpoints, and deduplicates collisions. It produces 2,597 causal normalized `[20,80] float32` windows: 2,210 uniform, 127 precursor, 141 Slip and 119 Support tags. It uses no model output or quantization result.

| Frozen item | Identity |
|---|---|
| Training config | `2da935d96c80452d69108ac14aa8a4df8edae297672cf25fd1945eb4deb64dbe` |
| Calibration NPZ | `cd82304d34b2cdc60a0feb3de3e84ca7bc7f45e73223e2df389bd695cddbab5f` (12,351,989 bytes) |
| Research release manifest | `d5d4e7225a35d7547e373b0ac62dbaf552d45c1a3290f214882a032355589dc7` |
| Research packaging/release commits | `ff9873d` / `314ded6` |

The raw representative values span `[-40.3635,314.6293]`; absolute p99 is `4.132843623161313`. Full-range min/max calibration spends almost all input codes on rare TRAIN tails, giving scale `1.3921286`, 22 threshold mismatches, 10 final-state mismatches and all three lost onsets. The selected model-blind policy therefore clamps only representative calibration values to symmetric TRAIN absolute p99. It covers 99% of representative elements by definition, preserves sign symmetry, and does not alter causal preprocessing. Runtime out-of-range behavior is the normal INT8 saturation rule.

## 3. Representations evaluated

| Representation | Ensemble max / p95 / bias | Crossing/count/reflex/onset mismatch | Conclusion |
|---|---:|---:|---|
| Raw TRAIN min/max, NPU INT8 softmax | `0.96339 / 0.75004 / -0.10408` | `22 / 22 / 10 / 3` | Reject: outlier-dominated input scale |
| Robust p99, INT8 logits, CPU Float32 softmax | `0.33345 / 0.18906 / -0.02032` | `0 / 0 / 0 / 0` | Output boundary does not fix recurrent error |
| Robust p99, NPU INT8 softmax | `0.33338 / 0.18842 / -0.02093` | `0 / 0 / 0 / 0` | Selected M3 candidate; numerical gate still FAIL |

NPU-side softmax is the cleanest boundary because it has effectively the same numerical behavior as host softmax, avoids a CPU nonlinear operation, and preserves full graph placement. Each member emits INT8 `[1,2]` softmax probabilities; CPU code would dequantize, select class 1, promote to Float64, average in fixed seed order, compare `>=0.99`, and apply five-sample persistence. This is a selected failure-analysis candidate, not an M4-authorized artifact.

## 4. Quantization parameters and artifacts

All members use INT8 input scale `0.03241445869207382`, zero point `0`; output probability scale `0.00390625` (`1/256`), zero point `-128`. Canonical golden input quantization saturates 20 of 193,600 elements (`0.0103306%`), has p99 dequantization error `0.0160335`, and maximum saturated error `1.11404`.

| Seed | Raw TFLite bytes | SHA-256 | Recurrent intermediate scale range |
|---:|---:|---|---:|
| 20260828 | 96,704 | `d6726d65a6bb63ca564b57b528112ab6e6db5a0e1b77b87cdae08c4ef363aec1` | `0.014988–0.028153` |
| 20260829 | 97,096 | `c793df173ecf73c80f05bae6dbb2061a552f2e4448fff02025892c547a19d3c3` | `0.018900–0.039196` |
| 20260830 | 97,096 | `f4538854a092d4d12a8909f06d2a01156c8097f54c4518b31d9df8a82ad1a854` | `0.019384–0.036918` |

Every learned matrix weight is INT8 per-axis: classifier has 2 scales, recurrent matrix 96, and input matrix 96. All 343 runtime activation tensors per member, including input/output, are per-tensor; 21 INT32 bias tensors are per-axis. Three repeated conversions are byte-identical. Input/output shapes and dtypes are exact.

## 5. Float-versus-INT8 probability parity

The selected graph's equivalent output is dequantized softmax rather than logits, so no post-softmax logit is inventively reconstructed. The separate logit-output alternative directly measures dequantized logits and obtains maximum logit error `3.22123`; its probability result remains unstable.

| Seed | Max | Mean | p50 | p90 | p95 | p99 | Bias |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260828 | 0.106519 | 0.011586 | 0.002047 | 0.037587 | 0.055821 | 0.084310 | -0.001244 |
| 20260829 | 0.887587 | 0.037635 | 0.002404 | 0.044648 | 0.313470 | 0.630818 | -0.028502 |
| 20260830 | 0.898282 | 0.034393 | 0.003135 | 0.046625 | 0.107334 | 0.685032 | -0.033042 |
| Ensemble | 0.333381 | 0.025237 | 0.002487 | 0.053197 | 0.188422 | 0.286233 | -0.020929 |

The largest seed-20260829 excursion is Float `0.953994` versus INT8 `0.066406` at window 105; seed-20260830 is Float `0.964688` versus INT8 `0.066406` at window 111. Both remain below the ensemble decision threshold but prove that global probability fidelity is not preserved.

## 6. Threshold, persistence, onset and final decision

The closest Float value above `0.99` is window 73: Float `0.990930001`, INT8 `0.990885417`, margin `0.000930001`, signed error `-0.000044584`. The closest below is window 83: Float `0.987859706`, INT8 `0.986979167`, margin `0.002140294`, signed error `-0.000880539`.

All 22 threshold crossings match. All 121 consecutive counts and `REFLEX_REQUIRED` values match. Float and INT8 onset window indices are `[46,71,88]`, corresponding to endpoints `[65,90,107]`; onset displacement is zero. Threshold `0.99` and persistence `5` were not changed.

## 7. INT8-specific numerical contract

Float tolerances are not reused. The contract first limits median member error to one output quantum (`1/256`), then independently rejects maximum member/ensemble excursions above `0.10`, ensemble p95 above `0.05`, and absolute bias above `0.01`. Canonical-input saturation must remain at or below `0.02%`. Exact crossing, count, onset and final state are separately mandatory.

Input saturation and median member error pass. Maximum member error (`0.89828`), ensemble maximum (`0.33338`), ensemble p95 (`0.18842`) and bias (`0.02093`) fail. These round semantic gates deliberately reject the observed result; they were not widened to fit it. Exact discrete parity also passes, but does not override continuous failure.

## 8. Formal Vela/U55 evidence

Vela 4.2.0 compiled each selected formal model for `ethos-u55-128` with Generic Arm `Ethos_U55_High_End_Embedded`. Every raw graph has 302 TFLite operations including softmax; every compiled graph reports `0 CPU / 192 NPU`, with zero unsupported-semantics warnings.

| Mode / seed | Compiled bytes | SRAM KiB | Constant KiB | Max subgraph KiB | NPU / total cycles |
|---|---:|---:|---:|---:|---:|
| Shared / 20260828 | 84,144 | 13.03 | 80.83 off-chip | 14.596 | 42,207 / 108,587 |
| Shared / 20260829 | 83,840 | 13.03 | 80.55 off-chip | 14.596 | 42,191 / 108,571 |
| Shared / 20260830 | 84,192 | 13.03 | 80.89 off-chip | 14.596 | 42,175 / 108,555 |
| SRAM-only / 20260828 | 84,016 | 3.44 | 80.72 on-chip | 5.002 | 19,748 / 19,799 |
| SRAM-only / 20260829 | 83,696 | 3.44 | 80.41 on-chip | 5.002 | 19,748 / 19,799 |
| SRAM-only / 20260830 | 84,032 | 3.44 | 80.73 on-chip | 5.002 | 19,748 / 19,799 |

Each member has 212,036 MACs; the ensemble has 636,108. Sequential three-member totals are 325,713 generic Shared-SRAM cycles or 59,397 generic SRAM-only cycles. Pure analytical scaling to 400 MHz is `814.28 us` or `148.49 us`, before preprocessing, quantization, invocation and CPU postprocessing. These are not E84 measurements. Largest-member Vela SRAM (`13.03` or `3.44 KiB`) may be reusable sequentially, but it is not a TFLM arena/linker measurement; actual RAM/Flash belongs to M4/M5.

## 9. Reproduction, tests and boundaries

```bash
python tools/deployment.py verify-reference
python tools/deployment.py evaluate-export
python tools/deployment.py evaluate-int8  # expected exit 2 after writing evidence
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python -m compileall -q src tools tests
```

Machine-readable evidence is regenerated at `reports/generated/int8_quantization_parity.json`; TFLite and Vela outputs remain in ignored generated-output boundaries. Regression coverage requires all three models, exact hashes/order/IO, byte-identical reconversion, alternatives, numerical fail-closed checks, exact decisions, and two-mode zero-fallback Vela compilation.

No firmware or board state was created or modified. The unresolved risk is post-training INT8 recurrent instability outside the near-threshold portions exercised by this single non-protected golden. M4, board performance claims and HIL remain blocked; any QAT/retraining or revised Research artifact is a separate reviewed task.
