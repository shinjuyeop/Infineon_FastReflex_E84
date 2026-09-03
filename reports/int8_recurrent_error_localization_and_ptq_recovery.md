# M3.1 INT8 Recurrent Error Localization and PTQ Recovery

## Verdict

`INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL`

The failure is a recurrent representation problem, not an output-softmax problem. A mathematically equivalent 16-channel projection partition substantially reduces the canonical ensemble error and retains complete U55 placement, but it still violates the unchanged member maximum-error gate. Formal M3 was therefore not rerun, M4 is not authorized, and the next reviewed action returns to Research for QAT or a deployment-aware recurrent model change.

The candidate remains `model_v2_anchor_refined_gru20_20260902`, role `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`, with scientific verdict `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`. No Research semantics, weights, preprocessing, threshold, persistence, firmware, or board state changed.

## 1. Starting state and reproduced M3 failure

- Deployment started clean at `0adbb09e608cd905c5fb34b2c69a2f3a56fd094c`, matching `origin/main`.
- Research started at `dd4c909584741606c965241061a98796ca5fc5db`, matching `origin/main`, with unrelated work already present in `src/fastreflex/dataset/sand_factor_conditioned.py`. Research remained read-only.
- The exact formal M3 artifacts reproduced byte-for-byte: `d6726d65...` / `c793df17...` / `f4538854...` for seeds `20260828` / `20260829` / `20260830`.
- The reproduced member maxima were `0.106519`, `0.887587`, and `0.898282`; ensemble max/p95/bias were `0.333381 / 0.188422 / -0.020929`.
- The worst member windows remained 107, 105, and 111. At window 105 seed `20260829` remained Float `0.953994` versus INT8 `0.066406`; at window 111 seed `20260830` remained Float `0.964688` versus INT8 `0.066406`.
- All baseline threshold crossings, consecutive counts, final reflex states, and onset endpoints `[65,90,107]` remained exact.

## 2. Actual graph and recurrent localization

The formal model contains 302 statically lowered built-in TFLite operations, not a high-level GRU operator. The trace follows the actual input `FULLY_CONNECTED`, gate split/slices, 20 reset/update/candidate recurrences, hidden projections, classifier, and softmax. Preserving all tensors in LiteRT allowed every quantized intermediate to be dequantized and compared with the corresponding Float32 PyTorch equation.

The existing contract's `0.10` member maximum-error budget is also used as the material internal-error marker. The first material representational error is already present in the input projection before recurrent timestep 0:

| Seed / worst window | Input-projection scale | Projection max error | First hidden-state error `>=0.10` | Hidden max error at t=5 / t=19 | Classifier-logit max error |
|---|---:|---:|---:|---:|---:|
| 20260828 / 107 | 0.209167 | 0.117935 | t=2 | 0.0664 / 0.1985 | 0.2964 |
| 20260829 / 105 | 0.215676 | 0.131755 | t=3 | 0.2379 / 1.0188 | 3.2212 |
| 20260830 / 111 | 0.237995 | 0.164384 | t=1 | 0.3499 / 1.0859 | 3.1042 |

At t=0, reset/update preactivation errors are already `0.15–0.19`, but sigmoid compresses their gate-output errors to `0.032–0.047`. Tanh similarly attenuates some early candidate error. There is no isolated sigmoid, tanh, classifier, or softmax collapse. Instead, the hidden perturbation is fed back through the next hidden projection and grows across the remaining steps. On the two unstable members, hidden-projection error reaches about `2.02` at t=19 before the classifier converts the diverged hidden state into the reversed logit margin.

The worst windows themselves contain no saturated input elements; their values lie within `[-2.8762,1.5778]`. Their input-projection tensors also contain no INT8 endpoint values. Canonical input saturation remains only 20 of 193,600 elements (`0.01033%`), below the existing gate, so saturation is not the primary cause. Seed `20260830` does reach six hidden-state endpoints late in its worst recurrence, but only after error has already become material.

## 3. Why the members differ

The initial projection quantization errors are similar across members, but their Float hidden transitions have very different perturbation gain:

| Seed | Maximum local hidden Jacobian norm | Full 20-step Jacobian-product norm | Maximum suffix-product norm | Classifier weight norm |
|---:|---:|---:|---:|---:|
| 20260828 | 1.354 | 5.124 | 5.431 | 1.281 |
| 20260829 | 1.512 | 26.662 | 26.662 | 1.645 |
| 20260830 | 1.611 | 19.795 | 30.286 | 1.605 |

Thus seeds `20260829` and `20260830` amplify comparable early quantization perturbations roughly five to six times more strongly by this local linear sensitivity measure, and their classifiers also have larger gain. This explains why seed `20260828` drifts but does not undergo the same probability reversal.

## 4. Error-source separation

An ablation dequantized the exact formal INT8 IO and per-axis weight tensors, then ran the recurrent intermediates in Float32. Maximum canonical probability errors were:

| Seed | INT8 IO only | INT8 weights only | IO + weights, Float recurrent intermediates | Actual full INT8 baseline |
|---:|---:|---:|---:|---:|
| 20260828 | 0.0368 | 0.0416 | 0.0775 | 0.1065 |
| 20260829 | 0.1163 | 0.0920 | 0.1661 | 0.8876 |
| 20260830 | 0.0835 | 0.0144 | 0.0820 | 0.8983 |

Input and weight quantization are non-negligible, especially for seed `20260829`, but they do not explain the full collapse. The large additional error appears only when per-tensor recurrent activations and feedback are quantized. M3's host-Float-softmax alternative already showed that classifier-output dequantization plus Float softmax has the same failure, and the present internal trace confirms that classifier and softmax are downstream witnesses rather than the origin.

## 5. Calibration integrity and focused PTQ candidates

The formal baseline remains the exact 442-run, 2,597-window TRAIN handoff with SHA-256 `cd82304d34b2cdc60a0feb3de3e84ca7bc7f45e73223e2df389bd695cddbab5f`. Its symmetric absolute p99 bound remains `±4.132843623161313`. No HOLDOUT was read and no new percentile was selected from golden behavior.

The prior M3 raw full-range policy and robust p99 policy remain the range comparison: raw min/max was rejected because its input scale `1.3921` destroyed discrete parity, while robust p99 preserves it. M3.1 therefore held p99 fixed and tested only projection representations motivated by the localization:

- One 32-channel block per gate, separating reset/update/new projections.
- Two fixed 16-channel blocks per gate.
- Four fixed 8-channel blocks per gate.
- An exploratory INT16-activation/INT8-weight logit graph was also converted outside source outputs. Vela mapped it as `0 CPU / 281 NPU`, but both installed LiteRT 2.1.0 and TensorFlow 2.19 aborted during tensor allocation, so Host accuracy could not be evaluated. It was rejected as an unqualified toolchain path; Vela acceptance alone is not parity evidence.

All three full-INT8 candidates retain the same weights, operation order within each output row, fresh-zero-state semantics, runtime IO, p99 calibration, and NPU softmax. Partitioning only gives narrower output groups independent per-tensor activation ranges. The candidate-selection metric was declared as ensemble p95 error over all 2,597 TRAIN-derived windows. It selected 16-channel blocks (`0.026847` versus `0.027146` for 32 and `0.028956` for 8). Canonical golden was not used by this selection.

## 6. Complete canonical-golden assessment

| Representation | Member max by seed | Ensemble max / p95 / bias | Exact discrete result |
|---|---|---|---|
| Formal combined projection | `0.1065 / 0.8876 / 0.8983` | `0.3334 / 0.1884 / -0.02093` | PASS |
| Gate-split 32 | `0.1526 / 0.5063 / 0.1775` | `0.1307 / 0.0566 / -0.00181` | PASS |
| Two blocks/gate 16 | `0.2190 / 0.2614 / 0.1286` | `0.0833 / 0.0407 / -0.00205` | PASS |
| Four blocks/gate 8 | `0.2034 / 0.4236 / 0.1494` | `0.1570 / 0.0425 / -0.00530` | FAIL |

The TRAIN-selected 16-channel representation is the best partial recovery. It eliminates the two original probability reversals and passes input saturation, member median, ensemble max, ensemble p95, bias, and every discrete requirement. It still fails the unchanged member maximum gate: observed `0.261351`, required `<=0.10`. Seed `20260828` is worse than baseline, which is another reason not to characterize this as general recovery.

The selected residual maxima move to windows 97, 104, and 114. Their hidden error first exceeds `0.10` at t=4, t=5, and t=4 and reaches `0.3240`, `0.2872`, and `0.3695` at t=19. The residual limiting behavior is therefore still recurrent hidden-state accumulation, not output quantization.

Regime analysis confirms that the failure is concentrated in transition windows. The selected candidate reduces transition-region ensemble max/mean from `0.3334/0.0562` to `0.0833/0.0205`. Strong-hazard max remains `0.00299`, threshold-vicinity max is `0.01125`, and benign-region max is `0.02331`. Exact threshold/count/reflex/onset parity, including endpoints `[65,90,107]`, is retained.

## 7. U55 feasibility of the best partial candidate

The selected source graphs are 700 TFLite operations and 165,064 / 165,280 / 165,208 bytes. Repeated conversion is byte-identical for all members; hashes are `04e1b579...`, `bda9b7f7...`, and `a60c1764...`.

Vela 4.2.0 maps every member in both memory modes as `0 CPU / 472 NPU` with no fallback. Each has 212,676 MACs. Three-member summaries are:

| Generic mode | Compiled bytes sum | Max member SRAM | NPU / total cycles |
|---|---:|---:|---:|
| Shared SRAM | 353,328 | 5.09 KiB | 185,070 / 389,520 |
| SRAM only | 352,592 | 3.50 KiB | 71,526 / 71,679 |

These generic Vela estimates preserve operator feasibility but are not board timing, TFLM arena, linker, or 1 kHz proof. No board was accessed or modified.

## 8. Gate decision and next action

The M3 numerical contract was not changed. Because the selected PTQ representation still fails member maximum probability error, it is not frozen and the full formal M3 process was not rerun. A diagnostic improvement cannot authorize M4.

The exact frozen GRU20 is too sensitive for acceptable full-INT8 PTQ fidelity under the practical representations validated here. The current evidence supports a Research-side reviewed intervention—preferably QAT, or a deployment-aware recurrent model change—before E84 deployment continues. No QAT or Research change was implemented in this repository.

## 9. Reproduction and generated evidence

```bash
python tools/deployment.py verify-reference
python tools/deployment.py evaluate-int8                   # expected exit 2
python tools/deployment.py evaluate-int8-recovery          # expected exit 2
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
ruff check src tools tests
python -m compileall -q src tools tests
```

Machine-readable evidence is regenerated at `reports/generated/int8_recurrent_error_localization.json`. It contains all 20 timestep traces for the three formal worst windows and benign/threshold/strong-hazard representatives, IO/weight ablations, TRAIN selection evidence, all candidate distributions, residual traces, exact decisions, artifact hashes, and two-mode Vela results. Generated models, compiled models, logs, and the mixed-precision probe remain outside tracked source boundaries.
