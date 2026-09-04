# INT8 16-channel HIL prototype freeze

## Verdict

`NON_RELEASE_HIL_PATH_PROTOTYPE_FROZEN`

This freeze exists only to exercise the E84 deployment and HIL path. It does
not authorize M4, change the scientific verdict, or convert the formal M3
failure into a pass.

```text
role = NON_RELEASE_HIL_PATH_PROTOTYPE
formal_m3_pass = false
numerical_contract_pass = false
scientific_release = false
real_robot_supported = false
production_ready = false
safety_certified = false
hil_path_only = true
```

The frozen machine-readable contract is
`model/prototype/model_v2_anchor_refined_gru20_int8_16ch_hil_prototype/manifest.json`.

## Starting-state integrity

- Deployment branch: `main`
- Deployment starting HEAD: `564d3db788f8a1fa4a02b405a8e8f227c1b2755f`
- Starting `origin/main`: `564d3db788f8a1fa4a02b405a8e8f227c1b2755f`
- Deployment worktree: clean, including untracked files
- Research HEAD inspected read-only: `2aca66d77fb6518af0d81c63f416d87e3ef12f57`
- Research worktree had pre-existing tracked changes. No Research file was
  written by this work.

The preserved formal states are:

```text
INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL
MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED
SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED
M4 = NOT AUTHORIZED
```

## Reproduction

The canonical `evaluate-int8-recovery` workflow was run before freezing. Its
expected exit status was 2 because the unchanged numerical contract remains a
failure. It reproduced:

- the same three reference checkpoints and ensemble order `20260828`,
  `20260829`, `20260830`;
- normalizer SHA-256
  `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a`;
- TRAIN calibration SHA-256
  `cd82304d34b2cdc60a0feb3de3e84ca7bc7f45e73223e2df389bd695cddbab5f`;
- p99 symmetric clipping bound `4.132843623161313`;
- two 16-channel blocks per gate, static `[1,20,80]` INT8 input, 700
  TFLite operations, in-graph two-class softmax, and hazard class index 1;
- input quantization `(scale=0.03241445869207382, zero_point=0)` and output
  quantization `(scale=0.00390625, zero_point=-128)`;
- byte-identical conversion for all members.

Frozen member hashes are:

| Seed | Bytes | SHA-256 |
|---:|---:|---|
| 20260828 | 165,064 | `04e1b5796fa0cac9240086230ee0d4389bbca89c133b9238758f1558dd266404` |
| 20260829 | 165,280 | `bda9b7f7776a981e0f0529052997fc5b4c57fabec2f67f1eec11804cd464a20c` |
| 20260830 | 165,208 | `a60c1764a05dd8b8a193a6534729e6a72d45f312f035f537a75cb23217666dc7` |

The formal M3 artifacts in `model/quantized/formal` were neither selected nor
overwritten. Their preserved hashes are `d6726d65...`, `c793df17...`, and
`f4538854...`.

## Numerical and discrete result

Member maximum errors are `0.2189599 / 0.2613510 / 0.1286247`. Ensemble
maximum/p95/bias are `0.0832680 / 0.0407164 / -0.00205461`.

Threshold crossing, consecutive count, `REFLEX_REQUIRED`, and Reflex onset are
exact on the non-protected canonical deployment golden. This is deployment
parity evidence only. The member maximum requirement remains `<= 0.10`, so the
numerical contract is still `FAIL`.

## Reproduction command

```bash
python tools/deployment.py evaluate-int8-recovery  # expected exit 2
python tools/deployment.py freeze-prototype
```

No QAT checkpoint or weight is present in this freeze.
