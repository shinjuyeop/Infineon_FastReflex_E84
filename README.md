# Infineon FastReflex E84

## 목적

Research repository에서 검증·동결된 Float model을 받아 KIT_PSE84_AI / PSoC Edge E84에 배포하고 검증한다.

## Pipeline

```text
Frozen Float Model
  -> Float TFLite Export
  -> Float Parity Gate
  -> Float Numerical Contract Resolution
  -> Minimum INT8 Operator Probe
  -> Vela / U55 Mapping
  -> Formal INT8 Parity
  -> INT8 Recurrent Error Localization / PTQ Recovery
  -> Firmware Integration
  -> E84
  -> HIL
  -> Runtime Validation
```

## Repository boundary

이 저장소가 담당하는 범위:

- 동결된 Float model과 계약 artifact의 검증
- quantization, target conversion, Vela
- firmware integration
- E84 HIL 및 runtime validation

이 저장소에서 하지 않는 작업:

- dataset 생성
- model training
- model architecture exploration

연구 작업과 Float model export는 [`Infineon_FastReflex`](https://github.com/shinjuyeop/Infineon_FastReflex)에서 수행한다. 기존 `/d/shin/Infineon`의 코드, model, firmware는 자동으로 복사하지 않으며 향후 명시적으로 검토된 migration만 허용한다.

## Research → Deployment 계약

현재 exact frozen reference는 다음 reviewed artifact 묶음으로 경계를 정의한다.

- 3-member Float `model`
- `model_manifest.json`
- `sensor_schema.json`
- `preprocessing.json`
- `label_map.json`
- `golden_inputs`
- `golden_outputs`
- `float_numerical_contract.json`
- `calibration_manifest.json`
- `calibration_inputs/int8_representative.npz`
- `metrics.json`

실제 계약과 checksum은 [`docs/model_contract.md`](docs/model_contract.md)를 참고한다.

## Current Status

`INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL`

Formal model status and non-release deployment status are independent:

| Gate | Status |
|---|---|
| M1 Host Float | PASS |
| M2.1 Float numerical contract | PASS |
| M3 Formal INT8 | FAIL |
| M3.1 PTQ recovery | FAIL |
| 16-channel non-release prototype freeze | PASS |
| E84 target-specific Vela, CPU fallback 0 | PASS |
| Release firmware build / flash / boot / three-member U55 execution | PASS |
| Full window/feature/raw deployment trace execution | PASS |
| Strict host INT8 vs target-Vela numerical parity | FAIL |
| Raw 1 kHz no-loss HIL | FAIL |

The TRAIN-only-selected 16-channel representation is frozen solely as
`NON_RELEASE_HIL_PATH_PROTOTYPE`. Its canonical member maximum errors remain
`0.2190 / 0.2614 / 0.1286`; ensemble max/p95/bias remain
`0.0833 / 0.0407 / -0.00205`. This still violates the formal member maximum
contract `<= 0.10`. It is not an approved, release, production, real-robot, or
safety-certified model, and M4 remains **NOT AUTHORIZED**.

ML Pack 3.0.0.2416 / CoreTools 3.0.0.8948 / Vela 4.3.0 compiled each member
for the actual `ethos-u55-128` and
`PSE8x_U55_400MHz_SOCMEM_200MHz_QUAD_XIP` configuration with 0 CPU and 472 NPU
operators. The three compiled blobs total 352,592 bytes, 638,028 MACs, and an
estimated 114,231 cycles.

The checked-in ModusToolbox project builds in Release, flashes
`PSE846GPS2DBZC4A`, boots with `CYBOOT_SUCCESS`, and executes all three U55
members. Actual raw-path timing is about 745 µs p95 for the three members and
831 µs p95 total processing, with zero board deadline misses. Firmware C
preprocessing matches the golden chain to `1.49e-6`, and the rolling INT8
window is exact.

HIL remains fail-closed. At the canonical 1 Mbps KitProg3 link, repeated 900 Hz
raw traces are complete and deterministic, 925 Hz is intermittent, and 1 kHz
produces result-frame CRC/loss even though the board receives/processes all 140
inputs with zero board drops or deadline misses. Target-Vela output differs
from the original host INT8 prototype by member max `0.04296875` and ensemble
max `0.0143229167`, exceeding the one-quantum diagnostic tolerance. Threshold,
persistence, reflex decision, and onset remain exact for every comparable
result. No model or decision tuning was performed.

Scientific status remains
`MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED` and
`SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`.

## 구조

```text
configs/deployment/       deployment configuration
model/source/             검토된 frozen Float handoff와 golden evidence
model/converted/          생성된 Float target-format model 경계
model/quantized/          생성된 quantized model 경계
model/vela/               생성된 Vela output 경계
src/fastreflex_e84/       독립 Host Float runtime과 handoff validator
tools/                    deployment 도구와 환경 점검
firmware/fastreflex_e84/  ModusToolbox CM33 secure/non-secure + CM55/U55 project
hil/host/                 binary protocol replay and fail-closed HIL runner
docs/                     pipeline 및 model contract
reports/                  runtime validation 보고서 경계
tests/                    handoff, parity와 operator-feasibility test suite
```

전체 흐름은 [`docs/deployment_pipeline.md`](docs/deployment_pipeline.md), tool 상태는 [`tools/README.md`](tools/README.md), firmware 경계는 [`firmware/README.md`](firmware/README.md), HIL 경계는 [`hil/README.md`](hil/README.md)를 참고한다.

## Verification

Python 3.10 이상에서 다음 명령으로 handoff와 layered parity를 검증한다.

```bash
python tools/deployment.py verify-reference
python tools/deployment.py evaluate-export
python tools/deployment.py evaluate-int8  # expected exit 2: numerical gate fails
python tools/deployment.py evaluate-int8-recovery  # expected exit 2
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

현재 `verify-reference`와 `evaluate-export`는 exit code 0으로 통과한다. `evaluate-int8`과 `evaluate-int8-recovery`는 전체 evidence를 생성한 뒤 현재의 의도된 numerical-contract 실패를 exit code 2로 반환한다. Contract 또는 discrete decision이 어긋나도 fail-closed한다.

현재 Python, M2 conversion/Vela 도구와 KitProg USB 열거 상태를 확인하려면:

```bash
python tools/verify_environment.py
```

생성된 TFLite/Vela 결과와 상세 log는 Git에 포함되지 않는다. Prototype과
hardware evidence는 [`reports/int8_16ch_hil_prototype_freeze.md`](reports/int8_16ch_hil_prototype_freeze.md),
[`reports/e84_target_vela_compilation.md`](reports/e84_target_vela_compilation.md),
[`reports/e84_firmware_integration.md`](reports/e84_firmware_integration.md),
[`reports/e84_hil_runtime_validation.md`](reports/e84_hil_runtime_validation.md)에
있다. Historical M2/M3/M3.1 reports는 `reports/`에 그대로 보존한다.
