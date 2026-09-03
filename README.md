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

`INT8_DECISION_PARITY_PASS_NUMERICAL_CONTRACT_FAIL`

Research가 exact effective TRAIN 442개 run에서 model/quantization 결과를 보지 않고 고른 2,597개 causal window를 18-file handoff에 추가했다. Deployment는 per-tensor 입력의 극단 tail 문제를 완화하기 위해 TRAIN 절대값 99 percentile인 `±4.1328436`을 calibration 범위로 사용했다. 입력은 INT8 scale `0.03241446`, zero point `0`, NPU softmax 출력은 scale `1/256`, zero point `-128`이다.

세 formal member는 모두 byte-deterministic하고 Vela 4.2.0에서 각각 `0 CPU / 192 NPU`로 전부 배치된다. Frozen golden의 threshold crossing, consecutive count, 5 ms persistence, onset `[65,90,107]`, 최종 decision도 exact다. 그러나 member 최대 확률 오차가 `0.1065 / 0.8876 / 0.8983`, ensemble 최대/p95 오차가 `0.3334 / 0.1884`, bias가 `-0.02093`으로 INT8-specific numerical contract를 크게 위반한다. 정확한 discrete 결과만으로 이 recurrent probability instability를 승인하지 않았다.

따라서 M3는 완료됐지만 FAIL이며 M4는 승인되지 않는다. 선택안은 NPU-side softmax 뒤 probability dequantization과 CPU Float64 ensemble/decision 경계이나, instability가 해소되기 전에는 M4 canonical input이 아니다. Candidate role은 계속 `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`, scientific verdict는 계속 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`다. Firmware, flash, board execution과 HIL은 시작하지 않았다.

## 구조

```text
configs/deployment/       deployment configuration
model/source/             검토된 frozen Float handoff와 golden evidence
model/converted/          생성된 Float target-format model 경계
model/quantized/          생성된 quantized model 경계
model/vela/               생성된 Vela output 경계
src/fastreflex_e84/       독립 Host Float runtime과 handoff validator
tools/                    deployment 도구와 환경 점검
firmware/                 향후 ModusToolbox project 경계
hil/host/                 향후 host-side HIL 경계
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
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

현재 `verify-reference`와 `evaluate-export`는 exit code 0으로 통과한다. `evaluate-int8`은 전체 evidence를 생성한 뒤 현재의 의도된 numerical-contract 실패를 exit code 2로 반환한다. Contract 또는 discrete decision이 어긋나도 fail-closed한다.

현재 Python, M2 conversion/Vela 도구와 KitProg USB 열거 상태를 확인하려면:

```bash
python tools/verify_environment.py
```

생성된 TFLite/Vela 결과와 상세 log는 Git에 포함되지 않는다. Historical M2 evidence는 [`reports/export_target_operator_feasibility.md`](reports/export_target_operator_feasibility.md), M2.1 resolution은 [`reports/float_numerical_contract_resolution.md`](reports/float_numerical_contract_resolution.md), M3 결과는 [`reports/int8_quantization_and_parity.md`](reports/int8_quantization_and_parity.md)에 있다.
