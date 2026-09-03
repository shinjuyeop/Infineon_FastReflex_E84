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
- `metrics.json`

실제 계약과 checksum은 [`docs/model_contract.md`](docs/model_contract.md)를 참고한다.

## Current Status

`FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED`

Research가 실제 runtime과 같은 독립 batch-1 `[1,20,80]`을 canonical Float execution으로 승인하고 새 golden과 layer별 contract를 16-file handoff에 고정했다. E84 독립 Host runtime은 모든 수치 layer를 bit-exact하게 재현했다. Float TFLite의 최대 오차는 member logits `2.384186e-6`, member probability `1.072884e-6`, ensemble `3.470729e-7`이며 Research contract를 모두 통과했다. Threshold crossing, consecutive count, 5 ms persistence, onset과 최종 decision은 exact다.

기존 M2 operator 결과도 유지된다. Vela 4.2.0에서 Float graph는 `301 CPU / 0 NPU`이고, non-protected golden windows만 사용한 최소 INT8+softmax probe는 `0 CPU / 192 NPU`로 Ethos-U55-128에 전부 배치된다. 이 probe는 INT8 parity나 quantization sign-off가 아니며 Generic Arm cycle estimate도 board 성능 증거가 아니다. Float gate가 해결되었으므로 다음 milestone `INT8_QUANTIZATION_AND_PARITY`는 승인된다.

Candidate role은 계속 `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`, scientific verdict는 계속 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`다. Firmware, flash, board execution과 HIL은 시작하지 않았다.

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
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

현재 `verify-reference`와 `evaluate-export`는 모두 Research-owned batch-1 contract를 소비해 exit code 0으로 통과한다. Contract 또는 discrete decision이 어긋나면 fail-closed한다.

현재 Python, M2 conversion/Vela 도구와 KitProg USB 열거 상태를 확인하려면:

```bash
python tools/verify_environment.py
```

생성된 TFLite/Vela 결과와 상세 log는 Git에 포함되지 않는다. Historical M2 evidence는 [`reports/export_target_operator_feasibility.md`](reports/export_target_operator_feasibility.md), M2.1 resolution은 [`reports/float_numerical_contract_resolution.md`](reports/float_numerical_contract_resolution.md)에 있다.
