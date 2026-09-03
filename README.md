# Infineon FastReflex E84

## 목적

Research repository에서 검증·동결된 Float model을 받아 KIT_PSE84_AI / PSoC Edge E84에 배포하고 검증한다.

## Pipeline

```text
Frozen Float Model
  -> Quantization
  -> Target Conversion
  -> Vela
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
- `metrics.json`

실제 계약과 checksum은 [`docs/model_contract.md`](docs/model_contract.md)를 참고한다.

## Current Status

`REFERENCE_MODEL_HANDOFF_AND_HOST_FLOAT_PARITY_PASS`

Exact `model_v2_anchor_refined_gru20_20260902` handoff의 14개 payload와 outer manifest를 검증했고, Research package를 import하지 않는 E84 Host Float 구현이 raw IMU6부터 각 GRU member, ensemble, inclusive threshold, 5 ms persistence와 최종 decision까지 layered golden parity를 통과했다. 모든 numeric layer의 현재 최대 절대 오차는 0이며 discrete parity는 exact다.

이 candidate의 role은 `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`이고 scientific verdict는 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`다. Release/real-robot/safety 모델이 아니다. INT8, Vela, firmware, board execution과 HIL은 시작하지 않았다.

## 구조

```text
configs/deployment/       deployment configuration
model/source/             검토된 frozen Float handoff와 golden evidence
model/quantized/          생성된 quantized model 경계
model/vela/               생성된 Vela output 경계
src/fastreflex_e84/       독립 Host Float runtime과 handoff validator
tools/                    deployment 도구와 환경 점검
firmware/                 향후 ModusToolbox project 경계
hil/host/                 향후 host-side HIL 경계
docs/                     pipeline 및 model contract
reports/                  runtime validation 보고서 경계
tests/                    향후 test suite
```

전체 흐름은 [`docs/deployment_pipeline.md`](docs/deployment_pipeline.md), tool 상태는 [`tools/README.md`](tools/README.md), firmware 경계는 [`firmware/README.md`](firmware/README.md), HIL 경계는 [`hil/README.md`](hil/README.md)를 참고한다.

## M1 verification

Python 3.10 이상에서 다음 명령으로 handoff와 layered parity를 검증한다.

```bash
python tools/deployment.py verify-reference
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

현재 Python 환경과 M1 dependency만 확인하려면:

```bash
python tools/verify_environment.py
```
