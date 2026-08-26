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

향후 다음 artifact 묶음으로 경계를 정의한다.

- `model`
- `model_manifest.json`
- `sensor_schema.json`
- `preprocessing.json`
- `label_map.json`
- `golden_inputs`
- `golden_outputs`
- `metrics.json`

초안은 [`docs/model_contract.md`](docs/model_contract.md)를 참고한다.

## Current Status

`PROJECT_SCAFFOLD_ONLY`

현재는 repository 골격, 계약 문서, environment placeholder만 존재한다. model, quantization, Vela, firmware, HIL은 아직 구현되지 않았다.

## 구조

```text
configs/deployment/       deployment configuration
model/source/             검토된 frozen Float model 경계
model/quantized/          생성된 quantized model 경계
model/vela/               생성된 Vela output 경계
tools/                    deployment 도구와 환경 점검
firmware/                 향후 ModusToolbox project 경계
hil/host/                 향후 host-side HIL 경계
docs/                     pipeline 및 model contract
reports/                  runtime validation 보고서 경계
tests/                    향후 test suite
```

전체 흐름은 [`docs/deployment_pipeline.md`](docs/deployment_pipeline.md), tool 상태는 [`tools/README.md`](tools/README.md), firmware 경계는 [`firmware/README.md`](firmware/README.md), HIL 경계는 [`hil/README.md`](hil/README.md)를 참고한다.

## Environment placeholder

Python 3.10 이상에서 다음 명령으로 현재 placeholder 환경을 확인한다.

```bash
python tools/verify_environment.py
```
