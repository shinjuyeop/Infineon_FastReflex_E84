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

Research가 exact effective TRAIN 442개 run에서 model/quantization 결과를 보지 않고 고른 2,597개 causal window를 18-file handoff에 추가했다. Deployment는 per-tensor 입력의 극단 tail 문제를 완화하기 위해 TRAIN 절대값 99 percentile인 `±4.1328436`을 calibration 범위로 사용했다. 입력은 INT8 scale `0.03241446`, zero point `0`, NPU softmax 출력은 scale `1/256`, zero point `-128`이다.

M3.1의 actual 302-op graph trace에서 오차는 recurrent step 이전의 shared input projection에서 이미 material해지고, hidden-state error가 seed별 t=`2 / 3 / 1`부터 `0.10`을 넘은 뒤 feedback으로 누적됨을 확인했다. Worst window에는 input saturation이 없고 sigmoid/tanh, classifier, softmax에서 단번에 생기는 collapse도 아니다. Seed `20260829/20260830`의 20-step hidden Jacobian gain은 `26.66/19.79`로 seed `20260828`의 `5.12`보다 훨씬 커서 비슷한 초기 양자화 오차를 크게 증폭한다.

Golden을 사용하지 않고 TRAIN 2,597개 전체의 ensemble p95로 선택한 best PTQ 표현은 gate마다 두 개의 고정 16-channel projection block을 둔다. Baseline member max `0.1065 / 0.8876 / 0.8983`은 `0.2190 / 0.2614 / 0.1286`으로, ensemble max/p95/bias는 `0.0833 / 0.0407 / -0.00205`로 개선됐다. 모든 discrete 결과와 onset `[65,90,107]`은 exact이고 Vela에서 각 member가 `0 CPU / 472 NPU`이지만, member max `0.2614`가 기존 `<=0.10` 계약을 여전히 위반한다.

따라서 M3.1은 partial recovery이지만 FAIL이다. Best 후보를 freeze하지 않았고 formal M3를 재실행하지 않았으며 M4도 승인되지 않는다. 다음 단계는 Research의 reviewed QAT 또는 deployment-aware recurrent model 변경 검토다. Candidate role은 계속 `DEPLOYMENT_ENGINEERING_REFERENCE_MODEL`, scientific verdict는 계속 `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`다. Research, firmware, flash, board execution과 HIL은 수정하거나 시작하지 않았다.

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
python tools/deployment.py evaluate-int8-recovery  # expected exit 2
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

현재 `verify-reference`와 `evaluate-export`는 exit code 0으로 통과한다. `evaluate-int8`과 `evaluate-int8-recovery`는 전체 evidence를 생성한 뒤 현재의 의도된 numerical-contract 실패를 exit code 2로 반환한다. Contract 또는 discrete decision이 어긋나도 fail-closed한다.

현재 Python, M2 conversion/Vela 도구와 KitProg USB 열거 상태를 확인하려면:

```bash
python tools/verify_environment.py
```

생성된 TFLite/Vela 결과와 상세 log는 Git에 포함되지 않는다. Historical M2 evidence는 [`reports/export_target_operator_feasibility.md`](reports/export_target_operator_feasibility.md), M2.1 resolution은 [`reports/float_numerical_contract_resolution.md`](reports/float_numerical_contract_resolution.md), M3 결과는 [`reports/int8_quantization_and_parity.md`](reports/int8_quantization_and_parity.md), M3.1 결과는 [`reports/int8_recurrent_error_localization_and_ptq_recovery.md`](reports/int8_recurrent_error_localization_and_ptq_recovery.md)에 있다.
