# Research → Deployment Model Contract

이 문서는 계약의 draft schema만 정의한다. 실제 shape, sampling rate, label, normalization 값은 아직 고정하지 않는다.

## Artifact bundle

```text
model
model_manifest.json
sensor_schema.json
preprocessing.json
label_map.json
golden_inputs/
golden_outputs/
metrics.json
```

## `model_manifest.json` draft

```json
{
  "model_version": null,
  "framework": null,
  "framework_version": null,
  "model_format": null,
  "input_shape": null,
  "input_dtype": null,
  "output_shape": null,
  "output_dtype": null,
  "sample_rate_hz": null,
  "window_ms": null,
  "feature_order": [],
  "normalization": null,
  "labels": [],
  "model_sha256": null,
  "source_repository": null,
  "source_commit": null,
  "exported_at": null
}
```

## 계약 원칙

- `model_sha256`는 전달받은 model 파일과 일치해야 한다.
- sensor 순서와 단위는 `sensor_schema.json`에서 명시한다.
- runtime에서 재현해야 할 preprocessing만 `preprocessing.json`에 기록한다.
- `label_map.json`은 output index와 label 의미를 고정한다.
- `golden_inputs`와 `golden_outputs`는 변환 전후 parity 확인에 사용한다.
- `metrics.json`은 research validation 결과와 평가 dataset provenance를 연결한다.
- schema versioning과 acceptance threshold는 첫 frozen artifact 검토 시 결정한다.
