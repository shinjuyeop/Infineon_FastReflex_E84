# Reports

Quantization parity, Vela conversion, firmware integration, HIL, runtime validation 결과의 경계다. 각 보고서는 source model checksum, contract revision, toolchain version, configuration, target 정보를 추적할 수 있어야 한다.

M1 reviewed result는 [`reference_model_handoff.md`](reference_model_handoff.md)에 있다. 명령별 임시 JSON이나 arbitrary output은 commit하지 않는다.

M2 mixed verdict와 target-operator evidence는 [`export_target_operator_feasibility.md`](export_target_operator_feasibility.md)에 있다. 재생성 가능한 TFLite, Vela output, CSV와 full log는 각각 `model/converted`, `model/quantized`, `model/vela`, `reports/generated`의 ignored 경계에 둔다.

M2.1 Research-owned batch-1 contract와 E84 revalidation 결과는 [`float_numerical_contract_resolution.md`](float_numerical_contract_resolution.md)에 있다.
