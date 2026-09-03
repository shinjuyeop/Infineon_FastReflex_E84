# Tools

Quantization, target conversion, Vela 및 artifact verification의 canonical 도구를 둔다. 기능별 version suffix가 붙은 script를 만들지 않고 configuration으로 차이를 표현한다.

- `deployment.py verify-reference`: accepted release checksum/contract와 독립 Host Float layered parity를 fail-closed로 검증한다.
- `verify_environment.py`: Python, NumPy, PyYAML, PyTorch와 현재 milestone 상태를 확인한다.

```bash
python tools/deployment.py verify-reference
python tools/verify_environment.py
```

Vendor SDK, target converter, Vela와 board 검사는 `EXPORT_AND_TARGET_OPERATOR_FEASIBILITY` 이후 필요한 canonical command에 추가한다.
