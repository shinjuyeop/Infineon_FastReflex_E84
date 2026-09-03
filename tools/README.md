# Tools

Quantization, target conversion, Vela 및 artifact verification의 canonical 도구를 둔다. 기능별 version suffix가 붙은 script를 만들지 않고 configuration으로 차이를 표현한다.

- `deployment.py verify-reference`: accepted release checksum/contract와 독립 Host Float layered parity를 fail-closed로 검증한다.
- `deployment.py evaluate-export`: three frozen member를 Float32 TFLite built-in graph로 내리고 golden parity를 측정한 뒤, 최소 INT8+softmax operator probe를 Vela/Ethos-U55-128에 컴파일한다. 생성 artifact와 log는 ignored deployment output 경계에 둔다.
- `verify_environment.py`: Python, NumPy, PyYAML, PyTorch와 현재 milestone 상태를 확인한다.

```bash
python tools/deployment.py verify-reference
python tools/deployment.py evaluate-export
python tools/verify_environment.py
```

`evaluate-export`는 먼저 M1 handoff를 다시 fail-closed 검증한다. Float export는 `[1,20,80] float32 -> [1,2] float32 logits`이고, INT8 artifact는 operator mapping 조사만을 위한 임시 probe다. Vela generic Arm config의 cycle number는 E84 board timing으로 해석하지 않는다.

현재 두 명령은 Research-owned batch-1 Float contract를 독립적으로 통과해 exit code 0을 반환한다. `evaluate-export`는 formal INT8 parity를 수행하지 않으며 M3용 operator evidence만 유지한다.
