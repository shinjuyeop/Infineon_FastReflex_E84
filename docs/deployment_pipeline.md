# Deployment Pipeline

현재 milestone은 deployment 경계만 정의하며 실제 변환이나 target 실행을 하지 않는다.

```text
Reviewed Frozen Float Model + Contract
  -> Contract and checksum validation
  -> Quantization
  -> Target conversion
  -> Vela
  -> Firmware integration
  -> KIT_PSE84_AI / PSoC Edge E84
  -> HIL with golden vectors
  -> Runtime validation report
```

각 단계는 입력 artifact, tool version, configuration, 출력 checksum을 추적할 수 있어야 한다. 변환 결과는 source model과 분리하고, 같은 기능의 canonical tool 하나를 config로 구동한다.

Dataset 생성, training, architecture exploration은 이 pipeline 밖이며 research repository가 담당한다. Acceptance criteria와 실제 toolchain version은 frozen model 계약이 준비된 뒤 고정한다.
