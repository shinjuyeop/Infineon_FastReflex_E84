# E84 target-specific Vela compilation

## Verdict

`E84_TARGET_VELA_COMPILATION_PASS`

This is a compiler/integration gate for the frozen
`NON_RELEASE_HIL_PATH_PROTOTYPE`. It does not change
`INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL`, does not authorize M4,
and is not scientific or release evidence.

## Verified target

- Board/BSP: `KIT_PSE84_AI`, device `PSE846GPS2DBZC4A`, BSP
  `release-v1.4.0` at `4894dd6a1ac2cee8be0d028476da4c7e9e640431`
- Reference application: Infineon `mtb-example-psoc-edge-ml-profiler`
  `release-v2.3.0` at `625e450523bd653eb02ef449b691c8679e28beb2`
- ML Pack: `3.0.0.2416`, artifact
  `708b03a5-1c48-4fa8-9069-1687d9bc900f`, Linux package SHA-256
  `1d1de67afddc15b9603fde8ba74027ec35490a896467a2dbf3c3b24434410b7e`
- ML CoreTools: `3.0.0.8948`; bundled Vela: `4.3.0`
- Target: `PSE84_M55_U55`; accelerator: `ethos-u55-128`
- System: `PSE8x_U55_400MHz_SOCMEM_200MHz_QUAD_XIP`
- Memory mode: `Sram_Only`; M55/U55 400 MHz, SoC memory 200 MHz
- Vela options: 2,936,012-byte arena cache, block dependency 3,
  `Performance`, recursion 1000, `HillClimb`

The exact system and memory-mode parameters are checked in at
`configs/vela/pse84_u55.ini`. They were extracted from the pinned Infineon ML
Pack, not inferred from generic Arm defaults.

## Results

| Seed | CPU | NPU | MACs | Compiled bytes | Command bytes | Encoded weights | Total cycles | Peak Vela SRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260828 | 0 | 472 | 212,676 | 117,504 | 33,776 | 11.84 KiB | 38,077 | 3.50 KiB |
| 20260829 | 0 | 472 | 212,676 | 117,280 | 33,804 | 11.84 KiB | 38,077 | 3.50 KiB |
| 20260830 | 0 | 472 | 212,676 | 117,808 | 33,820 | 11.84 KiB | 38,077 | 3.50 KiB |
| **Sequential total** | **0** | **1,416** | **638,028** | **352,592** | **101,400** | **35.52 KiB** | **114,231** | **3.50 KiB reused** |

The compiler estimates 0.10 ms per member and 0.30 ms for the three sequential
invocations. Each generated TFLM model reports a 16,384-byte tensor arena. The
exact compiled-model SHA-256 values and per-member constant-buffer sizes are in
`e84_target_vela_compilation.json`.

## Placement interpretation

In `Sram_Only`, Vela reports constants as `OnChipFlash` using SRAM
characteristics. The Infineon firmware integration places generated model blobs
in `.cy_socmem_data`; the application images execute from external QSPI XIP.
These compiler estimates are distinct from linker-map and physical runtime
measurements, which remain firmware/board gates.

## Reproduction

Install ML Pack `3.0.0.2416` with Infineon Launcher, then run:

```bash
ML_CORETOOLS=/opt/Tools/ModusToolbox/packs/ModusToolbox-Machine-Learning-Pack/tools/ml-coretools/ml-coretools
PYTHONPATH=src python tools/deployment.py compile-target-vela \
  --ml-coretools "$ML_CORETOOLS"
```

Generated models, C blobs, logs, and QEMU arena probes remain under the ignored
`model/vela/non_release_hil_prototype/` boundary. The checked-in JSON report is
the immutable machine-readable result; it records every generated path, hash,
command, and translated Vela option.
