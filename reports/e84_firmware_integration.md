# E84 firmware integration

## Verdict

`E84_BOARD_EXECUTION_PASS`

The Release firmware builds, flashes, boots, initializes all three models, runs
all members on the U55 without inference errors or hard faults, and completes a
full deployment trace. This is an engineering execution gate only. Formal M3
remains `INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL`; M4, production,
real-robot, and safety use remain unauthorized.

## Reproducible project

The canonical project is `firmware/fastreflex_e84`, derived from Infineon's
`mtb-example-psoc-edge-ml-profiler` release-v2.3.0
(`625e450523bd653eb02ef449b691c8679e28beb2`) with BSP release-v1.4.0
(`4894dd6a1ac2cee8be0d028476da4c7e9e640431`). The tested toolchain is
ModusToolbox 3.8 and GCC Arm 14.2.1.

The staging command verifies all source and target-Vela hashes, generates the
normalizer arrays, and places ignored C blobs under `proj_cm55/mtb_ml_gen`:

```bash
python tools/deployment.py stage-firmware-assets
cd firmware/fastreflex_e84
make build CY_TOOLS_DIR=/opt/Tools/ModusToolbox/tools_3.8
```

The Release build completed for CM33 secure, CM33 non-secure, and CM55/U55.
Generated build products, downloaded libraries, model conversions, and signed
images are not source-controlled.
The measured build produced combined-HEX SHA-256 `166ba89d...f42cc` and CM55
ELF SHA-256 `cd31bf92...cc30`; these identify the ignored physical artifacts
used for the final flash without moving them into the source tree.

## Actual flash and boot

The selected probe was `13070E98012D2400` (`/dev/ttyACM0`); the other two
attached kits were not programmed. OpenOCD detected `PSE846GPS2DBZC4A`, silicon
`0xED94`, family `0x115`, revision B0, lifecycle `DEVELOPMENT`, VTarget 1.801 V,
and `CYBOOT_SUCCESS`. Programming and verification completed successfully for
976,436 bytes. KitProg firmware 2.80.1529 is older than the installed 2.81.1663
image; it was deliberately not updated as part of this task.

## Firmware architecture

- CM33 secure performs the BSP/extended-boot security setup.
- CM33 non-secure starts CM55 and then sleeps.
- CM55 receives CRC32-framed packets over KitProg3 UART at 1 Mbps.
- Window mode quantizes a received normalized `[20,80]` diagnostic vector.
- Feature/raw modes maintain a rolling float and INT8 ring and quantize only the
  newest 80D row.
- Three `ethos-u55-128` models run sequentially with zero CPU fallback, followed
  by the fixed arithmetic mean, threshold 0.99, and five-sample persistence.

The middleware's `CONDITIONAL` cache policy is used. It performs whole-cache
maintenance once per invocation and is materially faster than the per-layer
policy measured during bring-up. No model, threshold, persistence, seed,
quantization range, or output calibration was changed from the frozen
prototype contract.

## Actual linker memory

| Region/item | Actual/configured bytes |
|---|---:|
| Total SMIF image use | 977,456 |
| CM55 NVM | 671,052 |
| `.cy_socmem_data` model section | 352,680 |
| Compiled model payloads | 352,592 |
| CM55 DTCM static use | 29,120 |
| CM55 ITCM use | 1,496 |
| SoC-memory heap capacity | 2,514,520 |
| Three configured model arenas | 49,152 (3 × 16,384) |
| CM55 stack reservation | 4,096 |

The static preprocessor object is 8,416 bytes: 408 bytes of base history/state,
a 6,400-byte float ring, a 1,600-byte INT8 ring, and counters. Separate static
buffers are 6,400 bytes for the largest UART payload, 6,400 bytes for the direct
float window, and 1,600 bytes for model input. Arena sizes are model metadata;
the runtime heap high-water mark was not instrumented, so the report does not
claim measured dynamic peak usage.

Compiler estimates remain separate in `e84_target_vela_compilation.md`; actual
runtime measurements are in `e84_hil_runtime_validation.md`.
