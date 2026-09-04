# FastReflex E84 non-release HIL firmware

This ModusToolbox application runs the frozen
`NON_RELEASE_HIL_PATH_PROTOTYPE` on the Cortex-M55 and Ethos-U55 of
`KIT_PSE84_AI`. It is deployment-path evidence only. The formal status remains
`INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL`; M4 and real-robot use are
not authorized.

## Pinned base and tools

- Infineon `mtb-example-psoc-edge-ml-profiler` `release-v2.3.0`, commit
  `625e450523bd653eb02ef449b691c8679e28beb2`
- `TARGET_KIT_PSE84_AI` `release-v1.4.0`, commit
  `4894dd6a1ac2cee8be0d028476da4c7e9e640431`
- ModusToolbox 3.8, GCC Arm 14.2.1, ML middleware 3.4.0
- Expected device `PSE846GPS2DBZC4A`; target `APP_KIT_PSE84_AI`

Library sources downloaded by `make getlibs` live in the ignored sibling
directory `firmware/mtb_shared/`. The target BSP is checked in so board and
memory configuration remain explicit. Generated model C blobs, build output,
and signed images are ignored.

## Prepare and build

From the repository root, stage the checksum-verified Vela artifacts and
normalizer:

```bash
python tools/deployment.py stage-firmware-assets
```

Then build the three-domain Release application:

```bash
cd firmware/fastreflex_e84
make getlibs CY_TOOLS_DIR=/opt/Tools/ModusToolbox/tools_3.8
make build CY_TOOLS_DIR=/opt/Tools/ModusToolbox/tools_3.8
```

Use `make clean CY_TOOLS_DIR=/opt/Tools/ModusToolbox/tools_3.8` to remove build
products. Staging can be rerun at any time; it rejects source or compiled-model
hash drift before replacing the ignored generated directory.

## Flash and serial

Select the probe explicitly when more than one kit is attached:

```bash
make qprogram \
  CY_TOOLS_DIR=/opt/Tools/ModusToolbox/tools_3.8 \
  MTB_PROBE_SERIAL=13070E98012D2400
```

The HIL channel is the KitProg3 virtual COM port at 1,000,000 baud, 8N1. It is
a CRC32-protected little-endian binary protocol, so a text serial monitor is
only useful for checking that the port opens. The host runner is the supported
monitor/client:

```bash
python hil/host/replay.py --port /dev/ttyACM0 --mode window --rate-hz 10
python hil/host/replay.py --port /dev/ttyACM0 --mode feature --rate-hz 100
python hil/host/replay.py --port /dev/ttyACM0 --mode raw --rate-hz 1000
```

Packet types carry a complete normalized `[20,80]` window, one normalized 80D
feature, one raw pelvis IMU6 sample, or a deterministic state reset. Every
result carries the sequence/window endpoint, three member probabilities,
ensemble/decision state, per-stage timing, status, and cumulative integrity
counters.

## Runtime

The raw path implements the frozen causal 10D base, 1/5/10 ms deltas, 5/10 ms
means and variances, normalizer, 20-sample float/INT8 rings, three sequential
U55 invocations, arithmetic mean, threshold 0.99, and five-sample persistence.
The INT8 ring quantizes only the new 80D row; the direct-window diagnostic path
quantizes all 1,600 values and is therefore intentionally slower.

Measured build, flash, boot, memory, parity, and HIL results are frozen in
`reports/e84_firmware_integration.md` and
`reports/e84_hil_runtime_validation.md`.
