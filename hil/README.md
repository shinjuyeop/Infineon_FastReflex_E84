# Hardware-in-the-loop

`host/replay.py` replays the frozen non-protected deployment chain through the
KitProg3 virtual COM link in three modes: normalized `[20,80]` windows,
normalized 80D features, and raw pelvis IMU6 samples. The shared implementation
under `src/fastreflex_e84/` owns framing, CRC32, reset/sequence tracking,
streaming preprocessing, timing summaries, parity, and fail-closed evaluation.

```bash
python hil/host/replay.py --port /dev/ttyACM0 --mode window --rate-hz 10
python hil/host/replay.py --port /dev/ttyACM0 --mode feature --rate-hz 100
python hil/host/replay.py --port /dev/ttyACM0 --mode raw --rate-hz 1000
```

The runner exits 2 when any result is missing, a sequence/counter/deadline gate
fails, or strict host-prototype numerical parity fails. This does not redefine
the formal M3 contract. Actual board results and the measured 900 Hz reliable
raw rate are in `reports/e84_hil_runtime_validation.md`.
