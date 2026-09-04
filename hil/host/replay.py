#!/usr/bin/env python3
"""Replay frozen non-protected FastReflex data to the E84 prototype."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fastreflex_e84.hil import replay_packets  # noqa: E402
from fastreflex_e84.protocol import Packet, PayloadType  # noqa: E402
from fastreflex_e84.conversion import run_frozen_int8_prototype  # noqa: E402


BUNDLE = ROOT / "model/source/model_v2_anchor_refined_gru20_20260902"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="FastReflex E84 HIL replay")
    result.add_argument("--port", required=True, help="KitProg3 virtual COM device")
    result.add_argument("--mode", choices=("window", "feature", "raw"), required=True)
    result.add_argument("--rate-hz", type=float)
    result.add_argument("--baud", type=int, default=1_000_000)
    result.add_argument("--timeout", type=float, default=5.0)
    result.add_argument(
        "--limit", type=int, help="replay only the first N data packets"
    )
    result.add_argument("--output", type=Path)
    return result


def load_packets(
    mode: str, limit: int | None = None
) -> tuple[list[Packet], dict[str, np.ndarray]]:
    with np.load(
        BUNDLE / "golden_inputs/runtime_chain.npz", allow_pickle=False
    ) as data:
        raw = data["raw_pelvis_imu6"].copy()
    with np.load(
        BUNDLE / "golden_outputs/deployment_runtime_chain.npz", allow_pickle=False
    ) as data:
        golden = {name: data[name].copy() for name in data.files}
    if mode == "window":
        values = golden["model_windows"]
        endpoints = golden["window_endpoints"]
        kind = PayloadType.WINDOW_FLOAT32
    elif mode == "feature":
        values = golden["normalized_features"]
        endpoints = np.arange(len(values), dtype=np.int64)
        kind = PayloadType.FEATURE_FLOAT32
    else:
        values = raw
        endpoints = np.arange(len(values), dtype=np.int64)
        kind = PayloadType.RAW_IMU6_FLOAT32
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        values = values[:limit]
        endpoints = endpoints[:limit]
    packets = [Packet(PayloadType.RESET_STATE, sequence_id=0, window_endpoint=0)] + [
        Packet(
            kind,
            sequence_id=index + 1,
            window_endpoint=int(endpoint),
            payload=np.asarray(value, dtype="<f4").tobytes(),
        )
        for index, (endpoint, value) in enumerate(zip(endpoints, values))
    ]
    expected = run_frozen_int8_prototype(ROOT, golden["model_windows"])
    expected["window_endpoints"] = golden["window_endpoints"]
    return packets, expected


def main() -> int:
    args = parser().parse_args()
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("install the 'hil' optional dependency (pyserial)") from exc
    packets, expected = load_packets(args.mode, args.limit)
    default_rate = {"window": 10.0, "feature": 100.0, "raw": 1000.0}[args.mode]
    with serial.Serial(args.port, args.baud, timeout=0.01) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        report = replay_packets(
            port,
            packets,
            args.rate_hz or default_rate,
            args.timeout,
            expected=expected,
        )
    report.update(
        {
            "mode": args.mode,
            "requested_rate_hz": args.rate_hz or default_rate,
            "baud": args.baud,
            "role": "NON_RELEASE_HIL_PATH_PROTOTYPE",
            "formal_status": "INT8_PTQ_PARTIAL_RECOVERY_NUMERICAL_CONTRACT_FAIL",
        }
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
