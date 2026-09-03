#!/usr/bin/env python3
"""Canonical command-line entry point for E84 deployment workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FastReflex E84 deployment CLI")
    subparsers = parser.add_subparsers(dest="command")
    verify = subparsers.add_parser(
        "verify-reference", help="validate handoff and run layered host-Float parity"
    )
    verify.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/deployment/reference_model.yaml",
    )
    export = subparsers.add_parser(
        "evaluate-export",
        help="export frozen members and evaluate TFLite/Vela operator feasibility",
    )
    export.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/deployment/reference_model.yaml",
    )
    export.add_argument(
        "--output-root",
        type=Path,
        help="override generated output root (primarily for isolated verification)",
    )
    quantize = subparsers.add_parser(
        "evaluate-int8",
        help="quantize all members and evaluate formal INT8 parity and Vela mapping",
    )
    quantize.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/deployment/reference_model.yaml",
    )
    quantize.add_argument(
        "--output-root",
        type=Path,
        help="override generated output root (primarily for isolated verification)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "verify-reference":
            from fastreflex_e84.handoff import verify_reference_handoff

            result = verify_reference_handoff(REPOSITORY_ROOT, args.config)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "evaluate-export":
            from fastreflex_e84.conversion import evaluate_export_feasibility

            result = evaluate_export_feasibility(
                REPOSITORY_ROOT, args.config, args.output_root
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["float_export"]["parity"]["status"] == "PASS" else 2
        if args.command == "evaluate-int8":
            from fastreflex_e84.conversion import (
                evaluate_int8_quantization,
            )

            result = evaluate_int8_quantization(
                REPOSITORY_ROOT, args.config, args.output_root
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return (
                0
                if result["status"] == "INT8_QUANTIZATION_AND_PARITY_PASS"
                else 2
            )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
