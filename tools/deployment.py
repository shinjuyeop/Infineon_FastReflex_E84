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
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
