#!/usr/bin/env python3
"""Environment check for the current E84 deployment milestone."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy
import torch
import yaml


MINIMUM_PYTHON = (3, 10)
M2_PACKAGES = ("tensorflow", "ai-edge-litert", "ethos-u-vela")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _kitprog_count() -> int:
    count = 0
    for vendor_path in Path("/sys/bus/usb/devices").glob("*/idVendor"):
        product_path = vendor_path.with_name("idProduct")
        if not product_path.is_file():
            continue
        if (
            vendor_path.read_text(encoding="ascii").strip().lower() == "04b4"
            and product_path.read_text(encoding="ascii").strip().lower() == "f155"
        ):
            count += 1
    return count


def main() -> int:
    current = sys.version_info[:2]
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("Project status: FLOAT_EXPORT_PARITY_FAIL_INT8_U55_OPERATOR_MAPPING_PASS")

    if current < MINIMUM_PYTHON:
        print("Result: Python 3.10 or newer is required.")
        return 1

    print(f"NumPy: {numpy.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"PyTorch: {torch.__version__}")
    missing: list[str] = []
    for package in M2_PACKAGES:
        version = _package_version(package)
        print(f"{package}: {version or 'NOT INSTALLED'}")
        if version is None:
            missing.append(package)
    vela = shutil.which("vela")
    print(f"Vela executable: {vela or 'NOT FOUND'}")
    if vela is not None:
        completed = subprocess.run(
            [vela, "--version"], capture_output=True, text=True, check=False
        )
        print(f"Vela CLI: {completed.stdout.strip() or completed.stderr.strip()}")
    else:
        missing.append("vela executable")
    print(f"KitProg3 USB devices (read-only enumeration): {_kitprog_count()}")
    if missing:
        print(f"Result: M2 environment incomplete: {', '.join(missing)}")
        return 1
    print("Result: M2 export/operator-analysis environment available.")
    print("Infineon ML Pack and board execution are not validated by this check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
