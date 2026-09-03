#!/usr/bin/env python3
"""Environment check for the current E84 deployment milestone."""

from __future__ import annotations

import platform
import sys

import numpy
import torch
import yaml


MINIMUM_PYTHON = (3, 10)


def main() -> int:
    current = sys.version_info[:2]
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("Project status: REFERENCE_MODEL_HANDOFF_AND_HOST_FLOAT_PARITY_PASS")

    if current < MINIMUM_PYTHON:
        print("Result: Python 3.10 or newer is required.")
        return 1

    print(f"NumPy: {numpy.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print("Result: M1 Host Float environment check passed.")
    print("Vendor SDK, Vela, firmware, and HIL checks have not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
