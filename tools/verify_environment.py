#!/usr/bin/env python3
"""Minimal environment check for the E84 deployment scaffold."""

from __future__ import annotations

import platform
import sys


MINIMUM_PYTHON = (3, 10)


def main() -> int:
    current = sys.version_info[:2]
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print("Project status: PROJECT_SCAFFOLD_ONLY")

    if current < MINIMUM_PYTHON:
        print("Result: Python 3.10 or newer is required.")
        return 1

    print("Result: placeholder environment check passed.")
    print("E84 SDK, Vela, firmware, and HIL checks are not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
