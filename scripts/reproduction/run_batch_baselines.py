#!/usr/bin/env python3
"""Thin entry point for the vendored stable_generalist batch-baseline runner."""

from __future__ import annotations

import runpy
from pathlib import Path
import sys

from _runner_utils import REPO_ROOT


RUNNER = REPO_ROOT / "vendor" / "stable_generalist_runners" / "scripts" / "run_batch_baselines.py"


def main() -> int:
    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing vendored runner: {RUNNER}")
    runner_dir = str(RUNNER.parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    runpy.run_path(str(RUNNER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
