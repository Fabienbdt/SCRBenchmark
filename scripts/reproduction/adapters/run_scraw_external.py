#!/usr/bin/env python3
"""
SCRBenchmark adapter to run the external scRAW package at /data2/fbidet/scRAW.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add external scRAW source directory to Python path
SCRAW_ROOT = Path("/data2/fbidet/scRAW")
SCRAW_SOURCE_DIR = SCRAW_ROOT / "src"
if SCRAW_SOURCE_DIR.exists():
    sys.path.insert(0, str(SCRAW_SOURCE_DIR))
else:
    raise RuntimeError(f"Could not find scRAW source directory at {SCRAW_SOURCE_DIR}")

SCRAW_PRESET_CONFIGS = {
    "default": SCRAW_ROOT / "configs" / "default_scraw.json",
    "baron": SCRAW_ROOT / "configs" / "baron_jobim.json",
}

from scraw.config import load_config
from scraw.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run scRAW externally")
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preset",
        choices=sorted(SCRAW_PRESET_CONFIGS),
        default="default",
        help=(
            "scRAW source configuration to load from /data2/fbidet/scRAW. "
            "`default` is the 0017/stable configuration; `baron` is the Baron configuration."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = SCRAW_PRESET_CONFIGS[str(args.preset)]
    if not config_path.exists():
        raise FileNotFoundError(f"Missing scRAW preset config: {config_path}")

    config = load_config(config_path)

    # Configure path values
    config.data.data_path = str(Path(args.data).expanduser().resolve())
    config.data.output_dir = str(Path(args.output).expanduser().resolve())
    config.data.label_key = str(args.label_key)

    # Configure runtime properties
    config.runtime.seed = int(args.seed)
    config.runtime.device = str(args.device)

    # Target number of clusters
    if int(args.n_labels) > 0:
        config.clustering.pseudo_k = int(args.n_labels)

    # Configure batch correction
    if args.batch_key:
        config.batch_correction.key = str(args.batch_key)
        config.batch_correction.enabled = True
    else:
        config.batch_correction.enabled = False

    # Execute the external pipeline
    run_pipeline(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
