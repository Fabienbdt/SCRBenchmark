#!/usr/bin/env python3
"""
SCRBenchmark adapter to run the external scRAW package at /data2/fbidet/scRAW.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add external scRAW source directory to Python path
SCRAW_SOURCE_DIR = Path("/data2/fbidet/scRAW/src")
if SCRAW_SOURCE_DIR.exists():
    sys.path.insert(0, str(SCRAW_SOURCE_DIR))
else:
    raise RuntimeError(f"Could not find scRAW source directory at {SCRAW_SOURCE_DIR}")

from scraw.config import ScRAWConfig
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load or instantiate a default ScRAWConfig
    config = ScRAWConfig()

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
