#!/usr/bin/env python3
"""Run the self-contained vendored scRAW backend through the method registry."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRAW_SOURCE_DIR = REPO_ROOT / "vendor" / "scraw_inductive" / "src"
if not SCRAW_SOURCE_DIR.exists():
    raise RuntimeError(f"Vendored scRAW source directory is missing: {SCRAW_SOURCE_DIR}")
sys.path.insert(0, str(SCRAW_SOURCE_DIR))

from scraw.pipeline import run_pipeline
from scraw.presets import resolve_preset_config


PUBLIC_PRESETS = ("baron", "default")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the vendored scRAW backend")
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
        choices=PUBLIC_PRESETS,
        default="default",
        help=(
            "Vendored public preset. `default` is the 0017/stable configuration; "
            "`baron` is the Baron configuration."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = resolve_preset_config(str(args.preset))

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

    # Execute the vendored pipeline.
    run_pipeline(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
