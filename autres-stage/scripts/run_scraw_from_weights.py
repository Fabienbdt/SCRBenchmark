#!/usr/bin/env python3
"""Run scRAW inference from an existing checkpoint artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAW_INDUCTIVE_SRC = REPO_ROOT / "vendor" / "scraw_inductive" / "src"
if str(SCRAW_INDUCTIVE_SRC) not in sys.path:
    sys.path.insert(0, str(SCRAW_INDUCTIVE_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["transductive", "inductive"], default="transductive")
    parser.add_argument("--config", required=True, help="config_used.json or scRAW config YAML/JSON.")
    parser.add_argument("--checkpoint", required=True, help="autoencoder.pt checkpoint.")
    parser.add_argument("--data", required=True, help="Input .h5ad file to encode.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--label-key", default=None)
    parser.add_argument("--preprocessing-state", default=None)
    parser.add_argument("--centroid-reference", default=None)
    parser.add_argument(
        "--no-filter-cells",
        action="store_true",
        help="Inductive mode only: keep all cells during preprocessing-state transform.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "transductive":
        from scraw import run_inference_from_checkpoint

        result = run_inference_from_checkpoint(
            config=args.config,
            checkpoint_path=args.checkpoint,
            data_path=args.data,
            output_dir=args.output,
            device=args.device,
        )
    else:
        if not args.preprocessing_state or not args.centroid_reference:
            raise ValueError(
                "Inductive mode requires --preprocessing-state and --centroid-reference."
            )
        from scraw import run_inductive_prediction

        result = run_inductive_prediction(
            config=args.config,
            checkpoint_path=args.checkpoint,
            preprocessing_state_path=args.preprocessing_state,
            centroid_reference_path=args.centroid_reference,
            data_path=args.data,
            output_dir=args.output,
            device=args.device,
            label_key=args.label_key,
            filter_cells=not args.no_filter_cells,
        )

    print(f"output_dir = {result['output_dir']}", flush=True)
    print(f"mode = {result['mode']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
