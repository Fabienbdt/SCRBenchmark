#!/usr/bin/env python3
"""Validate that a SCRBenchmark method spec is ready to run."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from _runner_utils import REPO_ROOT

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scrbenchmark.methods import get_method_spec  # noqa: E402
from run_method import (  # noqa: E402
    DEFAULT_RESOLUTIONS,
    _join,
    _source_path,
    build_command,
    expected_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, help="Method name or alias from methods/*.yaml.")
    parser.add_argument("--data", default="data/your_data.h5ad", help="Small .h5ad file used for validation.")
    parser.add_argument("--output", default="", help="Output directory for --run.")
    parser.add_argument("--dataset-key", default="")
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scib-n-jobs", type=int, default=1)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--harmony-max-iter", type=int, default=10)
    parser.add_argument("--harmony-nclust", type=int, default=50)
    parser.add_argument("--resolutions", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--selection-expected-n-classes", type=int, default=0)
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run", action="store_true", help="Actually execute the method on --data.")
    return parser.parse_args()


def _run_args(args: argparse.Namespace) -> list[str]:
    output = args.output or str(Path("/tmp") / f"scrbenchmark_validate_{args.method}")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "reproduction" / "run_method.py"),
        "--method",
        args.method,
        "--data",
        args.data,
        "--output",
        output,
        "--dataset-key",
        args.dataset_key,
        "--label-key",
        args.label_key,
        "--batch-key",
        args.batch_key,
        "--n-labels",
        str(int(args.n_labels)),
        "--seed",
        str(int(args.seed)),
        "--device",
        args.device,
        "--scib-n-jobs",
        str(int(args.scib_n_jobs)),
        "--python-bin",
        args.python_bin,
        "--n-top-genes",
        str(int(args.n_top_genes)),
        "--min-genes-per-cell",
        str(int(args.min_genes_per_cell)),
        "--max-genes-per-cell",
        str(int(args.max_genes_per_cell)),
        "--min-cells-per-gene",
        str(int(args.min_cells_per_gene)),
        "--target-sum",
        str(float(args.target_sum)),
        "--scale-max-value",
        str(float(args.scale_max_value)),
        "--hvg-flavor",
        args.hvg_flavor,
        "--n-pcs",
        str(int(args.n_pcs)),
        "--harmony-max-iter",
        str(int(args.harmony_max_iter)),
        "--harmony-nclust",
        str(int(args.harmony_nclust)),
        "--resolutions",
        args.resolutions,
        "--selection-expected-n-classes",
        str(int(args.selection_expected_n_classes)),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.verbose:
        command.append("--verbose")
    for param in args.param or []:
        command.extend(["--param", str(param)])
    return command


def _namespace_for_build(args: argparse.Namespace) -> argparse.Namespace:
    output = args.output or str(Path("/tmp") / f"scrbenchmark_validate_{args.method}")
    values = vars(args).copy()
    values["output"] = output
    values["dry_run"] = True
    return argparse.Namespace(**values)


def main() -> int:
    args = parse_args()
    spec = get_method_spec(args.method)
    if spec is None:
        raise SystemExit(f"Unknown method {args.method!r}. Check methods/*.yaml or run run_method.py --list.")

    build_args = _namespace_for_build(args)
    print(f"Method: {spec.name}")
    print(f"Runner: {spec.runner_kind}")
    print(f"Report UI: {'yes' if spec.report else 'no'}")

    source_path = _source_path(spec)
    if source_path:
        if Path(source_path).exists():
            print(f"Source: {source_path}")
        else:
            print(f"Warning: source path does not exist yet: {source_path}")

    command = build_command(spec, build_args)
    print()
    print("Generated command:")
    print(_join(command))

    if not args.run:
        print()
        print("Dry validation passed. Add --run to execute the method on the selected dataset.")
        return 0

    if int(args.n_labels) <= 0:
        raise SystemExit("--n-labels must be greater than 0 for an execution test.")

    run_command = _run_args(args)
    print()
    print("Running smoke test:")
    print(_join(run_command))
    completed = subprocess.run(run_command, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        return completed.returncode

    expected = expected_output_path(spec, build_args)
    if expected.exists():
        print(f"Output found: {expected}")
        return 0

    raise SystemExit(f"Run finished but expected output was not found: {expected}")


if __name__ == "__main__":
    raise SystemExit(main())
