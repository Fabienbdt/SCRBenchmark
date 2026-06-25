#!/usr/bin/env python3
"""Run SCRBenchmark CLI algorithms in leave-one-batch-out mode."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "src" / "scrbenchmark" / "cli.py"


PLAN_FIELDS = [
    "dataset_name",
    "split",
    "train_batches",
    "test_batch",
    "algorithms",
    "output_dir",
    "status",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Input .h5ad file.")
    parser.add_argument("--dataset-name", default="", help="Name written to plan/status CSV.")
    parser.add_argument("--algorithms", default="scname,sc_mae,scdeepcluster")
    parser.add_argument("--label-col", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "leave_one_batch" / "scrbenchmark"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Optional held-out batches to run. Defaults to all batches in the file.",
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--compute-scib", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-embeddings", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_batches(data_path: str, batch_key: str) -> list[str]:
    import scanpy as sc

    adata = sc.read_h5ad(data_path, backed="r")
    try:
        if batch_key not in adata.obs.columns:
            raise ValueError(f"Batch key {batch_key!r} was not found in adata.obs.")
        return sorted(adata.obs[batch_key].astype(str).unique().tolist())
    finally:
        adata.file.close()


def _safe_split_name(train_batches: Sequence[str], test_batch: str) -> str:
    split = "train_" + "-".join(train_batches) + "_to_" + test_batch
    return split.replace("/", "_").replace(" ", "_")


def build_command(args: argparse.Namespace, train_batches: list[str], test_batch: str, out_dir: Path) -> list[str]:
    cmd = [
        str(args.python_bin),
        str(CLI_PATH),
        "run",
        "--data",
        str(Path(args.data_path).expanduser().resolve()),
        "--algorithms",
        str(args.algorithms),
        "--output",
        str(out_dir),
        "--no-timestamp",
        "--benchmark-mode",
        "--train-ratio",
        "0.8",
        "--val-ratio",
        str(float(args.val_ratio)),
        "--test-ratio",
        "0.2",
        "--stratify-by",
        str(args.batch_key),
        "--stratify-mode",
        "batch+labels",
        "--train-batches",
        ",".join(train_batches),
        "--test-batches",
        str(test_batch),
        "--label-col",
        str(args.label_col),
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
        str(args.hvg_flavor),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--n-repeats",
        "1",
        "--n-clusters",
        "0",
        "--csv",
        "--save-labels",
    ]
    if args.save_embeddings:
        cmd.append("--save-embeddings")
    if not args.compute_scib:
        cmd.append("--no-scib-metrics")
    return cmd


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PLAN_FIELDS})


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    dataset_name = args.dataset_name or data_path.stem
    available_batches = read_batches(str(data_path), args.batch_key)
    if len(available_batches) < 2:
        raise ValueError("Need at least two batches for a leave-one-batch run.")
    if args.batches is None:
        test_batches = available_batches
    else:
        test_batches = list(dict.fromkeys(str(batch) for batch in args.batches))
        missing = sorted(set(test_batches) - set(available_batches))
        if missing:
            raise ValueError(f"Unknown held-out batch values for {args.batch_key!r}: {missing}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "planned_jobs.csv"
    status_path = output_root / "run_status.csv"

    env = os.environ.copy()
    py_paths = [str(REPO_ROOT / "src"), str(REPO_ROOT / "src" / "scrbenchmark")]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    rows: list[dict[str, Any]] = []
    for test_batch in test_batches:
        train_batches = [batch for batch in available_batches if batch != test_batch]
        split = _safe_split_name(train_batches, test_batch)
        out_dir = output_root / split
        command = build_command(args, train_batches, test_batch, out_dir)
        command_str = " ".join(shlex.quote(part) for part in command)
        expected = out_dir / "results" / "benchmark_summary.csv"

        status = "planned"
        if args.skip_existing and expected.exists():
            status = "existing"
        elif not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            log_file = out_dir / "run.log"
            with log_file.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            status = "ok" if completed.returncode == 0 and expected.exists() else f"failed_{completed.returncode}"

        row = {
            "dataset_name": dataset_name,
            "split": split,
            "train_batches": ",".join(train_batches),
            "test_batch": test_batch,
            "algorithms": args.algorithms,
            "output_dir": str(out_dir),
            "status": status,
            "command": command_str,
        }
        rows.append(row)
        print(f"[{split}] {status}", flush=True)

    write_rows(plan_path, rows)
    write_rows(status_path, rows)
    print(f"plan = {plan_path}", flush=True)
    print(f"status = {status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
