#!/usr/bin/env python3
"""Run one external stable_generalist method through local vendored code."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _runner_utils import REPO_ROOT, expose_output, reproduction_env, run_logged, write_failure, write_manifest


RUNNER_DIR = REPO_ROOT / "vendor" / "stable_generalist_runners" / "scripts"

DESC_WEIGHT_FLAGS = {
    "warmup_epochs": "--warmup-epochs",
    "dynamic_weight_update_interval": "--dynamic-weight-update-interval",
    "dynamic_weight_momentum": "--dynamic-weight-momentum",
    "pseudo_label_method": "--pseudo-label-method",
    "weight_exponent": "--weight-exponent",
    "weight_n_clusters": "--weight-n-clusters",
    "density_knn_k": "--density-knn-k",
    "density_weight_exponent": "--density-weight-exponent",
    "density_weight_clip": "--density-weight-clip",
    "weight_fusion_mode": "--weight-fusion-mode",
    "cluster_density_alpha": "--cluster-density-alpha",
    "cluster_weight_power": "--cluster-weight-power",
    "density_weight_power": "--density-weight-power",
    "min_cell_weight": "--min-cell-weight",
    "max_cell_weight": "--max-cell-weight",
}

RARE_METHODS = {
    "scCAD": ("scCAD", "standard", "sccad"),
    "scCAD+Harmony": ("scCAD", "harmony", "sccad"),
    "scAIDE": ("scAIDE", "standard", "scaide"),
    "scAIDE+Harmony": ("scAIDE", "harmony", "scaide"),
    "GiniClust": ("GiniClust", "standard", "giniclust"),
    "GiniClust+Harmony": ("GiniClust", "harmony", "giniclust"),
    "CellSIUS": ("CellSIUS", "standard", "cellsius"),
    "CellSIUS+Harmony": ("CellSIUS", "harmony", "cellsius"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Method parameter as key=value or method:key=value. Used by external adapters when supported.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_params(values: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            _, text = text.split(":", 1)
        if "=" not in text:
            raise ValueError(f"Invalid --param value {raw!r}; expected key=value.")
        key, value = text.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Invalid --param value {raw!r}; empty key.")
        params[key] = value.strip()
    return params


def manifest_for(args: argparse.Namespace, output_dir: Path) -> Path:
    manifest = output_dir / "_runner" / "manifest.csv"
    write_manifest(
        manifest,
        dataset_key=str(args.dataset_key),
        data_path=Path(args.data).expanduser().resolve(),
        label_key=str(args.label_key),
        batch_key=str(args.batch_key),
        n_labels=int(args.n_labels),
    )
    return manifest


def run_desc(args: argparse.Namespace, output_dir: Path, *, weighted: bool = False) -> None:
    cmd = [
        str(args.python_bin),
        str(RUNNER_DIR / "run_desc.py"),
        "--data",
        str(Path(args.data).expanduser().resolve()),
        "--output",
        str(output_dir),
        "--seed",
        str(int(args.seed)),
        "--label-key",
        str(args.label_key),
        "--batch-key",
        str(args.batch_key),
        "--verbose",
    ]
    if weighted:
        cmd.append("--weighted")
        params = parse_params(args.param)
        for key, flag in DESC_WEIGHT_FLAGS.items():
            if key in params:
                cmd.extend([flag, params[key]])
        device = str(args.device).lower()
        if device.startswith("cuda"):
            cmd.append("--use-gpu")
            if ":" in device:
                cmd.extend(["--gpu-id", device.split(":", 1)[1]])
    run_logged(cmd, output_dir / "logs" / "run_desc.log", env=reproduction_env())


def run_rare(args: argparse.Namespace, output_dir: Path) -> None:
    method, variant, slug = RARE_METHODS[str(args.method)]
    work_root = output_dir / "_runner" / "rare"
    manifest = manifest_for(args, output_dir)
    cmd = [
        str(args.python_bin),
        str(RUNNER_DIR / "run_rare_cell_methods.py"),
        "--manifest",
        str(manifest),
        "--output-root",
        str(work_root),
        "--datasets",
        str(args.dataset_key),
        "--methods",
        method,
        "--variants",
        variant,
        "--seed",
        str(int(args.seed)),
        "--scib-n-jobs",
        str(int(args.scib_n_jobs)),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.verbose:
        cmd.append("--verbose")
    run_logged(cmd, output_dir / "logs" / f"run_{slug}_{variant}.log", env=reproduction_env())
    expose_output(work_root / variant / str(args.dataset_key) / slug, output_dir)


def run_deepscena(args: argparse.Namespace, output_dir: Path) -> None:
    work_root = output_dir / "_runner" / "deepscena"
    manifest = manifest_for(args, output_dir)
    cmd = [
        str(args.python_bin),
        str(RUNNER_DIR / "run_deepscena.py"),
        "--manifest",
        str(manifest),
        "--output-root",
        str(work_root),
        "--datasets",
        str(args.dataset_key),
        "--seed",
        str(int(args.seed)),
        "--scib-n-jobs",
        str(int(args.scib_n_jobs)),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.verbose:
        cmd.append("--verbose")
    run_logged(cmd, output_dir / "logs" / "run_deepscena.log", env=reproduction_env())
    expose_output(work_root / "standard" / str(args.dataset_key) / "deepscena", output_dir)


def run_deepscena_harmony(args: argparse.Namespace, output_dir: Path) -> None:
    manifest = manifest_for(args, output_dir)
    base_root = output_dir / "_runner" / "deepscena_base"
    harmony_root = output_dir / "_runner" / "deepscena_harmony"

    base_cmd = [
        str(args.python_bin),
        str(RUNNER_DIR / "run_deepscena.py"),
        "--manifest",
        str(manifest),
        "--output-root",
        str(base_root),
        "--datasets",
        str(args.dataset_key),
        "--seed",
        str(int(args.seed)),
    ]
    if args.overwrite:
        base_cmd.append("--overwrite")
    harmony_cmd = [
        str(args.python_bin),
        str(RUNNER_DIR / "run_deepscena_harmony.py"),
        "--manifest",
        str(manifest),
        "--base-root",
        str(base_root),
        "--output-root",
        str(harmony_root),
        "--datasets",
        str(args.dataset_key),
        "--seed",
        str(int(args.seed)),
        "--scib-n-jobs",
        str(int(args.scib_n_jobs)),
    ]
    if args.overwrite:
        harmony_cmd.append("--overwrite")
    if args.verbose:
        base_cmd.append("--verbose")
        harmony_cmd.append("--verbose")
    env = reproduction_env(extra_paths=[RUNNER_DIR])
    run_logged(base_cmd, output_dir / "logs" / "run_deepscena_base.log", env=env)
    run_logged(harmony_cmd, output_dir / "logs" / "run_deepscena_harmony.log", env=env)
    expose_output(harmony_root / "harmony" / str(args.dataset_key) / "deepscena", output_dir)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        method = str(args.method)
        if method == "DESC":
            run_desc(args, output_dir)
        elif method == "DESC_scRAW_weighted":
            run_desc(args, output_dir, weighted=True)
        elif method == "DeepScena":
            run_deepscena(args, output_dir)
        elif method == "DeepScena+Harmony":
            run_deepscena_harmony(args, output_dir)
        elif method in RARE_METHODS:
            run_rare(args, output_dir)
        else:
            raise ValueError(f"Unsupported external method: {method}")
        failure_path = output_dir / "results" / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
    except Exception as exc:
        write_failure(output_dir / "results" / "failure.json", method=str(args.method), error=exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
