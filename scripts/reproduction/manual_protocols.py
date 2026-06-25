#!/usr/bin/env python3
"""Build or execute customizable report-style protocol commands.

This launcher is intentionally manual: it exposes the reusable protocol blocks
used in the report without forcing the fixed report plans. It can generate or
run jobs for loss-transfer variants, Harmony-composed methods, and inductive
train/test-group experiments.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping

from _runner_utils import REPO_ROOT, reproduction_env


SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scrbenchmark.methods import get_method_spec, load_method_specs  # noqa: E402


REPORT_LOSS_WEIGHT_PARAMS = {
    "warmup_epochs": 30,
    "dynamic_weight_update_interval": 20,
    "dynamic_weight_momentum": 0.6884621079434989,
    "pseudo_label_method": "leiden",
    "weight_component_mode": "full",
    "weight_exponent": 0.2,
    "density_knn_k": 15,
    "density_weight_exponent": 1.0,
    "density_weight_clip": 3.0,
    "weight_fusion_mode": "multiplicative",
    "cluster_density_alpha": 0.3483603718613933,
    "cluster_weight_power": 1.0,
    "density_weight_power": 1.0,
    "min_cell_weight": 0.3845423008053828,
    "max_cell_weight": 10.0,
}

LOSS_METHODS = {
    "scMAE": ("scMAE", "scMAE_scRAW_weighted", "sc_mae_scraw_weighted"),
    "scDeepCluster": ("scDeepCluster", "scDeepCluster_scRAW_weighted", "scdeepcluster_scraw_weighted"),
    "DESC": ("DESC", "DESC_scRAW_weighted", "desc_scraw_weighted"),
}

LOSS_VARIANT_PRESETS = {
    "weighted": {},
    "density_only": {"weight_component_mode": "density_only"},
    "kmeans": {"pseudo_label_method": "kmeans"},
    "triplet": {"rare_triplet_weight": 0.05, "rare_triplet_start_epoch": 60},
}

DEFAULT_HARMONY_METHODS = [
    "Harmony",
    "scMAE+Harmony",
    "scNAME+Harmony",
    "scvi+Harmony",
    "DeepScena+Harmony",
    "CellSIUS+Harmony",
    "GiniClust+Harmony",
    "scAIDE+Harmony",
    "scCAD+Harmony",
]

DEFAULT_INDUCTIVE_ALGORITHMS = ["scraw", "scname", "sc_mae", "scdeepcluster"]

PLAN_FIELDS = ["protocol", "job_id", "method", "seed", "output_dir", "command"]


@dataclass(frozen=True)
class ManualProtocolConfig:
    protocol: str
    data: str
    output_root: str
    dataset_key: str = ""
    label_key: str = "Group"
    batch_key: str = "batch"
    n_labels: int = 0
    seeds: tuple[int, ...] = (42,)
    device: str = "cuda"
    python_bin: str = sys.executable
    scib_n_jobs: int = 4
    n_top_genes: int = 2000
    min_genes_per_cell: int = 200
    max_genes_per_cell: int = 10000
    min_cells_per_gene: int = 3
    target_sum: float = 20000.0
    scale_max_value: float = 10.0
    hvg_flavor: str = "seurat"
    resolutions: str = "0.2,0.4,0.6,0.8,1.0,1.2,1.4"
    selection_expected_n_classes: int = 0
    params: tuple[str, ...] = ()
    overwrite: bool = False
    verbose: bool = False
    loss_methods: tuple[str, ...] = ("scMAE", "scDeepCluster", "DESC")
    loss_variants: tuple[str, ...] = ("baseline", "weighted")
    loss_weight_params: Mapping[str, Any] = field(default_factory=dict)
    harmony_methods: tuple[str, ...] = ("Harmony", "scMAE+Harmony", "scNAME+Harmony", "scvi+Harmony")
    harmony_max_iter: int = 10
    harmony_nclust: int = 50
    n_pcs: int = 50
    inductive_algorithms: tuple[str, ...] = tuple(DEFAULT_INDUCTIVE_ALGORITHMS)
    split_key: str = "batch"
    train_split_key: str = ""
    test_split_key: str = ""
    train_batches: tuple[str, ...] = ()
    test_batches: tuple[str, ...] = ()
    preset: str = "stable_generalist"
    trial_config_path: str = ""
    baseline_runtime_profile: str = "scrbenchmark-default"
    skip_existing: bool = False


def _tokens(raw: str | Iterable[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = []
        for item in raw:
            parts.extend(str(item).split(","))
    return [part.strip() for part in parts if part and part.strip()]


def parse_seeds(raw: str | Iterable[str] | None) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in _tokens(raw):
        if "-" in token:
            start_raw, end_raw = token.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            step = 1 if end >= start else -1
            seeds.extend(range(start, end + step, step))
        else:
            seeds.append(int(token))
    return tuple(dict.fromkeys(seeds)) or (42,)


def _safe_slug(value: Any) -> str:
    text = str(value).strip()
    out = []
    for char in text:
        out.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(out).strip("_") or "item"


def _join(cmd: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _parse_key_values(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid key=value override: {raw!r}")
        key, value = text.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Invalid key=value override: {raw!r}")
        parsed[key] = value.strip()
    return parsed


def _dataset_key(config: ManualProtocolConfig) -> str:
    return config.dataset_key or Path(config.data).stem


def _base_run_method_command(
    config: ManualProtocolConfig,
    *,
    method: str,
    output_dir: Path,
    seed: int,
    params: Iterable[str] = (),
) -> list[str]:
    cmd = [
        str(config.python_bin),
        str(REPO_ROOT / "scripts" / "reproduction" / "run_method.py"),
        "--method",
        method,
        "--data",
        str(Path(config.data).expanduser().resolve()),
        "--output",
        str(output_dir),
        "--dataset-key",
        _dataset_key(config),
        "--label-key",
        config.label_key,
        "--batch-key",
        config.batch_key,
        "--n-labels",
        str(int(config.n_labels)),
        "--seed",
        str(int(seed)),
        "--device",
        config.device,
        "--scib-n-jobs",
        str(int(config.scib_n_jobs)),
        "--n-top-genes",
        str(int(config.n_top_genes)),
        "--min-genes-per-cell",
        str(int(config.min_genes_per_cell)),
        "--max-genes-per-cell",
        str(int(config.max_genes_per_cell)),
        "--min-cells-per-gene",
        str(int(config.min_cells_per_gene)),
        "--target-sum",
        str(float(config.target_sum)),
        "--scale-max-value",
        str(float(config.scale_max_value)),
        "--hvg-flavor",
        config.hvg_flavor,
        "--n-pcs",
        str(int(config.n_pcs)),
        "--harmony-max-iter",
        str(int(config.harmony_max_iter)),
        "--harmony-nclust",
        str(int(config.harmony_nclust)),
        "--resolutions",
        config.resolutions,
    ]
    if int(config.selection_expected_n_classes) > 0:
        cmd.extend(["--selection-expected-n-classes", str(int(config.selection_expected_n_classes))])
    for param in params:
        cmd.extend(["--param", str(param)])
    if config.overwrite:
        cmd.append("--overwrite")
    if config.verbose:
        cmd.append("--verbose")
    return cmd


def _job(protocol: str, method: str, seed: int, output_dir: Path, command: list[str]) -> dict[str, str]:
    return {
        "protocol": protocol,
        "job_id": _safe_slug(f"{protocol}_{method}_seed_{seed}_{output_dir.name}"),
        "method": method,
        "seed": str(int(seed)),
        "output_dir": str(output_dir),
        "command": _join(command),
    }


def _loss_params_for(config: ManualProtocolConfig, algo_prefix: str, variant: str) -> list[str]:
    merged = dict(REPORT_LOSS_WEIGHT_PARAMS)
    merged.update(LOSS_VARIANT_PRESETS.get(variant, {}))
    merged.update(dict(config.loss_weight_params or {}))
    if algo_prefix in {"scdeepcluster_scraw_weighted", "desc_scraw_weighted"} and "weight_n_clusters" not in merged:
        merged["weight_n_clusters"] = int(config.n_labels)
    params = [f"{algo_prefix}:{key}={value}" for key, value in merged.items()]
    params.extend(config.params)
    return params


def build_loss_transfer_jobs(config: ManualProtocolConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    methods = _tokens(config.loss_methods) or list(LOSS_METHODS)
    variants = _tokens(config.loss_variants) or ["baseline", "weighted"]
    for method_key in methods:
        if method_key not in LOSS_METHODS:
            raise ValueError(f"Unknown loss-transfer method {method_key!r}. Known: {', '.join(LOSS_METHODS)}")
        baseline_method, weighted_method, algo_prefix = LOSS_METHODS[method_key]
        for variant in variants:
            if variant == "baseline":
                method = baseline_method
                params = list(config.params)
            else:
                if variant not in LOSS_VARIANT_PRESETS:
                    raise ValueError(
                        f"Unknown loss-transfer variant {variant!r}. "
                        "Use baseline, weighted, density_only, kmeans, or triplet."
                    )
                method = weighted_method
                params = _loss_params_for(config, algo_prefix, variant)
            for seed in config.seeds:
                output_dir = (
                    Path(config.output_root).expanduser().resolve()
                    / "loss_transfer"
                    / _safe_slug(method)
                    / _dataset_key(config)
                    / _safe_slug(variant)
                    / f"seed_{seed}"
                )
                cmd = _base_run_method_command(config, method=method, output_dir=output_dir, seed=seed, params=params)
                rows.append(_job("loss_transfer", f"{method}:{variant}", seed, output_dir, cmd))
    return rows


def _expand_harmony_methods(methods: Iterable[str]) -> list[str]:
    tokens = _tokens(methods)
    if not tokens:
        return list(DEFAULT_HARMONY_METHODS)
    if any(token.lower() == "all" for token in tokens):
        specs = {spec.name: spec for spec in load_method_specs().values() if spec.name == spec.name}
        harmony = [
            spec.name
            for spec in specs.values()
            if spec.name == "Harmony" or "+Harmony" in spec.name or "+Harmony" in spec.display_name
        ]
        return sorted(set(harmony), key=str.lower)
    return tokens


def build_harmony_jobs(config: ManualProtocolConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    harmony_params = list(config.params)
    for method in _expand_harmony_methods(config.harmony_methods):
        if get_method_spec(method) is None:
            raise ValueError(f"Unknown Harmony method {method!r}. Check methods/*.yaml.")
        for seed in config.seeds:
            output_dir = (
                Path(config.output_root).expanduser().resolve()
                / "harmony"
                / _safe_slug(method)
                / _dataset_key(config)
                / f"seed_{seed}"
            )
            cmd = _base_run_method_command(config, method=method, output_dir=output_dir, seed=seed, params=harmony_params)
            rows.append(_job("harmony", method, seed, output_dir, cmd))
    return rows


def build_inductive_jobs(config: ManualProtocolConfig) -> list[dict[str, str]]:
    algorithms = _tokens(config.inductive_algorithms) or list(DEFAULT_INDUCTIVE_ALGORITHMS)
    if not config.train_batches or not config.test_batches:
        raise ValueError("Inductive protocol requires train_batches and test_batches.")

    rows: list[dict[str, str]] = []
    for seed in config.seeds:
        output_dir = (
            Path(config.output_root).expanduser().resolve()
            / "inductive"
            / _dataset_key(config)
            / f"seed_{seed}"
        )
        cmd = [
            str(config.python_bin),
            str(REPO_ROOT / "scripts" / "reproduction" / "run_shared_train_inductive_algorithms.py"),
            "--data-path",
            str(Path(config.data).expanduser().resolve()),
            "--dataset-name",
            _dataset_key(config),
            "--output-root",
            str(output_dir),
            "--split-key",
            config.split_key,
            "--label-key",
            config.label_key,
            "--train-batches",
            *config.train_batches,
            "--test-batches",
            *config.test_batches,
            "--algorithms",
            *algorithms,
            "--preset",
            config.preset,
            "--device",
            config.device,
            "--seed",
            str(int(seed)),
            "--n-top-genes",
            str(int(config.n_top_genes)),
            "--baseline-runtime-profile",
            config.baseline_runtime_profile,
        ]
        if config.train_split_key:
            cmd.extend(["--train-split-key", config.train_split_key])
        if config.test_split_key:
            cmd.extend(["--test-split-key", config.test_split_key])
        if config.trial_config_path:
            cmd.extend(["--trial-config-path", config.trial_config_path])
        if config.skip_existing:
            cmd.append("--skip-existing")
        for param in config.params:
            cmd.extend(["--param", str(param)])
        rows.append(_job("inductive", "+".join(algorithms), seed, output_dir, cmd))
    return rows


def build_jobs(config: ManualProtocolConfig) -> list[dict[str, str]]:
    protocol = config.protocol.strip().lower().replace("-", "_")
    if protocol == "loss_transfer":
        return build_loss_transfer_jobs(config)
    if protocol == "harmony":
        return build_harmony_jobs(config)
    if protocol == "inductive":
        return build_inductive_jobs(config)
    raise ValueError("Unknown protocol. Use loss_transfer, harmony, or inductive.")


def write_plan(rows: list[Mapping[str, str]], csv_path: Path, shell_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PLAN_FIELDS})

    shell_path.parent.mkdir(parents=True, exist_ok=True)
    with shell_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        for row in rows:
            handle.write(f"# {row['protocol']} | {row['method']} | seed {row['seed']}\n")
            handle.write(f"mkdir -p {shlex.quote(str(Path(row['output_dir']).parent))}\n")
            handle.write(f"{row['command']}\n\n")
    shell_path.chmod(0o755)


def run_jobs(rows: list[Mapping[str, str]]) -> None:
    for row in rows:
        print(f"[{row['protocol']}] {row['method']} seed={row['seed']}", flush=True)
        subprocess.run(shlex.split(row["command"]), check=True, env=reproduction_env())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, choices=["loss_transfer", "loss-transfer", "harmony", "inductive"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-key", default="")
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, default=0)
    parser.add_argument("--seeds", default="42", help="Comma list or range, e.g. 42,43 or 42-46.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0,1.2,1.4")
    parser.add_argument("--selection-expected-n-classes", type=int, default=0)
    parser.add_argument("--param", action="append", default=[], help="Extra parameter override. Can be repeated.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--loss-methods", default="scMAE,scDeepCluster,DESC")
    parser.add_argument("--loss-variants", default="baseline,weighted")
    parser.add_argument(
        "--loss-weight-param",
        action="append",
        default=[],
        help="Override default loss-transfer weight parameter, key=value. Can be repeated.",
    )

    parser.add_argument("--harmony-methods", default="Harmony,scMAE+Harmony,scNAME+Harmony,scvi+Harmony")
    parser.add_argument("--harmony-max-iter", type=int, default=10)
    parser.add_argument("--harmony-nclust", type=int, default=50)
    parser.add_argument("--n-pcs", type=int, default=50)

    parser.add_argument("--inductive-algorithms", default="scraw,scname,sc_mae,scdeepcluster")
    parser.add_argument("--split-key", default="batch")
    parser.add_argument("--train-split-key", default="")
    parser.add_argument("--test-split-key", default="")
    parser.add_argument("--train-batches", default="")
    parser.add_argument("--test-batches", default="")
    parser.add_argument("--preset", default="stable_generalist", choices=["default", "0017", "stable_generalist"])
    parser.add_argument("--trial-config-path", default="")
    parser.add_argument(
        "--baseline-runtime-profile",
        choices=["scrbenchmark-default", "debug-fast"],
        default="scrbenchmark-default",
    )
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--plan-csv", default="")
    parser.add_argument("--script", default="")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ManualProtocolConfig:
    return ManualProtocolConfig(
        protocol=str(args.protocol).replace("-", "_"),
        data=args.data,
        output_root=args.output_root,
        dataset_key=args.dataset_key,
        label_key=args.label_key,
        batch_key=args.batch_key,
        n_labels=int(args.n_labels),
        seeds=parse_seeds(args.seeds),
        device=args.device,
        python_bin=args.python_bin,
        scib_n_jobs=int(args.scib_n_jobs),
        n_top_genes=int(args.n_top_genes),
        min_genes_per_cell=int(args.min_genes_per_cell),
        max_genes_per_cell=int(args.max_genes_per_cell),
        min_cells_per_gene=int(args.min_cells_per_gene),
        target_sum=float(args.target_sum),
        scale_max_value=float(args.scale_max_value),
        hvg_flavor=args.hvg_flavor,
        resolutions=args.resolutions,
        selection_expected_n_classes=int(args.selection_expected_n_classes),
        params=tuple(args.param or []),
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
        loss_methods=tuple(_tokens(args.loss_methods)),
        loss_variants=tuple(_tokens(args.loss_variants)),
        loss_weight_params=_parse_key_values(args.loss_weight_param or []),
        harmony_methods=tuple(_tokens(args.harmony_methods)),
        harmony_max_iter=int(args.harmony_max_iter),
        harmony_nclust=int(args.harmony_nclust),
        n_pcs=int(args.n_pcs),
        inductive_algorithms=tuple(_tokens(args.inductive_algorithms)),
        split_key=args.split_key,
        train_split_key=args.train_split_key,
        test_split_key=args.test_split_key,
        train_batches=tuple(_tokens(args.train_batches)),
        test_batches=tuple(_tokens(args.test_batches)),
        preset=args.preset,
        trial_config_path=args.trial_config_path,
        baseline_runtime_profile=args.baseline_runtime_profile,
        skip_existing=bool(args.skip_existing),
    )


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    rows = build_jobs(config)
    if args.plan_csv or args.script:
        output_root = Path(config.output_root).expanduser().resolve()
        csv_path = Path(args.plan_csv).expanduser().resolve() if args.plan_csv else output_root / "manual_protocol_jobs.csv"
        shell_path = Path(args.script).expanduser().resolve() if args.script else output_root / "run_manual_protocol_jobs.sh"
        write_plan(rows, csv_path, shell_path)
        print(f"planned_jobs = {csv_path}")
        print(f"ready_launcher = {shell_path}")
    else:
        for row in rows:
            print(row["command"])
    print(f"jobs = {len(rows)}")
    if args.execute:
        run_jobs(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
