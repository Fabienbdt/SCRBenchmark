#!/usr/bin/env python3
"""Build command plans for the complementary report experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Mapping

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reproducibility" / "stable_generalist"
DEFAULT_DATASET_TABLE = REFERENCE_ROOT / "stable_generalist_dataset_table.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "report_reproduction"
DEFAULT_EXISTING_SCRAW_BARON_LABELS = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "presentation_stable_generalist_nonbaron_20260324/"
    "Exp\u00e9riences/scRAW_default_from_scRAW_seed60_stage_umaps_20260421/"
    "baron_human_pancreas/seed_60/results/labels/labels_scraw_run0.csv"
)
DEFAULT_EXISTING_REPORT_BARON_LABELS = Path(
    "/data2/fbidet/Rapport_Stage_M2_git/Images/"
    "analyse_biologique_scraw_baron_stable_generalist/tsne_coordinates.csv"
)


PLAN_FIELDS = [
    "campaign",
    "job_id",
    "dataset_key",
    "method",
    "status",
    "data_file",
    "output_dir",
    "expected_file",
    "command",
    "notes",
]

LOSS_TRANSFER_DATASETS = [
    "baron_human_pancreas",
    "bbag094_zeisel",
    "bbag094_spleen",
    "kang_pbmc_gse96583_singlets_raw_counts",
    "paul15_bone_marrow_raw_counts",
]
LOSS_TRANSFER_SEEDS = [42, 43, 44, 45, 46]
LOSS_TRANSFER_WEIGHT_PARAMS = {
    "warmup_epochs": 30,
    "dynamic_weight_update_interval": 20,
    "dynamic_weight_momentum": 0.6884621079434989,
    "pseudo_label_method": "leiden",
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


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    data_file: Path
    label_key: str
    batch_key: str
    n_labels: int
    n_batches: int


@dataclass(frozen=True)
class InductiveSplit:
    dataset_key: str
    split_name: str
    train_batches: tuple[str, ...]
    test_batches: tuple[str, ...]
    split_key: str
    train_split_key: str | None = None
    test_split_key: str | None = None


INDUCTIVE_SPLITS = [
    InductiveSplit(
        "baron_human_pancreas", "h234_to_h1", ("human2", "human3", "human4"), ("human1",), "batch"
    ),
    InductiveSplit(
        "baron_human_pancreas", "h134_to_h2", ("human1", "human3", "human4"), ("human2",), "batch"
    ),
    InductiveSplit(
        "baron_human_pancreas", "h124_to_h3", ("human1", "human2", "human4"), ("human3",), "batch"
    ),
    InductiveSplit(
        "baron_human_pancreas", "h123_to_h4", ("human1", "human2", "human3"), ("human4",), "batch"
    ),
    InductiveSplit("bbag094_spleen", "3F56_to_3M8", ("3-F-56",), ("3-M-8",), "batch"),
    InductiveSplit(
        "gse112013_human_testis_raw_counts",
        "donor12_to_donor3",
        (
            "Donor1_scRNA-seq_rep1",
            "Donor1_scRNA-seq_rep2",
            "Donor2_scRNA-seq_rep1",
            "Donor2_scRNA-seq_rep2",
        ),
        ("Donor3_scRNA-seq_rep1", "Donor3_scRNA-seq_rep2"),
        "batch",
    ),
    InductiveSplit(
        "kang_pbmc_gse96583_singlets_raw_counts",
        "train_samples_to_donors_1039_107",
        (
            "1015_ctrl",
            "1015_stim",
            "1488_ctrl",
            "1488_stim",
            "1256_ctrl",
            "1256_stim",
            "1016_ctrl",
            "1016_stim",
            "1244_ctrl",
            "1244_stim",
            "101_ctrl",
            "101_stim",
        ),
        ("1039", "107"),
        "donor",
        "sample",
        "donor",
    ),
    InductiveSplit(
        "macaque_retina_gse118480_bipolar_raw_counts",
        "m1m2_to_m3m4",
        ("M1", "M2"),
        ("M3", "M4"),
        "macaque_id",
    ),
    InductiveSplit(
        "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2",
        "smartseq2_celseq2_to_celseq_fluidigmc1",
        ("smartseq2", "celseq2"),
        ("celseq", "fluidigmc1"),
        "batch",
    ),
]

INDUCTIVE_METHODS = [
    ("scRAW", "scraw"),
    ("scNAME", "scname"),
    ("scMAE", "sc_mae"),
    ("scDeepCluster", "scdeepcluster"),
    ("scAIDE", "scaide"),
    ("PCA+Harmony", "pca_harmony"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-table", default=str(DEFAULT_DATASET_TABLE))
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data" / "stable_generalist"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--campaigns", default="inductive,loss_transfer,deg")
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument(
        "--no-reuse-existing-artifacts",
        action="store_true",
        help=(
            "Always plan source runs, even when local report artifacts already "
            "exist. By default, the DEG campaign reuses the existing Baron "
            "scRAW labels when available."
        ),
    )
    return parser.parse_args()


def _tokens(raw: str) -> set[str]:
    return {token.strip() for token in str(raw).split(",") if token.strip()}


def _safe(value: Any) -> str:
    text = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text).strip("_")


def _join(cmd: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _target_name(raw_path: Any) -> str:
    name = Path(str(raw_path)).name
    aliases = {"pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad"}
    return aliases.get(name, name)


def read_specs(dataset_table: Path, data_root: Path) -> dict[str, DatasetSpec]:
    frame = pd.read_csv(dataset_table)
    specs: dict[str, DatasetSpec] = {}
    for _, row in frame.iterrows():
        key = str(row["dataset_key"]).strip()
        specs[key] = DatasetSpec(
            dataset_key=key,
            data_file=(data_root / _target_name(row["data_file"])).resolve(),
            label_key=str(row.get("label_key") or "Group").strip(),
            batch_key=str(row.get("dann_batch_column") or "batch").strip(),
            n_labels=int(row.get("n_labels") or 0),
            n_batches=int(row.get("n_batches") or 0),
        )
    return specs


def _row(
    *,
    campaign: str,
    dataset_key: str,
    method: str,
    status: str,
    data_file: Path,
    output_dir: Path,
    expected_file: Path,
    command: list[Any] | None,
    notes: str,
) -> dict[str, str]:
    return {
        "campaign": campaign,
        "job_id": _safe(f"{campaign}_{dataset_key}_{method}_{Path(output_dir).name}"),
        "dataset_key": dataset_key,
        "method": method,
        "status": status,
        "data_file": str(data_file),
        "output_dir": str(output_dir),
        "expected_file": str(expected_file),
        "command": _join(command or []),
        "notes": notes,
    }


def add_inductive(rows: list[dict[str, str]], args: argparse.Namespace, specs: Mapping[str, DatasetSpec]) -> None:
    root = Path(args.output_root).expanduser().resolve() / "inductive"
    for split in INDUCTIVE_SPLITS:
        spec = specs[split.dataset_key]
        for display_name, algorithm in INDUCTIVE_METHODS:
            out = root / algorithm / split.dataset_key / split.split_name
            cmd: list[Any] = [
                args.python_bin,
                REPO_ROOT / "scripts" / "reproduction" / "run_shared_train_inductive_algorithms.py",
                "--data-path",
                spec.data_file,
                "--dataset-name",
                split.dataset_key,
                "--output-root",
                out,
                "--split-key",
                split.split_key,
            ]
            if split.train_split_key:
                cmd.extend(["--train-split-key", split.train_split_key])
            if split.test_split_key:
                cmd.extend(["--test-split-key", split.test_split_key])
            cmd.extend(
                [
                    "--label-key",
                    spec.label_key,
                    "--train-batches",
                    *split.train_batches,
                    "--test-batches",
                    *split.test_batches,
                    "--algorithms",
                    algorithm,
                    "--preset",
                    "stable_generalist",
                    "--device",
                    args.device,
                    "--seed",
                    42,
                    "--n-top-genes",
                    2000,
                    "--baseline-runtime-profile",
                    "scrbenchmark-default",
                    "--skip-existing",
                ]
            )
            rows.append(
                _row(
                    campaign="inductive",
                    dataset_key=split.dataset_key,
                    method=display_name,
                    status="ready",
                    data_file=spec.data_file,
                    output_dir=out,
                    expected_file=out / "summary.csv",
                    command=cmd,
                    notes=(
                        "Representative inductive split from the report; train groups="
                        f"{','.join(split.train_batches)} test groups={','.join(split.test_batches)}."
                    ),
                )
            )


def _run_method_cmd(args: argparse.Namespace, spec: DatasetSpec, method: str, output_dir: Path, seed: int) -> list[Any]:
    return [
        args.python_bin,
        REPO_ROOT / "scripts" / "reproduction" / "run_method.py",
        "--method",
        method,
        "--data",
        spec.data_file,
        "--output",
        output_dir,
        "--dataset-key",
        spec.dataset_key,
        "--label-key",
        spec.label_key,
        "--batch-key",
        spec.batch_key,
        "--n-labels",
        spec.n_labels,
        "--seed",
        seed,
        "--device",
        args.device,
        "--verbose",
    ]


def add_loss_transfer(rows: list[dict[str, str]], args: argparse.Namespace, specs: Mapping[str, DatasetSpec]) -> None:
    root = Path(args.output_root).expanduser().resolve() / "loss_transfer"
    methods = [
        "scMAE",
        "scMAE_scRAW_weighted",
        "scDeepCluster",
        "scDeepCluster_scRAW_weighted",
        "DESC",
        "DESC_scRAW_weighted",
    ]
    for dataset_key in LOSS_TRANSFER_DATASETS:
        spec = specs[dataset_key]
        for seed in LOSS_TRANSFER_SEEDS:
            for method in methods:
                out = root / method / dataset_key / f"seed_{seed}"
                cmd = _run_method_cmd(args, spec, method, out, seed)
                if method.endswith("_scRAW_weighted"):
                    if method.startswith("scMAE"):
                        algo = "sc_mae_scraw_weighted"
                    elif method.startswith("scDeepCluster"):
                        algo = "scdeepcluster_scraw_weighted"
                    else:
                        algo = "desc_scraw_weighted"
                    for key, value in LOSS_TRANSFER_WEIGHT_PARAMS.items():
                        cmd.extend(["--param", f"{algo}:{key}={value}"])
                    if algo in {"scdeepcluster_scraw_weighted", "desc_scraw_weighted"}:
                        cmd.extend(["--param", f"{algo}:weight_n_clusters={spec.n_labels}"])
                rows.append(
                    _row(
                        campaign="loss_transfer",
                        dataset_key=dataset_key,
                        method=method,
                        status="ready",
                        data_file=spec.data_file,
                        output_dir=out,
                        expected_file=out
                        / ("results/analysis_results.csv" if method.startswith("DESC") else "results/results.csv"),
                        command=cmd,
                        notes="Five-seed report plugging/loss-transfer plan.",
                    )
                )


def add_deg(rows: list[dict[str, str]], args: argparse.Namespace, specs: Mapping[str, DatasetSpec]) -> None:
    spec = specs["baron_human_pancreas"]
    root = Path(args.output_root).expanduser().resolve() / "deg_marker_overlap" / "baron_human_pancreas"
    scraw_out = root / "scraw_source"
    labels_csv = scraw_out / "results" / "labels" / "labels_scraw_run0.csv"
    true_label_col = ""
    pred_label_col = ""
    can_reuse_existing = not bool(getattr(args, "no_reuse_existing_artifacts", False))
    if can_reuse_existing and DEFAULT_EXISTING_REPORT_BARON_LABELS.exists():
        labels_csv = DEFAULT_EXISTING_REPORT_BARON_LABELS
        true_label_col = "true_label"
        pred_label_col = "predicted_label"
        rows.append(
            _row(
                campaign="deg",
                dataset_key=spec.dataset_key,
                method="scRAW",
                status="reused_existing",
                data_file=spec.data_file,
                output_dir=DEFAULT_EXISTING_REPORT_BARON_LABELS.parent,
                expected_file=labels_csv,
                command=None,
                notes=(
                    "Existing Baron labels from the report annotation table reused; "
                    "no source model rerun is required for marker-overlap planning."
                ),
            )
        )
    elif can_reuse_existing and DEFAULT_EXISTING_SCRAW_BARON_LABELS.exists():
        labels_csv = DEFAULT_EXISTING_SCRAW_BARON_LABELS
        rows.append(
            _row(
                campaign="deg",
                dataset_key=spec.dataset_key,
                method="scRAW",
                status="reused_existing_legacy_default",
                data_file=spec.data_file,
                output_dir=DEFAULT_EXISTING_SCRAW_BARON_LABELS.parent.parent.parent,
                expected_file=labels_csv,
                command=None,
                notes=(
                    "Legacy Baron scRAW default labels are available, but they are "
                    "not the exact stable_generalist labels used by the report."
                ),
            )
        )
    else:
        rows.append(
            _row(
                campaign="deg",
                dataset_key=spec.dataset_key,
                method="scRAW",
                status="ready",
                data_file=spec.data_file,
                output_dir=scraw_out,
                expected_file=labels_csv,
                command=_run_method_cmd(args, spec, "scRAW", scraw_out, 42),
                notes="Source scRAW run for Baron marker-overlap analysis.",
            )
        )
    deg_out = root / "marker_overlap"
    rows.append(
        _row(
            campaign="deg",
            dataset_key=spec.dataset_key,
            method="marker_overlap",
            status="ready",
            data_file=spec.data_file,
            output_dir=deg_out,
            expected_file=deg_out / "results" / "metrics_summary.csv",
            command=[
                args.python_bin,
                REPO_ROOT / "scripts" / "reproduction" / "run_marker_overlap.py",
                "--data",
                spec.data_file,
                "--labels-csv",
                labels_csv,
                "--output",
                deg_out,
                "--label-key",
                spec.label_key,
                *(
                    ["--true-label-col", true_label_col, "--pred-label-col", pred_label_col]
                    if true_label_col and pred_label_col
                    else []
                ),
                "--n-top-genes",
                100,
                "--method",
                "wilcoxon",
            ],
            notes="DEG top-100 marker-overlap annotation used by the report.",
        )
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PLAN_FIELDS})


def write_shell(path: Path, rows: list[Mapping[str, Any]]) -> None:
    ready = [row for row in rows if row.get("status") == "ready" and row.get("command")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        handle.write(f"REPO_ROOT={shlex.quote(str(REPO_ROOT))}\n")
        handle.write(
            'export PYTHONPATH="${REPO_ROOT}/vendor/scraw_dedicated/src:'
            '${REPO_ROOT}/vendor/scraw_inductive/src:${REPO_ROOT}/src:'
            '${REPO_ROOT}/src/scrbenchmark:${REPO_ROOT}/external/original_code/aide'
            '${PYTHONPATH:+:${PYTHONPATH}}"\n'
        )
        handle.write('export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-scrbenchmark-repro}"\n')
        handle.write('export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-scrbenchmark-repro}"\n')
        handle.write('mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"\n\n')
        for row in ready:
            handle.write(f"# {row['campaign']} | {row['dataset_key']} | {row['method']}\n")
            handle.write(f"mkdir -p {shlex.quote(str(Path(str(row['output_dir'])).parent))}\n")
            handle.write(f"{row['command']}\n\n")


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('campaign')}:{row.get('status')}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    specs = read_specs(Path(args.dataset_table), Path(args.data_root).expanduser().resolve())
    campaigns = _tokens(args.campaigns)
    rows: list[dict[str, str]] = []

    if "all" in campaigns or "inductive" in campaigns:
        add_inductive(rows, args, specs)
    if "all" in campaigns or "loss_transfer" in campaigns:
        add_loss_transfer(rows, args, specs)
    if "all" in campaigns or "deg" in campaigns:
        add_deg(rows, args, specs)

    if args.strict_data:
        for row in rows:
            if row["status"] == "ready" and row["data_file"] and not Path(row["data_file"]).exists():
                row["status"] = "blocked_missing_data"
                row["notes"] = f"{row['notes']} Missing data file: {row['data_file']}"

    rows.sort(key=lambda row: (row["campaign"], row["status"], row["dataset_key"], row["method"], row["job_id"]))
    plan_csv = output_root / "report_planned_jobs.csv"
    shell_path = output_root / "run_ready_report_jobs.sh"
    write_csv(plan_csv, rows)
    write_shell(shell_path, rows)
    shell_path.chmod(0o755)

    print(f"planned_jobs = {plan_csv}")
    print(f"ready_launcher = {shell_path}")
    print(f"jobs = {len(rows)}")
    for key, count in summarize(rows).items():
        print(f"{key} = {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
