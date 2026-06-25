#!/usr/bin/env python3
"""Build a reproducible command plan for the stable_generalist result table."""

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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scrbenchmark.methods import get_method_spec  # noqa: E402

REFERENCE_ROOT = REPO_ROOT / "reproducibility" / "stable_generalist"
DEFAULT_RESULTS_TABLE = REFERENCE_ROOT / "stable_generalist_all_results_table.csv"
DEFAULT_DATASET_TABLE = REFERENCE_ROOT / "stable_generalist_dataset_table.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "stable_generalist_reproduction"


PLAN_FIELDS = [
    "job_id",
    "result_row_id",
    "dataset_key",
    "dataset",
    "method",
    "family",
    "status",
    "data_file",
    "output_dir",
    "expected_file",
    "command",
    "notes",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    dataset: str
    data_file: Path
    label_key: str
    batch_key: str
    n_labels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-table", default=str(DEFAULT_RESULTS_TABLE))
    parser.add_argument("--dataset-table", default=str(DEFAULT_DATASET_TABLE))
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "data" / "stable_generalist"),
        help="Portable location expected to contain the 13 .h5ad files.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset_key subset.")
    parser.add_argument("--methods", default="", help="Comma-separated method subset using display names.")
    parser.add_argument("--strict-data", action="store_true", help="Mark jobs blocked when the .h5ad file is missing.")
    return parser.parse_args()


def _tokens(raw: str) -> set[str]:
    return {token.strip() for token in str(raw).split(",") if token.strip()}


def _safe_name(value: Any) -> str:
    text = str(value).strip()
    out = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            out.append(char)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"


def _resolve_data_file(raw_path: Any, data_root: Path) -> Path:
    raw = Path(str(raw_path)).expanduser()
    aliases = {
        "pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad",
    }
    filename = aliases.get(raw.name, raw.name)
    return (data_root / filename).resolve()


def read_dataset_specs(dataset_table: Path, data_root: Path) -> dict[str, DatasetSpec]:
    frame = pd.read_csv(dataset_table)
    specs: dict[str, DatasetSpec] = {}
    for _, row in frame.iterrows():
        dataset_key = str(row["dataset_key"]).strip()
        specs[dataset_key] = DatasetSpec(
            dataset_key=dataset_key,
            dataset=str(row["dataset"]).strip(),
            data_file=_resolve_data_file(row["data_file"], data_root),
            label_key=str(row.get("label_key") or "Group").strip(),
            batch_key=str(row.get("dann_batch_column") or "batch").strip(),
            n_labels=int(row.get("n_labels") or 0),
        )
    return specs


def build_command(
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    result_row_id: str,
    method: str,
    output_dir: Path,
    selection_expected_n_classes: int | None = None,
) -> tuple[str, str, str, str]:
    """Return family, status, command, notes."""
    method_spec = get_method_spec(method)
    if method_spec is None:
        return "unknown", "blocked_unmapped", "", f"No method spec is registered for method {method!r}."

    cmd = [
        str(args.python_bin),
        str(REPO_ROOT / "scripts" / "reproduction" / "run_method.py"),
        "--method",
        method,
        "--data",
        str(spec.data_file),
        "--output",
        str(output_dir),
        "--dataset-key",
        spec.dataset_key,
        "--label-key",
        spec.label_key,
        "--batch-key",
        spec.batch_key,
        "--n-labels",
        str(spec.n_labels),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
        "--scib-n-jobs",
        str(int(args.scib_n_jobs)),
        "--verbose",
    ]
    runner_command = method_spec.runner.get("command") if method_spec.runner else None
    runner_uses_selection = isinstance(runner_command, list) and "--selection-expected-n-classes" in runner_command
    if (method_spec.runner_kind == "batch_baseline" or runner_uses_selection) and selection_expected_n_classes:
        cmd.extend(["--selection-expected-n-classes", str(int(selection_expected_n_classes))])
    notes = f"{method_spec.core_status}. {method_spec.notes}".strip()
    return method_spec.runner_kind, "ready", _join(cmd), notes


def _join(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _expected_file(method: str, output_dir: Path) -> Path:
    method_spec = get_method_spec(method)
    if method_spec is None:
        return output_dir
    return output_dir / method_spec.expected_file


def build_plan(args: argparse.Namespace) -> list[dict[str, str]]:
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    specs = read_dataset_specs(Path(args.dataset_table), data_root)
    specs_by_dataset_name = {spec.dataset: spec for spec in specs.values()}
    results = pd.read_csv(args.results_table)

    selected_datasets = _tokens(args.datasets)
    selected_methods = _tokens(args.methods)
    rows: list[dict[str, str]] = []

    for _, result in results.iterrows():
        dataset_key = "" if pd.isna(result.get("dataset_key")) else str(result["dataset_key"]).strip()
        dataset_name = "" if pd.isna(result.get("dataset")) else str(result["dataset"]).strip()
        method = str(result["method"]).strip()
        if selected_methods and method not in selected_methods:
            continue

        spec = specs.get(dataset_key)
        if spec is None and dataset_name:
            spec = specs_by_dataset_name.get(dataset_name)
        if spec is None:
            continue
        dataset_key = spec.dataset_key
        if selected_datasets and dataset_key not in selected_datasets:
            continue

        result_row_id = str(result.get("result_row_id") or f"{dataset_key}_{method}")
        family_slug = _safe_name(method)
        output_dir = output_root / family_slug / dataset_key / result_row_id
        selected_cluster_count = None
        try:
            raw_cluster_count = result.get("n_clusters_found")
            if not pd.isna(raw_cluster_count):
                selected_cluster_count = int(round(float(raw_cluster_count)))
        except Exception:
            selected_cluster_count = None
        family, status, command, notes = build_command(
            args=args,
            spec=spec,
            result_row_id=result_row_id,
            method=method,
            output_dir=output_dir,
            selection_expected_n_classes=selected_cluster_count,
        )
        if args.strict_data and not spec.data_file.exists():
            status = "blocked_missing_data"
            notes = f"{notes} Missing data file: {spec.data_file}".strip()

        expected = _expected_file(method, output_dir)
        rows.append(
            {
                "job_id": _safe_name(f"{result_row_id}_{method}"),
                "result_row_id": result_row_id,
                "dataset_key": dataset_key,
                "dataset": spec.dataset,
                "method": method,
                "family": family,
                "status": status,
                "data_file": str(spec.data_file),
                "output_dir": str(output_dir),
                "expected_file": str(expected),
                "command": command,
                "notes": notes,
            }
        )

    rows.sort(key=lambda row: (row["status"], row["family"], row["dataset_key"], row["method"], row["result_row_id"]))
    return rows


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
        handle.write('export PYTHONPATH="${REPO_ROOT}/vendor/scraw_dedicated/src:${REPO_ROOT}/src:${REPO_ROOT}/src/scrbenchmark${PYTHONPATH:+:${PYTHONPATH}}"\n')
        handle.write('export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-scrbenchmark-repro}"\n')
        handle.write('export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba-scrbenchmark-repro}"\n')
        handle.write('mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"\n\n')
        for row in ready:
            handle.write(f"# {row['result_row_id']} | {row['dataset_key']} | {row['method']}\n")
            handle.write(f"mkdir -p {shlex.quote(str(Path(str(row['output_dir'])).parent))}\n")
            handle.write(f"{row['command']}\n\n")


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('status')}:{row.get('family')}"
        summary[key] = summary.get(key, 0) + 1
    return dict(sorted(summary.items()))


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    plan_csv = output_root / "planned_jobs.csv"
    shell_path = output_root / "run_ready_jobs.sh"

    rows = build_plan(args)
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
