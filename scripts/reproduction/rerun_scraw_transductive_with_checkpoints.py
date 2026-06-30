#!/usr/bin/env python3
"""Rerun transductive scRAW stable-generalist jobs and save final checkpoints.

The original stable-generalist search used the ``search_minimal`` profile, which
kept metrics but intentionally did not persist model checkpoints. This launcher
reuses the same scRAW preset/configuration while switching to the standard
artifact profile plus ``--metrics-only`` so training/evaluation stay comparable
and per-cell outputs are exported. A small in-process patch saves the final
autoencoder state after ``fit`` returns.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
SCRAW_ROOT = Path(os.environ.get("SCRAW_EXPERIMENTAL_ROOT", WORKSPACE_ROOT / "scRAW_EXPERIMENTAL"))
DEFAULT_SOURCE_RESULTS_TABLE = (
    SCRAW_ROOT
    / "results"
    / "presentation_stable_generalist_nonbaron_20260324"
    / "00_source_tables"
    / "stable_generalist_all_results_table.csv"
)
DEFAULT_DATASET_TABLE = (
    SCRAW_ROOT
    / "results"
    / "presentation_stable_generalist_nonbaron_20260324"
    / "00_source_tables"
    / "stable_generalist_dataset_table.csv"
)
DEFAULT_TRIAL_ROOT = (
    SCRAW_ROOT
    / "results"
    / "optuna_stable_generalist_search_20260415_161134"
    / "phase1"
    / "stable_generalist"
    / "stage1"
    / "trials"
    / "stable_generalist_trial_0017"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "scraw-transductive-stable-generalist"
DEFAULT_PYTHON = WORKSPACE_ROOT / "scrbenchmark_venv" / "bin" / "python"

METRIC_COLUMNS = [
    "NMI",
    "ARI",
    "ACC",
    "F1_Macro",
    "BalancedACC",
    "RareACC",
    "UltraRareACC",
    "n_clusters_found",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _as_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("_")


def _resolve_data_file(raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.exists():
        return path.resolve()

    candidates = [
        SCRAW_ROOT / "data" / path.name,
        SCRAW_ROOT / "data" / "scclubench_table_rows" / path.name,
        SCRAW_ROOT / "data" / "scclubench_extra" / path.name,
        REPO_ROOT / "data" / "stable_generalist" / path.name,
        REPO_ROOT / "data" / path.name,
    ]
    if path.name == "pancreas_raw_counts.h5ad":
        candidates.append(SCRAW_ROOT / "data" / "pancreas_raw_counts_no_smarter.h5ad")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset not found: {raw_path}")


def _read_dataset_table(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row["dataset_key"]).strip()
            out[key] = {
                "dataset_key": key,
                "dataset": str(row.get("dataset", key) or key),
                "data_file": _resolve_data_file(str(row["data_file"]).strip()),
                "batch_key": str(row.get("dann_batch_column", "batch") or "batch").strip(),
                "n_labels": int(row["n_labels"]),
            }
    return out


def _read_source_scraw_rows(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method", "")).strip().lower() != "scraw":
                continue
            if str(row.get("trial_id", "")).strip() != "stable_generalist":
                continue
            key = str(row["dataset_key"]).strip()
            rows[key] = dict(row)
    return rows


def _iter_first(items: Iterable[Path]) -> Path:
    for item in items:
        return item
    raise FileNotFoundError("No matching config_used.json found for the requested trial.")


def _load_trial_spec(trial_root: Path) -> dict[str, Any]:
    trial_root = trial_root.resolve()
    trial_payload = _read_json(trial_root / "trial_config.json")
    search_context_path = trial_root.parents[4] / "config" / "search_context.json"
    search_context: dict[str, Any] = {}
    if search_context_path.exists():
        search_context = _read_json(search_context_path)

    runs_root = trial_root.parents[1] / "runs" / trial_root.name
    sample_config_path = _iter_first(sorted(runs_root.glob("*/seed_*/config/config_used.json")))
    config_payload = _read_json(sample_config_path)

    preset_name = (
        str(trial_payload.get("preset_name") or "").strip()
        or str(config_payload.get("context", {}).get("preset") or "").strip()
    )
    if not preset_name:
        raise ValueError(f"Unable to resolve preset name from {trial_root}")

    seed = int(config_payload.get("execution", {}).get("random_seed", trial_payload.get("seed", 42)))
    algorithm_params = dict(config_payload.get("algorithm_params", {}).get("scraw", {}))
    preprocessing_params = dict(config_payload.get("preprocessing", {}))
    if not algorithm_params:
        raise ValueError(f"Missing algorithm_params.scraw in {sample_config_path}")

    return {
        "trial_root": str(trial_root),
        "candidate_id": str(trial_payload.get("candidate_id") or trial_root.name),
        "branch": str(trial_payload.get("branch") or ""),
        "preset": preset_name,
        "seed": seed,
        "python_bin": str(search_context.get("python_bin", DEFAULT_PYTHON)),
        "param_overrides": algorithm_params,
        "preprocess_overrides": preprocessing_params,
        "sample_config_path": str(sample_config_path),
    }


def _build_cli_argv(
    *,
    dataset: Mapping[str, Any],
    output_dir: Path,
    trial_spec: Mapping[str, Any],
    scib_n_jobs: int,
) -> list[str]:
    argv = [
        "--preset",
        str(trial_spec["preset"]),
        "--data",
        str(dataset["data_file"]),
        "--output",
        str(output_dir),
        "--seed",
        str(int(trial_spec["seed"])),
        "--device",
        "cuda",
        "--output-profile",
        "standard",
        "--metrics-only",
        "--capture-snapshots",
        "off",
        "--compute-scib-metrics",
        "on",
        "--scib-n-jobs",
        str(int(scib_n_jobs)),
        "--save-processed-data",
        "off",
        "--auto-hparams",
        "off",
        "--dann",
        "auto",
        "--batch-key",
        str(dataset["batch_key"]),
        "--leiden-target-clusters",
        str(int(dataset["n_labels"])),
        "--verbose",
    ]

    exclude = {"batch_correction_key", "device", "random_state", "resume_checkpoint_path", "save_checkpoint_path", "seed"}
    for key, value in sorted(dict(trial_spec["preprocess_overrides"]).items()):
        if value is not None:
            argv.extend(["--preprocess", f"{key}={_as_cli_value(value)}"])
    for key, value in sorted(dict(trial_spec["param_overrides"]).items()):
        if key not in exclude and value is not None:
            argv.extend(["--param", f"{key}={_as_cli_value(value)}"])
    argv.extend(["--param", f"batch_correction_key={_as_cli_value(dataset['batch_key'])}"])
    return argv


def _run_worker(args: argparse.Namespace) -> int:
    os.environ["PYTHONPATH"] = str(SCRAW_ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

    from scraw_dedicated import cli as scraw_cli
    from scraw_dedicated.algorithms.scraw_algorithm import ScRAWAlgorithm
    import torch

    checkpoint_path = Path(args.worker_checkpoint).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    original_fit = ScRAWAlgorithm.fit

    def fit_and_save(self: ScRAWAlgorithm, *fit_args: Any, **fit_kwargs: Any) -> ScRAWAlgorithm:
        result = original_fit(self, *fit_args, **fit_kwargs)
        if self.model is None:
            raise RuntimeError("scRAW fit completed but model is not initialized.")
        model_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        final_cell_weights = None
        get_final_cell_weights = getattr(self, "get_final_cell_weights", None)
        if callable(get_final_cell_weights):
            final_cell_weights = get_final_cell_weights()
        final_weight_components: dict[str, Any] = {}
        get_final_weight_components = getattr(self, "get_final_weight_components", None)
        if callable(get_final_weight_components):
            final_weight_components = dict(get_final_weight_components() or {})
        final_embeddings = self.get_embeddings()
        final_labels = self.predict()
        payload = {
            "format": "scraw_dedicated_final_autoencoder_v1",
            "dataset_key": str(args.worker_dataset_key),
            "created_at": datetime.now().isoformat(),
            "source_results_table": str(Path(args.source_results_table).expanduser().resolve()),
            "trial_root": str(Path(args.trial_root).expanduser().resolve()),
            "data_path": str(Path(cli_args.data).expanduser().resolve()),
            "run_output": str(Path(cli_args.output).expanduser().resolve()),
            "cli_argv": json.loads(args.worker_cli_argv_json),
            "model_state": model_state,
            "params": dict(self.params),
            "effective_params": dict(self.get_effective_params()),
            "final_cell_weights": None if final_cell_weights is None else np.asarray(final_cell_weights, dtype=np.float32),
            "final_cluster_component_weights": None
            if final_weight_components.get("cluster_component_weights") is None
            else np.asarray(final_weight_components["cluster_component_weights"], dtype=np.float32),
            "final_density_component_weights": None
            if final_weight_components.get("density_component_weights") is None
            else np.asarray(final_weight_components["density_component_weights"], dtype=np.float32),
            "final_embeddings": None if final_embeddings is None else np.asarray(final_embeddings, dtype=np.float32),
            "final_labels": None if final_labels is None else np.asarray(final_labels),
            "final_clustering_info": dict(self.get_final_clustering_info()),
            "pseudo_clustering_info": dict(self.get_pseudo_clustering_info()),
        }
        torch.save(payload, str(checkpoint_path))
        return result

    ScRAWAlgorithm.fit = fit_and_save
    try:
        parser = scraw_cli.build_arg_parser()
        cli_args = parser.parse_args(json.loads(args.worker_cli_argv_json))
        return int(scraw_cli.run_once(cli_args))
    finally:
        ScRAWAlgorithm.fit = original_fit


def _plot_combined_umap(per_cell_csv: Path, output_png: Path) -> bool:
    if not per_cell_csv.exists():
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(per_cell_csv)
    required = {"umap_1", "umap_2"}
    if not required.issubset(df.columns):
        return False

    panels: list[tuple[str, str]] = []
    if "batch" in df.columns:
        panels.append(("batch", "Batch"))
    if "predicted_label" in df.columns:
        panels.append(("predicted_label", "Label predit"))
    if "true_label" in df.columns:
        panels.append(("true_label", "Ground truth"))
    if "scraw_reconstruction_weight" in df.columns:
        panels.append(("scraw_reconstruction_weight", "scRAW cell weight"))
    if not panels:
        return False

    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 4.2), squeeze=False)
    x = df["umap_1"].to_numpy()
    y = df["umap_2"].to_numpy()
    for ax, (column, title) in zip(axes.ravel(), panels):
        values = df[column]
        if column == "scraw_reconstruction_weight":
            scatter = ax.scatter(x, y, c=values.astype(float), s=4, cmap="viridis", linewidths=0)
            fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        else:
            codes = values.astype(str).astype("category").cat.codes
            ax.scatter(x, y, c=codes, s=4, cmap="tab20", linewidths=0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float_or_none(value: Any) -> float | None:
    text = str(value).strip()
    if text in {"", "NA", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _best_metric_match(source_row: Mapping[str, str], candidates: list[dict[str, str]]) -> tuple[dict[str, str] | None, float | None]:
    best_row: dict[str, str] | None = None
    best_diff: float | None = None
    for row in candidates:
        diffs = []
        for column in METRIC_COLUMNS:
            source_value = _float_or_none(source_row.get(column))
            candidate_value = _float_or_none(row.get(column))
            if source_value is None or candidate_value is None:
                continue
            diffs.append(abs(source_value - candidate_value))
        if not diffs:
            continue
        max_diff = max(diffs)
        if best_diff is None or max_diff < best_diff:
            best_diff = max_diff
            best_row = row
    return best_row, best_diff


def _validate_outputs(output_root: Path, source_rows: Mapping[str, Mapping[str, str]]) -> Path:
    validation_rows: list[dict[str, Any]] = []
    for dataset_key, source_row in source_rows.items():
        run_dir = output_root / "runs" / dataset_key / "seed_42"
        candidates = _read_metric_rows(run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv")
        candidates.extend(_read_metric_rows(run_dir / "results" / "analysis_results.csv"))
        best_row, max_abs_diff = _best_metric_match(source_row, candidates)
        out: dict[str, Any] = {
            "dataset_key": dataset_key,
            "status": "missing_metrics" if best_row is None else "ok",
            "best_method": "" if best_row is None else best_row.get("method", ""),
            "max_abs_diff": "" if max_abs_diff is None else f"{max_abs_diff:.12g}",
        }
        if max_abs_diff is not None and max_abs_diff > 1e-6:
            out["status"] = "different"
        for column in METRIC_COLUMNS:
            out[f"source_{column}"] = source_row.get(column, "")
            out[f"rerun_{column}"] = "" if best_row is None else best_row.get(column, "")
        validation_rows.append(out)

    path = output_root / "validation" / "metrics_comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(validation_rows[0].keys()) if validation_rows else ["dataset_key", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(validation_rows)
    return path


def _run_one_subprocess(
    *,
    dataset: Mapping[str, Any],
    gpu_id: str,
    args: argparse.Namespace,
    trial_spec: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_key = str(dataset["dataset_key"])
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / "runs" / dataset_key / "seed_42"
    checkpoint_path = output_root / "model_weights" / "checkpoints" / f"model_{_safe_name(dataset_key)}.pt"
    figure_path = output_root / "figures" / "umaps" / f"{_safe_name(dataset_key)}_batch_label_groundtruth_weights_umap.png"
    log_path = output_root / "logs" / f"{_safe_name(dataset_key)}.log"

    if run_dir.exists() and checkpoint_path.exists() and not args.overwrite:
        made_figure = _plot_combined_umap(run_dir / "results" / "per_cell" / "per_cell_scraw_run0.csv", figure_path)
        return {
            "dataset_key": dataset_key,
            "status": "skipped_existing",
            "gpu_id": gpu_id,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "umap": str(figure_path) if made_figure or figure_path.exists() else "",
            "log": str(log_path),
            "return_code": 0,
        }

    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    if args.overwrite and checkpoint_path.exists():
        checkpoint_path.unlink()

    cli_argv = _build_cli_argv(
        dataset=dataset,
        output_dir=run_dir,
        trial_spec=trial_spec,
        scib_n_jobs=int(args.scib_n_jobs),
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-dataset-key",
        dataset_key,
        "--worker-checkpoint",
        str(checkpoint_path),
        "--worker-cli-argv-json",
        json.dumps(cli_argv),
        "--source-results-table",
        str(Path(args.source_results_table).expanduser().resolve()),
        "--trial-root",
        str(Path(args.trial_root).expanduser().resolve()),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(SCRAW_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    for cache_name in ("MPLCONFIGDIR", "NUMBA_CACHE_DIR", "XDG_CACHE_HOME"):
        cache_dir = output_root / ".cache" / dataset_key / cache_name.lower()
        cache_dir.mkdir(parents=True, exist_ok=True)
        env[cache_name] = str(cache_dir)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"dataset_key={dataset_key}\n")
        log_handle.write(f"gpu_id={gpu_id}\n")
        log_handle.write(f"checkpoint={checkpoint_path}\n")
        log_handle.write("command=" + " ".join(command) + "\n\n")
        log_handle.flush()
        proc = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=log_handle, stderr=subprocess.STDOUT)

    made_figure = False
    if proc.returncode == 0:
        made_figure = _plot_combined_umap(run_dir / "results" / "per_cell" / "per_cell_scraw_run0.csv", figure_path)

    return {
        "dataset_key": dataset_key,
        "status": "completed" if proc.returncode == 0 else "failed",
        "gpu_id": gpu_id,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else "",
        "umap": str(figure_path) if made_figure or figure_path.exists() else "",
        "log": str(log_path),
        "return_code": int(proc.returncode),
    }


def _write_run_status(output_root: Path, rows: list[Mapping[str, Any]]) -> Path:
    path = output_root / "run_status.csv"
    fieldnames = ["dataset_key", "status", "gpu_id", "run_dir", "checkpoint", "umap", "log", "return_code"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def _run_launcher(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).expanduser().resolve()
    source_results_table = Path(args.source_results_table).expanduser().resolve()
    dataset_table = Path(args.dataset_table).expanduser().resolve()
    trial_root = Path(args.trial_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for subdir in ["metadata", "logs", "runs", "model_weights/checkpoints", "figures/umaps", "validation"]:
        (output_root / subdir).mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_results_table, output_root / "metadata" / "stable_generalist_all_results_table.csv")
    shutil.copy2(dataset_table, output_root / "metadata" / "stable_generalist_dataset_table.csv")
    shutil.copy2(trial_root / "trial_config.json", output_root / "metadata" / "stable_generalist_trial_config.json")

    source_rows = _read_source_scraw_rows(source_results_table)
    datasets_by_key = _read_dataset_table(dataset_table)
    selected_keys = list(source_rows.keys())
    if args.dataset_keys:
        requested = {token.strip() for token in str(args.dataset_keys).split(",") if token.strip()}
        selected_keys = [key for key in selected_keys if key in requested]
    datasets = [datasets_by_key[key] for key in selected_keys]
    if not datasets:
        raise ValueError("No datasets selected.")

    gpu_ids = [token.strip() for token in str(args.gpus).split(",") if token.strip()]
    if not gpu_ids:
        raise ValueError("At least one GPU id is required.")

    trial_spec = _load_trial_spec(trial_root)
    _write_json(
        output_root / "run_metadata.json",
        {
            "started_at": datetime.now().isoformat(),
            "output_root": str(output_root),
            "source_results_table": str(source_results_table),
            "dataset_table": str(dataset_table),
            "trial_root": str(trial_root),
            "trial_spec": trial_spec,
            "gpus": gpu_ids,
            "scib_n_jobs": int(args.scib_n_jobs),
            "datasets": selected_keys,
            "note": "Rerun uses standard artifact profile plus metrics-only to save per-cell outputs and final checkpoints.",
        },
    )

    print("=== scRAW transductive stable-generalist rerun ===", flush=True)
    print(f"output_root = {output_root}", flush=True)
    print(f"datasets    = {len(datasets)}", flush=True)
    print(f"gpus        = {', '.join(gpu_ids)}", flush=True)
    print(f"preset      = {trial_spec['preset']}", flush=True)
    print(f"seed        = {trial_spec['seed']}", flush=True)

    work_queue: "queue.Queue[Mapping[str, Any]]" = queue.Queue()
    for dataset in datasets:
        work_queue.put(dataset)

    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def worker(gpu_id: str) -> None:
        while True:
            try:
                dataset = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                row = _run_one_subprocess(dataset=dataset, gpu_id=gpu_id, args=args, trial_spec=trial_spec)
            except Exception as exc:
                row = {
                    "dataset_key": str(dataset.get("dataset_key", "")),
                    "status": f"failed_exception:{type(exc).__name__}",
                    "gpu_id": gpu_id,
                    "run_dir": "",
                    "checkpoint": "",
                    "umap": "",
                    "log": "",
                    "return_code": "",
                }
                print(f"[error][gpu {gpu_id}] {dataset.get('dataset_key', '')}: {exc}", flush=True)
            with lock:
                results.append(row)
                _write_run_status(output_root, sorted(results, key=lambda r: str(r.get("dataset_key", ""))))
            print(f"[{row['status']}][gpu {gpu_id}] {row['dataset_key']}", flush=True)
            work_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=False) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    status_path = _write_run_status(output_root, sorted(results, key=lambda r: str(r.get("dataset_key", ""))))
    validation_path = _validate_outputs(output_root, source_rows)
    print(f"run_status = {status_path}", flush=True)
    print(f"validation = {validation_path}", flush=True)

    failed = [row for row in results if str(row.get("status")) not in {"completed", "skipped_existing"}]
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-table", default=str(DEFAULT_SOURCE_RESULTS_TABLE))
    parser.add_argument("--dataset-table", default=str(DEFAULT_DATASET_TABLE))
    parser.add_argument("--trial-root", default=str(DEFAULT_TRIAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--scib-n-jobs", type=int, default=6)
    parser.add_argument("--dataset-keys", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-checkpoint", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cli-argv-json", default="", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return _run_worker(args)
    return _run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
