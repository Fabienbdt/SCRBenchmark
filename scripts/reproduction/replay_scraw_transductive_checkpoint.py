#!/usr/bin/env python3
"""Regenerate scRAW transductive outputs from a saved final checkpoint.

The checkpoint produced by ``rerun_scraw_transductive_with_checkpoints.py``
stores the final autoencoder state dict plus final dynamic cell weights. The
model state is used to recompute embeddings and final clustering; the saved
cell weights are reused for per-cell exports and UMAP color panels because they
are a dynamic training artifact, not a deterministic function of the encoder
weights alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRAW_ROOT = REPO_ROOT / "vendor" / "scraw_dedicated"
SCRAW_ROOT = Path(os.environ.get("SCRAW_EXPERIMENTAL_ROOT", DEFAULT_SCRAW_ROOT))
SCRAW_SRC = SCRAW_ROOT / "src"
if str(SCRAW_SRC) not in sys.path:
    sys.path.insert(0, str(SCRAW_SRC))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _as_numpy(value: Any, dtype: Any | None = None) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _detect_obs_key(obs: Any, explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in obs.columns:
            raise ValueError(f"Column {explicit!r} not found in adata.obs.")
        return explicit
    for key in candidates:
        if key in obs.columns:
            return key
    return None


def _plot_combined_umap(per_cell_csv: Path, output_png: Path) -> bool:
    if not per_cell_csv.exists():
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(per_cell_csv)
    if not {"umap_1", "umap_2"}.issubset(df.columns):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="model_<dataset>.pt produced by the transductive rerun.")
    parser.add_argument("--config", required=True, help="config_used.json from the matching run.")
    parser.add_argument("--data", default="", help="Input .h5ad. Defaults to checkpoint data_path/config data.file.")
    parser.add_argument("--output", required=True, help="Output directory for regenerated results.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-key", default="")
    parser.add_argument("--batch-key", default="")
    parser.add_argument("--leiden-target-clusters", type=int, default=0)
    parser.add_argument("--compute-scib-metrics", action="store_true")
    parser.add_argument("--scib-n-jobs", type=int, default=1)
    return parser.parse_args()


def _resolve_data_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.exists():
        return path.resolve()

    filename = path.name
    aliases = {"pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad"}
    candidates = [
        REPO_ROOT / "data" / "stable_generalist" / aliases.get(filename, filename),
        REPO_ROOT / "data" / aliases.get(filename, filename),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def main() -> int:
    args = parse_args()

    import anndata as ad
    import torch

    from scraw_dedicated.algorithms.scraw_algorithm import ScRAWAlgorithm
    from scraw_dedicated.cli import (
        _leiden_method_name,
        _leiden_optimized_for_target_clusters,
        _metric_row_from_bundle,
    )
    from scraw_dedicated.metrics import align_labels, compute_metrics
    from scraw_dedicated.preprocessing import preprocess_adata
    from scraw_dedicated.visualization import compute_projection_2d

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError(f"Invalid checkpoint format: {checkpoint_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_data_path = str(args.data or checkpoint.get("data_path") or config.get("data", {}).get("file", ""))
    data_path = _resolve_data_path(raw_data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    params = dict(checkpoint.get("params") or config.get("algorithm_params", {}).get("scraw", {}))
    if not params:
        raise ValueError("No scRAW params found in checkpoint/config.")
    params["device"] = str(args.device)

    preprocess_cfg = dict(config.get("preprocessing", {}))
    adata = ad.read_h5ad(data_path)
    adata_proc = preprocess_adata(adata, preprocess_cfg)

    label_key = _detect_obs_key(
        adata_proc.obs,
        args.label_key or config.get("context", {}).get("label_key"),
        ["label", "labels", "cell_type", "celltype", "Group"],
    )
    batch_key = _detect_obs_key(
        adata_proc.obs,
        args.batch_key or config.get("context", {}).get("batch_key_detected"),
        ["batch", "Batch", "sample", "donor", "study"],
    )

    true_labels = None if label_key is None else adata_proc.obs[label_key].astype(str).to_numpy()
    batch_values = None if batch_key is None else adata_proc.obs[batch_key].astype(str).to_numpy()

    algo = ScRAWAlgorithm(params=params)
    X = algo._as_numpy_matrix(adata_proc)
    model = algo._build_model(X.shape[1])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(torch.device(args.device if args.device != "auto" else algo.get_device()))
    algo._fitted = True
    embeddings = algo._encode_full(X)
    pred_labels = algo._hdbscan_clustering(embeddings)
    algo._embeddings = embeddings
    algo._labels = pred_labels

    metrics = compute_metrics(
        true_labels,
        pred_labels,
        embeddings=embeddings,
        adata=adata_proc if args.compute_scib_metrics else None,
        batch_key=batch_key,
        label_key=label_key,
        compute_scib=bool(args.compute_scib_metrics),
        scib_n_jobs=max(1, int(args.scib_n_jobs)),
    )

    leiden_target = int(args.leiden_target_clusters or 0)
    if leiden_target <= 0 and true_labels is not None:
        leiden_target = int(len(np.unique(true_labels)))

    final_epoch = int((checkpoint.get("effective_params") or {}).get("epochs", params.get("epochs", 120)) or 120) - 1
    final_info = dict(algo.get_final_clustering_info())
    final_rows = [
        _metric_row_from_bundle(
            epoch=final_epoch,
            method="hdbscan_final",
            metrics=metrics,
            n_clusters=int(len(np.unique(pred_labels))),
            extra={
                "selection_metric": final_info.get("selection_metric"),
                "selection_score": final_info.get("selection_score"),
                "rare_weighted_silhouette": final_info.get("rare_weighted_silhouette"),
                "target_clusters": final_info.get("target_clusters"),
                "target_source": final_info.get("target_source"),
                "cluster_count_diff": final_info.get("cluster_diff"),
                "noise_fraction": final_info.get("noise_fraction"),
                "hdbscan_min_cluster_size": final_info.get("min_cluster_size"),
                "hdbscan_min_samples": final_info.get("min_samples"),
                "hdbscan_cluster_selection_method": final_info.get("cluster_selection_method"),
                "hdbscan_scan_enabled": final_info.get("scan_enabled"),
            },
        )
    ]

    leiden_labels = None
    if leiden_target > 0:
        leiden_labels, leiden_info = _leiden_optimized_for_target_clusters(
            embeddings=embeddings,
            seed=int(args.seed),
            target_clusters=leiden_target,
            labels_true=true_labels,
        )
        leiden_metrics = compute_metrics(true_labels, leiden_labels, embeddings=embeddings)
        final_rows.append(
            _metric_row_from_bundle(
                epoch=final_epoch,
                method=_leiden_method_name(leiden_target, final=True),
                metrics=leiden_metrics,
                n_clusters=int(leiden_info.get("n_clusters", len(np.unique(leiden_labels)))),
                extra={
                    "resolution": leiden_info.get("resolution"),
                    "selection_metric": leiden_info.get("selection_metric"),
                    "selection_score": leiden_info.get("selection_score"),
                    "target_clusters": leiden_target,
                },
            )
        )

    embeddings_dir = output / "results" / "embeddings"
    clustering_dir = output / "results" / "clustering_final"
    per_cell_dir = output / "results" / "per_cell"
    weights_dir = output / "results" / "weights"
    figures_dir = output / "figures" / "umaps"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_dir / "embeddings_scraw_run0.npy", np.asarray(embeddings, dtype=np.float32))
    _write_rows_csv(clustering_dir / "final_clustering_comparison.csv", final_rows)

    projection_2d, _ = compute_projection_2d(np.asarray(embeddings), random_state=int(args.seed))
    cell_weights = _as_numpy(checkpoint.get("final_cell_weights"), dtype=np.float32)
    cluster_weights = _as_numpy(checkpoint.get("final_cluster_component_weights"), dtype=np.float32)
    density_weights = _as_numpy(checkpoint.get("final_density_component_weights"), dtype=np.float32)
    saved_labels = _as_numpy(checkpoint.get("final_labels"))

    aligned_pred = None
    if true_labels is not None and len(true_labels) == len(pred_labels):
        aligned_pred = np.asarray(align_labels(true_labels, pred_labels), dtype=object)

    obs_names = np.asarray(adata_proc.obs_names.astype(str), dtype=object)
    per_cell_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for idx in range(len(pred_labels)):
        row: dict[str, Any] = {
            "cell_index": int(idx),
            "cell_id": str(obs_names[idx]),
            "umap_1": float(projection_2d[idx, 0]),
            "umap_2": float(projection_2d[idx, 1]),
            "predicted_label": str(pred_labels[idx]),
        }
        if saved_labels is not None and len(saved_labels) == len(pred_labels):
            row["saved_checkpoint_label"] = str(saved_labels[idx])
        if leiden_labels is not None:
            row["leiden_predicted_label"] = str(leiden_labels[idx])
        if true_labels is not None:
            row["true_label"] = str(true_labels[idx])
        if aligned_pred is not None:
            row["aligned_predicted_label"] = str(aligned_pred[idx])
        if batch_values is not None:
            row["batch"] = str(batch_values[idx])
        if cell_weights is not None and len(cell_weights) == len(pred_labels):
            row["scraw_reconstruction_weight"] = float(cell_weights[idx])
        if cluster_weights is not None and len(cluster_weights) == len(pred_labels):
            row["cluster_component_weight"] = float(cluster_weights[idx])
        if density_weights is not None and len(density_weights) == len(pred_labels):
            row["density_component_weight"] = float(density_weights[idx])
        per_cell_rows.append(row)

        weight_row = {
            "cell_index": int(idx),
            "cell_id": str(obs_names[idx]),
        }
        if cell_weights is not None and len(cell_weights) == len(pred_labels):
            weight_row["scraw_reconstruction_weight"] = float(cell_weights[idx])
        if cluster_weights is not None and len(cluster_weights) == len(pred_labels):
            weight_row["cluster_component_weight"] = float(cluster_weights[idx])
        if density_weights is not None and len(density_weights) == len(pred_labels):
            weight_row["density_component_weight"] = float(density_weights[idx])
        weight_rows.append(weight_row)

    per_cell_csv = per_cell_dir / "per_cell_scraw_run0.csv"
    _write_rows_csv(per_cell_csv, per_cell_rows)
    _write_rows_csv(weights_dir / "cell_weights_scraw_run0.csv", weight_rows)
    _plot_combined_umap(per_cell_csv, figures_dir / "batch_label_groundtruth_weights_umap.png")

    _write_json(
        output / "replay_metadata.json",
        {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "data": str(data_path),
            "device": str(args.device),
            "label_key": label_key,
            "batch_key": batch_key,
            "n_cells": int(len(pred_labels)),
            "checkpoint_format": checkpoint.get("format"),
            "cell_weights_reused_from_checkpoint": bool(cell_weights is not None),
        },
    )
    print(f"output = {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
