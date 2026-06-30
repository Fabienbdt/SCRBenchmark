#!/usr/bin/env python3
"""Run DEG marker-overlap annotation from a saved labels file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
SCRBENCHMARK_SRC = REPO_ROOT / "src" / "scrbenchmark"
if str(SCRBENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(SCRBENCHMARK_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--true-label-col", default="")
    parser.add_argument("--pred-label-col", default="")
    parser.add_argument("--n-top-genes", type=int, default=100)
    parser.add_argument("--method", default="wilcoxon")
    return parser.parse_args()


def _pick_column(frame: pd.DataFrame, explicit: str, candidates: list[str], role: str) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"{role} column {explicit!r} not found in {list(frame.columns)}")
        return explicit
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"Could not infer {role} column from {list(frame.columns)}")


def preprocess_for_deg(adata: Any) -> Any:
    import scanpy as sc

    work = adata.copy()
    sc.pp.filter_genes(work, min_cells=3)
    sc.pp.normalize_total(work, target_sum=10000)
    sc.pp.log1p(work)
    return work


def _balanced_acc(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    recalls = []
    for label in sorted(np.unique(labels_true.astype(str))):
        mask = labels_true.astype(str) == label
        if np.any(mask):
            recalls.append(float(np.mean(labels_pred.astype(str)[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _rare_balanced_acc(labels_true: np.ndarray, labels_pred: np.ndarray, threshold: float = 0.05) -> float:
    labels = labels_true.astype(str)
    preds = labels_pred.astype(str)
    recalls = []
    total = len(labels)
    for label in sorted(np.unique(labels)):
        mask = labels == label
        if np.sum(mask) / max(total, 1) < threshold:
            recalls.append(float(np.mean(preds[mask] == label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def write_heatmap(overlap: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_width = max(7.0, min(16.0, 0.55 * max(1, overlap.shape[1])))
    fig_height = max(5.0, min(16.0, 0.42 * max(1, overlap.shape[0])))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(overlap.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(overlap.shape[1]))
    ax.set_yticks(np.arange(overlap.shape[0]))
    ax.set_xticklabels(overlap.columns.astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(overlap.index.astype(str), fontsize=8)
    ax.set_xlabel("Gold-standard cell type")
    ax.set_ylabel("Predicted cluster")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Top-gene overlap")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_deg_tables(payload: dict[str, Any], results_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for source_kind, key in [
        ("ground_truth", "ground_truth_degs"),
        ("predicted_cluster", "predicted_cluster_degs"),
    ]:
        groups = payload.get(key) or {}
        for group, genes in sorted(groups.items(), key=lambda item: str(item[0])):
            for rank, gene in enumerate(genes, start=1):
                rows.append(
                    {
                        "source_kind": source_kind,
                        "group": str(group),
                        "rank": rank,
                        "gene": str(gene),
                    }
                )
    pd.DataFrame(rows).to_csv(results_dir / "marker_overlap_genes_long.csv", index=False)

    mapping = payload.get("cluster_to_type") or {}
    pd.DataFrame(
        [
            {"predicted_cluster": str(cluster), "assigned_cell_type": str(cell_type)}
            for cluster, cell_type in sorted(mapping.items(), key=lambda item: str(item[0]))
        ]
    ).to_csv(results_dir / "cluster_to_type.csv", index=False)


def main() -> int:
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    from scrbenchmark.utils.metrics import marker_overlap_annotation

    args = parse_args()
    data_path = Path(args.data).expanduser().resolve()
    labels_path = Path(args.labels_csv).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    config_dir = output_dir / "config"
    for path in [results_dir, figures_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(data_path)
    labels = pd.read_csv(labels_path)
    true_col = _pick_column(
        labels,
        args.true_label_col,
        ["true_label", args.label_key, "label", "Group", "cell_type"],
        "true label",
    )
    pred_col = _pick_column(
        labels,
        args.pred_label_col,
        ["predicted_label", "pred_label", "cluster", "leiden", "labels"],
        "predicted label",
    )
    labels_true = labels[true_col].astype(str).to_numpy()
    labels_pred = labels[pred_col].astype(str).to_numpy()
    if len(labels_true) != adata.n_obs:
        raise ValueError(f"Label count ({len(labels_true)}) does not match adata.n_obs ({adata.n_obs}).")

    adata_deg = preprocess_for_deg(adata)
    result = marker_overlap_annotation(
        adata=adata_deg,
        labels_true=labels_true,
        labels_pred=labels_pred,
        n_top_genes=int(args.n_top_genes),
        method=str(args.method),
    )

    hungarian = np.asarray(result["hungarian_labels"], dtype=str)
    marker = np.asarray(result["marker_labels"], dtype=str)
    true = labels_true.astype(str)
    pred = labels_pred.astype(str)
    annotation = pd.DataFrame(
        {
            "true_label": true,
            "predicted_cluster": pred,
            "hungarian_annotation": hungarian,
            "marker_overlap_annotation": marker,
        }
    )
    annotation.to_csv(results_dir / "annotation_comparison.csv", index=False)
    result["overlap_matrix"].to_csv(results_dir / "marker_overlap_matrix.csv")
    write_heatmap(result["overlap_matrix"], figures_dir / "marker_overlap_heatmap.png")

    deg_payload = {
        "ground_truth_degs": result["gold_degs"],
        "predicted_cluster_degs": result["pred_degs"],
        "cluster_to_type": result["cluster_to_type"],
    }
    (results_dir / "degs_top100.json").write_text(json.dumps(deg_payload, indent=2), encoding="utf-8")
    write_deg_tables(deg_payload, results_dir)

    metrics = {
        "n_cells": int(len(true)),
        "n_true_types": int(len(np.unique(true))),
        "n_predicted_clusters": int(len(np.unique(pred))),
        "hungarian_ARI": float(adjusted_rand_score(true, hungarian)),
        "hungarian_BalancedACC": _balanced_acc(true, hungarian),
        "hungarian_BalancedRareACC": _rare_balanced_acc(true, hungarian),
        "marker_ARI": float(adjusted_rand_score(true, marker)),
        "marker_BalancedACC": _balanced_acc(true, marker),
        "marker_BalancedRareACC": _rare_balanced_acc(true, marker),
        "annotation_agreement": float(np.mean(hungarian == marker)),
        "n_top_genes": int(args.n_top_genes),
        "deg_method": str(args.method),
    }
    pd.DataFrame([metrics]).to_csv(results_dir / "metrics_summary.csv", index=False)
    (config_dir / "config_used.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"metrics = {results_dir / 'metrics_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
