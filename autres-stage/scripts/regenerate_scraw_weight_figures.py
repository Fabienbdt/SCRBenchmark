#!/usr/bin/env python3
"""Regenerate simple UMAP figures from existing scRAW cell weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input .h5ad file.")
    parser.add_argument("--weights-csv", required=True, help="cell_weights_scraw_run0.csv or exported equivalent.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--label-key", default="")
    parser.add_argument("--batch-key", default="")
    parser.add_argument("--max-cells", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _detect_obs_key(obs: pd.DataFrame, explicit: str, candidates: list[str]) -> str | None:
    if explicit:
        if explicit not in obs.columns:
            raise ValueError(f"Column {explicit!r} not found in adata.obs.")
        return explicit
    for key in candidates:
        if key in obs.columns:
            return key
    return None


def _align_weights_to_adata(adata: object, weights: pd.DataFrame) -> tuple[object, np.ndarray]:
    if "cell_id" in weights.columns:
        by_id = weights.set_index(weights["cell_id"].astype(str))
        if not by_id.index.is_unique:
            raise ValueError("weights cell_id values must be unique.")
        obs_names = pd.Index(adata.obs_names.astype(str))
        if obs_names.isin(by_id.index).all():
            return adata, by_id.loc[obs_names, "scraw_reconstruction_weight"].to_numpy(dtype=float)
        weight_ids = pd.Index(by_id.index.astype(str))
        if weight_ids.isin(obs_names).all():
            return adata[weight_ids].copy(), by_id.loc[weight_ids, "scraw_reconstruction_weight"].to_numpy(dtype=float)
    if "cell_index" in weights.columns:
        ordered = weights.sort_values("cell_index")
        if len(ordered) == adata.n_obs:
            return adata, ordered["scraw_reconstruction_weight"].to_numpy(dtype=float)
    if len(weights) == adata.n_obs and "scraw_reconstruction_weight" in weights.columns:
        return adata, weights["scraw_reconstruction_weight"].to_numpy(dtype=float)
    raise ValueError("Could not align weights to AnnData observations.")


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scanpy as sc

    args = parse_args()
    adata = sc.read_h5ad(Path(args.data).expanduser().resolve())
    weights = pd.read_csv(Path(args.weights_csv).expanduser().resolve())
    adata, aligned_weights = _align_weights_to_adata(adata, weights)
    adata.obs["scraw_reconstruction_weight"] = aligned_weights

    if adata.n_obs > int(args.max_cells):
        sc.pp.subsample(adata, n_obs=int(args.max_cells), random_state=int(args.seed))

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=10000)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(2000, adata.n_vars), flavor="seurat")
    if "highly_variable" in adata.var:
        adata = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
    sc.tl.umap(adata, random_state=int(args.seed))

    label_key = _detect_obs_key(adata.obs, args.label_key, ["label", "Group", "cell_type", "celltype"])
    batch_key = _detect_obs_key(adata.obs, args.batch_key, ["batch", "Batch", "sample", "donor"])
    panels = ["scraw_reconstruction_weight"]
    if label_key:
        panels.append(label_key)
    if batch_key:
        panels.append(batch_key)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.5), squeeze=False)
    coords = adata.obsm["X_umap"]
    for ax, key in zip(axes.ravel(), panels):
        values = adata.obs[key]
        if key == "scraw_reconstruction_weight":
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=5, cmap="viridis", linewidths=0)
            fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
        else:
            codes = pd.Categorical(values.astype(str)).codes
            ax.scatter(coords[:, 0], coords[:, 1], c=codes, s=5, cmap="tab20", linewidths=0)
        ax.set_title(key)
        ax.set_xticks([])
        ax.set_yticks([])

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"figure = {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
