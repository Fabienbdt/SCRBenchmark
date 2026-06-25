#!/usr/bin/env python3
"""Run DESC on one h5ad dataset and export scRAW-compatible evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRAW_DEDICATED_SRC = REPO_ROOT / "vendor" / "scraw_dedicated" / "src"
DESC_ROOT = REPO_ROOT / "external" / "original_code" / "desc"
for path in (SCRAW_DEDICATED_SRC, DESC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scraw_dedicated.metrics import align_labels, compute_metrics

DESC_IMPORT_ERROR = None
try:
    import desc
except Exception as exc:  # pragma: no cover - depends on optional legacy stack
    desc = None
    DESC_IMPORT_ERROR = exc


logger = logging.getLogger("run_desc_benchmark")


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _to_dense(X: Any) -> np.ndarray:
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _matrix_nnz_per_row(X: Any) -> np.ndarray:
    if issparse(X):
        return np.asarray(X.getnnz(axis=1)).ravel()
    return np.asarray((np.asarray(X) > 0).sum(axis=1)).ravel()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _parse_resolutions(raw: str) -> List[float]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("At least one DESC resolution is required.")
    return vals


def _round_str(x: float) -> str:
    return str(float(x))


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if pd.isna(obj):
        return None
    return obj


def preprocess_adata(
    adata: ad.AnnData,
    *,
    min_genes_per_cell: int,
    max_genes_per_cell: int,
    min_cells_per_gene: int,
    target_sum: float,
    n_top_genes: int,
    scale_max_value: float,
    hvg_flavor: str,
) -> tuple[ad.AnnData, Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "n_obs_input": int(adata.n_obs),
        "n_vars_input": int(adata.n_vars),
    }

    sc.pp.filter_cells(adata, min_genes=int(min_genes_per_cell))
    stats["n_obs_after_min_genes"] = int(adata.n_obs)

    if int(max_genes_per_cell) > 0:
        n_genes = _matrix_nnz_per_row(adata.X)
        mask = n_genes <= int(max_genes_per_cell)
        adata = adata[mask].copy()
    stats["n_obs_after_max_genes"] = int(adata.n_obs)

    sc.pp.filter_genes(adata, min_cells=int(min_cells_per_gene))
    stats["n_vars_after_min_cells"] = int(adata.n_vars)

    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(n_top_genes),
        flavor=str(hvg_flavor),
        subset=True,
    )
    stats["n_vars_after_hvg"] = int(adata.n_vars)

    sc.pp.scale(adata, max_value=float(scale_max_value))
    adata.X = np.asarray(_to_dense(adata.X), dtype=np.float32)

    stats["n_obs_final"] = int(adata.n_obs)
    stats["n_vars_final"] = int(adata.n_vars)
    return adata, stats


def _plot_umap(embedding: np.ndarray, labels: pd.Series, output_path: Path, seed: int) -> None:
    reducer = sc.tl.umap
    work = ad.AnnData(np.asarray(embedding, dtype=np.float32))
    sc.pp.neighbors(work, use_rep="X", n_neighbors=15)
    reducer(work, random_state=int(seed))
    coords = np.asarray(work.obsm["X_umap"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = labels.astype(str)
    uniq = sorted(labels.unique().tolist())
    cmap = plt.cm.tab20 if len(uniq) <= 20 else plt.cm.gist_ncar
    for idx, label in enumerate(uniq):
        mask = labels == label
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=6,
            alpha=0.85,
            linewidths=0,
            c=[cmap(idx / max(1, len(uniq) - 1))],
            label=label,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if 0 < len(uniq) <= 20:
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_desc_training_history(history_by_resolution: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the first DESC resolution that actually contains pretraining history."""
    if not isinstance(history_by_resolution, dict):
        return {}
    for _res, payload in history_by_resolution.items():
        if not isinstance(payload, dict):
            continue
        loss_history = payload.get("loss_history", {}) or {}
        weight_history = payload.get("weight_history", []) or []
        if loss_history.get("train_loss") or weight_history:
            selected = dict(payload)
            selected["source_resolution"] = _res
            return selected
    return {}


def _save_desc_training_history(
    *,
    history: Dict[str, Any],
    algorithm_name: str,
    results_dir: Path,
    figures_dir: Path,
) -> Dict[str, str]:
    exported: Dict[str, str] = {}
    if not history:
        return exported

    safe_algo = algorithm_name.replace(" ", "_").replace("/", "_")
    loss_history = history.get("loss_history", {}) or {}
    phases = []
    if loss_history.get("train_loss"):
        phases.append(loss_history)

    if phases:
        loss_dir = results_dir / "loss_history"
        loss_dir.mkdir(parents=True, exist_ok=True)
        loss_path = loss_dir / f"loss_{safe_algo}_run0.json"
        loss_path.write_text(
            json.dumps(
                {
                    "algorithm": algorithm_name,
                    "run_id": 0,
                    "source_resolution": history.get("source_resolution"),
                    "phases": _safe_json(phases),
                },
                indent=2,
            )
        )
        exported["loss_json"] = str(loss_path)

        epochs = loss_history.get("epochs", list(range(len(loss_history.get("train_loss", [])))))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(epochs, loss_history.get("train_loss", []), color="#111111", linewidth=2, label="train_loss")
        components = loss_history.get("components", {}) or {}
        for name, values in components.items():
            if values and len(values) == len(epochs):
                ax.plot(epochs, values, linewidth=1, linestyle="--", label=str(name))
        ax.set_title(f"{algorithm_name} loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        loss_png = figures_dir / f"loss_curves_{safe_algo}_run0.png"
        fig.savefig(loss_png, bbox_inches="tight", dpi=150)
        plt.close(fig)
        exported["loss_png"] = str(loss_png)

    weight_history = history.get("weight_history", []) or []
    if weight_history:
        weight_dir = results_dir / "weight_history"
        weight_dir.mkdir(parents=True, exist_ok=True)
        weight_json = weight_dir / f"weights_{safe_algo}_run0.json"
        weight_json.write_text(
            json.dumps(
                {
                    "algorithm": algorithm_name,
                    "run_id": 0,
                    "source_resolution": history.get("source_resolution"),
                    "records": _safe_json(weight_history),
                },
                indent=2,
            )
        )
        exported["weight_json"] = str(weight_json)

        df = pd.DataFrame(weight_history)
        weight_csv = weight_dir / f"weights_{safe_algo}_run0.csv"
        df.to_csv(weight_csv, index=False)
        exported["weight_csv"] = str(weight_csv)

        if "epoch" in df.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for col, color, width, style in [
                ("std", "#1f77b4", 2.0, "-"),
                ("min", "#2ca02c", 1.5, "-"),
                ("max", "#d62728", 1.5, "-"),
                ("mean", "#444444", 1.0, "--"),
            ]:
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], color=color, linewidth=width, linestyle=style, label=col)
            ax.set_title(f"{algorithm_name} cell-weight dynamics")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cell weight")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False)
            fig.tight_layout()
            weight_png = figures_dir / f"weight_dynamics_{safe_algo}_run0.png"
            fig.savefig(weight_png, bbox_inches="tight", dpi=150)
            plt.close(fig)
            exported["weight_png"] = str(weight_png)

    return exported


def _load_reference_metrics(path: Path, seed: int) -> Dict[str, Any] | None:
    if not path:
        return None

    candidate_paths = []
    if path.is_file():
        candidate_paths.append(path)
    else:
        candidate_paths.append(path / "results" / "results.json")
        candidate_paths.append(path / "runs" / f"seed_{seed}" / "results" / "results.json")

    for candidate in candidate_paths:
        if candidate.exists():
            payload = json.loads(candidate.read_text())
            results = payload.get("results", [])
            if results:
                return results[0]
    return None


def _run_desc_dependency_fallback(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    data_path: Path,
    adata_proc: ad.AnnData,
    prep_stats: Dict[str, Any],
    resolutions: List[float],
    config_dir: Path,
    results_dir: Path,
    labels_dir: Path,
    embeddings_dir: Path,
    figures_dir: Path,
) -> int:
    """Write DESC-shaped outputs when the legacy TensorFlow stack is absent."""
    from sklearn.cluster import KMeans

    start = time.time()
    n_comps = min(50, int(adata_proc.n_obs) - 1, int(adata_proc.n_vars) - 1)
    if n_comps < 2:
        raise RuntimeError("DESC fallback needs at least two cells and two genes after preprocessing.")
    sc.pp.pca(adata_proc, n_comps=n_comps, random_state=int(args.seed))
    embedding = np.asarray(adata_proc.obsm["X_pca"], dtype=np.float32)
    labels_true = adata_proc.obs[args.label_key].astype(str).to_numpy()
    batch_values = (
        adata_proc.obs[args.batch_key].astype(str).to_numpy()
        if args.batch_key in adata_proc.obs.columns
        else np.array(["NA"] * adata_proc.n_obs, dtype=object)
    )
    n_clusters = max(2, len(np.unique(labels_true)))
    predicted = KMeans(n_clusters=n_clusters, n_init=10, random_state=int(args.seed)).fit_predict(embedding).astype(str)
    runtime = float(time.time() - start)

    rows = []
    metrics_by_resolution: Dict[str, Any] = {}
    aligned = align_labels(labels_true, predicted)
    for resolution in resolutions:
        res_key = _round_str(resolution)
        metrics = compute_metrics(
            labels_true=labels_true,
            labels_pred=predicted,
            embeddings=embedding,
            adata=adata_proc,
            batch_key=args.batch_key if args.batch_key in adata_proc.obs.columns else None,
            label_key=args.label_key,
            compute_scib=False,
        )
        metrics["legacy_dependency_fallback"] = "DESC TensorFlow stack unavailable"
        metrics["legacy_dependency_error"] = str(DESC_IMPORT_ERROR)
        per_cell_path = labels_dir / f"per_cell_desc_res_{res_key}.csv"
        pd.DataFrame(
            {
                "cell_id": adata_proc.obs_names.astype(str),
                "batch": batch_values,
                "true_label": labels_true,
                "predicted_label": predicted,
                "aligned_predicted_label": np.asarray(aligned, dtype=object).astype(str),
            }
        ).to_csv(per_cell_path, index=False)
        embedding_path = embeddings_dir / f"embedding_desc_res_{res_key}.npy"
        if not embedding_path.exists():
            np.save(embedding_path, embedding)
        _plot_umap(
            embedding=embedding,
            labels=pd.Series(aligned, index=adata_proc.obs_names, dtype="string"),
            output_path=figures_dir / f"umap_desc_res_{res_key}.png",
            seed=args.seed,
        )
        row = {"resolution": float(resolution), "runtime_total": runtime}
        row.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
        rows.append(row)
        metrics_by_resolution[res_key] = {
            "metrics": _safe_json(metrics),
            "label_column": "fallback_kmeans",
            "embedding_key": "X_pca",
            "per_cell_csv": str(per_cell_path),
            "embedding_npy": str(embedding_path.resolve()),
        }

    metrics_df = pd.DataFrame(rows).sort_values(["ARI", "NMI", "resolution"], ascending=[False, False, True])
    metrics_df.to_csv(results_dir / "analysis_results_by_resolution.csv", index=False)
    best_row = metrics_df.iloc[0].to_dict()
    best_resolution = _round_str(best_row["resolution"])
    best_metrics = metrics_by_resolution[best_resolution]["metrics"]
    pd.DataFrame([{k: v for k, v in best_row.items()}]).to_csv(results_dir / "analysis_results.csv", index=False)
    weighting_payload = {
        "weighted": bool(args.weighted),
        "warmup_epochs": args.warmup_epochs,
        "dynamic_weight_update_interval": args.dynamic_weight_update_interval,
        "dynamic_weight_momentum": args.dynamic_weight_momentum,
        "pseudo_label_method": args.pseudo_label_method,
        "weight_exponent": args.weight_exponent,
        "weight_n_clusters": args.weight_n_clusters,
        "density_knn_k": args.density_knn_k,
        "density_weight_exponent": args.density_weight_exponent,
        "density_weight_clip": args.density_weight_clip,
        "weight_fusion_mode": args.weight_fusion_mode,
        "cluster_density_alpha": args.cluster_density_alpha,
        "cluster_weight_power": args.cluster_weight_power,
        "density_weight_power": args.density_weight_power,
        "min_cell_weight": args.min_cell_weight,
        "max_cell_weight": args.max_cell_weight,
        "weight_component_mode": args.weight_component_mode,
    }

    config_payload = {
        "data": {"file": str(data_path)},
        "preprocessing": _safe_json(prep_stats),
        "desc_params": {"resolutions": resolutions, "seed": args.seed, "weighting": weighting_payload},
        "context": {
            "label_key": args.label_key,
            "batch_key": args.batch_key,
            "desc_version": "unavailable",
            "fallback": "PCA_KMeans_because_DESC_dependency_missing",
            "dependency_error": str(DESC_IMPORT_ERROR),
            "runtime_seconds": runtime,
        },
        "output": {"directory": str(output_dir)},
    }
    (config_dir / "config_used.json").write_text(json.dumps(_safe_json(config_payload), indent=2))
    (results_dir / "results.json").write_text(
        json.dumps(
            _safe_json(
                {
                    "results": [
                        {
                            "algorithm_name": "desc_scraw_weighted_dependency_fallback"
                            if args.weighted
                            else "desc_dependency_fallback",
                            "run_id": 0,
                            "runtime": runtime,
                            "metrics": best_metrics,
                            "best_resolution": float(best_row["resolution"]),
                            "embeddings_shape": list(embedding.shape),
                            "preprocess_stats": prep_stats,
                            "metrics_by_resolution": metrics_by_resolution,
                        }
                    ]
                }
            ),
            indent=2,
        )
    )
    (results_dir / "summary.json").write_text(
        json.dumps(
            _safe_json(
                {
                    "data_file": str(data_path),
                    "runtime_seconds": runtime,
                    "best_resolution": float(best_row["resolution"]),
                    "best_metrics": best_metrics,
                    "fallback": "PCA_KMeans_because_DESC_dependency_missing",
                }
            ),
            indent=2,
        )
    )
    logger.warning("DESC legacy dependency unavailable; wrote fallback outputs. Error: %s", DESC_IMPORT_ERROR)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--min-genes-per-cell", type=int, default=100)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--pretrain-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--epochs-fit", type=float, default=5.0)
    parser.add_argument("--num-cores", type=int, default=8)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--weighted", action="store_true", help="Enable scRAW-style weighted AE pretraining.")
    parser.add_argument("--warmup-epochs", type=int, default=30)
    parser.add_argument("--dynamic-weight-update-interval", type=int, default=10)
    parser.add_argument("--dynamic-weight-momentum", type=float, default=0.7)
    parser.add_argument("--pseudo-label-method", default="leiden", choices=["leiden", "kmeans"])
    parser.add_argument("--weight-exponent", type=float, default=0.2)
    parser.add_argument("--weight-n-clusters", type=int, default=0)
    parser.add_argument("--density-knn-k", type=int, default=15)
    parser.add_argument("--density-weight-exponent", type=float, default=1.0)
    parser.add_argument("--density-weight-clip", type=float, default=5.0)
    parser.add_argument("--weight-fusion-mode", default="additive", choices=["additive", "multiplicative"])
    parser.add_argument("--cluster-density-alpha", type=float, default=0.6)
    parser.add_argument("--cluster-weight-power", type=float, default=1.0)
    parser.add_argument("--density-weight-power", type=float, default=1.0)
    parser.add_argument("--min-cell-weight", type=float, default=0.25)
    parser.add_argument("--max-cell-weight", type=float, default=10.0)
    parser.add_argument("--weight-component-mode", default="full", choices=["full", "density_only"])
    parser.add_argument("--rare-triplet-weight", type=float, default=0.0)
    parser.add_argument("--rare-triplet-margin", type=float, default=0.4)
    parser.add_argument("--rare-triplet-min-weight", type=float, default=1.2)
    parser.add_argument("--rare-triplet-start-epoch", type=int, default=60)
    parser.add_argument("--max-triplet-anchors-per-batch", type=int, default=64)
    parser.add_argument("--triplet-pseudo-label-method", default="kmeans", choices=["leiden", "kmeans"])
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    _seed_everything(args.seed)

    output_dir = Path(args.output).resolve()
    config_dir = output_dir / "config"
    results_dir = output_dir / "results"
    labels_dir = results_dir / "labels"
    embeddings_dir = results_dir / "embeddings"
    figures_dir = output_dir / "figures"
    desc_tmp_dir = output_dir / "desc_tmp"
    for d in [config_dir, results_dir, labels_dir, embeddings_dir, figures_dir, desc_tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).resolve()
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    resolutions = _parse_resolutions(args.resolutions)

    logger.info("Reading %s", data_path)
    adata = ad.read_h5ad(data_path)
    adata_proc, prep_stats = preprocess_adata(
        adata.copy(),
        min_genes_per_cell=args.min_genes_per_cell,
        max_genes_per_cell=args.max_genes_per_cell,
        min_cells_per_gene=args.min_cells_per_gene,
        target_sum=args.target_sum,
        n_top_genes=args.n_top_genes,
        scale_max_value=args.scale_max_value,
        hvg_flavor=args.hvg_flavor,
    )

    if args.label_key not in adata_proc.obs.columns:
        raise KeyError(f"Missing label key '{args.label_key}' in adata.obs")

    if desc is None:
        return _run_desc_dependency_fallback(
            args=args,
            output_dir=output_dir,
            data_path=data_path,
            adata_proc=adata_proc,
            prep_stats=prep_stats,
            resolutions=resolutions,
            config_dir=config_dir,
            results_dir=results_dir,
            labels_dir=labels_dir,
            embeddings_dir=embeddings_dir,
            figures_dir=figures_dir,
        )

    start = time.time()
    adata_desc = desc.train(
        adata_proc,
        dims=None,
        louvain_resolution=resolutions,
        n_neighbors=int(args.n_neighbors),
        pretrain_epochs=int(args.pretrain_epochs),
        batch_size=int(args.batch_size),
        epochs_fit=int(args.epochs_fit) if float(args.epochs_fit).is_integer() else float(args.epochs_fit),
        max_iter=int(args.max_iter),
        num_Cores=int(args.num_cores),
        use_GPU=bool(args.use_gpu),
        GPU_id=args.gpu_id,
        random_seed=int(args.seed),
        use_ae_weights=False,
        save_encoder_weights=False,
        save_dir=str(desc_tmp_dir),
        verbose=bool(args.verbose),
        do_umap=False,
        weighted_training=bool(args.weighted),
        warmup_epochs=int(args.warmup_epochs),
        dynamic_weight_update_interval=int(args.dynamic_weight_update_interval),
        dynamic_weight_momentum=float(args.dynamic_weight_momentum),
        pseudo_label_method=str(args.pseudo_label_method),
        weight_exponent=float(args.weight_exponent),
        weight_n_clusters=int(args.weight_n_clusters),
        density_knn_k=int(args.density_knn_k),
        density_weight_exponent=float(args.density_weight_exponent),
        density_weight_clip=float(args.density_weight_clip),
        weight_fusion_mode=str(args.weight_fusion_mode),
        cluster_density_alpha=float(args.cluster_density_alpha),
        cluster_weight_power=float(args.cluster_weight_power),
        density_weight_power=float(args.density_weight_power),
        min_cell_weight=float(args.min_cell_weight),
        max_cell_weight=float(args.max_cell_weight),
        weight_component_mode=str(args.weight_component_mode),
        rare_triplet_weight=float(args.rare_triplet_weight),
        rare_triplet_margin=float(args.rare_triplet_margin),
        rare_triplet_min_weight=float(args.rare_triplet_min_weight),
        rare_triplet_start_epoch=int(args.rare_triplet_start_epoch),
        max_triplet_anchors_per_batch=int(args.max_triplet_anchors_per_batch),
        triplet_pseudo_label_method=str(args.triplet_pseudo_label_method),
    )
    runtime = float(time.time() - start)
    algorithm_name = "desc_scraw_weighted" if args.weighted else "desc"
    training_history = _select_desc_training_history(
        adata_desc.uns.get("desc_training_history", {})
    )
    training_history_exports = _save_desc_training_history(
        history=training_history,
        algorithm_name=algorithm_name,
        results_dir=results_dir,
        figures_dir=figures_dir,
    )

    labels_true = adata_desc.obs[args.label_key].astype(str).to_numpy()
    batch_values = (
        adata_desc.obs[args.batch_key].astype(str).to_numpy()
        if args.batch_key in adata_desc.obs.columns
        else np.array(["NA"] * adata_desc.n_obs, dtype=object)
    )

    rows = []
    metrics_by_resolution: Dict[str, Any] = {}
    for resolution in resolutions:
        res_key = _round_str(resolution)
        label_col = f"desc_{res_key}"
        embed_key = f"X_Embeded_z{res_key}"
        prob_key = f"prob_matrix{res_key}"
        if label_col not in adata_desc.obs or embed_key not in adata_desc.obsm:
            raise KeyError(f"DESC output missing expected keys for resolution {res_key}")

        predicted = adata_desc.obs[label_col].astype(str).to_numpy()
        embedding = np.asarray(adata_desc.obsm[embed_key], dtype=np.float32)
        metrics = compute_metrics(
            labels_true=labels_true,
            labels_pred=predicted,
            embeddings=embedding,
            adata=adata_desc,
            batch_key=args.batch_key if args.batch_key in adata_desc.obs.columns else None,
            label_key=args.label_key,
            compute_scib=False,
        )
        aligned = align_labels(labels_true, predicted)

        per_cell = pd.DataFrame(
            {
                "cell_id": adata_desc.obs_names.astype(str),
                "batch": batch_values,
                "true_label": labels_true,
                "predicted_label": predicted,
                "aligned_predicted_label": np.asarray(aligned, dtype=object).astype(str),
            }
        )
        per_cell_path = labels_dir / f"per_cell_desc_res_{res_key}.csv"
        per_cell.to_csv(per_cell_path, index=False)

        np.save(embeddings_dir / f"embedding_desc_res_{res_key}.npy", embedding)
        if prob_key in adata_desc.uns:
            np.save(embeddings_dir / f"prob_desc_res_{res_key}.npy", np.asarray(adata_desc.uns[prob_key]))

        _plot_umap(
            embedding=embedding,
            labels=pd.Series(aligned, index=adata_desc.obs_names, dtype="string"),
            output_path=figures_dir / f"umap_desc_res_{res_key}.png",
            seed=args.seed,
        )

        row = {"resolution": float(resolution), "runtime_total": runtime}
        for k, v in metrics.items():
            if isinstance(v, dict):
                continue
            row[k] = v
        rows.append(row)

        metrics_by_resolution[res_key] = {
            "metrics": _safe_json(metrics),
            "label_column": label_col,
            "embedding_key": embed_key,
            "per_cell_csv": str(per_cell_path),
            "embedding_npy": str((embeddings_dir / f"embedding_desc_res_{res_key}.npy").resolve()),
        }

    metrics_df = pd.DataFrame(rows).sort_values(["ARI", "NMI", "resolution"], ascending=[False, False, True])
    metrics_df.to_csv(results_dir / "analysis_results_by_resolution.csv", index=False)

    best_row = metrics_df.iloc[0].to_dict()
    best_resolution = _round_str(best_row["resolution"])
    best_metrics = metrics_by_resolution[best_resolution]["metrics"]

    analysis_best = pd.DataFrame([{k: v for k, v in best_row.items() if k != "resolution"}])
    analysis_best.insert(0, "resolution", float(best_row["resolution"]))
    analysis_best.to_csv(results_dir / "analysis_results.csv", index=False)

    reference_payload = (
        _load_reference_metrics(reference_dir, seed=args.seed) if reference_dir is not None else None
    )
    comparison_payload = None
    if reference_payload is not None:
        ref_metrics = reference_payload.get("metrics", {})
        metric_names = [
            "NMI",
            "ARI",
            "ACC",
            "F1_Macro",
            "BalancedACC",
            "RareACC",
            "UltraRareACC",
            "KNN_Purity",
            "Silhouette",
        ]
        comparison_rows = []
        for name in metric_names:
            if name in best_metrics and name in ref_metrics:
                try:
                    desc_value = float(best_metrics[name])
                    ref_value = float(ref_metrics[name])
                except Exception:
                    continue
                comparison_rows.append(
                    {
                        "metric": name,
                        "desc": desc_value,
                        "reference": ref_value,
                        "delta_desc_minus_reference": desc_value - ref_value,
                    }
                )
        if comparison_rows:
            comparison_df = pd.DataFrame(comparison_rows)
            comparison_df.to_csv(results_dir / "comparison_to_reference.csv", index=False)
            comparison_payload = {
                "reference_results_json": str(
                    (reference_dir / "results" / "results.json").resolve()
                    if reference_dir.is_dir() and (reference_dir / "results" / "results.json").exists()
                    else reference_dir
                ),
                "best_resolution": float(best_row["resolution"]),
                "comparisons": comparison_rows,
            }

    config_payload = {
        "data": {"file": str(data_path)},
        "reference_dir": str(reference_dir) if reference_dir is not None else "",
        "preprocessing": {
            "n_top_genes": args.n_top_genes,
            "min_genes_per_cell": args.min_genes_per_cell,
            "max_genes_per_cell": args.max_genes_per_cell,
            "min_cells_per_gene": args.min_cells_per_gene,
            "target_sum": args.target_sum,
            "scale_max_value": args.scale_max_value,
            "hvg_flavor": args.hvg_flavor,
        },
        "desc_params": {
            "resolutions": resolutions,
            "n_neighbors": args.n_neighbors,
            "pretrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "max_iter": args.max_iter,
            "epochs_fit": args.epochs_fit,
            "num_cores": args.num_cores,
            "use_gpu": args.use_gpu,
            "gpu_id": args.gpu_id,
            "seed": args.seed,
            "weighted": bool(args.weighted),
            "warmup_epochs": args.warmup_epochs,
            "dynamic_weight_update_interval": args.dynamic_weight_update_interval,
            "dynamic_weight_momentum": args.dynamic_weight_momentum,
            "pseudo_label_method": args.pseudo_label_method,
            "weight_exponent": args.weight_exponent,
            "weight_n_clusters": args.weight_n_clusters,
            "density_knn_k": args.density_knn_k,
            "density_weight_exponent": args.density_weight_exponent,
            "density_weight_clip": args.density_weight_clip,
            "weight_fusion_mode": args.weight_fusion_mode,
            "cluster_density_alpha": args.cluster_density_alpha,
            "cluster_weight_power": args.cluster_weight_power,
            "density_weight_power": args.density_weight_power,
            "min_cell_weight": args.min_cell_weight,
            "max_cell_weight": args.max_cell_weight,
            "weight_component_mode": args.weight_component_mode,
            "rare_triplet_weight": args.rare_triplet_weight,
            "rare_triplet_margin": args.rare_triplet_margin,
            "rare_triplet_min_weight": args.rare_triplet_min_weight,
            "rare_triplet_start_epoch": args.rare_triplet_start_epoch,
            "max_triplet_anchors_per_batch": args.max_triplet_anchors_per_batch,
            "triplet_pseudo_label_method": args.triplet_pseudo_label_method,
        },
        "context": {
            "label_key": args.label_key,
            "batch_key": args.batch_key,
            "desc_version": getattr(desc, "__version__", "unknown"),
            "runtime_seconds": runtime,
            "preprocess_stats": prep_stats,
        },
        "output": {"directory": str(output_dir)},
    }
    (config_dir / "config_used.json").write_text(json.dumps(_safe_json(config_payload), indent=2))

    result_payload = {
        "results": [
            {
                "algorithm_name": algorithm_name,
                "run_id": 0,
                "runtime": runtime,
                "metrics": best_metrics,
                "params": _safe_json(config_payload["desc_params"]),
                "best_resolution": float(best_row["resolution"]),
                "embeddings_shape": list(np.asarray(adata_desc.obsm[f"X_Embeded_z{best_resolution}"]).shape),
                "preprocess_stats": _safe_json(prep_stats),
                "metrics_by_resolution": metrics_by_resolution,
                "comparison_to_reference": comparison_payload,
                "loss_history": _safe_json(
                    [training_history["loss_history"]]
                    if training_history.get("loss_history", {}).get("train_loss")
                    else []
                ),
                "weight_history": _safe_json(training_history.get("weight_history", [])),
                "training_history_exports": _safe_json(training_history_exports),
            }
        ]
    }
    (results_dir / "results.json").write_text(json.dumps(_safe_json(result_payload), indent=2))

    summary_payload = {
        "data_file": str(data_path),
        "runtime_seconds": runtime,
        "best_resolution": float(best_row["resolution"]),
        "best_metrics": best_metrics,
        "reference_comparison": comparison_payload,
        "training_history_exports": _safe_json(training_history_exports),
    }
    (results_dir / "summary.json").write_text(json.dumps(_safe_json(summary_payload), indent=2))

    logger.info("Best DESC resolution: %s", best_row["resolution"])
    logger.info("Best metrics: %s", json.dumps(_safe_json(best_metrics), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
