#!/usr/bin/env python3
"""Run paper batch-correction baselines on one h5ad and export scRAW-style artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, issparse


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
SRC_ROOT = PROJECT_ROOT / "vendor" / "scraw_dedicated" / "src"
SCRBENCH_SRC_ROOT = PROJECT_ROOT / "src"
for path in (SRC_ROOT, SCRBENCH_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scraw_dedicated.metrics import align_labels, compute_metrics, compute_scib_metrics
from scraw_dedicated.resolution_selection import rank_resolutions, selection_metadata


logger = logging.getLogger("run_batch_baseline_benchmark")

COMBAT_SEQ_R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Expected: counts_mtx batch_txt corrected_mtx")
}
counts_path <- args[[1]]
batch_path <- args[[2]]
output_path <- args[[3]]

if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("Missing R package 'Matrix'.")
}
if (!requireNamespace("sva", quietly = TRUE)) {
  stop("Missing R package 'sva'. Install bioconductor-sva to run ComBat-seq.")
}

counts <- Matrix::readMM(counts_path)
batch <- readLines(batch_path, warn = FALSE)
if (ncol(counts) != length(batch)) {
  stop(sprintf("Batch vector has %d entries but count matrix has %d cells.", length(batch), ncol(counts)))
}

if (length(unique(batch)) < 2) {
  Matrix::writeMM(counts, output_path)
  quit(save = "no", status = 0)
}

counts <- as.matrix(counts)
counts <- round(counts)
counts[counts < 0] <- 0
storage.mode(counts) <- "integer"

corrected <- sva::ComBat_seq(counts = counts, batch = batch)
corrected <- round(corrected)
corrected[!is.finite(corrected)] <- 0
corrected[corrected < 0] <- 0
storage.mode(corrected) <- "integer"
Matrix::writeMM(Matrix::Matrix(corrected, sparse = TRUE), output_path)
"""


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


def _to_contiguous_float32(X: Any) -> np.ndarray:
    return np.array(_to_dense(X), dtype=np.float32, copy=True, order="C")


def _counts_for_combat_seq(adata: ad.AnnData) -> Any:
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if issparse(counts):
        out = counts.copy().tocsc()
        out.data = np.rint(out.data).astype(np.int32, copy=False)
        out.data[out.data < 0] = 0
        out.eliminate_zeros()
        return out

    out = np.rint(_to_dense(counts)).astype(np.int32, copy=False)
    out[out < 0] = 0
    return np.ascontiguousarray(out)


def _axis_sums(X: Any, axis: int) -> np.ndarray:
    return np.asarray(X.sum(axis=axis)).ravel()


def _resolve_combat_seq_rscript(rscript: str) -> str:
    if rscript:
        return str(rscript)

    candidates = [
        PROJECT_ROOT / "envs" / "combat_seq_r" / "bin" / "Rscript",
        PROJECT_ROOT / "envs" / "tran2020_r" / "bin" / "Rscript",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "Rscript"


def _run_combat_seq(
    work: ad.AnnData,
    *,
    batch_key: str,
    rscript: str,
) -> Any:
    batch_values = work.obs[batch_key].astype(str)
    counts = _counts_for_combat_seq(work)
    if int(batch_values.nunique()) < 2:
        logger.info("ComBat-seq skipped because only one batch level is present; reusing raw counts.")
        return counts

    rscript_exe = _resolve_combat_seq_rscript(rscript)
    with tempfile.TemporaryDirectory(prefix="combat_seq_") as tmp_raw:
        tmp = Path(tmp_raw)
        counts_path = tmp / "counts_gene_by_cell.mtx"
        batch_path = tmp / "batch.txt"
        corrected_path = tmp / "corrected_gene_by_cell.mtx"
        script_path = tmp / "run_combat_seq.R"

        counts_gene_by_cell = counts.T
        if not issparse(counts_gene_by_cell):
            counts_gene_by_cell = csc_matrix(counts_gene_by_cell)
        mmwrite(counts_path, counts_gene_by_cell)
        batch_path.write_text("\n".join(batch_values.tolist()) + "\n")
        script_path.write_text(COMBAT_SEQ_R_SCRIPT)

        logger.info("Running ComBat-seq via %s", rscript_exe)
        try:
            completed = subprocess.run(
                [rscript_exe, "--vanilla", str(script_path), str(counts_path), str(batch_path), str(corrected_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Rscript was not found for ComBat-seq. Pass --combat-seq-rscript or set up "
                "envs/combat_seq_r/bin/Rscript."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "ComBat-seq failed in R.\n"
                f"stdout:\n{exc.stdout}\n"
                f"stderr:\n{exc.stderr}"
            ) from exc

        if completed.stdout.strip():
            logger.info("ComBat-seq stdout:\n%s", completed.stdout.strip())
        if completed.stderr.strip():
            logger.info("ComBat-seq stderr:\n%s", completed.stderr.strip())

        corrected = mmread(corrected_path).T
        if issparse(corrected):
            corrected = corrected.tocsr()
            corrected.data = np.asarray(corrected.data, dtype=np.float32)
            corrected.eliminate_zeros()
            return corrected
        return np.asarray(corrected, dtype=np.float32)


def _ensure_obs_first_embedding(embedding: Any, n_obs: int, context: str) -> np.ndarray:
    arr = _to_contiguous_float32(embedding)
    if arr.ndim != 2:
        raise ValueError(f"{context} produced an embedding with ndim={arr.ndim}, expected 2.")
    if arr.shape[0] == int(n_obs):
        return arr
    if arr.shape[1] == int(n_obs):
        logger.info("%s returned a transposed embedding; transposing it back to obs-first.", context)
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(
        f"{context} produced an embedding with shape {tuple(arr.shape)}; expected one dimension to match n_obs={n_obs}."
    )


def _matrix_nnz_per_row(X: Any) -> np.ndarray:
    if issparse(X):
        return np.asarray(X.getnnz(axis=1)).ravel()
    return np.asarray((np.asarray(X) > 0).sum(axis=1)).ravel()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        # Torch is optional for the non-scVI baselines.
        pass


def _parse_resolutions(raw: str) -> List[float]:
    vals = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("At least one Leiden resolution is required.")
    return vals


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
) -> Tuple[ad.AnnData, Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "n_obs_input": int(adata.n_obs),
        "n_vars_input": int(adata.n_vars),
    }

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    sc.pp.filter_cells(adata, min_genes=int(min_genes_per_cell))
    stats["n_obs_after_min_genes"] = int(adata.n_obs)

    if int(max_genes_per_cell) > 0:
        n_genes = _matrix_nnz_per_row(adata.layers["counts"])
        mask = n_genes <= int(max_genes_per_cell)
        adata = adata[mask].copy()
    stats["n_obs_after_max_genes"] = int(adata.n_obs)

    sc.pp.filter_genes(adata, min_cells=int(min_cells_per_gene))
    stats["n_vars_after_min_cells"] = int(adata.n_vars)

    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=float(target_sum))
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(n_top_genes),
        flavor=str(hvg_flavor),
        subset=True,
    )
    stats["n_vars_after_hvg"] = int(adata.n_vars)

    # ComBat should see the log-normalized HVG matrix before z-scoring.
    adata.layers["pre_scale_log1p_hvg"] = adata.X.copy()
    sc.pp.scale(adata, max_value=float(scale_max_value))
    adata.X = np.asarray(_to_dense(adata.X), dtype=np.float32)

    stats["n_obs_final"] = int(adata.n_obs)
    stats["n_vars_final"] = int(adata.n_vars)
    return adata, stats


def _plot_umap(embedding: np.ndarray, labels: pd.Series, output_path: Path, seed: int) -> None:
    work = ad.AnnData(np.asarray(embedding, dtype=np.float32))
    sc.pp.neighbors(work, use_rep="X", n_neighbors=15)
    sc.tl.umap(work, random_state=int(seed))
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


def _integrate(
    adata: ad.AnnData,
    method: str,
    batch_key: str,
    seed: int,
    n_pcs: int,
    target_sum: float,
    scale_max_value: float,
    harmony_max_iter: int,
    harmony_nclust: int,
    scanorama_knn: int,
    scanorama_sigma: float,
    scanorama_alpha: float,
    scanorama_batch_size: int,
    scanorama_approx: bool,
    combat_seq_rscript: str,
) -> Tuple[ad.AnnData, np.ndarray]:
    method = str(method).strip().lower()
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Missing batch key '{batch_key}' in adata.obs")

    work = adata.copy()
    batch_values = work.obs[batch_key].astype(str)
    n_batches = int(batch_values.nunique())
    n_pcs_eff = max(2, min(int(n_pcs), work.n_obs - 1, work.n_vars - 1))

    if method == "pca_leiden":
        logger.info("Running PCA with %d components", n_pcs_eff)
        sc.pp.pca(work, n_comps=n_pcs_eff, svd_solver="arpack")
        embedding = _ensure_obs_first_embedding(work.obsm["X_pca"], work.n_obs, "PCA")
        work.obsm["X_integrated"] = embedding
        return work, embedding

    if method == "harmony":
        import harmonypy as hm

        logger.info("Running PCA before Harmony with %d components", n_pcs_eff)
        sc.pp.pca(work, n_comps=n_pcs_eff, svd_solver="arpack")
        if n_batches < 2:
            logger.info("Harmony skipped because only one batch level is present; reusing PCA embedding.")
            embedding = _to_contiguous_float32(work.obsm["X_pca"])
            work.obsm["X_integrated"] = embedding
            return work, embedding

        logger.info("Running Harmony on %d cells across %d batches", work.n_obs, n_batches)
        ho = hm.run_harmony(
            _to_contiguous_float32(work.obsm["X_pca"]),
            work.obs.copy(),
            vars_use=[batch_key],
            nclust=int(harmony_nclust),
            max_iter_harmony=int(harmony_max_iter),
        )
        embedding = _ensure_obs_first_embedding(ho.Z_corr, work.n_obs, "Harmony")
        work.obsm["X_integrated"] = embedding
        return work, embedding

    if method == "combat":
        logger.info("Preparing pre-scale matrix for ComBat on %d cells x %d genes", work.n_obs, work.n_vars)
        if "pre_scale_log1p_hvg" in work.layers:
            work.X = _to_contiguous_float32(work.layers["pre_scale_log1p_hvg"])
        else:
            logger.warning(
                "Missing pre_scale_log1p_hvg layer for ComBat; falling back to current adata.X."
            )
        logger.info("Running ComBat")
        sc.pp.combat(work, key=batch_key)
        work.X = _to_contiguous_float32(work.X)
        if not np.isfinite(work.X).all():
            n_bad = int(np.size(work.X) - np.isfinite(work.X).sum())
            logger.warning("ComBat produced %d non-finite values; replacing them with 0 before PCA.", n_bad)
            work.X = np.nan_to_num(work.X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        logger.info("Scaling ComBat-corrected matrix")
        sc.pp.scale(work, max_value=float(scale_max_value))
        work.X = _to_contiguous_float32(work.X)
        logger.info("Running PCA after ComBat with %d components", n_pcs_eff)
        sc.pp.pca(work, n_comps=n_pcs_eff, svd_solver="arpack")
        embedding = _ensure_obs_first_embedding(work.obsm["X_pca"], work.n_obs, "ComBat PCA")
        work.obsm["X_integrated"] = embedding
        return work, embedding

    if method == "combat_seq":
        batch_counts = work.obs[batch_key].astype(str).value_counts()
        singleton_batches = batch_counts[batch_counts == 1].index.tolist()
        if singleton_batches and int(batch_counts.size) > 1:
            logger.warning(
                "ComBat-seq cannot handle singleton batches; dropping %d cell(s) from batches: %s",
                len(singleton_batches),
                ", ".join(map(str, singleton_batches[:10])),
            )
            keep_mask = ~work.obs[batch_key].astype(str).isin(singleton_batches)
            work = work[keep_mask].copy()
            n_pcs_eff = max(2, min(int(n_pcs), work.n_obs - 1, work.n_vars - 1))
        counts_preview = _counts_for_combat_seq(work)
        cell_has_counts = _axis_sums(counts_preview, axis=1) > 0
        gene_has_counts = _axis_sums(counts_preview, axis=0) > 0
        if not bool(cell_has_counts.all()) or not bool(gene_has_counts.all()):
            logger.warning(
                "ComBat-seq requires positive library sizes; dropping %d cell(s) and %d gene(s) with zero HVG counts.",
                int((~cell_has_counts).sum()),
                int((~gene_has_counts).sum()),
            )
            work = work[cell_has_counts, gene_has_counts].copy()
            n_pcs_eff = max(2, min(int(n_pcs), work.n_obs - 1, work.n_vars - 1))
        logger.info("Preparing raw-count HVG matrix for ComBat-seq on %d cells x %d genes", work.n_obs, work.n_vars)
        corrected_counts = _run_combat_seq(work, batch_key=batch_key, rscript=combat_seq_rscript)
        work.X = corrected_counts
        logger.info("Normalizing and log-transforming ComBat-seq-corrected counts")
        sc.pp.normalize_total(work, target_sum=float(target_sum))
        sc.pp.log1p(work)
        logger.info("Scaling ComBat-seq-corrected matrix")
        sc.pp.scale(work, max_value=float(scale_max_value))
        work.X = _to_contiguous_float32(work.X)
        if not np.isfinite(work.X).all():
            n_bad = int(np.size(work.X) - np.isfinite(work.X).sum())
            logger.warning("ComBat-seq produced %d non-finite values; replacing them with 0 before PCA.", n_bad)
            work.X = np.nan_to_num(work.X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        logger.info("Running PCA after ComBat-seq with %d components", n_pcs_eff)
        sc.pp.pca(work, n_comps=n_pcs_eff, svd_solver="arpack")
        embedding = _ensure_obs_first_embedding(work.obsm["X_pca"], work.n_obs, "ComBat-seq PCA")
        work.obsm["X_integrated"] = embedding
        return work, embedding

    if method == "scanorama":
        logger.info("Running PCA before Scanorama with %d components", n_pcs_eff)
        sc.pp.pca(work, n_comps=n_pcs_eff, svd_solver="arpack")
        min_batch_size = int(batch_values.value_counts().min())
        knn_eff = max(1, min(int(scanorama_knn), min_batch_size))
        if knn_eff != int(scanorama_knn):
            logger.info(
                "Scanorama knn reduced from %d to %d because the smallest batch has %d cells.",
                int(scanorama_knn),
                knn_eff,
                min_batch_size,
            )
        sce.pp.scanorama_integrate(
            work,
            key=batch_key,
            basis="X_pca",
            adjusted_basis="X_scanorama",
            knn=knn_eff,
            sigma=float(scanorama_sigma),
            approx=bool(scanorama_approx),
            alpha=float(scanorama_alpha),
            batch_size=int(scanorama_batch_size),
        )
        embedding = _ensure_obs_first_embedding(work.obsm["X_scanorama"], work.n_obs, "Scanorama")
        work.obsm["X_integrated"] = embedding
        return work, embedding

    if method == "scvi":
        try:
            import scvi
        except Exception as exc:  # pragma: no cover - handled at runtime
            raise ImportError(
                "scvi-tools is required for method='scvi'. Install it in the runtime environment."
            ) from exc

        if "counts" not in work.layers:
            raise KeyError("scVI integration requires a raw-counts layer named 'counts'.")

        logger.info(
            "Running scVI on %d cells x %d genes with batch key '%s'",
            work.n_obs,
            work.n_vars,
            batch_key,
        )
        try:
            scvi.settings.seed = int(seed)
        except Exception:
            logger.warning("Unable to set scvi.settings.seed; continuing with external seeding only.")

        scvi.model.SCVI.setup_anndata(work, layer="counts", batch_key=batch_key)
        model = scvi.model.SCVI(
            work,
            n_hidden=128,
            n_latent=10,
            n_layers=1,
            dropout_rate=0.1,
            dispersion="gene",
            gene_likelihood="zinb",
            use_observed_lib_size=True,
            latent_distribution="normal",
        )
        try:
            import torch

            has_gpu = bool(torch.cuda.is_available())
        except Exception:
            has_gpu = False
        accelerator = "gpu" if has_gpu else "cpu"
        devices = 1 if has_gpu else "auto"
        model.train(
            max_epochs=400,
            batch_size=128,
            early_stopping=True,
            accelerator=accelerator,
            devices=devices,
        )
        embedding = _ensure_obs_first_embedding(
            model.get_latent_representation(),
            work.n_obs,
            "scVI",
        )
        work.obsm["X_scvi"] = embedding
        work.obsm["X_integrated"] = embedding
        return work, embedding

    raise ValueError(f"Unsupported method '{method}'")


def _cluster_embedding(
    adata: ad.AnnData,
    embedding: np.ndarray,
    resolutions: List[float],
    seed: int,
    n_neighbors: int,
) -> Dict[str, np.ndarray]:
    work = adata.copy()
    work.obsm["X_cluster"] = np.asarray(embedding, dtype=np.float32)
    sc.pp.neighbors(work, use_rep="X_cluster", n_neighbors=int(n_neighbors))

    labels_by_resolution: Dict[str, np.ndarray] = {}
    for resolution in resolutions:
        key = f"leiden_{resolution:.3f}".rstrip("0").rstrip(".")
        sc.tl.leiden(work, resolution=float(resolution), random_state=int(seed), key_added=key)
        labels_by_resolution[str(float(resolution))] = work.obs[key].astype(str).to_numpy()
    return labels_by_resolution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Input .h5ad file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--method", required=True, choices=["harmony", "combat", "combat_seq", "scanorama", "pca_leiden", "scvi"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-key", default="cell_type")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0,1.2,1.4")
    parser.add_argument(
        "--selection-expected-n-classes",
        type=int,
        default=0,
        help="Override the class count used for Leiden resolution selection.",
    )
    parser.add_argument("--min-genes-per-cell", type=int, default=100)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--harmony-max-iter", type=int, default=10)
    parser.add_argument("--harmony-nclust", type=int, default=50)
    parser.add_argument("--scanorama-knn", type=int, default=20)
    parser.add_argument("--scanorama-sigma", type=float, default=15.0)
    parser.add_argument("--scanorama-alpha", type=float, default=0.1)
    parser.add_argument("--scanorama-batch-size", type=int, default=5000)
    parser.add_argument("--scanorama-approx", choices=["true", "false"], default="true")
    parser.add_argument(
        "--combat-seq-rscript",
        default="",
        help="Rscript executable for sva::ComBat_seq. Defaults to envs/combat_seq_r/bin/Rscript when present.",
    )
    parser.add_argument("--reference-dir", default="")
    parser.add_argument("--compute-scib", action="store_true")
    parser.add_argument("--scib-n-jobs", type=int, default=1)
    parser.add_argument("--skip-umap-plots", action="store_true")
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
    for d in [config_dir, results_dir, labels_dir, embeddings_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data).resolve()
    reference_dir = Path(args.reference_dir).resolve() if args.reference_dir else None
    resolutions = _parse_resolutions(args.resolutions)

    logger.info("Reading %s", data_path)
    adata = ad.read_h5ad(data_path)
    logger.info("Starting preprocessing")
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
    logger.info("Finished preprocessing with %d cells x %d genes", adata_proc.n_obs, adata_proc.n_vars)

    if args.label_key not in adata_proc.obs.columns:
        raise KeyError(f"Missing label key '{args.label_key}' in adata.obs")
    if args.batch_key not in adata_proc.obs.columns:
        raise KeyError(f"Missing batch key '{args.batch_key}' in adata.obs")

    start = time.time()
    logger.info("Starting %s integration", args.method)
    adata_int, embedding = _integrate(
        adata=adata_proc,
        method=args.method,
        batch_key=args.batch_key,
        seed=args.seed,
        n_pcs=args.n_pcs,
        target_sum=args.target_sum,
        scale_max_value=args.scale_max_value,
        harmony_max_iter=args.harmony_max_iter,
        harmony_nclust=args.harmony_nclust,
        scanorama_knn=args.scanorama_knn,
        scanorama_sigma=args.scanorama_sigma,
        scanorama_alpha=args.scanorama_alpha,
        scanorama_batch_size=args.scanorama_batch_size,
        scanorama_approx=str(args.scanorama_approx).lower() == "true",
        combat_seq_rscript=args.combat_seq_rscript,
    )
    runtime = float(time.time() - start)
    logger.info("Finished %s integration in %.2fs", args.method, runtime)

    labels_true = adata_int.obs[args.label_key].astype(str).to_numpy()
    batch_values = adata_int.obs[args.batch_key].astype(str).to_numpy()
    logger.info("Clustering embedding across %d resolutions", len(resolutions))
    labels_by_resolution = _cluster_embedding(
        adata=adata_int,
        embedding=embedding,
        resolutions=resolutions,
        seed=args.seed,
        n_neighbors=args.n_neighbors,
    )
    logger.info("Finished clustering")
    shared_scib_metrics: Dict[str, Any] = {}
    if args.compute_scib:
        logger.info("Computing scIB metrics")
        shared_scib_metrics = compute_scib_metrics(
            adata=adata_int,
            embeddings=embedding,
            batch_key=args.batch_key,
            label_key=args.label_key,
            n_jobs=int(args.scib_n_jobs),
        )
        logger.info("Finished scIB metrics")

    rows = []
    metrics_by_resolution: Dict[str, Any] = {}
    logger.info("Computing clustering metrics and writing per-resolution outputs")
    for resolution in resolutions:
        res_key = str(float(resolution))
        predicted = labels_by_resolution[res_key]
        metrics = compute_metrics(
            labels_true=labels_true,
            labels_pred=predicted,
            embeddings=embedding,
            adata=None,
            batch_key=None,
            label_key=None,
            compute_scib=False,
            scib_n_jobs=1,
        )
        metrics.update(shared_scib_metrics)
        aligned = align_labels(labels_true, predicted)

        per_cell = pd.DataFrame(
            {
                "cell_id": adata_int.obs_names.astype(str),
                "batch": batch_values,
                "true_label": labels_true,
                "predicted_label": predicted,
                "aligned_predicted_label": np.asarray(aligned, dtype=object).astype(str),
            }
        )
        per_cell_path = labels_dir / f"per_cell_{args.method}_res_{res_key}.csv"
        per_cell.to_csv(per_cell_path, index=False)

        embedding_path = embeddings_dir / f"embedding_{args.method}.npy"
        if not embedding_path.exists():
            np.save(embedding_path, embedding)

        if not args.skip_umap_plots:
            _plot_umap(
                embedding=embedding,
                labels=pd.Series(aligned, index=adata_int.obs_names, dtype="string"),
                output_path=figures_dir / f"umap_{args.method}_res_{res_key}.png",
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
            "per_cell_csv": str(per_cell_path),
            "embedding_npy": str(embedding_path.resolve()),
        }

    expected_n_classes = (
        int(args.selection_expected_n_classes)
        if int(args.selection_expected_n_classes) > 0
        else int(len(np.unique(labels_true)))
    )
    metrics_df = rank_resolutions(pd.DataFrame(rows), expected_n_classes=expected_n_classes)
    metrics_df.to_csv(results_dir / "analysis_results_by_resolution.csv", index=False)

    best_row = metrics_df.iloc[0].to_dict()
    best_resolution = str(float(best_row["resolution"]))
    best_metrics = metrics_by_resolution[best_resolution]["metrics"]

    analysis_best = pd.DataFrame([{k: v for k, v in best_row.items()}])
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
            "Batch correction",
            "Inter cell-type conservation",
            "Intra cell-type conservation",
            "scIB-E Total score",
        ]
        comparison_rows = []
        for name in metric_names:
            if name in best_metrics and name in ref_metrics:
                try:
                    value = float(best_metrics[name])
                    ref_value = float(ref_metrics[name])
                except Exception:
                    continue
                comparison_rows.append(
                    {
                        "metric": name,
                        args.method: value,
                        "reference": ref_value,
                        f"delta_{args.method}_minus_reference": value - ref_value,
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

    method_params = {
        "method": args.method,
        "seed": args.seed,
        "resolutions": resolutions,
        "harmony_max_iter": args.harmony_max_iter,
        "harmony_nclust": args.harmony_nclust,
        "scanorama_knn": args.scanorama_knn,
        "scanorama_sigma": args.scanorama_sigma,
        "scanorama_alpha": args.scanorama_alpha,
        "scanorama_batch_size": args.scanorama_batch_size,
        "scanorama_approx": str(args.scanorama_approx).lower() == "true",
        "combat_seq_rscript": _resolve_combat_seq_rscript(args.combat_seq_rscript) if args.method == "combat_seq" else "",
        "compute_scib": args.compute_scib,
        "scib_n_jobs": args.scib_n_jobs,
        "resolution_selection": selection_metadata(expected_n_classes),
    }
    if args.method == "scvi":
        method_params.update(
            {
                "scvi_n_hidden": 128,
                "scvi_n_latent": 10,
                "scvi_n_layers": 1,
                "scvi_dropout_rate": 0.1,
                "scvi_dispersion": "gene",
                "scvi_gene_likelihood": "zinb",
                "scvi_use_observed_lib_size": True,
                "scvi_latent_distribution": "normal",
                "scvi_max_epochs": 400,
                "scvi_batch_size": 128,
                "scvi_early_stopping": True,
                "scvi_setup_layer": "counts",
            }
        )

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
            "n_pcs": args.n_pcs,
            "n_neighbors": args.n_neighbors,
        },
        "method_params": method_params,
        "context": {
            "label_key": args.label_key,
            "batch_key": args.batch_key,
            "runtime_seconds": runtime,
            "preprocess_stats": prep_stats,
        },
        "output": {"directory": str(output_dir)},
    }
    (config_dir / "config_used.json").write_text(json.dumps(_safe_json(config_payload), indent=2))

    result_payload = {
        "results": [
            {
                "algorithm_name": args.method,
                "run_id": 0,
                "runtime": runtime,
                "metrics": best_metrics,
                "params": _safe_json(config_payload["method_params"]),
                "best_resolution": float(best_row["resolution"]),
                "embeddings_shape": list(np.asarray(embedding).shape),
                "preprocess_stats": _safe_json(prep_stats),
                "metrics_by_resolution": metrics_by_resolution,
                "comparison_to_reference": comparison_payload,
            }
        ]
    }
    (results_dir / "results.json").write_text(json.dumps(_safe_json(result_payload), indent=2))

    summary_payload = {
        "data_file": str(data_path),
        "method": args.method,
        "runtime_seconds": runtime,
        "best_resolution": float(best_row["resolution"]),
        "best_metrics": best_metrics,
        "reference_comparison": comparison_payload,
    }
    (results_dir / "summary.json").write_text(json.dumps(_safe_json(summary_payload), indent=2))
    logger.info("Saved outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
