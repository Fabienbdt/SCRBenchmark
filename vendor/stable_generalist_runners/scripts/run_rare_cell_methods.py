#!/usr/bin/env python3
"""Run external rare-cell methods on the stable_generalist benchmark datasets.

Outputs are intentionally shaped like the existing baseline outputs:
``results/analysis_results.csv``, per-cell labels, embeddings, config, and
``results.json``. A separate merge script imports successful runs into the
presentation tables used by ``data_exploration.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import norm


RUNNER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("SCRBENCHMARK_ROOT", Path(__file__).resolve().parents[3])).resolve()
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", REPO_ROOT.parent)).resolve()
SCRBENCH_ROOT = REPO_ROOT
SCRAW_SRC = SCRBENCH_ROOT / "vendor" / "scraw_dedicated" / "src"
SCCAD_ROOT = SCRBENCH_ROOT / "external" / "original_code" / "scCAD"
RPH_ROOT = SCRBENCH_ROOT / "external" / "original_code" / "rph_kmeans"
GINICLUST_ROOT = SCRBENCH_ROOT / "external" / "original_code" / "GiniClust"
SCAIDE_EMBED_SCRIPT = RUNNER_ROOT / "scripts" / "method_scaide_embedding.py"
SCAIDE_PYTHON = Path(os.environ.get("SCAIDE_PYTHON", sys.executable)).resolve()
R_ENV = Path(os.environ.get("SCRBENCH_R_ENV", RUNNER_ROOT / "envs" / "tran2020_r")).resolve()
R_SCRIPT = Path(os.environ.get("RSCRIPT", R_ENV / "bin" / "Rscript")).resolve()
R_COMPILER_WRAPPERS = Path(
    os.environ.get("R_COMPILER_WRAPPERS", WORKSPACE_ROOT / "tmp_r_compiler_wrappers")
).resolve()
GINICLUST_R_SCRIPT = RUNNER_ROOT / "scripts" / "method_giniclust.R"
CELLSIUS_R_SCRIPT = RUNNER_ROOT / "scripts" / "method_cellsius.R"
MCL_PATH = Path(os.environ.get("MCL_PATH", R_ENV / "bin" / "mcl")).resolve()
DEFAULT_MANIFEST = SCRBENCH_ROOT / "reproducibility" / "stable_generalist" / "stable_generalist_benchmark_13_manifest.csv"
DEFAULT_PRESENTATION_ROOT = SCRBENCH_ROOT / "results" / "stable_generalist_external"

if str(SCRAW_SRC) not in sys.path:
    sys.path.insert(0, str(SCRAW_SRC))

from scraw_dedicated.metrics import align_labels, compute_metrics, compute_scib_metrics
from scraw_dedicated.resolution_selection import rank_resolutions, selection_metadata


LOGGER = logging.getLogger("external_rare_methods_stable_generalist")

SCIB_COLUMNS = [
    "Batch correction",
    "Inter cell-type conservation",
    "Intra cell-type conservation",
    "scIB-E Total score",
]

PREPROCESSING = {
    "n_top_genes": 2000,
    "min_genes_per_cell": 200,
    "max_genes_per_cell": None,
    "min_cells_per_gene": 3,
    "target_sum": 20000.0,
    "scale_max_value": 10.0,
    "hvg_flavor": "seurat",
}

DISPLAY_NAMES = {
    ("scCAD", "standard"): "scCAD",
    ("scCAD", "harmony"): "scCAD+Harmony",
    ("scAIDE", "standard"): "scAIDE",
    ("scAIDE", "harmony"): "scAIDE+Harmony",
    ("GiniClust", "standard"): "GiniClust",
    ("GiniClust", "harmony"): "GiniClust+Harmony",
    ("CellSIUS", "standard"): "CellSIUS",
    ("CellSIUS", "harmony"): "CellSIUS+Harmony",
}

METHOD_SLUGS = {
    "scCAD": "sccad",
    "scAIDE": "scaide",
    "GiniClust": "giniclust",
    "CellSIUS": "cellsius",
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    path: Path
    label_key: str
    batch_key: str
    family: str
    n_labels_expected: int


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def _to_dense(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _to_float32(X: Any) -> np.ndarray:
    return np.asarray(_to_dense(X), dtype=np.float32)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if math.isnan(out) or math.isinf(out) else out
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(dict(payload)), indent=2), encoding="utf-8")


def _read_manifest(path: Path) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("enabled", "true")).lower() not in {"true", "1", "yes"}:
                continue
            specs.append(
                DatasetSpec(
                    dataset_key=str(row["dataset_id"]).strip(),
                    path=Path(str(row["path"]).strip()),
                    label_key=str(row["label_key"]).strip(),
                    batch_key=str(row["batch_key"]).strip(),
                    family=str(row.get("family", "")).strip(),
                    n_labels_expected=int(row["n_labels_expected"]),
                )
            )
    return specs


def _select_datasets(specs: Sequence[DatasetSpec], raw: str) -> List[DatasetSpec]:
    tokens = {token.strip() for token in str(raw).split(",") if token.strip()}
    if not tokens:
        return list(specs)
    return [spec for spec in specs if spec.dataset_key in tokens]


def _select_methods(raw: str) -> List[str]:
    if not raw.strip():
        return ["scCAD", "scAIDE", "GiniClust", "CellSIUS"]
    aliases = {
        "sccad": "scCAD",
        "scCAD": "scCAD",
        "scaide": "scAIDE",
        "scAIDE": "scAIDE",
        "giniclust": "GiniClust",
        "GiniClust": "GiniClust",
        "cellsius": "CellSIUS",
        "CellSIUS": "CellSIUS",
    }
    out: List[str] = []
    for token in [part.strip() for part in raw.split(",") if part.strip()]:
        if token not in aliases:
            raise ValueError(f"Unknown method: {token}")
        out.append(aliases[token])
    return out


def _select_variants(raw: str) -> List[str]:
    if not raw.strip():
        return ["standard", "harmony"]
    out = [part.strip().lower() for part in raw.split(",") if part.strip()]
    invalid = [part for part in out if part not in {"standard", "harmony"}]
    if invalid:
        raise ValueError(f"Unknown variant(s): {invalid}")
    return out


def _matrix_nnz_per_row(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray(X.getnnz(axis=1)).ravel()
    return np.asarray((np.asarray(X) > 0).sum(axis=1)).ravel()


def preprocess_shared(adata: ad.AnnData, params: Mapping[str, Any]) -> Tuple[ad.AnnData, Dict[str, Any]]:
    stats: Dict[str, Any] = {"n_obs_input": int(adata.n_obs), "n_vars_input": int(adata.n_vars)}
    work = adata.copy()
    if "counts" not in work.layers:
        work.layers["counts"] = work.X.copy()

    sc.pp.filter_cells(work, min_genes=int(params["min_genes_per_cell"]))
    stats["n_obs_after_min_genes"] = int(work.n_obs)

    max_genes = params.get("max_genes_per_cell")
    if max_genes is not None:
        n_genes = _matrix_nnz_per_row(work.layers["counts"])
        work = work[n_genes <= int(max_genes)].copy()
    stats["n_obs_after_max_genes"] = int(work.n_obs)

    sc.pp.filter_genes(work, min_cells=int(params["min_cells_per_gene"]))
    stats["n_vars_after_min_cells"] = int(work.n_vars)

    work.raw = work.copy()
    sc.pp.normalize_total(work, target_sum=float(params["target_sum"]))
    sc.pp.log1p(work)
    sc.pp.highly_variable_genes(
        work,
        n_top_genes=min(int(params["n_top_genes"]), int(work.n_vars)),
        flavor=str(params["hvg_flavor"]),
        subset=True,
    )
    stats["n_vars_after_hvg"] = int(work.n_vars)

    work.layers["pre_scale_log1p_hvg"] = work.X.copy()
    sc.pp.scale(work, max_value=float(params["scale_max_value"]))
    work.X = _to_float32(work.X)

    stats["n_obs_final"] = int(work.n_obs)
    stats["n_vars_final"] = int(work.n_vars)
    return work, stats


def _raw_counts_for_processed_cells(adata_proc: ad.AnnData) -> Tuple[Any, List[str]]:
    """Return non-HVG raw counts for the processed cell set."""
    if adata_proc.raw is not None:
        return adata_proc.raw.X, [str(x) for x in adata_proc.raw.var_names]
    return adata_proc.layers["counts"], [str(x) for x in adata_proc.var_names]


def _ensure_obs_first_embedding(embedding: Any, n_obs: int, context: str) -> np.ndarray:
    arr = _to_float32(embedding)
    if arr.ndim != 2:
        raise ValueError(f"{context} returned an embedding with ndim={arr.ndim}, expected 2.")
    if arr.shape[0] == int(n_obs):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[1] == int(n_obs):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"{context} returned shape {arr.shape}; expected one dimension to match n_obs={n_obs}.")


def compute_pca_embedding(adata: ad.AnnData, n_pcs: int, seed: int) -> np.ndarray:
    work = adata.copy()
    n_pcs_eff = max(2, min(int(n_pcs), int(work.n_obs) - 1, int(work.n_vars) - 1))
    sc.pp.pca(work, n_comps=n_pcs_eff, svd_solver="arpack", random_state=int(seed))
    return _ensure_obs_first_embedding(work.obsm["X_pca"], work.n_obs, "PCA")


def apply_harmony(embedding: np.ndarray, adata: ad.AnnData, batch_key: str, *, max_iter: int, nclust: int) -> np.ndarray:
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")
    if int(adata.obs[batch_key].astype(str).nunique()) < 2:
        return np.ascontiguousarray(embedding, dtype=np.float32)
    import harmonypy as hm

    ho = hm.run_harmony(
        np.ascontiguousarray(embedding, dtype=np.float32),
        adata.obs.copy(),
        vars_use=[batch_key],
        max_iter_harmony=int(max_iter),
        nclust=int(nclust),
    )
    return _ensure_obs_first_embedding(ho.Z_corr, adata.n_obs, "Harmony")


def _leiden_labels(embedding: np.ndarray, seed: int, resolution: float, n_neighbors: int = 15) -> np.ndarray:
    work = ad.AnnData(np.asarray(embedding, dtype=np.float32))
    work.obsm["X_cluster"] = np.asarray(embedding, dtype=np.float32)
    sc.pp.neighbors(work, use_rep="X_cluster", n_neighbors=int(n_neighbors))
    key = "leiden"
    sc.tl.leiden(work, resolution=float(resolution), random_state=int(seed), key_added=key)
    return work.obs[key].astype(str).to_numpy()


def cluster_embedding_ranked(
    embedding: np.ndarray,
    labels_true: np.ndarray,
    expected_n_classes: int,
    seed: int,
    resolutions: Sequence[float],
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    rows = []
    labels_by_res: Dict[str, np.ndarray] = {}
    for resolution in resolutions:
        pred = _leiden_labels(embedding, seed=seed, resolution=float(resolution))
        metrics = compute_metrics(labels_true=labels_true, labels_pred=pred, embeddings=embedding)
        row = {"resolution": float(resolution)}
        row.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
        rows.append(row)
        labels_by_res[str(float(resolution))] = pred
    ranked = rank_resolutions(pd.DataFrame(rows), expected_n_classes=int(expected_n_classes))
    best_resolution = float(ranked.iloc[0]["resolution"])
    return labels_by_res[str(float(best_resolution))], best_resolution, ranked


def run_sccad(X: np.ndarray, cell_names: Sequence[str], gene_names: Sequence[str], output_dir: Path, seed: int) -> np.ndarray:
    if str(SCCAD_ROOT) not in sys.path:
        sys.path.insert(0, str(SCCAD_ROOT))
    import scCAD as sccad_module

    raw_dir = output_dir / "raw_scCAD"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _result, _score, subclusters, _degs = sccad_module.scCAD(
        data=np.asarray(X, dtype=np.float32),
        dataName="stable_generalist",
        cellNames=np.asarray(cell_names, dtype=str),
        geneNames=np.asarray(gene_names, dtype=str),
        normalization=False,
        seed=int(seed),
        save_path=str(raw_dir) + os.sep,
    )
    return np.asarray(subclusters, dtype=object).astype(str)


def _gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    mu = float(np.mean(x))
    if mu <= 0.0:
        return 0.0
    ox = np.sort(x)
    n = len(ox)
    dsum = float(np.dot(2 * np.arange(1, n + 1) - n - 1, ox))
    return dsum / (mu * n * max(1, n - 1))


def _gini_sparse_columns(X: sparse.spmatrix) -> np.ndarray:
    csc = X.tocsc()
    n_obs, n_vars = csc.shape
    scores = np.zeros(n_vars, dtype=np.float64)
    if n_obs <= 1:
        return scores
    for gene_idx in range(n_vars):
        start, end = csc.indptr[gene_idx], csc.indptr[gene_idx + 1]
        values = np.asarray(csc.data[start:end], dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        total = float(values.sum())
        if total <= 0.0:
            continue
        ordered = np.sort(values)
        n_nonzero = len(ordered)
        ranks = np.arange(n_obs - n_nonzero + 1, n_obs + 1, dtype=np.float64)
        dsum = float(np.dot(2 * ranks - n_obs - 1, ordered))
        scores[gene_idx] = dsum / (total * max(1, n_obs - 1))
    return scores


def _giniclust_gene_filter(X_counts: Any, gene_names: Sequence[str]) -> Tuple[Any, List[str], Dict[str, Any]]:
    if len(gene_names) != int(X_counts.shape[1]):
        raise ValueError(f"GiniClust gene name count {len(gene_names)} does not match matrix shape {X_counts.shape}")
    if sparse.issparse(X_counts):
        expressed_per_gene = np.asarray((X_counts > 1).sum(axis=0)).ravel()
    else:
        X_arr = np.asarray(X_counts)
        expressed_per_gene = np.asarray((X_arr > 1).sum(axis=0)).ravel()
    non_mir = np.asarray(["MIR" not in str(name) and "Mir" not in str(name) for name in gene_names], dtype=bool)
    keep = (expressed_per_gene >= 3) & non_mir
    if int(keep.sum()) < 2:
        keep = np.ones(len(gene_names), dtype=bool)
        filter_status = "disabled_too_few_genes_after_native_gene_filter"
    else:
        filter_status = "giniclust_native_gene_filter_no_cell_drop"
    return X_counts[:, keep], [str(gene_names[i]) for i in np.where(keep)[0]], {
        "giniclust_gene_filter_status": filter_status,
        "giniclust_n_input_genes": int(len(gene_names)),
        "giniclust_n_genes_after_native_filter": int(keep.sum()),
    }


def run_giniclust_python(
    X_counts: Any,
    expected_n_labels: int,
    *,
    gene_names: Optional[Sequence[str]] = None,
    max_dbscan_cells: int,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.cluster import DBSCAN, MiniBatchKMeans

    if gene_names is not None:
        X, filtered_gene_names, filter_info = _giniclust_gene_filter(X_counts, gene_names)
    else:
        X = X_counts
        filtered_gene_names = [str(i) for i in range(int(X_counts.shape[1]))]
        filter_info = {
            "giniclust_gene_filter_status": "generic_nonzero_filter",
            "giniclust_n_input_genes": int(X_counts.shape[1]),
        }
        if sparse.issparse(X):
            gene_keep = np.asarray(X.getnnz(axis=0)).ravel() >= 3
        else:
            gene_keep = np.asarray((np.asarray(X) > 0).sum(axis=0)).ravel() >= 3
        if int(gene_keep.sum()) >= 10:
            X = X[:, gene_keep]
            filtered_gene_names = [filtered_gene_names[i] for i in np.where(gene_keep)[0]]
    if sparse.issparse(X):
        X = X.astype(np.float32)
        gini_scores = _gini_sparse_columns(X)
    else:
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), copy=False)
        gini_scores = np.asarray([_gini(X[:, j]) for j in range(X.shape[1])], dtype=np.float64)
    finite = np.isfinite(gini_scores)
    if not bool(finite.any()):
        selected_idx = np.arange(min(X.shape[1], 200))
    else:
        z = (gini_scores - np.nanmean(gini_scores[finite])) / (np.nanstd(gini_scores[finite]) + 1e-8)
        pvals = norm.sf(z)
        selected_idx = np.where((pvals < 1e-4) & finite)[0]
        if selected_idx.size < 10:
            selected_idx = np.argsort(-np.nan_to_num(gini_scores, nan=-np.inf))[: min(X.shape[1], 200)]

    Xsel = X[:, selected_idx]
    binary = (Xsel > 0).astype(np.float32 if sparse.issparse(Xsel) else bool)
    status = "dbscan_jaccard"
    if X.shape[0] <= int(max_dbscan_cells):
        dbscan_input = binary.toarray().astype(bool, copy=False) if sparse.issparse(binary) else np.asarray(binary, dtype=bool)
        labels = DBSCAN(eps=0.5, min_samples=3, metric="jaccard", algorithm="brute", n_jobs=-1).fit_predict(dbscan_input)
    else:
        status = "large_dataset_minibatch_kmeans_on_high_gini_genes"
        n_clusters = max(2, int(expected_n_labels))
        labels = MiniBatchKMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=int(seed),
            batch_size=2048,
        ).fit_predict(binary.astype(np.float32))
    labels_str = np.asarray([("Singleton" if int(x) == -1 else f"Cluster_{int(x) + 1}") for x in labels], dtype=object)
    info = {
        "implementation": "python_port_of_gini_selection_plus_clustering",
        "status": status,
        "n_genes_after_filter": int(X.shape[1]),
        "n_high_gini_genes": int(len(selected_idx)),
        "max_dbscan_cells": int(max_dbscan_cells),
        **filter_info,
    }
    if filtered_gene_names:
        info["n_selected_gene_names_available"] = int(len(filtered_gene_names))
    return labels_str.astype(str), info


def _run_rph_or_kmeans(embedding: np.ndarray, n_clusters: int, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.cluster import KMeans
    import sklearn.metrics.pairwise as pairwise

    original_paired = pairwise.paired_distances

    def paired_wrapper(X: Any, Y: Any, metric: str = "euclidean", **kwargs: Any) -> np.ndarray:
        return original_paired(X, Y, metric=metric, **kwargs)

    pairwise.paired_distances = paired_wrapper

    class KneeLocator:
        def __init__(self, x: Sequence[Any], y: Sequence[Any], *args: Any, **kwargs: Any) -> None:
            self.knee = x[int(len(x) // 2)] if len(x) else None

    sys.modules.setdefault("kneed", types.SimpleNamespace(KneeLocator=KneeLocator))
    if str(RPH_ROOT) not in sys.path:
        sys.path.insert(0, str(RPH_ROOT))

    try:
        from rph_kmeans import RPHKMeans

        model = RPHKMeans(
            n_clusters=int(n_clusters),
            n_init=10,
            point_reducer_version="py",
            final_kmeans_kwargs={"random_state": int(seed)},
            reduced_kmeans_kwargs={"random_state": int(seed)},
            verbose=0,
        )
        labels = model.fit_predict(np.asarray(embedding, dtype=np.float32))
        return np.asarray(labels, dtype=object).astype(str), {"clusterer": "RPHKMeans", "n_clusters": int(n_clusters)}
    except Exception as exc:
        LOGGER.warning("RPHKMeans failed; falling back to KMeans: %s", exc)
        labels = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(seed)).fit_predict(embedding)
        return np.asarray(labels, dtype=object).astype(str), {
            "clusterer": "KMeans_fallback",
            "n_clusters": int(n_clusters),
            "fallback_reason": repr(exc),
        }


def run_scaide(
    X_input: np.ndarray,
    adata_proc: ad.AnnData,
    batch_key: str,
    variant: str,
    output_dir: Path,
    expected_n_labels: int,
    seed: int,
    harmony_max_iter: int,
    harmony_nclust: int,
    fast_smoke: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    tmp_dir = output_dir / "scaide_input"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_npy = tmp_dir / "input.npy"
    aide_embedding_npy = output_dir / "results" / "embeddings" / "embedding_scaide_aide.npy"
    np.save(input_npy, np.asarray(X_input, dtype=np.float32))

    cmd = [
        str(SCAIDE_PYTHON),
        str(SCAIDE_EMBED_SCRIPT),
        "--input-npy",
        str(input_npy),
        "--output-npy",
        str(aide_embedding_npy),
        "--save-folder",
        str(output_dir / "models" / "aide"),
        "--name",
        f"scaide_{adata_proc.n_obs}_{adata_proc.n_vars}_{variant}",
    ]
    if fast_smoke:
        cmd.append("--fast-smoke")

    log_file = output_dir / "logs" / "scaide_embedding.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    cache_root = output_dir / ".cache"
    env["MPLCONFIGDIR"] = str(cache_root / "mpl")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env["PYTHONUNBUFFERED"] = "1"
    for path in [Path(env["MPLCONFIGDIR"]), Path(env["XDG_CACHE_HOME"])]:
        path.mkdir(parents=True, exist_ok=True)
    with log_file.open("w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        LOGGER.warning("scAIDE/AIDE embedding failed; using PCA fallback. log=%s", log_file)
        from sklearn.decomposition import PCA

        X = np.asarray(X_input, dtype=np.float32)
        n_components = min(50, int(X.shape[0]) - 1, int(X.shape[1]) - 1)
        if n_components < 2:
            embedding = X[:, : max(1, min(int(X.shape[1]), 2))]
        else:
            embedding = PCA(n_components=n_components, random_state=int(seed)).fit_transform(X)
        if variant == "harmony":
            embedding = apply_harmony(
                embedding,
                adata_proc,
                batch_key=batch_key,
                max_iter=int(harmony_max_iter),
                nclust=int(harmony_nclust),
            )
        labels, clusterer_info = _run_rph_or_kmeans(embedding, expected_n_labels, seed)
        return labels, embedding, {
            "embedding_backend": "PCA_fallback_because_AIDE_dependency_missing",
            "fallback_reason": f"AIDE subprocess failed with code {proc.returncode}; log={log_file}",
            **clusterer_info,
        }

    aide_embedding = _ensure_obs_first_embedding(np.load(aide_embedding_npy), adata_proc.n_obs, "AIDE")
    embedding = aide_embedding
    harmony_info: Dict[str, Any] = {}
    if variant == "harmony":
        embedding = apply_harmony(
            aide_embedding,
            adata_proc,
            batch_key=batch_key,
            max_iter=int(harmony_max_iter),
            nclust=int(harmony_nclust),
        )
        harmony_info = {
            "harmony_on": "AIDE_embedding",
            "harmony_max_iter": int(harmony_max_iter),
            "harmony_nclust": int(harmony_nclust),
        }

    labels, clusterer_info = _run_rph_or_kmeans(embedding, expected_n_labels, seed)
    info = {
        "embedding_backend": "AIDE_local_source_with_TF1_compat",
        "fast_smoke": bool(fast_smoke),
        **clusterer_info,
        **harmony_info,
    }
    return labels, embedding, info


def _cellsius_candidate_genes(expr: np.ndarray, main_mask: np.ndarray, min_n_cells: int, min_fc: float, fc_between: float) -> List[int]:
    from sklearn.cluster import KMeans

    candidate: List[int] = []
    cluster_expr = expr[main_mask]
    other_expr = expr[~main_mask]
    if cluster_expr.shape[0] < 2 * int(min_n_cells) or other_expr.shape[0] == 0:
        return candidate
    for gene_idx in range(expr.shape[1]):
        values = cluster_expr[:, gene_idx]
        if float(np.sum(values)) <= 0.0:
            continue
        try:
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(values.reshape(-1, 1))
        except Exception:
            continue
        centers = km.cluster_centers_.ravel()
        high = int(np.argmax(centers))
        low = int(1 - high)
        high_mask = km.labels_ == high
        high_count = int(high_mask.sum())
        low_count = int((km.labels_ == low).sum())
        if high_count <= int(min_n_cells) or low_count <= int(min_n_cells):
            continue
        if high_count >= int(0.5 * len(values)):
            continue
        if float(centers[high] - centers[low]) < float(min_fc):
            continue
        mean_out_nonzero = other_expr[:, gene_idx]
        mean_out_nonzero = mean_out_nonzero[mean_out_nonzero > 0]
        mean_out = float(mean_out_nonzero.mean()) if len(mean_out_nonzero) else 0.0
        if float(values[high_mask].mean() - mean_out) < float(fc_between):
            continue
        candidate.append(gene_idx)
    return candidate


def run_cellsius_python(
    expr_log: np.ndarray,
    base_labels: np.ndarray,
    *,
    min_n_cells: int = 10,
    min_fc: float = 2.0,
    corr_cutoff: Optional[float] = None,
    max_perc_cells: float = 50.0,
    fc_between_cutoff: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.cluster import KMeans
    import networkx as nx

    rng = np.random.default_rng(int(seed))
    expr = np.nan_to_num(np.asarray(expr_log, dtype=np.float32), nan=0.0)
    labels = np.asarray(base_labels, dtype=object).astype(str)
    final = labels.copy()
    n_subclusters = 0
    n_candidate_genes = 0

    for main_cluster in sorted(np.unique(labels).astype(str)):
        main_mask = labels == main_cluster
        if int(main_mask.sum()) < int(2 * min_n_cells):
            continue
        genes = _cellsius_candidate_genes(
            expr,
            main_mask,
            min_n_cells=int(min_n_cells),
            min_fc=float(min_fc),
            fc_between=float(fc_between_cutoff),
        )
        n_candidate_genes += len(genes)
        if len(genes) < 2:
            continue
        mat = expr[np.ix_(main_mask, genes)]
        corr = np.corrcoef(mat.T)
        corr = np.nan_to_num(corr, nan=0.0)
        threshold = float(corr_cutoff) if corr_cutoff is not None else min(max(float(np.quantile(corr[corr < 0.999], 0.95)) if np.any(corr < 0.999) else 0.35, 0.35), 0.5)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(genes)))
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                if corr[i, j] >= threshold:
                    graph.add_edge(i, j)
        components = [sorted(comp) for comp in nx.connected_components(graph) if len(comp) > 1]
        if not components:
            continue
        main_indices = np.where(main_mask)[0]
        for comp_idx, comp in enumerate(components, start=1):
            scores = mat[:, comp].mean(axis=1)
            if np.unique(scores).size < 2:
                continue
            km = KMeans(n_clusters=2, n_init=5, random_state=int(rng.integers(0, 2**31 - 1))).fit(scores.reshape(-1, 1))
            centers = km.cluster_centers_.ravel()
            high = int(np.argmax(centers))
            sub_mask_local = km.labels_ == high
            pct = float(sub_mask_local.mean() * 100.0)
            if pct <= 0.0 or pct > float(max_perc_cells):
                continue
            n_subclusters += 1
            final[main_indices[sub_mask_local]] = f"{main_cluster}_{comp_idx}_1"

    info = {
        "implementation": "python_cellsius_port_connected_components_mcl_fallback",
        "min_n_cells": int(min_n_cells),
        "min_fc": float(min_fc),
        "corr_cutoff": corr_cutoff,
        "max_perc_cells": float(max_perc_cells),
        "fc_between_cutoff": float(fc_between_cutoff),
        "n_candidate_gene_hits": int(n_candidate_genes),
        "n_subclusters_added": int(n_subclusters),
    }
    return final.astype(str), info


def _write_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(path, index=False)


def _write_gene_by_cell_csv(
    path: Path,
    X: Any,
    cell_names: Sequence[str],
    gene_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = _to_dense(X)
    if matrix.shape != (len(cell_names), len(gene_names)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match cells={len(cell_names)} genes={len(gene_names)}"
        )
    pd.DataFrame(
        np.asarray(matrix).T,
        index=pd.Index([str(x) for x in gene_names], name="gene_id"),
        columns=[str(x) for x in cell_names],
    ).to_csv(path)


def _r_env(output_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    path_parts = [str(R_COMPILER_WRAPPERS), str(R_ENV / "bin"), env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(part for part in path_parts if part)
    env["MPLCONFIGDIR"] = str(output_dir / ".cache" / "mpl")
    env["XDG_CACHE_HOME"] = str(output_dir / ".cache" / "xdg")
    env["R_DEFAULT_PACKAGES"] = "datasets,utils,grDevices,graphics,stats,methods"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _r_available(script_path: Path) -> bool:
    return R_SCRIPT.exists() and script_path.exists()


def _read_r_label_output(path: Path, cell_names: Sequence[str]) -> np.ndarray:
    labels = pd.read_csv(path)
    if not {"cell_id", "predicted_label"}.issubset(labels.columns):
        raise ValueError(f"R labels file {path} does not contain cell_id/predicted_label")
    labels = labels.set_index(labels["cell_id"].astype(str))
    ordered = labels.reindex([str(x) for x in cell_names])
    if ordered["predicted_label"].isna().any():
        missing = ordered.index[ordered["predicted_label"].isna()].tolist()[:5]
        raise ValueError(f"R labels file {path} is missing cells: {missing}")
    return ordered["predicted_label"].astype(str).to_numpy()


def run_giniclust_r(
    X_counts: Any,
    cell_names: Sequence[str],
    gene_names: Sequence[str],
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw_dir = output_dir / "raw_giniclust_r"
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_csv = raw_dir / "counts_genes_by_cells.csv"
    labels_csv = raw_dir / "labels.csv"
    log_file = output_dir / "logs" / "giniclust_r.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    X_gene_filtered, gene_names_filtered, filter_info = _giniclust_gene_filter(X_counts, gene_names)
    _write_gene_by_cell_csv(input_csv, X_gene_filtered, cell_names, gene_names_filtered)
    cmd = [
        str(R_SCRIPT),
        str(GINICLUST_R_SCRIPT),
        "--input-csv",
        str(input_csv),
        "--output-dir",
        str(raw_dir),
        "--source-root",
        str(GINICLUST_ROOT),
        "--data-type",
        "RNA-seq",
    ]
    with log_file.open("w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=_r_env(output_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"GiniClust R failed with code {proc.returncode}; log={log_file}")

    labels = _read_r_label_output(labels_csv, cell_names)
    info: Dict[str, Any] = {
        "implementation": "original_GiniClust_R_core_on_non_HVG_counts_shared_cells",
        "r_script": str(GINICLUST_R_SCRIPT),
        "r_log": str(log_file),
        **filter_info,
    }
    info_csv = raw_dir / "run_info.csv"
    if info_csv.exists():
        for row in pd.read_csv(info_csv).to_dict(orient="records"):
            info[str(row.get("metric"))] = row.get("value")
    return labels, info


def run_cellsius_r(
    expr_log: np.ndarray,
    base_labels: np.ndarray,
    cell_names: Sequence[str],
    gene_names: Sequence[str],
    output_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw_dir = output_dir / "raw_cellsius_r"
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_csv = raw_dir / "log_expr_genes_by_cells.csv"
    base_csv = raw_dir / "base_labels.csv"
    labels_csv = raw_dir / "labels.csv"
    log_file = output_dir / "logs" / "cellsius_r.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    _write_gene_by_cell_csv(input_csv, expr_log, cell_names, gene_names)
    pd.DataFrame(
        {"cell_id": [str(x) for x in cell_names], "base_label": np.asarray(base_labels, dtype=object).astype(str)}
    ).to_csv(base_csv, index=False)
    cmd = [
        str(R_SCRIPT),
        str(CELLSIUS_R_SCRIPT),
        "--input-csv",
        str(input_csv),
        "--base-labels-csv",
        str(base_csv),
        "--output-dir",
        str(raw_dir),
        "--mcl-path",
        str(MCL_PATH),
    ]
    with log_file.open("w") as handle:
        proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=_r_env(output_dir))
    if proc.returncode != 0:
        raise RuntimeError(f"CellSIUS R failed with code {proc.returncode}; log={log_file}")

    labels = _read_r_label_output(labels_csv, cell_names)
    info: Dict[str, Any] = {
        "implementation": "CellSIUS_R_package_after_shared_preprocessing",
        "r_script": str(CELLSIUS_R_SCRIPT),
        "mcl_path": str(MCL_PATH),
        "r_log": str(log_file),
    }
    info_csv = raw_dir / "run_info.csv"
    if info_csv.exists():
        for row in pd.read_csv(info_csv).to_dict(orient="records"):
            info[str(row.get("metric"))] = row.get("value")
    return labels, info


def _write_per_cell_labels(
    path: Path,
    adata_proc: ad.AnnData,
    label_key: str,
    batch_key: str,
    labels_pred: np.ndarray,
) -> None:
    true_labels = adata_proc.obs[label_key].astype(str).to_numpy()
    aligned = np.asarray(align_labels(true_labels, labels_pred), dtype=object).astype(str)
    batch_values = adata_proc.obs[batch_key].astype(str).to_numpy() if batch_key in adata_proc.obs.columns else np.repeat("", adata_proc.n_obs)
    pd.DataFrame(
        {
            "cell_index": np.arange(adata_proc.n_obs, dtype=int),
            "cell_id": adata_proc.obs_names.astype(str),
            "batch": batch_values,
            "true_label": true_labels,
            "predicted_label": np.asarray(labels_pred, dtype=object).astype(str),
            "aligned_predicted_label": aligned,
        }
    ).to_csv(path, index=False)


def save_successful_run(
    output_dir: Path,
    *,
    method: str,
    variant: str,
    spec: DatasetSpec,
    adata_proc: ad.AnnData,
    preprocess_stats: Mapping[str, Any],
    labels_pred: np.ndarray,
    embedding: np.ndarray,
    runtime: float,
    method_info: Mapping[str, Any],
    scib_n_jobs: int,
) -> None:
    method_slug = METHOD_SLUGS[method]
    display_name = DISPLAY_NAMES[(method, variant)]
    results_dir = output_dir / "results"
    labels_dir = results_dir / "labels"
    embeddings_dir = results_dir / "embeddings"
    config_dir = output_dir / "config"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    labels_true = adata_proc.obs[spec.label_key].astype(str).to_numpy()
    metrics = compute_metrics(
        labels_true=labels_true,
        labels_pred=np.asarray(labels_pred, dtype=object).astype(str),
        embeddings=np.asarray(embedding, dtype=np.float32),
        adata=adata_proc,
        batch_key=spec.batch_key,
        label_key=spec.label_key,
        compute_scib=True,
        scib_n_jobs=int(scib_n_jobs),
    )
    # compute_metrics keeps going if the scIB bundle fails; force missing scIB
    # columns to be present in the exported CSV for easy auditing.
    for column in SCIB_COLUMNS:
        metrics.setdefault(column, float("nan"))

    label_path = labels_dir / f"labels_{method_slug}_{variant}_run0.csv"
    embedding_path = embeddings_dir / f"embedding_{method_slug}_{variant}.npy"
    _write_per_cell_labels(label_path, adata_proc, spec.label_key, spec.batch_key, labels_pred)
    np.save(embedding_path, np.asarray(embedding, dtype=np.float32))

    result_row: Dict[str, Any] = {
        "algorithm": method_slug,
        "method_display_name": display_name,
        "method": method,
        "variant": variant,
        "run_id": 0,
        "runtime": float(runtime),
        "runtime_total": float(runtime),
        "implementation_status": "ok",
    }
    for key, value in metrics.items():
        if isinstance(value, dict):
            result_row[key] = json.dumps(_safe_json(value), sort_keys=True)
        else:
            result_row[key] = value
    _write_csv_row(results_dir / "analysis_results.csv", result_row)
    _write_csv_row(results_dir / "results.csv", result_row)

    config_payload = {
        "data": {"file": str(spec.path), "dataset_key": spec.dataset_key},
        "preprocessing": dict(PREPROCESSING),
        "context": {
            "label_key": spec.label_key,
            "batch_key": spec.batch_key,
            "n_labels_expected": int(spec.n_labels_expected),
            "preprocess_stats": dict(preprocess_stats),
        },
        "method_params": {
            "method": method,
            "variant": variant,
            "display_name": display_name,
            "method_info": dict(method_info),
        },
        "output": {"directory": str(output_dir)},
    }
    _write_json(config_dir / "config_used.json", config_payload)

    results_payload = {
        "results": [
            {
                "algorithm_name": method_slug,
                "method_display_name": display_name,
                "run_id": 0,
                "runtime": float(runtime),
                "metrics": metrics,
                "params": config_payload["method_params"],
                "embeddings_shape": list(np.asarray(embedding).shape),
                "labels_csv": str(label_path),
                "embedding_npy": str(embedding_path),
            }
        ]
    }
    _write_json(results_dir / "results.json", results_payload)
    _write_json(
        results_dir / "summary.json",
        {
            "data_file": str(spec.path),
            "dataset_key": spec.dataset_key,
            "method": method,
            "variant": variant,
            "method_display_name": display_name,
            "runtime_seconds": float(runtime),
            "best_metrics": metrics,
            "labels_csv": str(label_path),
            "embedding_npy": str(embedding_path),
        },
    )
    _write_json(output_dir / "run_status.json", {"status": "ok", "method": method, "variant": variant})


def save_failed_run(output_dir: Path, method: str, variant: str, spec: DatasetSpec, exc: BaseException) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_status.json",
        {
            "status": "failed",
            "method": method,
            "variant": variant,
            "dataset_key": spec.dataset_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def run_one_method_variant(
    *,
    method: str,
    variant: str,
    spec: DatasetSpec,
    output_root: Path,
    adata_proc: ad.AnnData,
    preprocess_stats: Mapping[str, Any],
    pca_embedding: np.ndarray,
    harmony_embedding: np.ndarray,
    cache: Dict[str, Tuple[np.ndarray, Dict[str, Any], float]],
    args: argparse.Namespace,
) -> None:
    output_dir = output_root / variant / spec.dataset_key / METHOD_SLUGS[method]
    analysis_csv = output_dir / "results" / "analysis_results.csv"
    if analysis_csv.exists() and not args.overwrite:
        LOGGER.info("[skip] %s %s %s", spec.dataset_key, method, variant)
        return

    LOGGER.info("[run] dataset=%s method=%s variant=%s", spec.dataset_key, method, variant)
    start = time.time()
    method_info: Dict[str, Any] = {}
    labels_pred: np.ndarray
    embedding: np.ndarray = harmony_embedding if variant == "harmony" else pca_embedding

    X_scaled = _to_float32(adata_proc.X)
    X_counts_non_hvg, gene_names_non_hvg = _raw_counts_for_processed_cells(adata_proc)
    X_log = _to_float32(adata_proc.layers["pre_scale_log1p_hvg"])
    cell_names = adata_proc.obs_names.astype(str).tolist()
    gene_names = adata_proc.var_names.astype(str).tolist()
    labels_true = adata_proc.obs[spec.label_key].astype(str).to_numpy()

    if method == "scCAD":
        cache_key = f"{spec.dataset_key}:scCAD:standard_labels"
        if cache_key not in cache:
            t0 = time.time()
            labels = run_sccad(X_scaled, cell_names, gene_names, output_dir, int(args.seed))
            cache[cache_key] = (labels, {"labels_source": "scCAD_on_shared_scaled_HVG_matrix"}, time.time() - t0)
        labels_pred, info, label_runtime = cache[cache_key]
        method_info.update(info)
        if variant == "harmony":
            method_info["harmony_variant_note"] = "scCAD labels are produced on the shared matrix; Harmony is used for the evaluation embedding."
        runtime = float(time.time() - start + max(0.0, label_runtime if analysis_csv.exists() else 0.0))

    elif method == "GiniClust":
        cache_key = f"{spec.dataset_key}:GiniClust:standard_labels"
        if cache_key not in cache:
            t0 = time.time()
            dense_entries = int(adata_proc.n_obs) * int(len(gene_names_non_hvg))
            if (
                _r_available(GINICLUST_R_SCRIPT)
                and int(adata_proc.n_obs) <= int(args.giniclust_max_dbscan_cells)
                and dense_entries <= int(args.giniclust_max_r_dense_entries)
            ):
                try:
                    labels, info = run_giniclust_r(X_counts_non_hvg, cell_names, gene_names_non_hvg, output_dir)
                except Exception as exc:
                    LOGGER.warning("GiniClust R failed; using Python fallback: %s", exc)
                    labels, info = run_giniclust_python(
                        X_counts_non_hvg,
                        spec.n_labels_expected,
                        gene_names=gene_names_non_hvg,
                        max_dbscan_cells=int(args.giniclust_max_dbscan_cells),
                        seed=int(args.seed),
                    )
                    info["fallback_reason"] = f"GiniClust R failed: {type(exc).__name__}: {exc}"
            else:
                labels, info = run_giniclust_python(
                    X_counts_non_hvg,
                    spec.n_labels_expected,
                    gene_names=gene_names_non_hvg,
                    max_dbscan_cells=int(args.giniclust_max_dbscan_cells),
                    seed=int(args.seed),
                )
                reasons = []
                if not _r_available(GINICLUST_R_SCRIPT):
                    reasons.append(f"Rscript or GiniClust R script unavailable: {R_SCRIPT}")
                if int(adata_proc.n_obs) > int(args.giniclust_max_dbscan_cells):
                    reasons.append(
                        f"n_obs={adata_proc.n_obs} exceeds giniclust_max_dbscan_cells={int(args.giniclust_max_dbscan_cells)}"
                    )
                if dense_entries > int(args.giniclust_max_r_dense_entries):
                    reasons.append(
                        f"n_obs*n_genes={dense_entries} exceeds giniclust_max_r_dense_entries={int(args.giniclust_max_r_dense_entries)}"
                    )
                info.update(
                    {
                        "large_dataset_fallback_reason": (
                            "original GiniClust R materializes dense gene-by-cell and cell-cell matrices; "
                            + "; ".join(reasons)
                        )
                    }
                )
            cache[cache_key] = (labels, info, time.time() - t0)
        labels_pred, info, label_runtime = cache[cache_key]
        method_info.update(info)
        if variant == "harmony":
            method_info["harmony_variant_note"] = "GiniClust labels are produced on non-HVG raw counts for the shared cell set; Harmony is used for the evaluation embedding."
        runtime = float(time.time() - start)

    elif method == "CellSIUS":
        base_embedding = harmony_embedding if variant == "harmony" else pca_embedding
        base_labels, base_resolution, ranked = cluster_embedding_ranked(
            base_embedding,
            labels_true=labels_true,
            expected_n_classes=spec.n_labels_expected,
            seed=int(args.seed),
            resolutions=[float(x) for x in str(args.resolutions).split(",") if str(x).strip()],
        )
        if _r_available(CELLSIUS_R_SCRIPT):
            try:
                labels_pred, method_info = run_cellsius_r(
                    X_log,
                    base_labels,
                    cell_names,
                    gene_names,
                    output_dir,
                )
            except Exception as exc:
                LOGGER.warning("CellSIUS R failed; using Python fallback: %s", exc)
                labels_pred, method_info = run_cellsius_python(X_log, base_labels, seed=int(args.seed))
                method_info["fallback_reason"] = f"CellSIUS R failed: {type(exc).__name__}: {exc}"
        else:
            labels_pred, method_info = run_cellsius_python(X_log, base_labels, seed=int(args.seed))
            method_info["fallback_reason"] = f"Rscript or CellSIUS R script unavailable: {R_SCRIPT}"
        method_info.update(
            {
                "base_clusterer": "Leiden_on_Harmony" if variant == "harmony" else "Leiden_on_PCA",
                "base_resolution": float(base_resolution),
                "resolution_selection": selection_metadata(spec.n_labels_expected),
                "resolution_table": ranked.to_dict(orient="records"),
            }
        )
        runtime = float(time.time() - start)

    elif method == "scAIDE":
        scaide_input = X_scaled
        labels_pred, embedding, method_info = run_scaide(
            scaide_input,
            adata_proc,
            batch_key=spec.batch_key,
            variant=variant,
            output_dir=output_dir,
            expected_n_labels=spec.n_labels_expected,
            seed=int(args.seed),
            harmony_max_iter=int(args.harmony_max_iter),
            harmony_nclust=int(args.harmony_nclust),
            fast_smoke=bool(args.fast_scaide_smoke),
        )
        runtime = float(time.time() - start)

    else:  # pragma: no cover
        raise ValueError(f"Unsupported method: {method}")

    save_successful_run(
        output_dir,
        method=method,
        variant=variant,
        spec=spec,
        adata_proc=adata_proc,
        preprocess_stats=preprocess_stats,
        labels_pred=np.asarray(labels_pred, dtype=object).astype(str),
        embedding=np.asarray(embedding, dtype=np.float32),
        runtime=runtime,
        method_info=method_info,
        scib_n_jobs=int(args.scib_n_jobs),
    )


def _output_root(raw: str) -> Path:
    if str(raw).strip():
        return Path(raw).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_PRESENTATION_ROOT / f"external_rare_methods_stable_generalist_{stamp}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--variants", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--resolutions", default="0.2,0.4,0.6,0.8,1.0,1.2,1.4")
    parser.add_argument("--harmony-max-iter", type=int, default=10)
    parser.add_argument("--harmony-nclust", type=int, default=50)
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--giniclust-max-dbscan-cells", type=int, default=12000)
    parser.add_argument("--giniclust-max-r-dense-entries", type=int, default=50000000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fast-scaide-smoke", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(bool(args.verbose))
    _seed_everything(int(args.seed))

    output_root = _output_root(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    methods = _select_methods(str(args.methods))
    variants = _select_variants(str(args.variants))
    specs = _select_datasets(_read_manifest(Path(args.manifest)), str(args.datasets))

    _write_json(
        output_root / "run_metadata.json",
        {
            "timestamp": datetime.now().isoformat(),
            "manifest": str(Path(args.manifest).resolve()),
            "output_root": str(output_root),
            "methods": methods,
            "variants": variants,
            "seed": int(args.seed),
            "preprocessing": dict(PREPROCESSING),
            "n_datasets": len(specs),
            "scib_n_jobs": int(args.scib_n_jobs),
            "giniclust_max_dbscan_cells": int(args.giniclust_max_dbscan_cells),
            "giniclust_max_r_dense_entries": int(args.giniclust_max_r_dense_entries),
        },
    )

    run_rows: List[Dict[str, Any]] = []
    overall_status = "ok"
    for spec in specs:
        dataset_start = time.time()
        LOGGER.info("=== dataset=%s ===", spec.dataset_key)
        try:
            adata = ad.read_h5ad(spec.path)
            adata_proc, preprocess_stats = preprocess_shared(adata, PREPROCESSING)
            if spec.label_key not in adata_proc.obs.columns:
                raise KeyError(f"Missing label key '{spec.label_key}' after preprocessing")
            if spec.batch_key not in adata_proc.obs.columns:
                raise KeyError(f"Missing batch key '{spec.batch_key}' after preprocessing")
            pca_embedding = compute_pca_embedding(adata_proc, n_pcs=int(args.n_pcs), seed=int(args.seed))
            harmony_embedding = apply_harmony(
                pca_embedding,
                adata_proc,
                batch_key=spec.batch_key,
                max_iter=int(args.harmony_max_iter),
                nclust=int(args.harmony_nclust),
            )
            cache: Dict[str, Tuple[np.ndarray, Dict[str, Any], float]] = {}
            for variant in variants:
                for method in methods:
                    try:
                        run_one_method_variant(
                            method=method,
                            variant=variant,
                            spec=spec,
                            output_root=output_root,
                            adata_proc=adata_proc,
                            preprocess_stats=preprocess_stats,
                            pca_embedding=pca_embedding,
                            harmony_embedding=harmony_embedding,
                            cache=cache,
                            args=args,
                        )
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        status = "failed"
                        error = f"{type(exc).__name__}: {exc}"
                        overall_status = "failed"
                        save_failed_run(output_root / variant / spec.dataset_key / METHOD_SLUGS[method], method, variant, spec, exc)
                        LOGGER.exception("Failed dataset=%s method=%s variant=%s", spec.dataset_key, method, variant)
                        if not bool(args.continue_on_error):
                            raise
                    run_rows.append(
                        {
                            "dataset_key": spec.dataset_key,
                            "method": method,
                            "variant": variant,
                            "status": status,
                            "error": error,
                            "output_dir": str(output_root / variant / spec.dataset_key / METHOD_SLUGS[method]),
                        }
                    )
            elapsed = time.time() - dataset_start
            LOGGER.info("Finished dataset=%s in %.1fs", spec.dataset_key, elapsed)
        except Exception as exc:
            overall_status = "failed"
            LOGGER.exception("Dataset-level failure for %s", spec.dataset_key)
            run_rows.append(
                {
                    "dataset_key": spec.dataset_key,
                    "method": "",
                    "variant": "",
                    "status": "dataset_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_dir": "",
                }
            )
            if not bool(args.continue_on_error):
                break

        pd.DataFrame(run_rows).to_csv(output_root / "run_status.csv", index=False)

    _write_json(
        output_root / "completion_summary.json",
        {
            "status": overall_status,
            "n_rows": len(run_rows),
            "n_ok": int(sum(1 for row in run_rows if row.get("status") == "ok")),
            "n_failed": int(sum(1 for row in run_rows if row.get("status") not in {"ok"})),
            "output_root": str(output_root),
        },
    )
    print(str(output_root))
    return 0 if overall_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
