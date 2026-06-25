"""Preprocessing utilities for the scRAW pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import logging
import numpy as np
from scipy import sparse


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingState:
    """Frozen preprocessing parameters learned on a training split."""

    var_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    params: Dict[str, Any]
    looks_processed: bool
    scale_max_value: float


def _as_dict(params: Any) -> Dict[str, Any]:
    """Convert a dataclass or mapping-like config into a plain dictionary."""
    if is_dataclass(params):
        return asdict(params)
    return dict(params)


def _has_negative_values(matrix: Any) -> bool:
    """Check whether a dense or sparse expression matrix contains negative values."""
    if sparse.issparse(matrix):
        data = np.asarray(matrix.data)
        return bool(data.size and np.nanmin(data) < 0)

    arr = np.asarray(matrix)
    return bool(arr.size and np.nanmin(arr) < 0)


def _to_dense_float32(matrix: Any) -> np.ndarray:
    """Convert one matrix to a dense float32 NumPy array."""
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _copy_original_layer(adata: Any) -> None:
    """Keep the original matrix in a layer when it is not already present."""
    if "original_X" not in adata.layers:
        X_orig = adata.X
        if hasattr(X_orig, "copy"):
            X_orig = X_orig.copy()
        adata.layers["original_X"] = X_orig


def _filter_cells_for_qc(adata: Any, cfg: Dict[str, Any]) -> Any:
    """Apply deterministic cell-level QC filters."""
    import scanpy as sc

    min_genes = int(cfg.get("min_genes_per_cell", 0) or 0)
    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)

    max_genes = cfg.get("max_genes_per_cell")
    if max_genes is not None:
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        if "n_genes_by_counts" in adata.obs.columns:
            adata = adata[adata.obs["n_genes_by_counts"] <= int(max_genes)].copy()

    return adata


def _filter_genes_for_fit(adata: Any, cfg: Dict[str, Any]) -> Any:
    """Apply train-only gene filtering before learning the frozen gene space."""
    import scanpy as sc

    min_cells = int(cfg.get("min_cells_per_gene", 0) or 0)
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    return adata


def _normalize_and_select_hvgs_for_fit(adata: Any, cfg: Dict[str, Any]) -> tuple[Any, bool]:
    """Normalize raw counts and learn the train HVG subset."""
    import scanpy as sc

    looks_processed = _has_negative_values(adata.X)
    if looks_processed:
        logger.warning(
            "Input matrix contains negative values; assuming it is already preprocessed."
        )
        return adata, True

    sc.pp.normalize_total(adata, target_sum=float(cfg.get("target_sum", 20000.0)))
    sc.pp.log1p(adata)

    n_top_genes = int(cfg.get("n_top_genes", 2000) or 0)
    if n_top_genes > 0 and adata.n_vars > 1:
        sc.pp.highly_variable_genes(
            adata,
            flavor=str(cfg.get("hvg_flavor", "seurat")),
            n_top_genes=min(n_top_genes, int(adata.n_vars)),
            subset=True,
        )
    return adata, False


def _scale_with_stats(
    adata: Any,
    mean: np.ndarray,
    std: np.ndarray,
    scale_max_value: float,
) -> Any:
    """Apply a frozen standardization vector and clipping."""
    X = _to_dense_float32(adata.X)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if X.shape[1] != mean.shape[0] or X.shape[1] != std.shape[0]:
        raise ValueError("Frozen preprocessing statistics do not match the matrix width.")
    X = (X - mean) / std
    np.clip(X, -float(scale_max_value), float(scale_max_value), out=X)
    adata.X = X.astype(np.float32, copy=False)
    return adata


def _align_vars_to_state(adata: Any, var_names: Sequence[str]) -> Any:
    """Align an AnnData object to the frozen training gene order."""
    import anndata as ad
    import pandas as pd

    target_names = [str(name) for name in var_names]
    current_names = [str(name) for name in adata.var_names]
    if current_names == target_names:
        return adata

    current_index = {name: idx for idx, name in enumerate(current_names)}
    present_target_indices: list[int] = []
    present_source_indices: list[int] = []
    for target_idx, gene_name in enumerate(target_names):
        source_idx = current_index.get(gene_name)
        if source_idx is not None:
            present_target_indices.append(target_idx)
            present_source_indices.append(source_idx)

    if not present_target_indices:
        raise ValueError("No genes from the frozen preprocessing state were found in input data.")

    X_source = _to_dense_float32(adata.X)
    X_aligned = np.zeros((adata.n_obs, len(target_names)), dtype=np.float32)
    X_aligned[:, present_target_indices] = X_source[:, present_source_indices]
    var = pd.DataFrame(index=pd.Index(target_names, name=adata.var_names.name))
    return ad.AnnData(X=X_aligned, obs=adata.obs.copy(), var=var)


def fit_preprocess_adata(adata: Any, params: Any) -> tuple[Any, PreprocessingState]:
    """Preprocess a training AnnData object and return reusable frozen state."""
    cfg = _as_dict(params)
    adata = adata.copy()
    _copy_original_layer(adata)
    adata = _filter_cells_for_qc(adata, cfg)
    adata = _filter_genes_for_fit(adata, cfg)

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Preprocessing removed all cells or genes.")

    adata, looks_processed = _normalize_and_select_hvgs_for_fit(adata, cfg)

    X = _to_dense_float32(adata.X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    scale_max_value = float(cfg.get("scale_max_value", 10.0))
    adata = _scale_with_stats(adata, mean=mean, std=std, scale_max_value=scale_max_value)
    state = PreprocessingState(
        var_names=[str(name) for name in adata.var_names],
        mean=np.asarray(mean, dtype=np.float32),
        std=np.asarray(std, dtype=np.float32),
        params=cfg,
        looks_processed=bool(looks_processed),
        scale_max_value=scale_max_value,
    )
    return adata, state


def transform_adata_with_state(
    adata: Any,
    state: PreprocessingState,
    *,
    filter_cells: bool = True,
) -> Any:
    """Apply frozen training preprocessing to held-out or future observations."""
    import scanpy as sc

    cfg = dict(state.params)
    adata = adata.copy()
    _copy_original_layer(adata)
    if filter_cells:
        adata = _filter_cells_for_qc(adata, cfg)

    if adata.n_obs == 0:
        raise ValueError("Preprocessing removed all held-out cells.")

    if not bool(state.looks_processed):
        sc.pp.normalize_total(adata, target_sum=float(cfg.get("target_sum", 20000.0)))
        sc.pp.log1p(adata)

    adata = _align_vars_to_state(adata, state.var_names)
    adata = _scale_with_stats(
        adata,
        mean=state.mean,
        std=state.std,
        scale_max_value=float(state.scale_max_value),
    )
    return adata


def preprocess_adata(adata: Any, params: Any) -> Any:
    """Apply the default scRAW preprocessing path on a raw-count AnnData object."""
    adata_proc, _ = fit_preprocess_adata(adata, params)
    return adata_proc


def save_preprocessing_state(state: PreprocessingState, path: str | Path) -> None:
    """Persist frozen preprocessing state to an NPZ file."""
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        var_names=np.asarray(state.var_names, dtype=str),
        mean=np.asarray(state.mean, dtype=np.float32),
        std=np.asarray(state.std, dtype=np.float32),
        params_json=json.dumps(state.params, sort_keys=True),
        looks_processed=np.asarray([bool(state.looks_processed)], dtype=bool),
        scale_max_value=np.asarray([float(state.scale_max_value)], dtype=np.float32),
    )


def load_preprocessing_state(path: str | Path) -> PreprocessingState:
    """Load frozen preprocessing state from an NPZ file."""
    import json

    with np.load(Path(path), allow_pickle=False) as payload:
        return PreprocessingState(
            var_names=[str(name) for name in payload["var_names"].tolist()],
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            params=json.loads(str(payload["params_json"].item())),
            looks_processed=bool(payload["looks_processed"][0]),
            scale_max_value=float(payload["scale_max_value"][0]),
        )
