#!/usr/bin/env python3
"""Preprocessing utilities for standalone scRAW runs."""

from __future__ import annotations

from typing import Any, Dict
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_total_fallback(adata: Any, target_sum: float) -> None:
    """Normalize per cell without relying on Scanpy's CSR helper.

    Some Scanpy versions crash on CSR inputs because `_normalize_csr` returns an
    uninitialized `counts_per_cols` when `exclude_highly_expressed=False`.
    """
    X = adata.X
    target_sum = float(target_sum)

    if hasattr(X, "tocsr"):
        X_csr = X.tocsr(copy=True)
        counts_per_cell = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32, copy=False)
        scale = np.zeros_like(counts_per_cell, dtype=np.float32)
        nonzero = counts_per_cell > 0.0
        scale[nonzero] = target_sum / counts_per_cell[nonzero]
        adata.X = X_csr.multiply(scale[:, None]).tocsr().astype(np.float32)
        return

    X_dense = np.asarray(X, dtype=np.float32)
    counts_per_cell = np.sum(X_dense, axis=1, dtype=np.float32)
    nonzero = counts_per_cell > 0.0
    if np.any(nonzero):
        X_dense[nonzero] *= (target_sum / counts_per_cell[nonzero])[:, None]
    adata.X = np.asarray(X_dense, dtype=np.float32)


def preprocess_adata(adata: Any, params: Dict[str, Any]) -> Any:
    """Apply the standard preprocessing pipeline expected by scRAW.

    Expected params keys:
      - min_genes_per_cell
      - max_genes_per_cell (optional)
      - min_cells_per_gene
      - target_sum
      - n_top_genes
      - hvg_flavor
      - scale_max_value
    """
    import scanpy as sc

    adata = adata.copy()

    # Preserve original matrix for NB reconstruction path in scRAW.
    if "original_X" not in adata.layers:
        X_orig = adata.X
        if hasattr(X_orig, "copy"):
            X_orig = X_orig.copy()
        adata.layers["original_X"] = X_orig

    min_genes = int(params.get("min_genes_per_cell", 0) or 0)
    max_genes = params.get("max_genes_per_cell")
    min_cells = int(params.get("min_cells_per_gene", 0) or 0)

    if min_genes > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes)

    if max_genes is not None:
        max_genes = int(max_genes)
        sc.pp.calculate_qc_metrics(adata, inplace=True)
        col = "n_genes_by_counts"
        if col in adata.obs.columns:
            adata = adata[adata.obs[col] <= max_genes].copy()

    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Preprocessing removed all cells/genes. Check filtering thresholds.")

    # If matrix already looks transformed (contains negatives), avoid re-running count-based steps.
    X_probe = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    looks_processed = bool(X_probe.size and np.nanmin(X_probe) < 0)

    if looks_processed:
        logger.warning(
            "Input matrix contains negative values; assuming preprocessed input and skipping normalize/log1p/HVG."
        )
    else:
        target_sum = float(params.get("target_sum", 20000.0))
        try:
            sc.pp.normalize_total(adata, target_sum=target_sum)
        except Exception as exc:
            if hasattr(adata.X, "tocsr"):
                logger.warning(
                    "scanpy.normalize_total failed on sparse input (%s); using local fallback normalization.",
                    exc,
                )
                _normalize_total_fallback(adata, target_sum=target_sum)
            else:
                raise
        sc.pp.log1p(adata)

        n_top_genes = int(params.get("n_top_genes", 2000) or 0)
        if n_top_genes > 0 and adata.n_vars > 1:
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    flavor=str(params.get("hvg_flavor", "seurat")),
                    n_top_genes=min(n_top_genes, int(adata.n_vars)),
                    subset=True,
                )
            except Exception as exc:
                logger.warning("HVG selection failed (%s). Continuing without HVG subset.", exc)

    # Final explicit z-score normalization with clipping for numerical stability.
    scale_max = float(params.get("scale_max_value", 10.0))
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0.0] = 1.0
    X = (X - mean) / std
    X = np.clip(X, -scale_max, scale_max)
    adata.X = np.asarray(X, dtype=np.float32)

    return adata
