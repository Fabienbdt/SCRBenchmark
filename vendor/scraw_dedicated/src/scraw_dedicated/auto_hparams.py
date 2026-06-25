#!/usr/bin/env python3
"""Dataset-adaptive hyperparameter selection for standalone scRAW.

The goal is not to learn a black-box meta-model from only a handful of
benchmark datasets. Instead, this module encodes a small, interpretable set of
profiles inspired by:

- the stable top family from the large stage1 search
- the best-per-dataset winners used in KEY_04 / KEY_05
- simple dataset fingerprints observable before training

This lets standalone scRAW stay autonomous while remaining explainable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import math
import re

import numpy as np


AUTO_HPARAMS_VERSION = "2026-04-11-stage1-profiles-v2"

AUTO_DANN_KEYS: Set[str] = {
    "use_batch_conditioning",
    "batch_correction_key",
    "adversarial_batch_weight",
    "adversarial_lambda",
    "adversarial_start_epoch",
    "adversarial_ramp_epochs",
    "mmd_batch_weight",
}

AUTO_CONTROLLED_KEYS: Set[str] = {
    "hidden_layers",
    "z_dim",
    "dropout",
    "epochs",
    "warmup_epochs",
    "lr",
    "batch_size",
    "reconstruction_distribution",
    "nb_input_transform",
    "nb_theta",
    "masking_rate",
    "masked_recon_weight",
    "masking_apply_weighted",
    "weight_exponent",
    "cluster_density_alpha",
    "dynamic_weight_momentum",
    "dynamic_weight_update_interval",
    "weight_fusion_mode",
    "min_cell_weight",
    "max_cell_weight",
    "rare_triplet_weight",
    "rare_triplet_start_epoch",
    "rare_triplet_margin",
    "rare_triplet_min_weight",
    "max_triplet_anchors_per_batch",
    "pseudo_label_method",
    "hdbscan_min_cluster_size",
    "hdbscan_min_samples",
    "hdbscan_cluster_selection_method",
    "hdbscan_reassign_noise",
    "clustering_method",
    "use_batch_conditioning",
    "adversarial_batch_weight",
    "adversarial_lambda",
    "adversarial_start_epoch",
    "adversarial_ramp_epochs",
    "mmd_batch_weight",
    "batch_correction_key",
}


PROFILE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "stable_general_mse": {
        "description": (
            "Stable backbone from the best stage1 family (trial_0060 / 0029 / 0074), "
            "meant to generalize well across datasets."
        ),
        "inspired_by_trials": ["trial_0060", "trial_0029", "trial_0074", "trial_0057"],
        "params": {
            "hidden_layers": "512,256",
            "z_dim": 224,
            "dropout": 0.25,
            "epochs": 210,
            "warmup_epochs": 74,
            "lr": 0.0011,
            "batch_size": 192,
            "reconstruction_distribution": "mse",
            "nb_input_transform": "log1p",
            "nb_theta": 12.0,
            "masking_rate": 0.125,
            "masked_recon_weight": 0.8,
            "masking_apply_weighted": True,
            "weight_exponent": 0.1,
            "cluster_density_alpha": 0.54,
            "dynamic_weight_momentum": 0.80,
            "dynamic_weight_update_interval": 15,
            "weight_fusion_mode": "additive",
            "min_cell_weight": 0.40,
            "max_cell_weight": 8.0,
            "rare_triplet_weight": 0.19,
            "rare_triplet_start_epoch": 60,
            "rare_triplet_margin": 0.4,
            "rare_triplet_min_weight": 1.0,
            "max_triplet_anchors_per_batch": 128,
            "pseudo_label_method": "leiden",
            "hdbscan_min_cluster_size": 8,
            "hdbscan_min_samples": 8,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_reassign_noise": True,
            "clustering_method": "leiden",
            "use_batch_conditioning": True,
            "batch_correction_key": "auto",
            "adversarial_batch_weight": 0.08,
            "adversarial_lambda": 0.5,
            "adversarial_start_epoch": 40,
            "adversarial_ramp_epochs": 55,
            "mmd_batch_weight": 0.0,
        },
    },
    "droplet_batch_heavy": {
        "description": (
            "Droplet profile with strong batch structure, kept close to the stable "
            "general family so rare and ultra-rare classes remain protected."
        ),
        "inspired_by_trials": ["trial_0022", "trial_0047"],
        "params": {
            "hidden_layers": "512,256",
            "z_dim": 224,
            "dropout": 0.25,
            "epochs": 210,
            "warmup_epochs": 60,
            "lr": 0.0012,
            "batch_size": 192,
            "reconstruction_distribution": "mse",
            "nb_input_transform": "log1p",
            "nb_theta": 10.0,
            "masking_rate": 0.15,
            "masked_recon_weight": 0.8,
            "masking_apply_weighted": True,
            "weight_exponent": 0.1,
            "cluster_density_alpha": 0.50,
            "dynamic_weight_momentum": 0.82,
            "dynamic_weight_update_interval": 15,
            "weight_fusion_mode": "additive",
            "min_cell_weight": 0.40,
            "max_cell_weight": 8.0,
            "rare_triplet_weight": 0.19,
            "rare_triplet_start_epoch": 60,
            "rare_triplet_margin": 0.4,
            "rare_triplet_min_weight": 1.8,
            "max_triplet_anchors_per_batch": 128,
            "pseudo_label_method": "leiden",
            "hdbscan_min_cluster_size": 8,
            "hdbscan_min_samples": 8,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_reassign_noise": True,
            "clustering_method": "leiden",
            "use_batch_conditioning": True,
            "batch_correction_key": "auto",
            "adversarial_batch_weight": 0.05,
            "adversarial_lambda": 0.85,
            "adversarial_start_epoch": 50,
            "adversarial_ramp_epochs": 55,
            "mmd_batch_weight": 0.0,
        },
    },
    "ultrarare_rich_droplet": {
        "description": (
            "Droplet profile for large heterogeneous datasets with several ultra-rare "
            "states, inspired by Macaque-like winners."
        ),
        "inspired_by_trials": ["trial_0073", "trial_0062"],
        "params": {
            "hidden_layers": "512,256,128",
            "z_dim": 256,
            "dropout": 0.25,
            "epochs": 210,
            "warmup_epochs": 36,
            "lr": 0.0016,
            "batch_size": 192,
            "reconstruction_distribution": "mse",
            "nb_input_transform": "log1p",
            "nb_theta": 6.5,
            "masking_rate": 0.15,
            "masked_recon_weight": 0.8,
            "masking_apply_weighted": True,
            "weight_exponent": 0.15,
            "cluster_density_alpha": 0.32,
            "dynamic_weight_momentum": 0.74,
            "dynamic_weight_update_interval": 10,
            "weight_fusion_mode": "multiplicative",
            "min_cell_weight": 0.42,
            "max_cell_weight": 8.0,
            "rare_triplet_weight": 0.15,
            "rare_triplet_start_epoch": 60,
            "rare_triplet_margin": 0.4,
            "rare_triplet_min_weight": 1.8,
            "max_triplet_anchors_per_batch": 128,
            "pseudo_label_method": "leiden",
            "hdbscan_min_cluster_size": 8,
            "hdbscan_min_samples": 6,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_reassign_noise": True,
            "clustering_method": "leiden",
            "use_batch_conditioning": True,
            "batch_correction_key": "auto",
            "adversarial_batch_weight": 0.08,
            "adversarial_lambda": 1.75,
            "adversarial_start_epoch": 55,
            "adversarial_ramp_epochs": 60,
            "mmd_batch_weight": 0.0,
        },
    },
    "plate_small_rare_nb": {
        "description": (
            "Small plate-like profile that stays close to the stable default, with "
            "only mild rare-aware adjustments for lower-cell-count datasets."
        ),
        "inspired_by_trials": ["trial_0009", "trial_0011", "trial_0001"],
        "params": {
            "hidden_layers": "512,256",
            "z_dim": 224,
            "dropout": 0.25,
            "epochs": 180,
            "warmup_epochs": 60,
            "lr": 0.00105,
            "batch_size": 256,
            "reconstruction_distribution": "mse",
            "nb_input_transform": "log1p",
            "nb_theta": 12.0,
            "masking_rate": 0.15,
            "masked_recon_weight": 0.8,
            "masking_apply_weighted": True,
            "weight_exponent": 0.12,
            "cluster_density_alpha": 0.48,
            "dynamic_weight_momentum": 0.80,
            "dynamic_weight_update_interval": 15,
            "weight_fusion_mode": "additive",
            "min_cell_weight": 0.42,
            "max_cell_weight": 8.0,
            "rare_triplet_weight": 0.18,
            "rare_triplet_start_epoch": 60,
            "rare_triplet_margin": 0.4,
            "rare_triplet_min_weight": 1.2,
            "max_triplet_anchors_per_batch": 128,
            "pseudo_label_method": "leiden",
            "hdbscan_min_cluster_size": 8,
            "hdbscan_min_samples": 8,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_reassign_noise": True,
            "clustering_method": "leiden",
            "use_batch_conditioning": True,
            "batch_correction_key": "auto",
            "adversarial_batch_weight": 0.04,
            "adversarial_lambda": 0.75,
            "adversarial_start_epoch": 45,
            "adversarial_ramp_epochs": 55,
            "mmd_batch_weight": 0.0,
        },
    },
}


def _detect_batch_key(obs_columns: Sequence[str], preferred: Optional[str] = None) -> Optional[str]:
    if preferred and preferred in obs_columns:
        return preferred
    for c in ["batch", "Batch", "study", "dataset", "donor", "sample", "patient", "tech", "condition", "stim"]:
        if c in obs_columns:
            return c
    return None


def _normalize_token_string(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower())
    return re.sub(r"\s+", " ", text).strip()


def _dominant_technology_family(texts: Sequence[str]) -> Tuple[str, List[str]]:
    tokens = " ".join(_normalize_token_string(t) for t in texts if str(t).strip())
    evidence: List[str] = []
    if any(t in tokens for t in ["smart seq", "smartseq", "c1", "mars seq", "marsseq", "fluidigm"]):
        evidence.append("Detected plate/microfluidic technology tokens.")
        return "plate_like", evidence
    if any(t in tokens for t in ["10x", "chromium", "droplet", "drop seq", "dropseq"]):
        evidence.append("Detected droplet technology tokens.")
        return "droplet_like", evidence
    if "indrop" in tokens:
        evidence.append("Detected inDrop technology token.")
        return "droplet_like", evidence
    if any(t in tokens for t in ["umi", "scrna", "single cell"]):
        evidence.append("Detected generic UMI/single-cell technology tokens.")
        return "umi_like", evidence
    return "unknown", evidence


def _counts_and_genes_per_cell(X: Any) -> Tuple[np.ndarray, np.ndarray, float]:
    if hasattr(X, "tocsr"):
        X_csr = X.tocsr()
        counts = np.asarray(X_csr.sum(axis=1)).ravel().astype(np.float32, copy=False)
        genes = np.diff(X_csr.indptr).astype(np.int32, copy=False)
        total_nnz = float(X_csr.nnz)
        return counts, genes, total_nnz
    X_arr = np.asarray(X)
    if X_arr.ndim != 2:
        raise ValueError("Expected a 2D matrix for dataset fingerprinting.")
    X_arr = np.asarray(X_arr, dtype=np.float32)
    counts = np.sum(np.maximum(X_arr, 0.0), axis=1, dtype=np.float32)
    genes = np.count_nonzero(X_arr > 0.0, axis=1).astype(np.int32, copy=False)
    total_nnz = float(np.count_nonzero(X_arr > 0.0))
    return counts, genes, total_nnz


def _estimate_structure_proxy(
    adata: Any,
    *,
    max_cells: int = 3000,
    random_state: int = 0,
) -> Dict[str, Any]:
    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"sklearn import failed: {exc}",
            "estimated_n_clusters": None,
            "estimated_rare_clusters_lt5pct": None,
            "estimated_ultrarare_clusters_lt1pct": None,
            "smallest_cluster_fraction": None,
        }

    n_obs = int(getattr(adata, "n_obs", 0) or 0)
    if n_obs < 50:
        return {
            "status": "too_small",
            "reason": "Dataset too small for a robust structure sketch.",
            "estimated_n_clusters": None,
            "estimated_rare_clusters_lt5pct": None,
            "estimated_ultrarare_clusters_lt1pct": None,
            "smallest_cluster_fraction": None,
        }

    rng = np.random.default_rng(int(random_state))
    if n_obs > int(max_cells):
        idx = np.sort(rng.choice(n_obs, size=int(max_cells), replace=False))
        X = adata.X[idx]
    else:
        idx = None
        X = adata.X

    try:
        if hasattr(X, "tocsr"):
            X_work = X.tocsr(copy=True).astype(np.float32)
            X_work.data = np.log1p(np.clip(X_work.data, 0.0, None))
            n_components = int(max(2, min(20, X_work.shape[0] - 1, X_work.shape[1] - 1)))
            emb = TruncatedSVD(n_components=n_components, random_state=int(random_state)).fit_transform(X_work)
        else:
            X_work = np.asarray(X, dtype=np.float32)
            X_work = np.log1p(np.clip(X_work, 0.0, None))
            n_components = int(max(2, min(20, X_work.shape[0] - 1, X_work.shape[1] - 1)))
            emb = PCA(n_components=n_components, random_state=int(random_state)).fit_transform(X_work)
            emb = StandardScaler(with_mean=True, with_std=True).fit_transform(emb)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"Embedding sketch failed: {exc}",
            "estimated_n_clusters": None,
            "estimated_rare_clusters_lt5pct": None,
            "estimated_ultrarare_clusters_lt1pct": None,
            "smallest_cluster_fraction": None,
        }

    n_sample = int(emb.shape[0])
    if n_sample < 50:
        return {
            "status": "too_small",
            "reason": "Sketch sample too small after subsampling.",
            "estimated_n_clusters": None,
            "estimated_rare_clusters_lt5pct": None,
            "estimated_ultrarare_clusters_lt1pct": None,
            "smallest_cluster_fraction": None,
        }

    k = int(min(24, max(6, round(math.sqrt(n_sample) / 4.0))))
    try:
        labels = MiniBatchKMeans(
            n_clusters=k,
            random_state=int(random_state),
            batch_size=min(2048, n_sample),
            n_init=5,
        ).fit_predict(emb)
    except Exception as exc:
        return {
            "status": "failed",
            "reason": f"KMeans sketch failed: {exc}",
            "estimated_n_clusters": None,
            "estimated_rare_clusters_lt5pct": None,
            "estimated_ultrarare_clusters_lt1pct": None,
            "smallest_cluster_fraction": None,
        }

    counts = np.bincount(np.asarray(labels, dtype=np.int32), minlength=k).astype(np.int32)
    fracs = counts / float(max(1, counts.sum()))
    rare = int(np.sum(fracs < 0.05))
    ultrarare = int(np.sum(fracs < 0.01))
    smallest = float(np.min(fracs)) if len(fracs) else None
    return {
        "status": "ok",
        "reason": None,
        "sampled_n_cells": int(n_sample),
        "subsampled": bool(idx is not None),
        "estimated_n_clusters": int(np.sum(counts > 0)),
        "estimated_rare_clusters_lt5pct": rare,
        "estimated_ultrarare_clusters_lt1pct": ultrarare,
        "smallest_cluster_fraction": smallest,
    }


def _size_regime(n_cells: int) -> str:
    if int(n_cells) <= 4000:
        return "small"
    if int(n_cells) <= 12000:
        return "medium"
    if int(n_cells) <= 30000:
        return "large"
    return "very_large"


def _batch_regime(n_batches: int, imbalance_ratio: float) -> str:
    if int(n_batches) <= 1:
        return "no_batch"
    if int(n_batches) >= 6 or float(imbalance_ratio) >= 6.0:
        return "heavy"
    if int(n_batches) >= 3:
        return "intermediate"
    return "light"


def _rarity_regime(proxy: Dict[str, Any]) -> str:
    ultrarare = int(proxy.get("estimated_ultrarare_clusters_lt1pct") or 0)
    rare = int(proxy.get("estimated_rare_clusters_lt5pct") or 0)
    smallest = proxy.get("smallest_cluster_fraction")
    smallest = float(smallest) if smallest is not None else None
    if ultrarare >= 4 or (smallest is not None and smallest < 0.008):
        return "ultrarare_rich"
    if rare >= 4 or (smallest is not None and smallest < 0.025):
        return "rare_rich"
    return "standard"


def _infer_technology_family(
    *,
    adata: Any,
    data_path: Optional[Path],
    n_cells: int,
    median_genes_per_cell: float,
    sparsity: float,
) -> Tuple[str, List[str]]:
    evidence: List[str] = []
    obs = getattr(adata, "obs", None)
    tech_texts: List[str] = []
    if obs is not None:
        for col in ["technology", "tech", "platform", "protocol", "sequencing_technology"]:
            if col in obs.columns:
                try:
                    vals = obs[col].astype(str).tolist()
                except Exception:
                    vals = [str(x) for x in obs[col].tolist()]
                tech_texts.extend(vals[:50])
    if data_path is not None:
        tech_texts.append(data_path.stem)
        tech_texts.append(str(data_path))

    family, family_evidence = _dominant_technology_family(tech_texts)
    evidence.extend(family_evidence)
    if family != "unknown":
        return family, evidence

    if float(median_genes_per_cell) >= 1800 and float(sparsity) <= 0.94:
        evidence.append("High genes-per-cell and lower sparsity suggest a plate/full-length dataset.")
        return "plate_like", evidence
    if float(median_genes_per_cell) <= 1100 and float(sparsity) >= 0.965:
        evidence.append("Low genes-per-cell and high sparsity suggest a droplet-like dataset.")
        return "droplet_like", evidence
    if int(n_cells) <= 4000 and float(sparsity) <= 0.85 and float(median_genes_per_cell) >= 700:
        evidence.append("Small low-sparsity dataset with moderate genes-per-cell suggests a plate-like regime.")
        return "plate_like", evidence
    if float(median_genes_per_cell) >= 1400:
        evidence.append("Genes-per-cell profile suggests a fuller-count UMI / plate-like dataset.")
        return "umi_like", evidence
    evidence.append("No explicit technology signal found; defaulting to generic UMI-like assumptions.")
    return "umi_like", evidence


def _resolve_batch_info(adata: Any, preferred_batch_key: Optional[str]) -> Tuple[Optional[str], List[int]]:
    obs = getattr(adata, "obs", None)
    if obs is None:
        return None, []
    obs_columns = list(getattr(obs, "columns", []))
    candidates: List[str] = []
    seen = set()
    for key in [preferred_batch_key, "batch", "Batch", "study", "dataset", "donor", "sample", "patient", "condition", "stim", "tech"]:
        if not key or key not in obs_columns or key in seen:
            continue
        seen.add(key)
        candidates.append(str(key))

    fallback_key: Optional[str] = None
    fallback_counts: List[int] = []
    for key in candidates:
        series = obs[key].astype(str)
        vc = series.value_counts(dropna=False)
        counts = [int(x) for x in vc.tolist()]
        if fallback_key is None:
            fallback_key = key
            fallback_counts = counts
        if len(counts) > 1:
            return key, counts
    return fallback_key, fallback_counts


def compute_dataset_fingerprint(
    adata: Any,
    *,
    data_path: Optional[Path] = None,
    preferred_batch_key: Optional[str] = None,
    sketch_max_cells: int = 3000,
) -> Dict[str, Any]:
    n_cells = int(getattr(adata, "n_obs", 0) or 0)
    n_genes = int(getattr(adata, "n_vars", 0) or 0)
    counts_per_cell, genes_per_cell, total_nnz = _counts_and_genes_per_cell(adata.X)
    total_entries = float(max(1, n_cells * n_genes))
    sparsity = float(max(0.0, min(1.0, 1.0 - (total_nnz / total_entries))))
    median_counts = float(np.median(counts_per_cell)) if len(counts_per_cell) else 0.0
    median_genes = float(np.median(genes_per_cell)) if len(genes_per_cell) else 0.0
    library_cv = float(np.std(counts_per_cell) / max(np.mean(counts_per_cell), 1e-8)) if len(counts_per_cell) else 0.0
    negative_values_present = bool(np.nanmin(counts_per_cell) < 0.0) if len(counts_per_cell) else False

    batch_key, batch_counts = _resolve_batch_info(adata, preferred_batch_key)
    n_batches = int(len(batch_counts)) if batch_counts else 1
    min_batch = min(batch_counts) if batch_counts else n_cells
    max_batch = max(batch_counts) if batch_counts else n_cells
    imbalance_ratio = float(max_batch / max(min_batch, 1))

    technology_family, technology_evidence = _infer_technology_family(
        adata=adata,
        data_path=data_path,
        n_cells=n_cells,
        median_genes_per_cell=median_genes,
        sparsity=sparsity,
    )
    structure_proxy = _estimate_structure_proxy(
        adata,
        max_cells=int(max(200, sketch_max_cells)),
        random_state=0,
    )

    fingerprint = {
        "n_cells": n_cells,
        "n_genes": n_genes,
        "sparsity": sparsity,
        "median_counts_per_cell": median_counts,
        "median_genes_per_cell": median_genes,
        "library_size_cv": library_cv,
        "negative_values_present": negative_values_present,
        "batch_key_detected": batch_key,
        "n_batches": n_batches,
        "batch_counts": batch_counts,
        "batch_imbalance_ratio": imbalance_ratio,
        "size_regime": _size_regime(n_cells),
        "batch_regime": _batch_regime(n_batches, imbalance_ratio),
        "technology_family": technology_family,
        "technology_evidence": technology_evidence,
        "structure_proxy": structure_proxy,
        "rarity_regime": _rarity_regime(structure_proxy),
    }
    return fingerprint


def _choose_profile(fingerprint: Dict[str, Any]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    technology_family = str(fingerprint.get("technology_family", "unknown"))
    size_regime = str(fingerprint.get("size_regime", "medium"))
    batch_regime = str(fingerprint.get("batch_regime", "intermediate"))
    rarity_regime = str(fingerprint.get("rarity_regime", "standard"))
    n_batches = int(fingerprint.get("n_batches", 1) or 1)
    imbalance_ratio = float(fingerprint.get("batch_imbalance_ratio", 1.0) or 1.0)

    if technology_family == "plate_like" and size_regime == "small" and batch_regime == "no_batch":
        reasons.append("Small plate-like dataset with no batch structure.")
        return "plate_small_rare_nb", reasons
    if technology_family == "plate_like" and size_regime == "small" and rarity_regime != "standard":
        reasons.append("Small plate-like dataset with a rare-rich structure sketch.")
        return "plate_small_rare_nb", reasons
    if batch_regime == "heavy" and technology_family in {"droplet_like", "umi_like"} and (
        n_batches >= 12 or imbalance_ratio >= 8.0
    ):
        reasons.append("Extreme batch structure on a droplet/UMI-like dataset.")
        return "droplet_batch_heavy", reasons
    if rarity_regime == "ultrarare_rich" and technology_family in {"droplet_like", "umi_like"}:
        reasons.append("Ultra-rare-rich structure on a droplet/UMI-like dataset.")
        return "ultrarare_rich_droplet", reasons

    reasons.append("Falling back to the stable general profile.")
    return "stable_general_mse", reasons


def _coerce_like(reference: Any, value: Any) -> Any:
    if isinstance(reference, bool):
        return bool(value)
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(round(float(value)))
    if isinstance(reference, float):
        return float(value)
    return value


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= 1e-9
    return a == b


def _set_param(
    params: Dict[str, Any],
    *,
    key: str,
    value: Any,
    locked_keys: Set[str],
    reason: str,
    changes: List[Dict[str, Any]],
    phase: str,
) -> None:
    if key in locked_keys:
        return
    old_value = params.get(key)
    new_value = _coerce_like(old_value, value) if old_value is not None else value
    if _values_equal(old_value, new_value):
        return
    params[key] = new_value
    changes.append(
        {
            "phase": phase,
            "param": key,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
    )


def apply_auto_hparams(
    base_params: Dict[str, Any],
    adata: Any,
    *,
    data_path: Optional[Path] = None,
    preferred_batch_key: Optional[str] = None,
    locked_keys: Optional[Set[str]] = None,
    sketch_max_cells: int = 3000,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return auto-adapted params plus a detailed report."""
    locked = set(locked_keys or set())
    params = dict(base_params)
    fingerprint = compute_dataset_fingerprint(
        adata,
        data_path=data_path,
        preferred_batch_key=preferred_batch_key,
        sketch_max_cells=sketch_max_cells,
    )
    profile_key, profile_reasons = _choose_profile(fingerprint)
    profile = PROFILE_LIBRARY[profile_key]
    changes: List[Dict[str, Any]] = []

    for key, value in profile["params"].items():
        _set_param(
            params,
            key=key,
            value=value,
            locked_keys=locked,
            reason=f"Profile '{profile_key}': {profile['description']}",
            changes=changes,
            phase="profile",
        )

    size_regime = str(fingerprint.get("size_regime", "medium"))
    batch_regime = str(fingerprint.get("batch_regime", "intermediate"))
    technology_family = str(fingerprint.get("technology_family", "unknown"))
    rarity_regime = str(fingerprint.get("rarity_regime", "standard"))
    n_batches = int(fingerprint.get("n_batches", 1) or 1)
    imbalance_ratio = float(fingerprint.get("batch_imbalance_ratio", 1.0) or 1.0)
    median_genes = float(fingerprint.get("median_genes_per_cell", 0.0) or 0.0)
    structure_proxy = dict(fingerprint.get("structure_proxy", {}) or {})
    est_clusters = int(structure_proxy.get("estimated_n_clusters") or 0)
    est_rare = int(structure_proxy.get("estimated_rare_clusters_lt5pct") or 0)
    est_ultrarare = int(structure_proxy.get("estimated_ultrarare_clusters_lt1pct") or 0)

    if size_regime == "small":
        _set_param(
            params,
            key="batch_size",
            value=max(int(params.get("batch_size", 192)), 256),
            locked_keys=locked,
            reason="Small datasets usually tolerate somewhat larger mini-batches, but we keep them moderate.",
            changes=changes,
            phase="size_adjustment",
        )
        _set_param(
            params,
            key="warmup_epochs",
            value=max(int(params.get("warmup_epochs", 45)), 45),
            locked_keys=locked,
            reason="Small datasets tolerate a slightly longer warm-up before triplet pressure ramps in.",
            changes=changes,
            phase="size_adjustment",
        )
    elif size_regime in {"large", "very_large"}:
        _set_param(
            params,
            key="batch_size",
            value=192,
            locked_keys=locked,
            reason="Larger datasets in stage1 were more stable with moderate batch sizes.",
            changes=changes,
            phase="size_adjustment",
        )
        if est_clusters >= 16:
            _set_param(
                params,
                key="z_dim",
                value=max(int(params.get("z_dim", 224)), 256),
                locked_keys=locked,
                reason="High estimated cluster complexity benefits from a larger latent space.",
                changes=changes,
                phase="size_adjustment",
            )

    if batch_regime == "no_batch":
        for key, value in {
            "use_batch_conditioning": False,
            "adversarial_batch_weight": 0.0,
            "mmd_batch_weight": 0.0,
            "batch_correction_key": "auto",
        }.items():
            _set_param(
                params,
                key=key,
                value=value,
                locked_keys=locked,
                reason="No batch detected: turn off explicit batch-adversarial regularization.",
                changes=changes,
                phase="batch_adjustment",
            )
    else:
        _set_param(
            params,
            key="use_batch_conditioning",
            value=True,
            locked_keys=locked,
            reason="Batch structure detected: keep batch conditioning enabled.",
            changes=changes,
            phase="batch_adjustment",
        )
        detected_batch_key = fingerprint.get("batch_key_detected")
        if detected_batch_key:
            _set_param(
                params,
                key="batch_correction_key",
                value=str(detected_batch_key),
                locked_keys=locked,
                reason="Use the detected batch key automatically.",
                changes=changes,
                phase="batch_adjustment",
            )
        if batch_regime == "heavy":
            if n_batches >= 12 or imbalance_ratio >= 8.0:
                _set_param(
                    params,
                    key="adversarial_batch_weight",
                    value=max(float(params.get("adversarial_batch_weight", 0.0)), 0.08),
                    locked_keys=locked,
                    reason="Extreme batch structure: strengthen batch-adversarial pressure.",
                    changes=changes,
                    phase="batch_adjustment",
                )
                _set_param(
                    params,
                    key="adversarial_lambda",
                    value=max(float(params.get("adversarial_lambda", 0.5)), 1.25),
                    locked_keys=locked,
                    reason="Extreme batch structure: use a stronger GRL lambda.",
                    changes=changes,
                    phase="batch_adjustment",
                )
                _set_param(
                    params,
                    key="cluster_density_alpha",
                    value=min(float(params.get("cluster_density_alpha", 0.54)), 0.45),
                    locked_keys=locked,
                    reason="Extreme batch structure: reduce over-reliance on density-only weighting.",
                    changes=changes,
                    phase="batch_adjustment",
                )
            else:
                _set_param(
                    params,
                    key="adversarial_batch_weight",
                    value=max(float(params.get("adversarial_batch_weight", 0.0)), 0.05),
                    locked_keys=locked,
                    reason="Moderately heavy batch structure: keep batch-adversarial pressure conservative.",
                    changes=changes,
                    phase="batch_adjustment",
                )
                _set_param(
                    params,
                    key="adversarial_lambda",
                    value=min(max(float(params.get("adversarial_lambda", 0.5)), 0.75), 1.0),
                    locked_keys=locked,
                    reason="Moderately heavy batch structure: avoid over-correcting rare populations.",
                    changes=changes,
                    phase="batch_adjustment",
                )
                _set_param(
                    params,
                    key="cluster_density_alpha",
                    value=max(float(params.get("cluster_density_alpha", 0.45)), 0.50),
                    locked_keys=locked,
                    reason="Moderately heavy batch structure: retain stronger density support for rare populations.",
                    changes=changes,
                    phase="batch_adjustment",
                )
                _set_param(
                    params,
                    key="mmd_batch_weight",
                    value=0.0,
                    locked_keys=locked,
                    reason="Moderately heavy batch structure: disable extra MMD pressure to protect rare classes.",
                    changes=changes,
                    phase="batch_adjustment",
                )

    if rarity_regime == "ultrarare_rich":
        _set_param(
            params,
            key="masking_rate",
            value=max(float(params.get("masking_rate", 0.1)), 0.15),
            locked_keys=locked,
            reason="Rare-rich winners tended to keep higher masking to protect structured signal.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="rare_triplet_start_epoch",
            value=min(int(params.get("rare_triplet_start_epoch", 60)), 60),
            locked_keys=locked,
            reason="Ultra-rare-rich datasets benefit from earlier rare-aware metric pressure.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="rare_triplet_min_weight",
            value=max(float(params.get("rare_triplet_min_weight", 1.0)), 1.8),
            locked_keys=locked,
            reason="Protect ultra-rare anchors with a higher minimum anchor weight.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="dropout",
            value=min(float(params.get("dropout", 0.25)), 0.25),
            locked_keys=locked,
            reason="Ultra-rare-rich winners often preferred lower dropout than the batch-heavy profile.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="lr",
            value=min(float(params.get("lr", 0.0016)), 0.0016),
            locked_keys=locked,
            reason="Ultra-rare-rich winners tended to avoid the highest learning-rate values.",
            changes=changes,
            phase="rarity_adjustment",
        )
    elif rarity_regime == "rare_rich":
        _set_param(
            params,
            key="masking_rate",
            value=max(float(params.get("masking_rate", 0.1)), 0.15),
            locked_keys=locked,
            reason="Rare-rich datasets tended to benefit from a slightly higher masking rate.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="rare_triplet_start_epoch",
            value=min(int(params.get("rare_triplet_start_epoch", 65)), 60),
            locked_keys=locked,
            reason="Rare-rich datasets usually did not want to delay triplet supervision too much.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="rare_triplet_weight",
            value=max(float(params.get("rare_triplet_weight", 0.16)), 0.18),
            locked_keys=locked,
            reason="Rare-rich datasets still need meaningful triplet pressure to preserve minority classes.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="rare_triplet_min_weight",
            value=max(float(params.get("rare_triplet_min_weight", 1.0)), 1.4),
            locked_keys=locked,
            reason="Rare-rich datasets benefit from a stronger floor on anchor weighting.",
            changes=changes,
            phase="rarity_adjustment",
        )
    else:
        _set_param(
            params,
            key="rare_triplet_weight",
            value=min(float(params.get("rare_triplet_weight", 0.19)), 0.16),
            locked_keys=locked,
            reason="When the sketch does not indicate strong rarity, keep triplet pressure conservative.",
            changes=changes,
            phase="rarity_adjustment",
        )
        _set_param(
            params,
            key="rare_triplet_min_weight",
            value=min(float(params.get("rare_triplet_min_weight", 1.8)), 1.0),
            locked_keys=locked,
            reason="Without ultra-rare evidence, a lower anchor floor is sufficient.",
            changes=changes,
            phase="rarity_adjustment",
        )

    if technology_family == "plate_like":
        enable_plate_nb = batch_regime == "no_batch" and size_regime == "small" and median_genes >= 1200.0
        if enable_plate_nb:
            _set_param(
                params,
                key="reconstruction_distribution",
                value="nb",
                locked_keys=locked,
                reason="Small no-batch plate-like datasets can benefit from NB-style reconstruction.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="nb_input_transform",
                value="pearson_residuals",
                locked_keys=locked,
                reason="Small no-batch plate-like datasets can benefit from Pearson-style residual inputs.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="nb_theta",
                value=max(float(params.get("nb_theta", 12.0)), 16.0),
                locked_keys=locked,
                reason="Small no-batch plate-like datasets typically prefer slightly larger NB theta values.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="lr",
                value=min(float(params.get("lr", 0.0011)), 0.0011),
                locked_keys=locked,
                reason="Plate-like NB runs were usually stable with lower learning rates.",
                changes=changes,
                phase="technology_adjustment",
            )
        else:
            _set_param(
                params,
                key="reconstruction_distribution",
                value="mse",
                locked_keys=locked,
                reason="Plate-like datasets with batches or ambiguous structure stay close to the stable MSE backbone.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="nb_input_transform",
                value="log1p",
                locked_keys=locked,
                reason="Plate-like datasets with batches or ambiguous structure stay close to the stable MSE backbone.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="weight_fusion_mode",
                value="additive",
                locked_keys=locked,
                reason="Conservative plate-like fallback keeps additive fusion.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="nb_theta",
                value=max(float(params.get("nb_theta", 10.0)), 12.0),
                locked_keys=locked,
                reason="Conservative plate-like fallback uses a moderate NB theta.",
                changes=changes,
                phase="technology_adjustment",
            )
            _set_param(
                params,
                key="lr",
                value=min(float(params.get("lr", 0.0012)), 0.0012),
                locked_keys=locked,
                reason="Conservative plate-like fallback lowers the learning rate slightly.",
                changes=changes,
                phase="technology_adjustment",
            )
    elif technology_family == "droplet_like":
        _set_param(
            params,
            key="reconstruction_distribution",
            value="mse",
            locked_keys=locked,
            reason="Droplet-like winners were predominantly MSE + log1p.",
            changes=changes,
            phase="technology_adjustment",
        )
        _set_param(
            params,
            key="nb_input_transform",
            value="log1p",
            locked_keys=locked,
            reason="Droplet-like winners were predominantly MSE + log1p.",
            changes=changes,
            phase="technology_adjustment",
        )
        _set_param(
            params,
            key="weight_fusion_mode",
            value="additive" if rarity_regime != "ultrarare_rich" else params.get("weight_fusion_mode", "additive"),
            locked_keys=locked,
            reason="Most droplet-like stable winners used additive fusion.",
            changes=changes,
            phase="technology_adjustment",
        )
        _set_param(
            params,
            key="nb_theta",
            value=min(float(params.get("nb_theta", 12.0)), 10.0),
            locked_keys=locked,
            reason="Droplet-like winners tended to use lower NB theta than plate-like ones.",
            changes=changes,
            phase="technology_adjustment",
        )

    if est_clusters >= 18:
        _set_param(
            params,
            key="z_dim",
            value=max(int(params.get("z_dim", 224)), 256),
            locked_keys=locked,
            reason="High estimated cluster complexity suggests a larger latent space.",
            changes=changes,
            phase="complexity_adjustment",
        )
        _set_param(
            params,
            key="dropout",
            value=max(float(params.get("dropout", 0.25)), 0.25),
            locked_keys=locked,
            reason="High cluster complexity benefits from some regularization.",
            changes=changes,
            phase="complexity_adjustment",
        )
    elif est_clusters > 0 and est_clusters <= 8:
        _set_param(
            params,
            key="z_dim",
            value=min(int(params.get("z_dim", 224)), 192),
            locked_keys=locked,
            reason="Simpler cluster structure usually does not require the largest latent size.",
            changes=changes,
            phase="complexity_adjustment",
        )

    report = {
        "version": AUTO_HPARAMS_VERSION,
        "fingerprint": fingerprint,
        "selected_profile": {
            "name": profile_key,
            "description": profile["description"],
            "inspired_by_trials": profile["inspired_by_trials"],
            "selection_reasons": profile_reasons,
        },
        "changes": changes,
        "locked_keys": sorted(locked),
        "summary": {
            "n_changes": int(len(changes)),
            "technology_family": technology_family,
            "batch_regime": batch_regime,
            "size_regime": size_regime,
            "rarity_regime": rarity_regime,
            "estimated_n_clusters": est_clusters if est_clusters > 0 else None,
            "estimated_rare_clusters_lt5pct": est_rare if est_rare > 0 else 0,
            "estimated_ultrarare_clusters_lt1pct": est_ultrarare if est_ultrarare > 0 else 0,
        },
    }
    return params, report
