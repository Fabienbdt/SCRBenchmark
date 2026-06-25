#!/usr/bin/env python3
"""Single-run CLI for the standalone scRAW project."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .auto_hparams import AUTO_DANN_KEYS, apply_auto_hparams
from .defaults import DEFAULT_PRESET_NAME
from .metrics import align_labels, compute_metrics, marker_overlap_annotation, scib_metrics_available
from .presets import PRESETS, get_preset
from .preprocessing import preprocess_adata
from .resolution_selection import (
    LEIDEN_RESOLUTION_GRID,
    compute_candidate_silhouette,
    select_best_leiden_candidate,
)

logger = logging.getLogger("scraw_dedicated")

SCIB_RESULT_KEYS = [
    "Silhouette batch",
    "iLISI",
    "KBET",
    "Graph connectivity",
    "Isolated labels",
    "KMeans NMI",
    "KMeans ARI",
    "Silhouette label",
    "cLISI",
    "PCR comparison",
    "Jaccard index",
    "Batch correction",
    "Inter cell-type conservation",
    "Intra cell-type conservation",
    "scIB-E Total score",
]


def _setup_logging(verbose: bool) -> None:
    """Configure console logging level and format."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _configure_runtime_cache(output_dir: Path) -> None:
    """Configure writable cache directories for numba/matplotlib."""
    cache_root = output_dir / "tmp_cache"
    numba_dir = cache_root / "numba"
    mpl_dir = cache_root / "mpl"
    xdg_dir = cache_root / "xdg_cache"

    for p in (cache_root, numba_dir, mpl_dir, xdg_dir):
        p.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    # Keep runtime behavior deterministic and robust on different backends.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _detect_label_key(obs_columns: Sequence[str]) -> Optional[str]:
    """Detect the most likely biological label column in `adata.obs`."""
    # Keep a stable priority order for common single-cell datasets.
    candidates = [
        "Group",
        "label",
        "cell_type",
        "cluster",
        "Groupe",
        "Y",
        "celltype",
        "CellType",
        "cell_types",
        "labels",
        "clusters",
        "assigned_cluster",
    ]
    for c in candidates:
        if c in obs_columns:
            return c
    return None


def _detect_batch_key(obs_columns: Sequence[str], preferred: Optional[str] = None) -> Optional[str]:
    """Detect batch column in `adata.obs`, optionally honoring a preferred key."""
    if preferred and preferred in obs_columns:
        return preferred
    for c in ["batch", "Batch", "study", "dataset", "donor", "sample", "patient", "tech"]:
        if c in obs_columns:
            return c
    return None


def _detect_batch_key_in_file(data_path: Path, preferred: Optional[str]) -> Optional[str]:
    """Detect batch key from an `.h5ad` file without loading full matrix data."""
    import anndata as ad

    adata = ad.read_h5ad(data_path, backed="r")
    try:
        cols = list(adata.obs.columns)
        return _detect_batch_key(cols, preferred=preferred)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()


def _as_jsonable(value: Any) -> Any:
    """Recursively convert NumPy types/arrays into JSON-serializable values."""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_jsonable(v) for v in value]
    return value


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write one dictionary as pretty JSON."""
    path.write_text(json.dumps(_as_jsonable(payload), indent=2), encoding="utf-8")


def _save_csv(path: Path, row: Dict[str, Any]) -> None:
    """Write one dictionary as a single-row CSV file."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow({k: _as_jsonable(v) for k, v in row.items()})


def _safe_numpy(x: Any) -> np.ndarray:
    """Convert input to a NumPy array."""
    return np.asarray(x)


def _parse_scalar(raw: str) -> Any:
    """Parse CLI scalar values into bool/int/float/None/string."""
    text = raw.strip()
    low = text.lower()

    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None

    if "," in text:
        return text

    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except Exception:
            return text

    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", text):
        try:
            return float(text)
        except Exception:
            return text

    return text


def _parse_kv_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """Parse repeated CLI overrides like `KEY=VALUE` into a Python dict."""
    out: Dict[str, Any] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"Invalid override '{raw}'. Expected KEY=VALUE.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{raw}'. Empty key.")
        # Keep compatibility with namespaced legacy syntax: "scraw:param=value".
        if ":" in key:
            ns, bare_key = key.split(":", 1)
            ns = ns.strip().lower()
            if ns in {"scraw", "algorithm", "algo", "preprocess"}:
                key = bare_key.strip()
                if not key:
                    raise ValueError(f"Invalid override '{raw}'. Empty key after namespace.")
        out[key] = _parse_scalar(value)
    return out


def _build_scraw_params(
    preset_name: str,
    seed: int,
    device: str,
    dann_mode: str,
    batch_key_override: Optional[str],
    capture_snapshots: str,
    snapshot_interval: Optional[int],
    param_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Build final algorithm params from preset + CLI overrides."""
    preset = get_preset(preset_name)
    params = dict(preset.algorithm_params)

    params["random_state"] = int(seed)
    params["seed"] = int(seed)
    params["device"] = str(device)

    if capture_snapshots == "on":
        params["capture_embedding_snapshots"] = True
    elif capture_snapshots == "off":
        params["capture_embedding_snapshots"] = False

    if snapshot_interval is not None:
        params["snapshot_interval_epochs"] = int(snapshot_interval)

    if dann_mode == "off":
        params["use_batch_conditioning"] = False
        params["adversarial_batch_weight"] = 0.0
        params["mmd_batch_weight"] = 0.0
        params["batch_correction_key"] = str(params.get("batch_correction_key", "auto") or "auto")
    elif dann_mode == "on":
        params["use_batch_conditioning"] = True

    if batch_key_override:
        params["batch_correction_key"] = batch_key_override

    params.update(param_overrides)
    return params


def _label_encoding(labels: Sequence[Any]) -> Tuple[np.ndarray, Dict[str, str]]:
    """Encode arbitrary labels to contiguous integers and return reverse map."""
    labels = np.asarray([str(x) for x in labels], dtype=object)
    uniq = sorted(np.unique(labels).tolist())
    to_idx = {lab: i for i, lab in enumerate(uniq)}
    encoded = np.asarray([to_idx[l] for l in labels], dtype=np.int64)
    idx_to_label = {str(i): lab for lab, i in to_idx.items()}
    return encoded, idx_to_label


def _extract_final_cell_weights(snapshots: List[Dict[str, Any]], n_cells: int) -> Optional[np.ndarray]:
    """Extract the most recent valid per-cell fused weight vector from snapshots."""
    for snap in reversed(snapshots):
        w = snap.get("cell_weights")
        if w is None:
            continue
        w = np.asarray(w, dtype=np.float32)
        if len(w) == n_cells:
            return w
    return None


def _extract_final_weight_component(
    snapshots: List[Dict[str, Any]],
    n_cells: int,
    key: str,
) -> Optional[np.ndarray]:
    """Return last valid per-cell weight component from snapshots."""
    for snap in reversed(snapshots):
        vals = snap.get(key)
        if vals is None:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        if len(arr) == n_cells:
            return arr
    return None


def _snapshot_epoch(snap: Dict[str, Any]) -> Optional[int]:
    """Parse snapshot epoch as int."""
    try:
        return int(snap.get("epoch"))
    except Exception:
        return None


def _select_snapshots_for_requested_epochs(
    snapshots: List[Dict[str, Any]],
    warmup_epochs: int,
    step: int = 10,
) -> List[Dict[str, Any]]:
    """Select snapshots: epoch 0 pre-backward, epoch warmup-1, then every `step` epochs."""
    valid = [s for s in snapshots if s.get("embeddings") is not None]
    if not valid:
        return []

    pre0 = next((s for s in valid if str(s.get("snapshot_type", "")) == "pre_backward"), None)
    by_epoch: Dict[int, Dict[str, Any]] = {}
    for s in valid:
        e = _snapshot_epoch(s)
        if e is None:
            continue
        if e not in by_epoch:
            by_epoch[e] = s

    max_epoch = max(by_epoch.keys()) if by_epoch else 0
    anchor = int(max(0, warmup_epochs - 1))
    anchor = min(anchor, max_epoch)
    target_epochs = [anchor]
    e = anchor + int(max(1, step))
    while e <= max_epoch:
        target_epochs.append(e)
        e += int(max(1, step))
    if max_epoch not in target_epochs:
        target_epochs.append(max_epoch)

    out: List[Dict[str, Any]] = []
    if pre0 is not None:
        out.append(pre0)
    for te in target_epochs:
        s = by_epoch.get(int(te))
        if s is not None and s not in out:
            out.append(s)
    if not out:
        out = [by_epoch[e] for e in sorted(by_epoch.keys())]
    return out


def _snapshot_component_vector(snapshot: Dict[str, Any], key: str) -> Optional[np.ndarray]:
    """Read one component vector from a snapshot."""
    vals = snapshot.get(key)
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] == 0:
        return None
    return arr


def _lagged_component_vectors(
    snapshots: List[Dict[str, Any]],
    key: str,
    lag: int,
    phase2_start_epoch: int,
) -> List[Optional[np.ndarray]]:
    """Build per-snapshot lagged vectors (epoch n-`lag`) for epoch n projections."""
    by_epoch: Dict[int, Dict[str, Any]] = {}
    for s in snapshots:
        e = _snapshot_epoch(s)
        if e is None:
            continue
        by_epoch[e] = s

    out: List[Optional[np.ndarray]] = []
    for s in snapshots:
        e = _snapshot_epoch(s)
        if e is None or e < int(phase2_start_epoch):
            out.append(None)
            continue
        prev = by_epoch.get(int(e - lag))
        if prev is None:
            out.append(None)
            continue
        out.append(_snapshot_component_vector(prev, key))
    return out


def _leiden_optimized_for_target_clusters(
    embeddings: np.ndarray,
    seed: int,
    target_clusters: int = 14,
    labels_true: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run Leiden resolution search on 0..1, targeting `target_clusters`, with silhouette tie-breaks."""
    import anndata as ad
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    emb = np.asarray(embeddings, dtype=np.float32)
    n_cells = int(emb.shape[0])
    if n_cells < 3:
        labels = np.zeros(n_cells, dtype=np.int64)
        return labels, {"resolution": 0.0, "n_clusters": 1, "target_clusters": int(target_clusters)}

    adata = ad.AnnData(X=emb)
    n_neighbors = max(2, min(15, n_cells - 1))
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep="X",
        method="gauss",
        transformer="sklearn",
        random_state=int(seed),
    )

    true_arr = None if labels_true is None else np.asarray(labels_true, dtype=object)
    has_truth = true_arr is not None and len(true_arr) == n_cells

    candidates: List[Dict[str, Any]] = []
    for res in LEIDEN_RESOLUTION_GRID:
        try:
            sc.tl.leiden(adata, resolution=float(res), random_state=int(seed), key_added="_leiden_tmp")
        except Exception as exc:
            logger.debug("Skipping Leiden resolution %.2f after failure: %s", float(res), exc)
            continue

        labels = adata.obs["_leiden_tmp"].astype(int).to_numpy(dtype=np.int64, copy=False)
        n_found = int(len(np.unique(labels)))
        silhouette = compute_candidate_silhouette(
            emb,
            labels,
            random_state=int(seed),
        )
        ari = float("nan")
        if has_truth:
            try:
                ari = float(adjusted_rand_score(true_arr, labels))
            except Exception:
                ari = float("nan")
        candidates.append(
            {
                "resolution": float(res),
                "labels": labels.copy(),
                "n_clusters": n_found,
                "diff": abs(n_found - int(target_clusters)),
                "silhouette": silhouette,
                "ari": ari,
            }
        )

    if not candidates:
        labels = np.zeros(n_cells, dtype=np.int64)
        return labels, {"resolution": 0.0, "n_clusters": 1, "target_clusters": int(target_clusters)}

    best = select_best_leiden_candidate(candidates, expected_n_classes=int(target_clusters))
    if best is None:
        labels = np.zeros(n_cells, dtype=np.int64)
        return labels, {"resolution": 0.0, "n_clusters": 1, "target_clusters": int(target_clusters)}

    info = {
        "resolution": float(best["resolution"]),
        "n_clusters": int(best["n_clusters"]),
        "target_clusters": int(target_clusters),
        "selection_metric": "cluster_count_abs_error_then_silhouette",
        "selection_score": float(best["silhouette"]) if np.isfinite(float(best["silhouette"])) else None,
        "silhouette": float(best["silhouette"]) if np.isfinite(float(best["silhouette"])) else None,
        "ARI_proxy": float(best["ari"]) if np.isfinite(float(best["ari"])) else None,
    }
    return np.asarray(best["labels"], dtype=np.int64), info


def _leiden_method_name(target_clusters: int, *, final: bool = False) -> str:
    """Return the exported method name for a Leiden target-cluster run."""
    base = f"leiden_target{int(target_clusters)}"
    return f"{base}_final" if final else base


def _resolve_leiden_target_clusters(
    *,
    true_labels_raw: Optional[np.ndarray],
    effective_params: Optional[Dict[str, Any]],
    scraw_params: Dict[str, Any],
    fallback_labels: Optional[np.ndarray] = None,
    override_target: Optional[int] = None,
) -> int:
    """Resolve the cluster target used for exported Leiden comparison runs."""
    if override_target is not None:
        try:
            value = int(override_target)
        except Exception:
            value = 0
        if value > 1:
            return value

    if true_labels_raw is not None:
        try:
            value = int(len(np.unique(np.asarray(true_labels_raw, dtype=object))))
        except Exception:
            value = 0
        if value > 1:
            return value

    effective = effective_params or {}
    for key in ("n_clusters_effective", "unsupervised_k_selected", "_pseudo_n_clusters"):
        try:
            value = int(effective.get(key, 0) or 0)
        except Exception:
            value = 0
        if value > 1:
            return value

    for key in ("n_clusters", "unsupervised_k_selected", "unsupervised_k_fallback"):
        try:
            value = int(scraw_params.get(key, 0) or 0)
        except Exception:
            value = 0
        if value > 1:
            return value

    if fallback_labels is not None:
        try:
            value = int(len(np.unique(np.asarray(fallback_labels))))
        except Exception:
            value = 0
        if value > 1:
            return value

    return 14


def _metric_row_from_bundle(
    epoch: int,
    method: str,
    metrics: Dict[str, Any],
    n_clusters: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize one metric row for CSV/plots."""
    row: Dict[str, Any] = {
        "epoch": int(epoch),
        "method": str(method),
        "NMI": metrics.get("NMI"),
        "ARI": metrics.get("ARI"),
        "ACC": metrics.get("ACC"),
        "BalancedACC": metrics.get("BalancedACC"),
        "F1_Macro": metrics.get("F1_Macro"),
        "RareACC": metrics.get("RareACC"),
        "UltraRareACC": metrics.get("UltraRareACC"),
        "n_clusters_found": int(n_clusters),
    }
    if extra:
        row.update(extra)
    return row


def _write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dict rows to CSV."""
    if not rows:
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(str(k))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _as_jsonable(v) for k, v in row.items()})


def _is_search_minimal_profile(name: str) -> bool:
    """Return True when the output profile should keep only search-critical artifacts."""
    return str(name).strip().lower() == "search_minimal"


def _hyperparams_declared() -> List[Dict[str, Any]]:
    """Export declared scRAW hyperparameters in a JSON-friendly structure."""
    from .algorithms.scraw_algorithm import ScRAWAlgorithm

    out: List[Dict[str, Any]] = []
    for hp in ScRAWAlgorithm.get_hyperparameters():
        out.append(
            {
                "name": hp.name,
                "display_name": hp.display_name,
                "type": hp.param_type.value,
                "default": hp.default,
                "description": hp.description,
                "min_value": hp.min_value,
                "max_value": hp.max_value,
                "choices": hp.choices,
                "category": hp.category,
                "advanced": hp.advanced,
            }
        )
    return out


def run_once(args: argparse.Namespace) -> int:
    """Run a full scRAW job: preprocess, train, evaluate, and export artifacts."""
    preset = get_preset(args.preset)
    data_path = Path(args.data).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    preprocess_overrides = _parse_kv_overrides(args.preprocess or [])
    param_overrides = _parse_kv_overrides(args.param or [])

    preprocess_cfg = dict(preset.preprocessing)
    preprocess_cfg.update(preprocess_overrides)

    scraw_params = _build_scraw_params(
        preset_name=args.preset,
        seed=args.seed,
        device=args.device,
        dann_mode=args.dann,
        batch_key_override=args.batch_key,
        capture_snapshots=args.capture_snapshots,
        snapshot_interval=args.snapshot_interval,
        param_overrides=param_overrides,
    )
    output_profile = str(getattr(args, "output_profile", "standard") or "standard").lower()
    search_minimal = _is_search_minimal_profile(output_profile)
    metrics_only = bool(getattr(args, "metrics_only", False) or search_minimal)
    umap_only = bool(getattr(args, "umap_only", False)) and not search_minimal
    save_processed_data = (
        str(getattr(args, "save_processed_data", "off")).lower() == "on" and not search_minimal
    )
    skip_evaluation_metrics = bool(
        getattr(args, "skip_evaluation_metrics", False) or umap_only
    )
    compute_scib_mode = str(getattr(args, "compute_scib_metrics", "off")).lower()
    compute_scib_requested = compute_scib_mode != "off"
    if skip_evaluation_metrics:
        compute_scib_requested = False
    scib_n_jobs = max(1, int(getattr(args, "scib_n_jobs", 1) or 1))
    show_context_annotation = (not search_minimal) and (
        str(getattr(args, "umap_context_annotation", "on")).lower() != "off"
    )
    export_comparison_panels = bool(getattr(args, "export_comparison_panels", False)) and not search_minimal
    umap_point_size = max(0.1, float(getattr(args, "umap_point_size", 5) or 5))
    figure_format = str(getattr(args, "figure_format", "png") or "png").strip().lower()
    if figure_format not in {"png", "svg"}:
        raise ValueError(f"Unsupported figure format: {figure_format}")
    umap_rasterized = figure_format != "svg"

    def _figure_output_path(path: Path) -> Path:
        return Path(path).with_suffix(f".{figure_format}")

    def _save_figure(fig: Any, path: Path, *, dpi: int = 150) -> None:
        fig.savefig(_figure_output_path(path), bbox_inches="tight", dpi=dpi)

    if search_minimal:
        # Search runs should avoid any model/checkpoint/UMAP side effects.
        scraw_params["capture_embedding_snapshots"] = False
        scraw_params["save_checkpoint_path"] = ""
        scraw_params["resume_checkpoint_path"] = ""

    if not metrics_only and args.capture_snapshots == "auto":
        # Figures epoch-wise require latent snapshots (enabled by default when plotting).
        scraw_params["capture_embedding_snapshots"] = True
        scraw_params.setdefault("snapshot_interval_epochs", 10)

    output.mkdir(parents=True, exist_ok=True)
    config_dir = output / "config"
    data_dir = output / "data"
    figures_dir = output / "figures"
    fig_umap_dir = figures_dir / "umaps"
    fig_umap_overview_dir = fig_umap_dir / "overview"
    fig_umap_labels_dir = fig_umap_dir / "labels"
    fig_umap_batch_dir = fig_umap_dir / "batch"
    fig_umap_weights_dir = fig_umap_dir / "weights"
    fig_umap_weights_cluster_dir = fig_umap_weights_dir / "cluster_component"
    fig_umap_weights_density_dir = fig_umap_weights_dir / "density_component"
    fig_umap_weights_fused_dir = fig_umap_weights_dir / "fused_weight"
    fig_loss_dir = figures_dir / "loss"
    fig_metrics_dir = figures_dir / "metrics"
    results_dir = output / "results"
    clustering_dir = results_dir / "clustering_final"
    embeddings_dir = results_dir / "embeddings"
    epoch_metrics_dir = results_dir / "epoch_metrics"
    labels_dir = results_dir / "labels"
    loss_dir = results_dir / "loss_history"
    per_cell_dir = results_dir / "per_cell"
    weights_export_dir = results_dir / "weights"

    if (not skip_evaluation_metrics) and compute_scib_mode == "on" and not scib_metrics_available():
        raise RuntimeError(
            "scib_metrics was requested via --compute-scib-metrics on, "
            "but the package is not available in the current environment."
        )

    dirs_to_create = [
        config_dir,
        results_dir,
        clustering_dir,
    ]
    if not search_minimal:
        dirs_to_create.extend(
            [
                figures_dir,
                fig_umap_dir,
                fig_umap_overview_dir,
                fig_umap_labels_dir,
                fig_umap_batch_dir,
                fig_umap_weights_dir,
                fig_umap_weights_cluster_dir,
                fig_umap_weights_density_dir,
                fig_umap_weights_fused_dir,
                fig_loss_dir,
                fig_metrics_dir,
                embeddings_dir,
                epoch_metrics_dir,
                labels_dir,
                loss_dir,
                per_cell_dir,
                weights_export_dir,
            ]
        )
    for p in dirs_to_create:
        p.mkdir(parents=True, exist_ok=True)
    if save_processed_data:
        data_dir.mkdir(parents=True, exist_ok=True)

    _configure_runtime_cache(output)

    import anndata as ad

    from .algorithms.scraw_algorithm import ScRAWAlgorithm
    if not search_minimal:
        import matplotlib.pyplot as plt

        from .visualization import (
            compute_projection_2d,
            compute_projection_2d_per_snapshot,
            export_axes_panels,
            format_cluster_labels_for_display,
            plot_loss_curves,
            plot_loss_curves_timeline,
            plot_marker_overlap_heatmap,
            plot_umap_batch,
            plot_umap_comparison,
            plot_umap_evolution,
            plot_umap_snapshots_categorical_panels,
            plot_umap_snapshots_gradient_panels,
            plot_metric_evolution_curves,
            plot_umap_weighted,
            plot_umap_weighted_gradient,
        )

    logger.info("Loading dataset: %s", data_path)
    adata = ad.read_h5ad(data_path)

    auto_hparams_mode = str(getattr(args, "auto_hparams", "auto") or "auto").lower()
    auto_hparams_requested = auto_hparams_mode == "on" or (
        auto_hparams_mode == "auto" and str(args.preset).strip().lower() == DEFAULT_PRESET_NAME
    )
    auto_hparams_report: Optional[Dict[str, Any]] = None
    if auto_hparams_requested:
        locked_param_keys = set(param_overrides.keys())
        if str(args.dann).lower() != "auto":
            locked_param_keys.update(AUTO_DANN_KEYS)
        if args.batch_key:
            locked_param_keys.add("batch_correction_key")
        scraw_params, auto_hparams_report = apply_auto_hparams(
            scraw_params,
            adata,
            data_path=data_path,
            preferred_batch_key=args.batch_key,
            locked_keys=locked_param_keys,
            sketch_max_cells=int(getattr(args, "auto_hparams_sketch_max_cells", 3000) or 3000),
        )
        _save_json(config_dir / "auto_hparams_report.json", auto_hparams_report)

    # Strict DANN validation with auto-batch-key fallback.
    use_batch = bool(scraw_params.get("use_batch_conditioning", False)) or float(
        scraw_params.get("adversarial_batch_weight", 0.0) or 0.0
    ) > 0.0
    if use_batch:
        bkey = str(scraw_params.get("batch_correction_key", "")).strip() or None
        if bkey is None or bkey == "auto":
            bkey = _detect_batch_key(list(adata.obs.columns), preferred=args.batch_key)
            if bkey is None:
                raise ValueError(
                    "DANN/batch conditioning is enabled but no batch key was found in adata.obs. "
                    "Pass --batch-key or --param batch_correction_key=..."
                )
            scraw_params["batch_correction_key"] = bkey

    logger.info("Applying preprocessing...")
    adata_proc = preprocess_adata(adata, preprocess_cfg)
    if save_processed_data:
        adata_proc.write(data_dir / "processed.h5ad")
    else:
        logger.info("Skipping export of processed.h5ad (save_processed_data=off).")

    label_key = _detect_label_key(list(adata_proc.obs.columns))
    batch_key = _detect_batch_key(
        list(adata_proc.obs.columns),
        preferred=str(scraw_params.get("batch_correction_key", "")).strip() or None,
    )

    true_labels_raw: Optional[np.ndarray] = None
    fit_labels: Optional[np.ndarray] = None
    label_map: Dict[str, str] = {}

    if label_key is not None:
        true_labels_raw = np.asarray(adata_proc.obs[label_key].astype(str).to_numpy(), dtype=object)
        fit_labels, label_map = _label_encoding(true_labels_raw)

    if args.unsupervised:
        fit_labels = None

    algo = ScRAWAlgorithm(params=scraw_params)
    t0 = time.time()
    algo.fit(adata_proc, labels=fit_labels)
    pred_labels = _safe_numpy(algo.predict())
    runtime = float(time.time() - t0)

    embeddings = algo.get_embeddings()
    if embeddings is not None:
        embeddings = _safe_numpy(embeddings)

    loss_history = algo.get_loss_history() or []
    snapshots = algo.get_embedding_snapshots() or []
    effective_params = algo.get_effective_params()
    num_params = algo.get_num_parameters()
    final_clustering_info: Dict[str, Any] = {}
    get_final_clustering_info = getattr(algo, "get_final_clustering_info", None)
    if callable(get_final_clustering_info):
        try:
            final_clustering_info = dict(get_final_clustering_info() or {})
        except Exception as exc:
            logger.warning("Could not retrieve final clustering info: %s", exc)
    pseudo_clustering_info: Dict[str, Any] = {}
    get_pseudo_clustering_info = getattr(algo, "get_pseudo_clustering_info", None)
    if callable(get_pseudo_clustering_info):
        try:
            pseudo_clustering_info = dict(get_pseudo_clustering_info() or {})
        except Exception as exc:
            logger.warning("Could not retrieve pseudo clustering info: %s", exc)
    leiden_target_clusters = _resolve_leiden_target_clusters(
        true_labels_raw=true_labels_raw,
        effective_params=effective_params,
        scraw_params=scraw_params,
        fallback_labels=pred_labels,
        override_target=getattr(args, "leiden_target_clusters", None),
    )
    leiden_epoch_method = _leiden_method_name(leiden_target_clusters, final=False)

    metrics: Dict[str, Any] = {}
    if skip_evaluation_metrics:
        logger.info("Skipping clustering/evaluation metrics as requested.")
    else:
        metrics = compute_metrics(
            true_labels_raw,
            pred_labels,
            embeddings=embeddings,
            adata=adata_proc if compute_scib_requested else None,
            batch_key=batch_key,
            label_key=label_key,
            compute_scib=compute_scib_requested,
            scib_n_jobs=scib_n_jobs,
        )
    leiden_final_labels: Optional[np.ndarray] = None
    leiden_final_info: Dict[str, Any] = {}
    leiden_final_metrics: Dict[str, Any] = {}
    if (not skip_evaluation_metrics) and embeddings is not None and len(embeddings) == len(pred_labels):
        try:
            leiden_final_labels, leiden_final_info = _leiden_optimized_for_target_clusters(
                embeddings=embeddings,
                seed=args.seed,
                target_clusters=leiden_target_clusters,
                labels_true=true_labels_raw,
            )
            leiden_final_metrics = compute_metrics(
                true_labels_raw,
                leiden_final_labels,
                embeddings=embeddings,
            )
        except Exception as exc:
            logger.warning(
                "Final Leiden target=%s export failed: %s",
                leiden_target_clusters,
                exc,
            )
            leiden_final_labels = None
            leiden_final_info = {}
            leiden_final_metrics = {}

    warmup_epochs_eff = int(
        effective_params.get(
            "warmup_epochs",
            scraw_params.get("warmup_epochs", 30),
        )
        or 30
    )
    epochs_eff = int(effective_params.get("epochs", scraw_params.get("epochs", 120)) or 120)
    snapshot_step = int(
        effective_params.get(
            "snapshot_interval_epochs",
            scraw_params.get("snapshot_interval_epochs", 10),
        )
        or 10
    )
    snapshot_step = max(1, snapshot_step)
    final_epoch = max(0, epochs_eff - 1)

    selected_snapshots = _select_snapshots_for_requested_epochs(
        snapshots=snapshots,
        warmup_epochs=warmup_epochs_eff,
        step=snapshot_step,
    )

    epoch_metric_rows: List[Dict[str, Any]] = []
    if (not skip_evaluation_metrics) and selected_snapshots and true_labels_raw is not None:
        for snap in selected_snapshots:
            epoch_idx = _snapshot_epoch(snap)
            emb_snap = snap.get("embeddings")
            if epoch_idx is None or emb_snap is None:
                continue
            emb_arr = np.asarray(emb_snap, dtype=np.float32)
            if emb_arr.ndim != 2 or emb_arr.shape[0] != len(true_labels_raw):
                continue
            try:
                labels_h = algo._hdbscan_clustering(emb_arr)
                m_h = compute_metrics(true_labels_raw, labels_h, embeddings=emb_arr)
                epoch_metric_rows.append(
                    _metric_row_from_bundle(
                        epoch=epoch_idx,
                        method="hdbscan",
                        metrics=m_h,
                        n_clusters=int(len(np.unique(labels_h))),
                    )
                )
            except Exception as exc:
                logger.warning("Epoch metric (HDBSCAN) failed at epoch %s: %s", epoch_idx, exc)

            try:
                labels_l, l_info = _leiden_optimized_for_target_clusters(
                    embeddings=emb_arr,
                    seed=args.seed,
                    target_clusters=leiden_target_clusters,
                    labels_true=true_labels_raw,
                )
                m_l = compute_metrics(true_labels_raw, labels_l, embeddings=emb_arr)
                epoch_metric_rows.append(
                    _metric_row_from_bundle(
                        epoch=epoch_idx,
                        method=leiden_epoch_method,
                        metrics=m_l,
                        n_clusters=int(l_info.get("n_clusters", len(np.unique(labels_l)))),
                        extra={
                            "resolution": l_info.get("resolution"),
                            "target_clusters": int(leiden_target_clusters),
                            "selection_metric": l_info.get("selection_metric"),
                            "selection_score": l_info.get("selection_score"),
                        },
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Epoch metric (Leiden target=%s) failed at epoch %s: %s",
                    leiden_target_clusters,
                    epoch_idx,
                    exc,
                )

    if epoch_metric_rows and not search_minimal:
        _write_rows_csv(epoch_metrics_dir / "metrics_by_epoch.csv", epoch_metric_rows)
        _save_json(
            epoch_metrics_dir / "metrics_by_epoch.json",
            {
                "n_rows": len(epoch_metric_rows),
                "methods": sorted({str(r.get("method")) for r in epoch_metric_rows}),
                "rows": epoch_metric_rows,
            },
        )

    final_clustering_rows: List[Dict[str, Any]] = []
    final_clustering_rows.append(
        _metric_row_from_bundle(
            epoch=final_epoch,
            method="hdbscan_final",
            metrics=metrics,
            n_clusters=int(len(np.unique(pred_labels))),
            extra={
                "resolution": None,
                "selection_metric": final_clustering_info.get("selection_metric"),
                "selection_score": final_clustering_info.get("selection_score"),
                "rare_weighted_silhouette": final_clustering_info.get("rare_weighted_silhouette"),
                "target_clusters": final_clustering_info.get("target_clusters"),
                "target_source": final_clustering_info.get("target_source"),
                "cluster_count_diff": final_clustering_info.get("cluster_diff"),
                "noise_fraction": final_clustering_info.get("noise_fraction"),
                "hdbscan_min_cluster_size": final_clustering_info.get("min_cluster_size"),
                "hdbscan_min_samples": final_clustering_info.get("min_samples"),
                "hdbscan_cluster_selection_method": final_clustering_info.get("cluster_selection_method"),
                "hdbscan_scan_enabled": final_clustering_info.get("scan_enabled"),
            },
        )
    )
    if leiden_final_labels is not None:
        final_clustering_rows.append(
            _metric_row_from_bundle(
                epoch=final_epoch,
                method=_leiden_method_name(leiden_target_clusters, final=True),
                metrics=leiden_final_metrics,
                n_clusters=int(leiden_final_info.get("n_clusters", len(np.unique(leiden_final_labels)))),
                extra={
                    "resolution": leiden_final_info.get("resolution"),
                    "selection_metric": leiden_final_info.get("selection_metric"),
                    "selection_score": leiden_final_info.get("selection_score"),
                    "target_clusters": int(leiden_target_clusters),
                },
            )
        )

    _write_rows_csv(clustering_dir / "final_clustering_comparison.csv", final_clustering_rows)
    if not search_minimal:
        _save_json(
            clustering_dir / "final_clustering_comparison.json",
            {
                "final_epoch": final_epoch,
                "rows": final_clustering_rows,
                "hdbscan_info": final_clustering_info,
                "leiden_info": leiden_final_info,
            },
        )

    # Save final clustering labels even in search_minimal: downstream report
    # diagnostics need the exact selected Leiden/HDBSCAN assignments.
    labels_compare_payload: Dict[str, List[str]] = {
        "hdbscan_final": [str(x) for x in np.asarray(pred_labels)],
    }
    if leiden_final_labels is not None:
        labels_compare_payload[_leiden_method_name(leiden_target_clusters, final=True)] = [
            str(x) for x in np.asarray(leiden_final_labels)
        ]
    if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
        labels_compare_payload["true_label"] = [str(x) for x in np.asarray(true_labels_raw)]
    with (clustering_dir / "final_clustering_labels.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(labels_compare_payload.keys()))
        writer.writeheader()
        rows = zip(*[labels_compare_payload[k] for k in labels_compare_payload.keys()])
        for vals in rows:
            writer.writerow({k: v for k, v in zip(labels_compare_payload.keys(), vals)})

    cluster_count = int(len(np.unique(pred_labels)))
    n_samples = int(len(pred_labels))

    result_row: Dict[str, Any] = {
        "algorithm": "scraw",
        "run_id": 0,
        "runtime": runtime,
        "num_parameters": num_params,
        "NMI": metrics.get("NMI"),
        "ARI": metrics.get("ARI"),
        "ACC": metrics.get("ACC"),
        "UCA": metrics.get("UCA"),
        "F1_Macro": metrics.get("F1_Macro"),
        "BalancedACC": metrics.get("BalancedACC"),
        "RareACC": metrics.get("RareACC"),
        "UltraRareACC": metrics.get("UltraRareACC"),
        "KNN_Purity": metrics.get("KNN_Purity"),
        "ClassWise": metrics.get("ClassWise"),
        "Silhouette": metrics.get("Silhouette"),
        "RareWeightedSilhouette": final_clustering_info.get("rare_weighted_silhouette"),
        "FinalClusteringScore": final_clustering_info.get("selection_score"),
        "n_clusters_found": metrics.get("n_clusters_found", cluster_count),
        "n_samples_evaluated": metrics.get("n_samples_evaluated", n_samples),
        "AutoHparamsMode": auto_hparams_mode,
        "AutoHparamsApplied": bool(auto_hparams_report is not None),
        "AutoHparamsProfile": (
            auto_hparams_report.get("selected_profile", {}).get("name")
            if auto_hparams_report is not None
            else ""
        ),
    }
    for key in SCIB_RESULT_KEYS:
        if key in metrics:
            result_row[key] = metrics.get(key)
    for k, v in sorted(effective_params.items()):
        result_row[f"param_{k}"] = v

    _save_csv(results_dir / "results.csv", result_row)
    _save_csv(results_dir / "analysis_results.csv", result_row)

    cell_ids: Optional[np.ndarray] = None
    if hasattr(adata_proc, "obs_names"):
        try:
            obs_names = np.asarray(adata_proc.obs_names, dtype=object).astype(str)
            if len(obs_names) == len(pred_labels):
                cell_ids = obs_names
        except Exception:
            cell_ids = None

    batch_values: Optional[np.ndarray] = None
    if batch_key is not None and batch_key in adata_proc.obs.columns:
        bvals = adata_proc.obs[batch_key].astype(str).to_numpy()
        if len(bvals) == len(pred_labels):
            batch_values = np.asarray(bvals, dtype=object)

    aligned_pred_labels: Optional[np.ndarray] = None
    if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
        try:
            aligned_pred_labels = np.asarray(align_labels(true_labels_raw, pred_labels), dtype=object)
        except Exception:
            aligned_pred_labels = None

    if not search_minimal:
        label_payload: Dict[str, Any] = {}
        if cell_ids is not None:
            label_payload["cell_index"] = [int(i) for i in range(len(pred_labels))]
            label_payload["cell_id"] = [str(x) for x in cell_ids]
        label_payload["predicted_label"] = [str(x) for x in pred_labels]
        if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
            label_payload["true_label"] = [str(x) for x in true_labels_raw]
            if aligned_pred_labels is not None:
                label_payload["aligned_predicted_label"] = [str(x) for x in aligned_pred_labels]
        if batch_values is not None:
            label_payload["batch"] = [str(x) for x in batch_values]

        with (labels_dir / "labels_scraw_run0.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(label_payload.keys()))
            writer.writeheader()
            rows = zip(*[label_payload[k] for k in label_payload.keys()])
            for vals in rows:
                writer.writerow({k: v for k, v in zip(label_payload.keys(), vals)})

        if label_map:
            _save_json(labels_dir / "label_map.json", label_map)

        _save_json(
            loss_dir / "loss_scraw_run0.json",
            {
                "algorithm": "scraw",
                "run_id": 0,
                "phases": loss_history,
            },
        )

        _save_json(
            results_dir / "results.json",
            {
                "results": [
                    {
                        "algorithm_name": "scraw",
                        "run_id": 0,
                        "runtime": runtime,
                        "metrics": metrics,
                        "final_clustering_info": final_clustering_info,
                        "pseudo_clustering_info": pseudo_clustering_info,
                        "params": effective_params,
                        "num_parameters": num_params,
                        "embeddings_shape": list(embeddings.shape) if embeddings is not None else None,
                        "final_clustering_comparison": final_clustering_rows,
                    }
                ],
                "summary": {
                    "scraw": {
                        "NMI_mean": metrics.get("NMI"),
                        "ARI_mean": metrics.get("ARI"),
                        "ACC_mean": metrics.get("ACC"),
                        "F1_Macro_mean": metrics.get("F1_Macro"),
                        "BalancedACC_mean": metrics.get("BalancedACC"),
                        "RareACC_mean": metrics.get("RareACC"),
                        "UltraRareACC_mean": metrics.get("UltraRareACC"),
                        "Silhouette_mean": metrics.get("Silhouette"),
                        "RareWeightedSilhouette_mean": final_clustering_info.get("rare_weighted_silhouette"),
                        "FinalClusteringScore_mean": final_clustering_info.get("selection_score"),
                        "runtime_mean": runtime,
                        "n_clusters_found_mean": metrics.get("n_clusters_found", cluster_count),
                    }
                },
                "timestamp": datetime.now().isoformat(),
            },
        )

    config_used = {
        "data": {"file": str(data_path)},
        "preprocessing": preprocess_cfg,
        "algorithms": ["scraw"],
        "algorithm_params": {"scraw": scraw_params},
        "algorithm_effective_params_by_algorithm": {"scraw": effective_params},
        "execution": {
            "device": args.device,
            "n_repeats": 1,
            "random_seed": args.seed,
            "output_profile": output_profile,
            "metrics_only": metrics_only,
            "compute_scib_metrics": compute_scib_requested,
            "compute_scib_metrics_mode": compute_scib_mode,
            "scib_n_jobs": scib_n_jobs,
            "skip_evaluation_metrics": skip_evaluation_metrics,
            "umap_only": umap_only,
            "save_processed_data": save_processed_data,
            "auto_hparams_mode": auto_hparams_mode,
            "auto_hparams_applied": bool(auto_hparams_report is not None),
        },
        "output": {"directory": str(output)},
        "context": {
            "preset": preset.name,
            "description": preset.description,
            "timestamp": datetime.now().isoformat(),
            "label_key": label_key,
            "batch_key_detected": batch_key,
            "unsupervised": bool(args.unsupervised),
            "auto_hparams_profile": (
                auto_hparams_report.get("selected_profile", {}).get("name")
                if auto_hparams_report is not None
                else None
            ),
        },
    }
    if auto_hparams_report is not None:
        config_used["auto_hparams"] = auto_hparams_report
    _save_json(config_dir / "config_used.json", config_used)

    _save_json(
        config_dir / "algorithm_hyperparams_used.json",
        {
            "timestamp": datetime.now().isoformat(),
            "algorithms": ["scraw"],
            "declared_hyperparameters": {"scraw": _hyperparams_declared()},
            "defaults_by_algorithm": {
                "scraw": {hp["name"]: hp["default"] for hp in _hyperparams_declared()}
            },
            "overrides_by_algorithm": {"scraw": scraw_params},
            "effective_params_by_algorithm": {"scraw": effective_params},
            "execution": config_used["execution"],
            "context": {
                "mode": output_profile,
                "data_file": str(data_path),
                "output_dir": str(output),
                "status": "completed",
                "preset": preset.name,
            },
            "per_run_params": [
                {
                    "algorithm": "scraw",
                    "run_id": 0,
                    "params_used": effective_params,
                }
            ],
        },
    )

    final_projection_2d: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    cluster_component_weights: Optional[np.ndarray] = None
    density_component_weights: Optional[np.ndarray] = None
    if (not search_minimal) and embeddings is not None and len(embeddings) == len(pred_labels):
        get_final_cell_weights = getattr(algo, "get_final_cell_weights", None)
        if callable(get_final_cell_weights):
            try:
                weights_candidate = get_final_cell_weights()
            except Exception as exc:
                logger.warning("Direct retrieval of final cell weights failed: %s", exc)
            else:
                if weights_candidate is not None:
                    weights_arr = np.asarray(weights_candidate, dtype=np.float32)
                    if len(weights_arr) == len(pred_labels):
                        weights = weights_arr
        if weights is None:
            weights = _extract_final_cell_weights(snapshots, n_cells=len(pred_labels))

        cluster_component_weights = _extract_final_weight_component(
            snapshots,
            n_cells=len(pred_labels),
            key="cluster_component_weights",
        )
        density_component_weights = _extract_final_weight_component(
            snapshots,
            n_cells=len(pred_labels),
            key="density_component_weights",
        )

        final_projection_2d, final_projection_used_fallback = compute_projection_2d(
            np.asarray(embeddings),
            random_state=args.seed,
        )
        if final_projection_used_fallback:
            logger.warning(
                "Final embedding projection fell back to the first two embedding dimensions."
            )
        np.save(embeddings_dir / "embeddings_scraw_run0.npy", np.asarray(embeddings, dtype=np.float32))

    if (not search_minimal) and weights is not None:
        weight_rows: List[Dict[str, Any]] = []
        for idx in range(len(weights)):
            row: Dict[str, Any] = {
                "cell_index": int(idx),
                "scraw_reconstruction_weight": float(weights[idx]),
            }
            if cell_ids is not None:
                row["cell_id"] = str(cell_ids[idx])
            if cluster_component_weights is not None:
                row["cluster_component_weight"] = float(cluster_component_weights[idx])
            if density_component_weights is not None:
                row["density_component_weight"] = float(density_component_weights[idx])
            weight_rows.append(row)
        _write_rows_csv(weights_export_dir / "cell_weights_scraw_run0.csv", weight_rows)

    if (not search_minimal) and final_projection_2d is not None:
        per_cell_rows: List[Dict[str, Any]] = []
        for idx in range(len(pred_labels)):
            row = {
                "cell_index": int(idx),
                "umap_1": float(final_projection_2d[idx, 0]),
                "umap_2": float(final_projection_2d[idx, 1]),
                "predicted_label": str(pred_labels[idx]),
            }
            if cell_ids is not None:
                row["cell_id"] = str(cell_ids[idx])
            if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
                row["true_label"] = str(true_labels_raw[idx])
            if aligned_pred_labels is not None and len(aligned_pred_labels) == len(pred_labels):
                row["aligned_predicted_label"] = str(aligned_pred_labels[idx])
            if batch_values is not None:
                row["batch"] = str(batch_values[idx])
            if weights is not None:
                row["scraw_reconstruction_weight"] = float(weights[idx])
            if cluster_component_weights is not None:
                row["cluster_component_weight"] = float(cluster_component_weights[idx])
            if density_component_weights is not None:
                row["density_component_weight"] = float(density_component_weights[idx])
            per_cell_rows.append(row)
        _write_rows_csv(per_cell_dir / "per_cell_scraw_run0.csv", per_cell_rows)

    if not metrics_only and embeddings is not None and len(embeddings) == len(pred_labels):
        logger.info("Generating figures...")

        reverse_label_map: Optional[Dict[int, str]] = None
        if label_map:
            reverse_label_map = {}
            for k, v in label_map.items():
                try:
                    reverse_label_map[int(k)] = str(v)
                except Exception:
                    continue

        params_info = {
            "normalization": effective_params.get("nb_input_transform", "log1p"),
            "DANN_weight": effective_params.get("adversarial_batch_weight", 0),
            "MMD_weight": effective_params.get("mmd_batch_weight", 0),
            "clustering": effective_params.get("clustering_method", "hdbscan"),
            "HVG_flavor": effective_params.get("internal_hvg_flavor", "seurat"),
            "epochs": effective_params.get("epochs", "?"),
            "z_dim": effective_params.get("z_dim", "?"),
        }
        dataset_info = f"{data_path.stem} | Full data"
        batch_labels: Optional[np.ndarray] = None
        if batch_key is not None and batch_key in adata_proc.obs.columns:
            tmp_batch = adata_proc.obs[batch_key].astype(str).to_numpy()
            if len(tmp_batch) == len(pred_labels):
                batch_labels = np.asarray(tmp_batch, dtype=object)

        if true_labels_raw is not None and len(true_labels_raw) == len(pred_labels):
            fig = plot_umap_comparison(
                embeddings=embeddings,
                true_labels=true_labels_raw,
                predicted_labels=pred_labels,
                algorithm_name="scraw",
                label_names=reverse_label_map,
                show_cluster_centroids=False,
                point_size=umap_point_size,
                params_info=params_info,
                dataset_info=dataset_info,
                projection_2d=final_projection_2d,
                show_context_annotation=show_context_annotation,
                rasterized=umap_rasterized,
            )
            _save_figure(fig, fig_umap_overview_dir / "umap_comparison_scraw.png")
            if export_comparison_panels:
                export_axes_panels(
                    fig=fig,
                    axes=[fig.axes[0], fig.axes[1], fig.axes[2]],
                    output_paths=[
                        _figure_output_path(
                            fig_umap_overview_dir
                            / "umap_comparison_panels"
                            / "ground_truth_cell_types.png"
                        ),
                        _figure_output_path(
                            fig_umap_overview_dir
                            / "umap_comparison_panels"
                            / "predicted_clusters_raw_ids.png"
                        ),
                        _figure_output_path(
                            fig_umap_overview_dir
                            / "umap_comparison_panels"
                            / "predicted_hungarian_aligned.png"
                        ),
                    ],
                )
            plt.close(fig)

            if not umap_only:
                try:
                    overlap_result = marker_overlap_annotation(
                        adata=adata_proc,
                        labels_true=true_labels_raw,
                        labels_pred=pred_labels,
                        n_top_genes=100,
                        method="wilcoxon",
                    )
                    fig_hm = plot_marker_overlap_heatmap(
                        overlap_matrix=overlap_result["overlap_matrix"],
                        algorithm_name="scraw",
                    )
                    if fig_hm is not None:
                        _save_figure(fig_hm, fig_umap_overview_dir / "marker_overlap_heatmap_scraw.png")
                        plt.close(fig_hm)

                    import pandas as pd

                    annot_df = pd.DataFrame(
                        {
                            "true_label": np.asarray(true_labels_raw, dtype=str),
                            "predicted_cluster": np.asarray(pred_labels, dtype=str),
                            "hungarian_annotation": np.asarray(overlap_result["hungarian_labels"], dtype=str),
                            "marker_overlap_annotation": np.asarray(overlap_result["marker_labels"], dtype=str),
                        }
                    )
                    annot_df.to_csv(results_dir / "annotation_comparison_scraw.csv", index=False)
                    overlap_result["overlap_matrix"].to_csv(results_dir / "marker_overlap_matrix_scraw.csv")
                except Exception as exc:
                    logger.warning("Marker-overlap annotation failed for scraw: %s", exc)

        if batch_labels is not None:
            fig_b = plot_umap_batch(
                embeddings=embeddings,
                batch_labels=batch_labels,
                title=f"scraw (Batch: {batch_key})",
                point_size=umap_point_size,
                params_info=params_info,
                dataset_info=dataset_info,
                projection_2d=final_projection_2d,
                show_context_annotation=show_context_annotation,
                rasterized=umap_rasterized,
            )
            _save_figure(fig_b, fig_umap_overview_dir / "umap_batch_scraw.png")
            plt.close(fig_b)

        if weights is not None:
            labels_for_weight_plot = true_labels_raw if true_labels_raw is not None else pred_labels
            fig_w = plot_umap_weighted(
                embeddings=embeddings,
                labels=labels_for_weight_plot,
                cell_weights=weights,
                title="scraw (Cell Weights)",
                label_names=reverse_label_map,
                point_size=umap_point_size,
                params_info=params_info,
                dataset_info=dataset_info,
                projection_2d=final_projection_2d,
                show_context_annotation=show_context_annotation,
                rasterized=umap_rasterized,
            )
            _save_figure(fig_w, fig_umap_weights_fused_dir / "umap_scraw_weighted_alpha.png")
            plt.close(fig_w)

            fig_wg = plot_umap_weighted_gradient(
                embeddings=embeddings,
                cell_weights=weights,
                title="scraw (Cell Weights Gradient)",
                point_size=umap_point_size,
                params_info=params_info,
                dataset_info=dataset_info,
                projection_2d=final_projection_2d,
                show_context_annotation=show_context_annotation,
                rasterized=umap_rasterized,
            )
            _save_figure(fig_wg, fig_umap_weights_fused_dir / "umap_scraw_weighted_gradient_final.png")
            plt.close(fig_wg)

        if selected_snapshots:
            projection_2d_per_snapshot, projection_used_fallback = compute_projection_2d_per_snapshot(
                [np.asarray(s["embeddings"]) for s in selected_snapshots],
                random_state=args.seed,
            )
            if projection_used_fallback:
                logger.warning(
                    "At least one snapshot projection fell back to the first two embedding dimensions."
                )

            labels_for_panels = true_labels_raw if true_labels_raw is not None else pred_labels
            labels_panel_title = (
                "UMAP snapshots (ground-truth labels, per-snapshot fit)"
                if true_labels_raw is not None
                else "UMAP snapshots (final predicted labels, per-snapshot fit)"
            )
            fig_labels_panel = plot_umap_snapshots_categorical_panels(
                embedding_snapshots=selected_snapshots,
                labels=np.asarray(labels_for_panels),
                title=labels_panel_title,
                point_size=umap_point_size,
                random_state=args.seed,
                projection_2d_per_snapshot=projection_2d_per_snapshot,
                projection_mode="per_snapshot",
                params_info=params_info,
                dataset_info=dataset_info,
                rasterized=umap_rasterized,
            )
            if fig_labels_panel is not None:
                _save_figure(fig_labels_panel, fig_umap_labels_dir / "umap_labels_snapshots_panels.png")
                plt.close(fig_labels_panel)

            if batch_labels is not None:
                fig_batch_panel = plot_umap_snapshots_categorical_panels(
                    embedding_snapshots=selected_snapshots,
                    labels=np.asarray(batch_labels),
                    title=f"UMAP snapshots (batch={batch_key}, per-snapshot fit)",
                    point_size=umap_point_size,
                    random_state=args.seed,
                    projection_2d_per_snapshot=projection_2d_per_snapshot,
                    projection_mode="per_snapshot",
                    params_info=params_info,
                    dataset_info=dataset_info,
                    rasterized=umap_rasterized,
                )
                if fig_batch_panel is not None:
                    _save_figure(fig_batch_panel, fig_umap_batch_dir / "umap_batch_snapshots_panels.png")
                    plt.close(fig_batch_panel)

            component_specs = [
                ("cluster_component_weights", "Cluster Component", fig_umap_weights_cluster_dir),
                ("density_component_weights", "Density Component", fig_umap_weights_density_dir),
                ("cell_weights", "Fused Reconstruction Weight (Cluster + Density)", fig_umap_weights_fused_dir),
            ]
            for comp_key, comp_name, comp_dir in component_specs:
                current_vectors = [
                    _snapshot_component_vector(s, comp_key) for s in selected_snapshots
                ]
                lag_vectors = _lagged_component_vectors(
                    snapshots=selected_snapshots,
                    key=comp_key,
                    lag=snapshot_step,
                    phase2_start_epoch=warmup_epochs_eff,
                )
                fig_comp = plot_umap_snapshots_gradient_panels(
                    embedding_snapshots=selected_snapshots,
                    current_weights=current_vectors,
                    lagged_weights=lag_vectors,
                    title=f"UMAP snapshots ({comp_name}, per-snapshot fit)",
                    point_size=umap_point_size,
                    random_state=args.seed,
                    projection_2d_per_snapshot=projection_2d_per_snapshot,
                    projection_mode="per_snapshot",
                    current_row_label="Current epoch n weights",
                    lagged_row_label=f"Lagged epoch n-{snapshot_step} weights on epoch n latent",
                    params_info=params_info,
                    dataset_info=dataset_info,
                    rasterized=umap_rasterized,
                )
                if fig_comp is not None:
                    out_name = comp_key.replace("_weights", "").replace("_", "-")
                    _save_figure(fig_comp, comp_dir / f"umap_gradient_panels_{out_name}.png")
                    plt.close(fig_comp)

            pseudo_cluster_labels_per_snapshot: List[Optional[np.ndarray]] = []
            for snap in selected_snapshots:
                pseudo_labels = snap.get("pseudo_labels")
                if pseudo_labels is None:
                    pseudo_cluster_labels_per_snapshot.append(None)
                    continue
                pseudo_cluster_labels_per_snapshot.append(
                    format_cluster_labels_for_display(np.asarray(pseudo_labels))
                )
            fig_evo = plot_umap_evolution(
                embedding_snapshots=selected_snapshots,
                labels=np.asarray(labels_for_panels),
                algorithm_name="scraw",
                point_size=umap_point_size,
                random_state=args.seed,
                projection_mode="per_snapshot",
                color_mode="pseudo_cluster",
                labels_per_snapshot=pseudo_cluster_labels_per_snapshot,
                projection_2d_per_snapshot=projection_2d_per_snapshot,
                params_info=params_info,
                dataset_info=dataset_info,
                rasterized=umap_rasterized,
            )
            if fig_evo is not None:
                _save_figure(fig_evo, fig_umap_overview_dir / "umap_evolution_scraw_run0.png")
                plt.close(fig_evo)

        if not umap_only:
            fig_loss = plot_loss_curves(loss_history, algorithm_name="scraw")
            if fig_loss is not None:
                _save_figure(fig_loss, fig_loss_dir / "loss_curves_by_phase_scraw_run0.png")
                plt.close(fig_loss)

            fig_loss_timeline = plot_loss_curves_timeline(loss_history, algorithm_name="scraw")
            if fig_loss_timeline is not None:
                _save_figure(
                    fig_loss_timeline,
                    fig_loss_dir / "loss_curves_timeline_scraw_run0.png",
                )
                plt.close(fig_loss_timeline)

            fig_metrics = plot_metric_evolution_curves(
                epoch_metric_rows,
                title=f"Epoch-wise Metrics (HDBSCAN vs {leiden_epoch_method})",
            )
            if fig_metrics is not None:
                _save_figure(
                    fig_metrics,
                    fig_metrics_dir / "metrics_evolution_by_epoch_scraw_run0.png",
                )
                plt.close(fig_metrics)

    logger.info("Run completed. Output: %s", output)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for standalone scRAW execution."""
    p = argparse.ArgumentParser(description="Standalone strict scRAW runner")
    p.add_argument(
        "--preset",
        default=DEFAULT_PRESET_NAME,
        choices=sorted(PRESETS.keys()),
        help=f"Preset name (default: {DEFAULT_PRESET_NAME})",
    )
    p.add_argument("--data", required=True, help="Input .h5ad file")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    p.add_argument(
        "--output-profile",
        choices=["standard", "search_minimal"],
        default="standard",
        help="Artifact profile: standard exports all usual outputs, search_minimal keeps only search-critical files.",
    )
    p.add_argument(
        "--save-processed-data",
        choices=["on", "off"],
        default="off",
        help="Save data/processed.h5ad in output directory (default: off)",
    )
    p.add_argument("--metrics-only", action="store_true", help="Skip figure generation")
    p.add_argument(
        "--skip-evaluation-metrics",
        action="store_true",
        help="Skip clustering/evaluation metric computation and export metric files with empty scores.",
    )
    p.add_argument(
        "--umap-only",
        action="store_true",
        help="Generate only UMAP visualizations and skip non-UMAP analyses/plots.",
    )
    p.add_argument(
        "--umap-context-annotation",
        choices=["on", "off"],
        default="on",
        help="Show or hide the bottom-left run context annotation on UMAP figures.",
    )
    p.add_argument(
        "--umap-point-size",
        type=float,
        default=5.0,
        help="Scatter point size for UMAP figures (default: 5.0).",
    )
    p.add_argument(
        "--figure-format",
        choices=["png", "svg"],
        default="png",
        help="Figure export format for generated plots (default: png).",
    )
    p.add_argument(
        "--export-comparison-panels",
        action="store_true",
        help="Also export the 3 panels of the final comparison UMAP as separate PNG files.",
    )
    p.add_argument("--unsupervised", action="store_true", help="Hide labels during training")
    p.add_argument("--dann", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--batch-key", default=None, help="Batch key override when DANN is enabled")
    p.add_argument(
        "--leiden-target-clusters",
        type=int,
        default=None,
        help=(
            "Override the target cluster count used for exported Leiden comparison runs. "
            "Default: auto-detect from truth labels when available, otherwise fall back "
            "to the effective pseudo-label K."
        ),
    )
    p.add_argument(
        "--compute-scib-metrics",
        choices=["off", "auto", "on"],
        default="off",
        help="Compute scIB/scIB-E metrics when possible (default: off).",
    )
    p.add_argument("--scib-n-jobs", type=int, default=1, help="Worker count for scIB metric computation.")
    p.add_argument(
        "--auto-hparams",
        choices=["off", "auto", "on"],
        default="auto",
        help=(
            "Dataset-adaptive hyperparameter tuning. 'auto' applies it to the default preset only, "
            "'on' forces it, and 'off' keeps the raw preset unchanged."
        ),
    )
    p.add_argument(
        "--auto-hparams-sketch-max-cells",
        type=int,
        default=3000,
        help="Max cells sampled for the pre-training structure sketch used by auto_hparams.",
    )
    p.add_argument("--capture-snapshots", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--snapshot-interval", type=int, default=None)
    p.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override algorithm param: KEY=VALUE (repeatable)",
    )
    p.add_argument(
        "--preprocess",
        action="append",
        default=[],
        help="Override preprocessing param: KEY=VALUE (repeatable)",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint with centralized error handling and logging."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return run_once(args)
    except Exception as exc:
        logger.exception("scRAW dedicated run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
