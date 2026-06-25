#!/usr/bin/env python3
"""Lightweight metric computation for standalone scRAW runs."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

NOISE_LABELS = {-1, "-1", "noise", "Noise", "NOISE", "unassigned", "Unassigned"}
logger = logging.getLogger(__name__)

_SCIB_BENCHMARKER = None
_SCIB_IMPORT_ATTEMPTED = False
_SCIB_IMPORT_ERROR: Optional[str] = None
_SCIB_IMPORT_WARNING_EMITTED = False


def _to_array(values: Any) -> np.ndarray:
    """Convert any array-like input to a NumPy array."""
    return np.asarray(values)


def _get_scib_benchmarker():
    """Lazy-load `scib_metrics.Benchmarker` if the package is available."""
    global _SCIB_BENCHMARKER
    global _SCIB_IMPORT_ATTEMPTED
    global _SCIB_IMPORT_ERROR
    global _SCIB_IMPORT_WARNING_EMITTED

    if not _SCIB_IMPORT_ATTEMPTED:
        try:
            from scib_metrics.benchmark import Benchmarker  # type: ignore

            _SCIB_BENCHMARKER = Benchmarker
        except Exception as exc:
            _SCIB_BENCHMARKER = None
            _SCIB_IMPORT_ERROR = str(exc)
        finally:
            _SCIB_IMPORT_ATTEMPTED = True

    if _SCIB_BENCHMARKER is None and not _SCIB_IMPORT_WARNING_EMITTED:
        err = _SCIB_IMPORT_ERROR or "unknown import error"
        logger.warning(
            "scib_metrics not available: %s. scIB/scIB-E metrics will be skipped.",
            err,
        )
        _SCIB_IMPORT_WARNING_EMITTED = True

    return _SCIB_BENCHMARKER


def scib_metrics_available() -> bool:
    """Return True when `scib_metrics` can be imported in the current environment."""
    return _get_scib_benchmarker() is not None


def _filter_noise(
    labels_true: Optional[np.ndarray], labels_pred: np.ndarray, embeddings: Optional[np.ndarray]
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Remove rows marked as noise in predicted/true labels (and embeddings)."""
    mask_pred = np.ones(len(labels_pred), dtype=bool)
    labels_pred_str = labels_pred.astype(str)
    for v in NOISE_LABELS:
        mask_pred &= labels_pred_str != str(v)

    if labels_true is None:
        if embeddings is not None:
            embeddings = embeddings[mask_pred]
        return None, labels_pred[mask_pred], embeddings

    labels_true_str = labels_true.astype(str)
    mask_true = np.ones(len(labels_true), dtype=bool)
    for v in NOISE_LABELS:
        mask_true &= labels_true_str != str(v)

    mask = mask_pred & mask_true
    emb_f = embeddings[mask] if embeddings is not None else None
    return labels_true[mask], labels_pred[mask], emb_f


def _get_expression_matrix_for_pca(adata: Any) -> Any:
    """Pick a reasonable matrix for PCA when approximating scIB-E Jaccard."""
    for layer in ["counts", "raw_counts", "original_X", "X_raw"]:
        if hasattr(adata, "layers") and layer in adata.layers:
            return adata.layers[layer]
    if hasattr(adata, "raw") and adata.raw is not None:
        return adata.raw.X
    return adata.X


def _matrix_looks_like_raw_counts(X: Any, sample_size: int = 4096) -> bool:
    """Heuristic used to avoid double normalizing/log-transforming already processed matrices."""
    if hasattr(X, "data"):
        values = np.asarray(X.data)
    else:
        values = np.asarray(X).ravel()

    if values.size == 0:
        return False

    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    if values.size > sample_size:
        values = values[:sample_size]
    if np.nanmin(values) < 0:
        return False

    rounded = np.round(values)
    frac_integer_like = float(np.mean(np.abs(values - rounded) < 1e-6))
    return frac_integer_like >= 0.98


def compute_jaccard_index(
    adata: Any,
    batch_key: str,
    embedding_key: str,
    k: int = 15,
    n_pcs: int = 50,
    pca_key: str = "X_pca_batch",
) -> Optional[float]:
    """
    Compute the per-batch kNN graph Jaccard index used in the scIB-E aggregate.
    """
    try:
        import scanpy as sc
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:
        logger.warning("Jaccard index skipped (missing dependency): %s", exc)
        return None

    if not hasattr(adata, "obs") or batch_key not in adata.obs.columns:
        return None
    if not hasattr(adata, "obsm") or embedding_key not in adata.obsm:
        return None

    try:
        if hasattr(adata, "obsm") and pca_key in adata.obsm:
            x_pca_batch = adata.obsm[pca_key]
        else:
            x_pca_batch = np.zeros((adata.n_obs, n_pcs), dtype=np.float32)
            batches = adata.obs[batch_key].unique()
            for batch in batches:
                mask = adata.obs[batch_key] == batch
                if int(np.sum(mask)) < 2:
                    continue
                try:
                    adata_batch = adata[mask].copy()
                    base_matrix = _get_expression_matrix_for_pca(adata_batch)
                    adata_batch.X = base_matrix.copy() if hasattr(base_matrix, "copy") else base_matrix
                    if _matrix_looks_like_raw_counts(adata_batch.X):
                        sc.pp.normalize_total(adata_batch, target_sum=1e4)
                        sc.pp.log1p(adata_batch)

                    n_obs_batch, n_vars_batch = adata_batch.shape
                    n_comps_eff = min(int(n_pcs), int(n_obs_batch) - 1, int(n_vars_batch) - 1)
                    if n_comps_eff < 2:
                        continue

                    sc.pp.pca(adata_batch, n_comps=n_comps_eff, svd_solver="arpack")
                    x_batch = np.asarray(adata_batch.obsm["X_pca"], dtype=np.float32)
                    x_pca_batch[mask, :x_batch.shape[1]] = x_batch
                except Exception as exc:
                    logger.warning("Jaccard index skipped for batch %s: %s", batch, exc)
                    continue

        embeddings = adata.obsm[embedding_key]
        all_ja: List[float] = []
        batches = adata.obs[batch_key].unique()
        for batch in batches:
            mask = adata.obs[batch_key] == batch
            n_batch = int(np.sum(mask))
            if n_batch < 2:
                continue

            x_pca = x_pca_batch[mask]
            x_emb = embeddings[mask]
            k_eff = min(k, n_batch - 1)
            if k_eff <= 0:
                continue

            nbrs1 = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(x_pca)
            a1 = nbrs1.kneighbors_graph(x_pca, k_eff, mode="connectivity")

            nbrs2 = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(x_emb)
            a2 = nbrs2.kneighbors_graph(x_emb, k_eff, mode="connectivity")

            a1.setdiag(0)
            a2.setdiag(0)
            a1 = a1.maximum(a1.T)
            a2 = a2.maximum(a2.T)

            intersection = a1.multiply(a2)
            union = a1.maximum(a2)
            if union.nnz == 0:
                continue
            all_ja.append(float(intersection.nnz) / float(union.nnz))
    except Exception as exc:
        logger.warning("Jaccard index skipped: %s", exc)
        return None

    if not all_ja:
        return None
    return float(np.mean(all_ja))


def _extract_scib_metric(row: Any, columns: List[str], canonical: str, aliases: List[str]) -> Optional[float]:
    """Extract one scIB metric robustly across minor naming differences."""
    for name in [canonical] + aliases:
        if name in columns:
            try:
                value = row[name]
                if value is None:
                    return None
                return float(value)
            except Exception:
                return None
    return None


def _compute_exact_neighbors(X: np.ndarray, n_neighbors: int, n_jobs: int = 1):
    """Compute exact kNN results for scIB using scikit-learn."""
    from scib_metrics.nearest_neighbors import NeighborsResults
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("Exact neighbor computation requires a non-empty 2D array.")

    k = min(int(n_neighbors), int(X.shape[0]))
    if k <= 0:
        raise ValueError("Exact neighbor computation requires at least one sample.")

    nn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
        algorithm="auto",
        n_jobs=max(1, int(n_jobs)),
    )
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    return NeighborsResults(indices=np.asarray(indices), distances=np.asarray(distances))


def _safe_scib_float(value: Any) -> Optional[float]:
    """Convert one scIB result to a finite float when possible."""
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _ensure_scib_preintegrated_pca(adata: Any) -> Optional[np.ndarray]:
    """Return the pre-integration PCA used by PCR comparison."""
    import scanpy as sc

    if hasattr(adata, "obsm") and "X_pca" in adata.obsm:
        return np.asarray(adata.obsm["X_pca"], dtype=np.float32)

    try:
        sc.tl.pca(adata, mask_var=None)
    except TypeError:
        sc.tl.pca(adata, use_highly_variable=False)

    if hasattr(adata, "obsm") and "X_pca" in adata.obsm:
        return np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    return None


def _compute_scib_neighbor_sets(embeddings: np.ndarray, n_jobs: int) -> Dict[int, Any]:
    """Compute the 15/50/90-NN structures expected by scIB metrics."""
    max_neighbors = min(90, int(embeddings.shape[0]))
    if max_neighbors < 2:
        raise ValueError("scIB metrics require at least two samples.")

    neighbors_max = _compute_exact_neighbors(
        embeddings,
        n_neighbors=max_neighbors,
        n_jobs=n_jobs,
    )
    out: Dict[int, Any] = {}
    for n in (15, 50, 90):
        out[n] = neighbors_max.subset_neighbors(min(n, int(neighbors_max.n_neighbors)))
    return out


def _compute_kbet_per_label_exact(
    embeddings: np.ndarray,
    batches: np.ndarray,
    labels: np.ndarray,
    *,
    alpha: float = 0.05,
    n_jobs: int = 1,
) -> Optional[float]:
    """Stable fallback for KBET that avoids the fragile UMAP connectivity path."""
    import pandas as pd
    import scib_metrics

    embeddings = np.asarray(embeddings, dtype=np.float32)
    batches = np.asarray(batches)
    labels = np.asarray(labels)

    size_max = 2**31 - 1
    scores: List[float] = []

    for clus in np.unique(labels):
        mask = labels == clus
        emb_sub = embeddings[mask]
        batches_sub = batches[mask]
        n_obs = int(emb_sub.shape[0])

        if n_obs < 10 or len(np.unique(batches_sub)) == 1:
            logger.info("%s consists of a single batch or is too small. Skip.", clus)
            continue

        batch_counts = pd.Series(batches_sub).value_counts()
        quarter_mean = int(np.floor(float(batch_counts.mean()) / 4.0))
        k0 = int(min(70, max(10, quarter_mean)))
        if k0 * n_obs >= size_max:
            k0 = int(np.floor(size_max / max(1, n_obs)))

        n_neighbors = min(max(2, k0 + 1), n_obs)
        if n_neighbors < 2:
            continue

        try:
            nn_sub = _compute_exact_neighbors(
                emb_sub,
                n_neighbors=n_neighbors,
                n_jobs=n_jobs,
            )
            score, _, _ = scib_metrics.kbet(
                nn_sub,
                batches=batches_sub,
                alpha=alpha,
            )
        except Exception as exc:
            logger.warning("scIB KBET fallback failed for label %s: %s", clus, exc)
            score = 0.0

        score_f = _safe_scib_float(score)
        if score_f is not None:
            scores.append(score_f)

    if not scores:
        return None
    return float(np.nanmean(scores))


def compute_scib_metrics(
    adata: Any,
    embeddings: np.ndarray,
    batch_key: Optional[str],
    label_key: Optional[str],
    n_jobs: int = 1,
    min_max_scale: bool = False,
    compute_jaccard: bool = True,
    jaccard_k: int = 15,
    jaccard_pcs: int = 50,
) -> Dict[str, float]:
    """Compute scIB/scIB-E metrics on one embedding."""
    if _get_scib_benchmarker() is None:
        return {}
    if adata is None or embeddings is None or batch_key is None or label_key is None:
        return {}
    if batch_key not in getattr(adata, "obs", {}):
        return {}
    if label_key not in getattr(adata, "obs", {}):
        return {}

    try:
        adata_scib = adata.copy()
    except Exception:
        return {}

    if len(embeddings) != adata_scib.n_obs:
        logger.warning(
            "scIB metrics skipped: embeddings length (%s) != adata.n_obs (%s)",
            len(embeddings),
            adata_scib.n_obs,
        )
        return {}

    embed_key = "__scib_embed"
    embeddings = np.asarray(embeddings, dtype=np.float32)
    adata_scib.obsm[embed_key] = embeddings

    if not np.isfinite(embeddings).all():
        n_bad = int(np.size(embeddings) - np.isfinite(embeddings).sum())
        logger.warning("scIB metrics skipped: embeddings contain %d non-finite values.", n_bad)
        return {}

    import scib_metrics

    labels = np.asarray(adata_scib.obs[label_key].to_numpy())
    batches = np.asarray(adata_scib.obs[batch_key].to_numpy())

    try:
        neighbors = _compute_scib_neighbor_sets(embeddings, n_jobs=max(1, int(n_jobs)))
    except Exception as exc:
        logger.warning("scIB metrics skipped during neighbor preparation: %s", exc, exc_info=True)
        return {}

    metrics: Dict[str, float] = {}

    def _record_metric(metric_name: str, compute_fn) -> Optional[float]:
        try:
            value = compute_fn()
        except Exception as exc:
            logger.warning("scIB metric %s skipped: %s", metric_name, exc)
            return None
        value_f = _safe_scib_float(value)
        if value_f is not None:
            metrics[metric_name] = value_f
        return value_f

    _record_metric(
        "Isolated labels",
        lambda: scib_metrics.isolated_labels(embeddings, labels, batches),
    )

    try:
        nmi_ari = scib_metrics.nmi_ari_cluster_labels_kmeans(embeddings, labels)
    except Exception as exc:
        logger.warning("scIB metric nmi_ari_cluster_labels_kmeans skipped: %s", exc)
    else:
        nmi = _safe_scib_float(nmi_ari.get("nmi"))
        ari = _safe_scib_float(nmi_ari.get("ari"))
        if nmi is not None:
            metrics["KMeans NMI"] = nmi
        if ari is not None:
            metrics["KMeans ARI"] = ari

    _record_metric(
        "Silhouette label",
        lambda: scib_metrics.silhouette_label(embeddings, labels),
    )
    _record_metric(
        "cLISI",
        lambda: scib_metrics.clisi_knn(neighbors[90], labels),
    )
    _record_metric(
        "Silhouette batch",
        lambda: scib_metrics.silhouette_batch(embeddings, labels, batches),
    )
    _record_metric(
        "iLISI",
        lambda: scib_metrics.ilisi_knn(neighbors[90], batches),
    )

    try:
        kbet_score = scib_metrics.kbet_per_label(neighbors[50], batches, labels)
    except Exception as exc:
        if "get_call_template" in str(exc):
            logger.warning(
                "scIB metric KBET hit the UMAP/numba connectivity path (%s); retrying with exact within-label neighbors.",
                exc,
            )
        else:
            logger.warning(
                "scIB metric KBET primary path failed (%s); retrying with exact within-label neighbors.",
                exc,
            )
        kbet_score = _compute_kbet_per_label_exact(
            embeddings,
            batches,
            labels,
            n_jobs=max(1, int(n_jobs)),
        )
    kbet_score_f = _safe_scib_float(kbet_score)
    if kbet_score_f is not None:
        metrics["KBET"] = kbet_score_f

    _record_metric(
        "Graph connectivity",
        lambda: scib_metrics.graph_connectivity(neighbors[15], labels),
    )

    x_pre = _ensure_scib_preintegrated_pca(adata_scib)
    if x_pre is not None:
        _record_metric(
            "PCR comparison",
            lambda: scib_metrics.pcr_comparison(x_pre, embeddings, batches, categorical=True),
        )

    if compute_jaccard:
        jaccard = compute_jaccard_index(
            adata_scib,
            batch_key=batch_key,
            embedding_key=embed_key,
            k=jaccard_k,
            n_pcs=jaccard_pcs,
        )
        if jaccard is not None:
            metrics["Jaccard index"] = float(jaccard)

    def _mean_available(keys: List[str]) -> Optional[float]:
        values = [metrics.get(k) for k in keys if metrics.get(k) is not None and not np.isnan(metrics[k])]
        if not values:
            return None
        return float(np.mean(values))

    batch_keys = ["Silhouette batch", "iLISI", "KBET", "Graph connectivity"]
    inter_keys = ["Isolated labels", "KMeans NMI", "KMeans ARI", "Silhouette label", "cLISI"]
    intra_keys = ["PCR comparison", "Jaccard index"]

    batch_score = _mean_available(batch_keys)
    inter_score = _mean_available(inter_keys)
    intra_score = _mean_available(intra_keys)

    if batch_score is not None:
        metrics["Batch correction"] = batch_score
    if inter_score is not None:
        metrics["Inter cell-type conservation"] = inter_score
    if intra_score is not None:
        metrics["Intra cell-type conservation"] = intra_score
    if batch_score is not None and inter_score is not None and intra_score is not None:
        metrics["scIB-E Total score"] = 0.2 * batch_score + 0.4 * inter_score + 0.4 * intra_score

    return metrics


def align_labels(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Align predicted clusters to true labels with Hungarian matching."""
    labels_true = _to_array(labels_true)
    labels_pred = _to_array(labels_pred)
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must have same length")

    true_u = np.unique(labels_true)
    pred_u = np.unique(labels_pred)
    true_map = {lab: i for i, lab in enumerate(true_u)}
    pred_map = {lab: i for i, lab in enumerate(pred_u)}

    w = np.zeros((len(pred_u), len(true_u)), dtype=np.int64)
    for t, p in zip(labels_true, labels_pred):
        w[pred_map[p], true_map[t]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {pred_u[r]: true_u[c] for r, c in zip(row_ind, col_ind)}

    out = np.array([mapping.get(p, f"Unmatched_{p}") for p in labels_pred], dtype=object)
    return out


def marker_overlap_annotation(
    adata: Any,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    n_top_genes: int = 100,
    method: str = "wilcoxon",
) -> Dict[str, Any]:
    """
    Annotate predicted clusters using marker-gene overlap with gold labels.

    Returns a dict with marker labels, overlap matrix, DEG lists, and Hungarian labels.
    """
    import pandas as pd
    import scanpy as sc

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if len(labels_true) != adata.n_obs or len(labels_pred) != adata.n_obs:
        raise ValueError(
            f"Label lengths ({len(labels_true)}, {len(labels_pred)}) "
            f"must match adata.n_obs ({adata.n_obs})."
        )

    def _compute_degs_per_group(adata_work: Any, labels: np.ndarray, group_name: str) -> Dict[str, list]:
        adata_copy = adata_work.copy()
        adata_copy.obs[group_name] = labels.astype(str)

        unique_groups = sorted(adata_copy.obs[group_name].unique())
        degs_dict: Dict[str, list] = {}

        try:
            sc.tl.rank_genes_groups(
                adata_copy,
                groupby=group_name,
                method=method,
                n_genes=n_top_genes,
                use_raw=False,
            )
        except Exception:
            sc.tl.rank_genes_groups(
                adata_copy,
                groupby=group_name,
                method="t-test",
                n_genes=n_top_genes,
                use_raw=False,
            )

        for grp in unique_groups:
            try:
                df = sc.get.rank_genes_groups_df(adata_copy, group=grp)
                degs_dict[grp] = df["names"].head(n_top_genes).tolist()
            except Exception as exc:
                logger.warning("Could not get DEGs for group %s: %s", grp, exc)
                degs_dict[grp] = []

        return degs_dict

    logger.info("Computing gold-standard DEGs from ground-truth labels...")
    gold_degs = _compute_degs_per_group(adata, labels_true, "_gold")

    logger.info("Computing DEGs from predicted clusters...")
    pred_degs = _compute_degs_per_group(adata, labels_pred, "_pred")

    gold_types = sorted(gold_degs.keys())
    pred_clusters = sorted(pred_degs.keys())
    overlap_data = np.zeros((len(pred_clusters), len(gold_types)))

    for i, pred_cluster in enumerate(pred_clusters):
        pred_genes = set(pred_degs.get(pred_cluster, []))
        for j, gold_type in enumerate(gold_types):
            gold_genes = set(gold_degs.get(gold_type, []))
            if pred_genes and gold_genes:
                overlap_data[i, j] = len(pred_genes & gold_genes) / float(n_top_genes)
            else:
                overlap_data[i, j] = 0.0

    overlap_df = pd.DataFrame(
        overlap_data,
        index=pred_clusters,
        columns=gold_types,
    )
    overlap_df.index.name = "Predicted Cluster"
    overlap_df.columns.name = "Gold Standard Type"

    cluster_to_type: Dict[str, str] = {}
    for i, pred_cluster in enumerate(pred_clusters):
        best_idx = int(np.argmax(overlap_data[i]))
        best_score = overlap_data[i, best_idx]
        best_type = gold_types[best_idx]
        cluster_to_type[pred_cluster] = best_type if best_score > 0 else f"Unknown_{pred_cluster}"

    marker_labels = np.array(
        [cluster_to_type.get(str(lbl), f"Unknown_{lbl}") for lbl in labels_pred],
        dtype=object,
    )
    hungarian_labels = align_labels(labels_true, labels_pred)

    return {
        "marker_labels": marker_labels,
        "overlap_matrix": overlap_df,
        "gold_degs": gold_degs,
        "pred_degs": pred_degs,
        "hungarian_labels": hungarian_labels,
        "cluster_to_type": cluster_to_type,
    }


def _accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute accuracy after Hungarian alignment."""
    labels_true = _to_array(labels_true)
    labels_pred = _to_array(labels_pred)
    aligned = align_labels(labels_true, labels_pred)
    return float(np.mean(aligned == labels_true))


def _balanced_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    """Compute macro F1 and balanced accuracy after label alignment."""
    from sklearn.metrics import balanced_accuracy_score, f1_score

    aligned = align_labels(labels_true, labels_pred)
    return {
        "F1_Macro": float(f1_score(labels_true, aligned, average="macro", zero_division=0)),
        "BalancedACC": float(balanced_accuracy_score(labels_true, aligned)),
    }


def _rare_acc(labels_true: np.ndarray, labels_pred: np.ndarray, threshold: float = 0.05) -> Optional[float]:
    """Compute accuracy restricted to rare classes below `threshold` frequency."""
    labels_true = _to_array(labels_true)
    aligned = align_labels(labels_true, labels_pred)
    classes, counts = np.unique(labels_true, return_counts=True)
    freq = counts / max(len(labels_true), 1)
    rare = classes[freq < threshold]
    if len(rare) == 0:
        return None
    mask = np.isin(labels_true, rare)
    if not np.any(mask):
        return None
    return float(np.mean(aligned[mask] == labels_true[mask]))


def _ultra_rare_acc(labels_true: np.ndarray, labels_pred: np.ndarray) -> Optional[float]:
    """Compute accuracy restricted to classes below 1% frequency."""
    return _rare_acc(labels_true, labels_pred, threshold=0.01)


def _classwise(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Return per-class precision/recall/F1/support after alignment."""
    from sklearn.metrics import precision_recall_fscore_support

    labels_true = _to_array(labels_true)
    aligned = align_labels(labels_true, labels_pred)
    classes = np.unique(labels_true)
    p, r, f, s = precision_recall_fscore_support(
        labels_true, aligned, labels=classes, zero_division=0
    )

    out: Dict[str, Dict[str, float]] = {}
    for i, cls in enumerate(classes):
        out[str(cls)] = {
            "Precision": float(p[i]),
            "Recall": float(r[i]),
            "F1": float(f[i]),
            "Support": int(s[i]),
        }
    return out


def _silhouette(embeddings: np.ndarray, labels_pred: np.ndarray, sample_size: Optional[int] = 5000) -> float:
    """Compute silhouette score with optional random subsampling."""
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels_pred)) < 2:
        return 0.0

    X = embeddings
    y = labels_pred
    if sample_size is not None and len(y) > sample_size:
        idx = np.random.choice(len(y), sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    try:
        return float(silhouette_score(X, y))
    except Exception:
        return 0.0


def _knn_purity(latent: np.ndarray, labels: np.ndarray, n_neighbors: int = 30) -> float:
    """Compute class-balanced kNN purity in latent space."""
    from sklearn.neighbors import NearestNeighbors

    latent = _to_array(latent)
    labels = _to_array(labels)
    if len(labels) < 2:
        return float("nan")

    k = min(n_neighbors, len(labels) - 1)
    if k <= 0:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    idx = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neigh_labels = labels[idx]
    per_cell = (neigh_labels == labels.reshape(-1, 1)).mean(axis=1)
    per_class = [np.mean(per_cell[labels == c]) for c in np.unique(labels)]
    return float(np.mean(per_class))


def compute_metrics(
    labels_true: Optional[np.ndarray],
    labels_pred: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    *,
    adata: Any = None,
    batch_key: Optional[str] = None,
    label_key: Optional[str] = None,
    compute_scib: bool = False,
    scib_n_jobs: int = 1,
) -> Dict[str, Any]:
    """Compute the full clustering metric bundle used by scRAW outputs."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    labels_pred_raw = _to_array(labels_pred)
    labels_true_raw = _to_array(labels_true) if labels_true is not None else None
    embeddings_raw = _to_array(embeddings) if embeddings is not None else None
    labels_true, labels_pred, embeddings = _filter_noise(labels_true_raw, labels_pred_raw, embeddings_raw)

    out: Dict[str, Any] = {
        "NMI": float("nan"),
        "ARI": float("nan"),
        "ACC": float("nan"),
        "UCA": float("nan"),
        "F1_Macro": float("nan"),
        "BalancedACC": float("nan"),
        "RareACC": float("nan"),
        "UltraRareACC": float("nan"),
        "KNN_Purity": float("nan"),
        "ClassWise": {},
        "Silhouette": float("nan"),
        "n_clusters_found": int(len(np.unique(labels_pred))) if len(labels_pred) else 0,
        "n_samples_evaluated": int(len(labels_pred)),
    }

    if labels_true is not None and len(labels_true) > 0:
        out["NMI"] = float(normalized_mutual_info_score(labels_true, labels_pred))
        out["ARI"] = float(adjusted_rand_score(labels_true, labels_pred))
        out["ACC"] = _accuracy(labels_true, labels_pred)
        out["UCA"] = out["ACC"]

        bm = _balanced_metrics(labels_true, labels_pred)
        out.update(bm)

        rare = _rare_acc(labels_true, labels_pred)
        if rare is not None:
            out["RareACC"] = float(rare)

        ultra_rare = _ultra_rare_acc(labels_true, labels_pred)
        if ultra_rare is not None:
            out["UltraRareACC"] = float(ultra_rare)

        out["ClassWise"] = _classwise(labels_true, labels_pred)

        if embeddings is not None and len(embeddings) == len(labels_true):
            out["KNN_Purity"] = _knn_purity(embeddings, labels_true)

    if embeddings is not None and len(embeddings) == len(labels_pred):
        out["Silhouette"] = _silhouette(embeddings, labels_pred)

    if compute_scib and adata is not None and embeddings_raw is not None:
        try:
            adata_scib = adata
            scib_label_key = label_key
            if scib_label_key is None and labels_true_raw is not None:
                adata_scib = adata.copy()
                scib_label_key = "__labels__"
                adata_scib.obs[scib_label_key] = np.asarray(labels_true_raw)
            elif scib_label_key is not None and scib_label_key not in getattr(adata, "obs", {}):
                if labels_true_raw is not None:
                    adata_scib = adata.copy()
                    adata_scib.obs[scib_label_key] = np.asarray(labels_true_raw)
                else:
                    adata_scib = adata

            out.update(
                compute_scib_metrics(
                    adata=adata_scib,
                    embeddings=embeddings_raw,
                    batch_key=batch_key,
                    label_key=scib_label_key,
                    n_jobs=scib_n_jobs,
                )
            )
        except Exception as exc:
            logger.warning("scIB metrics skipped: %s", exc, exc_info=True)

    return out
