"""
Clustering metrics for evaluation.

Includes:
- Standard metrics: NMI, ARI, ACC, UCA, Silhouette, KNN Purity, NNO
- Group-wise metrics: metrics computed per batch/dataset source
- Comparison utilities for benchmark mode
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment
import logging
import os

logger = logging.getLogger(__name__)


# Special label values that indicate noise/unassigned samples
NOISE_LABELS = {-1, '-1', 'noise', 'Noise', 'NOISE', 'unassigned', 'Unassigned'}


_SCIB_BENCHMARKER = None
_SCIB_IMPORT_DONE = False
_SCIB_IMPORT_ERROR: Optional[str] = None
_SCIB_WARNING_EMITTED = False
_SCIB_DISABLED_INFO_EMITTED = False


def _get_scib_benchmarker():
    """Lazy-load scib_metrics.Benchmarker with warning emitted at most once."""
    global _SCIB_BENCHMARKER
    global _SCIB_IMPORT_DONE
    global _SCIB_IMPORT_ERROR
    global _SCIB_WARNING_EMITTED
    global _SCIB_DISABLED_INFO_EMITTED

    disable_env = os.getenv("SCRB_DISABLE_SCIB_METRICS", "").strip().lower()
    if disable_env in {"1", "true", "yes", "on"}:
        if not _SCIB_DISABLED_INFO_EMITTED:
            logger.info("scIB/scIB-E metrics disabled via SCRB_DISABLE_SCIB_METRICS.")
            _SCIB_DISABLED_INFO_EMITTED = True
        return None

    if not _SCIB_IMPORT_DONE:
        try:
            from scib_metrics.benchmark import Benchmarker  # type: ignore
            _SCIB_BENCHMARKER = Benchmarker
        except Exception as e:
            _SCIB_BENCHMARKER = None
            _SCIB_IMPORT_ERROR = str(e)
        finally:
            _SCIB_IMPORT_DONE = True

    if _SCIB_BENCHMARKER is None:
        if not _SCIB_WARNING_EMITTED:
            err = _SCIB_IMPORT_ERROR or "unknown import error"
            logger.warning(
                f"scib_metrics not available: {err}. "
                "scIB/scIB-E metrics will be skipped for this run."
            )
            _SCIB_WARNING_EMITTED = True
        return None

    return _SCIB_BENCHMARKER


def _scib_benchmarker_supports_current_sparse_indexing() -> bool:
    """Return whether scipy sparse matrices accept Benchmarker pandas masks.

    scib-metrics 0.5.1 passes pandas Series masks to scipy sparse matrices.
    Recent scipy releases reject those masks with ``Series.nonzero`` errors.
    The direct scib-metrics API below already converts labels to NumPy arrays.
    """
    try:
        import pandas as pd
        from scipy.sparse import csr_matrix

        probe = csr_matrix(np.eye(2, dtype=np.float32))
        probe[pd.Series([True, False])]
    except (AttributeError, TypeError):
        return False
    return True


def _filter_noise_samples(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Filter out samples marked as noise (-1 or 'noise') in either true or predicted labels.

    Many clustering algorithms (HDBSCAN, DBSCAN, some graph methods) assign -1 to
    noise points. These should be excluded from evaluation as they are not assigned
    to any cluster.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        data: Optional data matrix (for Silhouette score)

    Returns:
        Tuple of (filtered_true, filtered_pred, filtered_data, valid_mask) with noise removed
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    # Create mask for non-noise samples
    # Check both numeric -1 and string representations
    mask_true = np.ones(len(labels_true), dtype=bool)
    mask_pred = np.ones(len(labels_pred), dtype=bool)

    for noise_val in NOISE_LABELS:
        if np.issubdtype(labels_true.dtype, np.number):
            if isinstance(noise_val, (int, float)):
                mask_true &= (labels_true != noise_val)
        else:
            mask_true &= (labels_true.astype(str) != str(noise_val))

        if np.issubdtype(labels_pred.dtype, np.number):
            if isinstance(noise_val, (int, float)):
                mask_pred &= (labels_pred != noise_val)
        else:
            mask_pred &= (labels_pred.astype(str) != str(noise_val))

    # Combine masks - exclude if EITHER is noise
    valid_mask = mask_true & mask_pred

    n_filtered = len(labels_true) - np.sum(valid_mask)
    if n_filtered > 0:
        logger.info(f"Filtered {n_filtered} noise samples (label=-1 or 'noise') from metric computation")

    filtered_data = data[valid_mask] if data is not None else None

    return labels_true[valid_mask], labels_pred[valid_mask], filtered_data, valid_mask


def compute_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels

    Returns:
        NMI score in [0, 1]
    """
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(labels_true, labels_pred)


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels

    Returns:
        ARI score in [-1, 1]
    """
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels_true, labels_pred)


def compute_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute clustering accuracy using Hungarian algorithm.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels

    Returns:
        Accuracy in [0, 1]
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    # Encode string labels or mixed labels to integers
    def force_int_labels(labels):
        # Already clean integers?
        if np.issubdtype(labels.dtype, np.integer) and labels.min() >= 0:
            return labels.astype(np.int64)
        
        # Otherwise, cast to string to handle mixed types (e.g. string labels vs integer clusters)
        # or negative values (e.g. noise as -1)
        labels_str = labels.astype(str)
        unique_labels = np.unique(labels_str)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        return np.array([label_map[l] for l in labels_str], dtype=np.int64)

    labels_true = force_int_labels(labels_true)
    labels_pred = force_int_labels(labels_pred)

    assert labels_pred.size == labels_true.size

    D = max(labels_pred.max(), labels_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(labels_pred.size):
        w[labels_pred[i], labels_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / labels_pred.size


def compute_unsupervised_clustering_accuracy(
    labels_true: Union[np.ndarray, 'torch.Tensor'],
    labels_pred: Union[np.ndarray, 'torch.Tensor']
) -> Tuple[float, np.ndarray]:
    """
    Compute Unsupervised Clustering Accuracy (UCA) using Hungarian algorithm.

    This follows the scIB-E implementation and returns both accuracy and the
    optimal assignment matrix (pred_idx -> true_idx).
    """
    # Convert torch tensors if needed
    try:
        import torch  # local import to avoid hard dependency
        if isinstance(labels_true, torch.Tensor):
            labels_true = labels_true.detach().cpu().numpy()
        if isinstance(labels_pred, torch.Tensor):
            labels_pred = labels_pred.detach().cpu().numpy()
    except Exception:
        pass

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    if len(labels_pred) != len(labels_true):
        raise ValueError("len(labels_pred) != len(labels_true)")

    # Use string keys to handle mixed label types safely
    labels_true_str = labels_true.astype(str)
    labels_pred_str = labels_pred.astype(str)

    u = np.unique(np.concatenate((labels_true_str, labels_pred_str)))
    n_clusters = len(u)
    mapping = dict(zip(u, range(n_clusters)))

    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for y_pred_, y_ in zip(labels_pred_str, labels_true_str):
        if y_ in mapping:
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1

    cost_matrix = reward_matrix.max() - reward_matrix
    row_assign, col_assign = linear_sum_assignment(cost_matrix)

    # Construct optimal assignments matrix
    row_assign = row_assign.reshape((-1, 1))
    col_assign = col_assign.reshape((-1, 1))
    assignments = np.concatenate((row_assign, col_assign), axis=1)

    optimal_reward = reward_matrix[row_assign, col_assign].sum() * 1.0
    return optimal_reward / labels_pred.size, assignments


def compute_knn_purity(
    latent: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 30
) -> float:
    """
    Compute KNN purity of a latent space with respect to labels.

    This follows the scIB-E definition: per-cell purity averaged per class.
    """
    from sklearn.neighbors import NearestNeighbors

    latent = np.asarray(latent)
    labels = np.asarray(labels)

    n_samples = len(labels)
    if n_samples < 2:
        return np.nan

    n_neighbors = min(n_neighbors, n_samples - 1)
    if n_neighbors <= 0:
        return np.nan

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = nbrs.kneighbors(latent, return_distance=False)[:, 1:]
    neighbors_labels = labels[indices]

    # Per-cell purity scores
    scores = (neighbors_labels == labels.reshape(-1, 1)).mean(axis=1)
    per_class = [np.mean(scores[labels == cls]) for cls in np.unique(labels)]

    return float(np.mean(per_class))


def compute_nearest_neighbor_overlap(
    x1: np.ndarray,
    x2: np.ndarray,
    k: int = 100,
    sample_size: Optional[int] = 2000,
    random_state: int = 0
) -> Tuple[float, float]:
    """
    Compute the overlap between the k-nearest neighbor graphs of x1 and x2.

    Returns:
        (spearman_correlation, fold_enrichment)

    Notes:
    - Matches scIB-E implementation but optionally subsamples for scalability.
    - If sample_size is None, uses all samples.
    """
    import scipy
    from sklearn.neighbors import NearestNeighbors

    if len(x1) != len(x2):
        raise ValueError("len(x1) != len(x2)")

    n_samples = len(x1)
    if n_samples < 2:
        return np.nan, np.nan

    if sample_size is not None and n_samples > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, sample_size, replace=False)
        x1 = x1[idx]
        x2 = x2[idx]
        n_samples = sample_size

    k = min(k, n_samples - 1)
    if k <= 0:
        return np.nan, np.nan

    nne = NearestNeighbors(n_neighbors=k + 1)
    nne.fit(x1)
    kmatrix_1 = nne.kneighbors_graph(x1) - scipy.sparse.identity(n_samples)
    nne.fit(x2)
    kmatrix_2 = nne.kneighbors_graph(x2) - scipy.sparse.identity(n_samples)

    m1 = kmatrix_1.toarray()
    m2 = kmatrix_2.toarray()

    spearman_correlation = scipy.stats.spearmanr(m1.flatten(), m2.flatten())[0]

    set_1 = set(np.where(m1.flatten() == 1)[0])
    set_2 = set(np.where(m2.flatten() == 1)[0])

    if len(set_1) == 0 or len(set_2) == 0:
        fold_enrichment = np.nan
    else:
        fold_enrichment = (
            len(set_1.intersection(set_2)) * n_samples**2 / (float(len(set_1)) * len(set_2))
        )

    return spearman_correlation, fold_enrichment


def _detect_label_key(adata: Any) -> Optional[str]:
    """Detect label column name in AnnData obs."""
    if not hasattr(adata, 'obs'):
        return None
    candidates = [
        'cell_type', 'celltype', 'cell_types',
        'Group', 'labels', 'labels_encoded', 'label',
        'original_celltype', 'cluster', 'clusters', 'assigned_cluster'
    ]
    for key in candidates:
        if key in adata.obs.columns:
            return key
    return None


def _get_expression_matrix_for_pca(adata: Any) -> Any:
    """Pick a reasonable matrix for PCA (prefer raw counts if available)."""
    # Prefer raw counts layers if present
    for layer in ['counts', 'raw_counts', 'original_X', 'X_raw']:
        if hasattr(adata, 'layers') and layer in adata.layers:
            return adata.layers[layer]
    # Fall back to adata.raw if available
    if hasattr(adata, 'raw') and adata.raw is not None:
        return adata.raw.X
    return adata.X


def _matrix_looks_like_raw_counts(X: Any, sample_size: int = 4096) -> bool:
    """Heuristic guard to avoid normalizing/log-transforming already processed matrices."""
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
    pca_key: str = 'X_pca_batch'
) -> Optional[float]:
    """
    Compute Jaccard index between kNN graphs from per-batch PCA and embeddings.
    Matches scIB-E definition (mean over batches).
    """
    try:
        import scanpy as sc
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        logger.warning(f"Jaccard index skipped (missing dependency): {e}")
        return None

    if not hasattr(adata, 'obs') or batch_key not in adata.obs.columns:
        return None
    if not hasattr(adata, 'obsm') or embedding_key not in adata.obsm:
        return None

    # Use precomputed batch PCA if available
    if hasattr(adata, 'obsm') and pca_key in adata.obsm:
        X_pca_batch = adata.obsm[pca_key]
    else:
        # Compute PCA per batch (as in scIB-E preprocessing)
        X_pca_batch = np.zeros((adata.n_obs, n_pcs), dtype=np.float32)
        batches = adata.obs[batch_key].unique()
        for batch in batches:
            mask = adata.obs[batch_key] == batch
            if np.sum(mask) < 2:
                continue
            try:
                adata_batch = adata[mask].copy()
                X_mat = _get_expression_matrix_for_pca(adata_batch)
                adata_batch.X = X_mat.copy() if hasattr(X_mat, "copy") else X_mat
                if _matrix_looks_like_raw_counts(adata_batch.X):
                    sc.pp.normalize_total(adata_batch, target_sum=1e4)
                    sc.pp.log1p(adata_batch)

                n_obs_batch, n_vars_batch = adata_batch.shape
                n_comps_eff = min(int(n_pcs), int(n_obs_batch) - 1, int(n_vars_batch) - 1)
                if n_comps_eff < 2:
                    continue

                sc.pp.pca(adata_batch, n_comps=n_comps_eff, svd_solver='arpack')
                x_batch = np.asarray(adata_batch.obsm['X_pca'], dtype=np.float32)
                X_pca_batch[mask, :x_batch.shape[1]] = x_batch
            except Exception as e:
                logger.warning(f"Jaccard index skipped for batch {batch}: {e}")
                continue

    embeddings = adata.obsm[embedding_key]
    all_ja = []
    batches = adata.obs[batch_key].unique()
    for batch in batches:
        mask = adata.obs[batch_key] == batch
        n_batch = int(np.sum(mask))
        if n_batch < 2:
            continue
        X_pca = X_pca_batch[mask]
        X_emb = embeddings[mask]

        k_eff = min(k, n_batch - 1)
        if k_eff <= 0:
            continue

        nbrs1 = NearestNeighbors(n_neighbors=k_eff, metric='euclidean').fit(X_pca)
        A1 = nbrs1.kneighbors_graph(X_pca, k_eff, mode='connectivity')

        nbrs2 = NearestNeighbors(n_neighbors=k_eff, metric='euclidean').fit(X_emb)
        A2 = nbrs2.kneighbors_graph(X_emb, k_eff, mode='connectivity')

        A1.setdiag(0)
        A2.setdiag(0)
        A1 = A1.maximum(A1.T)
        A2 = A2.maximum(A2.T)

        intersection = A1.multiply(A2)
        union = A1.maximum(A2)

        if union.nnz == 0:
            continue
        jaccard_index = (intersection.nnz) / (union.nnz)
        all_ja.append(jaccard_index)

    if not all_ja:
        return None
    return float(np.mean(all_ja))


def _extract_scib_metric(row: Any, columns: List[str], canonical: str, aliases: List[str]) -> Optional[float]:
    """Extract a metric value using canonical name or aliases."""
    for name in [canonical] + aliases:
        if name in columns:
            try:
                val = row[name]
                if val is None:
                    return None
                return float(val)
            except Exception:
                return None
    return None


def _compute_exact_neighbors(X: np.ndarray, n_neighbors: int, n_jobs: int = 1):
    """Compute exact kNN structures for direct scIB metric fallback."""
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
    """Convert one scIB output to a finite float when possible."""
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
    except Exception as e:
        logger.warning(f"PCR comparison PCA preparation failed: {e}")
        return None

    if hasattr(adata, "obsm") and "X_pca" in adata.obsm:
        return np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    return None


def _compute_scib_neighbor_sets(embeddings: np.ndarray, n_jobs: int) -> Dict[int, Any]:
    """Compute the 15/50/90-neighbor structures expected by direct scIB metrics."""
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
    """Stable KBET fallback that avoids the fragile UMAP/numba connectivity path."""
    import pandas as pd
    try:
        import scib_metrics
    except ImportError:
        return None

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
        except Exception as e:
            logger.warning(f"scIB KBET fallback failed for label {clus}: {e}")
            score = 0.0

        score_f = _safe_scib_float(score)
        if score_f is not None:
            scores.append(score_f)

    if not scores:
        return None
    return float(np.nanmean(scores))


def _compute_scib_metrics_direct(
    adata: Any,
    embeddings: np.ndarray,
    batch_key: Optional[str],
    label_key: Optional[str],
    n_jobs: int = 1,
    compute_jaccard: bool = True,
    jaccard_k: int = 15,
    jaccard_pcs: int = 50,
) -> Dict[str, float]:
    """Robust direct scIB/scIB-E computation used when Benchmarker is incomplete or fails."""
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

    try:
        import scib_metrics
    except ImportError:
        return {}

    labels = np.asarray(adata_scib.obs[label_key].to_numpy())
    batches = np.asarray(adata_scib.obs[batch_key].to_numpy())

    try:
        neighbors = _compute_scib_neighbor_sets(embeddings, n_jobs=max(1, int(n_jobs)))
    except Exception as e:
        logger.warning(f"scIB metrics skipped during neighbor preparation: {e}", exc_info=True)
        return {}

    metrics: Dict[str, float] = {}

    def _record_metric(metric_name: str, compute_fn) -> Optional[float]:
        try:
            value = compute_fn()
        except Exception as e:
            logger.warning(f"scIB metric {metric_name} skipped: {e}")
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
    except Exception as e:
        logger.warning(f"scIB metric nmi_ari_cluster_labels_kmeans skipped: {e}")
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
    except Exception as e:
        logger.warning(
            "scIB metric KBET primary path failed (%s); retrying with exact within-label neighbors.",
            e,
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


def compute_scib_metrics(
    adata: Any,
    embeddings: np.ndarray,
    batch_key: Optional[str],
    label_key: Optional[str],
    n_jobs: int = 1,
    min_max_scale: bool = False,
    compute_jaccard: bool = True,
    jaccard_k: int = 15,
    jaccard_pcs: int = 50
) -> Dict[str, float]:
    """
    Compute scIB/scIB-E metrics using scib_metrics.Benchmarker.
    Returns a dict with scIB-E metric names.
    """
    if adata is None or embeddings is None:
        return {}

    disable_env = os.getenv("SCRB_DISABLE_SCIB_METRICS", "").strip().lower()
    if disable_env in {"1", "true", "yes", "on"}:
        _get_scib_benchmarker()
        return {}

    if batch_key is None or label_key is None:
        return {}

    # Make a safe copy to avoid mutating caller adata
    try:
        adata_scib = adata.copy()
    except Exception:
        return {}

    if len(embeddings) != adata_scib.n_obs:
        logger.warning(
            f"scIB metrics skipped: embeddings length ({len(embeddings)}) "
            f"!= adata.n_obs ({adata_scib.n_obs})"
        )
        return {}

    if batch_key not in adata_scib.obs.columns:
        return {}
    if label_key not in adata_scib.obs.columns:
        return {}

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        n_bad = int(np.size(embeddings) - np.isfinite(embeddings).sum())
        logger.warning("scIB metrics skipped: embeddings contain %d non-finite values.", n_bad)
        return {}

    direct_metrics_needed = False
    metrics: Dict[str, float] = {}
    embed_key = "__scib_embed"
    adata_scib.obsm[embed_key] = embeddings

    Benchmarker = _get_scib_benchmarker()
    benchmarker_compatible = (
        Benchmarker is not None and _scib_benchmarker_supports_current_sparse_indexing()
    )
    if benchmarker_compatible:
        try:
            bm = Benchmarker(
                adata_scib,
                batch_key=batch_key,
                label_key=label_key,
                embedding_obsm_keys=[embed_key],
                n_jobs=n_jobs,
            )
            bm.benchmark()
            results = bm.get_results(min_max_scale=min_max_scale)

            if embed_key in results.index:
                row = results.loc[embed_key]
            else:
                row = results.iloc[0]

            columns = list(results.columns)
            metric_aliases = {
                "Silhouette batch": ["ASW batch", "silhouette_batch", "Silhouette_batch", "BRAS"],
                "iLISI": ["iLISI", "ilsi", "iLISI_batch"],
                "KBET": ["kBET", "KBET", "kbet"],
                "Graph connectivity": ["Graph connectivity", "graph_conn", "graph_connectivity"],
                "Isolated labels": ["Isolated labels", "isolated_labels", "isolated_label"],
                "KMeans NMI": ["KMeans NMI", "NMI_cluster", "KMeans_NMI"],
                "KMeans ARI": ["KMeans ARI", "ARI_cluster", "KMeans_ARI"],
                "Silhouette label": ["Silhouette label", "ASW label", "silhouette_label"],
                "cLISI": ["cLISI", "clisi", "cLISI_label"],
                "PCR comparison": ["PCR comparison", "PCR", "pcr_comparison"],
            }

            for canonical, aliases in metric_aliases.items():
                val = _extract_scib_metric(row, columns, canonical, aliases)
                if val is not None:
                    metrics[canonical] = val
        except Exception as e:
            logger.warning(f"scIB Benchmarker failed, falling back to direct metric computation: {e}")
            direct_metrics_needed = True
    else:
        if Benchmarker is not None:
            logger.info(
                "Using direct scIB metric APIs because this scipy/scib-metrics "
                "combination does not support Benchmarker sparse masks."
            )
        direct_metrics_needed = True

    if compute_jaccard:
        jaccard = compute_jaccard_index(
            adata_scib,
            batch_key=batch_key,
            embedding_key=embed_key,
            k=jaccard_k,
            n_pcs=jaccard_pcs
        )
        if jaccard is not None:
            metrics["Jaccard index"] = float(jaccard)

    # Aggregate category scores (scIB-E)
    def _mean_available(keys: List[str]) -> Optional[float]:
        vals = [metrics.get(k) for k in keys if metrics.get(k) is not None and not np.isnan(metrics.get(k))]
        if not vals:
            return None
        return float(np.mean(vals))

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
        metrics["scIB-E Total score"] = (
            0.2 * batch_score + 0.4 * inter_score + 0.4 * intra_score
        )

    if direct_metrics_needed or any(
        key not in metrics
        for key in (
            "Batch correction",
            "Inter cell-type conservation",
            "Intra cell-type conservation",
            "scIB-E Total score",
        )
    ):
        direct_metrics = _compute_scib_metrics_direct(
            adata=adata_scib,
            embeddings=embeddings,
            batch_key=batch_key,
            label_key=label_key,
            n_jobs=n_jobs,
            compute_jaccard=compute_jaccard,
            jaccard_k=jaccard_k,
            jaccard_pcs=jaccard_pcs,
        )
        for key, value in direct_metrics.items():
            metrics.setdefault(key, value)

    return metrics


def compute_balanced_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute balanced metrics (Macro F1, Balanced Accuracy) to account for rare cell types.
    Requires aligning clusters to ground truth first.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels

    Returns:
        Dictionary with 'F1_Macro' and 'BalancedACC'
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score

    # Align predictions to truth for valid supervised comparison
    aligned_pred = align_labels(labels_true, labels_pred)
    
    # Check if alignment returned valid labels (no "Unmatched_" strings or mixed types that fail scoring)
    # The align_labels returns strings if input was string.
    # scikit-learn metrics handle string labels fine as long as they match classes.
    
    # Filter out unmatched clusters from scoring if they don't map to any true class?
    # Actually, if a prediction is "Unmatched_...", it counts as an error for whatever the true label was.
    # So we don't need to filter, just let sklearn handle the mismatch.
    
    return {
        'F1_Macro': f1_score(labels_true, aligned_pred, average='macro'),
        'BalancedACC': balanced_accuracy_score(labels_true, aligned_pred)
    }


def compute_rare_class_accuracy(
    labels_true: np.ndarray, 
    labels_pred: np.ndarray, 
    threshold: float = 0.05
) -> Optional[float]:
    """
    Compute accuracy ONLY for samples belonging to rare classes.
    Rare classes are defined as those with frequency < threshold.
    
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        threshold: Frequency threshold (default 0.05 for 5%)
        
    Returns:
        Accuracy on rare classes, or None if no rare classes exist.
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    # Align predictions first (Global alignment strategy)
    aligned_pred = align_labels(labels_true, labels_pred)
    
    # Identify rare classes
    unique_classes, counts = np.unique(labels_true, return_counts=True)
    total_samples = len(labels_true)
    freqs = counts / total_samples
    
    rare_classes = unique_classes[freqs < threshold]
    
    if len(rare_classes) == 0:
        return None
        
    # Create mask for samples belonging to rare classes
    # Use broadcasting for efficiency instead of list comprehension if possible, 
    # but unique_classes might be strings.
    # np.isin is robust for this.
    rare_mask = np.isin(labels_true, rare_classes)
    
    if np.sum(rare_mask) == 0:
        return None
        
    # Compute accuracy on this subset
    # Since we already aligned predictions globally, we just check equality
    matches = (labels_true[rare_mask] == aligned_pred[rare_mask])
    
    return float(np.mean(matches))


def compute_balanced_rare_class_accuracy(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """
    Compute balanced accuracy restricted to rare ground-truth classes.

    Rare classes are defined by their frequency in ``labels_true``. Predictions
    are aligned globally first, then the metric averages recall across rare
    classes so that a rare class with many cells does not dominate the score.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        threshold: Frequency threshold (default 0.05 for 5%)

    Returns:
        Balanced rare accuracy, or NaN if no rare classes exist.
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    if len(labels_true) != len(labels_pred) or len(labels_true) == 0:
        return float("nan")

    aligned_pred = align_labels(labels_true, labels_pred)
    unique_classes, counts = np.unique(labels_true, return_counts=True)
    rare_classes = unique_classes[(counts / len(labels_true)) < threshold]

    if len(rare_classes) == 0:
        return float("nan")

    recalls = []
    for cls in rare_classes:
        mask = labels_true == cls
        if np.any(mask):
            recalls.append(float(np.mean(aligned_pred[mask] == labels_true[mask])))

    return float(np.mean(recalls)) if recalls else float("nan")


def compute_class_wise_metrics(labels_true: np.ndarray, labels_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics (Precision, Recall, F1) to identify discrepancies 
    on specific key cell types (especially rare ones).

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels

    Returns:
        Dictionary mapping class_name -> {precision, recall, f1, support}
    """
    from sklearn.metrics import precision_recall_fscore_support

    # Align predictions
    aligned_pred = align_labels(labels_true, labels_pred)
    
    # Get unique classes from TRUE labels (we care about performance on known biology)
    classes = np.unique(labels_true)
    
    # Compute metrics
    # labels=classes ensures we get a result for every true class, even if not predicted
    p, r, f, s = precision_recall_fscore_support(labels_true, aligned_pred, labels=classes, zero_division=0)
    
    results = {}
    for i, cls in enumerate(classes):
        results[str(cls)] = {
            'Precision': float(p[i]),
            'Recall': float(r[i]),
            'F1': float(f[i]),
            'Support': int(s[i])
        }
        
    return results


def compute_silhouette(data: np.ndarray, labels: np.ndarray,
                      sample_size: Optional[int] = 5000) -> float:
    """
    Compute Silhouette Score.

    Args:
        data: Data matrix or embeddings
        labels: Cluster labels
        sample_size: Sample size for large datasets (None = use all)

    Returns:
        Silhouette score in [-1, 1]
    """
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2:
        return 0.0

    # SAFETY CHECK: Ensure data and labels have the same number of samples
    if len(data) != len(labels):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Silhouette skipped: shape mismatch between data ({len(data)}) and labels ({len(labels)})")
        return 0.0

    if sample_size is not None and len(labels) > sample_size:
        idx = np.random.choice(len(labels), sample_size, replace=False)
        data = data[idx]
        labels = labels[idx]

    return silhouette_score(data, labels)


def compute_metrics(labels_true: Optional[np.ndarray],
                   labels_pred: np.ndarray,
                   embeddings: Optional[np.ndarray] = None,
                   data: Optional[np.ndarray] = None,
                   filter_noise: bool = True,
                   neighbor_overlap_k: int = 100,
                   neighbor_overlap_sample_size: Optional[int] = 2000,
                   compute_neighbor_overlap: bool = True,
                   adata: Optional[Any] = None,
                   batch_key: Optional[str] = None,
                   label_key: Optional[str] = None,
                   compute_scib: bool = True,
                   scib_n_jobs: int = 1) -> Dict[str, float]:
    """
    Compute all available metrics.

    Args:
        labels_true: Ground truth labels (optional)
        labels_pred: Predicted labels
        embeddings: Latent embeddings for silhouette (optional)
        data: Original data for silhouette if no embeddings (optional)
        filter_noise: If True, filter out samples with noise labels (-1, 'noise')
                     before computing metrics. Default True.
        neighbor_overlap_k: k for nearest-neighbor overlap (NNO)
        neighbor_overlap_sample_size: Optional subsample size for NNO
        compute_neighbor_overlap: If True, compute NNO when embeddings and data are provided
        adata: Optional AnnData for scIB metrics
        batch_key: Optional batch column in adata.obs for scIB metrics
        label_key: Optional label column in adata.obs for scIB metrics
        compute_scib: If True, compute scIB/scIB-E metrics when possible
        scib_n_jobs: Number of jobs for scib_metrics.Benchmarker

    Returns:
        Dictionary of metric names to values (includes UCA, KNN_Purity, NNO when applicable)
    """
    metrics = {}

    # Count original samples and noise
    n_total = len(labels_pred)
    labels_pred_filtered = np.array(labels_pred)
    labels_true_filtered = np.array(labels_true) if labels_true is not None else None
    embeddings_filtered = embeddings
    data_filtered = data

    # Filter noise samples if requested
    if filter_noise:
        if labels_true is not None:
            # Use data if available (so we can filter it), else embeddings
            filter_data = data if data is not None else embeddings
            labels_true_filtered, labels_pred_filtered, filter_data, valid_mask = _filter_noise_samples(
                labels_true, labels_pred, filter_data
            )
            if data is not None:
                data_filtered = filter_data
                if embeddings is not None and valid_mask is not None:
                    embeddings_filtered = np.asarray(embeddings)[valid_mask]
            elif embeddings is not None:
                embeddings_filtered = filter_data
        else:
            # Only filter predicted labels for silhouette
            filter_data = data if data is not None else embeddings
            _, labels_pred_filtered, filter_data, valid_mask = _filter_noise_samples(
                labels_pred, labels_pred, filter_data  # Use pred as both to filter its noise
            )
            if data is not None:
                data_filtered = filter_data
                if embeddings is not None and valid_mask is not None:
                    embeddings_filtered = np.asarray(embeddings)[valid_mask]
            elif embeddings is not None:
                embeddings_filtered = filter_data

    # Count noise samples filtered
    n_noise = n_total - len(labels_pred_filtered)
    if n_noise > 0:
        metrics['n_noise_filtered'] = n_noise

    # Supervised metrics (require ground truth)
    if labels_true_filtered is not None and len(labels_true_filtered) > 0:
        metrics['NMI'] = compute_nmi(labels_true_filtered, labels_pred_filtered)
        metrics['ARI'] = compute_ari(labels_true_filtered, labels_pred_filtered)
        metrics['ACC'] = compute_accuracy(labels_true_filtered, labels_pred_filtered)
        metrics['UCA'] = compute_unsupervised_clustering_accuracy(
            labels_true_filtered, labels_pred_filtered
        )[0]
        
        # New Balanced Metrics for Rare Cell Types
        bal_metrics = compute_balanced_metrics(labels_true_filtered, labels_pred_filtered)
        metrics.update(bal_metrics)
        
        rare_acc = compute_rare_class_accuracy(labels_true_filtered, labels_pred_filtered)
        if rare_acc is not None:
            metrics['RareACC'] = rare_acc
        metrics['BalancedRareACC'] = compute_balanced_rare_class_accuracy(
            labels_true_filtered,
            labels_pred_filtered,
        )

        # KNN purity (requires embeddings + labels)
        if embeddings_filtered is not None and len(embeddings_filtered) == len(labels_true_filtered):
            try:
                metrics['KNN_Purity'] = compute_knn_purity(
                    embeddings_filtered, labels_true_filtered
                )
            except Exception as e:
                logger.warning(f"KNN Purity skipped: {e}")
        
        # Store detailed class-wise metrics (useful for deep dive, maybe not shown in main summary table)
        metrics['ClassWise'] = compute_class_wise_metrics(labels_true_filtered, labels_pred_filtered)

    # Unsupervised metrics
    if embeddings_filtered is not None and len(labels_pred_filtered) > 1:
        # Silhouette requires at least 2 samples
        n_unique = len(np.unique(labels_pred_filtered))
        if n_unique > 1 and n_unique < len(labels_pred_filtered):
            metrics['Silhouette'] = compute_silhouette(embeddings_filtered, labels_pred_filtered)
    elif data_filtered is not None and len(labels_pred_filtered) > 1:
        n_unique = len(np.unique(labels_pred_filtered))
        if n_unique > 1 and n_unique < len(labels_pred_filtered):
            metrics['Silhouette'] = compute_silhouette(data_filtered, labels_pred_filtered)

    # Representation overlap metrics (requires both embeddings and data)
    if compute_neighbor_overlap and embeddings_filtered is not None and data_filtered is not None:
        if len(embeddings_filtered) == len(data_filtered):
            try:
                spearman, fold = compute_nearest_neighbor_overlap(
                    embeddings_filtered,
                    data_filtered,
                    k=neighbor_overlap_k,
                    sample_size=neighbor_overlap_sample_size
                )
                metrics['NNO_Spearman'] = spearman
                metrics['NNO_FoldEnrichment'] = fold
            except Exception as e:
                logger.warning(f"NNO skipped: {e}")
        else:
            logger.warning(
                f"NNO skipped: embeddings/data length mismatch "
                f"({len(embeddings_filtered)} vs {len(data_filtered)})"
            )

    # Additional info (from unfiltered predictions to show actual clustering result)
    metrics['n_clusters_found'] = len(np.unique(labels_pred))
    metrics['n_samples_evaluated'] = len(labels_pred_filtered)

    # scIB/scIB-E metrics (requires AnnData + embeddings + batch/label keys)
    if compute_scib and adata is not None and embeddings is not None:
        try:
            # Detect keys if not provided
            if batch_key is None:
                from .dataset_splitter import get_batch_column
                batch_key = get_batch_column(adata)

            if label_key is None:
                label_key = _detect_label_key(adata)

            # If labels_true provided but label_key missing, add to a copy
            adata_scib = adata
            if label_key is None and labels_true is not None:
                adata_scib = adata.copy()
                label_key = "__labels__"
                adata_scib.obs[label_key] = np.asarray(labels_true)
            elif label_key is not None and label_key not in getattr(adata, 'obs', {}):
                if labels_true is not None:
                    adata_scib = adata.copy()
                    adata_scib.obs[label_key] = np.asarray(labels_true)
                else:
                    adata_scib = adata

            scib_metrics = compute_scib_metrics(
                adata=adata_scib,
                embeddings=embeddings,
                batch_key=batch_key,
                label_key=label_key,
                n_jobs=scib_n_jobs,
                min_max_scale=False
            )
            metrics.update(scib_metrics)
        except Exception as e:
            logger.warning(f"scIB metrics skipped: {e}")

    return metrics


def align_labels(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Permute labels_pred to best match labels_true (using Hungarian algorithm).
    Used for visualization alignment.
    
    Returns:
        Mapped predicted labels that correspond to labels_true classes.
    """
    # Ensure arrays
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    # Use string keys for robust matching while preserving original label types
    true_str = labels_true.astype(str)
    pred_str = labels_pred.astype(str)

    unique_true_str, true_first_idx = np.unique(true_str, return_index=True)
    unique_pred_str = np.unique(pred_str)

    map_true_idx = {s: i for i, s in enumerate(unique_true_str)}
    map_pred_idx = {s: i for i, s in enumerate(unique_pred_str)}

    y_true_int = np.array([map_true_idx[s] for s in true_str], dtype=np.int64)
    y_pred_int = np.array([map_pred_idx[s] for s in pred_str], dtype=np.int64)

    # Sizes might differ (e.g. true has 3 clusters, pred has 5)
    dim = max(len(unique_true_str), len(unique_pred_str))
    w = np.zeros((dim, dim), dtype=np.int64)

    for p, t in zip(y_pred_int, y_true_int):
        w[p, t] += 1

    # Solve assignment problem (max weight matching)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Map from pred_int -> true_int
    map_dict = {r: c for r, c in zip(row_ind, col_ind)}

    # Convert prediction INTs to aligned INTs
    aligned_pred_int = np.array([map_dict.get(x, x) for x in y_pred_int], dtype=np.int64)

    n_classes = len(unique_true_str)
    valid_mask = aligned_pred_int < n_classes

    # Preserve original true label types in the output
    true_values = labels_true[true_first_idx]

    def _choose_numeric_sentinel(values: np.ndarray, dtype: np.dtype) -> Union[int, float]:
        if values.size == 0:
            return -1 if np.issubdtype(dtype, np.integer) else -1.0

        existing = set(np.unique(values).tolist())

        if np.issubdtype(dtype, np.integer):
            candidate = -1
            if candidate not in existing and candidate not in NOISE_LABELS:
                return candidate
            candidate = int(np.max(values)) + 1
            while candidate in existing:
                candidate += 1
            return candidate

        # Floating labels are unusual but supported
        candidate = float(np.min(values)) - 1.0
        while candidate in existing:
            candidate -= 1.0
        return candidate

    if np.issubdtype(labels_true.dtype, np.number):
        aligned_labels = np.empty(labels_true.shape, dtype=labels_true.dtype)
        if np.any(valid_mask):
            aligned_labels[valid_mask] = true_values[aligned_pred_int[valid_mask]]
        if np.any(~valid_mask):
            sentinel = _choose_numeric_sentinel(true_values, labels_true.dtype)
            aligned_labels[~valid_mask] = sentinel
    else:
        # Use object dtype to avoid truncation of longer unmatched labels
        aligned_labels = np.empty(labels_true.shape, dtype=object)
        if np.any(valid_mask):
            aligned_labels[valid_mask] = true_values[aligned_pred_int[valid_mask]]
        if np.any(~valid_mask):
            for idx in np.where(~valid_mask)[0]:
                aligned_labels[idx] = f"Unmatched_Cluster_{aligned_pred_int[idx]}"

    return aligned_labels


# =============================================================================
# Benchmark Mode Metrics - Group-wise evaluation
# =============================================================================

def _compute_global_alignment(
    labels_true: np.ndarray,
    labels_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """
    Compute global alignment between predicted and true labels using Hungarian algorithm.

    This ensures consistent cluster-to-label mapping across all groups/batches.

    Args:
        labels_true: Ground truth labels (can be strings or integers)
        labels_pred: Predicted labels (can be strings or integers)

    Returns:
        Tuple of:
        - y_true_idx: True labels encoded as integers
        - y_pred_aligned: Predicted labels aligned to true label space
        - alignment_map: Dictionary mapping pred_idx -> true_idx
    """
    # Encode labels to integers (handles strings and mixed types)
    unique_true = np.unique(labels_true.astype(str))
    unique_pred = np.unique(labels_pred.astype(str))

    map_true = {l: i for i, l in enumerate(unique_true)}
    map_pred = {l: i for i, l in enumerate(unique_pred)}

    y_true_idx = np.array([map_true[str(l)] for l in labels_true])
    y_pred_idx = np.array([map_pred[str(l)] for l in labels_pred])

    # Build co-occurrence matrix w[pred, true]
    dim = max(len(unique_true), len(unique_pred))
    w = np.zeros((dim, dim), dtype=np.int64)

    for t, p in zip(y_true_idx, y_pred_idx):
        w[p, t] += 1

    # Solve assignment problem (maximize overlap)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    alignment_map = {r: c for r, c in zip(row_ind, col_ind)}

    # Apply global alignment to predictions
    y_pred_aligned = np.array([alignment_map.get(p, -1) for p in y_pred_idx])

    return y_true_idx, y_pred_aligned, alignment_map


def _compute_acc_with_alignment(
    y_true_idx: np.ndarray,
    y_pred_aligned: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute accuracy using pre-aligned predictions.

    Args:
        y_true_idx: True labels as integer indices
        y_pred_aligned: Predicted labels aligned to true label space
        mask: Optional boolean mask to select subset of samples

    Returns:
        Accuracy score in [0, 1]
    """
    if mask is not None:
        y_true_idx = y_true_idx[mask]
        y_pred_aligned = y_pred_aligned[mask]

    if len(y_true_idx) == 0:
        return np.nan

    return float(np.mean(y_true_idx == y_pred_aligned))


def compute_metrics_by_group(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    group_ids: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    data: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each group (batch/dataset source).

    This is useful for identifying which batches/datasets contribute
    most to prediction errors in the benchmark mode.

    IMPORTANT: ACC is computed using GLOBAL alignment across all samples,
    then evaluated per group. This ensures consistent cluster-to-label
    mapping across groups and avoids optimistic per-group alignment.
    NMI and ARI are permutation-invariant and computed per group.

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        group_ids: Group identifiers (e.g., batch names)
        embeddings: Optional embeddings for silhouette score
        data: Optional data matrix for silhouette if no embeddings

    Returns:
        Dictionary mapping group_name -> {metric_name -> value}
        (includes KNN_Purity when embeddings are provided)

    Example:
        >>> metrics_by_batch = compute_metrics_by_group(
        ...     y_true, y_pred, batch_ids
        ... )
        >>> print(metrics_by_batch['Pancreas1']['NMI'])
        0.85
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    group_ids = np.array(group_ids)

    unique_groups = np.unique(group_ids)
    results = {}

    # --- GLOBAL ALIGNMENT FOR ACC ---
    # Compute alignment using ALL samples to ensure consistent mapping across groups
    # This prevents optimistic per-group alignment where each group gets its own
    # best-case cluster-to-label mapping
    y_true_idx, y_pred_aligned, _ = _compute_global_alignment(labels_true, labels_pred)

    for group in unique_groups:
        mask = group_ids == group
        n_samples = np.sum(mask)

        if n_samples == 0:
            continue

        group_true = labels_true[mask]
        group_pred = labels_pred[mask]

        group_metrics = {}

        # Compute supervised metrics
        try:
            # NMI and ARI are permutation-invariant, compute per group
            group_metrics['NMI'] = compute_nmi(group_true, group_pred)
            group_metrics['ARI'] = compute_ari(group_true, group_pred)

            # ACC uses GLOBAL alignment to ensure consistent mapping across groups
            group_metrics['ACC'] = _compute_acc_with_alignment(
                y_true_idx, y_pred_aligned, mask
            )
        except Exception as e:
            logger.warning(f"Error computing metrics for group {group}: {e}")
            group_metrics['NMI'] = np.nan
            group_metrics['ARI'] = np.nan
            group_metrics['ACC'] = np.nan

        # Compute silhouette if data available
        if embeddings is not None:
            group_emb = embeddings[mask]
            if len(np.unique(group_pred)) >= 2:
                try:
                    group_metrics['Silhouette'] = compute_silhouette(
                        group_emb, group_pred, sample_size=None
                    )
                except Exception:
                    group_metrics['Silhouette'] = np.nan
            else:
                group_metrics['Silhouette'] = np.nan

            # KNN Purity (requires embeddings + true labels)
            if len(group_emb) > 1 and len(group_true) == len(group_emb):
                try:
                    group_metrics['KNN_Purity'] = compute_knn_purity(
                        group_emb, group_true
                    )
                except Exception:
                    group_metrics['KNN_Purity'] = np.nan
        elif data is not None:
            group_data = data[mask]
            if len(np.unique(group_pred)) >= 2:
                try:
                    group_metrics['Silhouette'] = compute_silhouette(
                        group_data, group_pred, sample_size=None
                    )
                except Exception:
                    group_metrics['Silhouette'] = np.nan
            else:
                group_metrics['Silhouette'] = np.nan

        # Additional info
        group_metrics['n_samples'] = int(n_samples)
        group_metrics['n_clusters_true'] = len(np.unique(group_true))
        group_metrics['n_clusters_pred'] = len(np.unique(group_pred))

        results[str(group)] = group_metrics

    return results


def compute_error_analysis(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    group_ids: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze where prediction errors come from.

    Returns detailed information about:
    - Which groups have the most errors
    - Which cell types are most confused
    - Confusion patterns between groups

    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted labels
        group_ids: Group identifiers

    Returns:
        Dictionary with error analysis results
    """
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    group_ids = np.array(group_ids)

    # Align predictions to true labels for fair comparison
    aligned_pred = align_labels(labels_true, labels_pred)

    # Overall error rate
    errors = labels_true != aligned_pred
    overall_error_rate = np.mean(errors)

    # Error rate by group
    error_by_group = {}
    unique_groups = np.unique(group_ids)
    for group in unique_groups:
        mask = group_ids == group
        if np.sum(mask) > 0:
            group_error_rate = np.mean(errors[mask])
            error_by_group[str(group)] = {
                'error_rate': float(group_error_rate),
                'n_errors': int(np.sum(errors[mask])),
                'n_samples': int(np.sum(mask))
            }

    # Error rate by cell type (true label)
    error_by_celltype = {}
    unique_celltypes = np.unique(labels_true)
    for ct in unique_celltypes:
        mask = labels_true == ct
        if np.sum(mask) > 0:
            ct_error_rate = np.mean(errors[mask])
            error_by_celltype[str(ct)] = {
                'error_rate': float(ct_error_rate),
                'n_errors': int(np.sum(errors[mask])),
                'n_samples': int(np.sum(mask))
            }

    # Error rate by cell type within each group (e.g., batch)
    error_by_celltype_by_group = {}
    for group in unique_groups:
        group_mask = group_ids == group
        if np.sum(group_mask) == 0:
            continue
        group_key = str(group)
        error_by_celltype_by_group[group_key] = {}
        for ct in unique_celltypes:
            ct_mask = labels_true == ct
            mask = group_mask & ct_mask
            if np.sum(mask) > 0:
                ct_error_rate = np.mean(errors[mask])
                error_by_celltype_by_group[group_key][str(ct)] = {
                    'error_rate': float(ct_error_rate),
                    'n_errors': int(np.sum(errors[mask])),
                    'n_samples': int(np.sum(mask))
                }

    # Most confused pairs
    confusion_pairs = []
    for ct_true in unique_celltypes:
        mask_true = labels_true == ct_true
        wrong_mask = mask_true & errors
        if np.sum(wrong_mask) > 0:
            wrong_preds = aligned_pred[wrong_mask]
            unique_wrong, counts = np.unique(wrong_preds, return_counts=True)
            for wrong_ct, count in zip(unique_wrong, counts):
                confusion_pairs.append({
                    'true': str(ct_true),
                    'predicted': str(wrong_ct),
                    'count': int(count),
                    'percentage': float(count / np.sum(mask_true))
                })

    # Sort by count
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    return {
        'overall_error_rate': float(overall_error_rate),
        'total_errors': int(np.sum(errors)),
        'total_samples': len(labels_true),
        'error_by_group': error_by_group,
        'error_by_celltype': error_by_celltype,
        'error_by_celltype_by_group': error_by_celltype_by_group,
        'top_confusion_pairs': confusion_pairs[:10]  # Top 10
    }


def compute_generalization_gap(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    metric_names: List[str] = None
) -> Dict[str, float]:
    """
    Compute the generalization gap between train and test metrics.

    The generalization gap indicates how much performance drops
    when moving from training to test data. A large gap suggests
    overfitting.

    Args:
        train_metrics: Metrics computed on training data
        test_metrics: Metrics computed on test data
        metric_names: Which metrics to compare (default: NMI, ARI, ACC)

    Returns:
        Dictionary with gap values (train - test) for each metric
    """
    if metric_names is None:
        metric_names = ['NMI', 'ARI', 'ACC', 'UCA']

    gaps = {}
    for metric in metric_names:
        if metric in train_metrics and metric in test_metrics:
            gap = train_metrics[metric] - test_metrics[metric]
            gaps[f'{metric}_gap'] = float(gap)
            gaps[f'{metric}_train'] = float(train_metrics[metric])
            gaps[f'{metric}_test'] = float(test_metrics[metric])
            
    # Also compute gaps for additional metrics if available
    extra_metrics = [
        'F1_Macro', 'BalancedACC', 'RareACC', 'BalancedRareACC', 'KNN_Purity',
        'NNO_Spearman', 'NNO_FoldEnrichment',
        # scIB / scIB-E metrics
        'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
        'Isolated labels', 'KMeans NMI', 'KMeans ARI',
        'Silhouette label', 'cLISI', 'PCR comparison',
        'Jaccard index',
        'Batch correction', 'Inter cell-type conservation',
        'Intra cell-type conservation', 'scIB-E Total score',
    ]
    for metric in extra_metrics:
        if metric in train_metrics and metric in test_metrics:
            gap = train_metrics[metric] - test_metrics[metric]
            gaps[f'{metric}_gap'] = float(gap)
            gaps[f'{metric}_train'] = float(train_metrics[metric])
            gaps[f'{metric}_test'] = float(test_metrics[metric])

    return gaps


def aggregate_group_metrics(
    group_metrics: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Aggregate group-wise metrics into overall metrics.

    Args:
        group_metrics: Metrics per group from compute_metrics_by_group
        weights: Optional weights per group (default: sample-weighted)

    Returns:
        Aggregated metrics (weighted mean, std, min, max per metric)
    """
    if not group_metrics:
        return {}

    # Collect all metric names
    metric_names = set()
    for group_data in group_metrics.values():
        metric_names.update(group_data.keys())

    # Remove non-numeric fields
    metric_names -= {'n_samples', 'n_clusters_true', 'n_clusters_pred'}

    # Use sample counts as default weights
    if weights is None:
        weights = {
            g: data.get('n_samples', 1)
            for g, data in group_metrics.items()
        }

    total_weight = sum(weights.values())
    if total_weight == 0:
        total_weight = 1

    aggregated = {}
    for metric in metric_names:
        values = []
        w = []
        for group, data in group_metrics.items():
            if metric in data and not np.isnan(data[metric]):
                values.append(data[metric])
                w.append(weights.get(group, 1))

        if values:
            values = np.array(values)
            w = np.array(w)
            w_sum = w.sum()

            aggregated[f'{metric}_mean'] = float(np.average(values, weights=w))
            aggregated[f'{metric}_std'] = float(np.sqrt(
                np.average((values - np.average(values, weights=w))**2, weights=w)
            ))
            aggregated[f'{metric}_min'] = float(np.min(values))
            aggregated[f'{metric}_max'] = float(np.max(values))

    return aggregated


# =============================================================================
# Benchmark comparison utilities
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """Container for benchmark mode metrics."""
    train_metrics: Dict[str, float]
    val_metrics: Optional[Dict[str, float]]
    test_metrics: Dict[str, float]
    train_by_group: Optional[Dict[str, Dict[str, float]]] = None
    test_by_group: Optional[Dict[str, Dict[str, float]]] = None
    generalization_gap: Optional[Dict[str, float]] = None
    error_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'test_metrics': self.test_metrics,
            'train_by_group': self.train_by_group,
            'test_by_group': self.test_by_group,
            'generalization_gap': self.generalization_gap,
            'error_analysis': self.error_analysis
        }


def compute_benchmark_metrics(
    labels_true_train: np.ndarray,
    labels_pred_train: np.ndarray,
    labels_true_test: np.ndarray,
    labels_pred_test: np.ndarray,
    group_ids_train: Optional[np.ndarray] = None,
    group_ids_test: Optional[np.ndarray] = None,
    labels_true_val: Optional[np.ndarray] = None,
    labels_pred_val: Optional[np.ndarray] = None,
    embeddings_train: Optional[np.ndarray] = None,
    embeddings_test: Optional[np.ndarray] = None,
    embeddings_val: Optional[np.ndarray] = None,
    data_train: Optional[np.ndarray] = None,
    data_test: Optional[np.ndarray] = None,
    data_val: Optional[np.ndarray] = None,
    adata_train: Optional[Any] = None,
    adata_test: Optional[Any] = None,
    adata_val: Optional[Any] = None,
    compute_scib: bool = True,
    compute_neighbor_overlap: bool = True
) -> BenchmarkMetrics:
    """
    Compute comprehensive benchmark metrics for train/val/test evaluation.

    Args:
        labels_true_train: Ground truth for training set
        labels_pred_train: Predictions on training set
        labels_true_test: Ground truth for test set
        labels_pred_test: Predictions on test set
        group_ids_train: Optional group IDs for training samples
        group_ids_test: Optional group IDs for test samples
        labels_true_val: Optional ground truth for validation set
        labels_pred_val: Optional predictions on validation set
        embeddings_train: Optional embeddings for training
        embeddings_test: Optional embeddings for test
        embeddings_val: Optional embeddings for validation
        data_train: Optional original data matrix for training (for NNO)
        data_test: Optional original data matrix for test (for NNO)
        data_val: Optional original data matrix for validation (for NNO)
        adata_train: Optional AnnData for scIB metrics (train)
        adata_test: Optional AnnData for scIB metrics (test)
        adata_val: Optional AnnData for scIB metrics (val)

    Returns:
        BenchmarkMetrics object with all computed metrics
    """
    # Compute basic metrics
    train_metrics = compute_metrics(
        labels_true_train, labels_pred_train, embeddings_train, data_train,
        adata=adata_train, compute_scib=compute_scib, 
        compute_neighbor_overlap=compute_neighbor_overlap
    )
    test_metrics = compute_metrics(
        labels_true_test, labels_pred_test, embeddings_test, data_test,
        adata=adata_test, compute_scib=compute_scib,
        compute_neighbor_overlap=compute_neighbor_overlap
    )

    val_metrics = None
    if labels_true_val is not None and labels_pred_val is not None:
        val_metrics = compute_metrics(
            labels_true_val, labels_pred_val, embeddings_val, data_val,
            adata=adata_val, compute_scib=compute_scib,
            compute_neighbor_overlap=compute_neighbor_overlap
        )

    # Compute group-wise metrics
    train_by_group = None
    test_by_group = None

    if group_ids_train is not None:
        train_by_group = compute_metrics_by_group(
            labels_true_train, labels_pred_train, group_ids_train,
            embeddings_train, data_train
        )

    if group_ids_test is not None:
        test_by_group = compute_metrics_by_group(
            labels_true_test, labels_pred_test, group_ids_test,
            embeddings_test, data_test
        )

    # Compute generalization gap
    gen_gap = compute_generalization_gap(train_metrics, test_metrics)

    # Error analysis on test set
    error_analysis = None
    if group_ids_test is not None:
        error_analysis = compute_error_analysis(
            labels_true_test, labels_pred_test, group_ids_test
        )

    return BenchmarkMetrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_by_group=train_by_group,
        test_by_group=test_by_group,
        generalization_gap=gen_gap,
        error_analysis=error_analysis
    )


# =============================================================================
# Marker-Overlap Annotation (DEG-based cell type assignment)
# =============================================================================

def marker_overlap_annotation(
    adata,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    n_top_genes: int = 100,
    method: str = 'wilcoxon',
) -> Dict[str, Any]:
    """
    Annotate predicted clusters using marker gene overlap with gold-standard.

    For each ground-truth cell type, the top `n_top_genes` DEGs are computed
    (gold-standard marker list). The same is done for each predicted cluster.
    The overlap score between a predicted cluster p and a gold-standard type g
    is: score(p, g) = |DEG_p ∩ DEG_g| / n_top_genes.

    Each predicted cluster is then assigned to the gold-standard type with the
    highest overlap score. Unlike Hungarian (1-to-1), this allows **N-to-1**
    mapping (multiple predicted clusters can map to the same gold-standard type,
    which detects cluster fragmentation).

    Reference: scCDCG benchmark methodology (marker-overlap annotation).

    Args:
        adata: AnnData object with expression data (normalized, log-transformed).
               Must have the same number of observations as labels_true/labels_pred.
        labels_true: Ground truth cell type labels (1D array, len = adata.n_obs).
        labels_pred: Predicted cluster labels (1D array, len = adata.n_obs).
        n_top_genes: Number of top DEGs per cluster to use (default: 100).
        method: Statistical test for DEG detection ('wilcoxon', 't-test').

    Returns:
        Dictionary with keys:
        - 'marker_labels': np.ndarray of marker-overlap annotations (same length as labels_pred)
        - 'overlap_matrix': pd.DataFrame (rows=predicted clusters, cols=gold-standard types)
        - 'gold_degs': Dict[str, List[str]] — top DEGs per gold-standard type
        - 'pred_degs': Dict[str, List[str]] — top DEGs per predicted cluster
        - 'hungarian_labels': np.ndarray of Hungarian annotations for comparison
    """
    import scanpy as sc
    import pandas as pd

    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)

    if len(labels_true) != adata.n_obs or len(labels_pred) != adata.n_obs:
        raise ValueError(
            f"Label lengths ({len(labels_true)}, {len(labels_pred)}) "
            f"must match adata.n_obs ({adata.n_obs})."
        )

    # --- Helper: compute top DEGs for each group in a label vector ---
    def _compute_degs_per_group(adata_work, labels, group_name='_group'):
        """Run rank_genes_groups for all groups and return top gene lists."""
        adata_copy = adata_work.copy()
        adata_copy.obs[group_name] = labels.astype(str)

        unique_groups = sorted(adata_copy.obs[group_name].unique())
        degs_dict = {}

        # Run DEG analysis for all groups at once
        try:
            sc.tl.rank_genes_groups(
                adata_copy,
                groupby=group_name,
                method=method,
                n_genes=n_top_genes,
                use_raw=False,
            )
        except Exception:
            # Fallback to t-test if wilcoxon fails (e.g., very small groups)
            sc.tl.rank_genes_groups(
                adata_copy,
                groupby=group_name,
                method='t-test',
                n_genes=n_top_genes,
                use_raw=False,
            )

        for grp in unique_groups:
            try:
                df = sc.get.rank_genes_groups_df(adata_copy, group=grp)
                gene_list = df['names'].head(n_top_genes).tolist()
                degs_dict[grp] = gene_list
            except Exception as e:
                logger.warning(f"Could not get DEGs for group {grp}: {e}")
                degs_dict[grp] = []

        return degs_dict

    # --- Step 1: Gold-standard DEGs (ground truth clusters) ---
    logger.info("Computing gold-standard DEGs from ground truth labels...")
    gold_degs = _compute_degs_per_group(adata, labels_true, '_gold')

    # --- Step 2: Predicted cluster DEGs ---
    logger.info("Computing DEGs from predicted clusters...")
    pred_degs = _compute_degs_per_group(adata, labels_pred, '_pred')

    # --- Step 3: Build overlap matrix ---
    gold_types = sorted(gold_degs.keys())
    pred_clusters = sorted(pred_degs.keys())

    overlap_data = np.zeros((len(pred_clusters), len(gold_types)))

    for i, pc in enumerate(pred_clusters):
        pred_genes = set(pred_degs.get(pc, []))
        for j, gt in enumerate(gold_types):
            gold_genes = set(gold_degs.get(gt, []))
            if len(pred_genes) > 0 and len(gold_genes) > 0:
                intersection = len(pred_genes & gold_genes)
                overlap_data[i, j] = intersection / n_top_genes
            else:
                overlap_data[i, j] = 0.0

    overlap_df = pd.DataFrame(
        overlap_data,
        index=pred_clusters,
        columns=gold_types,
    )
    overlap_df.index.name = 'Predicted Cluster'
    overlap_df.columns.name = 'Gold Standard Type'

    # --- Step 4: Assign each predicted cluster to best gold-standard type ---
    cluster_to_type = {}
    for i, pc in enumerate(pred_clusters):
        best_idx = np.argmax(overlap_data[i])
        best_score = overlap_data[i, best_idx]
        best_type = gold_types[best_idx]

        if best_score > 0:
            cluster_to_type[pc] = best_type
        else:
            cluster_to_type[pc] = f"Unknown_{pc}"

    logger.info(f"Marker-overlap mapping: {cluster_to_type}")

    # Build per-cell marker-overlap labels
    marker_labels = np.array(
        [cluster_to_type.get(str(lbl), f"Unknown_{lbl}") for lbl in labels_pred],
        dtype=object,
    )

    # --- Step 5: Hungarian labels for comparison ---
    hungarian_labels = align_labels(labels_true, labels_pred)

    return {
        'marker_labels': marker_labels,
        'overlap_matrix': overlap_df,
        'gold_degs': gold_degs,
        'pred_degs': pred_degs,
        'hungarian_labels': hungarian_labels,
        'cluster_to_type': cluster_to_type,
    }
