from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

LEIDEN_RESOLUTION_GRID: tuple[float, ...] = tuple(
    float(v) for v in np.round(np.linspace(0.0, 1.0, 21), 2)
)


def selection_rule_name() -> str:
    return "minimize_cluster_count_error_then_maximize_silhouette"


def _cluster_stratified_sample_indices(
    labels: np.ndarray,
    max_size: int,
    random_state: int,
) -> np.ndarray:
    """Subsample while keeping every cluster represented for silhouette scoring."""
    labs = np.asarray(labels, dtype=np.int64)
    n = int(labs.shape[0])
    max_size = int(max(2, max_size))
    if n <= max_size:
        return np.arange(n, dtype=np.int64)

    uniq = np.unique(labs)
    rng = np.random.RandomState(int(random_state))
    min_per_cluster = max(4, int(max_size // max(10, 2 * len(uniq))))

    picked: list[int] = []
    leftovers: list[np.ndarray] = []
    for clu in uniq:
        idx = np.where(labs == int(clu))[0]
        if idx.size <= min_per_cluster:
            picked.extend(idx.tolist())
            continue
        take = rng.choice(idx, size=min_per_cluster, replace=False)
        picked.extend(np.asarray(take, dtype=np.int64).tolist())
        leftovers.append(np.setdiff1d(idx, take, assume_unique=False))

    if len(picked) >= max_size:
        picked = rng.choice(np.asarray(picked, dtype=np.int64), size=max_size, replace=False).tolist()
        return np.sort(np.asarray(picked, dtype=np.int64))

    budget = int(max_size - len(picked))
    if budget > 0 and leftovers:
        pool = np.concatenate([arr for arr in leftovers if arr.size > 0], axis=0)
        if pool.size > 0:
            if pool.size > budget:
                extra = rng.choice(pool, size=budget, replace=False)
            else:
                extra = pool
            picked.extend(np.asarray(extra, dtype=np.int64).tolist())

    return np.sort(np.asarray(picked[:max_size], dtype=np.int64))


def compute_candidate_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: Optional[int] = 5000,
    random_state: int = 0,
) -> float:
    """Compute a deterministic silhouette score for one Leiden candidate."""
    from sklearn.metrics import silhouette_score

    emb = np.asarray(embeddings, dtype=np.float32)
    labs = np.asarray(labels, dtype=np.int64)
    if emb.ndim != 2 or emb.shape[0] != labs.shape[0] or np.unique(labs).size < 2:
        return float("nan")

    emb_eval = emb
    labs_eval = labs
    if sample_size is not None and emb.shape[0] > int(sample_size):
        idx = _cluster_stratified_sample_indices(
            labs,
            max_size=int(sample_size),
            random_state=int(random_state),
        )
        emb_eval = emb[idx]
        labs_eval = labs[idx]
        if np.unique(labs_eval).size < 2:
            return float("nan")

    try:
        return float(silhouette_score(emb_eval, labs_eval, metric="euclidean"))
    except Exception:
        return float("nan")


def select_best_leiden_candidate(
    candidates: Sequence[Mapping[str, Any]],
    expected_n_classes: int,
) -> Optional[Mapping[str, Any]]:
    """Pick an exact class-count match first, then the best silhouette."""
    if not candidates:
        return None

    expected = max(int(expected_n_classes), 1)

    def _sort_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, float, float, float, int]:
        idx, candidate = item
        try:
            n_clusters = int(candidate.get("n_clusters", 0))
        except Exception:
            n_clusters = 0
        try:
            resolution = float(candidate.get("resolution", float("inf")))
        except Exception:
            resolution = float("inf")
        try:
            silhouette = float(candidate.get("silhouette", float("nan")))
        except Exception:
            silhouette = float("nan")
        abs_error = abs(n_clusters - expected)
        exact_match_key = 0 if abs_error == 0 else 1
        silhouette_key = -silhouette if np.isfinite(silhouette) else float("inf")
        return (exact_match_key, abs_error, silhouette_key, resolution, idx)

    _, best = min(enumerate(candidates), key=_sort_key)
    return best


def rank_resolutions(metrics_df: pd.DataFrame, expected_n_classes: int) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df.copy()

    ranked = metrics_df.copy()
    expected = max(int(expected_n_classes), 1)

    silhouette = pd.to_numeric(ranked.get("Silhouette"), errors="coerce")
    n_clusters = pd.to_numeric(ranked.get("n_clusters_found"), errors="coerce")
    abs_error = (n_clusters - expected).abs()
    rel_error = abs_error / float(expected)

    ranked["expected_n_classes"] = expected
    ranked["cluster_count_abs_error"] = abs_error
    ranked["cluster_count_rel_error"] = rel_error
    ranked["has_expected_n_classes"] = abs_error.eq(0)

    selection_score = silhouette
    ranked["selection_score"] = selection_score

    ranked["_has_expected_n_classes_sort"] = np.where(abs_error.eq(0), 0, 1)
    ranked["_cluster_count_abs_error_sort"] = np.where(np.isfinite(abs_error), abs_error, np.inf)
    ranked["_silhouette_sort"] = np.where(np.isfinite(silhouette), -silhouette, np.inf)

    ranked = ranked.sort_values(
        ["_has_expected_n_classes_sort", "_cluster_count_abs_error_sort", "_silhouette_sort", "resolution"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return ranked.drop(
        columns=["_has_expected_n_classes_sort", "_cluster_count_abs_error_sort", "_silhouette_sort"],
        errors="ignore",
    )


def selection_metadata(expected_n_classes: int) -> dict[str, Any]:
    return {
        "rule": selection_rule_name(),
        "expected_n_classes": max(int(expected_n_classes), 1),
    }
