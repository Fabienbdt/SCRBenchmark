"""
Visualization utilities for SCRBenchmark.
Provides shared plotting functions for both Streamlit UI and CLI specific outputs.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

def plot_metrics_comparison(
    results_summary: Dict[str, Any],
    results_list: List[Any],
    show_stats: bool = True,
    stat_method: bool = 'nonparametric',
    alpha: float = 0.05,
    benchmark_split: str = 'test'
) -> plt.Figure:
    """
    Plot boxplots of metrics comparison with optional statistical significance.
    """
    # Import here to avoid circular dependencies if any
    from utils.statistics import compute_significance_groups
    from core.algorithm_registry import AlgorithmRegistry
    
    metrics = ['NMI', 'ARI', 'ACC', 'Silhouette']
    algo_names = list(results_summary.keys())
    
    # Filter valid metrics that exist in results
    valid_metrics = []
    
    # Helper to get metrics from a result object (handles both Standard and Benchmark modes)
    def _get_result_metrics(r):
        if hasattr(r, 'metrics'):
            return r.metrics
        elif hasattr(r, 'benchmark_metrics'):
            if benchmark_split == 'train':
                return r.benchmark_metrics.train_metrics
            elif benchmark_split == 'val':
                return r.benchmark_metrics.val_metrics or {}
            else: # test
                return r.benchmark_metrics.test_metrics
        return {}

    for m in metrics:
        # Check if any result has this metric
        if any(m in _get_result_metrics(r) for r in results_list):
            valid_metrics.append(m)
            
    if not valid_metrics:
        # Fallback to check if metrics are there but maybe the keys differ
        # (e.g. if list contains only BenchmarkResults)
        return None

    fig, axes = plt.subplots(1, len(valid_metrics), figsize=(4 * len(valid_metrics), 5))
    if len(valid_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(valid_metrics):
        ax = axes[idx]
        data_by_algo = {}
        data_list = []
        labels = []
        
        for algo_name in algo_names:
            algo_results = [r for r in results_list if r.algorithm_name == algo_name]
            values = []
            for r in algo_results:
                m_dict = _get_result_metrics(r)
                if metric in m_dict:
                    values.append(m_dict[metric])
            
            if values:
                try:
                    algo_info = AlgorithmRegistry.get(algo_name).get_info()
                    display_name = algo_info.display_name
                except:
                    display_name = algo_name
                    
                # Filter out None and NaN
                clean_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
                
                if clean_values:
                    # Use cleaned values for both plotting and stats
                    data_by_algo[display_name] = clean_values
                    data_list.append(clean_values)
                    labels.append(display_name)
        
        if data_list:
            bp = ax.boxplot(data_list, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            
            # Add CLD letters if enabled
            if show_stats and len(data_by_algo) >= 2:
                try:
                    cld, _, _ = compute_significance_groups(
                        data_by_algo, method=stat_method, alpha=alpha
                    )
                    
                    # Add letters above each boxplot
                    for i, label in enumerate(labels):
                        letter = cld.get(label, '')
                        if letter:
                            # Get the top of the boxplot data
                            # Note: data_list[i] is already cleaned in the loop above
                            box_data = data_list[i]
                            
                            if box_data:
                                current_max = max(box_data)
                            else:
                                current_max = ax.get_ylim()[1]
                                
                            y_pos = current_max + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                            
                            ax.text(i + 1, y_pos, letter,
                                   ha='center', va='bottom',
                                   fontsize=12, fontweight='bold',
                                   color='darkred')
                except Exception as e:
                    print(f"Warning: Failed to compute statistics for {metric}: {e}")
            
            ax.set_title(metric)
            ax.set_ylabel('Score')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust y-axis to make room for letters
            if show_stats:
                ylim = ax.get_ylim()
                ax.set_ylim(ylim[0], ylim[1] * 1.1)
                
    plt.tight_layout()
    return fig


def _compute_umap_or_2d(
    embeddings: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, bool]:
    """Return a 2D representation and whether we fell back to first 2 dimensions."""
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 2 or embeddings.shape[1] < 2:
        raise ValueError("Embeddings must be a 2D array with at least 2 columns.")

    if embeddings.shape[1] <= 2:
        return embeddings, False

    # Intermediate PCA if too many dimensions (standard Scanpy pipeline)
    if embeddings.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_comps = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=n_comps, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(embeddings), False
    except Exception:
        return embeddings[:, :2], True


def _select_snapshot_indices_for_gallery(
    snapshots: List[Dict[str, Any]],
    max_snapshots: int,
) -> List[int]:
    """
    Select representative snapshot indices for gallery-like displays.

    The selection preserves key milestones (start/end, phase transitions,
    warm-up end) and fills remaining slots with evenly spaced epochs.
    """
    n = len(snapshots)
    if n == 0:
        return []
    if max_snapshots <= 0 or n <= max_snapshots:
        return list(range(n))

    must_keep: set = {0, n - 1}

    # Preserve phase transitions and warm-up completion marker.
    prev_phase = str(snapshots[0].get("phase", ""))
    warmup_indices: List[int] = []
    for i, snap in enumerate(snapshots):
        phase = str(snap.get("phase", ""))
        if phase.lower().startswith("warm"):
            warmup_indices.append(i)
        if i > 0 and phase != prev_phase:
            must_keep.add(i - 1)
            must_keep.add(i)
        prev_phase = phase

    if warmup_indices:
        must_keep.add(max(warmup_indices))

    # Keep a few explicit weight-refresh milestones when present.
    refresh_indices = [
        i for i, snap in enumerate(snapshots)
        if snap.get("snapshot_type") == "weight_refresh"
    ]
    if 0 < len(refresh_indices) <= 3:
        must_keep.update(refresh_indices)
    elif len(refresh_indices) > 3:
        must_keep.update(
            [
                refresh_indices[0],
                refresh_indices[len(refresh_indices) // 2],
                refresh_indices[-1],
            ]
        )

    selected = sorted(i for i in must_keep if 0 <= i < n)
    if len(selected) >= max_snapshots:
        # If key points alone exceed capacity, downsample keys uniformly.
        key_positions = np.linspace(0, len(selected) - 1, num=max_snapshots)
        picked = sorted({selected[int(round(pos))] for pos in key_positions})
        return picked[:max_snapshots]

    # Fill the remaining slots with evenly spaced indices over the full range.
    selected_set = set(selected)
    fill_candidates = [
        int(round(pos))
        for pos in np.linspace(0, n - 1, num=max(max_snapshots * 3, n))
    ]
    for idx in fill_candidates:
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
            if len(selected) >= max_snapshots:
                break

    return sorted(selected)


def _compute_shared_umap_sequence(
    embeddings_per_snapshot: List[np.ndarray],
    random_state: int = 42,
) -> Tuple[List[np.ndarray], bool]:
    """
    Compute a shared 2D projection for a sequence of snapshot embeddings.

    Returns:
        (projected_list, used_fallback_to_first_2d)
    """
    if not embeddings_per_snapshot:
        return [], True

    arrays = [np.asarray(e) for e in embeddings_per_snapshot]
    if len(arrays) == 1:
        one, used_fb = _compute_umap_or_2d(arrays[0], random_state=random_state)
        return [one], used_fb

    # If dimensions are inconsistent, fallback to per-snapshot projection.
    first_dim = arrays[0].shape[1] if arrays[0].ndim == 2 else -1
    if first_dim < 2 or any(a.ndim != 2 or a.shape[1] != first_dim for a in arrays):
        out = []
        used_any_fallback = False
        for emb in arrays:
            emb2d, used_fb = _compute_umap_or_2d(emb, random_state=random_state)
            out.append(emb2d)
            used_any_fallback = used_any_fallback or used_fb
        return out, used_any_fallback

    common = np.vstack(arrays)
    transformed_arrays = arrays
    pca_model = None
    if common.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_comps = min(50, common.shape[0] - 1, common.shape[1])
        pca_model = PCA(n_components=n_comps, random_state=random_state)
        common = pca_model.fit_transform(common)
        transformed_arrays = [pca_model.transform(emb) for emb in arrays]

    if common.shape[1] <= 2:
        return [emb[:, :2] for emb in transformed_arrays], False

    try:
        import umap

        # Shared projection improves panel-to-panel comparability.
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        reducer.fit(common)
        out = [reducer.transform(emb) for emb in transformed_arrays]
        return out, False
    except Exception:
        # Robust fallback to the first two dimensions of the shared transformed space.
        return [emb[:, :2] for emb in transformed_arrays], True


def _encode_labels(labels: np.ndarray) -> Tuple[np.ndarray, List[Any], Dict[Any, int]]:
    """Encode arbitrary labels to integer IDs for plotting."""
    labels_arr = np.asarray(labels, dtype=object)
    sentinel = "__MISSING__"
    normalized = np.array([sentinel if pd.isna(x) else x for x in labels_arr], dtype=object)

    if isinstance(labels, pd.Series) and isinstance(labels.dtype, pd.CategoricalDtype):
        unique_labels = [sentinel if pd.isna(x) else x for x in labels.cat.categories]
        # Ensure unseen non-categorical values are still encoded safely
        extra = [x for x in pd.unique(normalized) if x not in unique_labels]
        unique_labels.extend(extra)
    else:
        unique_labels = list(pd.unique(normalized))

    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    encoded = np.array([label_map[x] for x in normalized])
    return encoded, unique_labels, label_map


def _decode_label_name(label: Any, label_names: Optional[Dict[int, str]]) -> str:
    """Map numeric labels back to readable names when a map is available."""
    if label == "__MISSING__":
        return "NA"
    label_text = str(label)
    if label_text.startswith("Unmatched_Cluster_"):
        return label_text.replace("Unmatched_Cluster_", "Unmatched cluster ")
    if label_names is None:
        return label_text

    try:
        key = int(float(label))
        if key in label_names:
            return str(label_names[key])
    except (ValueError, TypeError):
        pass
    return label_text


def _tag_unmatched_predicted_labels(
    predicted_labels: np.ndarray,
    label_names: Optional[Dict[int, str]]
) -> np.ndarray:
    """Mark predicted labels not present in label_names as unmatched clusters.

    This avoids ambiguous legends such as a bare numeric label (e.g. "14")
    when true classes are 0..13 and an extra predicted cluster appears.
    """
    if not label_names:
        return np.asarray(predicted_labels, dtype=object)

    known_ids = set()
    for k in label_names.keys():
        try:
            known_ids.add(int(k))
        except (TypeError, ValueError):
            continue

    out = []
    for raw in np.asarray(predicted_labels, dtype=object):
        txt = str(raw)
        if txt.startswith("Unmatched_Cluster_"):
            out.append(txt)
            continue
        try:
            idx = int(float(txt))
        except (TypeError, ValueError):
            out.append(raw)
            continue
        if idx in known_ids:
            out.append(raw)
        else:
            out.append(f"Unmatched_Cluster_{idx}")
    return np.asarray(out, dtype=object)


def _draw_cluster_overlays(
    ax: plt.Axes,
    points_2d: np.ndarray,
    labels: np.ndarray,
    outline_mode: str = "ellipse",
    n_std: float = 1.8,
    show_centroids: bool = False,
    centroid_label_names: Optional[Dict[int, str]] = None,
):
    """Draw optional cluster outlines and/or center labels for UMAP readability."""
    from matplotlib.patches import Ellipse, Polygon

    labels_arr = np.asarray(labels)
    _, unique_labels, _ = _encode_labels(labels_arr)
    n_labels = len(unique_labels)
    cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
    norm = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    mode = (outline_mode or "ellipse").lower()
    if mode in ("hull", "convex"):
        mode = "convex_hull"
    if mode not in {"none", "ellipse", "convex_hull", "density"}:
        mode = "ellipse"

    top_for_density = set(unique_labels)
    if mode == "density" and n_labels > 16:
        counts = pd.Series(labels_arr).value_counts()
        top_for_density = set(counts.index[:16].tolist())

    for idx, group_label in enumerate(unique_labels):
        mask = labels_arr == group_label
        points = points_2d[mask]
        if len(points) < 3:
            continue

        color = cmap(norm(idx))
        try:
            if mode == "ellipse":
                mean = np.mean(points, axis=0)
                cov = np.cov(points.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                width = 2 * n_std * np.sqrt(max(eigenvalues[0], 1e-3))
                height = 2 * n_std * np.sqrt(max(eigenvalues[1], 1e-3))

                ax.add_patch(
                    Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor=color,
                        edgecolor='none',
                        alpha=0.10,
                    )
                )
                ax.add_patch(
                    Ellipse(
                        xy=mean,
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor='none',
                        edgecolor=color,
                        linewidth=1.8,
                        alpha=0.70,
                    )
                )
            elif mode == "convex_hull":
                from scipy.spatial import ConvexHull

                hull = ConvexHull(points)
                hull_pts = points[hull.vertices]
                ax.add_patch(
                    Polygon(
                        hull_pts,
                        closed=True,
                        facecolor=color,
                        edgecolor='none',
                        alpha=0.10,
                    )
                )
                ax.add_patch(
                    Polygon(
                        hull_pts,
                        closed=True,
                        facecolor='none',
                        edgecolor=color,
                        linewidth=1.8,
                        alpha=0.75,
                    )
                )
            elif mode == "density":
                if group_label in top_for_density and len(points) >= 20:
                    sns.kdeplot(
                        x=points[:, 0],
                        y=points[:, 1],
                        ax=ax,
                        levels=1,
                        color=color,
                        linewidths=1.5,
                        fill=False,
                        thresh=0.20,
                    )
        except Exception:
            continue

        if show_centroids:
            center = points.mean(axis=0)
            label_text = _decode_label_name(group_label, centroid_label_names)
            ax.text(
                center[0],
                center[1],
                label_text,
                fontsize=7,
                ha='center',
                va='center',
                color='black',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'),
            )


def plot_umap_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Projection",
    show_legend: bool = True,
    save_path: Optional[str] = None,
    predicted_labels: Optional[np.ndarray] = None,
    show_cluster_hulls: bool = True,
    label_names: Optional[Dict[int, str]] = None,
    cluster_outline_mode: str = "none",
    show_cluster_centroids: bool = True,
    show_label_centroids: bool = False,
    projection_name: str = "UMAP",
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> plt.Figure:
    """
    Plot embeddings with automatic UMAP reduction and optional cluster overlays.

    Args:
        embeddings: N-dimensional embeddings (auto-reduced to 2D via UMAP if >2D).
        labels: Labels for coloring points (typically ground truth).
        title: Plot title.
        show_legend: Whether to show a text legend.
        save_path: Optional path to save the figure.
        predicted_labels: Optional predicted cluster labels for overlays.
        show_cluster_hulls: Backward-compatible flag to enable/disable overlays.
        label_names: Optional numeric-label decoding map.
        cluster_outline_mode: One of {'ellipse', 'convex_hull', 'density', 'none'}.
        show_cluster_centroids: If True, annotate overlay centroids.
        show_label_centroids: If True, annotate centroids for `labels` (typically ground truth).
        projection_name: Label prefix for axes/titles (e.g., 'UMAP', 'Latent 2D').
    """
    embeddings_2d, used_fallback = _compute_umap_or_2d(np.asarray(embeddings))
    if used_fallback and embeddings_2d.shape[1] == 2 and np.asarray(embeddings).shape[1] > 2:
        title += " (first 2 dims, no UMAP)"

    fig, ax = plt.subplots(figsize=(10, 8))

    encoded, unique_labels, _ = _encode_labels(labels)
    n_labels = len(unique_labels)
    cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
    norm = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=encoded,
        cmap=cmap,
        norm=norm,
        s=2,
        alpha=0.7,
    )

    if show_cluster_hulls and predicted_labels is not None:
        pred_arr = np.asarray(predicted_labels)
        if len(pred_arr) == len(embeddings_2d):
            _draw_cluster_overlays(
                ax=ax,
                points_2d=embeddings_2d,
                labels=pred_arr,
                outline_mode=cluster_outline_mode,
                n_std=2.0,
                show_centroids=show_cluster_centroids,
                centroid_label_names=label_names,
            )

    # Optional centroid labels for the main color labels (e.g. ground truth).
    if show_label_centroids:
        labels_arr = np.asarray(labels)
        if len(labels_arr) == len(embeddings_2d):
            _draw_cluster_overlays(
                ax=ax,
                points_2d=embeddings_2d,
                labels=labels_arr,
                outline_mode="none",
                n_std=2.0,
                show_centroids=True,
                centroid_label_names=label_names,
            )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{projection_name} 1")
    ax.set_ylabel(f"{projection_name} 2")

    if show_legend:
        if n_labels <= 30:
            handles = []
            for i, lbl in enumerate(unique_labels):
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker='o',
                        color='w',
                        markerfacecolor=cmap(norm(i)),
                        markersize=6,
                        label=_decode_label_name(lbl, label_names),
                    )
                )

            ax.legend(
                handles=handles,
                fontsize=8,
                markerscale=1.5,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                borderaxespad=0,
                framealpha=0.9,
            )
        else:
            ax.text(
                0.02,
                0.02,
                f"({n_labels} clusters — legend hidden)",
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.7,
            )

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_umap_comparison(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    algorithm_name: str = "Algorithm",
    save_path: Optional[str] = None,
    n_std: float = 1.8,
    label_names: Optional[Dict[int, str]] = None,
    outline_mode: str = "none",
    show_cluster_centroids: bool = True,
    projection_name: str = "UMAP",
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> plt.Figure:
    """
    Plot side-by-side UMAP comparison: Ground Truth vs Predicted Clusters.

    Args:
        embeddings: N-dimensional embeddings (auto-reduced to 2D via UMAP if >2D).
        true_labels: Ground truth labels (cell types).
        predicted_labels: Predicted cluster labels from algorithm.
        algorithm_name: Name of the algorithm for the title.
        save_path: Optional path to save the figure.
        n_std: Number of standard deviations for ellipse size (used in ellipse mode).
        label_names: Optional dict mapping numeric labels to text names.
        outline_mode: One of {'ellipse', 'convex_hull', 'density', 'none'}.
        show_cluster_centroids: If True, annotate cluster centers.
        projection_name: Label prefix for axes/titles (e.g., 'UMAP', 'Latent 2D').
    """
    from .metrics import align_labels

    embeddings = np.asarray(embeddings)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)

    try:
        predicted_labels = align_labels(true_labels, predicted_labels)
    except Exception:
        pass
    predicted_labels = _tag_unmatched_predicted_labels(predicted_labels, label_names)

    embeddings_2d, _ = _compute_umap_or_2d(embeddings)

    # --- Shared color mapping across Ground Truth and Predicted panels ---
    _UNMATCHED_PREFIX = "Unmatched_Cluster_"
    _UNMATCHED_COLOR = "#888888"

    true_str = np.asarray(true_labels, dtype=object).astype(str)
    pred_str = np.asarray(predicted_labels, dtype=object).astype(str)

    # Real labels = true labels first (defines colour order), then any extra
    # matched predicted labels not already seen.
    _true_unique = list(pd.unique(true_str))
    _pred_unmatched = set(x for x in pd.unique(pred_str) if str(x).startswith(_UNMATCHED_PREFIX))
    _pred_matched_extra = [x for x in pd.unique(pred_str)
                           if not str(x).startswith(_UNMATCHED_PREFIX) and x not in _true_unique]
    _real_unique: List[Any] = list(_true_unique) + list(_pred_matched_extra)
    _real_map: Dict[Any, int] = {lbl: i for i, lbl in enumerate(_real_unique)}
    _n_real = len(_real_unique)
    _shared_cmap = plt.cm.tab20 if _n_real <= 20 else plt.cm.gist_ncar
    _shared_norm = plt.Normalize(vmin=0, vmax=max(_n_real - 1, 1))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    def _draw_panel(ax, labels, panel_title, use_label_names=False, custom_label_names=None):
        labels_arr = np.asarray(labels, dtype=object).astype(str)
        target_map = custom_label_names if custom_label_names is not None else (label_names if use_label_names else None)

        is_unmatched = np.array([str(x) in _pred_unmatched for x in labels_arr], dtype=bool)
        matched_mask = ~is_unmatched

        # Plot matched points with shared colormap.
        if np.any(matched_mask):
            encoded = np.array([_real_map.get(str(x), 0) for x in labels_arr[matched_mask]])
            ax.scatter(
                embeddings_2d[matched_mask, 0],
                embeddings_2d[matched_mask, 1],
                c=encoded,
                cmap=_shared_cmap,
                norm=_shared_norm,
                s=3,
                alpha=0.8,
            )
        # Plot unmatched points distinctly.
        if np.any(is_unmatched):
            ax.scatter(
                embeddings_2d[is_unmatched, 0],
                embeddings_2d[is_unmatched, 1],
                c=_UNMATCHED_COLOR,
                s=5,
                alpha=0.9,
                marker="x",
                linewidths=0.5,
            )

        if outline_mode != "none" or show_cluster_centroids:
            _draw_cluster_overlays(
                ax=ax,
                points_2d=embeddings_2d,
                labels=labels_arr,
                outline_mode=outline_mode,
                n_std=n_std,
                show_centroids=show_cluster_centroids,
                centroid_label_names=target_map,
            )

        ax.set_title(panel_title, fontsize=13, fontweight='bold')
        ax.set_xlabel(f"{projection_name} 1", fontsize=10)
        ax.set_ylabel(f"{projection_name} 2", fontsize=10)

        # Legend: show only labels present in this panel.
        present = set(labels_arr)
        all_labels_ordered = list(_real_unique) + sorted(_pred_unmatched)
        n_present = sum(1 for lbl in all_labels_ordered if lbl in present)
        if n_present <= 30:
            handles = []
            for lbl in all_labels_ordered:
                if lbl not in present:
                    continue
                if str(lbl) in _pred_unmatched:
                    handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker='x',
                            color=_UNMATCHED_COLOR,
                            markerfacecolor=_UNMATCHED_COLOR,
                            markersize=7,
                            linestyle='None',
                            label=_decode_label_name(lbl, target_map),
                        )
                    )
                else:
                    i = _real_map.get(str(lbl), 0)
                    handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker='o',
                            color='w',
                            markerfacecolor=_shared_cmap(_shared_norm(i)),
                            markersize=7,
                            label=_decode_label_name(lbl, target_map),
                        )
                    )

            ax.legend(
                handles=handles,
                fontsize=7,
                markerscale=1.2,
                bbox_to_anchor=(1.01, 1),
                loc='upper left',
                borderaxespad=0,
                framealpha=0.9,
            )

    _draw_panel(axes[0], true_str, "Ground Truth (Cell Types)", use_label_names=True)
    _draw_panel(axes[1], predicted_labels, "Predicted Clusters (IDs)", use_label_names=False)
    _draw_panel(
        axes[2],
        pred_str,
        "Predicted (Aligned & Named)",
        use_label_names=True,
        custom_label_names=label_names,
    )

    fig.suptitle(f"{projection_name} Comparison: {algorithm_name}", fontsize=16, fontweight='bold', y=1.02)

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_umap_diagnostic(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
    algorithm_name: str = "Algorithm",
    outline_mode: str = "none",
    show_cluster_centroids: bool = True,
    max_points: int = 30000,
    random_state: int = 42,
    n_std: float = 1.8,
    save_path: Optional[str] = None,
    label_names: Optional[Dict[int, str]] = None,
    projection_name: str = "UMAP",
) -> plt.Figure:
    """
    Plot a 2x2 UMAP diagnostic view: true labels, predicted labels, batch, and errors.
    """
    from .metrics import align_labels

    embeddings = np.asarray(embeddings)
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    batch_arr = np.asarray(batch_labels) if batch_labels is not None else None

    n = min(len(embeddings), len(true_labels), len(predicted_labels))
    if batch_arr is not None:
        n = min(n, len(batch_arr))

    embeddings = embeddings[:n]
    true_labels = true_labels[:n]
    predicted_labels = predicted_labels[:n]
    if batch_arr is not None:
        batch_arr = batch_arr[:n]

    if max_points and n > max_points:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(n, size=max_points, replace=False))
        embeddings = embeddings[idx]
        true_labels = true_labels[idx]
        predicted_labels = predicted_labels[idx]
        if batch_arr is not None:
            batch_arr = batch_arr[idx]
        n = len(idx)

    try:
        predicted_aligned = np.asarray(align_labels(true_labels, predicted_labels))
    except Exception:
        predicted_aligned = predicted_labels
    predicted_aligned = _tag_unmatched_predicted_labels(predicted_aligned, label_names)

    embeddings_2d, used_fallback = _compute_umap_or_2d(embeddings, random_state=random_state)
    true_str = np.asarray(true_labels, dtype=object).astype(str)
    pred_str = np.asarray(predicted_aligned, dtype=object).astype(str)
    error_mask = true_str != pred_str

    # Build a shared label->color mapping so True Labels and Predicted Labels
    # panels use the same colour for the same cell-type name.
    # Separate "real" labels from unmatched clusters so we can colour them
    # distinctly (gray) instead of giving them a random colormap slot.
    _UNMATCHED_PREFIX = "Unmatched_Cluster_"
    _true_unique = list(pd.unique(true_str))
    _pred_matched = [x for x in pd.unique(pred_str) if not str(x).startswith(_UNMATCHED_PREFIX)]
    _pred_unmatched = [x for x in pd.unique(pred_str) if str(x).startswith(_UNMATCHED_PREFIX)]

    # Merge unique real labels (true first, then any extra matched predicted).
    _real_unique: List[Any] = list(_true_unique)
    for lbl in _pred_matched:
        if lbl not in _real_unique:
            _real_unique.append(lbl)

    _n_real = len(_real_unique)
    _real_map: Dict[Any, int] = {lbl: i for i, lbl in enumerate(_real_unique)}
    _real_cmap = plt.cm.tab20 if _n_real <= 20 else plt.cm.gist_ncar
    _real_norm = plt.Normalize(vmin=0, vmax=max(_n_real - 1, 1))

    # Unmatched cluster color: distinctive gray.
    _UNMATCHED_COLOR = "#888888"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def _scatter_panel(ax, labels, title, names_map=None, draw_outline=False,
                       shared_color_map=None):
        labels_obj = np.asarray(labels, dtype=object).astype(str)

        if shared_color_map is not None:
            cmap, norm, real_labels, label_map, unmatched_set = shared_color_map

            # Split into matched vs unmatched points.
            is_unmatched = np.array([str(x) in unmatched_set for x in labels_obj], dtype=bool)
            matched_mask = ~is_unmatched

            # Plot matched points with colormap.
            if np.any(matched_mask):
                encoded = np.array([label_map.get(str(x), 0) for x in labels_obj[matched_mask]])
                ax.scatter(
                    embeddings_2d[matched_mask, 0],
                    embeddings_2d[matched_mask, 1],
                    c=encoded,
                    cmap=cmap,
                    norm=norm,
                    s=3,
                    alpha=0.8,
                )
            # Plot unmatched points in distinctive gray.
            if np.any(is_unmatched):
                ax.scatter(
                    embeddings_2d[is_unmatched, 0],
                    embeddings_2d[is_unmatched, 1],
                    c=_UNMATCHED_COLOR,
                    s=5,
                    alpha=0.9,
                    marker="x",
                    linewidths=0.5,
                )

            n_labels = len(real_labels) + len(unmatched_set)
            unique_labels_all = list(real_labels) + sorted(unmatched_set)
        else:
            encoded, unique_labels_raw, label_map = _encode_labels(labels_obj)
            n_labels = len(unique_labels_raw)
            cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
            norm = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=encoded,
                cmap=cmap,
                norm=norm,
                s=3,
                alpha=0.8,
            )
            unique_labels_all = unique_labels_raw
            unmatched_set = set()

        if draw_outline:
            _draw_cluster_overlays(
                ax=ax,
                points_2d=embeddings_2d,
                labels=labels_obj,
                outline_mode=outline_mode,
                n_std=n_std,
                show_centroids=show_cluster_centroids,
                centroid_label_names=names_map,
            )
        if n_labels <= 30:
            present = set(labels_obj)
            handles = []
            for lbl in unique_labels_all:
                if lbl not in present:
                    continue
                if str(lbl) in unmatched_set:
                    handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker='x',
                            color=_UNMATCHED_COLOR,
                            markerfacecolor=_UNMATCHED_COLOR,
                            markersize=6,
                            linestyle='None',
                            label=_decode_label_name(lbl, names_map),
                        )
                    )
                else:
                    i = label_map.get(str(lbl), 0) if shared_color_map else label_map.get(lbl, 0)
                    handles.append(
                        plt.Line2D(
                            [0], [0],
                            marker='o',
                            color='w',
                            markerfacecolor=cmap(norm(i)),
                            markersize=6,
                            label=_decode_label_name(lbl, names_map),
                        )
                    )
            ax.legend(handles=handles, fontsize=7, loc='upper right', framealpha=0.9)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(f"{projection_name} 1")
        ax.set_ylabel(f"{projection_name} 2")

    _unmatched_set = set(str(x) for x in _pred_unmatched)
    _shared = (_real_cmap, _real_norm, _real_unique, _real_map, _unmatched_set)

    _scatter_panel(
        axes[0, 0],
        true_str,
        "True Labels",
        names_map=label_names,
        draw_outline=False,
        shared_color_map=_shared,
    )
    _scatter_panel(
        axes[0, 1],
        pred_str,
        "Predicted Labels",
        names_map=label_names,
        draw_outline=True,
        shared_color_map=_shared,
    )

    if batch_arr is not None:
        _scatter_panel(axes[1, 0], batch_arr, "Batch View", names_map=None, draw_outline=False)
    else:
        axes[1, 0].text(0.5, 0.5, "No batch labels available", ha='center', va='center', fontsize=12)
        axes[1, 0].set_axis_off()

    ax_err = axes[1, 1]
    if np.any(~error_mask):
        ax_err.scatter(
            embeddings_2d[~error_mask, 0],
            embeddings_2d[~error_mask, 1],
            c='#b8b8b8',
            s=2,
            alpha=0.35,
            label='Correct',
        )
    if np.any(error_mask):
        ax_err.scatter(
            embeddings_2d[error_mask, 0],
            embeddings_2d[error_mask, 1],
            c='#c12b2b',
            s=6,
            alpha=0.8,
            label='Misclassified',
        )
    error_rate = float(error_mask.mean()) if len(error_mask) else 0.0
    ax_err.set_title(f"Error Focus (error={error_rate:.1%})", fontsize=12, fontweight='bold')
    ax_err.set_xlabel(f"{projection_name} 1")
    ax_err.set_ylabel(f"{projection_name} 2")
    if np.any(error_mask):
        confusion_pairs = pd.Series(list(zip(true_str[error_mask], pred_str[error_mask]))).value_counts().head(3)
        pairs_text = "\n".join(
            [f"{a} -> {b}: {int(count)}" for (a, b), count in confusion_pairs.items()]
        )
        ax_err.text(
            0.02,
            0.02,
            f"Top confusions:\n{pairs_text}",
            transform=ax_err.transAxes,
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
        )
    ax_err.legend(loc='upper right', fontsize=8, framealpha=0.9)

    projection_note = " (first 2 dims fallback)" if used_fallback else ""
    fig.suptitle(
        f"{projection_name} Diagnostic - {algorithm_name}{projection_note} | n={n}",
        fontsize=15,
        fontweight='bold',
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_benchmark_comparison(results_summary: Dict[str, Any]) -> plt.Figure:
    """
    Plot Train vs Test performance comparison for benchmark results.
    """
    from core.algorithm_registry import AlgorithmRegistry
    
    metrics = ['NMI', 'ARI', 'ACC']
    algo_names = list(results_summary.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_vals = []
        test_vals = []
        labels = []
        
        for algo_name in algo_names:
            stats = results_summary[algo_name]
            train_key = f'train_{metric}_mean'
            test_key = f'test_{metric}_mean'
            
            if train_key in stats and test_key in stats:
                try:
                    algo_info = AlgorithmRegistry.get(algo_name).get_info()
                    display_name = algo_info.display_name
                except:
                    display_name = algo_name
                    
                labels.append(display_name)
                train_vals.append(stats[train_key])
                test_vals.append(stats[test_key])
                
        if labels:
            x = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color='steelblue')
            bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='coral')
            
            ax.set_ylabel('Score')
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar in bars1 + bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
                           
    plt.tight_layout()
    return fig


def plot_generalization_gap(results_summary: Dict[str, Any]) -> plt.Figure:
    """
    Plot generalization gap (Train - Test).
    """
    from core.algorithm_registry import AlgorithmRegistry
    
    metrics = ['NMI', 'ARI', 'ACC']
    algo_names = list(results_summary.keys())
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    gap_data = []
    for algo_name in algo_names:
        stats = results_summary[algo_name]
        try:
            algo_info = AlgorithmRegistry.get(algo_name).get_info()
            display_name = algo_info.display_name
        except:
            display_name = algo_name
            
        for metric in metrics:
            gap_key = f'{metric}_gap_mean'
            if gap_key in stats:
                gap_data.append({
                    'Algorithm': display_name,
                    'Metric': metric,
                    'Gap': stats[gap_key]
                })
                
    if gap_data:
        df_gap = pd.DataFrame(gap_data)
        df_pivot = df_gap.pivot(index='Algorithm', columns='Metric', values='Gap')
        
        df_pivot.plot(kind='bar', ax=ax, color=['steelblue', 'coral', 'seagreen'])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Gap (Train - Test)')
        ax.set_title('Generalization Gap by Algorithm\n(Lower is better, negative means test > train)')
        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Metric')
        
    plt.tight_layout()
    return fig

def plot_batch_metrics_heatmap(df_groups: pd.DataFrame, title: str = "Metrics by Batch") -> plt.Figure:
    """
    Plot heatmap of metrics by batch.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    metrics_to_show = ['NMI', 'ARI', 'ACC']
    # Ensure metrics exist in dataframe
    existing_metrics = [m for m in metrics_to_show if m in df_groups.columns]
    
    if not existing_metrics:
        plt.close(fig)
        return None
        
    heatmap_data = df_groups.set_index('Batch')[existing_metrics].astype(float)
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        ax=ax,
        vmin=0,
        vmax=1
    )
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_celltype_errors_by_batch(
    error_by_celltype_by_group: Dict[str, Dict[str, Dict[str, Any]]],
    title: str = "Error Rate by Cell Type and Batch"
) -> plt.Figure:
    """
    Plot heatmap of error rates by cell type (rows) and batch/group (columns).

    Args:
        error_by_celltype_by_group: Dict[batch][cell_type] -> {'error_rate': ...}
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if not error_by_celltype_by_group:
        ax.text(0.5, 0.5, 'No cell type error-by-batch data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    batches = sorted(error_by_celltype_by_group.keys())
    celltypes = sorted({ct for b in batches for ct in error_by_celltype_by_group[b].keys()})

    if not batches or not celltypes:
        ax.text(0.5, 0.5, 'No cell type error-by-batch data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    matrix = np.full((len(celltypes), len(batches)), np.nan, dtype=float)
    for j, batch in enumerate(batches):
        ct_map = error_by_celltype_by_group.get(batch, {})
        for i, ct in enumerate(celltypes):
            val = ct_map.get(ct)
            if isinstance(val, dict):
                val = val.get('error_rate', np.nan)
            if val is not None:
                try:
                    matrix[i, j] = float(val)
                except Exception:
                    matrix[i, j] = np.nan

    # Sort cell types and batches by mean error (descending)
    with np.errstate(invalid='ignore'):
        ct_means = np.nanmean(matrix, axis=1)
        batch_means = np.nanmean(matrix, axis=0)
    ct_order = np.argsort(ct_means)[::-1]
    batch_order = np.argsort(batch_means)[::-1]
    matrix = matrix[ct_order, :][:, batch_order]
    celltypes = [celltypes[i] for i in ct_order]
    batches = [batches[i] for i in batch_order]

    df_matrix = pd.DataFrame(matrix, index=celltypes, columns=batches)

    sns.heatmap(
        df_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Error rate'},
        annot_kws={'size': 9},
        mask=df_matrix.isna()
    )

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Batch', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cell type', fontweight='bold', fontsize=12)

    plt.tight_layout()
    return fig

def plot_top_genes(df_genes: pd.DataFrame, title: str = "Top Genes") -> plt.Figure:
    """
    Plot bar chart of top genes.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    top_10 = df_genes.head(10)
    
    sns.barplot(
        data=top_10, 
        x='Gene', 
        y='Mean Expression', 
        hue='Gene', 
        ax=ax, 
        palette='viridis', 
        legend=False
    )
    
    ax.set_title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_radar_chart(results_summary: Dict[str, Any]) -> plt.Figure:
    """
    Plot radar chart comparing multiple algorithms across standard metrics.
    """
    from core.algorithm_registry import AlgorithmRegistry
    
    metrics = ['NMI', 'ARI', 'ACC', 'Silhouette']
    labels = metrics
    num_vars = len(labels)
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + labels
    plt.xticks(angles[:-1], labels)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot each algorithm
    # Use tab10 color map which has distinct colors
    colors = plt.cm.tab10.colors
    
    for idx, (algo_name, stats) in enumerate(results_summary.items()):
        values = []
        for m in metrics:
            # Try exact match, then test_*_mean, then train_*_mean
            val = stats.get(f'{m}_mean')
            if val is None:
                val = stats.get(f'test_{m}_mean')
            if val is None:
                val = stats.get(f'train_{m}_mean', 0.0)
            values.append(val)
        
        # Close the loop
        values += values[:1]
        
        try:
            algo_info = AlgorithmRegistry.get(algo_name).get_info()
            display_name = algo_info.display_name
        except:
            display_name = algo_name
            
        color = colors[idx % len(colors)]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=display_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Algorithm Comparison (Mean Performance)', y=1.08)

    return fig


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    algorithm_name: str = "Algorithm",
    normalize: bool = True,
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plot confusion matrix comparing predicted clusters to true cell type labels.

    Args:
        true_labels: Array of ground truth labels
        predicted_labels: Array of predicted cluster labels
        algorithm_name: Name of the algorithm for the title
        normalize: If True, normalize by row (true labels)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment

    # Explicitly cast to strings to avoid "Mix of label input types" error in sklearn
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    if true_labels.size == 0 or predicted_labels.size == 0:
        return None

    if true_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError("True and predicted labels must have the same length.")

    true_labels = true_labels.astype(str)
    predicted_labels = predicted_labels.astype(str)

    cm = contingency_matrix(true_labels, predicted_labels)
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predicted_labels)

    # Optimal assignment using Hungarian algorithm to reorder predicted labels
    # This helps visualize the correspondence between clusters and cell types
    row_ind, col_ind = linear_sum_assignment(-cm)
    cm_reordered = cm[:, col_ind]
    pred_labels_reordered = unique_pred[col_ind]

    if normalize:
        # Normalize by row (each true label sums to 1)
        cm_normalized = cm_reordered.astype('float') / cm_reordered.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        cm_display = cm_normalized
        fmt = '.2f'
        title_suffix = " (Normalized)"
    else:
        cm_display = cm_reordered
        fmt = 'd'
        title_suffix = ""

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(
        xticks=np.arange(cm_display.shape[1]),
        yticks=np.arange(cm_display.shape[0]),
        xticklabels=pred_labels_reordered,
        yticklabels=unique_true,
        ylabel='True Labels (Cell Types)',
        xlabel='Predicted Clusters',
        title=f'Confusion Matrix: {algorithm_name}{title_suffix}'
    )

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            value = cm_display[i, j]
            if normalize:
                text = f'{value:.2f}'
            else:
                text = f'{int(value)}'
            ax.text(j, i, text,
                   ha="center", va="center",
                   color="white" if value > thresh else "black",
                   fontsize=8)

    fig.tight_layout()
    return fig


def plot_confusion_matrix_multi(
    results_list: List[Any],
    true_labels: np.ndarray,
    normalize: bool = True,
    max_algorithms: int = 4
) -> plt.Figure:
    """
    Plot confusion matrices for multiple algorithms in a grid.

    Args:
        results_list: List of algorithm results with .labels attribute
        true_labels: Ground truth labels
        normalize: If True, normalize matrices
        max_algorithms: Maximum number of algorithms to show

    Returns:
        Matplotlib figure with subplots
    """
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment
    from core.algorithm_registry import AlgorithmRegistry

    # Get unique algorithms
    true_labels = np.array(true_labels)
    if true_labels.size == 0:
        return None

    algo_results = {}
    for r in results_list:
        if r.algorithm_name not in algo_results:
            pred_labels = np.array(r.labels)
            if pred_labels.size == 0 or pred_labels.shape[0] != true_labels.shape[0]:
                continue
            algo_results[r.algorithm_name] = r

    algo_names = list(algo_results.keys())[:max_algorithms]
    n_algos = len(algo_names)

    if n_algos == 0:
        return None

    # Determine grid layout
    n_cols = min(2, n_algos)
    n_rows = (n_algos + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_algos == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    unique_true = np.unique(true_labels.astype(str))

    for idx, algo_name in enumerate(algo_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        result = algo_results[algo_name]
        pred_labels = result.labels

        # Explicitly cast to strings to avoid "Mix of label input types" error in sklearn
        true_labels_str = np.array(true_labels).astype(str)
        pred_labels_str = np.array(pred_labels).astype(str)

        # Compute confusion matrix
        cm = contingency_matrix(true_labels_str, pred_labels_str)
        unique_pred = np.unique(pred_labels_str)
        unique_true = np.unique(true_labels_str) # Refresh unique_true as strings for labeling

        # Reorder using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-cm)
        cm_reordered = cm[:, col_ind]
        pred_labels_reordered = unique_pred[col_ind] if len(col_ind) <= len(unique_pred) else unique_pred

        if normalize:
            cm_normalized = cm_reordered.astype('float') / cm_reordered.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)
            cm_display = cm_normalized
        else:
            cm_display = cm_reordered

        # Plot
        im = ax.imshow(cm_display, interpolation='nearest', cmap='Blues')

        # Get display name
        try:
            algo_info = AlgorithmRegistry.get(algo_name).get_info()
            display_name = algo_info.display_name
        except:
            display_name = algo_name

        ax.set_title(display_name, fontsize=10)
        ax.set_ylabel('True Labels')
        ax.set_xlabel('Predicted')
        ax.set_xticks(np.arange(len(pred_labels_reordered)))
        ax.set_xticklabels(pred_labels_reordered, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(unique_true)))
        ax.set_yticklabels(unique_true)

        # Smaller font for tick labels if many categories
        if len(unique_true) > 6:
            ax.tick_params(axis='both', labelsize=6)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Hide unused subplots
    for idx in range(n_algos, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    fig.suptitle('Confusion Matrices (Clusters vs True Labels)', fontsize=12)
    fig.tight_layout()
    return fig


def plot_volcano(df_markers: pd.DataFrame, 
                 title: str = "Volcano Plot", 
                 p_threshold: float = 0.05, 
                 fc_threshold: float = 1.0) -> plt.Figure:
    """
    Plot Volcano Plot (Log2FC vs -Log10(Adj P-value)).
    
    Args:
        df_markers: DataFrame with 'Log2FC', 'Adj P-value', 'Gene'
        title: Plot title
        p_threshold: Adjusted p-value threshold for significance
        fc_threshold: Log2 Fold Change threshold
        
    Returns:
        Matplotlib figure
    """
    if df_markers is None or df_markers.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate -log10(p-value)
    # Handle zeros by replacing with smallest non-zero
    min_pval = df_markers['Adj P-value'][df_markers['Adj P-value'] > 0].min()
    if pd.isna(min_pval):
        min_pval = 1e-300
        
    pvals = df_markers['Adj P-value'].replace(0, min_pval * 0.1)
    neg_log_p = -np.log10(pvals)
    
    # Define categories
    # Significant: P < thresh AND abs(FC) > thresh
    is_sig = (df_markers['Adj P-value'] < p_threshold) & (np.abs(df_markers['Log2FC']) > fc_threshold)
    
    # Up/Down
    is_up = is_sig & (df_markers['Log2FC'] > 0)
    is_down = is_sig & (df_markers['Log2FC'] < 0)
    is_ns = ~is_sig
    
    # Plot points
    ax.scatter(
        df_markers.loc[is_ns, 'Log2FC'], 
        neg_log_p.loc[is_ns], 
        c='grey', alpha=0.5, s=10, label='Not Significant'
    )
    
    ax.scatter(
        df_markers.loc[is_up, 'Log2FC'], 
        neg_log_p.loc[is_up], 
        c='firebrick', alpha=0.7, s=20, label='Upregulated'
    )
    
    ax.scatter(
        df_markers.loc[is_down, 'Log2FC'], 
        neg_log_p.loc[is_down], 
        c='steelblue', alpha=0.7, s=20, label='Downregulated'
    )
    
    # Add lines
    ax.axhline(-np.log10(p_threshold), color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(fc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(-fc_threshold, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Label top genes
    # Top 5 most significant up-regulated
    top_genes = df_markers[is_up].sort_values('Score', ascending=False).head(5)
    
    texts = []
    for _, row in top_genes.iterrows():
        idx = row.name
        texts.append(ax.text(
            row['Log2FC'], 
            neg_log_p.loc[idx], 
            row['Gene'],
            fontsize=8
        ))
    
    # Labels
    ax.set_xlabel("Log2 Fold Change")
    ax.set_ylabel("-Log10(Adjusted P-value)")
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_marker_dotplot(adata, marker_genes_dict: Dict[str, List[str]], title: str = "Marker Genes DotPlot") -> plt.Figure:
    """
    Plot a DotPlot of marker genes across clusters using Scanpy.
    
    Args:
        adata: AnnData object
        marker_genes_dict: Dictionary mapping cluster names/IDs to list of marker genes
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    import scanpy as sc
    
    # Scanpy expects genes to be in var_names
    valid_genes = {}
    for cluster, genes in marker_genes_dict.items():
        valid = [g for g in genes if g in adata.var_names]
        if valid:
            valid_genes[str(cluster)] = valid[:5] # Top 5 per cluster
            
    if not valid_genes:
        return None

    # Use scanpy's dotplot
    cluster_key = adata.uns.get('main_cluster_key', 'clusters')
    if cluster_key not in adata.obs.columns:
        return None
        
    # Ensure categorical for scanpy
    if not isinstance(adata.obs[cluster_key].dtype, pd.CategoricalDtype):
        adata.obs[cluster_key] = adata.obs[cluster_key].astype(str).astype('category')

    dp = sc.pl.dotplot(
        adata, 
        valid_genes, 
        groupby=cluster_key,
        title=title,
        show=False,
        expression_cutoff=0.1,
        mean_only_expressed=True,
        standard_scale='var',
        return_fig=True
    )
    
    dp.make_figure()
    return dp.fig


def plot_cluster_silhouette(
    sample_silhouette_values: np.ndarray, 
    labels: np.ndarray,
    title: str = "Silhouette Analysis per Cluster"
) -> plt.Figure:
    """
    Plot silhouette scores per cluster to see clustering quality distribution.
    
    Args:
        sample_silhouette_values: Silhouette score for each sample
        labels: Cluster labels for each sample
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    import matplotlib.cm as cm

    sample_silhouette_values = np.asarray(sample_silhouette_values)
    labels = np.asarray(labels)

    if sample_silhouette_values.size == 0 or labels.size == 0:
        return None

    if sample_silhouette_values.shape[0] != labels.shape[0]:
        raise ValueError("Silhouette values and labels must have the same length.")

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_lower = 10
    for i, cluster in enumerate(unique_labels):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == cluster]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title(title)
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    avg_score = np.mean(sample_silhouette_values)
    ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Average ({avg_score:.2f})")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_top_genes_by_group_heatmap(
    df_genes_by_group: Dict[str, pd.DataFrame],
    n_genes: int = 10,
    title: str = "Top Expressed Genes by Group"
) -> plt.Figure:
    """
    Plot heatmap showing top expressed genes for each group.
    
    Args:
        df_genes_by_group: Dictionary mapping group names to DataFrames with gene stats
        n_genes: Number of top genes to display per group
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not df_genes_by_group:
        return None
    
    # Collect top genes from each group
    all_genes = set()
    for group_df in df_genes_by_group.values():
        top_genes = group_df.head(n_genes)['Gene'].tolist()
        all_genes.update(top_genes)
    
    all_genes = sorted(all_genes)
    groups = sorted(df_genes_by_group.keys())
    
    # Build matrix: genes x groups
    matrix = np.zeros((len(all_genes), len(groups)))
    
    for j, group in enumerate(groups):
        df = df_genes_by_group[group]
        gene_to_expr = dict(zip(df['Gene'], df['Mean Expression']))
        
        for i, gene in enumerate(all_genes):
            matrix[i, j] = gene_to_expr.get(gene, 0)
    
    # Create DataFrame for heatmap
    df_heatmap = pd.DataFrame(matrix, index=all_genes, columns=groups)
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), max(6, len(all_genes) * 0.3)))
    
    sns.heatmap(
        df_heatmap,
        annot=False,
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Mean Expression'},
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    return fig


def plot_top_genes_by_group_bars(
    df_genes_by_group: Dict[str, pd.DataFrame],
    n_genes: int = 10,
    title: str = "Top Expressed Genes by Group"
) -> plt.Figure:
    """
    Plot grouped bar charts showing top expressed genes for each group.
    
    Args:
        df_genes_by_group: Dictionary mapping group names to DataFrames with gene stats
        n_genes: Number of top genes to display per group
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not df_genes_by_group:
        return None
    
    groups = sorted(df_genes_by_group.keys())
    n_groups = len(groups)
    
    # Determine grid layout
    n_cols = min(3, n_groups)
    n_rows = (n_groups + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    
    # Handle single subplot case
    if n_groups == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, group in enumerate(groups):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        df = df_genes_by_group[group].head(n_genes)
        
        # Create bar plot
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(df)))
        bars = ax.barh(
            range(len(df)),
            df['Mean Expression'],
            color=colors
        )
        
        # Set labels
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Gene'])
        ax.set_xlabel('Mean Expression', fontsize=10)
        ax.set_title(f'{group}', fontsize=12, fontweight='bold')
        ax.invert_yaxis()  # Highest at top
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, df['Mean Expression'])):
            ax.text(
                val, i, f' {val:.2f}',
                va='center', ha='left', fontsize=8
            )
    
    # Hide unused subplots
    for idx in range(n_groups, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig


def plot_loss_curves(
    loss_history: List[Dict[str, Any]],
    algorithm_name: str,
    show_components: bool = True
) -> Optional[plt.Figure]:
    """
    Plot training loss curves from a loss_history list.

    Args:
        loss_history: List of phase dicts with keys: name, epochs, train_loss,
            optional val_loss, and optional components
        algorithm_name: Display name for the figure title
        show_components: Whether to show individual loss components as dashed lines

    Returns:
        matplotlib Figure or None if no data
    """
    if not loss_history:
        return None

    n_phases = len(loss_history)
    fig, axes = plt.subplots(1, n_phases, figsize=(6 * n_phases, 4), squeeze=False)

    component_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    for idx, phase in enumerate(loss_history):
        ax = axes[0, idx]
        train_loss = phase.get('train_loss', [])
        val_loss = phase.get('val_loss', [])
        default_len = max(len(train_loss), len(val_loss))
        epochs = phase.get('epochs', list(range(default_len)))
        phase_name = phase.get('name', f'Phase {idx + 1}')

        if not train_loss and not val_loss:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(phase_name)
            continue

        # Main total loss line(s)
        if train_loss:
            train_epochs = epochs[:len(train_loss)] if len(epochs) >= len(train_loss) else list(range(len(train_loss)))
            ax.plot(train_epochs, train_loss, color='black', linewidth=2, label='Train loss')
        if val_loss:
            val_epochs = epochs[:len(val_loss)] if len(epochs) >= len(val_loss) else list(range(len(val_loss)))
            ax.plot(val_epochs, val_loss, color='#1f77b4', linewidth=2, label='Val loss')

        # Component lines (dashed)
        if show_components and 'components' in phase:
            for c_idx, (comp_name, comp_values) in enumerate(phase['components'].items()):
                if comp_values == train_loss or comp_values == val_loss:
                    continue
                comp_epochs = epochs[:len(comp_values)] if len(epochs) >= len(comp_values) else list(range(len(comp_values)))
                color = component_colors[c_idx % len(component_colors)]
                ax.plot(comp_epochs, comp_values, color=color, linewidth=1, linestyle='--',
                        alpha=0.7, label=comp_name)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(phase_name, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    has_val = any(phase.get('val_loss') for phase in loss_history)
    title_suffix = 'Training/Validation Loss' if has_val else 'Training Loss'
    fig.suptitle(f'{algorithm_name} - {title_suffix}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Parameter / dataset annotation helper
# =============================================================================

def _add_param_annotation(
    fig: plt.Figure,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> None:
    """Draw a semi-transparent text box with key parameters and dataset info on a figure."""
    lines: List[str] = []
    if dataset_info:
        lines.append(dataset_info)
    if params_info:
        for key, val in params_info.items():
            lines.append(f"{key}: {val}")
    if not lines:
        return
    text = "\n".join(lines)
    fig.text(
        0.01, 0.01, text,
        fontsize=7, fontfamily='monospace',
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'),
        transform=fig.transFigure,
    )


# =============================================================================
# Static UMAP: colored by batch
# =============================================================================

def plot_umap_batch(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    title: str = "UMAP - Batch",
    point_size: int = 2,
    random_state: int = 42,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """Plot UMAP colored by batch labels (strings or integers)."""
    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] == 0:
        return None

    embeddings_2d, _ = _compute_umap_or_2d(embeddings, random_state=random_state)
    encoded, unique_labels, _ = _encode_labels(np.asarray(batch_labels))
    n_labels = len(unique_labels)
    cmap = plt.cm.tab10 if n_labels <= 10 else plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=encoded, cmap=cmap, norm=norm, s=point_size, alpha=0.7, rasterized=True,
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if n_labels <= 30:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(norm(i)), markersize=6, label=str(lbl))
            for i, lbl in enumerate(unique_labels)
        ]
        ax.legend(handles=handles, fontsize=8, markerscale=1.5,
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, framealpha=0.9)

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    return fig


# =============================================================================
# Static UMAP: ground truth + cell-weight opacity
# =============================================================================

def _normalize_cell_weights_robust(cell_weights: np.ndarray) -> np.ndarray:
    """
    Robustly normalize weights to [0, 1] using 5th/95th percentiles.
    """
    w = np.asarray(cell_weights, dtype=np.float64)
    if len(w) == 0:
        return np.asarray([], dtype=np.float64)

    w_lo = float(np.nanpercentile(w, 5))
    w_hi = float(np.nanpercentile(w, 95))
    if not np.isfinite(w_lo) or not np.isfinite(w_hi) or w_hi <= w_lo:
        w_lo = float(np.nanmin(w))
        w_hi = float(np.nanmax(w))
    if np.isfinite(w_hi) and w_hi > w_lo:
        return np.clip((w - w_lo) / (w_hi - w_lo), 0.0, 1.0)
    return np.full(len(w), 0.5, dtype=np.float64)


def _weights_to_alpha_exponential(
    w_norm: np.ndarray,
    strength: float = 14.0,
    min_alpha: float = 0.002,
) -> np.ndarray:
    """
    Convert normalized weights to alpha values with very strong exponential contrast.

    The mapping ``alpha = exp((w_norm - 1) * strength)`` compresses low-weight
    cells to near-invisibility while keeping high-weight cells fully visible.
    With ``strength=14`` (default):
      - w_norm = 0.0  →  alpha ≈ 0.000001  (effectively invisible)
      - w_norm = 0.5  →  alpha ≈ 0.0009    (very faint)
      - w_norm = 0.75 →  alpha ≈ 0.030     (slightly visible)
      - w_norm = 1.0  →  alpha = 1.0       (fully visible)

    This is *quasi-exponential* in that the contrast between low and high weights
    is roughly 500× — much more dramatic than a linear ramp.

    Args:
        w_norm: Normalized weights in [0, 1].
        strength: Exponential slope; higher = more contrast (default 14).
        min_alpha: Hard floor so cells are never completely invisible (default 0.002).
    """
    w = np.asarray(w_norm, dtype=np.float64)
    return np.clip(np.exp((w - 1.0) * strength), min_alpha, 1.0)


def plot_umap_weighted(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cell_weights: np.ndarray,
    title: str = "UMAP - Cell Weights (Opacity ∝ loss weight)",
    point_size: int = 3,
    random_state: int = 42,
    label_names: Optional[Dict[int, str]] = None,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot UMAP colored by cell-type labels with per-cell opacity driven by
    reconstruction loss weight.

    Rendering strategy:
    - Cells are sorted by weight (ascending) and drawn bottom-up so that
      high-weight (rare) cells are always visible on top.
    - Opacity follows a quasi-exponential curve (strength=14): low-weight cells
      are nearly invisible while high-weight cells are fully opaque.
    - The legend shows one opaque marker per cell type for readability.
    - A subtitle annotation reminds the viewer of the opacity encoding.
    """
    embeddings = np.asarray(embeddings)
    cell_weights = np.asarray(cell_weights, dtype=np.float64)
    if embeddings.shape[0] == 0:
        return None

    embeddings_2d, _ = _compute_umap_or_2d(embeddings, random_state=random_state)
    encoded, unique_labels, _ = _encode_labels(np.asarray(labels))
    n_labels = len(unique_labels)
    cmap = plt.cm.tab20 if n_labels <= 20 else plt.cm.gist_ncar
    norm_c = plt.Normalize(vmin=0, vmax=max(n_labels - 1, 1))

    w_norm = _normalize_cell_weights_robust(cell_weights)
    alpha_arr = _weights_to_alpha_exponential(w_norm, strength=14.0, min_alpha=0.002)

    # Sort all cells by ascending weight: low-weight cells drawn first (below).
    sort_order = np.argsort(w_norm)  # ascending: faint cells at the bottom
    e2d_sorted = embeddings_2d[sort_order]
    enc_sorted = encoded[sort_order]
    alpha_sorted = alpha_arr[sort_order]
    label_idx_sorted = enc_sorted  # same integer coding

    fig, ax = plt.subplots(figsize=(12, 8))

    # Build per-point RGBA array in sorted order
    colors_per_point = np.array(
        [list(cmap(norm_c(i))) for i in label_idx_sorted], dtype=np.float64
    )  # shape (N, 4)
    colors_per_point[:, 3] = alpha_sorted

    ax.scatter(
        e2d_sorted[:, 0], e2d_sorted[:, 1],
        c=colors_per_point,
        s=float(point_size),
        rasterized=True,
        linewidths=0,
    )

    ax.set_title(
        title + "\n"
        r"(opacity ∝ reconstruction weight — rare ↑ opaque, common ↓ transparent)",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Legend: one fully-opaque proxy marker per cell type
    if n_labels <= 30:
        handles = [
            plt.Line2D(
                [0], [0], marker='o', color='w',
                markerfacecolor=cmap(norm_c(i)),
                markersize=7,
                label=_decode_label_name(unique_labels[i], label_names),
            )
            for i in range(n_labels)
        ]
        ax.legend(
            handles=handles,
            fontsize=8,
            markerscale=1.0,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0,
            framealpha=0.9,
            title="Cell type",
            title_fontsize=8,
        )

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    return fig


def plot_umap_weighted_gradient(
    embeddings: np.ndarray,
    cell_weights: np.ndarray,
    title: str = "UMAP - Cell Weights (Gradient)",
    point_size: int = 3,
    random_state: int = 42,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot UMAP with a continuous color gradient encoding per-cell reconstruction
    loss weights.

    Color encoding:
    - **plasma** colormap: dark-purple (low weight) → vivid yellow (high weight).
      This perceptually-uniform ramp maximises contrast at both ends.
    - Cells are sorted ascending by weight so high-weight (bright) cells are
      drawn on top and remain visible even in dense regions.
    - No gamma compression: linear mapping preserves the true weight distribution.
    - Colorbar ticks annotated with min/max values for quick inspection.
    """
    embeddings = np.asarray(embeddings)
    cell_weights = np.asarray(cell_weights, dtype=np.float64)
    if embeddings.shape[0] == 0:
        return None

    embeddings_2d, _ = _compute_umap_or_2d(embeddings, random_state=random_state)
    w_norm = _normalize_cell_weights_robust(cell_weights)

    # Sort ascending so high-weight cells are rendered on top.
    sort_order = np.argsort(w_norm)
    e2d_s = embeddings_2d[sort_order]
    w_s = w_norm[sort_order]

    fig, ax = plt.subplots(figsize=(11, 8))
    sc = ax.scatter(
        e2d_s[:, 0],
        e2d_s[:, 1],
        c=w_s,
        cmap='plasma',
        vmin=0.0,
        vmax=1.0,
        s=float(point_size),
        alpha=0.95,
        rasterized=True,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Reconstruction weight (normalized)\nLow → High",
        rotation=90,
        labelpad=12,
        fontsize=10,
    )
    # Annotate colorbar extremes
    w_raw = cell_weights
    cbar.ax.text(
        0.5, -0.02,
        f"min={float(np.nanmin(w_raw)):.2f}",
        transform=cbar.ax.transAxes,
        ha='center', va='top', fontsize=8, color='#444'
    )
    cbar.ax.text(
        0.5, 1.02,
        f"max={float(np.nanmax(w_raw)):.2f}",
        transform=cbar.ax.transAxes,
        ha='center', va='bottom', fontsize=8, color='#444'
    )

    ax.set_title(
        title + "\n"
        "(plasma gradient: dark purple = low weight, bright yellow = high weight)",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    plt.tight_layout()
    return fig


def plot_umap_evolution(
    embedding_snapshots: List[Dict[str, Any]],
    labels: np.ndarray,
    algorithm_name: str = '',
    max_cols: int = 4,
    point_size: int = 3,
    random_state: int = 42,
    max_points: int = 5000,
    max_snapshots: int = 18,
    projection_mode: str = 'shared',
    color_mode: str = 'ground_truth',
    cell_weights_per_snapshot: Optional[List[np.ndarray]] = None,
    batch_labels: Optional[np.ndarray] = None,
    labels_per_snapshot: Optional[List[np.ndarray]] = None,
    params_info: Optional[Dict[str, Any]] = None,
    dataset_info: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Plot a grid of UMAP projections showing embedding evolution across training epochs.

    Args:
        embedding_snapshots: List of snapshot dicts with keys: epoch, phase, embeddings
        labels: Ground truth labels for coloring (n_cells,)
        algorithm_name: Display name for the figure title
        max_cols: Maximum number of columns in the grid
        point_size: Size of scatter points
        random_state: Random state for UMAP
        max_points: Maximum number of cells to plot per snapshot (for speed)
        max_snapshots: Maximum snapshots to render (gallery-style capping)
        projection_mode: 'shared' (single 2D projection for all epochs) or
            'per_snapshot' (independent projection per epoch)
        color_mode: 'ground_truth' | 'ground_truth_weighted' | 'batch' | 'pseudo_cluster'
        cell_weights_per_snapshot: Per-snapshot cell weights (required for 'ground_truth_weighted')
        batch_labels: Batch labels for 'batch' mode (n_cells,)
        labels_per_snapshot: Per-snapshot labels for 'pseudo_cluster' mode (n_cells each)
        params_info: Dict of key parameters to annotate on the figure
        dataset_info: Dataset name + split info string

    Returns:
        matplotlib Figure or None if no valid snapshots
    """
    from math import ceil

    fixed_label_mode = color_mode in {'ground_truth', 'ground_truth_weighted', 'batch'}
    # For batch mode, use batch_labels instead of ground truth labels
    plot_labels = batch_labels if color_mode == 'batch' and batch_labels is not None else labels

    # Filter valid snapshots and keep original indices for aligned side-data.
    valid_snapshots_all = [
        {"snapshot": s, "orig_idx": i}
        for i, s in enumerate(embedding_snapshots)
        if s.get('embeddings') is not None and len(s['embeddings']) > 0
    ]
    selected_indices = _select_snapshot_indices_for_gallery(
        [x["snapshot"] for x in valid_snapshots_all],
        max_snapshots=max_snapshots,
    )
    valid_snapshots = [valid_snapshots_all[i] for i in selected_indices]
    if not valid_snapshots:
        return None

    if projection_mode not in {'shared', 'per_snapshot'}:
        projection_mode = 'shared'

    n_snapshots = len(valid_snapshots)
    n_cols = min(n_snapshots, max_cols)
    n_rows = ceil(n_snapshots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    # Encode labels once for fixed-label modes
    encoded_fixed = None
    unique_labels_fixed = None
    colors_fixed = None
    if fixed_label_mode:
        encoded_fixed, unique_labels_fixed, _ = _encode_labels(np.asarray(plot_labels))
        n_unique_fixed = len(unique_labels_fixed)
        if n_unique_fixed <= 10:
            cmap = plt.cm.tab10
        elif n_unique_fixed <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral
        colors_fixed = [cmap(i / max(n_unique_fixed - 1, 1)) for i in range(n_unique_fixed)]

    # Optional deterministic subsampling to keep rendering fast on large datasets.
    first_emb = np.asarray(valid_snapshots[0]["snapshot"]["embeddings"])
    n_cells = first_emb.shape[0]
    use_subsample = int(max_points) > 0 and n_cells > int(max_points)
    subsample_idx = None
    if use_subsample:
        rng = np.random.default_rng(int(random_state))
        subsample_idx = np.sort(rng.choice(n_cells, size=int(max_points), replace=False))
        if encoded_fixed is not None:
            encoded_fixed = encoded_fixed[subsample_idx]

    legend_unique_labels = None
    legend_colors = None

    shared_umap_fallback = False
    shared_umap_2d_by_snapshot: Optional[List[np.ndarray]] = None
    if projection_mode == 'shared':
        shared_input = []
        for entry in valid_snapshots:
            emb = np.asarray(entry["snapshot"]["embeddings"])
            if use_subsample and subsample_idx is not None:
                emb = emb[subsample_idx]
            shared_input.append(emb)
        shared_umap_2d_by_snapshot, shared_umap_fallback = _compute_shared_umap_sequence(
            shared_input,
            random_state=random_state,
        )

    for idx, entry in enumerate(valid_snapshots):
        snapshot = entry["snapshot"]
        orig_idx = int(entry["orig_idx"])
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        emb = snapshot['embeddings']
        if use_subsample and subsample_idx is not None:
            emb = np.asarray(emb)[subsample_idx]
        epoch = snapshot.get('epoch', '?')
        phase = snapshot.get('phase', '')

        # Compute 2D projection.
        if shared_umap_2d_by_snapshot is not None and idx < len(shared_umap_2d_by_snapshot):
            umap_2d = shared_umap_2d_by_snapshot[idx]
        else:
            umap_2d, _ = _compute_umap_or_2d(emb, random_state=random_state)

        # Resolve per-point opacity for weighted mode.
        use_weighted_alpha = (
            color_mode == 'ground_truth_weighted'
            and cell_weights_per_snapshot is not None
            and orig_idx < len(cell_weights_per_snapshot)
            and cell_weights_per_snapshot[orig_idx] is not None
        )
        alpha_arr = None
        if use_weighted_alpha:
            w = np.asarray(cell_weights_per_snapshot[orig_idx], dtype=np.float64)
            if use_subsample and subsample_idx is not None:
                w = w[subsample_idx]
            w_norm = _normalize_cell_weights_robust(w)
            alpha_arr = _weights_to_alpha_exponential(w_norm)

        encoded_curr = None
        unique_curr = None
        colors_curr = None
        if fixed_label_mode:
            encoded_curr = encoded_fixed
            unique_curr = unique_labels_fixed
            colors_curr = colors_fixed
        else:
            snap_labels = None
            if labels_per_snapshot is not None and orig_idx < len(labels_per_snapshot):
                snap_labels = labels_per_snapshot[orig_idx]
            if snap_labels is None:
                snap_labels = snapshot.get('pseudo_labels')
            if snap_labels is None:
                ax.text(0.5, 0.5, 'No pseudo-labels', ha='center', va='center', fontsize=9)
                ax.set_title(f"Epoch {epoch} ({phase})", fontsize=9, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            snap_labels = np.asarray(snap_labels)
            if len(snap_labels) != len(snapshot['embeddings']):
                ax.text(0.5, 0.5, 'Pseudo-label size mismatch', ha='center', va='center', fontsize=9)
                ax.set_title(f"Epoch {epoch} ({phase})", fontsize=9, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if use_subsample and subsample_idx is not None:
                snap_labels = snap_labels[subsample_idx]

            encoded_curr, unique_curr, _ = _encode_labels(snap_labels)
            n_unique_curr = len(unique_curr)
            if n_unique_curr <= 10:
                cmap_curr = plt.cm.tab10
            elif n_unique_curr <= 20:
                cmap_curr = plt.cm.tab20
            else:
                cmap_curr = plt.cm.nipy_spectral
            colors_curr = [cmap_curr(i / max(n_unique_curr - 1, 1)) for i in range(n_unique_curr)]

        if encoded_curr is None or unique_curr is None or colors_curr is None:
            ax.text(0.5, 0.5, 'No labels', ha='center', va='center', fontsize=9)
            ax.set_title(f"Epoch {epoch} ({phase})", fontsize=9, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if legend_unique_labels is None and len(unique_curr) <= 20:
            legend_unique_labels = unique_curr
            legend_colors = colors_curr

        # Plot each label group
        for label_idx, label_name in enumerate(unique_curr):
            mask = encoded_curr == label_idx
            if mask.sum() == 0:
                continue

            if alpha_arr is not None:
                base_rgba = np.array(colors_curr[label_idx], dtype=np.float64)
                rgba = np.tile(base_rgba, (int(mask.sum()), 1))
                rgba[:, 3] = alpha_arr[mask]
                ax.scatter(
                    umap_2d[mask, 0], umap_2d[mask, 1],
                    c=rgba, s=float(point_size),
                    label=str(label_name) if idx == 0 else None,
                    rasterized=True,
                )
            else:
                ax.scatter(
                    umap_2d[mask, 0], umap_2d[mask, 1],
                    c=[colors_curr[label_idx]], s=point_size, alpha=0.75,
                    label=str(label_name) if idx == 0 else None,
                    rasterized=True,
                )

        title = f"Epoch {epoch} ({phase})"
        if snapshot.get('snapshot_type') == 'weight_refresh':
            wr_idx = snapshot.get('weight_refresh_index')
            if wr_idx is not None:
                title += f" | refresh#{wr_idx}"
        if color_mode == 'pseudo_cluster':
            ax.set_title(
                f"{title} | k={len(unique_curr)}",
                fontsize=9,
                fontweight='bold',
            )
        else:
            ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_snapshots, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row][col].axis('off')

    # Add legend if not too many labels
    if legend_unique_labels is not None and legend_colors is not None and len(legend_unique_labels) <= 20:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=legend_colors[i],
                       markersize=6, label=str(legend_unique_labels[i]))
            for i in range(len(legend_unique_labels))
        ]
        fig.legend(
            handles=handles, loc='lower center',
            ncol=min(len(legend_unique_labels), 8), fontsize=7,
            bbox_to_anchor=(0.5, -0.02), frameon=True
        )

    # Title reflects the color mode
    mode_suffix = {
        'ground_truth': 'Ground Truth',
        'ground_truth_weighted': 'Ground Truth + Cell Weights (Alpha)',
        'batch': 'Batch',
        'pseudo_cluster': 'Pseudo-Clusters',
    }.get(color_mode, 'Ground Truth')
    proj_suffix = (
        "shared-UMAP"
        if projection_mode == 'shared' and not shared_umap_fallback
        else ("shared-fallback-2D" if projection_mode == 'shared' else "per-snapshot")
    )
    coverage_suffix = f"{n_snapshots}/{len(valid_snapshots_all)} snapshots"
    fig.suptitle(
        f'UMAP Evolution - {algorithm_name} ({mode_suffix} | {proj_suffix} | {coverage_suffix})',
        fontsize=13,
        fontweight='bold',
    )

    # Add parameter annotation
    if params_info or dataset_info:
        _add_param_annotation(fig, params_info, dataset_info)

    legend_size = len(legend_unique_labels) if legend_unique_labels is not None else 0
    plt.tight_layout(rect=[0, 0.03 if legend_size <= 20 and legend_size > 0 else 0, 1, 0.96])
    return fig


# =============================================================================
# Marker-Overlap Annotation Visualizations
# =============================================================================

def plot_marker_overlap_heatmap(
    overlap_matrix: 'pd.DataFrame',
    algorithm_name: str = "Algorithm",
    figsize: tuple = None,
) -> Optional[plt.Figure]:
    """
    Plot heatmap of marker gene overlap scores.

    Rows = predicted clusters, Columns = gold-standard cell types.
    Each cell = |DEG_pred ∩ DEG_gold| / n_top_genes.

    Args:
        overlap_matrix: DataFrame from marker_overlap_annotation (pred × gold).
        algorithm_name: Name for the title.
        figsize: Optional figure size override.

    Returns:
        Matplotlib figure or None if empty.
    """
    if overlap_matrix is None or overlap_matrix.empty:
        return None

    n_rows, n_cols = overlap_matrix.shape
    if figsize is None:
        figsize = (max(8, n_cols * 0.9), max(5, n_rows * 0.5))

    fig, ax = plt.subplots(figsize=figsize)

    im = sns.heatmap(
        overlap_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="lightgray",
        cbar_kws={"label": "Overlap Score (∩ / 100)"},
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"Marker Gene Overlap — {algorithm_name}\n"
        f"(Predicted Clusters × Gold-Standard Types)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Gold-Standard Cell Type", fontsize=11)
    ax.set_ylabel("Predicted Cluster", fontsize=11)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    return fig


def plot_annotation_sankey(
    labels_true: np.ndarray,
    hungarian_labels: np.ndarray,
    marker_labels: np.ndarray,
    algorithm_name: str = "Algorithm",
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Plot a Sankey diagram comparing Hungarian and Marker-Overlap annotations.

    Three columns:
    - Left: Hungarian (best-mapping) annotation
    - Center: Marker-Overlap annotation
    - Right: Gold Standard (ground truth)

    Flows show how many cells travel between each annotation assignment.
    Divergences between the two methods are visually obvious.

    Args:
        labels_true: Ground truth labels (1D array).
        hungarian_labels: Labels from Hungarian (align_labels) annotation.
        marker_labels: Labels from marker-overlap annotation.
        algorithm_name: Name for the title.
        save_path: Optional path to save the figure (HTML for plotly, PNG for fallback).

    Returns:
        Plotly Figure object if plotly is available, else matplotlib Figure.
    """
    labels_true = np.asarray(labels_true, dtype=str)
    hungarian_labels = np.asarray(hungarian_labels, dtype=str)
    marker_labels = np.asarray(marker_labels, dtype=str)

    try:
        import plotly.graph_objects as go

        # --- Build node and link lists ---
        # Nodes: Hungarian types + Marker-Overlap types + Gold Standard types
        hung_types = sorted(set(hungarian_labels))
        marker_types = sorted(set(marker_labels))
        gold_types = sorted(set(labels_true))

        # Prefix to ensure uniqueness across columns
        hung_nodes = [f"H: {t}" for t in hung_types]
        marker_nodes = [f"M: {t}" for t in marker_types]
        gold_nodes = [f"GT: {t}" for t in gold_types]

        all_nodes = hung_nodes + marker_nodes + gold_nodes
        node_index = {name: i for i, name in enumerate(all_nodes)}

        # Links: Hungarian → Marker-Overlap (left → center)
        from collections import Counter
        link_hm = Counter()
        for h, m in zip(hungarian_labels, marker_labels):
            link_hm[(f"H: {h}", f"M: {m}")] += 1

        # Links: Marker-Overlap → Gold Standard (center → right)
        link_mg = Counter()
        for m, g in zip(marker_labels, labels_true):
            link_mg[(f"M: {m}", f"GT: {g}")] += 1

        sources, targets, values, link_colors = [], [], [], []

        # Color palette
        n_types = len(gold_types)
        if n_types <= 10:
            palette = [f"hsla({int(i * 360 / n_types)}, 70%, 50%, 0.4)" for i in range(n_types)]
        else:
            palette = [f"hsla({int(i * 360 / n_types)}, 60%, 45%, 0.35)" for i in range(n_types)]
        type_to_color = {t: palette[i % len(palette)] for i, t in enumerate(gold_types)}

        # Node colors (solid)
        node_colors = []
        for n in all_nodes:
            # Extract type name
            type_name = n.split(": ", 1)[1] if ": " in n else n
            base_color = type_to_color.get(type_name, "hsla(0, 0%, 70%, 0.6)")
            # Make node colors more opaque
            node_colors.append(base_color.replace("0.4)", "0.8)").replace("0.35)", "0.8)"))

        for (src, tgt), val in link_hm.items():
            sources.append(node_index[src])
            targets.append(node_index[tgt])
            values.append(val)
            tgt_type = tgt.split(": ", 1)[1] if ": " in tgt else tgt
            link_colors.append(type_to_color.get(tgt_type, "hsla(0, 0%, 80%, 0.3)"))

        for (src, tgt), val in link_mg.items():
            sources.append(node_index[src])
            targets.append(node_index[tgt])
            values.append(val)
            tgt_type = tgt.split(": ", 1)[1] if ": " in tgt else tgt
            link_colors.append(type_to_color.get(tgt_type, "hsla(0, 0%, 80%, 0.3)"))

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=node_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
            ),
        )])

        fig.update_layout(
            title_text=(
                f"Annotation Comparison — {algorithm_name}<br>"
                f"<sub>Hungarian (left) → Marker-Overlap (center) → Gold Standard (right)</sub>"
            ),
            font_size=11,
            width=1200,
            height=max(600, n_types * 45),
        )

        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                try:
                    fig.write_image(save_path)
                except Exception:
                    # kaleido not installed — save as HTML instead
                    html_path = save_path.rsplit('.', 1)[0] + '.html'
                    fig.write_html(html_path)

        return fig

    except ImportError:
        # Fallback: matplotlib-based alluvial-like visualization
        import logging
        logging.getLogger(__name__).info(
            "plotly not available — generating matplotlib fallback for annotation comparison."
        )

        # Build a comparison table instead
        from collections import Counter

        match_count = np.sum(hungarian_labels == marker_labels)
        total = len(labels_true)
        agreement_pct = 100.0 * match_count / total if total > 0 else 0.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel 1: Agreement heatmap (Hungarian vs Marker-Overlap)
        hung_types = sorted(set(hungarian_labels))
        marker_types = sorted(set(marker_labels))
        hm_matrix = np.zeros((len(hung_types), len(marker_types)))
        hung_idx = {t: i for i, t in enumerate(hung_types)}
        mark_idx = {t: i for i, t in enumerate(marker_types)}
        for h, m in zip(hungarian_labels, marker_labels):
            hm_matrix[hung_idx[h], mark_idx[m]] += 1

        hm_df = pd.DataFrame(hm_matrix, index=hung_types, columns=marker_types)
        sns.heatmap(hm_df, annot=True, fmt=".0f", cmap="Blues", ax=axes[0],
                    linewidths=0.5, linecolor="lightgray")
        axes[0].set_title(f"Hungarian vs Marker-Overlap\nAgreement: {agreement_pct:.1f}%",
                          fontweight="bold")
        axes[0].set_xlabel("Marker-Overlap")
        axes[0].set_ylabel("Hungarian")

        # Panel 2: Per-type agreement bar chart
        types_all = sorted(set(labels_true))
        agree_by_type = {}
        for t in types_all:
            mask = labels_true == t
            n = mask.sum()
            agree = np.sum(
                (hungarian_labels[mask] == t) & (marker_labels[mask] == t)
            )
            agree_by_type[t] = 100.0 * agree / n if n > 0 else 0.0

        types_sorted = sorted(agree_by_type.keys(), key=lambda x: agree_by_type[x])
        colors_bar = [
            "#2ecc71" if agree_by_type[t] >= 80 else
            "#f39c12" if agree_by_type[t] >= 50 else
            "#e74c3c" for t in types_sorted
        ]
        axes[1].barh(types_sorted, [agree_by_type[t] for t in types_sorted],
                     color=colors_bar)
        axes[1].set_xlabel("Both Methods Correct (%)")
        axes[1].set_title("Per-Type Annotation Agreement", fontweight="bold")
        axes[1].set_xlim(0, 105)
        for i, t in enumerate(types_sorted):
            axes[1].text(agree_by_type[t] + 1, i, f"{agree_by_type[t]:.0f}%",
                         va="center", fontsize=8)

        fig.suptitle(f"Annotation Comparison — {algorithm_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fallback_path = save_path
            if str(fallback_path).lower().endswith(".html"):
                fallback_path = str(fallback_path)[:-5] + ".png"
            fig.savefig(fallback_path, dpi=150, bbox_inches="tight")

        return fig
