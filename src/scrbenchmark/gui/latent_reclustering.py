"""
Latent re-clustering page.

Lets users:
- reuse embeddings produced by SCRBenchmark (from current session), or
- load a saved latent embedding file,
then run a new clustering method and visualize results.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import utils.visualization as viz
from utils.dataset_splitter import get_batch_column
from utils.metrics import compute_accuracy, compute_ari, compute_error_analysis, compute_nmi


_RESULT_STATE_KEY = "latent_reclustering_result"


def _to_2d_numeric(x: Any) -> np.ndarray:
  arr = np.asarray(x)
  if arr.ndim == 1:
    arr = arr.reshape(-1, 1)
  if arr.ndim != 2:
    raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
  if arr.shape[0] < 2:
    raise ValueError("At least 2 cells are required for clustering.")
  if not np.issubdtype(arr.dtype, np.number):
    arr = arr.astype(np.float64)
  return arr.astype(np.float32, copy=False)


def _read_uploaded_array(uploaded_file: Any) -> np.ndarray:
  suffix = Path(uploaded_file.name).suffix.lower()
  raw = uploaded_file.getvalue()

  if suffix == ".npy":
    arr = np.load(BytesIO(raw), allow_pickle=False)
    return _to_2d_numeric(arr)

  if suffix == ".npz":
    with np.load(BytesIO(raw), allow_pickle=False) as data:
      keys = list(data.files)
      if not keys:
        raise ValueError("NPZ file is empty.")
      selected_key = st.selectbox(
        "Array key in NPZ",
        options=keys,
        key="latent_npz_key_selector",
      )
      return _to_2d_numeric(data[selected_key])

  if suffix in {".csv", ".tsv", ".txt"}:
    sep = "\t" if suffix == ".tsv" else ","
    df = pd.read_csv(BytesIO(raw), sep=sep)
    return _to_2d_numeric(df.to_numpy())

  raise ValueError("Unsupported file format. Use .npy, .npz, .csv or .tsv.")


def _read_uploaded_labels(uploaded_file: Any, n_cells: int) -> Optional[np.ndarray]:
  if uploaded_file is None:
    return None

  suffix = Path(uploaded_file.name).suffix.lower()
  raw = uploaded_file.getvalue()

  if suffix == ".npy":
    labels = np.load(BytesIO(raw), allow_pickle=False).reshape(-1)
  else:
    sep = "\t" if suffix == ".tsv" else ","
    df = pd.read_csv(BytesIO(raw), sep=sep)
    if df.shape[1] == 1:
      labels = df.iloc[:, 0].to_numpy()
    else:
      col = st.selectbox(
        "Label column",
        options=list(df.columns),
        key="latent_labels_column_selector",
      )
      labels = df[col].to_numpy()

  if len(labels) != n_cells:
    st.warning(
      f"Label length mismatch: expected {n_cells}, got {len(labels)}. Ignoring labels for metrics."
    )
    return None
  return np.asarray(labels)


def _read_uploaded_groups(uploaded_file: Any, n_cells: int) -> Optional[np.ndarray]:
  if uploaded_file is None:
    return None

  suffix = Path(uploaded_file.name).suffix.lower()
  raw = uploaded_file.getvalue()

  if suffix == ".npy":
    groups = np.load(BytesIO(raw), allow_pickle=False).reshape(-1)
  else:
    sep = "\t" if suffix == ".tsv" else ","
    df = pd.read_csv(BytesIO(raw), sep=sep)
    if df.shape[1] == 1:
      groups = df.iloc[:, 0].to_numpy()
    else:
      col = st.selectbox(
        "Batch/group column",
        options=list(df.columns),
        key="latent_groups_column_selector",
      )
      groups = df[col].to_numpy()

  if len(groups) != n_cells:
    st.warning(
      f"Group length mismatch: expected {n_cells}, got {len(groups)}. Ignoring group labels."
    )
    return None
  return np.asarray(groups).astype(str)


def _infer_reverse_label_map(adata: Any) -> Optional[Dict[int, str]]:
  if adata is None or not hasattr(adata, "uns"):
    return None

  raw_map = adata.uns.get("label_map")
  if not isinstance(raw_map, dict):
    return None

  reverse: Dict[int, str] = {}
  for key, value in raw_map.items():
    try:
      reverse[int(value)] = str(key)
    except (TypeError, ValueError):
      pass

    try:
      key_int = int(key)
      value_text = str(value)
      try:
        float(value_text)
      except ValueError:
        reverse[key_int] = value_text
    except (TypeError, ValueError):
      pass

  return reverse or None


def _decode_true_labels_for_display(
  labels: Optional[np.ndarray],
  label_name_map: Optional[Dict[int, str]],
) -> Optional[np.ndarray]:
  if labels is None:
    return None
  arr = np.asarray(labels)
  if arr.size == 0 or not label_name_map:
    return arr.astype(str)

  decoded = []
  for value in arr:
    try:
      decoded.append(label_name_map.get(int(float(value)), str(value)))
    except (TypeError, ValueError):
      decoded.append(str(value))
  return np.asarray(decoded, dtype=object)


def _extract_group_ids(adata: Any, expected_n: int) -> Tuple[np.ndarray, str]:
  fallback = np.asarray(["all"] * expected_n, dtype=object)
  if adata is None or not hasattr(adata, "obs") or expected_n <= 0:
    return fallback, "all"

  batch_col = None
  try:
    batch_col = get_batch_column(adata)
  except Exception:
    batch_col = None

  if batch_col is None:
    for candidate in ["batch", "Batch", "tech", "dataset", "study", "sample", "donor", "platform"]:
      if candidate in adata.obs.columns:
        batch_col = candidate
        break

  if batch_col is None or batch_col not in adata.obs.columns:
    return fallback, "all"

  values = np.asarray(adata.obs[batch_col].astype(str).to_numpy())
  if len(values) != expected_n:
    return fallback, "all"
  return values, str(batch_col)


def _build_latent_export_df(embeddings: np.ndarray, cell_ids: Optional[np.ndarray]) -> pd.DataFrame:
  n_cells, n_dims = embeddings.shape
  export_ids = (
    np.asarray(cell_ids)
    if cell_ids is not None and len(cell_ids) == n_cells
    else np.arange(n_cells, dtype=int)
  )
  cols = [f"z_{i}" for i in range(n_dims)]
  df = pd.DataFrame(embeddings, columns=cols)
  df.insert(0, "cell_id", export_ids)
  return df


def _collect_session_latent_candidates() -> List[Dict[str, Any]]:
  candidates: List[Dict[str, Any]] = []
  data_handler = st.session_state.get("data_handler")
  adata_full = None
  if data_handler is not None and hasattr(data_handler, "get_data"):
    try:
      adata_full = data_handler.get_data()
    except Exception:
      adata_full = None

  standard_results = st.session_state.get("analysis_results")
  if standard_results is not None and hasattr(standard_results, "results"):
    for result in getattr(standard_results, "results", []):
      emb = getattr(result, "embeddings", None)
      if emb is None:
        continue
      emb_arr = _to_2d_numeric(emb)
      group_ids, group_key = _extract_group_ids(adata_full, len(emb_arr))
      algo_name = getattr(result, "algorithm_name", "algorithm")
      run_id = getattr(result, "run_id", 0)
      candidates.append(
        {
          "id": f"standard::{algo_name}::run{run_id}",
          "label": f"Standard | {algo_name} | run {run_id}",
          "embeddings": emb_arr,
          "true_labels": getattr(result, "true_labels", None),
          "cell_ids": np.arange(len(emb_arr), dtype=int),
          "group_ids": group_ids,
          "group_key": group_key,
          "label_name_map": _infer_reverse_label_map(adata_full),
        }
      )

  benchmark_setup = st.session_state.get("benchmark_setup", {}) or {}
  split_adata_map = {
    "train": benchmark_setup.get("adata_train"),
    "test": benchmark_setup.get("adata_test"),
    "val": benchmark_setup.get("adata_val"),
  }

  bench_results = st.session_state.get("benchmark_results")
  if bench_results is not None and hasattr(bench_results, "results"):
    for result in getattr(bench_results, "results", []):
      algo_name = getattr(result, "algorithm_name", "algorithm")
      run_id = getattr(result, "run_id", 0)
      split_specs = [
        (
          "train",
          getattr(result, "train_embeddings", None),
          getattr(result, "train_true_labels", None),
          getattr(result, "train_batch_ids", None),
        ),
        (
          "test",
          getattr(result, "test_embeddings", None),
          getattr(result, "test_true_labels", None),
          getattr(result, "test_batch_ids", None),
        ),
        (
          "val",
          getattr(result, "val_embeddings", None),
          getattr(result, "val_true_labels", None),
          getattr(result, "val_batch_ids", None),
        ),
      ]
      for split_name, emb, true_labels, batch_ids in split_specs:
        if emb is None:
          continue
        emb_arr = _to_2d_numeric(emb)
        split_adata = split_adata_map.get(split_name)
        group_ids = None
        group_key = "all"
        batch_arr = None
        if batch_ids is not None:
          try:
            batch_arr = np.asarray(batch_ids).reshape(-1)
          except Exception:
            batch_arr = None

        if batch_arr is not None and len(batch_arr) == len(emb_arr):
          group_ids = batch_arr.astype(str)
          group_key = "batch"
        else:
          group_ids, group_key = _extract_group_ids(split_adata, len(emb_arr))
        candidates.append(
          {
            "id": f"benchmark::{algo_name}::run{run_id}::{split_name}",
            "label": f"Benchmark | {algo_name} | run {run_id} | {split_name}",
            "embeddings": emb_arr,
            "true_labels": true_labels,
            "cell_ids": np.arange(len(emb_arr), dtype=int),
            "group_ids": group_ids,
            "group_key": group_key,
            "label_name_map": _infer_reverse_label_map(split_adata),
          }
        )

  return candidates


def _resolution_grid(res_min: float, res_max: float, res_step: float) -> np.ndarray:
  if res_step <= 0:
    res_step = 0.1
  if res_max < res_min:
    res_min, res_max = res_max, res_min
  grid = np.arange(res_min, res_max + 1e-12, res_step)
  if len(grid) == 0:
    return np.array([1.0], dtype=np.float64)
  return grid


def _run_kmeans(
  embeddings: np.ndarray,
  *,
  n_clusters: int,
  n_init: int,
  random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
  from sklearn.cluster import KMeans

  n_clusters = int(max(1, min(n_clusters, embeddings.shape[0])))
  n_init = int(max(1, n_init))

  model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
  labels = model.fit_predict(embeddings)
  return labels.astype(np.int64), {
    "method": "kmeans",
    "n_clusters_selected": int(len(np.unique(labels))),
    "n_clusters_requested": int(n_clusters),
    "n_init": int(n_init),
  }


def _run_leiden(
  embeddings: np.ndarray,
  *,
  n_neighbors: int,
  search_mode: str,
  manual_resolution: float,
  target_clusters: int,
  res_min: float,
  res_max: float,
  res_step: float,
  silhouette_max_cells: int,
  true_labels: Optional[np.ndarray],
  random_state: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
  try:
    import anndata as ad
    import scanpy as sc
  except ImportError as exc:
    raise ImportError(
      "Leiden reclustering requires scanpy + anndata. Install with: pip install scanpy anndata"
    ) from exc

  from sklearn.metrics import adjusted_rand_score, silhouette_score

  n_cells = embeddings.shape[0]
  if n_cells <= 2:
    labels = np.zeros(n_cells, dtype=np.int64)
    return labels, {
      "method": "leiden",
      "selection_mode": "degenerate_small_n",
      "selected_resolution": 1.0,
      "n_clusters_selected": 1,
    }

  n_neighbors = int(max(2, min(n_neighbors, n_cells - 1)))
  adata_tmp = ad.AnnData(X=embeddings)
  sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, use_rep="X")

  selected_resolution = float(manual_resolution)
  selected_mode = search_mode
  selected_score: Optional[float] = None

  if search_mode == "manual":
    selected_resolution = float(manual_resolution)
  else:
    grid = _resolution_grid(res_min, res_max, res_step)
    best_resolution = float(grid[0])

    if search_mode == "target_clusters":
      target = int(max(1, target_clusters))
      best_dist = float("inf")
      for res in grid:
        sc.tl.leiden(adata_tmp, resolution=float(res), random_state=random_state, key_added="_latent_tmp")
        labels = adata_tmp.obs["_latent_tmp"].astype(int).to_numpy()
        dist = abs(len(np.unique(labels)) - target)
        if dist < best_dist or (dist == best_dist and float(res) > best_resolution):
          best_dist = dist
          best_resolution = float(res)
      selected_resolution = best_resolution
      selected_score = float(best_dist)

    elif search_mode == "ari":
      if true_labels is None:
        raise ValueError("ARI search requires true labels.")
      best_ari = -np.inf
      for res in grid:
        sc.tl.leiden(adata_tmp, resolution=float(res), random_state=random_state, key_added="_latent_tmp")
        labels = adata_tmp.obs["_latent_tmp"].astype(int).to_numpy()
        if len(np.unique(labels)) <= 1:
          continue
        ari = float(adjusted_rand_score(true_labels, labels))
        if ari > best_ari or (np.isclose(ari, best_ari) and float(res) > best_resolution):
          best_ari = ari
          best_resolution = float(res)
      selected_resolution = best_resolution
      selected_score = None if best_ari == -np.inf else float(best_ari)

    else:  # silhouette
      rng = np.random.default_rng(int(random_state))
      max_cells = int(max(200, silhouette_max_cells))
      if n_cells > max_cells:
        eval_idx = np.sort(rng.choice(n_cells, size=max_cells, replace=False))
        emb_eval = embeddings[eval_idx]
      else:
        eval_idx = None
        emb_eval = embeddings

      best_sil = -np.inf
      for res in grid:
        sc.tl.leiden(adata_tmp, resolution=float(res), random_state=random_state, key_added="_latent_tmp")
        labels = adata_tmp.obs["_latent_tmp"].astype(int).to_numpy()
        if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
          continue
        labels_eval = labels if eval_idx is None else labels[eval_idx]
        if len(np.unique(labels_eval)) <= 1:
          continue
        try:
          sil = float(silhouette_score(emb_eval, labels_eval, metric="euclidean"))
        except Exception:
          continue
        if sil > best_sil or (np.isclose(sil, best_sil) and float(res) > best_resolution):
          best_sil = sil
          best_resolution = float(res)
      selected_resolution = best_resolution
      selected_score = None if best_sil == -np.inf else float(best_sil)

  sc.tl.leiden(adata_tmp, resolution=float(selected_resolution), random_state=random_state, key_added="leiden")
  final_labels = adata_tmp.obs["leiden"].astype(int).to_numpy()

  meta: Dict[str, Any] = {
    "method": "leiden",
    "selection_mode": selected_mode,
    "selected_resolution": float(selected_resolution),
    "n_neighbors": int(n_neighbors),
    "n_clusters_selected": int(len(np.unique(final_labels))),
  }
  if selected_mode == "target_clusters":
    meta["target_cluster_distance"] = selected_score
  elif selected_mode == "ari":
    meta["ari_selected"] = selected_score
  elif selected_mode == "silhouette":
    meta["silhouette_selected"] = selected_score

  return final_labels.astype(np.int64), meta


def _render_metrics(true_labels: Optional[np.ndarray], pred_labels: np.ndarray) -> None:
  if true_labels is None:
    st.info("No ground-truth labels provided: supervised metrics are not available.")
    return
  if len(true_labels) != len(pred_labels):
    st.warning("Ground-truth labels length does not match predictions. Metrics skipped.")
    return

  try:
    nmi = compute_nmi(true_labels, pred_labels)
    ari = compute_ari(true_labels, pred_labels)
    acc = compute_accuracy(true_labels, pred_labels)
  except Exception as exc:
    st.warning(f"Could not compute supervised metrics: {exc}")
    return

  c1, c2, c3 = st.columns(3)
  c1.metric("NMI", f"{nmi:.4f}")
  c2.metric("ARI", f"{ari:.4f}")
  c3.metric("ACC", f"{acc:.4f}")


def _render_cluster_distribution(pred_labels: np.ndarray) -> None:
  counts = pd.Series(pred_labels).value_counts().sort_values(ascending=False)
  df = counts.rename_axis("cluster").reset_index(name="n_cells")
  df["percent"] = (df["n_cells"] / max(1, len(pred_labels)) * 100.0).round(2)
  st.dataframe(df, use_container_width=True, hide_index=True)


def _render_latent_export_panel(embeddings: np.ndarray, cell_ids: Optional[np.ndarray]) -> None:
  st.markdown("### Export latent space")
  latent_df = _build_latent_export_df(embeddings, cell_ids)
  csv_bytes = latent_df.to_csv(index=False).encode("utf-8")

  npy_buffer = BytesIO()
  np.save(npy_buffer, embeddings)
  npy_buffer.seek(0)

  c1, c2 = st.columns(2)
  c1.download_button(
    "Download latent (.csv)",
    data=csv_bytes,
    file_name="latent_space.csv",
    mime="text/csv",
    use_container_width=True,
  )
  c2.download_button(
    "Download latent (.npy)",
    data=npy_buffer.getvalue(),
    file_name="latent_space.npy",
    mime="application/octet-stream",
    use_container_width=True,
  )


def _render_error_analysis(
  *,
  true_labels_display: np.ndarray,
  pred_labels: np.ndarray,
  group_ids: np.ndarray,
  group_key: str,
  algo_title: str,
) -> None:
  try:
    error_analysis = compute_error_analysis(true_labels_display, pred_labels, group_ids)
  except Exception as exc:
    st.warning(f"Could not compute error analysis: {exc}")
    return

  st.markdown("### Error analysis")
  top = st.columns(3)
  top[0].metric("Overall Error Rate", f"{float(error_analysis.get('overall_error_rate', np.nan)):.3f}")
  top[1].metric("Total Errors", int(error_analysis.get("total_errors", 0)))
  top[2].metric("Total Samples", int(error_analysis.get("total_samples", len(pred_labels))))

  error_by_celltype = error_analysis.get("error_by_celltype", {}) or {}
  if error_by_celltype:
    rows = []
    for cell_type, values in error_by_celltype.items():
      rows.append(
        {
          "cell_type": str(cell_type),
          "error_rate": float(values.get("error_rate", np.nan)),
          "n_errors": int(values.get("n_errors", 0)),
          "n_samples": int(values.get("n_samples", 0)),
        }
      )
    df_ct = pd.DataFrame(rows).sort_values("error_rate", ascending=False, na_position="last")
    max_top = max(1, min(60, len(df_ct)))
    top_n = st.slider(
      "Cell types to display",
      min_value=1,
      max_value=max_top,
      value=min(20, max_top),
    )
    plot_df = df_ct.head(top_n).iloc[::-1].copy()
    plot_df["error_rate"] = plot_df["error_rate"].fillna(0.0)

    fig_h = max(4.0, 0.34 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = plt.cm.RdYlGn_r(np.clip(plot_df["error_rate"].to_numpy(), 0, 1))
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["error_rate"].to_numpy(), color=colors, alpha=0.95)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["cell_type"].to_numpy())
    ax.set_xlim(0, 1)
    ax.set_xlabel("Error rate")
    ax.set_ylabel("Cell type")
    ax.set_title("Error by cell type")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(df_ct, use_container_width=True, hide_index=True)

  confusion_pairs = error_analysis.get("top_confusion_pairs", []) or []
  if confusion_pairs:
    st.markdown("#### Top confusion pairs")
    st.dataframe(pd.DataFrame(confusion_pairs), use_container_width=True, hide_index=True)

  st.markdown("#### Confusion matrix")
  normalize_cm = st.checkbox("Normalize confusion matrix", value=True, key="latent_confusion_normalized")
  try:
    fig_cm = viz.plot_confusion_matrix(
      true_labels=true_labels_display,
      predicted_labels=pred_labels,
      algorithm_name=algo_title,
      normalize=normalize_cm,
    )
    if fig_cm is not None:
      st.pyplot(fig_cm, use_container_width=True)
      plt.close(fig_cm)
  except Exception as exc:
    st.warning(f"Could not render confusion matrix: {exc}")

  nested = error_analysis.get("error_by_celltype_by_group", {}) or {}
  if nested and group_key != "all" and len(nested) > 1:
    st.markdown(f"#### Error by `{group_key}` and cell type")
    try:
      fig_hm = viz.plot_celltype_errors_by_batch(
        nested,
        title=f"Error rate by `{group_key}` and cell type",
      )
      st.pyplot(fig_hm, use_container_width=True)
      plt.close(fig_hm)
    except Exception as exc:
      st.warning(f"Could not render group x cell type error heatmap: {exc}")


def render_latent_reclustering_page() -> None:
  st.markdown("# Latent Re-clustering")
  st.caption(
    "Cluster an existing latent space (from current session or from a saved file), then visualize clusters."
  )

  source_mode = st.radio(
    "Latent source",
    options=["Current session results", "Upload latent file"],
    horizontal=True,
    key="latent_source_mode",
  )

  embeddings: Optional[np.ndarray] = None
  true_labels: Optional[np.ndarray] = None
  group_ids: Optional[np.ndarray] = None
  group_key = "all"
  label_name_map: Optional[Dict[int, str]] = None
  source_label = "latent"
  cell_ids: Optional[np.ndarray] = None

  if source_mode == "Current session results":
    candidates = _collect_session_latent_candidates()
    if not candidates:
      st.warning("No embeddings found in session. Run an analysis first, or upload a latent file.")
      return
    selected_label = st.selectbox(
      "Select embeddings",
      options=[c["label"] for c in candidates],
      key="latent_candidate_selector",
    )
    selected = next(c for c in candidates if c["label"] == selected_label)
    embeddings = selected["embeddings"]
    true_labels = selected.get("true_labels", None)
    group_ids = selected.get("group_ids", None)
    group_key = selected.get("group_key", "all")
    label_name_map = selected.get("label_name_map", None)
    source_label = selected["label"]
    cell_ids = selected.get("cell_ids", None)
  else:
    latent_file = st.file_uploader(
      "Latent file",
      type=["npy", "npz", "csv", "tsv"],
      help="Expected shape: n_cells x latent_dim",
      key="latent_file_uploader",
    )
    if latent_file is None:
      st.info("Upload a latent file to continue.")
      return

    try:
      embeddings = _read_uploaded_array(latent_file)
      source_label = latent_file.name
      cell_ids = np.arange(len(embeddings), dtype=int)
    except Exception as exc:
      st.error(f"Could not read latent file: {exc}")
      return

    labels_file = st.file_uploader(
      "Optional ground-truth labels",
      type=["csv", "tsv", "npy"],
      key="latent_labels_uploader",
    )
    true_labels = _read_uploaded_labels(labels_file, n_cells=embeddings.shape[0])

    groups_file = st.file_uploader(
      "Optional batch/group labels",
      type=["csv", "tsv", "npy"],
      key="latent_groups_uploader",
    )
    group_ids = _read_uploaded_groups(groups_file, n_cells=embeddings.shape[0])
    group_key = "group" if group_ids is not None else "all"

  if true_labels is not None and len(true_labels) != embeddings.shape[0]:
    st.warning(
      f"Ground-truth label length mismatch: expected {embeddings.shape[0]}, got {len(true_labels)}. "
      "Supervised metrics disabled."
    )
    true_labels = None

  if group_ids is None:
    group_ids = np.asarray(["all"] * embeddings.shape[0], dtype=object)
    group_key = "all"

  st.markdown("---")
  st.write(f"**Source:** `{source_label}`")
  st.write(f"**Shape:** `{embeddings.shape[0]}` cells x `{embeddings.shape[1]}` dimensions")
  if true_labels is not None and len(true_labels) == embeddings.shape[0]:
    st.write(f"**Ground-truth labels:** `{len(np.unique(true_labels))}` unique classes")
  if group_ids is not None and len(group_ids) == embeddings.shape[0] and group_key != "all":
    st.write(f"**{group_key} values:** `{len(np.unique(group_ids))}` groups")

  _render_latent_export_panel(embeddings, cell_ids)

  method = st.selectbox(
    "Clustering method",
    options=["Leiden", "KMeans"],
    index=0,
    key="latent_cluster_method",
  )
  random_state = int(st.number_input("Random state", min_value=0, max_value=999999, value=42, step=1))

  outline_mode = st.selectbox(
    "Cluster outline mode",
    options=["none", "ellipse", "convex_hull", "density"],
    index=0,
    key="latent_outline_mode",
  )

  clustering_params: Dict[str, Any] = {"method": method.lower(), "random_state": random_state}
  if method == "KMeans":
    n_clusters = int(st.number_input("n_clusters", min_value=2, max_value=max(2, embeddings.shape[0]), value=10, step=1))
    n_init = int(st.number_input("n_init", min_value=1, max_value=100, value=10, step=1))
    clustering_params.update({"n_clusters": n_clusters, "n_init": n_init})
  else:
    n_neighbors = int(st.number_input("leiden_neighbors", min_value=2, max_value=max(2, embeddings.shape[0] - 1), value=min(15, embeddings.shape[0] - 1), step=1))
    search_options = ["manual", "target_clusters", "silhouette"]
    default_search_idx = 0
    if true_labels is not None and len(true_labels) == embeddings.shape[0]:
      search_options = ["ari", "manual", "target_clusters", "silhouette"]
      default_search_idx = 0
      st.caption("Ground truth detected: Leiden resolution defaults to ARI maximization.")
    search_mode = st.selectbox("leiden_search_mode", options=search_options, index=default_search_idx)

    manual_resolution = float(st.number_input("leiden_resolution (manual)", min_value=0.01, max_value=10.0, value=1.0, step=0.05))
    target_clusters = int(st.number_input("target_clusters", min_value=2, max_value=max(2, embeddings.shape[0]), value=10, step=1))
    res_min = float(st.number_input("resolution_min", min_value=0.01, max_value=10.0, value=0.1, step=0.05))
    res_max = float(st.number_input("resolution_max", min_value=0.05, max_value=10.0, value=2.5, step=0.05))
    res_step = float(st.number_input("resolution_step", min_value=0.01, max_value=1.0, value=0.1, step=0.01))
    silhouette_max_cells = int(st.number_input("silhouette_max_cells", min_value=200, max_value=max(200, embeddings.shape[0]), value=min(5000, embeddings.shape[0]), step=200))

    clustering_params.update(
      {
        "n_neighbors": n_neighbors,
        "search_mode": search_mode,
        "manual_resolution": manual_resolution,
        "target_clusters": target_clusters,
        "res_min": res_min,
        "res_max": res_max,
        "res_step": res_step,
        "silhouette_max_cells": silhouette_max_cells,
      }
    )

  if st.button("Run re-clustering", type="primary", use_container_width=True):
    try:
      if method == "KMeans":
        pred_labels, meta = _run_kmeans(
          embeddings,
          n_clusters=clustering_params["n_clusters"],
          n_init=clustering_params["n_init"],
          random_state=random_state,
        )
      else:
        pred_labels, meta = _run_leiden(
          embeddings,
          n_neighbors=clustering_params["n_neighbors"],
          search_mode=clustering_params["search_mode"],
          manual_resolution=clustering_params["manual_resolution"],
          target_clusters=clustering_params["target_clusters"],
          res_min=clustering_params["res_min"],
          res_max=clustering_params["res_max"],
          res_step=clustering_params["res_step"],
          silhouette_max_cells=clustering_params["silhouette_max_cells"],
          true_labels=true_labels,
          random_state=random_state,
        )

      st.session_state[_RESULT_STATE_KEY] = {
        "source_label": source_label,
        "embeddings": embeddings,
        "true_labels": true_labels,
        "true_labels_display": _decode_true_labels_for_display(true_labels, label_name_map),
        "group_ids": group_ids,
        "group_key": group_key,
        "pred_labels": pred_labels,
        "meta": meta,
        "method": method,
        "cell_ids": cell_ids,
        "label_name_map": label_name_map,
      }
      st.success("Re-clustering completed.")
    except Exception as exc:
      st.error(f"Re-clustering failed: {exc}")

  result = st.session_state.get(_RESULT_STATE_KEY)
  if not result:
    return

  st.markdown("---")
  st.subheader("Results")
  st.json(result.get("meta", {}))
  _render_metrics(result.get("true_labels"), result["pred_labels"])

  st.markdown("### Cluster distribution")
  _render_cluster_distribution(result["pred_labels"])

  st.markdown("### Visualization")
  algo_title = f"Latent Re-clustering ({result['method']})"
  true_labels_display = result.get("true_labels_display")
  has_supervised = (
    true_labels_display is not None
    and len(true_labels_display) == len(result["pred_labels"])
  )
  has_group = (
    result.get("group_ids") is not None
    and len(result.get("group_ids")) == len(result["pred_labels"])
  )

  viz_modes = ["Predicted clusters only"]
  if has_supervised:
    viz_modes = ["UMAP comparison", "UMAP diagnostic", "Predicted clusters only"]
  viz_mode = st.selectbox("Visualization mode", options=viz_modes, index=0)

  try:
    if viz_mode == "UMAP comparison" and has_supervised:
      fig = viz.plot_umap_comparison(
        embeddings=result["embeddings"],
        true_labels=true_labels_display,
        predicted_labels=result["pred_labels"],
        algorithm_name=algo_title,
        outline_mode=outline_mode,
        label_names=result.get("label_name_map"),
      )
    elif viz_mode == "UMAP diagnostic" and has_supervised:
      batch_vals = result.get("group_ids") if has_group and result.get("group_key") != "all" else None
      fig = viz.plot_umap_diagnostic(
        embeddings=result["embeddings"],
        true_labels=true_labels_display,
        predicted_labels=result["pred_labels"],
        batch_labels=batch_vals,
        algorithm_name=algo_title,
        outline_mode=outline_mode,
        label_names=result.get("label_name_map"),
      )
    else:
      fig = viz.plot_umap_embeddings(
        embeddings=result["embeddings"],
        labels=result["pred_labels"],
        title=algo_title,
        predicted_labels=result["pred_labels"],
        cluster_outline_mode=outline_mode,
      )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
  except Exception as exc:
    st.warning(f"Could not render UMAP visualization: {exc}")

  if has_supervised:
    group_vals = result.get("group_ids")
    if group_vals is None or len(group_vals) != len(result["pred_labels"]):
      group_vals = np.asarray(["all"] * len(result["pred_labels"]), dtype=object)
    _render_error_analysis(
      true_labels_display=np.asarray(true_labels_display),
      pred_labels=np.asarray(result["pred_labels"]),
      group_ids=np.asarray(group_vals).astype(str),
      group_key=str(result.get("group_key", "all")),
      algo_title=algo_title,
    )

  out_df = pd.DataFrame(
    {
      "cell_id": (
        result["cell_ids"]
        if result.get("cell_ids") is not None and len(result["cell_ids"]) == len(result["pred_labels"])
        else np.arange(len(result["pred_labels"]))
      ),
      "predicted_cluster": result["pred_labels"],
    }
  )
  if result.get("true_labels") is not None and len(result["true_labels"]) == len(result["pred_labels"]):
    out_df["true_label"] = result.get("true_labels_display", result["true_labels"])
  if result.get("group_ids") is not None and len(result["group_ids"]) == len(result["pred_labels"]):
    out_df[str(result.get("group_key", "group"))] = result["group_ids"]

  csv_bytes = out_df.to_csv(index=False).encode("utf-8")
  st.download_button(
    "Download clustering labels (CSV)",
    data=csv_bytes,
    file_name="latent_reclustering_labels.csv",
    mime="text/csv",
    use_container_width=True,
  )
