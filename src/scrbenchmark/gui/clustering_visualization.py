"""
Standalone UMAP generator page.

Allows users to:
- choose a local dataset path (H5AD),
- select one or more algorithms,
- customize algorithm hyperparameters,
- run clustering and regenerate UMAP figures.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from core.algorithm_registry import AlgorithmRegistry
from core.config import HyperparameterConfig
from utils.analysis_runner import AnalysisRunner
from utils.data_handler import DataHandler
import utils.visualization as viz
from gui.widgets import render_param_input as render_param_input_widget


def _algo_display_name(algo_name: str) -> str:
  algo_class = AlgorithmRegistry.get(algo_name)
  if algo_class is None:
    return algo_name
  try:
    return algo_class.get_info().display_name
  except Exception:
    return algo_name


def _load_hyperparameters(algo_name: str) -> List[HyperparameterConfig]:
  algo_class = AlgorithmRegistry.get(algo_name)
  if algo_class is None:
    return []
  try:
    return algo_class.get_hyperparameters()
  except Exception as exc:
    st.warning(f"Unable to load hyperparameters for `{algo_name}`: {exc}")
    return []


def _get_reverse_label_map(adata: Any) -> Optional[Dict[int, str]]:
  if adata is None:
    return None

  def _is_mostly_numeric_label(values: List[str]) -> bool:
    if not values:
      return True
    numeric_count = 0
    for value in values:
      txt = str(value).strip()
      if txt == "":
        continue
      try:
        float(txt)
        numeric_count += 1
      except ValueError:
        continue
    return (numeric_count / max(1, len(values))) >= 0.8

  def _build_one_to_one_map(code_series: pd.Series, name_series: pd.Series) -> Optional[Dict[int, str]]:
    tmp = pd.DataFrame({"code": code_series, "name": name_series}).dropna()
    if tmp.empty:
      return None
    per_code = tmp.groupby("code", observed=True)["name"].nunique()
    if not (per_code == 1).all():
      return None

    raw_mapping = tmp.drop_duplicates("code").set_index("code")["name"].to_dict()
    name_values = [str(v) for v in raw_mapping.values()]
    if _is_mostly_numeric_label(name_values):
      return None

    mapping: Dict[int, str] = {}
    for raw_code, raw_name in raw_mapping.items():
      try:
        code_key = int(float(raw_code))
      except (TypeError, ValueError):
        continue
      mapping[code_key] = str(raw_name)
    return mapping or None

  reverse: Dict[int, str] = {}
  if hasattr(adata, "uns"):
    raw_map = adata.uns.get("label_map")
    if isinstance(raw_map, dict):
      for k, v in raw_map.items():
        try:
          reverse[int(v)] = str(k)
        except (TypeError, ValueError):
          continue

  # If label_map already contains readable names, keep it.
  if reverse and not _is_mostly_numeric_label(list(reverse.values())):
    return reverse

  # Try to infer readable mapping from obs columns (common in pancreas datasets).
  if hasattr(adata, "obs") and adata.obs is not None:
    infer_pairs = [
      ("Group", "cell_type"),
      ("Group", "celltype"),
      ("labels_encoded", "cell_type"),
      ("labels_encoded", "celltype"),
      ("labels", "cell_type"),
      ("labels", "celltype"),
    ]
    for code_col, name_col in infer_pairs:
      if code_col in adata.obs.columns and name_col in adata.obs.columns:
        inferred = _build_one_to_one_map(adata.obs[code_col], adata.obs[name_col])
        if inferred:
          return inferred

  return reverse or None


def _find_batch_column(adata: Any) -> Optional[str]:
  if adata is None or not hasattr(adata, "obs"):
    return None
  for col in ["batch", "Batch", "donor", "sample", "dataset", "tech"]:
    if col in adata.obs.columns:
      return col
  return None


def _fallback_true_labels_from_adata(
  adata: Any,
  pred_labels: Optional[np.ndarray]
) -> Optional[np.ndarray]:
  """Best-effort fallback to recover true labels for diagnostic plotting."""
  if adata is None or pred_labels is None or not hasattr(adata, "obs"):
    return None

  n_pred = len(np.asarray(pred_labels))
  if n_pred <= 0:
    return None

  candidates = [
    "labels_encoded",
    "labels",
    "Group",
    "cell_type",
    "celltype",
    "CellType",
    "cell_types",
  ]
  for col in candidates:
    if col in adata.obs.columns:
      values = adata.obs[col].to_numpy()
      if len(values) == n_pred:
        return values
  return None


def _default_algo_params(algo_name: str) -> Dict[str, Any]:
  defaults: Dict[str, Any] = {}
  for hp in _load_hyperparameters(algo_name):
    defaults[hp.name] = hp.default
  return defaults


def _extract_final_reconstruction_loss(result: Any) -> Dict[str, Any]:
  """Extract final loss values from algorithm loss_history."""
  loss_history = getattr(result, "loss_history", []) or []
  warmup_final: Optional[float] = None
  weighted_final: Optional[float] = None
  final_loss: Optional[float] = None
  final_phase: Optional[str] = None

  for phase in loss_history:
    name = str(phase.get("name", "")).strip().lower()
    train_loss = phase.get("train_loss", []) or []
    if not train_loss:
      continue
    last = float(train_loss[-1])
    if name == "warm-up":
      warmup_final = last
    elif name == "weighted":
      weighted_final = last
    final_loss = last
    final_phase = str(phase.get("name", ""))

  return {
    "final_loss": final_loss,
    "final_phase": final_phase,
    "warmup_mse_final": warmup_final,
    "weighted_mse_final": weighted_final,
  }





def _extract_leiden_resolution_record(result: Any, display_name: str) -> Optional[Dict[str, Any]]:
  params = getattr(result, "params", {}) or {}
  clustering_method = str(params.get("clustering_method", "")).strip().lower()
  if clustering_method != "leiden":
    return None

  selected = params.get("leiden_resolution_selected")
  if selected is None:
    return None

  try:
    selected_float = float(selected)
  except (TypeError, ValueError):
    return None

  requested = params.get("leiden_resolution", 0.0)
  try:
    requested_float = float(requested)
  except (TypeError, ValueError):
    requested_float = 0.0

  return {
    "Algorithm": display_name,
    "Requested Resolution": round(requested_float, 6),
    "Selected Resolution": round(selected_float, 6),
    "Selection Mode": str(params.get("leiden_resolution_selection_mode", "unknown")),
  }


def _save_labels_csv(
    output_dir: Path,
    algo_name: str,
    predicted_labels: np.ndarray,
    true_labels: Optional[np.ndarray],
    batch_values: Optional[np.ndarray],
) -> Optional[Path]:
  labels_dir = output_dir / "labels"
  labels_dir.mkdir(parents=True, exist_ok=True)
  safe_algo = algo_name.replace(" ", "_").replace("/", "_")
  csv_path = labels_dir / f"labels_{safe_algo}_run0.csv"

  n = len(predicted_labels)
  data = {"predicted_label": np.asarray(predicted_labels)[:n]}
  if true_labels is not None and len(true_labels) == n:
    data["true_label"] = np.asarray(true_labels)
  if batch_values is not None and len(batch_values) == n:
    data["batch"] = np.asarray(batch_values)

  pd.DataFrame(data).to_csv(csv_path, index=False)
  return csv_path


def _persist_uploaded_h5ad(uploaded_file: Any) -> Optional[Path]:
  """Persist Streamlit uploaded file to a stable temp path and return it."""
  if uploaded_file is None:
    return None

  content = uploaded_file.getvalue()
  if not content:
    return None

  safe_name = Path(uploaded_file.name).name
  digest = hashlib.sha1(content).hexdigest()[:12]
  tmp_dir = Path(tempfile.gettempdir()) / "scrbenchmark_uploads"
  tmp_dir.mkdir(parents=True, exist_ok=True)
  target = tmp_dir / f"{digest}_{safe_name}"

  if not target.exists() or target.stat().st_size != len(content):
    with open(target, "wb") as f:
      f.write(content)

  return target


def render_umap_generator_page() -> None:
  """Render standalone page to regenerate UMAPs from selected algorithms/dataset."""
  st.subheader("Figure Generation")
  st.caption(
    "Run one or more algorithms on a `.h5ad` file and generate clustering visualizations "
    "(UMAP or native 2D latent space depending on returned embeddings)."
  )

  if "umap_gen_algo_params" not in st.session_state:
    st.session_state.umap_gen_algo_params = {}

  data_path_default = st.session_state.get(
    "umap_gen_data_path",
    st.session_state.get("uploaded_file_path", "data/pancreas_raw_counts.h5ad")
  )

  with st.container(border=True):
    st.subheader("1) Data")
    st.caption("You can either paste a path or select a file directly via Finder.")

    data_path = st.text_input(
      ".h5ad file path (optional if selected via Finder)",
      value=data_path_default,
      key="umap_gen_data_path",
      help="Exemple: data/Pancreas_RawCount_DCTCorr_Processed.h5ad"
    )

    uploaded_h5ad = st.file_uploader(
      "Select a .h5ad file via Finder",
      type=["h5ad"],
      accept_multiple_files=False,
      key="umap_gen_uploaded_h5ad",
      help="Opens the native macOS file picker (Finder) in your browser."
    )

    uploaded_path = _persist_uploaded_h5ad(uploaded_h5ad)
    if uploaded_path is not None:
      path_obj = uploaded_path
      st.success(f"File selected via Finder: `{uploaded_h5ad.name}`")
      st.caption(f"Temporary copy used for analysis: `{uploaded_path}`")
    else:
      path_obj = Path(data_path).expanduser()
      if path_obj.exists():
        st.success(f"File detected: `{path_obj}`")
      else:
        st.warning("File not found at this path yet.")

    use_as_is = st.checkbox(
      "Use data as-is (skip preprocessing)",
      value=True,
      key="umap_gen_skip_preprocessing",
      help="Enable if your file is already preprocessed."
    )

    preprocess_params: Dict[str, Any] = {"skip": True}
    if not use_as_is:
      col1, col2, col3 = st.columns(3)
      with col1:
        n_top_genes = st.number_input("HVGs", min_value=100, max_value=20000, value=2000, step=100)
        min_cells_per_gene = st.number_input("Min cells/gene", min_value=1, max_value=100, value=1, step=1)
      with col2:
        target_sum = st.number_input("Target sum", min_value=1000, max_value=100000, value=30000, step=1000)
        scale_max_value = st.number_input("Scale max", min_value=1.0, max_value=50.0, value=10.0, step=1.0)
      with col3:
        hvg_flavor = st.selectbox("HVG flavor", options=["seurat", "seurat_v3", "cell_ranger"], index=0)
      preprocess_params = {
        "skip": False,
        "n_top_genes": int(n_top_genes),
        "min_cells_per_gene": int(min_cells_per_gene),
        "target_sum": float(target_sum),
        "scale_max_value": float(scale_max_value),
        "hvg_flavor": hvg_flavor,
      }

  with st.container(border=True):
    st.subheader("2) Algorithms")
    all_algos = AlgorithmRegistry.get_all()
    algo_names = sorted(all_algos.keys(), key=lambda a: _algo_display_name(a).lower())

    default_selection: List[str] = []

    selected_algorithms = st.multiselect(
      "Select one or more algorithms",
      options=algo_names,
      default=default_selection,
      format_func=_algo_display_name,
      key="umap_gen_selected_algorithms"
    )

    for algo_name in selected_algorithms:
      algo_class = AlgorithmRegistry.get(algo_name)
      if algo_class is None:
        continue

      info = algo_class.get_info()
      hyperparams = _load_hyperparameters(algo_name)
      defaults = {hp.name: hp.default for hp in hyperparams}

      if algo_name not in st.session_state.umap_gen_algo_params:
        st.session_state.umap_gen_algo_params[algo_name] = defaults.copy()

      current_params = st.session_state.umap_gen_algo_params[algo_name]
      for k, v in defaults.items():
        current_params.setdefault(k, v)

      with st.expander(
        f"{info.display_name} ({algo_name})",
        expanded=False
      ):
        st.caption(info.description)
        if st.button("Reset to defaults", key=f"umap_gen_reset_{algo_name}"):
          st.session_state.umap_gen_algo_params[algo_name] = defaults.copy()
          st.rerun()

        categories: Dict[str, List[HyperparameterConfig]] = {}
        for hp in hyperparams:
          categories.setdefault(hp.category or "General", []).append(hp)

        for category_name, hp_list in categories.items():
          st.markdown(f"**{category_name}**")
          for hp in hp_list:
            widget_prefix = f"umap_gen_{algo_name}"
            value = current_params.get(hp.name, hp.default)
            current_params[hp.name] = render_param_input_widget(
              hp=hp,
              current_value=value,
              key_prefix=widget_prefix,
              compact=True,
            )

      st.session_state.umap_gen_algo_params[algo_name] = current_params

  with st.container(border=True):
    st.subheader("3) Execution")
    col1, col2, col3 = st.columns(3)
    with col1:
      random_seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
        key="umap_gen_seed"
      )
    with col2:
      output_dir_str = st.text_input(
        "Output directory",
        value="results/umap_generator",
        key="umap_gen_output_dir"
      )
    with col3:
      save_labels = st.checkbox("Save labels CSV", value=True, key="umap_gen_save_labels")



    show_diagnostic = st.checkbox(
      "Show error diagnostic view (Error Focus)",
      value=True,
      key="umap_gen_show_diagnostic",
      help="Shows a 2x2 view with true labels, predicted labels, batch, and highlighted misclassifications."
    )

    diag_col1, diag_col2 = st.columns(2)
    if show_diagnostic:
      with diag_col1:
        outline_mode = st.selectbox(
          "Cluster outlines",
          options=["none", "ellipse", "convex_hull", "density"],
          index=0,
          key="umap_gen_outline_mode",
          help="Outline style around predicted clusters."
        )
      with diag_col2:
        show_centroids = st.checkbox(
          "Show centroids",
          value=True,
          key="umap_gen_show_centroids",
          help="Display cluster label at the group centroid."
        )
    else:
      outline_mode = "none"
      show_centroids = True

    run_clicked = st.button("Generate visualizations", type="primary", use_container_width=True)

  # --------------- Generation (only when button is clicked) ---------------
  if run_clicked:
    if not selected_algorithms:
      st.error("Select at least one algorithm.")
      return

    if not path_obj.exists():
      st.error(f"File not found: `{path_obj}`")
      return

    st.session_state.umap_gen_leiden_resolution_records = []
    st.session_state.umap_gen_ae_recon_losses = []

    output_dir = Path(output_dir_str).expanduser()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    progress = st.progress(0)
    status = st.empty()

    def _progress_cb(message: str, ratio: Optional[float] = None) -> None:
      status.text(message)
      if ratio is not None:
        progress.progress(float(max(0.0, min(1.0, ratio))))

    try:
      status.text("Loading data...")
      handler = DataHandler().load(path_obj)
      handler.preprocess(preprocess_params)
      info = handler.get_info()
      st.info(f"Data ready: {info['n_cells']:,} cells, {info['n_genes']:,} genes")

      params_to_run: Dict[str, Dict[str, Any]] = {}
      for algo_name in selected_algorithms:
        base_params = st.session_state.umap_gen_algo_params.get(algo_name, {}).copy()
        base_params["random_state"] = int(random_seed)
        base_params["seed"] = int(random_seed)
        params_to_run[algo_name] = base_params



      runner = AnalysisRunner(output_dir=output_dir)
      results = runner.run_comparison(
        algorithm_names=selected_algorithms,
        data=handler,
        labels=None,
        params=params_to_run,
        n_repeats=1,
        compute_scib_metrics=False,
        progress_callback=_progress_cb
      )

      progress.progress(1.0)
      status.text("Completed.")

      if not results.results:
        st.error("No usable result was produced.")
        return



      adata = handler.get_data()
      reverse_label_map = _get_reverse_label_map(adata)
      batch_col = _find_batch_column(adata)

      generated: List[str] = []
      figures_store: List[Dict[str, Any]] = []
      skipped_algorithms: List[str] = []
      leiden_resolution_records: List[Dict[str, Any]] = []

      for result in results.results:
        algo_name = result.algorithm_name
        safe_algo = algo_name.replace(" ", "_").replace("/", "_")
        display_name = _algo_display_name(algo_name)

        leiden_record = _extract_leiden_resolution_record(result, display_name)
        if leiden_record is not None:
          leiden_resolution_records.append(leiden_record)

        true_labels = result.true_labels
        pred_labels = result.labels
        if true_labels is None:
          true_labels = _fallback_true_labels_from_adata(adata, pred_labels)

        # Resolve embeddings robustly: prefer algorithm output, fallback to data matrix
        # so UMAP plots are still generated when an algorithm only returns labels.
        embeddings = None
        try:
          if result.embeddings is not None:
            emb_arr = np.asarray(result.embeddings)
            if emb_arr.ndim == 2 and emb_arr.shape[0] > 0 and emb_arr.shape[1] >= 2:
              embeddings = emb_arr
            elif emb_arr.ndim == 2 and emb_arr.shape[0] > 0 and emb_arr.shape[1] == 1:
              # Keep plotting available even for 1D embeddings.
              zeros = np.zeros((emb_arr.shape[0], 1), dtype=emb_arr.dtype)
              embeddings = np.hstack([emb_arr, zeros])
        except Exception:
          embeddings = None

        native_2d_algorithms = set()
        if embeddings is None and algo_name not in native_2d_algorithms and pred_labels is not None and adata is not None:
          try:
            X_fallback = adata.X
            if hasattr(X_fallback, "toarray"):
              X_fallback = X_fallback.toarray()
            X_fallback = np.asarray(X_fallback)
            if X_fallback.ndim == 2 and X_fallback.shape[0] > 0 and X_fallback.shape[1] >= 2:
              n_pred = len(pred_labels)
              n_obs = X_fallback.shape[0]
              if n_pred == n_obs:
                embeddings = X_fallback
              elif n_pred < n_obs:
                embeddings = X_fallback[:n_pred, :]
          except Exception:
            embeddings = None

        if algo_name in native_2d_algorithms:
          if embeddings is None:
            skipped_algorithms.append(
              f"- {display_name}: 2D embeddings unavailable (adata.X fallback disabled for this algorithm)"
            )
            continue
          emb_arr = np.asarray(embeddings)
          if emb_arr.ndim != 2 or emb_arr.shape[1] != 2:
            skipped_algorithms.append(
              f"- {display_name}: expected 2D embeddings, got shape={emb_arr.shape}"
            )
            continue
          embeddings = emb_arr

        if embeddings is None:
          skipped_algorithms.append(
            f"- {display_name}: embeddings unavailable or invalid"
          )
          continue

        n_plot = embeddings.shape[0]
        if pred_labels is not None:
          pred_labels = np.asarray(pred_labels)[:n_plot]
        if true_labels is not None:
          true_labels = np.asarray(true_labels)[:n_plot]

        can_compare = (
          true_labels is not None
          and pred_labels is not None
          and len(true_labels) == len(pred_labels)
        )
        is_native_2d = embeddings.ndim == 2 and embeddings.shape[1] == 2
        projection_name = "Latent 2D" if is_native_2d else "UMAP"
        file_prefix = "latent2d" if is_native_2d else "umap"

        # --- Main projection figure ---
        if can_compare:
          fig = viz.plot_umap_comparison(
            embeddings=embeddings,
            true_labels=np.asarray(true_labels),
            predicted_labels=np.asarray(pred_labels),
            algorithm_name=display_name,
            label_names=reverse_label_map,
            projection_name=projection_name,
          )
          out_name = f"{file_prefix}_comparison_{safe_algo}.png"
        else:
          fig = viz.plot_umap_embeddings(
            embeddings=embeddings,
            labels=np.asarray(pred_labels),
            title=f"{projection_name}: {display_name}",
            predicted_labels=np.asarray(pred_labels),
            label_names=reverse_label_map,
            projection_name=projection_name,
          )
          out_name = f"{file_prefix}_{safe_algo}.png"

        out_path = figures_dir / out_name
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        generated.append(str(out_path))

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        figures_store.append({
          "label": f"{projection_name} - {display_name}",
          "filename": out_name,
          "bytes": buf.getvalue(),
          "key_suffix": f"{file_prefix}_{safe_algo}",
        })
        plt.close(fig)

        # --- Diagnostic figure ---
        if show_diagnostic and can_compare:
          batch_vals_diag = None
          if batch_col and adata is not None and len(adata.obs[batch_col]) == len(pred_labels):
            batch_vals_diag = adata.obs[batch_col].to_numpy()

          fig_diag = viz.plot_umap_diagnostic(
            embeddings=embeddings,
            true_labels=np.asarray(true_labels),
            predicted_labels=np.asarray(pred_labels),
            batch_labels=batch_vals_diag,
            algorithm_name=display_name,
            outline_mode=outline_mode,
            show_cluster_centroids=show_centroids,
            random_state=int(random_seed),
            label_names=reverse_label_map,
            projection_name=projection_name,
          )
          diag_name = f"{file_prefix}_diagnostic_{safe_algo}.png"
          diag_out_path = figures_dir / diag_name
          fig_diag.savefig(diag_out_path, bbox_inches="tight", dpi=150)
          generated.append(str(diag_out_path))

          buf_diag = io.BytesIO()
          fig_diag.savefig(buf_diag, format="png", bbox_inches="tight", dpi=150)
          figures_store.append({
            "label": f"Diagnostic - {display_name}",
            "filename": diag_name,
            "bytes": buf_diag.getvalue(),
            "key_suffix": f"diag_{safe_algo}",
          })
          plt.close(fig_diag)
        elif show_diagnostic:
          if true_labels is None:
            skipped_algorithms.append(
              f"- {display_name}: diagnostic skipped (true labels unavailable)"
            )
          elif pred_labels is None:
            skipped_algorithms.append(
              f"- {display_name}: diagnostic skipped (predicted labels unavailable)"
            )
          else:
            skipped_algorithms.append(
              f"- {display_name}: diagnostic skipped (size mismatch true={len(true_labels)} vs pred={len(pred_labels)})"
            )

        # --- Save labels CSV ---
        if save_labels and pred_labels is not None:
          batch_vals = None
          if batch_col and adata is not None and len(adata.obs[batch_col]) == len(pred_labels):
            batch_vals = adata.obs[batch_col].to_numpy()
          _save_labels_csv(
            output_dir=output_dir,
            algo_name=algo_name,
            predicted_labels=np.asarray(pred_labels),
            true_labels=np.asarray(true_labels) if true_labels is not None else None,
            batch_values=batch_vals,
          )

      # Persist figures in session state
      st.session_state.umap_gen_figures = figures_store
      st.session_state.umap_gen_leiden_resolution_records = leiden_resolution_records

      if skipped_algorithms:
        st.warning(
          "Some visualizations could not be generated:\n" + "\n".join(skipped_algorithms)
        )

      if not figures_store:
        st.warning(
          "No figure was generated. Check returned embeddings "
          "and label consistency."
        )
        return

      meta = {
        "data_path": str(path_obj),
        "algorithms": selected_algorithms,
        "preprocess_params": preprocess_params,
        "output_dir": str(output_dir),
        "generated_figures": generated,
        "leiden_resolution_selected": leiden_resolution_records,
        "ae_reconstruction_losses": [],
      }
      meta_path = output_dir / "umap_generator_run.json"
      with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

      st.success(f"Visualizations generated in `{figures_dir}`")
      st.caption(f"Run metadata: `{meta_path}`")

    except Exception as exc:
      progress.empty()
      status.empty()
      st.error(f"UMAP generation failed: {exc}")
      import traceback
      st.code(traceback.format_exc())
      return

  # --------------- Display (persisted across reruns) ---------------
  figures_store = st.session_state.get("umap_gen_figures")
  if not figures_store:
    return

  leiden_resolution_records = st.session_state.get("umap_gen_leiden_resolution_records", [])
  if leiden_resolution_records:
    st.markdown("### Selected Leiden Resolution")
    st.dataframe(pd.DataFrame(leiden_resolution_records), use_container_width=True)



  st.markdown("### Generated visualizations")
  for entry in figures_store:
    st.markdown(f"**{entry['label']}**")
    st.image(entry["bytes"], use_container_width=True)
    st.download_button(
      label=f"Download {entry['label']}",
      data=entry["bytes"],
      file_name=entry["filename"],
      mime="image/png",
      key=f"umap_gen_dl_{entry['key_suffix']}"
    )

  # --- Download all as ZIP ---
  if len(figures_store) > 1:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
      for entry in figures_store:
        zf.writestr(entry["filename"], entry["bytes"])
    st.download_button(
      label="Download all figures (ZIP)",
      data=zip_buf.getvalue(),
      file_name="umap_figures.zip",
      mime="application/zip",
      key="umap_gen_dl_all_zip"
    )
