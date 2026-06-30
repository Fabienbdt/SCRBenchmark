"""
Analysis page for running clustering and viewing results.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import io
import json
import seaborn as sns

# Add package root to path (legacy.py moved into gui/analysis/)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.algorithm_registry import AlgorithmRegistry
from utils.metrics import align_labels
from utils.analysis_runner import AnalysisRunner, BenchmarkComparisonResult
from utils.dataset_splitter import DatasetSplitter, get_batch_column
import utils.visualization as viz
from gui.algorithm_config import _generate_cli_command
from gui.widgets import check_prerequisites, detect_batch_column, display_error


def render_analysis_page():
  """Render the analysis page."""
  st.header("Run Analysis")

  # Check prerequisites
  if not check_prerequisites(
    require_data=True,
    require_preprocessing=True,
    require_algorithms=True,
  ):
    return

  handler = st.session_state.data_handler
  info = handler.get_info()

  # =========================================================================
  # Branching: Benchmark Mode vs Standard Mode
  # =========================================================================
  benchmark_setup = st.session_state.get('benchmark_setup', {})
  is_benchmark_mode = (
    st.session_state.get('benchmark_configured', False) and
    benchmark_setup.get('mode') == 'benchmark'
  )
  
  if is_benchmark_mode:
    # Benchmark Mode UI
    _render_benchmark_settings(handler, info)
  else:
    # Standard Mode UI
    st.subheader("Analysis Settings")

    col1, col2 = st.columns(2)

    with col1:
      n_repeats = st.number_input(
        "Number of repetitions",
        min_value=1,
        max_value=100,
        value=1,
        help="Number of times to run each algorithm (for statistical analysis)"
      )

    with col2:
      random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=42,
        help="Base random seed for reproducibility"
      )

    compute_scib_metrics = st.checkbox(
      "Compute scIB/scIB-E metrics",
      value=st.session_state.get('analysis_compute_scib_metrics', False),
      help="Disable to speed up analysis when scIB/scIB-E metrics are not needed."
    )
    st.session_state['analysis_compute_scib_metrics'] = compute_scib_metrics

    # Batch balancing options (for datasets with batch effects)
    st.markdown("---")
    st.subheader("Batch Balancing (Optional)")
    st.caption("Reduce larger batches to equalize contributions from each batch.")

    # Check if batch column exists
    adata = handler.get_data()
    detected_batch_col = detect_batch_column(adata)

    balance_batches_standard = st.checkbox(
      "Enable batch balancing",
      value=False,
      help="Subsample larger batches so each batch contributes equally to the analysis. "
         "This can help reduce batch effects without explicit correction.",
      key="balance_batches_standard"
    )

    balance_settings = {}
    if balance_batches_standard:
      col_bal1, col_bal2 = st.columns(2)

      with col_bal1:
        # Batch column selection
        available_cols = [c for c in adata.obs.columns if adata.obs[c].dtype == 'object' or adata.obs[c].dtype.name == 'category']
        available_cols = [c for c in available_cols if adata.obs[c].nunique() > 1 and adata.obs[c].nunique() < 50]

        if detected_batch_col and detected_batch_col in available_cols:
          default_idx = available_cols.index(detected_batch_col)
        else:
          default_idx = 0

        if available_cols:
          batch_col_standard = st.selectbox(
            "Batch column",
            options=available_cols,
            index=default_idx,
            help="Column containing batch identifiers",
            key="batch_col_standard"
          )
          balance_settings['batch_col'] = batch_col_standard

          # Show batch distribution
          batch_counts = adata.obs[batch_col_standard].value_counts()
          st.caption(f"Batches: {dict(batch_counts)}")
        else:
          st.warning("No suitable batch column found.")
          batch_col_standard = None

      with col_bal2:
        if available_cols and batch_col_standard:
          batch_counts = adata.obs[batch_col_standard].value_counts()
          min_batch = int(batch_counts.min())
          max_batch = int(batch_counts.max())

          balance_target_standard = st.number_input(
            "Target cells per batch",
            min_value=10,
            max_value=max_batch,
            value=min_batch,
            help=f"Smallest batch has {min_batch} cells. Larger batches will be subsampled to this target.",
            key="balance_target_standard"
          )
          balance_settings['target'] = balance_target_standard

          # Preview
          total_after = min(balance_target_standard, min_batch) * len(batch_counts)
          st.caption(f"Expected cells after balancing: ~{total_after:,}")

    st.session_state['balance_settings_standard'] = balance_settings if balance_batches_standard else {}

    # Summary
    st.markdown("---")
    st.subheader("Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
      st.metric("Cells", f"{info['n_cells']:,}")
    with col2:
      st.metric("Genes", f"{info['n_genes']:,}")
    with col3:
      st.metric("Algorithms", len(st.session_state.selected_algorithms))

    st.markdown("**Selected Algorithms & Hyperparameters:**")
    
    # Get params from session state, defaulting to empty dict if not present
    all_params = st.session_state.get('algorithm_params', {})
    
    for algo_name in st.session_state.selected_algorithms:
      algo_class = AlgorithmRegistry.get(algo_name)
      algo_info = algo_class.get_info()
      hyperparams = algo_class.get_hyperparameters()
      
      # Get current params for this algorithm
      current_algo_params = all_params.get(algo_name, {})
      
      with st.expander(f"**{algo_info.display_name}** - *Click to view parameters*"):
        st.markdown(f"*{algo_info.description}*")
        
        # Group params by category
        categories = {}
        for hp in hyperparams:
          if hp.category not in categories:
            categories[hp.category] = []
          categories[hp.category].append(hp)
        
        # Display params by category
        for category, hps in categories.items():
          st.markdown(f"**{category}**")
          for hp in hps:
            # Use session state value or default
            val = current_algo_params.get(hp.name, hp.default)
            
            # Highlight modified values
            is_modified = val != hp.default
            val_str = f"**{val}**" if is_modified else f"{val}"
            modified_tag = " *(modified)*" if is_modified else ""
            
            st.markdown(f"- {hp.display_name}: {val_str}{modified_tag}")
            if hp.description:
              st.caption(f"  {hp.description}")

    # CLI Command Preview
    st.markdown("---")
    st.subheader("CLI Command")
    st.caption("Run this analysis from the terminal:")
    
    try:
      cli_cmd = _generate_cli_command(n_repetitions=n_repeats, seed=random_seed)
      st.code(cli_cmd, language="bash")
      
      st.download_button(
        label="Download Shell Script",
        data=f"#!/bin/bash\n# SCRBenchmark CLI Command\n# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{cli_cmd}\n",
        file_name=f"run_analysis_{time.strftime('%Y%m%d_%H%M%S')}.sh",
        mime="text/x-shellscript",
        width="stretch"
      )
    except Exception as e:
      st.warning(f"Could not generate CLI command: {e}")

    # Run button
    st.markdown("---")

    if st.button("Run Analysis", type="primary", width="stretch"):
      _run_analysis(handler, n_repeats, random_seed, compute_scib_metrics)

    # Display results (only in standard mode)
    if 'analysis_results' in st.session_state and st.session_state.analysis_results is not None:
      _display_results()


def _run_analysis(handler, n_repeats: int, random_seed: int, compute_scib_metrics: bool = True):
  """Run the analysis."""
  try:
    current_preproc = _snapshot_preprocessing_params(
      st.session_state.get('preprocessing_params', {})
    )
    applied_preproc = st.session_state.get('preprocessing_applied_params')
    if isinstance(applied_preproc, dict) and applied_preproc != current_preproc:
      st.error(
        "Preprocessing parameters changed since last preprocessing run. "
        "Please re-run 'Run Preprocessing' before launching analysis."
      )
      st.session_state.data_preprocessed = False
      return

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(message: str, progress: float = None):
      status_text.text(message)
      if progress is not None:
        progress_bar.progress(progress)

    runner = AnalysisRunner()

    # Apply batch balancing if enabled
    balance_settings = st.session_state.get('balance_settings_standard', {})
    if balance_settings:
      from utils.dataset_splitter import DatasetSplitter

      status_text.text("Balancing batches...")
      progress_bar.progress(0.05)

      splitter = DatasetSplitter(random_state=random_seed)
      original_adata = handler.get_data()

      batch_col = balance_settings.get('batch_col')
      target = balance_settings.get('target')

      if batch_col and target:
        # Get label column for stratified subsampling
        label_col = None
        for candidate in ['Group', 'labels', 'cell_type', 'cluster']:
          if candidate in original_adata.obs.columns:
            label_col = candidate
            break

        balanced_adata = splitter.balance_by_batch(
          original_adata,
          batch_col=batch_col,
          target_per_batch=target,
          preserve_labels=True,
          label_col=label_col or 'Group'
        )

        # Update handler with balanced data
        handler._adata = balanced_adata
        handler._preprocessed_adata = balanced_adata

        st.info(f"Balanced data: {balanced_adata.n_obs} cells (from {original_adata.n_obs})")

    # Get data and labels
    data = handler.get_data()
    labels = handler.get_labels()

    # Prepare params
    params = st.session_state.get('algorithm_params', {})

    # Add seed to all algorithms
    for algo_name in st.session_state.selected_algorithms:
      if algo_name not in params:
        params[algo_name] = {}
      params[algo_name]['random_state'] = random_seed
      params[algo_name]['random_state'] = random_seed
      params[algo_name]['seed'] = random_seed
      
      # Force scCDCG to use raw_original data if not specified
      if algo_name == 'sccdcg' and 'input_type' not in params[algo_name]:
        params[algo_name]['input_type'] = 'raw_original'

    # Run comparison
    # Pass the handler as 'data' so the proper data version can be fetched inside
    results = runner.run_comparison(
      algorithm_names=st.session_state.selected_algorithms,
      data=handler, 
      labels=None, # Labels will be fetched from handler based on input type
      params=params,
      n_repeats=n_repeats,
      compute_scib_metrics=compute_scib_metrics,
      progress_callback=progress_callback
    )

    st.session_state.analysis_results = results
    st.session_state.analysis_runner = runner

    progress_bar.progress(1.0)
    status_text.text("Analysis completed!")
    time.sleep(1)

    st.rerun()

  except Exception as e:
    display_error(
      e,
      user_message="Standard analysis failed.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="analysis_standard_run",
    )


def _display_results():
  """Display analysis results."""
  results = st.session_state.analysis_results

  st.markdown("---")
  st.subheader("Results")

  # Check if we have any results
  if not results.results:
    st.error("No results available. All algorithms may have failed. Check the terminal for error messages.")
    return

  # Summary table
  st.markdown("### Summary Statistics")

  summary_data = []
  # Prepare data for display (keep numeric values for ProgressColumn)
  display_data = []
  
  for algo_name, stats in results.summary.items():
    algo_info = AlgorithmRegistry.get(algo_name).get_info()
    
    # Row for DataFrame (numeric for progress bars)
    row = {'Algorithm': algo_info.display_name}
    for metric in ['NMI', 'ARI', 'ACC', 'Silhouette', 'F1_Macro', 'BalancedACC', 'BalancedRareACC']:
      if f'{metric}_mean' in stats:
        row[metric] = stats[f'{metric}_mean']
    
    if 'runtime_mean' in stats:
      row['Runtime (s)'] = stats['runtime_mean']
      
    # Get parameter count from the first result for this algorithm
    first_result = next((r for r in results.results if r.algorithm_name == algo_name), None)
    if first_result and getattr(first_result, 'num_parameters', None) is not None:
       row['Num Parameters'] = first_result.num_parameters
      
    display_data.append(row)

  df_summary = pd.DataFrame(display_data)
  
  st.dataframe(
    df_summary,
    column_config={
      "Num Parameters": st.column_config.NumberColumn(
        "Num Parameters", format="%d", help="Number of trainable parameters in the model"
      ),
      "NMI": st.column_config.ProgressColumn(
        "NMI", min_value=0, max_value=1, format="%.3f"
      ),
      "ARI": st.column_config.ProgressColumn(
        "ARI", min_value=0, max_value=1, format="%.3f"
      ),
      "ACC": st.column_config.ProgressColumn(
        "ACC", min_value=0, max_value=1, format="%.3f"
      ),
      "Silhouette": st.column_config.ProgressColumn(
        "Silhouette", min_value=-1, max_value=1, format="%.3f"
      ),
      "Runtime (s)": st.column_config.NumberColumn(
        "Runtime (s)", format="%.2f"
      ),
      "F1_Macro": st.column_config.ProgressColumn(
        "F1 (Macro)", min_value=0, max_value=1, format="%.3f"
      ),
      "BalancedACC": st.column_config.ProgressColumn(
        "Balanced ACC", min_value=0, max_value=1, format="%.3f"
      ),
      "BalancedRareACC": st.column_config.ProgressColumn(
        "Balanced Rare ACC", min_value=0, max_value=1, format="%.3f"
      ),
    },
    width="stretch"
  )

  # Show mean ± std when multiple repeats are available
  has_repeats = any(r.run_id > 0 for r in results.results)
  if has_repeats:
    st.markdown("### Summary (Mean ± Std)")
    summary_text = []
    for algo_name, stats in results.summary.items():
      algo_info = AlgorithmRegistry.get(algo_name).get_info()
      algo_info = AlgorithmRegistry.get(algo_name).get_info()
      row = {'Algorithm': algo_info.display_name}
      for metric in ['NMI', 'ARI', 'ACC', 'Silhouette', 'F1_Macro', 'BalancedACC', 'BalancedRareACC']:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        if mean_key in stats:
          if std_key in stats:
            row[metric] = f"{stats[mean_key]:.3f} ± {stats[std_key]:.3f}"
          else:
            row[metric] = f"{stats[mean_key]:.3f}"
      if 'runtime_mean' in stats:
        if 'runtime_std' in stats:
          row['Runtime (s)'] = f"{stats['runtime_mean']:.2f} ± {stats['runtime_std']:.2f}"
        else:
          row['Runtime (s)'] = f"{stats['runtime_mean']:.2f}"
      summary_text.append(row)

    df_summary_text = pd.DataFrame(summary_text)
    st.dataframe(df_summary_text, width="stretch")

  # Best algorithm
  if results.summary:
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    metrics = ['NMI', 'ARI', 'ACC', 'Silhouette', 'F1_Macro', 'BalancedACC', 'BalancedRareACC']
    cols = [col1, col2, col3, col4, col5, col6, col7]
    
    for i, metric in enumerate(metrics):
      best = results.get_best_algorithm(metric)
      if best:
        best_info = AlgorithmRegistry.get(best).get_info()
        best_val = results.summary[best].get(f'{metric}_mean', 0)
        with cols[i]:
          st.success(f"Best {metric}\n**{best_info.display_name}**\n({best_val:.3f})")

  # Show PCA components info if auto was used
  pca_info_shown = False
  for result in results.results:
    if result.extra_info.get('n_pca_components_used'):
      if not pca_info_shown:
        st.markdown("### PCA Components (Auto-detected)")
        pca_info_shown = True
      algo_info = AlgorithmRegistry.get(result.algorithm_name).get_info()
      n_comp = result.extra_info['n_pca_components_used']
      cum_var = result.extra_info.get('cumulative_variance', 0) * 100
      st.info(f"**{algo_info.display_name}**: {n_comp} components "
          f"(Cattell's scree test, {cum_var:.1f}% variance)")
      break # Only show once per algorithm

  # Detailed results
  with st.expander("Detailed Results"):
    detail_data = []
    for result in results.results:
      row = {
        'Algorithm': result.algorithm_name,
        'Run': result.run_id + 1,
        **result.metrics,
        'Runtime (s)': result.runtime
      }
      # Add PCA info if available
      if result.extra_info.get('n_pca_components_used'):
        row['PCA Components'] = result.extra_info['n_pca_components_used']
      if result.extra_info.get('num_parameters'):
        row['Num Parameters'] = result.extra_info['num_parameters']
      detail_data.append(row)

    df_detail = pd.DataFrame(detail_data)
    st.dataframe(df_detail, width="stretch")

  # Loss curves (if available)
  has_loss = any(getattr(r, 'loss_history', None) for r in results.results)
  if has_loss:
    with st.expander("Training Loss Curves"):
      try:
        import utils.visualization as viz
        import matplotlib.pyplot as plt_loss

        # Group by algorithm (show first run only to keep compact)
        seen_algos = set()
        for result in results.results:
          if not getattr(result, 'loss_history', None):
            continue
          if result.algorithm_name in seen_algos:
            continue
          seen_algos.add(result.algorithm_name)

          fig = viz.plot_loss_curves(result.loss_history, result.algorithm_name)
          if fig:
            st.pyplot(fig)
            plt_loss.close(fig)
      except Exception as e:
        st.warning(f"Could not display loss curves: {e}")

  # UMAP evolution (if available)
  has_umap_evolution = any(getattr(r, 'embedding_snapshots', None) for r in results.results)
  if has_umap_evolution:
    with st.expander("UMAP Evolution"):
      try:
        import matplotlib.pyplot as plt_umap

        selected_algorithms = sorted({r.algorithm_name for r in results.results})
        fig = _plot_umap_evolution_gallery_current(
          results,
          selected_algorithms=selected_algorithms,
          max_panels=18,
        )
        if fig is not None:
          st.pyplot(fig)
          plt_umap.close(fig)
        else:
          st.info("No UMAP evolution snapshots available for display.")
      except Exception as e:
        st.warning(f"Could not display UMAP evolution: {e}")

  # Visualization
  _render_visualizations(results)

  # Export
  st.markdown("---")
  st.subheader("Export Results")

  col1, col2 = st.columns(2)

  with col1:
    if st.button("Export to CSV"):
      _export_csv(results)

  with col2:
    if st.button("Export to JSON"):
      runner = st.session_state.analysis_runner
      path = runner.save_results()
      st.success(f"Saved to {path}")


def _safe_json_for_explorer(value):
  """Serialize complex values for Results Explorer compatibility."""
  def _default(obj):
    if isinstance(obj, (np.integer, np.floating)):
      return obj.item()
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, (set, tuple)):
      return list(obj)
    return str(obj)

  try:
    return json.dumps(value, ensure_ascii=False, default=_default)
  except Exception:
    return str(value)


def _build_explorer_label_map(handler) -> dict:
  """Convert DataHandler label_map ({name: id}) to explorer format ({id: name})."""
  if handler is None:
    return {}
  try:
    adata = handler.get_data()
  except Exception:
    adata = None
  if adata is None or not hasattr(adata, "uns"):
    return {}

  raw_map = adata.uns.get("label_map")
  if not isinstance(raw_map, dict):
    return {}

  mapped = {}
  for k, v in raw_map.items():
    try:
      mapped[str(int(v))] = str(k)
    except Exception:
      continue
  return mapped


def _build_explorer_payload_from_current_results(results):
  """
  Build an in-memory payload compatible with gui.results_explorer helpers.
  """
  if not hasattr(results, "results") or not results.results:
    return None, None, None, [], []

  from gui import results_explorer as rex

  condition_name = "run_analysis_current"
  runner = st.session_state.get("analysis_runner")
  output_path = str(getattr(runner, "output_dir", "results"))
  handler = st.session_state.get("data_handler")
  adata = handler.get_data() if handler is not None else None
  label_map = _build_explorer_label_map(handler)

  first_result = results.results[0]
  is_benchmark = hasattr(first_result, "benchmark_metrics")

  if is_benchmark:
    rows = []
    labels_data = {}
    for result in results.results:
      bm = result.benchmark_metrics
      row = {
        "algorithm": result.algorithm_name,
        "algorithm_name": result.algorithm_name,
        "run_id": int(result.run_id),
        "fit_time": float(result.fit_time),
        "predict_time": float(result.predict_time),
        "total_time": float(result.total_time),
      }

      for metric_name, metric_value in (bm.train_metrics or {}).items():
        if isinstance(metric_value, (int, float, np.integer, np.floating)):
          row[f"train_{metric_name}"] = float(metric_value)
      for metric_name, metric_value in (bm.test_metrics or {}).items():
        if isinstance(metric_value, (int, float, np.integer, np.floating)):
          row[f"test_{metric_name}"] = float(metric_value)
      for metric_name, metric_value in (bm.val_metrics or {}).items():
        if isinstance(metric_value, (int, float, np.integer, np.floating)):
          row[f"val_{metric_name}"] = float(metric_value)
      for metric_name, metric_value in (bm.generalization_gap or {}).items():
        if isinstance(metric_value, (int, float, np.integer, np.floating)):
          row[f"{metric_name}_gap"] = float(metric_value)

      row["test_by_group"] = _safe_json_for_explorer(bm.test_by_group or {})
      row["error_analysis"] = _safe_json_for_explorer(bm.error_analysis or {})
      rows.append(row)

      for split in ["train", "test", "val"]:
        pred = getattr(result, f"{split}_labels", None)
        if pred is None:
          continue
        pred = np.asarray(pred)
        if pred.size == 0:
          continue

        n = len(pred)
        df_labels = pd.DataFrame({"predicted_label": pred})
        true_split = getattr(result, f"{split}_true_labels", None)
        if true_split is not None and len(true_split) == n:
          df_labels["true_label"] = np.asarray(true_split)
        batch_split = getattr(result, f"{split}_batch_ids", None)
        if batch_split is not None and len(batch_split) == n:
          df_labels["batch"] = np.asarray(batch_split).astype(str)

        labels_data.setdefault(result.algorithm_name, {})[
          f"run{int(result.run_id)}_{split}"
        ] = df_labels

    df = pd.DataFrame(rows)
    all_data = {
      condition_name: {
        "df": df,
        "type": "benchmark_detailed",
        "condition": condition_name,
        "path": output_path,
        "labels": labels_data,
        "label_map": label_map,
      }
    }
  else:
    rows = []
    labels_data = {}
    global_true = None
    if handler is not None and hasattr(handler, "get_labels"):
      try:
        global_true = handler.get_labels(data_source="processed")
      except Exception:
        global_true = handler.get_labels()

    batch_col = get_batch_column(adata) if adata is not None else None

    for result in results.results:
      row = {
        "algorithm_name": result.algorithm_name,
        "algorithm": result.algorithm_name,
        "run_id": int(result.run_id),
        "runtime": float(result.runtime),
      }
      for metric_name, metric_value in (result.metrics or {}).items():
        if isinstance(metric_value, dict):
          row[metric_name] = _safe_json_for_explorer(metric_value)
        elif isinstance(metric_value, (int, float, np.integer, np.floating)):
          row[metric_name] = float(metric_value)
      rows.append(row)

      pred = np.asarray(result.labels) if result.labels is not None else None
      if pred is None or pred.size == 0:
        continue

      n = len(pred)
      df_labels = pd.DataFrame({"predicted_label": pred})
      true_arr = np.asarray(result.true_labels) if result.true_labels is not None else None
      if true_arr is None and global_true is not None:
        global_true_arr = np.asarray(global_true)
        if len(global_true_arr) == n:
          true_arr = global_true_arr
      if true_arr is not None and len(true_arr) == n:
        df_labels["true_label"] = true_arr
      if (
        batch_col
        and adata is not None
        and batch_col in adata.obs.columns
        and len(adata.obs[batch_col]) == n
      ):
        df_labels["batch"] = adata.obs[batch_col].to_numpy().astype(str)

      labels_data.setdefault(result.algorithm_name, {})[
        f"run{int(result.run_id)}_full"
      ] = df_labels

    df = pd.DataFrame(rows)
    all_data = {
      condition_name: {
        "df": df,
        "type": "analysis_results",
        "condition": condition_name,
        "path": output_path,
        "labels": labels_data,
        "label_map": label_map,
      }
    }

  agg_df = rex.aggregate_metrics(all_data)
  all_algorithms = sorted(
    {
      getattr(row, "algorithm_name", None)
      for row in results.results
      if getattr(row, "algorithm_name", None) is not None
    }
  )
  all_conditions = [condition_name]
  return rex, all_data, agg_df, all_algorithms, all_conditions


def _figure_to_rgb_array(fig):
  """Render a Matplotlib figure into an RGB numpy image."""
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  return buf.reshape(h, w, 3)


def _compose_figure_gallery(figure_entries, title: str):
  """Compose multiple matplotlib figures into a single gallery figure."""
  if not figure_entries:
    return None

  import matplotlib.pyplot as plt_gallery

  if len(figure_entries) == 1:
    label, fig_single = figure_entries[0]
    fig_single.suptitle(label, fontsize=12, fontweight="bold")
    return fig_single

  n = len(figure_entries)
  n_cols = min(2, n)
  n_rows = (n + n_cols - 1) // n_cols
  fig_out, axes = plt_gallery.subplots(
    n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows), squeeze=False
  )

  for i, (label, src_fig) in enumerate(figure_entries):
    r, c = divmod(i, n_cols)
    ax = axes[r][c]
    try:
      img = _figure_to_rgb_array(src_fig)
      ax.imshow(img)
      ax.set_title(label, fontsize=10, fontweight="bold")
      ax.axis("off")
    finally:
      plt_gallery.close(src_fig)

  for i in range(n, n_rows * n_cols):
    r, c = divmod(i, n_cols)
    axes[r][c].axis("off")

  fig_out.suptitle(title, fontsize=13, fontweight="bold")
  fig_out.tight_layout()
  return fig_out


def _plot_loss_curves_gallery_current(results, selected_algorithms):
  """Build a loss curve gallery from in-memory run_analysis results."""
  entries = []
  for result in results.results:
    if result.algorithm_name not in selected_algorithms:
      continue
    if not getattr(result, "loss_history", None):
      continue
    fig_local = viz.plot_loss_curves(result.loss_history, result.algorithm_name)
    if fig_local is not None:
      algo_name = AlgorithmRegistry.get(result.algorithm_name).get_info().display_name
      entries.append((f"{algo_name} (run {int(result.run_id) + 1})", fig_local))
  return _compose_figure_gallery(entries, "Loss curves (run_analysis)")


def _plot_umap_evolution_gallery_current(results, selected_algorithms, max_panels: int = 18):
  """Build a UMAP-evolution gallery from in-memory run_analysis results."""
  entries = []
  for result in results.results:
    if result.algorithm_name not in selected_algorithms:
      continue
    snapshots = getattr(result, "embedding_snapshots", None) or []
    if not snapshots:
      continue

    first_valid_snapshot = next(
      (
        s for s in snapshots
        if s.get("embeddings") is not None and len(s.get("embeddings")) > 0
      ),
      None,
    )
    if first_valid_snapshot is None:
      continue

    labels_for_plot = result.true_labels if result.true_labels is not None else result.labels
    if labels_for_plot is None or len(labels_for_plot) != len(first_valid_snapshot.get("embeddings")):
      continue

    fig_local = viz.plot_umap_evolution(
      embedding_snapshots=snapshots,
      labels=labels_for_plot,
      algorithm_name=result.algorithm_name,
    )
    if fig_local is not None:
      algo_name = AlgorithmRegistry.get(result.algorithm_name).get_info().display_name
      entries.append((f"{algo_name} (run {int(result.run_id) + 1})", fig_local))
      if max_panels > 0 and len(entries) >= max_panels:
        break

  return _compose_figure_gallery(entries, "UMAP evolution (run_analysis)")


def _render_results_explorer_tools(results, key_prefix: str):
  """Render the same visualization toolbox as results_explorer on current in-memory results."""
  st.markdown("### Results Explorer Tools")
  st.caption(
    "Same visualization engine as `results_explorer`, applied directly to this run's results "
    "(without reloading folders)."
  )

  try:
    payload = _build_explorer_payload_from_current_results(results)
  except Exception as e:
    st.warning(f"Could not initialize Results Explorer tools: {e}")
    return

  if not payload or payload[0] is None:
    st.info("No compatible data available for Results Explorer tools.")
    return

  rex, all_data, agg_df, default_algorithms, default_conditions = payload
  if not default_algorithms:
    st.info("No algorithm outputs available for advanced visualization.")
    return

  with st.expander("Open Results Explorer-Compatible Visualizations", expanded=False):
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
      selected_algorithms = st.multiselect(
        "Algorithms",
        options=default_algorithms,
        default=default_algorithms,
        format_func=lambda x: rex.ALGO_DISPLAY_NAMES.get(x, x),
        key=f"{key_prefix}_rex_algos",
      )
    with col_sel2:
      selected_conditions = st.multiselect(
        "Conditions",
        options=default_conditions,
        default=default_conditions,
        key=f"{key_prefix}_rex_conds",
      )

    if not selected_algorithms or not selected_conditions:
      st.warning("Select at least one algorithm and one condition.")
      return

    figure_type = st.selectbox(
      "Results Explorer Figure Type",
      options=list(rex.FIGURE_TYPES.keys()),
      format_func=lambda x: rex.FIGURE_TYPES[x]["name"],
      key=f"{key_prefix}_rex_figure_type",
    )
    st.caption(rex.FIGURE_TYPES[figure_type]["description"])

    metric = "NMI"
    metric_figures = [
      "algorithm_comparison",
      "metrics_heatmap",
      "generalization_gap",
      "generalization_gap_heatmap",
      "statistical_test",
    ]
    if figure_type in metric_figures:
      require_train_test = figure_type in ["generalization_gap", "generalization_gap_heatmap"]
      metric_options = rex.get_available_metrics(agg_df, require_train_test=require_train_test)
      if not metric_options:
        metric_options = ["NMI", "ARI", "ACC", "Silhouette"]
      metric = st.selectbox(
        "Target Metric",
        options=metric_options,
        key=f"{key_prefix}_rex_metric",
      )

    test_mode = "pairwise_conditions"
    if figure_type == "statistical_test":
      test_mode = st.radio(
        "Statistical comparison mode",
        options=["pairwise_conditions", "pairwise_algorithms"],
        format_func=lambda x: (
          "Compare conditions (per algorithm)"
          if x == "pairwise_conditions"
          else "Compare algorithms (per condition)"
        ),
        key=f"{key_prefix}_rex_test_mode",
      )

    export_publication_format = "latex"
    if figure_type == "publication_export":
      export_publication_format = st.radio(
        "Publication export format",
        options=["latex", "markdown"],
        horizontal=True,
        key=f"{key_prefix}_rex_pub_fmt",
      )

    umap_split = "all"
    umap_max_images = 18
    if figure_type in {"saved_umap_gallery", "umap_evolution"}:
      if figure_type == "saved_umap_gallery":
        umap_split = st.selectbox(
          "UMAP Split",
          options=["all", "train", "test", "val", "full"],
          key=f"{key_prefix}_rex_umap_split",
        )
      umap_max_images = st.slider(
        "Max gallery images",
        min_value=1,
        max_value=60,
        value=18,
        step=1,
        key=f"{key_prefix}_rex_umap_max",
      )

    diag_condition = selected_conditions[0]
    diag_algorithm = selected_algorithms[0]
    diag_run_id = 0
    diag_split = "test"
    diag_outline_mode = "none"
    diag_max_points = 15000
    diag_seed = 42
    diag_show_centroids = True
    if figure_type == "umap_diagnostic":
      diag_entries = rex.list_umap_diagnostic_entries(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
      if diag_entries:
        cond_opts = sorted({e["condition"] for e in diag_entries})
        diag_condition = st.selectbox(
          "Diagnostic condition",
          options=cond_opts,
          key=f"{key_prefix}_rex_diag_condition",
        )

        cond_entries = [e for e in diag_entries if e["condition"] == diag_condition]
        algo_opts = sorted({e["algorithm"] for e in cond_entries})
        diag_algorithm = st.selectbox(
          "Diagnostic algorithm",
          options=algo_opts,
          format_func=lambda x: rex.ALGO_DISPLAY_NAMES.get(x, x),
          key=f"{key_prefix}_rex_diag_algo",
        )

        algo_entries = [e for e in cond_entries if e["algorithm"] == diag_algorithm]
        run_opts = sorted({int(e["run_id"]) for e in algo_entries})
        diag_run_id = st.selectbox(
          "Run",
          options=run_opts,
          format_func=lambda x: f"run{x}",
          key=f"{key_prefix}_rex_diag_run",
        )

        run_entries = [e for e in algo_entries if int(e["run_id"]) == int(diag_run_id)]
        split_opts = [s for s in ["train", "test", "val", "full"] if any(e["split"] == s for e in run_entries)]
        if split_opts:
          diag_split = st.selectbox(
            "Split",
            options=split_opts,
            key=f"{key_prefix}_rex_diag_split",
          )

        diag_outline_mode = st.selectbox(
          "Cluster outline mode",
          options=["none", "convex_hull", "ellipse", "density"],
          key=f"{key_prefix}_rex_diag_outline",
        )
        diag_show_centroids = st.checkbox(
          "Show cluster centroids",
          value=True,
          key=f"{key_prefix}_rex_diag_centroids",
        )
        diag_max_points = st.slider(
          "Max cells for diagnostic",
          min_value=2000,
          max_value=60000,
          value=15000,
          step=1000,
          key=f"{key_prefix}_rex_diag_points",
        )
        diag_seed = st.number_input(
          "Diagnostic random seed",
          min_value=0,
          max_value=1_000_000,
          value=42,
          step=1,
          key=f"{key_prefix}_rex_diag_seed",
        )
      else:
        st.info("No UMAP diagnostic entries found for current run.")

    with st.expander("Figure Style", expanded=False):
      fig_context = st.selectbox(
        "Context",
        options=["notebook", "paper", "talk", "poster"],
        index=0,
        key=f"{key_prefix}_rex_context",
      )
      fig_style = st.selectbox(
        "Style",
        options=["whitegrid", "darkgrid", "ticks", "white"],
        index=0,
        key=f"{key_prefix}_rex_style",
      )
      export_format = st.selectbox(
        "Export format",
        options=["png", "svg", "pdf"],
        index=0,
        key=f"{key_prefix}_rex_export_fmt",
      )
      fig_dpi = st.number_input(
        "DPI",
        value=300,
        min_value=72,
        max_value=600,
        step=50,
        key=f"{key_prefix}_rex_dpi",
      )
      cld_figures = [
        "algorithm_comparison",
        "generalization_gap",
        "generalization_gap_combined",
        "runtime_comparison",
      ]
      show_cld = False
      if getattr(rex, "HAS_CLD", False) and figure_type in cld_figures:
        show_cld = st.checkbox(
          "Show CLD significance letters",
          value=False,
          key=f"{key_prefix}_rex_show_cld",
        )
      elif figure_type in cld_figures:
        st.caption("CLD unavailable (utils.statistics import failed).")

    if not st.button("Generate Explorer Visualization", type="primary", key=f"{key_prefix}_rex_generate"):
      return

    sns.set_theme(context=fig_context, style=fig_style, palette="deep")

    if figure_type == "statistical_test":
      stats_df = rex.perform_statistical_test(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        metric=metric,
        test_mode=test_mode,
      )
      if stats_df.empty:
        st.info("Not enough data to run statistical tests (need repeated runs).")
      else:
        st.dataframe(stats_df, use_container_width=True)
      return

    if figure_type == "summary_table":
      summary_df = rex.create_summary_dataframe(
        agg_df=agg_df,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
      st.dataframe(summary_df, use_container_width=True)
      return

    if figure_type == "metrics_by_celltype_table":
      metrics_df = rex.create_metrics_by_celltype_table(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
      if metrics_df.empty:
        st.info("No label data available for per-cell-type metrics.")
      else:
        st.dataframe(metrics_df, use_container_width=True)
      return

    if figure_type == "publication_export":
      output = rex.export_publication_ready(
        agg_df=agg_df,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        format=export_publication_format,
      )
      st.code(output, language="latex" if export_publication_format == "latex" else "markdown")
      return

    fig = None
    if figure_type == "algorithm_comparison":
      fig = rex.plot_algorithm_comparison(
        agg_df=agg_df,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        metric=metric,
        all_data=all_data,
        show_cld=show_cld,
      )
    elif figure_type == "metrics_heatmap":
      fig = rex.plot_metrics_heatmap(
        agg_df=agg_df,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        metric=metric,
      )
    elif figure_type == "generalization_gap":
      fig = rex.plot_generalization_gap_boxplot(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        metric=metric,
        show_cld=show_cld,
      )
    elif figure_type == "generalization_gap_heatmap":
      fig = rex.plot_generalization_gap_heatmap_fig(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        metric=metric,
      )
    elif figure_type == "runtime_comparison":
      fig = rex.plot_runtime_comparison(
        agg_df=agg_df,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        all_data=all_data,
        show_cld=show_cld,
      )
    elif figure_type == "celltype_errors":
      fig = rex.analyze_celltype_errors_fig(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        sort_by="n_samples",
      )
    elif figure_type == "celltype_errors_by_batch":
      fig = rex.analyze_celltype_errors_by_batch_fig(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        sort_by="n_samples",
      )
    elif figure_type == "confusion_patterns":
      fig = rex.plot_confusion_patterns_fig(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
    elif figure_type == "confusion_matrix_detailed":
      fig = rex.plot_confusion_matrix_detailed(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        sort_by="n_samples",
      )
    elif figure_type == "train_vs_test":
      fig = rex.plot_train_vs_test_comparison(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
    elif figure_type == "generalization_gap_combined":
      fig = rex.plot_generalization_gap_combined(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        show_cld=show_cld,
      )
    elif figure_type == "batch_generalization":
      fig = rex.plot_batch_generalization_fig(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
    elif figure_type == "test_metrics_by_batch":
      fig = rex.plot_test_metrics_by_batch(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
    elif figure_type == "saved_umap_gallery":
      fig = rex.plot_saved_umap_gallery(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
        split_filter=umap_split,
        max_images=int(umap_max_images),
      )
    elif figure_type == "umap_evolution":
      fig = _plot_umap_evolution_gallery_current(
        results=results,
        selected_algorithms=selected_algorithms,
        max_panels=int(umap_max_images),
      )
    elif figure_type == "umap_diagnostic":
      fig = rex.plot_umap_diagnostic_from_results(
        all_data=all_data,
        condition=diag_condition,
        algorithm=diag_algorithm,
        run_id=int(diag_run_id),
        split=diag_split,
        outline_mode=diag_outline_mode,
        max_points=int(diag_max_points),
        random_state=int(diag_seed),
        show_centroids=bool(diag_show_centroids),
      )
    elif figure_type == "error_rate_by_batch":
      fig = rex.plot_error_rate_by_batch(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
    elif figure_type == "confusion_matrix_by_batch":
      maybe_fig = rex.plot_confusion_matrix_by_batch(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
      if maybe_fig == "no_batch_column":
        st.warning("No `batch` column available in labels for confusion-by-batch plotting.")
        fig = None
      else:
        fig = maybe_fig
    elif figure_type == "loss_curves":
      fig = _plot_loss_curves_gallery_current(results, selected_algorithms)
    elif figure_type == "batch_composition":
      maybe_fig = rex.plot_batch_composition(
        all_data=all_data,
        selected_algos=selected_algorithms,
        selected_conditions=selected_conditions,
      )
      if maybe_fig == "no_batch_column":
        st.warning("No `batch` column available for batch composition plotting.")
        fig = None
      else:
        fig = maybe_fig

    if fig is None:
      st.info("No figure generated for this selection.")
      return

    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format=export_format, bbox_inches="tight", dpi=int(fig_dpi))
    st.download_button(
      label=f"Download ({export_format.upper()})",
      data=buf.getvalue(),
      file_name=f"{figure_type}.{export_format}",
      mime=(
        "image/png" if export_format == "png"
        else "image/svg+xml" if export_format == "svg"
        else "application/pdf"
      ),
      key=f"{key_prefix}_rex_download",
      width="stretch",
    )


def _render_visualizations(results):
  """Render result visualizations with a gallery selector."""
  import matplotlib.pyplot as plt

  st.markdown("### Visualization Gallery")

  # Define available plots
  plot_options = {
    "radar": "Radar Chart (Performance Profile)",
    "boxplot": "Statistical Comparison (Boxplots)",
    "confusion": "Confusion Matrix (Clusters vs Labels)",
    "silhouette": "Silhouette Analysis (Cluster Quality)",
  }
  
  # Check if the results object is a BenchmarkComparisonResult
  is_benchmark_results = hasattr(results, 'summary') and any(any(k.startswith('test_') for k in v.keys()) for v in results.summary.values())
  
  if is_benchmark_results:
    plot_options["benchmark_comp"] = "Benchmark Comparison (Train vs Test)"
    plot_options["batch_heatmap"] = "Batch Performance (Heatmap)"

  # Select plot type
  selected_plot_key = st.selectbox(
    "Choose Visualization",
    options=list(plot_options.keys()),
    format_func=lambda x: plot_options[x],
    index=0
  )
  
  fig = None
  filename = "plot.png"
  
  # -------------------------------------------------------------------------
  # 1. Radar Chart
  # -------------------------------------------------------------------------
  if selected_plot_key == "radar":
    st.caption("Compare algorithms across multiple metrics simultaneously (Mean values).")
    try:
      fig = viz.plot_radar_chart(results.summary)
      filename = "radar_chart_performance.png"
    except Exception as e:
      st.warning(f"Could not create Radar Chart: {e}")

  # -------------------------------------------------------------------------
  # 2. Boxplots (Statistical Comparison)
  # -------------------------------------------------------------------------
  elif selected_plot_key == "boxplot":
    st.caption("Distribution of scores across multiple runs with statistical significance tests.")

    # Statistical test settings
    with st.expander("Statistical Settings", expanded=False):
      sc1, sc2, sc3 = st.columns(3)
      with sc1:
        show_stats = st.checkbox("Show CLD letters", value=True)
      with sc2:
        stat_method = st.selectbox(
          "Method",
          ['nonparametric', 'parametric'],
          format_func=lambda x: 'Kruskal-Wallis' if x == 'nonparametric' else 'ANOVA'
        )
      with sc3:
        alpha = st.selectbox("Alpha", [0.01, 0.05, 0.1], index=1)

    if len(results.results) > 0:
      fig = viz.plot_metrics_comparison(
        results.summary,
        results.results,
        show_stats=show_stats,
        stat_method=stat_method,
        alpha=alpha
      )
      filename = "metrics_boxplots_comparison.png"

  # -------------------------------------------------------------------------
  # 3. Confusion Matrix
  # -------------------------------------------------------------------------
  # -------------------------------------------------------------------------
  # 3. Confusion Matrix
  # -------------------------------------------------------------------------
  elif selected_plot_key == "confusion":
    st.caption("Compare predicted clusters to true cell type labels for each algorithm.")

    # Get true labels from session state
    handler = st.session_state.get('data_handler')
    true_labels = None

    if handler and handler.adata is not None:
      adata = handler.adata
      # Priority columns for cell types
      for col in ['Group', 'labels', 'celltype', 'cell_type']:
        if col in adata.obs.columns:
          # Get the correct subset of labels if this is a benchmark (test set only)
          # For now, let's use the labels from the first result as a length check
          first_res = results.results[0]
          if len(first_res.labels) < adata.n_obs:
            # Extract labels corresponding to the results (usually the test set)
            # The runner should ideally store labels_true in result
            if hasattr(first_res, 'true_labels') and first_res.true_labels is not None:
               true_labels = first_res.true_labels
            else:
               # Fallback: assume it's the last N cells (approximate)
               true_labels = adata.obs[col].values[-len(first_res.labels):]
          else:
            true_labels = adata.obs[col].values
          st.info(f"Using '{col}' as ground truth labels")
          break

    if true_labels is None:
      st.warning("No ground truth labels found in data. Cannot display confusion matrix.")
    elif len(results.results) > 0:
      # Options
      col1, col2 = st.columns(2)
      with col1:
        normalize_cm = st.checkbox("Normalize (by row)", value=True,
                     help="Show proportions instead of counts")
      with col2:
        max_algos = st.slider("Max algorithms to show", 1, 6, 4)

      try:
        fig = viz.plot_confusion_matrix_multi(
          results.results,
          true_labels,
          normalize=normalize_cm,
          max_algorithms=max_algos
        )
        filename = "confusion_matrices.png"
      except Exception as e:
        st.error(f"Could not create confusion matrix: {e}")
    else:
      st.warning("No results available for confusion matrix.")

  # -------------------------------------------------------------------------
  # 4. Silhouette Analysis
  # -------------------------------------------------------------------------
  elif selected_plot_key == "silhouette":
    st.caption("Detailed view of cluster cohesion and separation.")
    
    # Algorithm selection for silhouette
    algo_names = list(results.summary.keys())
    selected_algo = st.selectbox("Choose Algorithm", options=algo_names, 
                  format_func=lambda x: AlgorithmRegistry.get(x).get_info().display_name)
    
    # Get result
    algo_result = next((r for r in results.results if r.algorithm_name == selected_algo), None)
    
    if algo_result and algo_result.embeddings is not None:
      from sklearn.metrics import silhouette_samples
      with st.spinner("Computing silhouette samples..."):
        try:
          sample_scores = silhouette_samples(algo_result.embeddings, algo_result.labels)
          fig = viz.plot_cluster_silhouette(
            sample_scores,
            algo_result.labels,
            title=f"Silhouette Analysis: {AlgorithmRegistry.get(selected_algo).get_info().display_name}"
          )
          filename = f"silhouette_{selected_algo}.png"
        except Exception as e:
          st.warning(f"Could not create silhouette plot: {e}")
    else:
      st.warning("Embeddings not available for this individual run or algorithm. Ensure the algorithm returns embeddings.")

  # -------------------------------------------------------------------------
  # 5. Benchmark Comparison
  # -------------------------------------------------------------------------
  elif selected_plot_key == "benchmark_comp":
    st.caption("Comparison of performance metrics between training and testing sets.")
    try:
      fig = viz.plot_benchmark_comparison(results.summary)
      filename = "benchmark_train_vs_test.png"
    except Exception as e:
      st.error(f"Could not create benchmark comparison: {e}")

  # -------------------------------------------------------------------------
  # 6. Batch Heatmap
  # -------------------------------------------------------------------------
  elif selected_plot_key == "batch_heatmap":
    st.caption("Heatmap of metrics across different batches/datasets (Test set).")
    
    # This requires grouping results by batch
    if hasattr(results, 'test_by_group') and results.test_by_group:
      # We need to pick one algorithm or show aggregate? 
      # plot_batch_metrics_heatmap expects a dataframe with Batch column
      algo_names = list(results.summary.keys())
      selected_algo = st.selectbox("Choose Algorithm", options=algo_names,
                     format_func=lambda x: AlgorithmRegistry.get(x).get_info().display_name,
                     key="batch_heatmap_algo")
      
      # Extract data for this algo from results.test_by_group 
      # Note: results.test_by_group is typically Dict[algo_name, Dict[batch_name, metrics]]
      # Let's verify results structure for BenchmarkComparisonResult
      batch_data = []
      if selected_algo in results.test_by_group:
        for batch, metrics in results.test_by_group[selected_algo].items():
          row = {'Batch': batch}
          row.update(metrics)
          batch_data.append(row)
        
        if batch_data:
          df_batch = pd.DataFrame(batch_data)
          fig = viz.plot_batch_metrics_heatmap(df_batch, 
                            title=f"Batch Performance: {AlgorithmRegistry.get(selected_algo).get_info().display_name}")
          filename = f"batch_metrics_{selected_algo}.png"
        else:
          st.warning("No batch-wise metrics found for this algorithm.")
      else:
        st.warning("Test set batch evaluation not found.")
    else:
       # Fallback if structure is different
       st.warning("Batch-wise performance data is not available for this run.")
  
  # Display Plot
  if fig:
    st.pyplot(fig)
    
    # Download Button
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    st.download_button(
      label=f"Download {plot_options[selected_plot_key]}",
      data=buf.getvalue(),
      file_name=filename,
      mime="image/png",
      width="stretch"
    )
    plt.close()

  # Independent Inspectors (Always visible or in their own section)
  st.markdown("---")
  _render_cell_inspector(results)
  _render_group_inspector(results)
  
  # NEW: Batch and Cell Type Gene Inspectors
  st.markdown("---")
  _render_batch_gene_inspector(results)
  
  st.markdown("---")
  _render_celltype_gene_inspector(results)
  
  # UMAP is distinct enough to keep separate or add to gallery later if desired
  # For now, keeping it as a major standalone visualization
  st.markdown("---")
  _render_umap(results)

  st.markdown("---")
  _render_results_explorer_tools(results, key_prefix="run_analysis_standard")


def _render_group_inspector(results):
  """Render inspector to see top marker genes for a specific cluster."""
  st.markdown("### Group Gene Inspector")
  st.caption("Identify marker genes that distinguish a cluster from the rest using differential expression analysis.")

  if 'data_handler' not in st.session_state or st.session_state.data_handler is None:
    return

  handler = st.session_state.data_handler
  adata = handler.get_data()
  
  if adata is None:
    return

  # Check available results
  if not results.results:
    return
    
  col1, col2 = st.columns(2)
  
  with col1:
    # Algorithm selection
    algo_names = [r.algorithm_name for r in results.results]
    # Use display names for better UX
    algo_display_map = {
      name: AlgorithmRegistry.get(name).get_info().display_name 
      for name in algo_names
    }
    
    selected_algo_name = st.selectbox(
      "Select Algorithm",
      options=algo_names,
      format_func=lambda x: algo_display_map.get(x, x),
      key="group_inspector_algo"
    )
    
  # Get result for selected algo
  selected_result = next((r for r in results.results if r.algorithm_name == selected_algo_name), None)
  
  if not selected_result:
    return

  # Get unique clusters
  labels = selected_result.labels
  unique_clusters = np.unique(labels)
  unique_clusters.sort()
  
  with col2:
    selected_cluster = st.selectbox(
      "Select Cluster",
      options=unique_clusters,
      key="group_inspector_cluster"
    )

  # Compute marker genes
  if st.button("Find Marker Genes", key="analyze_genes_btn", type="primary"):
    with st.spinner(f"Computing differential expression for Cluster {selected_cluster} (Wilcoxon)..."):
      from utils.statistics import compute_marker_genes
      
      try:
        # Run DE analysis
        # We ask for top 100 genes initially to allow for filtering
        df_markers = compute_marker_genes(
          adata, 
          labels=labels, 
          target_cluster=selected_cluster, 
          method='wilcoxon',
          n_genes=200
        )
        
        # Filter for significant results
        df_sig = df_markers[
          (df_markers['Adj P-value'] < 0.05) & 
          (df_markers['Log2FC'] > 0.5)
        ].sort_values('Score', ascending=False)
        
        # Global Markers for DotPlot
        st.markdown("---")
        if st.checkbox("Show DotPlot (Top 5 markers for ALL clusters)", value=False):
          with st.spinner("Preparing DotPlot for all clusters..."):
            # We need top markers for ALL groups to make a good dotplot
            from utils.statistics import compute_marker_genes
            marker_dict = {}
            for cluster in unique_clusters:
              # Faster if we just get top 5
              df_c = compute_marker_genes(adata, labels=labels, target_cluster=cluster, n_genes=5)
              marker_dict[str(cluster)] = df_c['Gene'].tolist()
            
            # Store main cluster key for the viz module
            adata.uns['main_cluster_key'] = 'temp_clusters'
            adata.obs['temp_clusters'] = pd.Categorical(labels)
            
            fig_dot = viz.plot_marker_dotplot(adata, marker_dict)
            if fig_dot:
              st.pyplot(fig_dot)
            else:
              st.warning("Could not generate DotPlot. Ensure genes exist in the dataset.")

        # Tabs for Table vs Plot
        tab1, tab2 = st.tabs(["Top Marker Genes", "Volcano Plot"])
        
        with tab1:
          st.write(f"Found **{len(df_sig)}** significant marker genes (Adj P < 0.05, Log2FC > 0.5)")
          
          st.dataframe(
            df_sig.head(50),
            column_config={
              "Gene": "Gene Symbol",
              "Log2FC": st.column_config.NumberColumn(format="%.2f", help="Log2 Fold Change: Measures how much expression is higher in this cluster compared to others (log base 2)."),
              "P-value": st.column_config.NumberColumn(format="%.2e", help="Raw P-value from Wilcoxon test."),
              "Adj P-value": st.column_config.NumberColumn(format="%.2e", help="Adjusted P-value (Benjamini-Hochberg) to correct for multiple testing."),
              "Mean Expr": st.column_config.NumberColumn(format="%.2f", help="Mean gene expression in this cluster."),
              "Score": st.column_config.NumberColumn(format="%.1f", help="Z-score: Standardized score from statistical test.")
            },
            width="stretch",
            hide_index=True
          )

          with st.expander("Calculation Details (Methodology)"):
            st.markdown("""
            **Calculation Method:**
            *  **Statistical Test:** Wilcoxon Rank-Sum test via `scanpy.tl.rank_genes_groups`.
            *  **Log2FC:** `log2(Mean of cluster) - log2(Mean of rest)`. A value of 1 means expression is doubled.
            *  **Adj P-value:** Benjamini-Hochberg correction to control False Discovery Rate (FDR).
            *  **Score:** Z-score approximating the test statistic.
            """)
        
        with tab2:
          st.caption("Visualization of significance (Y) vs magnitude of change (X).", 
                help="Genes in top-right are the most reliable markers (strongly upregulated and highly significant).")
          
          fig = viz.plot_volcano(
            df_markers, 
            title=f"Volcano Plot: Cluster {selected_cluster} vs Rest",
            p_threshold=0.05,
            fc_threshold=1.0
          )
          if fig:
            st.pyplot(fig)
            
            # Export
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
            st.download_button(
              "Download Volcano Plot",
              data=buf.getvalue(),
              file_name=f"volcano_cluster_{selected_cluster}.png",
              mime="image/png"
            )
            
        # Top genes barplot (simplified view)
        st.markdown("#### Top 10 Upregulated Genes")
        if not df_sig.empty:
          fig_bar = viz.plot_top_genes(
            df_sig.rename(columns={'Score': 'Mean Expression'}), # Reuse existing plotting func logic
            title=f"Top Markers by Z-Score (Cluster {selected_cluster})"
          )
          st.pyplot(fig_bar)
          
      except Exception as e:
        st.error(f"Error computing marker genes: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_batch_gene_inspector(results):
  """Render inspector to analyze highly expressed genes by batch."""
  st.markdown("### Batch Gene Inspector")
  st.caption("Analyze the most highly expressed genes in each batch (technical/experimental groups).")

  if 'data_handler' not in st.session_state or st.session_state.data_handler is None:
    return

  handler = st.session_state.data_handler
  adata = handler.get_data()
  
  if adata is None:
    return

  # Auto-detect batch column
  from utils.dataset_splitter import get_batch_column
  batch_col = get_batch_column(adata)
  
  if not batch_col:
    st.info("No batch column detected in the dataset. This analysis requires batch annotations.")
    st.caption("Common batch column names: 'batch', 'Batch', 'tech', 'sample', 'donor'")
    return
  
  # Settings
  col1, col2, col3 = st.columns(3)
  
  with col1:
    n_genes = st.number_input(
      "Number of genes per batch",
      min_value=5,
      max_value=50,
      value=10,
      step=5,
      help="How many top genes to show for each batch",
      key="batch_n_genes"
    )
  
  with col2:
    method = st.selectbox(
      "Ranking method",
      options=['mean', 'median', 'fraction_expressed'],
      format_func=lambda x: {
        'mean': 'Mean Expression',
        'median': 'Median Expression',
        'fraction_expressed': 'Fraction Cells Expressing'
      }[x],
      help="Method used to rank genes",
      key="batch_method"
    )
  
  with col3:
    n_batches = adata.obs[batch_col].nunique()
    st.caption(f"{n_batches} batch(es) detected")
  
  with st.expander("ℹ️ How are top genes determined?"):
    st.markdown("""
    **⚠️ Important: This is NOT differential expression analysis**
    
    This analysis identifies genes with **high expression levels** within each batch, not genes that are **specific** to a batch.
    
    **Methodology:**
    - For each batch independently, we calculate gene expression statistics
    - **Mean Expression**: Average expression across all cells in the batch
    - **Median Expression**: Median expression (more robust to outliers)
    - **Fraction Expressed**: Percentage of cells expressing the gene (expression > 0)
    - Genes are then ranked by the selected metric
    
    **Example:** A housekeeping gene like ACTB (actin) will appear in the top genes of ALL batches because it's highly expressed everywhere.
    
    **For differential expression** (finding batch-specific markers), use the "Group Gene Inspector" which performs statistical tests (Wilcoxon) to compare one group vs. all others.
    """)
  
  # Compute button
  if st.button("Compute Top Genes by Batch", key="analyze_batch_genes_btn", type="primary"):
    with st.spinner(f"Computing top {n_genes} genes for each batch..."):
      from utils.statistics import compute_highly_expressed_genes_by_group
      
      try:
        genes_by_batch = compute_highly_expressed_genes_by_group(
          adata,
          groupby=batch_col,
          n_genes=n_genes,
          method=method
        )
        
        # Store in session state
        st.session_state['batch_genes_results'] = genes_by_batch
        st.success(f"✓ Analysis complete for {len(genes_by_batch)} batches")
        
      except Exception as e:
        st.error(f"Error computing genes by batch: {e}")
        import traceback
        st.code(traceback.format_exc())
  
  # Display results if available
  if 'batch_genes_results' in st.session_state:
    genes_by_batch = st.session_state['batch_genes_results']
    
    st.markdown("---")
    st.markdown("#### Visualization")
    
    viz_type = st.radio(
      "Visualization type",
      options=['Heatmap', 'Bar Charts', 'Tables'],
      horizontal=True,
      key="batch_viz_type"
    )
    
    if viz_type == 'Heatmap':
      st.caption("Heatmap showing expression levels of top genes across batches")
      fig = viz.plot_top_genes_by_group_heatmap(
        genes_by_batch,
        n_genes=n_genes,
        title=f"Top {n_genes} Expressed Genes by Batch"
      )
      if fig:
        st.pyplot(fig)
        
        # Download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        st.download_button(
          label="Download Heatmap",
          data=buf.getvalue(),
          file_name=f"batch_genes_heatmap_{n_genes}.png",
          mime="image/png"
        )
        plt.close(fig)
    
    elif viz_type == 'Bar Charts':
      st.caption("Bar charts showing top genes for each batch")
      fig = viz.plot_top_genes_by_group_bars(
        genes_by_batch,
        n_genes=n_genes,
        title=f"Top {n_genes} Expressed Genes by Batch"
      )
      if fig:
        st.pyplot(fig)
        
        # Download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        st.download_button(
          label="Download Bar Charts",
          data=buf.getvalue(),
          file_name=f"batch_genes_bars_{n_genes}.png",
          mime="image/png"
        )
        plt.close(fig)
    
    else:  # Tables
      st.caption("Detailed tables for each batch")
      
      # Create tabs for each batch
      batch_names = sorted(genes_by_batch.keys())
      if len(batch_names) <= 10:
        tabs = st.tabs(batch_names)
        for tab, batch_name in zip(tabs, batch_names):
          with tab:
            df = genes_by_batch[batch_name]
            st.dataframe(
              df,
              column_config={
                "Gene": "Gene Symbol",
                "Mean Expression": st.column_config.NumberColumn(format="%.3f"),
                "Median Expression": st.column_config.NumberColumn(format="%.3f"),
                "Std Expression": st.column_config.NumberColumn(format="%.3f"),
                "Fraction Expressed": st.column_config.ProgressColumn(
                  "Fraction Expressed",
                  format="%.1f%%",
                  min_value=0,
                  max_value=1
                ),
                "Rank": st.column_config.NumberColumn(format="%d")
              },
              hide_index=True,
              width="stretch"
            )
      else:
        # Too many batches for tabs, use selectbox
        selected_batch = st.selectbox(
          "Select batch to view",
          options=batch_names,
          key="batch_table_selector"
        )
        df = genes_by_batch[selected_batch]
        st.dataframe(
          df,
          column_config={
            "Gene": "Gene Symbol",
            "Mean Expression": st.column_config.NumberColumn(format="%.3f"),
            "Median Expression": st.column_config.NumberColumn(format="%.3f"),
            "Std Expression": st.column_config.NumberColumn(format="%.3f"),
            "Fraction Expressed": st.column_config.ProgressColumn(
              "Fraction Expressed",
              format="%.1f%%",
              min_value=0,
              max_value=1
            ),
            "Rank": st.column_config.NumberColumn(format="%d")
          },
          hide_index=True,
          width="stretch"
        )
    
    # Export all data as CSV
    st.markdown("---")
    if st.button("Export All Data to CSV", key="batch_export_csv"):
      # Combine all DataFrames
      combined_data = []
      for batch_name, df in genes_by_batch.items():
        df_copy = df.copy()
        df_copy['Batch'] = batch_name
        combined_data.append(df_copy)
      
      df_combined = pd.concat(combined_data, ignore_index=True)
      csv = df_combined.to_csv(index=False)
      
      st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"top_genes_by_batch_{n_genes}.csv",
        mime="text/csv"
      )


def _render_celltype_gene_inspector(results):
  """Render inspector to analyze highly expressed genes by cell type."""
  st.markdown("### Cell Type Gene Inspector")
  st.caption("Analyze the most highly expressed genes in each cell type.")

  if 'data_handler' not in st.session_state or st.session_state.data_handler is None:
    return

  handler = st.session_state.data_handler
  adata = handler.get_data()
  
  if adata is None:
    return

  # Auto-detect cell type column
  celltype_col = None
  for col in ['Group', 'labels', 'cell_type', 'celltype', 'CellType']:
    if col in adata.obs.columns:
      celltype_col = col
      break
  
  if not celltype_col:
    st.info("No cell type column detected in the dataset. This analysis requires cell type annotations.")
    st.caption("Common cell type column names: 'Group', 'labels', 'cell_type', 'celltype'")
    return
  
  # Settings
  col1, col2, col3 = st.columns(3)
  
  with col1:
    n_genes = st.number_input(
      "Number of genes per cell type",
      min_value=5,
      max_value=50,
      value=10,
      step=5,
      help="How many top genes to show for each cell type",
      key="celltype_n_genes"
    )
  
  with col2:
    method = st.selectbox(
      "Ranking method",
      options=['mean', 'median', 'fraction_expressed'],
      format_func=lambda x: {
        'mean': 'Mean Expression',
        'median': 'Median Expression',
        'fraction_expressed': 'Fraction Cells Expressing'
      }[x],
      help="Method used to rank genes",
      key="celltype_method"
    )
  
  with col3:
    st.metric("Cell Type Column", celltype_col)
    n_celltypes = adata.obs[celltype_col].nunique()
    st.caption(f"{n_celltypes} cell type(s) detected")
  
  # Batch filtering option
  from utils.dataset_splitter import get_batch_column
  batch_col = get_batch_column(adata)
  
  filter_by_batch = False
  selected_batch = None
  
  if batch_col:
    st.markdown("#### Batch Filter (Optional)")
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
      filter_by_batch = st.checkbox(
        "Analyze cell types within a specific batch",
        value=False,
        help="Filter data to a single batch before computing top genes per cell type",
        key="celltype_filter_batch"
      )
    
    if filter_by_batch:
      with col_b2:
        batches = sorted(adata.obs[batch_col].unique().astype(str))
        selected_batch = st.selectbox(
          "Select batch",
          options=batches,
          key="celltype_batch_selector"
        )
        n_cells_in_batch = (adata.obs[batch_col].astype(str) == selected_batch).sum()
        st.caption(f"{n_cells_in_batch:,} cells in this batch")
  
  with st.expander("ℹ️ How are top genes determined?"):
    st.markdown("""
    **⚠️ Important: This is NOT differential expression analysis**
    
    This analysis identifies genes with **high expression levels** within each cell type, not genes that are **specific** to a cell type.
    
    **Methodology:**
    - For each cell type independently, we calculate gene expression statistics
    - **Mean Expression**: Average expression across all cells of this type
    - **Median Expression**: Median expression (more robust to outliers)
    - **Fraction Expressed**: Percentage of cells expressing the gene (expression > 0)
    - Genes are then ranked by the selected metric
    
    **Example:** A housekeeping gene like GAPDH will appear in the top genes of ALL cell types because it's highly expressed everywhere.
    
    **For differential expression** (finding cell type-specific markers), use the "Group Gene Inspector" which performs statistical tests (Wilcoxon) to compare one group vs. all others.
    """)
  
  # Compute button
  if st.button("Compute Top Genes by Cell Type", key="analyze_celltype_genes_btn", type="primary"):
    with st.spinner(f"Computing top {n_genes} genes for each cell type..."):
      from utils.statistics import compute_highly_expressed_genes_by_group
      
      try:
        # Filter by batch if selected
        if filter_by_batch and selected_batch:
          adata_subset = adata[adata.obs[batch_col].astype(str) == selected_batch, :].copy()
          st.info(f"Analyzing {adata_subset.n_obs} cells from batch '{selected_batch}'")
        else:
          adata_subset = adata
        
        genes_by_celltype = compute_highly_expressed_genes_by_group(
          adata_subset,
          groupby=celltype_col,
          n_genes=n_genes,
          method=method
        )
        
        # Store in session state
        st.session_state['celltype_genes_results'] = genes_by_celltype
        st.session_state['celltype_genes_batch_filter'] = selected_batch if filter_by_batch else None
        st.success(f"✓ Analysis complete for {len(genes_by_celltype)} cell types")
        
      except Exception as e:
        st.error(f"Error computing genes by cell type: {e}")
        import traceback
        st.code(traceback.format_exc())
  
  # Display results if available
  if 'celltype_genes_results' in st.session_state:
    genes_by_celltype = st.session_state['celltype_genes_results']
    
    st.markdown("---")
    st.markdown("#### Visualization")
    
    viz_type = st.radio(
      "Visualization type",
      options=['Heatmap', 'Bar Charts', 'Tables'],
      horizontal=True,
      key="celltype_viz_type"
    )
    
    if viz_type == 'Heatmap':
      st.caption("Heatmap showing expression levels of top genes across cell types")
      fig = viz.plot_top_genes_by_group_heatmap(
        genes_by_celltype,
        n_genes=n_genes,
        title=f"Top {n_genes} Expressed Genes by Cell Type"
      )
      if fig:
        st.pyplot(fig)
        
        # Download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        st.download_button(
          label="Download Heatmap",
          data=buf.getvalue(),
          file_name=f"celltype_genes_heatmap_{n_genes}.png",
          mime="image/png"
        )
        plt.close(fig)
    
    elif viz_type == 'Bar Charts':
      st.caption("Bar charts showing top genes for each cell type")
      fig = viz.plot_top_genes_by_group_bars(
        genes_by_celltype,
        n_genes=n_genes,
        title=f"Top {n_genes} Expressed Genes by Cell Type"
      )
      if fig:
        st.pyplot(fig)
        
        # Download button
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        st.download_button(
          label="Download Bar Charts",
          data=buf.getvalue(),
          file_name=f"celltype_genes_bars_{n_genes}.png",
          mime="image/png"
        )
        plt.close(fig)
    
    else:  # Tables
      st.caption("Detailed tables for each cell type")
      
      # Create tabs for each cell type
      celltype_names = sorted(genes_by_celltype.keys())
      if len(celltype_names) <= 10:
        tabs = st.tabs(celltype_names)
        for tab, celltype_name in zip(tabs, celltype_names):
          with tab:
            df = genes_by_celltype[celltype_name]
            st.dataframe(
              df,
              column_config={
                "Gene": "Gene Symbol",
                "Mean Expression": st.column_config.NumberColumn(format="%.3f"),
                "Median Expression": st.column_config.NumberColumn(format="%.3f"),
                "Std Expression": st.column_config.NumberColumn(format="%.3f"),
                "Fraction Expressed": st.column_config.ProgressColumn(
                  "Fraction Expressed",
                  format="%.1f%%",
                  min_value=0,
                  max_value=1
                ),
                "Rank": st.column_config.NumberColumn(format="%d")
              },
              hide_index=True,
              width="stretch"
            )
      else:
        # Too many cell types for tabs, use selectbox
        selected_celltype = st.selectbox(
          "Select cell type to view",
          options=celltype_names,
          key="celltype_table_selector"
        )
        df = genes_by_celltype[selected_celltype]
        st.dataframe(
          df,
          column_config={
            "Gene": "Gene Symbol",
            "Mean Expression": st.column_config.NumberColumn(format="%.3f"),
            "Median Expression": st.column_config.NumberColumn(format="%.3f"),
            "Std Expression": st.column_config.NumberColumn(format="%.3f"),
            "Fraction Expressed": st.column_config.ProgressColumn(
              "Fraction Expressed",
              format="%.1f%%",
              min_value=0,
              max_value=1
            ),
            "Rank": st.column_config.NumberColumn(format="%d")
          },
          hide_index=True,
          width="stretch"
        )
    
    # Export all data as CSV
    st.markdown("---")
    if st.button("Export All Data to CSV", key="celltype_export_csv"):
      # Combine all DataFrames
      combined_data = []
      for celltype_name, df in genes_by_celltype.items():
        df_copy = df.copy()
        df_copy['Cell_Type'] = celltype_name
        combined_data.append(df_copy)
      
      df_combined = pd.concat(combined_data, ignore_index=True)
      csv = df_combined.to_csv(index=False)
      
      st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"top_genes_by_celltype_{n_genes}.csv",
        mime="text/csv"
      )
def _render_cell_inspector(results):
  """Render cell inspector to check individual cell predictions."""
  st.markdown("### Cell Inspector")
  st.caption("Select a specific cell to see how each algorithm classified it.")

  # Get data handler to access cell names
  if 'data_handler' not in st.session_state or st.session_state.data_handler is None:
    return

  handler = st.session_state.data_handler
  adata = handler.get_data()
  
  if adata is None:
    return

  # maximize performance by using session state for cell list if large
  cell_names = list(adata.obs_names)
  
  col1, col2 = st.columns([1, 2])
  
  with col1:
    selected_cell = st.selectbox(
      "Select Cell ID",
      options=cell_names,
      index=0,
      help="Choose a cell barcode/name to inspect"
    )
  
  if not selected_cell:
    return

  # Find index of selected cell
  try:
    cell_idx = cell_names.index(selected_cell)
  except ValueError:
    st.error(f"Cell {selected_cell} not found.")
    return

  # Ground truth
  true_labels = handler.get_labels()
  true_label = "Unknown"
  if true_labels is not None:
    true_label = true_labels[cell_idx]
    if hasattr(adata, 'uns') and 'label_map' in adata.uns:
       # Try to map back to string label if possible
       inv_map = {v: k for k, v in adata.uns['label_map'].items()}
       if true_label in inv_map:
         true_label = f"{inv_map[true_label]} ({true_label})"

  with col2:
    st.info(f"**Ground Truth**: {true_label}")

  # Build predictions table
  # Get ground truth for alignment
  handler = st.session_state.data_handler
  true_labels = handler.get_labels()
  
  inspector_data = []
  
  for result in results.results:
    # Get raw prediction
    pred = result.labels[cell_idx]
    
    # Calculate aligned prediction if GT exists
    aligned_pred_val = ""
    if true_labels is not None:
      # We align the full set of labels to find the mapping context
      aligned_labels = align_labels(true_labels, result.labels)
      aligned_pred_val = aligned_labels[cell_idx]
    
    row = {
      'Algorithm': result.algorithm_name,
      'Run': result.run_id + 1,
      'Raw Cluster': int(pred) if isinstance(pred, (int, np.integer)) else pred,
    }
    
    if true_labels is not None:
       row['Aligned Cluster (Match to GT)'] = aligned_pred_val
       
    inspector_data.append(row)
  
  if inspector_data:
    df_inspector = pd.DataFrame(inspector_data)
    if "Aligned Cluster (Match to GT)" in df_inspector.columns:
      df_inspector["Aligned Cluster (Match to GT)"] = df_inspector["Aligned Cluster (Match to GT)"].astype(str)
      
    st.dataframe(
      df_inspector, 
      column_config={
        "Raw Cluster": st.column_config.NumberColumn(format="%d"),
        "Aligned Cluster (Match to GT)": st.column_config.TextColumn(help="Predicted cluster mapped to the closest ground truth label")
      },
      width="stretch",
      hide_index=True
    )


def _render_umap(results):
  """Render 2D projection visualization (native latent 2D when available, else UMAP)."""
  try:
    # Collect all unique algorithms with embeddings (one per algorithm, first run)
    algorithms_with_embeddings = {}
    for result in results.results:
      if result.embeddings is not None and result.algorithm_name not in algorithms_with_embeddings:
        algorithms_with_embeddings[result.algorithm_name] = result

    if not algorithms_with_embeddings:
      return

    st.markdown("### Projection Visualization")
    st.caption("Uses native latent 2D embeddings when available; otherwise computes a UMAP projection.")

    # Get ground truth labels
    handler = st.session_state.data_handler
    true_labels = handler.get_labels()
    adata = handler.get_data()

    # Let user select which algorithm's embeddings to use for visualization
    algo_names = list(algorithms_with_embeddings.keys())
    algo_display_names = [
      AlgorithmRegistry.get(name).get_info().display_name
      for name in algo_names
    ]

    selected_algo_display = st.selectbox(
      "Select embeddings for 2D projection",
      options=algo_display_names,
      index=0,
      help="Choose which algorithm's embeddings to visualize in 2D"
    )
    selected_algo_name = algo_names[algo_display_names.index(selected_algo_display)]
    base_result = algorithms_with_embeddings[selected_algo_name]

    base_embeddings = np.asarray(base_result.embeddings)
    if base_embeddings.ndim != 2 or base_embeddings.shape[0] == 0 or base_embeddings.shape[1] < 2:
      st.warning("Embeddings are not valid for 2D visualization.")
      return

    is_native_2d = base_embeddings.shape[1] == 2
    if is_native_2d:
      projection_name = "Latent 2D"
      embedding_2d = base_embeddings
    else:
      projection_name = "UMAP"
      try:
        import umap
      except ImportError:
        st.info("Install umap-learn for UMAP visualization (required for embeddings with >2 dimensions).")
        return
      with st.spinner("Computing UMAP..."):
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(base_embeddings)

    import matplotlib.pyplot as plt

    # Coloring options
    st.markdown(f"#### {projection_name} Display Options")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
      color_by = st.selectbox("Color by", options=["Cluster", "Ground Truth", "Batch"], index=0)

    n_points = embedding_2d.shape[0]

    def _match_label_length(label_values):
      if label_values is None:
        return None
      labels_array = np.asarray(label_values)
      if labels_array.shape[0] != n_points:
        return None
      return labels_array
    
    # Determine labels to color by
    if color_by == "Cluster":
      plot_labels = np.asarray(base_result.labels)
    elif color_by == "Ground Truth":
      plot_labels = _match_label_length(base_result.true_labels)
      if plot_labels is None:
        plot_labels = _match_label_length(true_labels)
      if plot_labels is None:
        st.warning("No ground truth available. Defaulting to Cluster.")
        plot_labels = np.asarray(base_result.labels)
    else: # Batch
      from utils.dataset_splitter import get_batch_column
      batch_col = get_batch_column(adata) if adata is not None else None
      if batch_col:
        plot_labels = _match_label_length(adata.obs[batch_col].values)
      else:
        plot_labels = None
      if plot_labels is None:
        st.warning("No compatible batch labels detected. Defaulting to Cluster.")
        plot_labels = np.asarray(base_result.labels)

    fig = viz.plot_umap_embeddings(
      embedding_2d,
      plot_labels,
      title=f"{projection_name}: {selected_algo_display} (Colored by {color_by})",
      projection_name=projection_name,
    )
    st.pyplot(fig)

    # Export figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    out_name = "latent2d_projection.png" if is_native_2d else "umap_projection.png"
    st.download_button(
      label=f"Download {projection_name} Plot",
      data=buf.getvalue(),
      file_name=out_name,
      mime="image/png"
    )
    plt.close()

  except Exception as e:
    st.warning(f"Could not create projection visualization: {e}")


def _export_csv(results):
  """Export results to CSV."""
  import io

  detail_data = []
  for result in results.results:
    row = {
      'Algorithm': result.algorithm_name,
      'Run': result.run_id + 1,
      **result.metrics,
      'Runtime (s)': result.runtime
    }
    detail_data.append(row)

  df = pd.DataFrame(detail_data)

  csv = df.to_csv(index=False)

  st.download_button(
    label="Download CSV",
    data=csv,
    file_name="analysis_results.csv",
    mime="text/csv"
  )


# =============================================================================
# Benchmark Mode Functions
# =============================================================================

def _snapshot_preprocessing_params(params: dict) -> dict:
  """Return a stable subset of preprocessing params for staleness checks."""
  if not isinstance(params, dict):
    return {}
  tracked_keys = [
    'skip',
    'do_cell_filtering',
    'min_genes_per_cell',
    'max_genes_per_cell',
    'do_gene_filtering',
    'min_cells_per_gene',
    'do_normalization',
    'target_sum',
    'do_log_transform',
    'do_hvg',
    'n_top_genes',
    'hvg_flavor',
    'hvg_strategy',
    'ensure_celltype_markers',
    'min_markers_per_celltype',
    'max_additional_hvg_genes',
    'do_batch_correction',
    'batch_correction_method',
    'batch_correction_batch_key',
    'batch_correction_labels_key',
    'batch_correction_epochs',
    'batch_correction_n_latent',
    'batch_correction_batch_size',
    'batch_correction_dct_weight',
    'batch_correction_corr_mse_weight',
    'batch_correction_cycle_weight',
    'batch_correction_kl_weight',
    'batch_correction_prior',
    'batch_correction_n_prior_components',
    'batch_correction_embed_covariates',
    'do_scaling',
    'scale_max_value',
    'dropout_method',
    'dropout_rate',
    'dropout_mid',
    'dropout_shape',
    'noise_level',
    'batch_col',
  ]
  snap = {}
  for key in tracked_keys:
    if key in params:
      value = params.get(key)
      if isinstance(value, (list, tuple)):
        snap[key] = list(value)
      else:
        snap[key] = value
  return snap

def _generate_benchmark_cli_command() -> str:
  """Generate benchmark CLI command using the shared canonical generator."""
  settings = st.session_state.get('benchmark_settings', {}) or {}
  n_repeats = int(settings.get('n_repeats', 1))
  random_seed = int(settings.get('random_seed', 42))
  return _generate_cli_command(
    compact=False,
    n_repetitions=n_repeats,
    seed=random_seed,
  )

def _render_benchmark_settings(handler, info):
  """Render settings for benchmark mode (Summary only + Execution controls)."""
  st.markdown("---")
  st.subheader("Benchmark Execution Settings")

  setup = st.session_state.get('benchmark_setup', {})
  split_info = setup.get('split_info', {})
  original_settings = setup.get('original_settings', {})
  mode = setup.get('mode', 'standard')
  
  # 1. Concise Summary of active split
  with st.expander("Active Benchmark Split Summary", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
      st.write(f"**Strategy**: {original_settings.get('mode', mode).replace('_', ' ').title()}")
      if original_settings.get('batch_col'):
        st.write(f"**Batch Col**: `{original_settings.get('batch_col')}`")
    with col2:
      st.write(f"**Train Size**: {split_info.get('n_train', 0):,} cells")
      st.write(f"**Test Size**: {split_info.get('n_test', 0):,} cells")
    with col3:
      st.write(f"**Val Size**: {split_info.get('n_val', 0):,} cells")
      if 'label_col' in original_settings:
        st.write(f"**Label Col**: `{original_settings.get('label_col')}`")
    
    st.info("To change the split strategy or ratios, go back to the **Data Split** page.")

  # 2. Execution Controls (Only things that can be changed without re-splitting)
  st.markdown("#### Execution Parameters")
  col_e1, col_e2 = st.columns(2)
  
  with col_e1:
    n_repeats = st.number_input(
      "Number of repetitions",
      min_value=1,
      max_value=100,
      value=1,
      help="Number of times to run each algorithm (for statistical analysis)"
    )

  with col_e2:
    random_seed = st.number_input(
      "Random seed",
      min_value=0,
      max_value=99999,
      value=st.session_state.get('benchmark_settings', {}).get('random_seed', 42),
      help="Seed for reproducibility of algorithm initialization"
    )

  compute_scib_metrics = st.checkbox(
    "Compute scIB/scIB-E metrics",
    value=st.session_state.get('benchmark_settings', {}).get('compute_scib_metrics', False),
    help="Disable to speed up benchmark runs when scIB/scIB-E metrics are not needed."
  )

  # 3. Running Options
  st.markdown("#### Running Options")
  
  # Auto-detect batch column for group metrics
  adata = handler.get_data()
  batch_col = original_settings.get('batch_col') or get_batch_column(adata)
  
  use_batch_metrics = False
  if batch_col:
    use_batch_metrics = st.checkbox(
      "Compute metrics per batch",
      value=True, # DEFAULT ENABLED as requested
      help=f"Calculate NMI/ARI/ACC for each batch in '{batch_col}' separately."
    )
  else:
    st.info("No batch column detected. Group metrics unavailable.")

  # Update session state with only execution-related settings
  # We preserve the structural split settings from setup
  st.session_state['benchmark_settings'] = {
    **original_settings, # Keep original split logic
    'n_repeats': n_repeats,
    'random_seed': random_seed,
    'compute_scib_metrics': compute_scib_metrics,
    'batch_col': batch_col if use_batch_metrics else None
  }

  # Balance batches option
  st.markdown("---")
  st.markdown("#### Dataset Balancing")
  
  balance_batches = st.checkbox(
    "Balance batches (equalize cell counts)",
    value=False,
    help="Subsample larger batches to match the smallest batch size. "
       "Useful when batches have very different sizes."
  )
  
  if balance_batches:
    col1, col2 = st.columns(2)
    with col1:
      # Show current batch distribution
      if batch_col:
        batch_counts = adata.obs[batch_col].value_counts()
        min_batch = batch_counts.min()
        st.caption(f"Smallest batch: {min_batch:,} cells")
        st.caption(f"Largest batch: {batch_counts.max():,} cells")
    
    with col2:
      balance_target = st.number_input(
        "Target cells per batch",
        min_value=100,
        max_value=10000,
        value=int(batch_counts.min()) if batch_col else 500,
        step=100,
        help="Number of cells to keep per batch. Default is smallest batch size."
      )
    
    st.session_state['benchmark_settings']['balance_batches'] = True
    st.session_state['benchmark_settings']['balance_target'] = balance_target
  else:
    st.session_state['benchmark_settings']['balance_batches'] = False

  # Summary of selected algorithms

  # Hyperparameter search is separate
  st.markdown("---")
  st.info(
    "Hyperparameter search is independent from benchmark runs. "
    "Use the **Hyperparam Search** page to optimize algorithms."
  )


  # Summary of selected algorithms
  st.markdown("---")
  st.subheader("Selected Algorithms")

  algo_names = st.session_state.get('selected_algorithms', [])
  for algo_name in algo_names:
    algo_info = AlgorithmRegistry.get(algo_name).get_info()
    st.markdown(f"- **{algo_info.display_name}**")

  # CLI Command Generation
  st.markdown("---")
  st.subheader("CLI Command")
  st.caption("Run this analysis from the terminal:")
  
  cli_cmd = _generate_benchmark_cli_command()
  st.code(cli_cmd, language="bash")
  
  # Download as shell script
  st.download_button(
    label="Download Shell Script",
    data=f"#!/bin/bash\n# SCRBenchmark Benchmark CLI Command\n# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{cli_cmd}\n",
    file_name=f"benchmark_run_{time.strftime('%Y%m%d_%H%M%S')}.sh",
    mime="text/x-shellscript",
    width="stretch"
  )

  # Run button
  st.markdown("---")
  if st.button("Run Benchmark Analysis", type="primary", width="stretch"):
    _run_benchmark_analysis(handler)

  # Display benchmark results
  if 'benchmark_results' in st.session_state and st.session_state.benchmark_results is not None:
    _display_benchmark_results()


def _run_benchmark_analysis(handler):
  """Run the benchmark analysis with pre-split/pre-processed data."""
  try:
    setup = st.session_state.get('benchmark_setup', {})
    settings = st.session_state.get('benchmark_settings', {})
    if not settings and setup.get('original_settings'):
      settings = setup.get('original_settings', {})
    if not settings:
      st.error("No benchmark settings found. Please configure the split in 'Data Split'.")
      return

    # NEW LOGIC: Use pre-processed data from session state
    if 'benchmark_processed' not in st.session_state:
      st.error("Benchmark data not found! Please run 'Preprocessing' first.")
      return
      
    processed_data = st.session_state.benchmark_processed
    current_preproc = _snapshot_preprocessing_params(
      st.session_state.get('preprocessing_params', {})
    )
    applied_preproc = processed_data.get('preprocessing_params')
    if applied_preproc is None:
      applied_preproc = st.session_state.get('preprocessing_applied_params')
    if isinstance(applied_preproc, dict) and applied_preproc != current_preproc:
      st.error(
        "Preprocessing parameters changed since last preprocessing run. "
        "Please re-run 'Run Preprocessing' before launching benchmark analysis."
      )
      st.session_state.data_preprocessed = False
      return

    adata_train = processed_data['train']
    adata_test = processed_data['test']
    adata_val = processed_data['val']
    use_validation = settings.get('use_validation', True)
    if not use_validation:
      adata_val = None
    
    # Determine label column from original handler
    # (Though processed data usually has 'labels' or 'Group' preserved)
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(message: str, progress: float = None):
      status_text.text(message)
      if progress is not None:
        progress_bar.progress(progress)

    progress_callback("Preparing benchmark...", 0.1)

    # Get labels from processed data
    # Note: Preprocessing preserves labels in .obs
    def extract_labels(adata):
      if adata is None: return None
      if 'labels_encoded' in adata.obs:
        return adata.obs['labels_encoded'].values
      elif 'labels' in adata.obs:
        return adata.obs['labels'].values
      elif 'Group' in adata.obs:
        # Need to encode if string
        return adata.obs['Group'].values 
      return None
      
    # We should use the Helper in DataHandler or DatasetSplitter if available, 
    # but here we can try to rely on what's in .obs
    labels_train = extract_labels(adata_train)
    labels_test = extract_labels(adata_test)
    labels_val = extract_labels(adata_val) if adata_val is not None else None

    # Run benchmark
    runner = AnalysisRunner()

    params = st.session_state.get('algorithm_params', {})
    compute_scib_metrics = settings.get('compute_scib_metrics', True)
    for algo_name in st.session_state.selected_algorithms:
      if algo_name not in params:
        params[algo_name] = {}
      # Ensure random seed consistency
      params[algo_name]['random_state'] = settings.get('random_seed', 42)

    progress_callback("Running benchmark...", 0.3)
    
    # Split algorithms: graph-based get classical mode (no split)
    # Separate algorithms by their capability to handle out-of-sample prediction
    # - benchmark_algos: algorithms that support out-of-sample (inductive) -> use train/test split
    # - transductive_algos_by_input: algorithms that DON'T support out-of-sample -> run on full data
    transductive_algos_by_input = {'processed': [], 'raw_filtered': [], 'raw_original': []}
    benchmark_algos = []
    for algo_name in st.session_state.selected_algorithms:
      algo_class = AlgorithmRegistry.get(algo_name)
      if algo_class:
        algo_info = algo_class.get_info()
        # Use supports_out_of_sample as the primary criterion
        # If an algorithm supports out-of-sample, it can use the train/test split
        supports_oos = getattr(algo_info, 'supports_out_of_sample', True)
        if not supports_oos:
          # Transductive algorithm -> run on full dataset
          input_type = params.get(algo_name, {}).get('input_type', 'processed')
          if input_type not in transductive_algos_by_input:
            input_type = 'processed'
          transductive_algos_by_input[input_type].append(algo_name)
        else:
          # Inductive algorithm -> use train/test split
          benchmark_algos.append(algo_name)
      else:
        benchmark_algos.append(algo_name)

    all_results = []

    if benchmark_algos:
      results_benchmark = runner.run_benchmark_comparison(
        algorithm_names=benchmark_algos,
        data_train=adata_train,
        data_test=adata_test,
        labels_train=labels_train,
        labels_test=labels_test,
        data_val=adata_val,
        labels_val=labels_val,
        params=params,
        batch_col=settings.get('batch_col'),
        n_repeats=settings.get('n_repeats', 1),
        compute_scib_metrics=compute_scib_metrics,
        progress_callback=progress_callback
      )
      all_results.extend(results_benchmark.results)

    if any(transductive_algos_by_input.values()):
      import anndata as ad

      def concat_processed_parts():
        parts = [adata_train, adata_val, adata_test]
        parts = [p for p in parts if p is not None]
        if len(parts) == 1:
          return parts[0].copy()
        return ad.concat(parts, axis=0, join="inner", merge="same")

      # Processed data group
      if transductive_algos_by_input['processed']:
        adata_full = concat_processed_parts()
        labels_full = extract_labels(adata_full)
        results_graph = runner.run_benchmark_comparison(
          algorithm_names=transductive_algos_by_input['processed'],
          data_train=adata_full,
          data_test=adata_full,
          labels_train=labels_full,
          labels_test=labels_full,
          data_val=None,
          labels_val=None,
          params=params,
          batch_col=settings.get('batch_col'),
          n_repeats=settings.get('n_repeats', 1),
          compute_scib_metrics=compute_scib_metrics,
          progress_callback=progress_callback
        )
        all_results.extend(results_graph.results)

      # Raw filtered data group
      if transductive_algos_by_input['raw_filtered']:
        adata_raw_filtered = handler.get_raw_filtered_data()
        if adata_raw_filtered is None:
          adata_raw_filtered = concat_processed_parts()
        labels_raw_filtered = handler.get_labels(data_source='processed')
        if labels_raw_filtered is None or len(labels_raw_filtered) != adata_raw_filtered.n_obs:
          labels_raw_filtered = extract_labels(adata_raw_filtered)
        results_graph = runner.run_benchmark_comparison(
          algorithm_names=transductive_algos_by_input['raw_filtered'],
          data_train=adata_raw_filtered,
          data_test=adata_raw_filtered,
          labels_train=labels_raw_filtered,
          labels_test=labels_raw_filtered,
          data_val=None,
          labels_val=None,
          params=params,
          batch_col=settings.get('batch_col'),
          n_repeats=settings.get('n_repeats', 1),
          compute_scib_metrics=compute_scib_metrics,
          progress_callback=progress_callback
        )
        all_results.extend(results_graph.results)

      # Raw original data group
      if transductive_algos_by_input['raw_original']:
        adata_raw_original = handler.get_original_data()
        if adata_raw_original is None:
          adata_raw_original = concat_processed_parts()
        labels_raw_original = handler.get_labels(data_source='original')
        if labels_raw_original is None or len(labels_raw_original) != adata_raw_original.n_obs:
          labels_raw_original = extract_labels(adata_raw_original)
        results_graph = runner.run_benchmark_comparison(
          algorithm_names=transductive_algos_by_input['raw_original'],
          data_train=adata_raw_original,
          data_test=adata_raw_original,
          labels_train=labels_raw_original,
          labels_test=labels_raw_original,
          data_val=None,
          labels_val=None,
          params=params,
          batch_col=settings.get('batch_col'),
          n_repeats=settings.get('n_repeats', 1),
          compute_scib_metrics=compute_scib_metrics,
          progress_callback=progress_callback
        )
        all_results.extend(results_graph.results)

    results = BenchmarkComparisonResult(
      results=all_results,
      summary=runner._compute_benchmark_summary(all_results)
    )

    st.session_state.benchmark_results = results
    # Use existing split info
    setup = st.session_state.get('benchmark_setup', {})
    st.session_state.benchmark_split_info = setup.get('split_info', {})

    progress_bar.progress(1.0)
    status_text.text("Benchmark analysis completed!")
    time.sleep(1)

    st.rerun()

  except Exception as e:
    display_error(
      e,
      user_message="Benchmark analysis failed.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="analysis_benchmark_run",
    )


def _display_benchmark_results():
  """Display benchmark analysis results."""
  results = st.session_state.benchmark_results
  split_info = st.session_state.get('benchmark_split_info', {})

  st.markdown("---")
  st.subheader("Benchmark Results")

  # Split information
  st.markdown("### Data Split Summary")
  col1, col2, col3 = st.columns(3)
  with col1:
    st.metric("Train Samples", f"{split_info.get('n_train', 0):,}")
  with col2:
    st.metric("Validation Samples", f"{split_info.get('n_val', 0):,}")
  with col3:
    st.metric("Test Samples", f"{split_info.get('n_test', 0):,}")

  # Summary table
  st.markdown("### Performance Summary")

  summary_data = []
  has_repeats = any(r.run_id > 0 for r in results.results)
  for algo_name, stats in results.summary.items():
    algo_info = AlgorithmRegistry.get(algo_name).get_info()
    row = {'Algorithm': algo_info.display_name}

    # Train metrics
    for metric in ['NMI', 'ARI', 'ACC']:
      train_key = f'train_{metric}_mean'
      if train_key in stats:
        if has_repeats and f'train_{metric}_std' in stats:
          row[f'Train {metric}'] = f"{stats[train_key]:.4f} ± {stats[f'train_{metric}_std']:.4f}"
        else:
          row[f'Train {metric}'] = f"{stats[train_key]:.4f}"

    # Test metrics
    for metric in ['NMI', 'ARI', 'ACC']:
      test_key = f'test_{metric}_mean'
      if test_key in stats:
        if has_repeats and f'test_{metric}_std' in stats:
          row[f'Test {metric}'] = f"{stats[test_key]:.4f} ± {stats[f'test_{metric}_std']:.4f}"
        else:
          row[f'Test {metric}'] = f"{stats[test_key]:.4f}"

    # Generalization gap
    for metric in ['NMI', 'ARI', 'ACC']:
      gap_key = f'{metric}_gap_mean'
      if gap_key in stats:
        gap = stats[gap_key]
        if has_repeats and f'{metric}_gap_std' in stats:
          row[f'{metric} Gap'] = f"{gap:+.4f} ± {stats[f'{metric}_gap_std']:.4f}"
        else:
          row[f'{metric} Gap'] = f"{gap:+.4f}"

    summary_data.append(row)

  df_summary = pd.DataFrame(summary_data)
  st.dataframe(df_summary, width="stretch")

  # Best algorithms
  st.markdown("### Best Algorithms")

  col1, col2 = st.columns(2)

  with col1:
    best_test = results.get_best_algorithm('NMI', 'test')
    if best_test:
      best_info = AlgorithmRegistry.get(best_test).get_info()
      test_nmi = results.summary[best_test].get('test_NMI_mean', 0)
      st.success(f"**Best Test NMI:** {best_info.display_name} ({test_nmi:.4f})")

  with col2:
    # Best generalization (smallest gap)
    ranking = results.get_generalization_ranking('NMI')
    if ranking:
      best_gen = ranking[0]
      best_info = AlgorithmRegistry.get(best_gen[0]).get_info()
      st.success(f"**Best Generalization:** {best_info.display_name} (gap: {best_gen[1]:+.4f})")

  # Visualization
  _render_benchmark_visualizations(results)

  # Note: _render_group_metrics is now integrated as an option in _render_benchmark_visualizations
  # but we can keep it here as a fallback or if not standalone
  # _render_group_metrics(results)

  # Export
  st.markdown("---")
  st.subheader("Export Results")

  if st.button("Export Benchmark Results to CSV"):
    _export_benchmark_csv(results)


def _render_benchmark_visualizations(results: BenchmarkComparisonResult):
  """Render visualizations for benchmark results with gallery selector."""
  import matplotlib.pyplot as plt

  st.markdown("### Benchmark Visualizations")
  
  # Define available plots
  plot_options = {
    "train_test": " Train vs Test Performance",
    "gap": " Generalization Gap",
    "radar": " Algorithm Radar (NMI/ARI/ACC)",
    "boxplot": " Statistical Comparison (Boxplots)",
    "batch": " Metrics by Batch Heatmap",
  }
  
  selected_plot_key = st.selectbox(
    "Choose Comparison Plot",
    options=list(plot_options.keys()),
    format_func=lambda x: plot_options[x],
    index=0
  )
  
  fig = None
  filename = "benchmark_plot.png"

  # -------------------------------------------------------------------------
  # 1. Train vs Test
  # -------------------------------------------------------------------------
  if selected_plot_key == "train_test":
    st.caption("Compare performance on Training set vs Test set. Good generalization means Test scores are close to Train scores.")
    fig = viz.plot_benchmark_comparison(results.summary)
    filename = "train_vs_test_comparison.png"

  # -------------------------------------------------------------------------
  # 2. Generalization Gap
  # -------------------------------------------------------------------------
  elif selected_plot_key == "gap":
    st.caption("Difference between Train and Test scores (Gap = Train - Test). Lower (or negative) bars are better.")
    fig = viz.plot_generalization_gap(results.summary)
    filename = "generalization_gap.png"
  
  # -------------------------------------------------------------------------
  # 3. Radar Chart
  # -------------------------------------------------------------------------
  elif selected_plot_key == "radar":
    st.caption("Overview of mean performance across different metrics. Larger area is better.")
    fig = viz.plot_radar_chart(results.summary)
    filename = "radar_comparison.png"

  # -------------------------------------------------------------------------
  # 4. Boxplots (Statistical Comparison)
  # -------------------------------------------------------------------------
  elif selected_plot_key == "boxplot":
    col_b1, col_b2 = st.columns(2)
    with col_b1:
      split = st.radio("Select Split for Boxplot", options=["test", "train", "val"], index=0, horizontal=True)
    with col_b2:
      stat_method = st.selectbox("Statistical Method", options=["nonparametric", "parametric"], index=0)
      
    st.caption(f"Statistical distribution of {split.capitalize()} scores across all repetitions.")
    fig = viz.plot_metrics_comparison(
      results.summary, 
      results.results, 
      benchmark_split=split,
      stat_method=stat_method
    )
    filename = f"benchmark_boxplot_{split}.png"

  # -------------------------------------------------------------------------
  # 5. Batch Heatmap (Delegated)
  # -------------------------------------------------------------------------
  elif selected_plot_key == "batch":
    fig = _render_group_metrics(results, standalone=True)
    # Note: fig can be None if no group metrics
  
  # Display & Download
  if fig:
    st.pyplot(fig)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=300)
    st.download_button(
      label=f"Download {plot_options[selected_plot_key]}",
      data=buf.getvalue(),
      file_name=filename,
      mime="image/png",
      width="stretch"
    )
    plt.close()

  st.markdown("---")
  _render_results_explorer_tools(results, key_prefix="run_analysis_benchmark")


def _render_group_metrics(results: BenchmarkComparisonResult, standalone: bool = False):
  """Render metrics broken down by batch/group."""
  if not standalone:
    st.markdown("### Metrics by Batch/Dataset Source")

  # Check if we have group metrics
  has_group_metrics = False
  for result in results.results:
    if result.benchmark_metrics.test_by_group:
      has_group_metrics = True
      break

  if not has_group_metrics:
    if not standalone:
      st.info("No batch information available for group metrics.")
    return None

  # Select algorithm to view
  algo_names = list(results.summary.keys())
  algo_display_map = {
    name: AlgorithmRegistry.get(name).get_info().display_name
    for name in algo_names
  }

  selected_algo = st.selectbox(
    "Select Algorithm",
    options=algo_names,
    format_func=lambda x: algo_display_map.get(x, x),
    key="group_metrics_algo"
  )

  # Get first result for this algorithm
  selected_result = next(
    (r for r in results.results if r.algorithm_name == selected_algo),
    None
  )

  if selected_result and selected_result.benchmark_metrics.test_by_group:
    group_metrics = selected_result.benchmark_metrics.test_by_group

    # Create dataframe
    group_data = []
    for group_name, metrics in group_metrics.items():
      row = {'Batch': group_name}
      row.update({k: v for k, v in metrics.items() if k not in ['n_samples', 'n_clusters_true', 'n_clusters_pred']})
      row['N Samples'] = metrics.get('n_samples', 0)
      group_data.append(row)

    df_groups = pd.DataFrame(group_data)
    st.dataframe(df_groups, width="stretch")

    # Heatmap visualization
    fig = viz.plot_batch_metrics_heatmap(df_groups, title=f'Metrics by Batch - {algo_display_map[selected_algo]}')
    if fig and not standalone:
      st.pyplot(fig)
      plt.close()
    
    # ... (error analysis code remains but we return fig if available)
    # return fig at the end of the block

    # Error analysis
    if selected_result.benchmark_metrics.error_analysis:
      st.markdown("#### Error Analysis")
      error_info = selected_result.benchmark_metrics.error_analysis

      col1, col2 = st.columns(2)

      with col1:
        st.metric(
          "Overall Error Rate",
          f"{error_info['overall_error_rate']:.2%}"
        )

      with col2:
        st.metric(
          "Total Errors",
          f"{error_info['total_errors']:,} / {error_info['total_samples']:,}"
        )

      # Error by batch
      if error_info.get('error_by_group'):
        st.markdown("**Error Rate by Batch:**")
        error_df = pd.DataFrame([
          {'Batch': k, 'Error Rate': v['error_rate'], 'Errors': v['n_errors'], 'Samples': v['n_samples']}
          for k, v in error_info['error_by_group'].items()
        ]).sort_values('Error Rate', ascending=False)
        st.dataframe(error_df, width="stretch")

      # Error by cell type within each batch
      if error_info.get('error_by_celltype_by_group'):
        st.markdown("**Error Rate by Cell Type and Batch:**")
        fig_ct_batch = viz.plot_celltype_errors_by_batch(
          error_info['error_by_celltype_by_group'],
          title=f"Error by Cell Type and Batch - {algo_display_map[selected_algo]}"
        )
        if fig_ct_batch and not standalone:
          st.pyplot(fig_ct_batch)
          plt.close(fig_ct_batch)

      # Top confusion pairs
      if error_info.get('top_confusion_pairs'):
        st.markdown("**Most Common Confusions:**")
        conf_df = pd.DataFrame(error_info['top_confusion_pairs'][:5])
        st.dataframe(conf_df, width="stretch")
    
    return fig
  
  return None


def _export_benchmark_csv(results: BenchmarkComparisonResult):
  """Export benchmark results to CSV."""

  # Summary data
  summary_data = []
  for algo_name, stats in results.summary.items():
    algo_info = AlgorithmRegistry.get(algo_name).get_info()
    row = {'Algorithm': algo_info.display_name}
    row.update(stats)
    summary_data.append(row)

  df = pd.DataFrame(summary_data)
  csv = df.to_csv(index=False)

  st.download_button(
    label="Download Summary CSV",
    data=csv,
    file_name="benchmark_summary.csv",
    mime="text/csv"
  )

  # Detailed results
  detail_data = []
  for result in results.results:
    row = result.to_dict()
    detail_data.append(row)

  df_detail = pd.DataFrame(detail_data)

  # Flatten nested dicts
  for col in ['train_metrics', 'test_metrics', 'generalization_gap']:
    if col in df_detail.columns:
      expanded = df_detail[col].apply(pd.Series)
      expanded.columns = [f'{col}_{c}' for c in expanded.columns]
      df_detail = pd.concat([df_detail.drop(col, axis=1), expanded], axis=1)

  csv_detail = df_detail.to_csv(index=False)

  st.download_button(
    label="Download Detailed CSV",
    data=csv_detail,
    file_name="benchmark_detailed.csv",
    mime="text/csv"
  )
