"""
Results Explorer Page.

Allows users to load benchmark results and generate interactive visualizations
for comparing algorithms across multiple conditions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import io
import ast
import re
import glob
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from gui.widgets import render_export_panel
from .constants import (
  ALGO_COLORS,
  ALGO_DISPLAY_NAMES,
  FIGURE_CAPTIONS,
  FIGURE_CATEGORY_MAP,
  FIGURE_TYPES,
)
from .registry import FigureRegistry, debug_registry_state

# Setup path to import shared visualization utilities (same strategy as CLI)
# legacy.py now lives in gui/results_explorer/, so go three levels up to src/scrbenchmark.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

try:
  from utils import visualization as viz_utils
except Exception:
  viz_utils = None

# Optional import for multiple testing correction
try:
  from statsmodels.stats.multitest import multipletests
  HAS_STATSMODELS = True
except ImportError:
  HAS_STATSMODELS = False

# Optional import for Compact Letter Display (CLD)
try:
  from utils.statistics import compute_significance_groups
  HAS_CLD = True
except ImportError:
  HAS_CLD = False


# =============================================================================
# CLD (Compact Letter Display) Helpers
# =============================================================================

def _collect_metric_values(all_data: Dict[str, Dict], selected_algos: List[str],
                           selected_conditions: List[str], metric: str,
                           use_gap: bool = False) -> Dict[str, List[float]]:
  """Collect raw per-run metric values for each algorithm, pooled across conditions.

  Args:
    all_data: All loaded results
    selected_algos: Algorithms to include
    selected_conditions: Conditions to pool
    metric: Metric name (NMI, ARI, ACC, etc.)
    use_gap: If True, compute train-test gap instead of raw metric

  Returns:
    Dict mapping algorithm name to list of float values
  """
  values_by_algo: Dict[str, List[float]] = {a: [] for a in selected_algos}

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue
    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    if use_gap:
      train_col = f'train_{metric}'
      test_col = f'test_{metric}'
      if train_col not in df.columns or test_col not in df.columns:
        continue
      for _, row in df.iterrows():
        algo = row.get(algo_col)
        if algo not in selected_algos:
          continue
        train_val = pd.to_numeric(row.get(train_col), errors='coerce')
        test_val = pd.to_numeric(row.get(test_col), errors='coerce')
        if not np.isnan(train_val) and not np.isnan(test_val):
          values_by_algo.setdefault(algo, []).append(train_val - test_val)
    else:
      metric_col = f'test_{metric}' if f'test_{metric}' in df.columns else metric
      if metric_col not in df.columns:
        continue
      for algo in selected_algos:
        vals = pd.to_numeric(df[df[algo_col] == algo][metric_col], errors='coerce').dropna()
        values_by_algo.setdefault(algo, []).extend(vals.tolist())

  # Remove algorithms with no values
  return {k: v for k, v in values_by_algo.items() if len(v) >= 1}


def _collect_metric_values_by_condition(
    all_data: Dict[str, Dict],
    selected_algos: List[str],
    selected_conditions: List[str],
    metric: str,
    use_gap: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
  """Collect raw per-run metric values for each algorithm, split by condition."""
  values_by_condition: Dict[str, Dict[str, List[float]]] = {
    condition: {algo: [] for algo in selected_algos}
    for condition in selected_conditions
  }

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data.get('df')
    if df is None or df.empty:
      continue

    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'
    condition_values = values_by_condition.setdefault(
      condition, {algo: [] for algo in selected_algos}
    )

    if use_gap:
      train_col = f'train_{metric}'
      test_col = f'test_{metric}'
      if train_col not in df.columns or test_col not in df.columns:
        continue
      for _, row in df.iterrows():
        algo = row.get(algo_col)
        if algo not in selected_algos:
          continue
        train_val = pd.to_numeric(row.get(train_col), errors='coerce')
        test_val = pd.to_numeric(row.get(test_col), errors='coerce')
        if not np.isnan(train_val) and not np.isnan(test_val):
          condition_values.setdefault(algo, []).append(train_val - test_val)
    else:
      metric_col = f'test_{metric}' if f'test_{metric}' in df.columns else metric
      if metric_col not in df.columns:
        continue
      for algo in selected_algos:
        vals = pd.to_numeric(df[df[algo_col] == algo][metric_col], errors='coerce').dropna()
        condition_values.setdefault(algo, []).extend(vals.tolist())

  cleaned: Dict[str, Dict[str, List[float]]] = {}
  for condition, algo_values in values_by_condition.items():
    filtered = {algo: vals for algo, vals in algo_values.items() if len(vals) >= 1}
    if filtered:
      cleaned[condition] = filtered
  return cleaned


def _collect_runtime_values(all_data: Dict[str, Dict], selected_algos: List[str],
                            selected_conditions: List[str]) -> Dict[str, List[float]]:
  """Collect per-run runtime values for each algorithm across selected conditions."""
  values_by_algo: Dict[str, List[float]] = {a: [] for a in selected_algos}

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue
    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    runtime_col = None
    if 'total_time' in df.columns:
      runtime_col = 'total_time'
    elif 'runtime' in df.columns:
      runtime_col = 'runtime'
    if runtime_col is None:
      continue

    for algo in selected_algos:
      vals = pd.to_numeric(df[df[algo_col] == algo][runtime_col], errors='coerce').dropna()
      values_by_algo.setdefault(algo, []).extend(vals.tolist())

  return {k: v for k, v in values_by_algo.items() if len(v) >= 1}


def _annotate_cld(ax, cld_dict: Dict[str, str], algo_order: List[str],
                  x_positions=None, y_offset_ratio: float = 0.03):
  """Annotate Compact Letter Display letters above bars/boxes.

  Groups sharing the same letter are NOT significantly different.

  Args:
    ax: Matplotlib axes
    cld_dict: Dict mapping algo name to CLD letter(s)
    algo_order: List of algorithm names in x-axis order
    x_positions: x positions for annotations (default: 0, 1, 2, ...)
    y_offset_ratio: Fraction of y-range to offset above current ylim
  """
  if not cld_dict:
    return

  ymin, ymax = ax.get_ylim()
  y_pos = ymax + (ymax - ymin) * y_offset_ratio

  if x_positions is None:
    x_positions = list(range(len(algo_order)))

  for i, algo in enumerate(algo_order):
    letter = cld_dict.get(algo, '')
    if letter and i < len(x_positions):
      ax.text(x_positions[i], y_pos, letter, ha='center', va='bottom',
              fontweight='bold', fontsize=11, color='#222',
              bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffcc',
                        edgecolor='#cccc88', alpha=0.85))

  # Expand y-axis to make room for letters
  ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.12)


def _annotate_cld_grouped_bars(
    ax,
    cld_dict: Dict[str, str],
    algo_order: List[str],
    x_positions: np.ndarray,
    bar_tops: np.ndarray,
    y_offset_ratio: float = 0.015,
):
  """Annotate CLD letters above grouped bars (one condition at a time)."""
  if not cld_dict or len(x_positions) == 0 or len(bar_tops) == 0:
    return

  ymin, ymax = ax.get_ylim()
  y_span = max(ymax - ymin, 1e-9)
  y_pad = y_span * y_offset_ratio
  max_text_y = ymax

  for i, algo in enumerate(algo_order):
    if i >= len(x_positions) or i >= len(bar_tops):
      continue
    letter = cld_dict.get(algo, '')
    if not letter:
      continue
    y_top = float(bar_tops[i])
    if not np.isfinite(y_top):
      continue
    y = y_top + y_pad
    max_text_y = max(max_text_y, y)
    ax.text(
      x_positions[i],
      y,
      letter,
      ha='center',
      va='bottom',
      fontweight='bold',
      fontsize=9,
      color='#222',
      bbox=dict(boxstyle='round,pad=0.12', facecolor='#ffffcc', edgecolor='#cccc88', alpha=0.8),
    )

  if max_text_y > ymax:
    ax.set_ylim(ymin, max_text_y + y_span * 0.06)


def _annotate_cld_horizontal(ax, cld_dict: Dict[str, str], algo_order: List[str], x_offset_ratio: float = 0.02):
  """Annotate CLD letters to the right of horizontal bars."""
  if not cld_dict:
    return

  xmin, xmax = ax.get_xlim()
  x_pad = (xmax - xmin) * x_offset_ratio

  for patch, algo in zip(ax.patches, algo_order):
    letter = cld_dict.get(algo, '')
    if not letter:
      continue
    x = patch.get_width() + x_pad
    y = patch.get_y() + patch.get_height() / 2
    ax.text(
      x,
      y,
      letter,
      ha='left',
      va='center',
      fontweight='bold',
      fontsize=10,
      color='#222',
      bbox=dict(boxstyle='round,pad=0.15', facecolor='#ffffcc', edgecolor='#cccc88', alpha=0.85)
    )

  ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.12)


# =============================================================================
# Utility Functions
# =============================================================================

def safe_parse_error_analysis(error_str: str) -> Optional[dict]:
  """
  Safely parse the error_analysis string without using eval().

  Uses ast.literal_eval which is safe as it can only evaluate literal
  Python data structures (dict, list, str, int, float, bool, None).
  """
  if pd.isna(error_str) or error_str in ('nan', '', 'None'):
    return None

  try:
    # Clean the string from np.float64() wrapping
    cleaned = str(error_str)
    # Replace numpy type wrappers: np.float64(X) -> X
    cleaned = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', cleaned)

    # Use ast.literal_eval (safer than eval)
    result = ast.literal_eval(cleaned)
    return result if isinstance(result, dict) else None
  except (ValueError, SyntaxError, TypeError):
    return None


def _natural_sort_key(value: Any) -> Tuple[Any, ...]:
  """Stable alphanumeric sort key (e.g. Human2 < Human10)."""
  text = str(value)
  parts = re.split(r'(\d+)', text)
  return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def extract_celltype_errors_from_classwise(
    classwise_str: str,
    label_map: Optional[Dict[str, str]] = None
) -> Dict[str, float]:
  """
  Extract error rates by cell type from ClassWise data.

  The error rate is calculated as 1 - Recall for each cell type.
  Recall represents the proportion of cells of a given type correctly identified.

  Args:
    classwise_str: String representing the ClassWise dictionary
    label_map: Optional dictionary to convert numeric indices to cell type names
               (e.g., {'0': 'alpha', '1': 'beta', ...})

  Returns:
    Dictionary {cell_type: error_rate}
  """
  if pd.isna(classwise_str) or classwise_str in ('nan', '', 'None'):
    return {}

  try:
    # Clean the string from np.float64() wrapping
    cleaned = str(classwise_str)
    cleaned = re.sub(r'np\.\w+\(([^)]+)\)', r'\1', cleaned)

    # Parse the dictionary
    classwise_data = ast.literal_eval(cleaned)
    if not isinstance(classwise_data, dict):
      return {}

    errors = {}
    for ct_key, ct_metrics in classwise_data.items():
      if not isinstance(ct_metrics, dict):
        continue

      # Extract Recall to calculate the error rate
      recall = ct_metrics.get('Recall')
      if recall is not None:
        try:
          recall_val = float(recall)
          error_rate = 1.0 - recall_val

          # Convert numeric index to cell type name if label_map is provided
          if label_map and str(ct_key) in label_map:
            ct_name = label_map[str(ct_key)]
          else:
            ct_name = str(ct_key)

          errors[ct_name] = error_rate
        except (ValueError, TypeError):
          continue

    return errors
  except (ValueError, SyntaxError, TypeError):
    return {}


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
  """
  Calculate the confidence interval for a series of values.

  Args:
    values: Array of values
    confidence: Confidence level (0.95 = 95%)

  Returns:
    Tuple (lower_bound, upper_bound) of the confidence interval
  """
  n = len(values)
  if n < 2:
    return (np.nan, np.nan)

  mean = np.mean(values)
  se = stats.sem(values)

  # Use Student's t-distribution for small samples
  ci = stats.t.interval(confidence, n-1, loc=mean, scale=se)
  return ci


def add_figure_caption(fig: plt.Figure, analysis_type: str, metric: str = None,
            n_runs: int = None, extra_info: str = None):
  """
  Add an explanatory caption suitable for biologists at the bottom of the figure.

  Args:
    fig: Matplotlib figure
    analysis_type: Analysis type (key from FIGURE_CAPTIONS)
    metric: Metric used (optional)
    n_runs: Number of runs (optional)
    extra_info: Additional information (optional)
  """
  caption = FIGURE_CAPTIONS.get(analysis_type, "")

  if not caption:
    return

  # Customize with parameters
  if metric:
    caption = caption.replace("NMI/ARI/ACC", metric)
  if n_runs:
    caption = caption.replace("N repetitions", f"{n_runs} repetitions")
  if extra_info:
    caption = f"{caption}\n{extra_info}"

  # Add the legend
  fig.text(0.5, -0.02, caption, ha='center', va='top', fontsize=8,
       style='italic', wrap=True,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
       transform=fig.transFigure)


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_benchmark_detailed(filepath: str) -> Optional[pd.DataFrame]:
  """Load and parse benchmark_detailed.csv file."""
  try:
    df = pd.read_csv(filepath, on_bad_lines='warn')
    # Generic renaming for benchmark exports
    column_map = {'algorithm_name': 'algorithm'}
    for col in df.columns:
      if col.startswith('test_metrics_'):
        column_map[col] = col.replace('test_metrics_', 'test_')
      elif col.startswith('train_metrics_'):
        column_map[col] = col.replace('train_metrics_', 'train_')
      elif col.startswith('val_metrics_'):
        column_map[col] = col.replace('val_metrics_', 'val_')
    df.rename(columns=column_map, inplace=True)
    return df
  except Exception as e:
    st.error(f"Error while loading {filepath}: {e}")
    return None


def load_analysis_results(filepath: str) -> Optional[pd.DataFrame]:
  """Load results.csv or analysis_results.csv file."""
  try:
    df = pd.read_csv(filepath, on_bad_lines='warn')
    return df
  except Exception as e:
    st.error(f"Error while loading {filepath}: {e}")
    return None


def load_labels_from_directory(labels_dir: str) -> Dict[str, Dict[str, pd.DataFrame]]:
  """
  Load all label files from a labels directory.

  Supports two patterns:
  - Benchmark mode: benchmark_{algo}_run{id}_{split}.csv
  - Standard mode: labels_{algo}_run{id}.csv

  Returns a dict structure:
  {
    'algorithm_name': {
      'run0_train': DataFrame with 'predicted_label' and 'true_label' columns,
      'run0_test': ...,
      'run0_full': ... (for standard mode without split)
    }
  }
  """
  labels_data = {}

  if not os.path.isdir(labels_dir):
    return labels_data

  import re
  # Pattern 1: benchmark_{algo}_run{id}_{split}.csv (benchmark mode with split)
  pattern_benchmark = re.compile(r'^benchmark_(.+)_run(\d+)_(train|test|val)\.csv$')
  # Pattern 2: labels_{algo}_run{id}.csv (standard mode without split)
  pattern_standard = re.compile(r'^labels_(.+)_run(\d+)\.csv$')

  for filename in os.listdir(labels_dir):
    filepath = os.path.join(labels_dir, filename)

    # Try benchmark pattern first
    match = pattern_benchmark.match(filename)
    if match:
      algo = match.group(1)
      run_id = match.group(2)
      split = match.group(3)

      try:
        df = pd.read_csv(filepath)
        if algo not in labels_data:
          labels_data[algo] = {}
        labels_data[algo][f'run{run_id}_{split}'] = df
      except Exception:
        pass # Skip files that can't be read
      continue

    # Try standard pattern
    match = pattern_standard.match(filename)
    if match:
      algo = match.group(1)
      run_id = match.group(2)

      try:
        df = pd.read_csv(filepath)
        if algo not in labels_data:
          labels_data[algo] = {}
        # Use 'full' as split name for standard mode (no train/test split)
        labels_data[algo][f'run{run_id}_full'] = df
      except Exception:
        pass # Skip files that can't be read

  return labels_data


def _inject_batch_from_h5ad(labels_data: Dict[str, Dict[str, pd.DataFrame]],
               load_dir: str) -> None:
  """Inject batch column into label DataFrames from saved H5AD files.

  When label CSVs were saved without the batch column, this function
  reconstructs the batch assignments by loading H5AD files saved during
  the run:
  - benchmark mode: data/benchmark/{train,test,val}.h5ad
  - standard mode: data/processed.h5ad (mapped to split "full")

  Args:
    labels_data: Mutable dict of {algo: {run_split: DataFrame}}.
    load_dir: Path to the results directory (contains ../data/benchmark/).
  """
  # Check if any DataFrames already have 'batch'
  needs_batch = False
  for algo_runs in labels_data.values():
    for df in algo_runs.values():
      if 'batch' not in df.columns and 'true_label' in df.columns:
        needs_batch = True
        break
    if needs_batch:
      break

  if not needs_batch:
    return

  run_root = os.path.dirname(load_dir)

  def _read_batch_array(h5ad_path: str) -> Optional[np.ndarray]:
    """Read batch labels from an H5AD file with minimal dependencies."""
    try:
      import anndata as ad
      adata = ad.read_h5ad(h5ad_path, backed='r')
    except Exception:
      return None

    try:
      for col in ['batch', 'Batch', 'donor', 'sample']:
        if col in adata.obs.columns:
          return np.asarray(adata.obs[col]).astype(str)
    except Exception:
      return None
    finally:
      if getattr(adata, 'file', None) is not None:
        try:
          adata.file.close()
        except Exception:
          pass

    return None

  # Load batch info per split from saved H5AD files
  split_batches: Dict[str, np.ndarray] = {}
  for split_name in ['train', 'test', 'val']:
    candidates = [
      os.path.join(run_root, 'data', 'benchmark', f'{split_name}.h5ad'),
      os.path.join(run_root, 'data', f'{split_name}.h5ad'),
    ]
    for h5ad_path in candidates:
      if not os.path.isfile(h5ad_path):
        continue
      batch_arr = _read_batch_array(h5ad_path)
      if batch_arr is not None:
        split_batches[split_name] = batch_arr
        break

  # Standard mode fallback (single file, mapped to runX_full labels)
  full_candidates = [
    os.path.join(run_root, 'data', 'processed.h5ad'),
    os.path.join(run_root, 'data', 'input.h5ad'),
    os.path.join(run_root, 'data', 'full.h5ad'),
  ]
  for h5ad_path in full_candidates:
    if not os.path.isfile(h5ad_path):
      continue
    batch_arr = _read_batch_array(h5ad_path)
    if batch_arr is not None:
      split_batches['full'] = batch_arr
      break

  # Legacy standard-run fallback:
  # If saved H5AD files do not carry batch metadata, try to reconstruct
  # the batch vector from config_used.json + original source H5AD.
  if 'full' not in split_batches:
    try:
      import itertools
      import json
      import anndata as ad
      from utils.dataset_splitter import DatasetSplitter, get_batch_column

      # Find a target row count from any run*_full labels file.
      full_lengths = []
      for algo_runs in labels_data.values():
        for run_key, df in algo_runs.items():
          if isinstance(run_key, str) and run_key.endswith('_full') and isinstance(df, pd.DataFrame):
            full_lengths.append(len(df))
      target_len = max(full_lengths) if full_lengths else None

      if target_len is not None and target_len > 0:
        config_candidates = [
          os.path.join(run_root, 'config', 'config_used.json'),
          os.path.join(load_dir, 'config', 'config_used.json'),
        ]
        config = None
        for cfg_path in config_candidates:
          if os.path.isfile(cfg_path):
            try:
              with open(cfg_path) as f:
                config = json.load(f)
              break
            except Exception:
              continue

        if config:
          data_file = (config.get('data') or {}).get('file')
          source_candidates = []
          if data_file:
            source_candidates.append(data_file)
            basename = os.path.basename(data_file)
            source_candidates.append(os.path.join(os.getcwd(), 'data', basename))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            source_candidates.append(os.path.join(project_root, 'data', basename))
            source_candidates.append(os.path.join(load_dir, '..', '..', '..', '..', 'data', basename))

          source_path = None
          for cand in source_candidates:
            cand = os.path.normpath(cand)
            if os.path.isfile(cand):
              source_path = cand
              break

          if source_path:
            adata_src = ad.read_h5ad(source_path, backed='r')
            try:
              batch_col = get_batch_column(adata_src)
              if batch_col and batch_col in adata_src.obs.columns:
                batches_all = np.asarray(adata_src.obs[batch_col]).astype(str)
                unique_batches = list(pd.unique(batches_all))

                # Infer excluded batches by matching row count.
                # Keep search small: no exclusion, singles, then pairs.
                exclusion_sets = [tuple()]
                exclusion_sets.extend((b,) for b in unique_batches)
                exclusion_sets.extend(itertools.combinations(unique_batches, 2))

                balance_cfg = config.get('batch_balancing') or {}
                balance_enabled = bool(balance_cfg.get('enabled', False))
                balance_target = balance_cfg.get('target', None)
                random_seed = (config.get('execution') or {}).get('random_seed', 42)

                def _apply_exclusion(excluded: tuple) -> np.ndarray:
                  if not excluded:
                    return np.ones(len(batches_all), dtype=bool)
                  return ~pd.Series(batches_all).isin(list(excluded)).to_numpy()

                reconstructed = None
                best_excluded = None

                for excluded in exclusion_sets:
                  keep_mask = _apply_exclusion(excluded)
                  sub_batches = batches_all[keep_mask]

                  if not balance_enabled:
                    if len(sub_batches) == target_len:
                      reconstructed = sub_batches
                      best_excluded = excluded
                      break
                    continue

                  if balance_target is None:
                    continue

                  # Fast length check before running deterministic balancing.
                  counts = pd.Series(sub_batches).value_counts()
                  expected_len = int(sum(min(int(c), int(balance_target)) for c in counts.values))
                  if expected_len != target_len:
                    continue

                  # Reproduce CLI standard balancing order with DatasetSplitter.
                  try:
                    adata_tmp = ad.AnnData(
                      X=np.zeros((len(sub_batches), 1), dtype=np.float32),
                      obs=pd.DataFrame({batch_col: sub_batches.astype(str)})
                    )
                    splitter = DatasetSplitter(random_state=int(random_seed))
                    balanced = splitter.balance_by_batch(
                      adata_tmp,
                      batch_col=batch_col,
                      target_per_batch=int(balance_target),
                      preserve_labels=False,
                      label_col='Group'
                    )
                    rec_batches = np.asarray(balanced.obs[batch_col]).astype(str)
                    if len(rec_batches) == target_len:
                      reconstructed = rec_batches
                      best_excluded = excluded
                      break
                  except Exception:
                    continue

                if reconstructed is not None:
                  split_batches['full'] = reconstructed
                  if best_excluded:
                    st.info(
                      "Fallback batch reconstructed from config/H5AD source "
                      f"(excluded batches inferred: {list(best_excluded)})."
                    )
            finally:
              if getattr(adata_src, 'file', None) is not None:
                try:
                  adata_src.file.close()
                except Exception:
                  pass
    except Exception:
      pass

  if not split_batches:
    return

  # Inject batch column into DataFrames that match by split and row count
  for algo_runs in labels_data.values():
    for run_key, df in algo_runs.items():
      if 'batch' in df.columns:
        continue
      # Extract split name from run key (e.g. 'run0_test' -> 'test')
      parts = run_key.rsplit('_', 1)
      if len(parts) != 2:
        continue
      split_name = parts[1]
      if split_name not in split_batches:
        continue
      batch_arr = split_batches[split_name]
      if len(batch_arr) == len(df):
        df['batch'] = batch_arr
      elif len(batch_arr) > len(df):
        # Keep alignment-by-order as fallback when labels were truncated.
        df['batch'] = batch_arr[:len(df)]


def detect_result_type(results_dir: str) -> str:
  """Detect the type of results based on available files."""
  # Check current dir
  if os.path.exists(os.path.join(results_dir, 'benchmark_detailed.csv')):
    return 'benchmark_detailed'
  elif os.path.exists(os.path.join(results_dir, 'results.csv')):
    return 'analysis_results'
  elif os.path.exists(os.path.join(results_dir, 'analysis_results.csv')):
    return 'analysis_results'
  
  # Check subdirectory 'results'
  sub_results = os.path.join(results_dir, 'results')
  if os.path.isdir(sub_results):
    if os.path.exists(os.path.join(sub_results, 'benchmark_detailed.csv')):
      return 'benchmark_detailed_sub'
    elif os.path.exists(os.path.join(sub_results, 'results.csv')):
      return 'analysis_results_sub'
    elif os.path.exists(os.path.join(sub_results, 'analysis_results.csv')):
      return 'analysis_results_sub'
      
  return 'unknown'


def _try_load_label_map_from_data(load_dir: str) -> Optional[Dict[str, str]]:
  """Try to reconstruct label_map from config_used.json -> original data file."""
  import json
  # Look for config in parent or sibling directories
  for config_path in [
    os.path.join(load_dir, '..', 'config', 'config_used.json'),
    os.path.join(load_dir, 'config', 'config_used.json'),
  ]:
    config_path = os.path.normpath(config_path)
    if not os.path.isfile(config_path):
      continue
    try:
      with open(config_path) as f:
        config = json.load(f)
      data_file = config.get('data', {}).get('file')
      if not data_file or not data_file.endswith('.h5ad'):
        continue

      # Try multiple paths: original, local data/, relative to project
      candidates = [data_file]
      basename = os.path.basename(data_file)
      # Try data/ relative to CWD
      candidates.append(os.path.join(os.getcwd(), 'data', basename))
      # Try data/ relative to project root (parent of src/)
      project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
      candidates.append(os.path.join(project_root, 'data', basename))
      # Try relative to results dir (up to results_full_benchmark_5runs/../../../data/)
      candidates.append(os.path.join(load_dir, '..', '..', '..', '..', 'data', basename))

      for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if not os.path.isfile(candidate):
          continue
        try:
          adata = None
          # Prefer lightweight anndata loading; scanpy import can fail in some envs.
          try:
            import anndata as ad
            adata = ad.read_h5ad(candidate, backed='r')
          except Exception:
            import scanpy as sc
            adata = sc.read_h5ad(candidate, backed='r')

          try:
            # Best source: original label_map stored in AnnData.
            raw_map = adata.uns.get('label_map')
            if isinstance(raw_map, dict) and raw_map:
              # DataHandler convention: {label_name: encoded_id}
              if all(not str(k).isdigit() for k in raw_map.keys()):
                inv_map = {}
                for k, v in raw_map.items():
                  try:
                    inv_map[str(int(v))] = str(k)
                  except Exception:
                    continue
                if inv_map:
                  return inv_map
              # Already explorer convention: {encoded_id: label_name}
              direct_map = {}
              for k, v in raw_map.items():
                if str(k).isdigit():
                  direct_map[str(k)] = str(v)
              if direct_map:
                return direct_map

            # Strong fallback: infer mapping from Group numeric ids to readable names.
            if 'Group' in adata.obs:
              group_series = adata.obs['Group']
              group_vals = group_series.astype(str)
              for name_col in ['celltype', 'cell_type', 'labels', 'CellType', 'label']:
                if name_col not in adata.obs.columns:
                  continue
                name_vals = adata.obs[name_col].astype(str)
                tmp = pd.DataFrame({'gid': group_vals, 'name': name_vals}).dropna()
                if tmp.empty:
                  continue
                # Keep only unambiguous mappings (one name per group id).
                per_gid_unique = tmp.groupby('gid')['name'].nunique()
                if not (per_gid_unique == 1).all():
                  continue
                gid_to_name = tmp.groupby('gid')['name'].first().to_dict()
                if gid_to_name:
                  return {str(k): str(v) for k, v in gid_to_name.items()}

              # Last fallback for numeric Group: preserve ids (avoid wrong re-indexing).
              unique_groups = sorted(set(group_vals.tolist()), key=_natural_sort_key)
              if unique_groups:
                return {str(g): str(g) for g in unique_groups}
          finally:
            if hasattr(adata, 'file') and adata.file is not None:
              adata.file.close()
        except Exception:
          pass
    except Exception:
      pass
  return None


def load_results_from_directory(results_dir: str, condition_name: str) -> Optional[Dict]:
  """Load results from a directory."""
  result_type = detect_result_type(results_dir)

  # Handle subdirectory cases
  load_dir = results_dir
  if result_type.endswith('_sub'):
    load_dir = os.path.join(results_dir, 'results')
    result_type = result_type.replace('_sub', '')

  if result_type == 'benchmark_detailed':
    filepath = os.path.join(load_dir, 'benchmark_detailed.csv')
    df = load_benchmark_detailed(filepath)
  elif result_type == 'analysis_results':
    for fname in ['results.csv', 'analysis_results.csv']:
      filepath = os.path.join(load_dir, fname)
      if os.path.exists(filepath):
        df = load_analysis_results(filepath)
        break
    else:
      df = None
  else:
    df = None

  if df is not None:
    # Try to load labels from the labels directory
    labels_dir = os.path.join(load_dir, 'labels')
    labels_data = load_labels_from_directory(labels_dir)

    # Inject batch column from saved H5AD files if missing from CSVs
    if labels_data:
      _inject_batch_from_h5ad(labels_data, load_dir)

    # Try to load label_map.json for decoding numeric labels
    label_map = None
    label_map_file = os.path.join(labels_dir, 'label_map.json')
    if os.path.isfile(label_map_file):
      try:
        import json
        with open(label_map_file) as f:
          label_map = json.load(f) # {int_str: original_name}
      except Exception:
        pass
    # Fallback: try to load from config_used.json -> original data file
    if label_map is None:
      label_map = _try_load_label_map_from_data(load_dir)

    return {
      'df': df,
      'type': result_type,
      'condition': condition_name,
      'path': load_dir,
      'labels': labels_data, # May be empty dict if no labels found
      'label_map': label_map # May be None
    }
  return None


def aggregate_metrics(all_data: Dict[str, Dict]) -> pd.DataFrame:
  """Aggregate metrics per algorithm across all conditions."""
  records = []

  for condition, data in all_data.items():
    df = data['df']
    result_type = data['type']

    # Determine metric columns based on result type
    if result_type == 'benchmark_detailed':
      metric_cols = [c for c in df.columns if c.startswith('test_')]
      train_cols = [c for c in df.columns if c.startswith('train_')]
    else:
      # Standard mode: keep numeric columns that are not metadata
      meta_cols = {'algorithm', 'algorithm_name', 'run_id', 'runtime', 'params', 'timestamp'}
      metric_cols = [c for c in df.columns if c not in meta_cols]
      train_cols = []

    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'
    if algo_col not in df.columns:
      continue

    for algo, group in df.groupby(algo_col):
      record = {
        'algorithm': algo,
        'condition': condition,
        'n_runs': len(group),
        'result_type': result_type
      }

      for col in metric_cols:
        if col in group.columns:
          values = pd.to_numeric(group[col], errors='coerce').dropna()
          if len(values) > 0:
            metric_name = col.replace('test_', '')
            record[f'{metric_name}_mean'] = values.mean()
            record[f'{metric_name}_std'] = values.std()

      for col in train_cols:
        if col in group.columns:
          values = pd.to_numeric(group[col], errors='coerce').dropna()
          if len(values) > 0:
            metric_name = col.replace('train_', '') + '_train'
            record[f'{metric_name}_mean'] = values.mean()
            record[f'{metric_name}_std'] = values.std()

      # Runtime
      if 'total_time' in group.columns:
        times = pd.to_numeric(group['total_time'], errors='coerce').dropna()
        if len(times) > 0:
          record['runtime_mean'] = times.mean()
          record['runtime_std'] = times.std()
      elif 'runtime' in group.columns:
        times = pd.to_numeric(group['runtime'], errors='coerce').dropna()
        if len(times) > 0:
          record['runtime_mean'] = times.mean()
          record['runtime_std'] = times.std()

      # Collect ClassWise dictionary if exists (take the first one as they should be similar or just indicative)
      # Note: We can't average dictionaries easily. 
      # We'll check if 'ClassWise' column exists and store it as is from the first row for reference
      if 'ClassWise' in group.columns:
          # It's stored as string representation of dict in CSV usually
          first_val = group['ClassWise'].iloc[0]
          if pd.notna(first_val):
              record['ClassWise_raw'] = first_val

      records.append(record)

  return pd.DataFrame(records)


def _sort_metrics(metrics: List[str]) -> List[str]:
  """Sort metrics with a preferred order, then alphabetical for the rest."""
  preferred = [
    'NMI', 'ARI', 'ACC', 'UCA', 'Silhouette',
    'F1_Macro', 'BalancedACC', 'RareACC', 'BalancedRareACC', 'KNN_Purity',
    'NNO_Spearman', 'NNO_FoldEnrichment',
    'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity',
    'Isolated labels', 'KMeans NMI', 'KMeans ARI',
    'Silhouette label', 'cLISI', 'PCR comparison',
    'Jaccard index',
    'Batch correction', 'Inter cell-type conservation',
    'Intra cell-type conservation', 'scIB-E Total score',
  ]
  preferred_set = set(preferred)
  ordered = [m for m in preferred if m in metrics]
  remaining = sorted([m for m in metrics if m not in preferred_set])
  return ordered + remaining


def get_available_metrics(agg_df: pd.DataFrame, require_train_test: bool = False) -> List[str]:
  """
  Extract available metric names from aggregated dataframe.
  If require_train_test=True, only keep metrics that have both train and test stats.
  """
  if agg_df is None or agg_df.empty:
    return []

  test_metrics = {
    col[:-5]
    for col in agg_df.columns
    if col.endswith('_mean')
    and not col.endswith('_train_mean')
    and not col.startswith('runtime')
  }

  if not require_train_test:
    return _sort_metrics(list(test_metrics))

  train_metrics = {
    col[:-11]  # strip "_train_mean"
    for col in agg_df.columns
    if col.endswith('_train_mean')
  }
  return _sort_metrics(list(test_metrics & train_metrics))


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_algorithm_comparison(agg_df: pd.DataFrame, selected_algos: List[str],
                selected_conditions: List[str], metric: str = 'NMI',
                all_data: Optional[Dict] = None, show_cld: bool = False) -> plt.Figure:
  """Create bar plot comparing algorithms across conditions."""
  # Filter data
  df = agg_df[
    (agg_df['algorithm'].isin(selected_algos)) &
    (agg_df['condition'].isin(selected_conditions))
  ]

  if df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
    return fig

  mean_col = f'{metric}_mean'
  std_col = f'{metric}_std'

  if mean_col not in df.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'Metric {metric} not available', ha='center', va='center', fontsize=14)
    return fig

  algorithms = sorted(df['algorithm'].unique())
  conditions = sorted(df['condition'].unique())

  # Dynamic figure size based on number of conditions and algorithms
  n_conditions = len(conditions)
  n_algos = len(algorithms)
  fig_width = max(12, 2 * n_algos + 2)  # Scale with algorithms
  fig_height = 8 if n_conditions <= 6 else 10  # Taller for more conditions

  fig, ax = plt.subplots(figsize=(fig_width, fig_height))

  x = np.arange(len(algorithms))
  width = 0.8 / len(conditions)

  # Use a high-contrast vibrant palette for many conditions
  colors = sns.color_palette("husl", len(conditions))
  all_means = []
  condition_x_positions: Dict[str, np.ndarray] = {}
  condition_bar_tops: Dict[str, np.ndarray] = {}

  for i, cond in enumerate(conditions):
    cond_data = df[df['condition'] == cond]

    means = []
    stds = []
    for algo in algorithms:
      algo_data = cond_data[cond_data['algorithm'] == algo]
      if len(algo_data) > 0:
        means.append(algo_data[mean_col].values[0])
        stds.append(algo_data[std_col].values[0] if std_col in algo_data.columns else 0)
      else:
        means.append(0)
        stds.append(0)

    offset = (i - len(conditions)/2 + 0.5) * width
    x_positions = x + offset
    ax.bar(x + offset, means, width, label=cond, yerr=stds, capsize=2, alpha=0.85, color=colors[i])
    condition_x_positions[cond] = x_positions
    condition_bar_tops[cond] = np.asarray(means, dtype=float) + np.asarray(stds, dtype=float)
    all_means.extend(means)

  ax.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
  ax.set_ylabel(metric, fontweight='bold', fontsize=12)
  ax.set_title(f'Comparison {metric} by Condition', fontweight='bold', fontsize=14)
  ax.set_xticks(x)
  ax.set_xticklabels([ALGO_DISPLAY_NAMES.get(a, a) for a in algorithms], rotation=45, ha='right')

  # Adjust legend placement based on number of conditions
  if n_conditions > 6:
    # Place legend below the plot for many conditions
    ax.legend(title='Condition', bbox_to_anchor=(0.5, -0.25), loc='upper center',
              fontsize=8, ncol=min(4, n_conditions), framealpha=0.9)
    plt.subplots_adjust(bottom=0.30)
  else:
    ax.legend(title='Condition', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    plt.subplots_adjust(bottom=0.15)

  ax.grid(axis='y', alpha=0.3)
  # Don't clamp y-axis; some metrics may not be strictly within [0,1]
  if all_means and all(0 <= m <= 1 for m in all_means):
    ax.set_ylim(0, 1.05)

  # CLD significance letters (after title and ylim are set)
  if show_cld and HAS_CLD and all_data is not None:
    values_by_condition = _collect_metric_values_by_condition(
      all_data, algorithms, conditions, metric
    )
    significant_conditions = []
    test_names = set()

    for cond in conditions:
      values_by_algo = values_by_condition.get(cond, {})
      if len(values_by_algo) < 2:
        continue

      cld, global_p, test_name = compute_significance_groups(values_by_algo)
      if global_p >= 0.05:
        continue

      _annotate_cld_grouped_bars(
        ax,
        cld,
        algorithms,
        condition_x_positions.get(cond, np.array([])),
        condition_bar_tops.get(cond, np.array([])),
      )
      significant_conditions.append(cond)
      test_names.add(test_name)

    if significant_conditions:
      tests = ", ".join(sorted(test_names))
      ax.set_title(
        ax.get_title()
        + f'\n(CLD by condition: {len(significant_conditions)}/{len(conditions)} significant, {tests})',
        fontsize=10,
        style='italic',
      )

  plt.tight_layout()
  return fig


def plot_metrics_heatmap(agg_df: pd.DataFrame, selected_algos: List[str],
             selected_conditions: List[str], metric: str = 'NMI') -> plt.Figure:
  """Create heatmap of metrics."""
  df = agg_df[
    (agg_df['algorithm'].isin(selected_algos)) &
    (agg_df['condition'].isin(selected_conditions))
  ]

  mean_col = f'{metric}_mean'

  if mean_col not in df.columns or df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, f'Metric {metric} not available', ha='center', va='center', fontsize=14)
    return fig

  # Pivot table
  pivot = df.pivot_table(values=mean_col, index='algorithm', columns='condition', aggfunc='mean')

  # Reorder algorithms
  algo_order = ['scdeepcluster', 'sccdcg', 'sc_mae', 'scname',
         'pca', 'pca_kmeans', 'pca_leiden']
  algo_order = [a for a in algo_order if a in pivot.index]
  # Append any algorithms present in the data but missing from the order list
  algo_order += [a for a in pivot.index if a not in algo_order]
  pivot = pivot.reindex(algo_order)

  fig, ax = plt.subplots(figsize=(12, 8))

  # Dynamic range: use [0,1] if values are within it, otherwise use data min/max
  try:
    vmin = np.nanmin(pivot.values)
    vmax = np.nanmax(pivot.values)
    if 0.0 <= vmin and vmax <= 1.0:
      vmin, vmax = 0.0, 1.0
  except Exception:
    vmin, vmax = 0.0, 1.0

  sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
        ax=ax, vmin=vmin, vmax=vmax, cbar_kws={'label': metric},
        annot_kws={'size': 10})

  ax.set_title(f'{metric} by Algorithm and Condition', fontweight='bold', fontsize=14)
  ax.set_xlabel('Condition', fontweight='bold', fontsize=12)
  ax.set_ylabel('Algorithm', fontweight='bold', fontsize=12)

  # Update y-axis labels
  new_labels = [ALGO_DISPLAY_NAMES.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()]
  ax.set_yticklabels(new_labels)

  plt.tight_layout()
  return fig


def plot_generalization_gap_boxplot(all_data: Dict[str, Dict], selected_algos: List[str],
                   selected_conditions: List[str], metric: str = 'NMI',
                   show_cld: bool = False) -> plt.Figure:
  """Create boxplot of generalization gap."""
  gap_records = []

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    result_type = data['type']

    if result_type != 'benchmark_detailed':
      continue

    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      train_col = f'train_{metric}'
      test_col = f'test_{metric}'

      train_val = pd.to_numeric(row.get(train_col), errors='coerce')
      test_val = pd.to_numeric(row.get(test_col), errors='coerce')

      if not np.isnan(train_val) and not np.isnan(test_val):
        gap_records.append({
          'algorithm': algo,
          'condition': condition,
          'gap': train_val - test_val
        })

  if not gap_records:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No train/test data disponibles', ha='center', va='center', fontsize=14)
    return fig

  gap_df = pd.DataFrame(gap_records)

  fig, ax = plt.subplots(figsize=(12, 6))

  algo_order = gap_df.groupby('algorithm')['gap'].mean().sort_values().index.tolist()

  sns.boxplot(data=gap_df, x='algorithm', y='gap', order=algo_order, ax=ax, palette='Set3')
  ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No gap')
  ax.axhline(y=0.05, color='orange', linestyle=':', alpha=0.7, label='Gap 5%')

  ax.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
  ax.set_ylabel(f'{metric} Gap (Train - Test)', fontweight='bold', fontsize=12)
  ax.set_title(f'Generalization Gap {metric}', fontweight='bold', fontsize=14)
  ax.set_xticklabels([ALGO_DISPLAY_NAMES.get(t.get_text(), t.get_text())
            for t in ax.get_xticklabels()], rotation=45, ha='right')
  ax.grid(axis='y', alpha=0.3)
  ax.legend()

  # CLD significance letters
  if show_cld and HAS_CLD:
    gap_values_by_algo = {}
    for algo in algo_order:
      vals = gap_df[gap_df['algorithm'] == algo]['gap'].tolist()
      if vals:
        gap_values_by_algo[algo] = vals
    if len(gap_values_by_algo) >= 2:
      cld, global_p, test_name = compute_significance_groups(gap_values_by_algo)
      if global_p < 0.05:
        _annotate_cld(ax, cld, algo_order)
        ax.set_title(ax.get_title() + f'\n(CLD: {test_name}, p={global_p:.2e})',
                     fontsize=10, style='italic')

  plt.tight_layout()
  return fig


def plot_generalization_gap_heatmap_fig(all_data: Dict[str, Dict], selected_algos: List[str],
                     selected_conditions: List[str], metric: str = 'NMI') -> plt.Figure:
  """Create detailed heatmap of generalization gap."""
  gap_records = []

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    result_type = data['type']

    if result_type != 'benchmark_detailed':
      continue

    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      train_col = f'train_{metric}'
      test_col = f'test_{metric}'

      train_val = pd.to_numeric(row.get(train_col), errors='coerce')
      test_val = pd.to_numeric(row.get(test_col), errors='coerce')

      if not np.isnan(train_val) and not np.isnan(test_val):
        gap_records.append({
          'algorithm': algo,
          'condition': condition,
          'gap': train_val - test_val
        })

  if not gap_records:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No train/test data disponibles', ha='center', va='center', fontsize=14)
    return fig

  gap_df = pd.DataFrame(gap_records)

  # Aggregate
  agg_df = gap_df.groupby(['algorithm', 'condition']).agg({
    'gap': ['mean', 'std']
  }).reset_index()
  agg_df.columns = ['algorithm', 'condition', 'gap_mean', 'gap_std']

  # Pivot
  pivot_mean = agg_df.pivot(index='algorithm', columns='condition', values='gap_mean')
  pivot_std = agg_df.pivot(index='algorithm', columns='condition', values='gap_std')

  # Order algorithms
  algo_order = ['scdeepcluster', 'sccdcg', 'sc_mae', 'scname',
         'pca', 'pca_kmeans', 'pca_leiden']
  algo_order = [a for a in algo_order if a in pivot_mean.index]
  algo_order += [a for a in pivot_mean.index if a not in algo_order]
  pivot_mean = pivot_mean.reindex(algo_order)
  pivot_std = pivot_std.reindex(algo_order)

  fig, ax = plt.subplots(figsize=(14, 8))

  # Create annotation labels
  annot_labels = np.empty_like(pivot_mean.values, dtype=object)
  for i in range(pivot_mean.shape[0]):
    for j in range(pivot_mean.shape[1]):
      mean_val = pivot_mean.iloc[i, j]
      std_val = pivot_std.iloc[i, j]
      if pd.isna(mean_val):
        annot_labels[i, j] = ''
      else:
        annot_labels[i, j] = f'{mean_val:.3f}\n±{std_val:.3f}'

  vmax = max(abs(pivot_mean.min().min()), abs(pivot_mean.max().max()))
  vmax = max(vmax, 0.1)

  sns.heatmap(pivot_mean, annot=annot_labels, fmt='', cmap='RdYlGn_r',
        ax=ax, vmin=-vmax, vmax=vmax, center=0,
        cbar_kws={'label': f'{metric} Gap (Train - Test)', 'shrink': 0.8},
        annot_kws={'size': 9}, linewidths=0.5)

  new_ylabels = [ALGO_DISPLAY_NAMES.get(t.get_text(), t.get_text())
          for t in ax.get_yticklabels()]
  ax.set_yticklabels(new_ylabels)

  ax.set_title(f'Generalization Gap {metric} by Condition', fontweight='bold', fontsize=14)
  ax.set_xlabel('Condition', fontweight='bold', fontsize=12)
  ax.set_ylabel('Algorithm', fontweight='bold', fontsize=12)

  plt.tight_layout()
  return fig


def perform_statistical_test(all_data: Dict[str, Dict], selected_algos: List[str],
               selected_conditions: List[str], metric: str = 'NMI',
               test_mode: str = 'pairwise_conditions') -> pd.DataFrame:
  """
  Perform statistical tests with multiple testing correction.

  Args:
    all_data: Dictionary of loaded results
    selected_algos: List of algorithms to include
    selected_conditions: List of conditions to compare
    metric: Metric to test (NMI, ARI, ACC, Silhouette)
    test_mode: 'pairwise_conditions' (compare conditions for each algo)
          or 'pairwise_algorithms' (compare algos for each condition)

  Returns:
    DataFrame with test results including raw and adjusted p-values
  """
  if len(selected_conditions) < 2:
    st.warning("Please select at least two conditions to run a statistical test.")
    return pd.DataFrame()

  records = []
  p_values = []

  # Helper to extract values for an algorithm/condition
  def get_values(condition: str, algo: str) -> Optional[np.ndarray]:
    if condition not in all_data:
      return None
    df = all_data[condition]['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    # Determine metric column
    metric_key = f'test_{metric}'
    col = metric_key if metric_key in df.columns else metric
    if col not in df.columns:
      return None

    vals = pd.to_numeric(df[df[algo_col] == algo][col], errors='coerce').dropna()
    return vals.values if len(vals) >= 3 else None

  if test_mode == 'pairwise_conditions':
    # Compare all pairs of conditions for each algorithm
    condition_pairs = [(selected_conditions[i], selected_conditions[j])
             for i in range(len(selected_conditions))
             for j in range(i+1, len(selected_conditions))]

    for algo in selected_algos:
      for c1, c2 in condition_pairs:
        vals1 = get_values(c1, algo)
        vals2 = get_values(c2, algo)

        if vals1 is None or vals2 is None:
          continue

        try:
          # Mann-Whitney U test (non-parametric, unpaired)
          stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')

          # Effect size: rank-biserial correlation
          n1, n2 = len(vals1), len(vals2)
          effect_size = 1 - (2 * stat) / (n1 * n2)

          records.append({
            'Algorithm': ALGO_DISPLAY_NAMES.get(algo, algo),
            'Condition 1': c1,
            'Condition 2': c2,
            'n1': n1,
            'n2': n2,
            'Mean 1': vals1.mean(),
            'Mean 2': vals2.mean(),
            'Delta': vals1.mean() - vals2.mean(),
            'U-statistic': stat,
            'Effect Size (r)': effect_size,
            'P-value (brut)': pval,
          })
          p_values.append(pval)
        except Exception:
          pass

  else: # pairwise_algorithms
    # Compare all pairs of algorithms for each condition
    algo_pairs = [(selected_algos[i], selected_algos[j])
           for i in range(len(selected_algos))
           for j in range(i+1, len(selected_algos))]

    for condition in selected_conditions:
      for a1, a2 in algo_pairs:
        vals1 = get_values(condition, a1)
        vals2 = get_values(condition, a2)

        if vals1 is None or vals2 is None:
          continue

        try:
          stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
          n1, n2 = len(vals1), len(vals2)
          effect_size = 1 - (2 * stat) / (n1 * n2)

          records.append({
            'Condition': condition,
            'Algo 1': ALGO_DISPLAY_NAMES.get(a1, a1),
            'Algo 2': ALGO_DISPLAY_NAMES.get(a2, a2),
            'n1': n1,
            'n2': n2,
            'Mean 1': vals1.mean(),
            'Mean 2': vals2.mean(),
            'Delta': vals1.mean() - vals2.mean(),
            'U-statistic': stat,
            'Effect Size (r)': effect_size,
            'P-value (brut)': pval,
          })
          p_values.append(pval)
        except Exception:
          pass

  if not records:
    return pd.DataFrame()

  result_df = pd.DataFrame(records)

  # Apply multiple testing correction
  if p_values and HAS_STATSMODELS:
    # FDR correction (Benjamini-Hochberg)
    _, p_adj_fdr, _, _ = multipletests(p_values, method='fdr_bh')
    # Bonferroni correction (more conservative)
    _, p_adj_bonf, _, _ = multipletests(p_values, method='bonferroni')

    result_df['P-value (FDR)'] = p_adj_fdr
    result_df['P-value (Bonf.)'] = p_adj_bonf

    # Significance stars based on FDR-corrected p-values
    def get_stars(p):
      if p < 0.001: return '***'
      elif p < 0.01: return '**'
      elif p < 0.05: return '*'
      return 'ns'

    result_df['Sig. (FDR)'] = [get_stars(p) for p in p_adj_fdr]
    result_df['Sig. (Bonf.)'] = [get_stars(p) for p in p_adj_bonf]
  else:
    # Fallback without statsmodels
    def get_stars(p):
      if p < 0.001: return '***'
      elif p < 0.01: return '**'
      elif p < 0.05: return '*'
      return 'ns'

    result_df['Sig. (brut)'] = [get_stars(p) for p in p_values]

    if not HAS_STATSMODELS and len(p_values) > 1:
      st.info("Install `statsmodels` for FDR correction: `pip install statsmodels`")

  return result_df


def plot_runtime_comparison(agg_df: pd.DataFrame, selected_algos: List[str],
               selected_conditions: List[str],
               all_data: Optional[Dict] = None, show_cld: bool = False) -> plt.Figure:
  """Create bar plot comparing runtimes."""
  df = agg_df[
    (agg_df['algorithm'].isin(selected_algos)) &
    (agg_df['condition'].isin(selected_conditions))
  ]

  if 'runtime_mean' not in df.columns or df.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Runtime data not available', ha='center', va='center', fontsize=14)
    return fig

  fig, ax = plt.subplots(figsize=(12, 6))

  # Average runtime across conditions for each algorithm
  runtime_df = df.groupby('algorithm').agg({
    'runtime_mean': 'mean',
    'runtime_std': 'mean'
  }).reset_index()
  runtime_df = runtime_df.sort_values('runtime_mean')

  colors = [ALGO_COLORS.get(a, '#333333') for a in runtime_df['algorithm']]

  bars = ax.barh(
    [ALGO_DISPLAY_NAMES.get(a, a) for a in runtime_df['algorithm']],
    runtime_df['runtime_mean'],
    xerr=runtime_df['runtime_std'],
    color=colors,
    alpha=0.8,
    capsize=3
  )

  ax.set_xlabel('Runtime (seconds)', fontweight='bold', fontsize=12)
  ax.set_title('Runtime Comparison', fontweight='bold', fontsize=14)
  ax.grid(axis='x', alpha=0.3)

  # Add value labels
  for bar, val in zip(bars, runtime_df['runtime_mean']):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
        f'{val:.1f}s', va='center', fontsize=10)

  if show_cld and HAS_CLD and all_data is not None:
    algo_order = runtime_df['algorithm'].tolist()
    runtime_values_by_algo = _collect_runtime_values(all_data, algo_order, selected_conditions)
    if len(runtime_values_by_algo) >= 2:
      cld, global_p, test_name = compute_significance_groups(runtime_values_by_algo)
      if global_p < 0.05:
        _annotate_cld_horizontal(ax, cld, algo_order)
        ax.set_title(ax.get_title() + f'\n(CLD: {test_name}, p={global_p:.2e})',
                     fontsize=10, style='italic')

  plt.tight_layout()
  return fig


def create_summary_dataframe(agg_df: pd.DataFrame, selected_algos: List[str],
               selected_conditions: List[str]) -> pd.DataFrame:
  """Compatibility wrapper for the table helper module."""
  from .tables import create_summary_dataframe as _create_summary_dataframe

  return _create_summary_dataframe(agg_df, selected_algos, selected_conditions)


def _collect_train_counts(all_data: Dict[str, Dict], selected_conditions: List[str],
              average: bool = True) -> Dict[str, int]:
  """
  Collect the number of training cells per cell type across selected conditions.

  For benchmark mode (split): counts from _train label files.
  For standard mode (no split): counts from _full label files (all data = train).

  Args:
    all_data: Loaded results dictionary.
    selected_conditions: Conditions to include.
    average: If True, return averaged counts per run. If False, return raw totals
         across all runs (useful when test counts are also raw totals).

  Returns:
    Dict mapping cell type name (str) -> count (averaged per run or raw total).
  """
  from collections import Counter
  raw_counts_per_run = Counter()
  n_runs = 0

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue
    labels_dict = data.get('labels', {})
    label_map = data.get('label_map')
    if not labels_dict:
      continue

    # Pick the first algorithm (train counts are the same for all algos)
    first_algo = next(iter(labels_dict))
    runs = labels_dict[first_algo]

    for run_key, labels_df in runs.items():
      if 'true_label' not in labels_df.columns:
        continue
      # Use _train for split mode, _full for standard mode
      if '_train' not in run_key and '_full' not in run_key:
        continue

      true_vals = labels_df['true_label'].values.tolist()
      run_counter = Counter()
      for v in true_vals:
        # Decode numeric labels if needed
        key = str(v)
        if label_map:
          key = label_map.get(key, key)
        run_counter[key] += 1

      for k, cnt in run_counter.items():
        raw_counts_per_run[k] += cnt
      n_runs += 1

  if n_runs == 0:
    return {}
  if average:
    return {k: round(v / n_runs) for k, v in raw_counts_per_run.items()}
  return dict(raw_counts_per_run)


def _collect_eval_counts_by_celltype(
    all_data: Dict[str, Dict],
    selected_conditions: List[str],
    selected_algos: List[str]
) -> Dict[str, Dict[str, List[float]]]:
  """Collect evaluation-set counts per cell type (test in split mode, full in standard mode)."""
  counts = defaultdict(lambda: defaultdict(list))

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    labels_dict = data.get('labels', {}) or {}
    label_map = data.get('label_map')

    for algo in selected_algos:
      runs = labels_dict.get(algo, {}) or {}
      for run_key, labels_df in runs.items():
        if labels_df is None or labels_df.empty:
          continue
        run_key_str = str(run_key)
        if not (run_key_str.endswith('_test') or run_key_str.endswith('_full')):
          continue

        true_col, _ = _select_label_columns(labels_df)
        if true_col is None:
          continue

        tmp = labels_df[[true_col]].dropna(subset=[true_col]).copy()
        if tmp.empty:
          continue

        ct_values = tmp[true_col].astype(str).values
        if label_map:
          ct_values = [label_map.get(v, v) for v in ct_values]

        run_counts = pd.Series(ct_values).value_counts()
        for ct, n_val in run_counts.items():
          counts[algo][str(ct)].append(float(n_val))

  return counts


def _collect_eval_counts_by_celltype_batch(
    all_data: Dict[str, Dict],
    selected_conditions: List[str],
    selected_algos: List[str]
) -> Tuple[Dict[str, Dict[str, Dict[str, List[float]]]], bool]:
  """Collect evaluation-set counts per (batch, cell type) for each algorithm."""
  counts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  used_synthetic_global_batch = False

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    labels_dict = data.get('labels', {}) or {}
    label_map = data.get('label_map')

    for algo in selected_algos:
      runs = labels_dict.get(algo, {}) or {}
      for run_key, labels_df in runs.items():
        if labels_df is None or labels_df.empty:
          continue
        run_key_str = str(run_key)
        if not (run_key_str.endswith('_test') or run_key_str.endswith('_full')):
          continue

        true_col, _ = _select_label_columns(labels_df)
        if true_col is None:
          continue

        if 'batch' in labels_df.columns:
          tmp = labels_df[[true_col, 'batch']].dropna(subset=[true_col, 'batch']).copy()
        else:
          tmp = labels_df[[true_col]].dropna(subset=[true_col]).copy()
          tmp['batch'] = 'GLOBAL (batch missing)'
          used_synthetic_global_batch = True

        if tmp.empty:
          continue

        ct_values = tmp[true_col].astype(str).values
        if label_map:
          ct_values = [label_map.get(v, v) for v in ct_values]
        tmp['_ct'] = [str(v) for v in ct_values]
        tmp['batch'] = tmp['batch'].astype(str)

        for (batch, ct), grp in tmp.groupby(['batch', '_ct']):
          counts[algo][str(batch)][str(ct)].append(float(len(grp)))

  return counts, used_synthetic_global_batch


def analyze_celltype_errors_fig(all_data: Dict[str, Dict], selected_algos: List[str],
                 selected_conditions: List[str],
                 sort_by: str = 'n_samples') -> plt.Figure:
  """Create heatmap of cell type errors.

  Extract errors from error_analysis (benchmark mode) or ClassWise (standard mode).
  For standard mode, the error rate is computed as 1 - Recall.

  Args:
    all_data: Dictionary of data per condition
    selected_algos: List of selected algorithms
    selected_conditions: List of selected conditions
    sort_by: Sorting method for cell types:
      - 'n_samples': sort by ascending cell count (rare populations at top) [default]
      - 'error': sort by descending average error rate
  """
  all_celltype_errors = defaultdict(lambda: defaultdict(list))
  all_celltype_counts = defaultdict(lambda: defaultdict(list))
  data_source = None  # Track which source was used

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'
    label_map = data.get('label_map')  # To convert indices -> names

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      # Method 1: Try to extract from error_analysis (benchmark mode)
      error_str = str(row.get('error_analysis', ''))
      error_data = safe_parse_error_analysis(error_str)

      if error_data is not None and 'error_by_celltype' in error_data:
        data_source = 'error_analysis'
        for ct, ct_data in error_data.get('error_by_celltype', {}).items():
          if isinstance(ct_data, dict) and 'error_rate' in ct_data:
            ct_key = str(ct)
            # Decode numeric true-label ids to cell-type names when available.
            if label_map:
              ct_key = label_map.get(ct_key, ct_key)
            all_celltype_errors[ct_key][algo].append(ct_data['error_rate'])
            n_samples = ct_data.get('n_samples_mean', ct_data.get('n_samples'))
            try:
              n_val = float(n_samples)
              if np.isfinite(n_val):
                all_celltype_counts[ct_key][algo].append(n_val)
            except (TypeError, ValueError):
              pass
      else:
        # Method 2: Fallback to ClassWise (standard mode)
        # Look for ClassWise in several possible columns
        classwise_str = None
        for col in ['ClassWise', 'test_ClassWise', 'test_metrics_ClassWise']:
          if col in row.index and pd.notna(row.get(col)):
            classwise_str = str(row.get(col))
            break

        if classwise_str:
          data_source = data_source or 'ClassWise'
          errors_from_classwise = extract_celltype_errors_from_classwise(
            classwise_str, label_map
          )
          for ct, error_rate in errors_from_classwise.items():
            all_celltype_errors[ct][algo].append(error_rate)

  if not all_celltype_errors:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No cell-type error data', ha='center', va='center', fontsize=14)
    return fig

  # Get all celltypes and algorithms
  all_celltypes = sorted(all_celltype_errors.keys())
  all_algorithms = sorted(set(algo for ct_data in all_celltype_errors.values() for algo in ct_data.keys()))

  # Collect train counts for y-axis annotation (needed for sorting)
  train_counts = _collect_train_counts(all_data, selected_conditions)
  # Fallback counts from labels when n_samples is missing in error_analysis/ClassWise
  eval_counts_fallback = _collect_eval_counts_by_celltype(
    all_data, selected_conditions, selected_algos
  )

  # Build matrix
  matrix = np.zeros((len(all_celltypes), len(all_algorithms)))
  count_matrix = np.full((len(all_celltypes), len(all_algorithms)), np.nan)
  for i, ct in enumerate(all_celltypes):
    for j, algo in enumerate(all_algorithms):
      if algo in all_celltype_errors[ct]:
        matrix[i, j] = np.mean(all_celltype_errors[ct][algo])
      else:
        matrix[i, j] = np.nan
      primary_counts = all_celltype_counts[ct].get(algo, [])
      if primary_counts:
        count_matrix[i, j] = float(np.mean(primary_counts))
      else:
        fallback_counts = eval_counts_fallback.get(algo, {}).get(ct, [])
        if fallback_counts:
          count_matrix[i, j] = float(np.mean(fallback_counts))

  # Sort celltypes according to sort_by parameter
  if sort_by == 'n_samples':
    # Sort by number of samples (ascending: rare populations on top)
    n_samples_list = [train_counts.get(ct, float('inf')) for ct in all_celltypes]
    sort_idx = np.argsort(n_samples_list)
  else:  # default: sort by error
    # Sort by mean error (descending: highest errors on top)
    mean_errors = np.nanmean(matrix, axis=1)
    sort_idx = np.argsort(mean_errors)[::-1]

  matrix = matrix[sort_idx]
  count_matrix = count_matrix[sort_idx]
  all_celltypes = [all_celltypes[i] for i in sort_idx]

  # Build y-labels with train counts
  y_labels = []
  for ct in all_celltypes:
    n_train = train_counts.get(ct)
    if n_train is not None:
      y_labels.append(f"{ct} (n_train={n_train})")
    else:
      y_labels.append(ct)

  df_matrix = pd.DataFrame(matrix, index=y_labels,
               columns=[ALGO_DISPLAY_NAMES.get(a, a) for a in all_algorithms])
  annot_matrix = np.empty(df_matrix.shape, dtype=object)
  annot_matrix[:] = ""
  for i in range(df_matrix.shape[0]):
    for j in range(df_matrix.shape[1]):
      val = matrix[i, j]
      if not np.isfinite(val):
        continue
      text = f"{val:.2f}"
      n_val = count_matrix[i, j]
      if np.isfinite(n_val):
        text += f"\n(n={int(round(n_val))})"
      annot_matrix[i, j] = text

  fig, ax = plt.subplots(figsize=(12, max(8, len(all_celltypes) * 0.5)))

  sns.heatmap(df_matrix, annot=annot_matrix, fmt='', cmap='RdYlGn_r',
        ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Error Rate'},
        annot_kws={'size': 9}, mask=df_matrix.isna())

  # Title with data-source information
  if data_source == 'ClassWise':
    title = 'Cell-Type Error Rate (1 - Recall)'
    subtitle = 'Computed from ClassWise (standard mode)'
  else:
    title = 'Cell-Type Error Rate'
    subtitle = None

  ax.set_title(title, fontweight='bold', fontsize=14)
  if subtitle:
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha='center',
            fontsize=10, style='italic', color='gray')
  ax.text(
    0.5, -0.12,
    "Value = error rate; n = mean number of evaluated cells (TEST split in benchmark mode, FULL in standard mode).",
    transform=ax.transAxes, ha='center', va='top', fontsize=9, color='gray'
  )
  ax.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
  ax.set_ylabel('Cell Type', fontweight='bold', fontsize=12)

  plt.tight_layout()
  return fig


def analyze_celltype_errors_by_batch_fig(
    all_data: Dict[str, Dict],
    selected_algos: List[str],
    selected_conditions: List[str],
    sort_by: str = 'n_samples'
) -> plt.Figure:
  """Create heatmaps of cell type errors by batch (one per algorithm).

  Args:
    all_data: Dictionary of data per condition
    selected_algos: List of selected algorithms
    selected_conditions: List of selected conditions
    sort_by: Sorting method for cell types:
      - 'n_samples': sort by ascending cell count (rare populations at top) [default]
      - 'error': sort by descending average error rate
      - 'stable': stable alphabetical/natural sort
  """
  all_errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  all_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
  used_synthetic_global_batch = False

  # Track (condition, algo) pairs that got data from error_analysis
  _pairs_with_ea = set()

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'
    label_map = data.get('label_map')

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      error_str = str(row.get('error_analysis', ''))
      error_data = safe_parse_error_analysis(error_str)
      if error_data is None:
        continue

      ct_by_group = error_data.get('error_by_celltype_by_group', {})
      if not ct_by_group:
        continue

      _pairs_with_ea.add((condition, algo))
      for batch, ct_map in ct_by_group.items():
        for ct, ct_data in ct_map.items():
          if isinstance(ct_data, dict) and 'error_rate' in ct_data:
            ct_key = str(ct)
            # Decode numeric true-label ids to cell-type names when available.
            if label_map:
              ct_key = label_map.get(ct_key, ct_key)
            all_errors[algo][str(batch)][ct_key].append(ct_data['error_rate'])
            n_samples = ct_data.get('n_samples_mean', ct_data.get('n_samples'))
            try:
              n_val = float(n_samples)
              if np.isfinite(n_val):
                all_counts[algo][str(batch)][ct_key].append(n_val)
            except (TypeError, ValueError):
              pass

  # Fallback: compute celltype-by-batch errors from label files.
  # Applies per (condition, algo) pair where error_analysis was absent.
  from scipy.optimize import linear_sum_assignment as _lsa
  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue
    labels_by_algo = data.get('labels', {}) or {}
    label_map = data.get('label_map')
    for algo in selected_algos:
      if (condition, algo) in _pairs_with_ea:
        continue
      runs = labels_by_algo.get(algo, {}) or {}
      for run_key, labels_df in runs.items():
        if labels_df is None or labels_df.empty:
          continue
        if not (str(run_key).endswith('_test') or str(run_key).endswith('_full')):
          continue
        true_col, pred_col = _select_label_columns(labels_df)
        if true_col is None or pred_col is None:
          continue

        if 'batch' in labels_df.columns:
          tmp = labels_df[[true_col, pred_col, 'batch']].dropna(subset=[true_col, pred_col, 'batch']).copy()
        else:
          # Legacy standard exports may not contain a batch column.
          # Keep a usable fallback by aggregating as a single pseudo-batch.
          tmp = labels_df[[true_col, pred_col]].dropna(subset=[true_col, pred_col]).copy()
          tmp['batch'] = 'GLOBAL (batch missing)'
          used_synthetic_global_batch = True

        if tmp.empty:
          continue
        y_true = tmp[true_col].astype(str).values
        y_pred = tmp[pred_col].astype(str).values
        # Decode numeric true labels via label_map if available
        if label_map:
          y_true = np.array([label_map.get(t, t) for t in y_true])
        # Hungarian matching: map cluster IDs to true label names
        true_set = sorted(set(y_true))
        pred_set = sorted(set(y_pred))
        if set(y_true) != set(y_pred):
          true_to_idx = {l: i for i, l in enumerate(true_set)}
          pred_to_idx = {l: i for i, l in enumerate(pred_set)}
          cost = np.zeros((len(pred_set), len(true_set)), dtype=int)
          for t, p in zip(y_true, y_pred):
            cost[pred_to_idx[p], true_to_idx[t]] += 1
          row_ind, col_ind = _lsa(-cost)
          pred_to_true = {}
          for ri, ci in zip(row_ind, col_ind):
            pred_to_true[pred_set[ri]] = true_set[ci]
          for p in pred_set:
            if p not in pred_to_true:
              p_idx = pred_to_idx[p]
              if cost[p_idx].sum() > 0:
                pred_to_true[p] = true_set[np.argmax(cost[p_idx])]
              else:
                pred_to_true[p] = true_set[0]
          y_pred = np.array([pred_to_true[p] for p in y_pred])
        tmp['_true'] = y_true
        tmp['_pred'] = y_pred
        tmp['batch'] = tmp['batch'].astype(str)
        tmp['_wrong'] = (tmp['_true'] != tmp['_pred'])
        for (batch, ct), grp in tmp.groupby(['batch', '_true']):
          err_rate = float(grp['_wrong'].mean())
          all_errors[algo][str(batch)][str(ct)].append(err_rate)
          all_counts[algo][str(batch)][str(ct)].append(float(len(grp)))

  # Independent label-based counts are used as fallback when n_samples is absent
  # from error_analysis payloads.
  fallback_counts, fallback_used_synthetic_batch = _collect_eval_counts_by_celltype_batch(
    all_data, selected_conditions, selected_algos
  )
  if fallback_used_synthetic_batch:
    used_synthetic_global_batch = True

  if not all_errors:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No cell-type-by-batch error data',
            ha='center', va='center', fontsize=14)
    ax.axis('off')
    return fig

  # Collect train counts for sorting by n_samples
  train_counts = _collect_train_counts(all_data, selected_conditions)

  # Global ordering
  global_ct = defaultdict(list)
  global_batches = defaultdict(list)
  for algo, batch_map in all_errors.items():
    for batch, ct_map in batch_map.items():
      for ct, vals in ct_map.items():
        global_ct[ct].extend(vals)
        global_batches[batch].extend(vals)

  # Sort celltypes according to sort_by parameter
  if sort_by == 'n_samples':
    # Sort by number of samples (ascending: rare populations on top)
    ct_order = sorted(
      global_ct.keys(),
      key=lambda k: (train_counts.get(k, float('inf')), _natural_sort_key(k))
    )
  elif sort_by == 'error':
    ct_order = sorted(
      global_ct.keys(),
      key=lambda k: (-np.mean(global_ct[k]), _natural_sort_key(k))
    )
  else:  # default: stable order
    ct_order = sorted(global_ct.keys(), key=_natural_sort_key)

  # Keep a deterministic order across conditions to ease visual comparisons.
  batch_order = sorted(global_batches.keys(), key=_natural_sort_key)
  ct_display_labels = []
  for ct in ct_order:
    n_train = train_counts.get(ct)
    if n_train is not None:
      ct_display_labels.append(f"{ct} (n_train={n_train})")
    else:
      ct_display_labels.append(ct)

  n_algos = len(all_errors)
  fig, axes = plt.subplots(n_algos, 1, figsize=(12, max(6, len(ct_order) * 0.4) * n_algos))
  if n_algos == 1:
    axes = [axes]

  for ax, algo in zip(axes, sorted(all_errors.keys())):
    matrix = np.full((len(ct_order), len(batch_order)), np.nan)
    count_matrix = np.full((len(ct_order), len(batch_order)), np.nan)
    for i, ct in enumerate(ct_order):
      for j, batch in enumerate(batch_order):
        vals = all_errors[algo].get(batch, {}).get(ct, [])
        if vals:
          matrix[i, j] = float(np.mean(vals))
        n_vals = all_counts[algo].get(batch, {}).get(ct, [])
        if n_vals:
          count_matrix[i, j] = float(np.mean(n_vals))
        else:
          fb_vals = fallback_counts.get(algo, {}).get(str(batch), {}).get(ct, [])
          if fb_vals:
            count_matrix[i, j] = float(np.mean(fb_vals))

    df_matrix = pd.DataFrame(
      matrix,
      index=ct_display_labels,
      columns=[str(b) for b in batch_order]
    )

    annot_matrix = np.empty(df_matrix.shape, dtype=object)
    annot_matrix[:] = ""
    for i in range(df_matrix.shape[0]):
      for j in range(df_matrix.shape[1]):
        val = matrix[i, j]
        if not np.isfinite(val):
          continue
        text = f"{val:.2f}"
        n_val = count_matrix[i, j]
        if np.isfinite(n_val):
          text += f"\n(n={int(round(n_val))})"
        annot_matrix[i, j] = text

    sns.heatmap(
      df_matrix,
      annot=annot_matrix,
      fmt='',
      cmap='RdYlGn_r',
      ax=ax,
      vmin=0,
      vmax=1,
      cbar_kws={'label': 'Error Rate'},
      annot_kws={'size': 7},
      mask=df_matrix.isna()
    )

    ax.set_title(f"Cell-Type Error Rate by Batch - {ALGO_DISPLAY_NAMES.get(algo, algo)}",
          fontweight='bold', fontsize=12)
    if used_synthetic_global_batch:
      ax.text(
        0.5, 1.02,
        "Some conditions have no batch column: reconstructed global view shown.",
        transform=ax.transAxes, ha='center', va='bottom',
        fontsize=9, color='gray', style='italic'
      )
    ax.text(
      0.5, -0.13,
      "Value = error rate; n = mean number of evaluated cells (TEST split in benchmark mode, FULL in standard mode).",
      transform=ax.transAxes, ha='center', va='top',
      fontsize=9, color='gray'
    )
    ax.set_xlabel('Batch', fontweight='bold', fontsize=11)
    ax.set_ylabel('Cell Type', fontweight='bold', fontsize=11)

  plt.tight_layout()
  return fig


def plot_error_rate_by_batch(all_data: Dict[str, Dict], selected_algos: List[str],
               selected_conditions: List[str]) -> plt.Figure:
  """Create heatmap of error rates per batch.

  Priority:
  1) error_analysis.error_by_group (direct error rates)
  2) test_by_group fallback (error_rate = 1 - ACC)
  3) labels fallback (if labels include/infer a batch column)
  """
  all_batch_errors = defaultdict(lambda: defaultdict(list))
  has_metric_source = set()

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      # Primary source: error_analysis.error_by_group
      found_error_by_group = False
      error_str = str(row.get('error_analysis', ''))
      error_data = safe_parse_error_analysis(error_str)
      if error_data is not None:
        for batch, batch_data in error_data.get('error_by_group', {}).items():
          if isinstance(batch_data, dict) and 'error_rate' in batch_data:
            try:
              err = float(batch_data['error_rate'])
            except (TypeError, ValueError):
              continue
            if not np.isnan(err):
              all_batch_errors[str(batch)][algo].append(err)
              found_error_by_group = True
              has_metric_source.add((condition, algo))

      # Fallback source: test_by_group -> error_rate = 1 - ACC
      if not found_error_by_group:
        tbg_str = str(row.get('test_by_group', ''))
        tbg_data = safe_parse_error_analysis(tbg_str)
        if tbg_data is None:
          continue
        for batch, metrics in tbg_data.items():
          if not isinstance(metrics, dict):
            continue
          acc = metrics.get('ACC')
          if acc is None:
            continue
          try:
            acc_val = float(acc)
          except (TypeError, ValueError):
            continue
          if np.isnan(acc_val):
            continue
          all_batch_errors[str(batch)][algo].append(1.0 - acc_val)
          has_metric_source.add((condition, algo))

  # Last fallback: compute per-batch error directly from label files.
  # Useful for "standard" runs where benchmark_detailed/error_analysis is absent.
  from scipy.optimize import linear_sum_assignment as _lsa
  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    labels_by_algo = data.get('labels', {}) or {}
    label_map = data.get('label_map')
    for algo in selected_algos:
      if (condition, algo) in has_metric_source:
        continue
      runs = labels_by_algo.get(algo, {}) or {}
      for run_key, labels_df in runs.items():
        if labels_df is None or labels_df.empty:
          continue
        # Keep evaluation splits only
        if not (str(run_key).endswith('_test') or str(run_key).endswith('_full')):
          continue
        if 'batch' not in labels_df.columns:
          continue
        true_col, pred_col = _select_label_columns(labels_df)
        if true_col is None or pred_col is None:
          continue

        tmp = labels_df[[true_col, pred_col, 'batch']].dropna(subset=[true_col, pred_col, 'batch']).copy()
        if tmp.empty:
          continue
        tmp['batch'] = tmp['batch'].astype(str)
        y_true = tmp[true_col].astype(str).values
        y_pred = tmp[pred_col].astype(str).values

        # Decode true labels (numeric ids -> names) when label_map is available.
        if label_map:
          y_true = np.array([label_map.get(t, t) for t in y_true])

        # Align predicted cluster ids to true label names before error computation.
        true_set = sorted(set(y_true))
        pred_set = sorted(set(y_pred))
        if true_set and pred_set and set(y_true) != set(y_pred):
          true_to_idx = {l: i for i, l in enumerate(true_set)}
          pred_to_idx = {l: i for i, l in enumerate(pred_set)}
          cost = np.zeros((len(pred_set), len(true_set)), dtype=int)
          for t, p in zip(y_true, y_pred):
            cost[pred_to_idx[p], true_to_idx[t]] += 1
          row_ind, col_ind = _lsa(-cost)
          pred_to_true = {}
          for ri, ci in zip(row_ind, col_ind):
            pred_to_true[pred_set[ri]] = true_set[ci]
          for p in pred_set:
            if p not in pred_to_true:
              p_idx = pred_to_idx[p]
              pred_to_true[p] = true_set[np.argmax(cost[p_idx])] if cost[p_idx].sum() > 0 else true_set[0]
          y_pred = np.array([pred_to_true[p] for p in y_pred])

        tmp['_true'] = y_true
        tmp['_pred'] = y_pred
        err_by_batch = (tmp['_true'] != tmp['_pred']).groupby(tmp['batch']).mean()
        for batch, err in err_by_batch.items():
          try:
            err_val = float(err)
          except (TypeError, ValueError):
            continue
          if np.isnan(err_val):
            continue
          all_batch_errors[str(batch)][algo].append(err_val)

  if not all_batch_errors:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No batch-level error data\n\n'
              '(Requires error_by_group in error_analysis\n'
              'or ACC in test_by_group\n'
              'or labels + batch column)',
        ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  # Get all batches and algorithms
  all_batches = sorted(all_batch_errors.keys())
  all_algorithms = sorted(set(algo for batch_data in all_batch_errors.values() for algo in batch_data.keys()))

  # Build matrix
  matrix = np.zeros((len(all_batches), len(all_algorithms)))
  for i, batch in enumerate(all_batches):
    for j, algo in enumerate(all_algorithms):
      if algo in all_batch_errors[batch]:
        matrix[i, j] = np.mean(all_batch_errors[batch][algo])
      else:
        matrix[i, j] = np.nan

  # Sort batches by mean error (highest first)
  mean_errors = np.nanmean(matrix, axis=1)
  sort_idx = np.argsort(mean_errors)[::-1]
  matrix = matrix[sort_idx]
  all_batches = [all_batches[i] for i in sort_idx]

  df_matrix = pd.DataFrame(matrix, index=all_batches,
               columns=[ALGO_DISPLAY_NAMES.get(a, a) for a in all_algorithms])

  fig, ax = plt.subplots(figsize=(12, max(6, len(all_batches) * 0.8)))

  sns.heatmap(df_matrix, annot=True, fmt='.2f', cmap='RdYlGn_r',
        ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Error Rate'},
        annot_kws={'size': 10}, mask=df_matrix.isna())

  ax.set_title('Test-Batch Error Rate', fontweight='bold', fontsize=14)
  ax.set_xlabel('Algorithm', fontweight='bold', fontsize=12)
  ax.set_ylabel('Batch', fontweight='bold', fontsize=12)

  plt.tight_layout()
  return fig


def plot_confusion_matrix_by_batch(all_data: Dict[str, Dict], selected_algos: List[str],
                  selected_conditions: List[str]) -> Optional[plt.Figure]:
  """Create per-batch confusion matrices for each algorithm.

  Requires the 'batch' column in label CSV files. Returns None if batch data
  is not available (handled as Streamlit widget output instead of figure).
  """
  from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
  from scipy.optimize import linear_sum_assignment
  from matplotlib.patches import Rectangle
  from scrbenchmark.utils.metrics import compute_rare_class_accuracy

  # Collect label_map from any condition
  global_label_map = None
  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue
    lmap = data.get('label_map')
    if lmap:
      global_label_map = lmap
      break

  # Collect train counts (averaged per run)
  train_counts = _collect_train_counts(all_data, selected_conditions)

  # Collect labels per algo per batch
  algo_batch_data: Dict[str, Dict[str, Dict[str, list]]] = {}
  algo_run_counts: Dict[str, int] = {} # Track num runs per algo
  has_batch_column = False

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    labels_dict = data.get('labels', {})
    if not labels_dict:
      continue

    for algo, runs in labels_dict.items():
      if algo not in selected_algos:
        continue

      for run_key, labels_df in runs.items():
        if '_train' in run_key or '_val' in run_key:
          continue
        
        # Count valid test runs for this algo
        algo_run_counts[algo] = algo_run_counts.get(algo, 0) + 1

        if 'true_label' not in labels_df.columns or 'predicted_label' not in labels_df.columns:
          continue
        if 'batch' not in labels_df.columns:
          continue

        has_batch_column = True
        if algo not in algo_batch_data:
          algo_batch_data[algo] = {}

        for batch_id, group_df in labels_df.groupby('batch'):
          batch_key = str(batch_id)
          if batch_key not in algo_batch_data[algo]:
            algo_batch_data[algo][batch_key] = {'true': [], 'pred': []}
          algo_batch_data[algo][batch_key]['true'].extend(group_df['true_label'].values.tolist())
          algo_batch_data[algo][batch_key]['pred'].extend(group_df['predicted_label'].values.tolist())

  if not has_batch_column:
    return 'no_batch_column'

  if not algo_batch_data:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No label data available with batch column',
        ha='center', va='center', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  # Get all batches across all algos
  all_batches = sorted(set(
    b for algo_data in algo_batch_data.values() for b in algo_data.keys()
  ))

  # Get sorted algorithm list
  sorted_algos = sorted(algo_batch_data.keys())

  # Build one figure per algorithm with subplots for each batch
  n_batches = len(all_batches)
  n_algos = len(sorted_algos)
  n_cols = min(4, n_batches)
  n_rows_per_algo = (n_batches + n_cols - 1) // n_cols
  total_rows = n_rows_per_algo * n_algos

  fig, axes = plt.subplots(total_rows, n_cols,
               figsize=(5 * n_cols, 8.0 * total_rows),
               squeeze=False)

  for algo_idx, algo in enumerate(sorted_algos):
    batch_data = algo_batch_data[algo]
    row_offset = algo_idx * n_rows_per_algo

    for batch_idx, batch_key in enumerate(all_batches):
      r = row_offset + batch_idx // n_cols
      c = batch_idx % n_cols
      ax = axes[r][c]

      if batch_key not in batch_data:
        ax.text(0.5, 0.5, f'No data\n({batch_key})',
            ha='center', va='center', fontsize=10)
        ax.set_title(f'{ALGO_DISPLAY_NAMES.get(algo, algo)} - Batch {batch_key}',
              fontsize=10, fontweight='bold')
        ax.axis('off')
        continue

      y_true = np.array(batch_data[batch_key]['true'], dtype=str)
      y_pred = np.array(batch_data[batch_key]['pred'], dtype=str)

      # Decode numeric true labels
      if global_label_map is not None:
        try:
          [int(t) for t in set(y_true)]
          y_true = np.array([global_label_map.get(t, t) for t in y_true])
        except ValueError:
          pass

      # Hungarian mapping if label spaces differ
      true_labels_set = set(y_true)
      pred_labels_set = set(y_pred)

      if true_labels_set != pred_labels_set:
        true_unique = sorted(true_labels_set)
        pred_unique = sorted(pred_labels_set)
        true_to_idx = {l: i for i, l in enumerate(true_unique)}
        pred_to_idx = {l: i for i, l in enumerate(pred_unique)}

        cost = np.zeros((len(pred_unique), len(true_unique)), dtype=int)
        for t, p in zip(y_true, y_pred):
          cost[pred_to_idx[p], true_to_idx[t]] += 1

        row_ind, col_ind = linear_sum_assignment(-cost)
        pred_to_true = {}
        for ri, ci in zip(row_ind, col_ind):
          pred_to_true[pred_unique[ri]] = true_unique[ci]

        # Unmapped clusters: assign to the most frequent type in this cluster
        for p in pred_unique:
          if p not in pred_to_true:
            p_idx = pred_to_idx[p]
            if cost[p_idx].sum() > 0:
              best_true_idx = np.argmax(cost[p_idx])
              pred_to_true[p] = true_unique[best_true_idx]
            else:
              pred_to_true[p] = true_unique[0]

        y_pred = np.array([pred_to_true[p] for p in y_pred])

      labels = sorted(set(y_true) | set(y_pred))
      cm = confusion_matrix(y_true, y_pred, labels=labels)
      
      # Divide raw counts by number of runs to show "average per run" stats
      n_runs = algo_run_counts.get(algo, 1)
      cm_avg = cm / n_runs
      
      cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
      cm_norm = np.nan_to_num(cm_norm)

      im = ax.imshow(cm_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

      # Annotate with AVERAGE values
      for i in range(len(labels)):
        for j in range(len(labels)):
          text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
          val_avg = int(round(cm_avg[i, j]))
          ax.text(j, i, f'{val_avg}',
              ha='center', va='center', fontsize=7, color=text_color)

      # Highlight the diagonal (correct predictions)
      for i in range(len(labels)):
        ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                    edgecolor='green', lw=1.5))

      ax.set_xticks(range(len(labels)))
      ax.set_yticks(range(len(labels)))
      ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
      
      # Build y-axis labels with train counts
      y_label_names = []
      for l in labels:
        name = str(l)
        n_train = train_counts.get(name)
        if n_train is not None:
          y_label_names.append(f"{name}\n(tr:{n_train})")
        else:
          y_label_names.append(name)
      
      ax.set_yticklabels(y_label_names, fontsize=7)
      # Title: Show average n_test per run
      n_samples = len(y_true)
      n_avg = int(round(n_samples / n_runs))
      ax.set_title(f'{ALGO_DISPLAY_NAMES.get(algo, algo)} - Batch {batch_key}\n(n_test≈{n_avg})',
            fontsize=10, fontweight='bold')
      ax.set_xlabel('Predicted', fontsize=8)
      ax.set_ylabel('Vrai', fontsize=8)
      
      # Compute metrics for this batch
      try:
        b_acc = balanced_accuracy_score(y_true, y_pred)
        f1_mac = f1_score(y_true, y_pred, average='macro')
        rare_acc = compute_rare_class_accuracy(y_true, y_pred, threshold=0.05)
        
        summary_text = (
          f"Macro F1: {f1_mac:.3f}\n"
          f"Balanced ACC: {b_acc:.3f}"
        )
        if rare_acc is not None:
             summary_text += f"\nRare-ACC (<5%): {rare_acc:.3f}"
        
        ax.text(0.5, -0.25, summary_text,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='lightgray'))
      except Exception:
        pass # metrics failed (e.g. single class)

    # Hide unused subplots for this algo
    for extra_idx in range(n_batches, n_rows_per_algo * n_cols):
      r = row_offset + extra_idx // n_cols
      c = extra_idx % n_cols
      axes[r][c].axis('off')

  plt.tight_layout()
  return fig


def plot_confusion_patterns_fig(all_data: Dict[str, Dict], selected_algos: List[str],
                 selected_conditions: List[str]) -> plt.Figure:
  """Create bar chart of top confusion patterns."""
  all_confusion_counts = defaultdict(int)

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      error_str = str(row.get('error_analysis', ''))
      error_data = safe_parse_error_analysis(error_str)

      if error_data is None:
        continue

      for pair in error_data.get('top_confusion_pairs', []):
        if isinstance(pair, dict) and 'true' in pair and 'predicted' in pair:
          key = f"{pair['true']} -> {pair['predicted']}"
          all_confusion_counts[key] += pair.get('count', 1)

  if not all_confusion_counts:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center', fontsize=14)
    return fig

  # Sort and take top 15
  sorted_conf = sorted(all_confusion_counts.items(), key=lambda x: x[1], reverse=True)[:15]
  pairs, counts = zip(*sorted_conf)

  fig, ax = plt.subplots(figsize=(12, 8))

  colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(counts)))
  bars = ax.barh(pairs[::-1], counts[::-1], color=colors[::-1])

  ax.set_xlabel('Total number of confusions', fontweight='bold', fontsize=12)
  ax.set_title('Top confusion pairs between cell types', fontweight='bold', fontsize=14)
  ax.grid(axis='x', alpha=0.3)

  for bar, count in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
        str(count), va='center', fontsize=10)

  plt.tight_layout()
  return fig


def plot_confusion_matrix_detailed(all_data: Dict[str, Dict], selected_algos: List[str],
                  selected_conditions: List[str],
                  sort_by: str = 'n_samples') -> plt.Figure:
  """
  Creates a confusion matrix for each algorithm with F1 score, precision, and recall.

  Args:
    all_data: Dictionary of loaded results
    selected_algos: List of selected algorithms
    selected_conditions: List of selected conditions
    sort_by: Cell type sorting method:
      - 'n_samples': sort by increasing number of cells (rare populations at the top) [default]
      - 'alphabetical': alphabetical sort

  Returns:
    Matplotlib figure with confusion matrices
  """
  from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
  from matplotlib.patches import Rectangle
  from scrbenchmark.utils.metrics import compute_rare_class_accuracy

  # Collect predictions and true labels for each algorithm
  algo_data = {}
  algo_n_runs: Dict[str, int] = {}  # number of test runs per algorithm

  # Collect train counts using shared helper (averaged per run).
  train_counts = _collect_train_counts(all_data, selected_conditions)

  # Collect label_map from any condition that has one (for decoding numeric labels)
  global_label_map = None
  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue
    lmap = data.get('label_map')
    if lmap:
      global_label_map = lmap
      break

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    # Try to load labels from files first
    labels_dict = data.get('labels', {})

    if labels_dict:
      # Use labels from files
      for algo, runs in labels_dict.items():
        if algo not in selected_algos:
          continue

        for run_key, labels_df in runs.items():
          # Skip train and val sets
          if '_train' in run_key or '_val' in run_key:
            continue

          if 'true_label' not in labels_df.columns or 'predicted_label' not in labels_df.columns:
            continue

          labels_true = labels_df['true_label'].values
          labels_pred = labels_df['predicted_label'].values

          if algo not in algo_data:
            algo_data[algo] = {'true': [], 'pred': []}

          algo_data[algo]['true'].extend(labels_true.tolist())
          algo_data[algo]['pred'].extend(labels_pred.tolist())
          algo_n_runs[algo] = algo_n_runs.get(algo, 0) + 1
    else:
      # Fallback: try to get labels from dataframe columns
      df = data['df']
      algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

      for _, row in df.iterrows():
        algo = row.get(algo_col)
        if algo not in selected_algos:
          continue

        # Extract true and predicted labels
        labels_true = row.get('true_labels')
        labels_pred = row.get('predicted_labels')

        if labels_true is None or labels_pred is None:
          continue

        # Convert to numpy if necessary
        if not isinstance(labels_true, np.ndarray):
          if isinstance(labels_true, str):
            try:
              labels_true = ast.literal_eval(labels_true)
            except:
              continue
          labels_true = np.array(labels_true)

        if not isinstance(labels_pred, np.ndarray):
          if isinstance(labels_pred, str):
            try:
              labels_pred = ast.literal_eval(labels_pred)
            except:
              continue
          labels_pred = np.array(labels_pred)

        if algo not in algo_data:
          algo_data[algo] = {'true': [], 'pred': []}

        algo_data[algo]['true'].extend(labels_true.tolist())
        algo_data[algo]['pred'].extend(labels_pred.tolist())
        algo_n_runs[algo] = algo_n_runs.get(algo, 0) + 1

  if not algo_data:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No label data available\n\n'
             '(Requires label files with true_label and predicted_label\n'
             'in results/labels/ or true_labels/predicted_labels in CSV files)',
        ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  # Determine grid
  n_algos = len(algo_data)
  n_cols = min(3, n_algos)
  n_rows = (n_algos + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
  axes = axes.flatten()

  for idx, (algo, data) in enumerate(sorted(algo_data.items())):
    ax = axes[idx]

    y_true = np.array(data['true'], dtype=str)
    y_pred = np.array(data['pred'], dtype=str)

    # Decode numeric true labels to original names if label_map is available
    if global_label_map is not None:
      # Check if true labels are numeric (encoded)
      try:
        [int(t) for t in set(y_true)]
        # All true labels are numeric - decode them
        y_true = np.array([global_label_map.get(t, t) for t in y_true])
      except ValueError:
        pass # Already string labels, no decoding needed

    # Check if true and predicted labels are in different spaces
    # (e.g., true = cell types, pred = cluster IDs)
    true_labels_set = set(y_true)
    pred_labels_set = set(y_pred)

    if true_labels_set != pred_labels_set:
      # Map predicted cluster IDs to true labels via Hungarian algorithm
      from scipy.optimize import linear_sum_assignment

      true_unique = sorted(true_labels_set)
      pred_unique = sorted(pred_labels_set)
      true_to_idx = {l: i for i, l in enumerate(true_unique)}
      pred_to_idx = {l: i for i, l in enumerate(pred_unique)}

      # Build cost matrix (n_pred x n_true): count co-occurrences
      cost = np.zeros((len(pred_unique), len(true_unique)), dtype=int)
      for t, p in zip(y_true, y_pred):
        cost[pred_to_idx[p], true_to_idx[t]] += 1

      # Hungarian assignment maximizes matches
      row_ind, col_ind = linear_sum_assignment(-cost)

      # Build mapping: predicted cluster -> true label
      pred_to_true = {}
      for r, c in zip(row_ind, col_ind):
        pred_to_true[pred_unique[r]] = true_unique[c]

      # Unmapped predicted clusters: assign to the most frequent true label in that cluster
      # (instead of creating "cluster_X" names that pollute the confusion matrix)
      for p in pred_unique:
        if p not in pred_to_true:
          # Find the most frequent true label for cells in this predicted cluster
          p_idx = pred_to_idx[p]
          if cost[p_idx].sum() > 0:
            best_true_idx = np.argmax(cost[p_idx])
            pred_to_true[p] = true_unique[best_true_idx]
          else:
            # No cells in this cluster (shouldn't happen), fallback to first true label
            pred_to_true[p] = true_unique[0]

      # Remap predictions
      y_pred = np.array([pred_to_true[p] for p in y_pred])

    # Get unique labels (now both in the same space)
    all_labels = set(y_true) | set(y_pred)

    # Sort labels according to sort_by parameter
    if sort_by == 'n_samples':
      # Sort by increasing number of cells (rare populations at the top)
      labels = sorted(all_labels, key=lambda l: train_counts.get(str(l), float('inf')))
    else:
      # Alphabetical sort by default
      labels = sorted(all_labels)

    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Calculate F1, precision, recall per class
    precision, recall, f1, support = precision_recall_fscore_support(
      y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Normalize matrix for visualization (percentage)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized) # Replace NaN by 0

    # Number of test runs for this algorithm (to average aggregated counts)
    n_runs = algo_n_runs.get(algo, 1)

    # Create heatmap
    im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Annotate with average values per run
    cm_avg = np.round(cm / n_runs).astype(int)
    for i in range(len(labels)):
      for j in range(len(labels)):
        # Text color depending on background
        text_color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
        ax.text(j, i, f'{cm_avg[i, j]}',
            ha='center', va='center', color=text_color, fontsize=9)

    # Highlight diagonal (correct predictions)
    for i in range(len(labels)):
      ax.add_patch(Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                  edgecolor='green', lw=2.5))

    # Configure axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Test counts from confusion matrix row sums, averaged per run
    test_row_counts = cm.sum(axis=1)

    # Build y-axis labels with average per-run train and test counts
    x_label_names = [str(l)[:15] for l in labels]
    y_label_names = []
    for i, l in enumerate(labels):
      name = str(l)[:15]
      n_test = round(int(test_row_counts[i]) / n_runs)
      n_train = train_counts.get(str(l))
      if n_train is not None:
        y_label_names.append(f"{name} (train:{n_train}, test:{n_test})")
      else:
        y_label_names.append(f"{name} (n={n_test})")

    ax.set_xticklabels(x_label_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(y_label_names, fontsize=7)

    ax.set_xlabel('Predicted type', fontweight='bold', fontsize=10)
    ax.set_ylabel('True type', fontweight='bold', fontsize=10)

    # Title with algorithm name
    algo_display = ALGO_DISPLAY_NAMES.get(algo, algo)
    ax.set_title(f'{algo_display}\nConfusion Matrix',
          fontweight='bold', fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', rotation=270, labelpad=15, fontsize=9)

    # Add summary table under the matrix
    # Calculate average metrics
    avg_f1 = np.mean(f1)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    # Summary text
    summary_text = (
      f"Global metrics:\n"
      f"Accuracy: {accuracy:.3f} | "
      f"Macro F1: {avg_f1:.3f}\n"
      f"Mean precision: {avg_precision:.3f} | "
      f"Balanced ACC: {avg_recall:.3f}"
    )
    
    # Calculate Rare-ACC using the raw vectors (they are already aligned/mapped above if needed)
    rare_acc = compute_rare_class_accuracy(y_true, y_pred, threshold=0.05)
    if rare_acc is not None:
        summary_text += f"\nRare-ACC (<5%): {rare_acc:.3f}"

    ax.text(0.5, -0.15, summary_text,
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

  # Hide unused subplots
  for idx in range(n_algos, len(axes.flat) if n_algos > 1 else 1):
    if n_algos > 1:
      axes.flat[idx].axis('off')

  fig.suptitle('Detailed Confusion Matrix by Algorithm',
        fontweight='bold', fontsize=15, y=0.98)

  plt.tight_layout()
  plt.subplots_adjust(bottom=0.08, top=0.95, hspace=0.4)

  return fig


def create_metrics_by_celltype_table(all_data: Dict[str, Dict], selected_algos: List[str],
                   selected_conditions: List[str]) -> pd.DataFrame:
  """Compatibility wrapper for the table helper module."""
  from .tables import create_metrics_by_celltype_table as _create_metrics_by_celltype_table

  return _create_metrics_by_celltype_table(all_data, selected_algos, selected_conditions)


def plot_train_vs_test_comparison(all_data: Dict[str, Dict], selected_algos: List[str],
                  selected_conditions: List[str]) -> plt.Figure:
  """
  Creates a scatter plot comparing train vs test performance to diagnose overfitting.

  Each point represents an algorithm/condition combination.
  Points on the diagonal indicate good generalization.
  """
  fig, axes = plt.subplots(1, 6, figsize=(36, 6))
  metrics = ['NMI', 'ARI', 'ACC', 'F1_Macro', 'BalancedACC', 'BalancedRareACC']

  for ax, metric in zip(axes, metrics):
    train_vals = []
    test_vals = []
    labels = []
    colors = []

    for condition, data in all_data.items():
      if condition not in selected_conditions:
        continue

      df = data['df']
      if data['type'] != 'benchmark_detailed':
        continue

      algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

      for algo in selected_algos:
        algo_df = df[df[algo_col] == algo]
        train_col = f'train_{metric}'
        test_col = f'test_{metric}'

        if train_col not in algo_df.columns or test_col not in algo_df.columns:
          continue

        train = pd.to_numeric(algo_df[train_col], errors='coerce').mean()
        test = pd.to_numeric(algo_df[test_col], errors='coerce').mean()

        if not np.isnan(train) and not np.isnan(test):
          train_vals.append(train)
          test_vals.append(test)
          labels.append(f"{ALGO_DISPLAY_NAMES.get(algo, algo)[:8]}\n({condition[:10]})")
          colors.append(ALGO_COLORS.get(algo, '#333333'))

    if not train_vals:
      ax.text(0.5, 0.5, 'No train/test data', ha='center', va='center', fontsize=12)
      ax.set_title(f'{metric}', fontweight='bold')
      continue

    # Scatter plot
    scatter = ax.scatter(train_vals, test_vals, s=120, alpha=0.7, c=colors, edgecolors='white', linewidth=1.5)

    # Identity line (no overfitting)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='No overfitting', linewidth=2)

    # Overfitting zone (above diagonal)
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='red', label='Overfitting zone')

    # Annotations
    for i, label in enumerate(labels):
      ax.annotate(label, (train_vals[i], test_vals[i]),
            fontsize=7, alpha=0.8, ha='center', va='bottom',
            xytext=(0, 5), textcoords='offset points')

    ax.set_xlabel(f'{metric} (Train)', fontweight='bold', fontsize=11)
    ax.set_ylabel(f'{metric} (Test)', fontweight='bold', fontsize=11)
    ax.set_title(f'{metric}: Train vs Test', fontweight='bold', fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

  fig.suptitle('Overfitting Diagnosis: Train vs Test Performance',
        fontweight='bold', fontsize=14, y=1.02)

  # Add explanatory caption
  add_figure_caption(fig, 'train_vs_test')

  plt.tight_layout()
  plt.subplots_adjust(bottom=0.15)
  return fig


def plot_generalization_gap_combined(all_data: Dict[str, Dict], selected_algos: List[str],
                   selected_conditions: List[str],
                   show_cld: bool = False) -> plt.Figure:
  """
  Creates a combined generalization gap plot with 6 subplots.

  Inspired by analyze_2000HVG_results_light.py - plot_generalization_gap().
  Displays a boxplot per metric with algorithms sorted by increasing gap.
  """
  # Collect gap data for all metrics
  metrics = ['NMI', 'ARI', 'ACC', 'F1_Macro', 'BalancedACC', 'BalancedRareACC']
  gap_data = []

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    result_type = data['type']

    # Gap requires separate train/test data
    if result_type != 'benchmark_detailed':
      continue

    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      for metric in metrics:
        train_col = f'train_{metric}'
        test_col = f'test_{metric}'

        train_val = pd.to_numeric(row.get(train_col), errors='coerce')
        test_val = pd.to_numeric(row.get(test_col), errors='coerce')

        if not np.isnan(train_val) and not np.isnan(test_val):
          gap_data.append({
            'algorithm': algo,
            'condition': condition,
            'metric': metric,
            'train': train_val,
            'test': test_val,
            'gap': train_val - test_val
          })

  if not gap_data:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No train/test data disponibles\n\n'
             '(Requires benchmark_detailed results\n'
             'with train/test split)',
        ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  gap_df = pd.DataFrame(gap_data)

  # Create 5 subplots
  fig, axes = plt.subplots(1, 5, figsize=(30, 7))

  for ax, metric in zip(axes, metrics):
    metric_df = gap_df[gap_df['metric'] == metric]

    if metric_df.empty:
      ax.text(0.5, 0.5, f'No data {metric}', ha='center', va='center')
      ax.set_title(f'{metric} Gap', fontweight='bold')
      continue

    # Sort algorithms by increasing average gap
    algo_order = metric_df.groupby('algorithm')['gap'].mean().sort_values().index.tolist()

    # Create the boxplot
    sns.boxplot(data=metric_df, x='algorithm', y='gap', order=algo_order,
          ax=ax, palette='Set3', width=0.6)

    # Add individual points to see the distribution
    sns.stripplot(data=metric_df, x='algorithm', y='gap', order=algo_order,
           ax=ax, color='black', alpha=0.4, size=4, jitter=True)

    # Reference lines
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Gap = 0 (ideal)')
    ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Gap = 5%')
    ax.axhline(y=0.10, color='red', linestyle=':', alpha=0.7, linewidth=1.5, label='Gap = 10%')

    # Color zoning for overfitting
    ymin, ymax = ax.get_ylim()
    ax.fill_between([-0.5, len(algo_order)-0.5], 0, ymax, alpha=0.05, color='red')
    ax.fill_between([-0.5, len(algo_order)-0.5], ymin, 0, alpha=0.05, color='green')

    ax.set_xlabel('', fontweight='bold', fontsize=11)
    ax.set_ylabel(f'{metric} Gap (Train - Test)', fontweight='bold', fontsize=11)
    ax.set_title(f'{metric} Generalization Gap', fontweight='bold', fontsize=13)

    # CLD significance letters per subplot
    if show_cld and HAS_CLD:
      gap_values_by_algo = {}
      for algo in algo_order:
        vals = metric_df[metric_df['algorithm'] == algo]['gap'].tolist()
        if vals:
          gap_values_by_algo[algo] = vals
      if len(gap_values_by_algo) >= 2:
        cld, global_p, _ = compute_significance_groups(gap_values_by_algo)
        if global_p < 0.05:
          _annotate_cld(ax, cld, algo_order, y_offset_ratio=0.02)

    # Algorithm labels
    ax.set_xticklabels([ALGO_DISPLAY_NAMES.get(t.get_text(), t.get_text())
             for t in ax.get_xticklabels()], rotation=45, ha='right', fontsize=9)

    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend only on the last subplot
    if metric == 'ACC':
      ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

  fig.suptitle('Generalization Gap Analysis by Metric',
        fontweight='bold', fontsize=15, y=1.02)

  # Explanatory caption
  caption_text = (
    "Method: boxplot of gap = (train performance) - (test performance) for each algorithm and metric.\n"
    "Interpretation: gap > 0 indicates overfitting. Gap close to 0 indicates stronger generalization. "
    "Gap < 0 means test > train (rare, sometimes observed with regularization).\n"
    "Algorithms are sorted by increasing mean gap. Black dots are individual run values."
  )
  fig.text(0.5, -0.02, caption_text, ha='center', va='top', fontsize=9,
       style='italic', wrap=True,
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8),
       transform=fig.transFigure)

  plt.tight_layout()
  plt.subplots_adjust(bottom=0.18, top=0.92)
  return fig


def export_publication_ready(agg_df: pd.DataFrame, selected_algos: List[str],
               selected_conditions: List[str],
               format: str = 'latex') -> str:
  """Compatibility wrapper for the table helper module."""
  from .tables import export_publication_ready as _export_publication_ready

  return _export_publication_ready(agg_df, selected_algos, selected_conditions, format)



def plot_batch_composition(all_data: Dict[str, Dict], selected_algos: List[str],
              selected_conditions: List[str]):
  """Create batch composition visualization with cell type × batch heatmaps.

  For each selected condition, shows:
  1. Stacked bar chart of overall batch proportions per split (Train/Test)
  2. Heatmaps of cell type × batch counts for Train and Test side by side
  3. Summary table with exact cell counts

  Uses labels from the first available algorithm (split is identical across algos).

  Returns:
    matplotlib Figure, or 'no_batch_column' if batch info is missing.
  """
  from collections import Counter
  import matplotlib.gridspec as gridspec

  # Collect data: condition -> split -> {batch_total, celltype_batch, n_runs}
  condition_data: Dict[str, Dict[str, dict]] = {}
  has_batch_column = False
  all_batches: set = set()
  all_celltypes: set = set()

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    labels_dict = data.get('labels', {})
    if not labels_dict:
      continue
    label_map = data.get('label_map')

    # Pick first available algorithm (batch composition is identical for all algos)
    first_algo = None
    for algo in selected_algos:
      if algo in labels_dict:
        first_algo = algo
        break
    if first_algo is None:
      first_algo = next(iter(labels_dict))

    runs = labels_dict[first_algo]

    for run_key, labels_df in runs.items():
      if 'batch' not in labels_df.columns:
        continue

      has_batch_column = True

      # Determine split type from run_key
      if '_train' in run_key:
        split = 'Train'
      elif '_test' in run_key:
        split = 'Test'
      elif '_val' in run_key:
        split = 'Val'
      else:
        continue

      if condition not in condition_data:
        condition_data[condition] = {}
      if split not in condition_data[condition]:
        condition_data[condition][split] = {
          'batch_total': Counter(),
          'celltype_batch': defaultdict(Counter),
          'n_runs': 0,
        }

      sd = condition_data[condition][split]
      sd['n_runs'] += 1

      # Overall batch counts
      for batch, cnt in labels_df['batch'].astype(str).value_counts().items():
        sd['batch_total'][batch] += cnt
        all_batches.add(batch)

      # Cell type × batch counts (using groupby for performance)
      if 'true_label' in labels_df.columns:
        ct_series = labels_df['true_label'].astype(str)
        batch_series = labels_df['batch'].astype(str)
        for (ct_raw, batch), group in labels_df.groupby([ct_series, batch_series]):
          ct = label_map.get(ct_raw, ct_raw) if label_map else ct_raw
          sd['celltype_batch'][ct][batch] += len(group)
          all_celltypes.add(ct)

  if not has_batch_column:
    return 'no_batch_column'

  if not condition_data:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No label data available',
        ha='center', va='center', fontsize=14)
    ax.axis('off')
    return fig

  all_batches = sorted(all_batches)
  all_celltypes = sorted(all_celltypes)
  conditions_sorted = sorted(condition_data.keys())
  splits_order = ['Train', 'Test', 'Val']

  # Build bar chart data and heatmap DataFrames (averaged across runs)
  bar_labels = []
  bar_proportions = []
  bar_absolutes = []
  heatmap_data: Dict[str, Dict[str, pd.DataFrame]] = {}

  for condition in conditions_sorted:
    heatmap_data[condition] = {}
    for split in splits_order:
      if split not in condition_data[condition]:
        continue
      sd = condition_data[condition][split]
      n = sd['n_runs']
      if n == 0:
        continue

      # Bar chart: average batch counts
      avg_batch = {b: sd['batch_total'].get(b, 0) / n for b in all_batches}
      total = sum(avg_batch.values())
      props = {b: (v / total * 100 if total > 0 else 0) for b, v in avg_batch.items()}
      absvals = {b: round(v) for b, v in avg_batch.items()}

      short_cond = condition.replace('split_', '').replace('standard_', '')
      bar_labels.append(f"{short_cond} ({split})")
      bar_proportions.append(props)
      bar_absolutes.append(absvals)

      # Heatmap: cell type × batch averaged counts
      hm = pd.DataFrame(0.0, index=all_celltypes, columns=all_batches)
      for ct in all_celltypes:
        for batch in all_batches:
          hm.loc[ct, batch] = sd['celltype_batch'].get(ct, Counter()).get(batch, 0) / n
      heatmap_data[condition][split] = hm

  if not bar_labels:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No train/test data disponibles',
        ha='center', va='center', fontsize=14)
    ax.axis('off')
    return fig

  n_bars = len(bar_labels)
  n_celltypes = len(all_celltypes)
  n_batches = len(all_batches)

  # Conditions that have at least Train or Test for heatmaps
  hm_conditions = [c for c in conditions_sorted
           if 'Train' in heatmap_data.get(c, {}) or 'Test' in heatmap_data.get(c, {})]
  n_hm_rows = len(hm_conditions)

  # --- Figure layout with GridSpec ---
  # Heights: bar chart + heatmap rows + table
  bar_h = max(4, n_bars * 0.4 + 2)
  hm_h = max(3, n_celltypes * 0.35 + 1.5)
  table_h = max(2.5, n_bars * 0.3 + 1)

  total_grid_rows = 1 + n_hm_rows + 1  # bar + heatmaps + table
  heights = [bar_h] + [hm_h] * n_hm_rows + [table_h]

  fig_width = max(12, n_batches * 1.5 + 4)
  fig_height = sum(heights) + 2

  fig = plt.figure(figsize=(fig_width, fig_height))
  gs = gridspec.GridSpec(total_grid_rows, 2, figure=fig,
              height_ratios=heights, hspace=0.5, wspace=0.35)

  # --- 1. Bar chart (top, spans both columns) ---
  ax_bar = fig.add_subplot(gs[0, :])
  y_pos = np.arange(n_bars)
  lefts = np.zeros(n_bars)

  cmap_bar = plt.colormaps.get_cmap('tab20')
  batch_colors = {b: cmap_bar(i / max(n_batches, 1)) for i, b in enumerate(all_batches)}

  for batch in all_batches:
    widths = [bar_proportions[i].get(batch, 0) for i in range(n_bars)]
    bars = ax_bar.barh(y_pos, widths, left=lefts, color=batch_colors[batch],
              label=batch, edgecolor='white', linewidth=0.5)
    for i, (bar, w) in enumerate(zip(bars, widths)):
      if w > 5:
        count = bar_absolutes[i].get(batch, 0)
        if count > 0:
          ax_bar.text(lefts[i] + w / 2, i, str(count),
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white' if w > 15 else 'black')
    lefts += widths

  ax_bar.set_yticks(y_pos)
  ax_bar.set_yticklabels(bar_labels, fontsize=8)
  ax_bar.set_xlabel('Proportion (%)', fontsize=10)
  ax_bar.set_xlim(0, 100)
  ax_bar.set_title('Composition Batch : Train vs Test (vue globale)', fontweight='bold', fontsize=13)
  ax_bar.legend(title='Batch', bbox_to_anchor=(1.01, 1), loc='upper left',
         fontsize=8, title_fontsize=9)
  ax_bar.invert_yaxis()

  # --- 2. Heatmaps: cell type × batch per condition (Train | Test) ---
  for idx, condition in enumerate(hm_conditions):
    short_cond = condition.replace('split_', '').replace('standard_', '')

    for col_idx, split in enumerate(['Train', 'Test']):
      ax = fig.add_subplot(gs[1 + idx, col_idx])

      if split not in heatmap_data.get(condition, {}):
        ax.text(0.5, 0.5, f'No {split} split', ha='center', va='center',
            fontsize=10, style='italic', color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f'{short_cond} — {split}', fontweight='bold', fontsize=10)
        continue

      hm_df = heatmap_data[condition][split]
      annot_df = hm_df.round(0).astype(int)

      # Determine vmax from the condition's data (shared between Train/Test)
      vmax_cond = 1
      for s in ['Train', 'Test']:
        if s in heatmap_data.get(condition, {}):
          vmax_cond = max(vmax_cond, heatmap_data[condition][s].values.max())

      sns.heatmap(hm_df, annot=annot_df, fmt='d', cmap='YlOrRd', ax=ax,
            vmin=0, vmax=vmax_cond,
            cbar_kws={'shrink': 0.7, 'label': 'n cells (mean/run)'},
            linewidths=0.5, linecolor='white')

      ax.set_title(f'{short_cond} — {split}', fontweight='bold', fontsize=10)
      ax.set_xlabel('Batch', fontsize=9)
      ax.set_ylabel('Cell Type' if col_idx == 0 else '', fontsize=9)
      ax.tick_params(axis='both', labelsize=7)

  # --- 3. Summary table (bottom, spans both columns) ---
  ax_table = fig.add_subplot(gs[-1, :])
  ax_table.axis('off')

  table_data = []
  col_headers = ['Condition', 'Split'] + all_batches + ['Total']

  for i, label in enumerate(bar_labels):
    parts = label.rsplit(' (', 1)
    cond_name = parts[0]
    split_name = parts[1].rstrip(')') if len(parts) > 1 else ''
    row = [cond_name, split_name]
    row_total = 0
    for b in all_batches:
      val = bar_absolutes[i].get(b, 0)
      row.append(str(val))
      row_total += val
    row.append(str(row_total))
    table_data.append(row)

  table = ax_table.table(
    cellText=table_data,
    colLabels=col_headers,
    loc='center',
    cellLoc='center',
  )
  table.auto_set_font_size(False)
  table.set_fontsize(8)
  table.scale(1, 1.3)

  # Style header row
  for j in range(len(col_headers)):
    cell = table[0, j]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(color='white', fontweight='bold')

  # Alternate row colors
  for i in range(len(table_data)):
    bg = '#F2F2F2' if i % 2 == 0 else 'white'
    for j in range(len(col_headers)):
      table[i + 1, j].set_facecolor(bg)

  ax_table.set_title('Mean counts per run (n cells)', fontweight='bold', fontsize=11, pad=10)

  fig.suptitle('Batch Composition of Train/Test Splits', fontweight='bold', fontsize=15, y=1.01)
  add_figure_caption(fig, 'batch_composition')
  return fig


def plot_batch_generalization_fig(all_data: Dict[str, Dict], selected_algos: List[str],
                  selected_conditions: List[str]) -> plt.Figure:
  """Create matrix of generalization between batches."""
  group_performance = []

  # Filter for batch conditions
  batch_conditions = {k: v for k, v in all_data.items() if 'batch' in k and k in selected_conditions}

  if not batch_conditions:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No "batch" condition selected\n(ex: split_batch_Human1)', 
        ha='center', va='center', fontsize=14)
    return fig

  for condition, data in batch_conditions.items():
    df = data['df']
    train_batch = condition.replace('split_batch_', '').replace('batch_', '')
    # Clean up name if needed
    if '_reinject' in train_batch:
      train_batch = train_batch.replace('_reinject', ' (R)')
    
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      tbg_str = str(row.get('test_by_group', ''))
      
      # Safe parse dictionary
      try:
        # Handle potential numpy string representation
        cleaned = tbg_str.replace('np.float64(', '').replace(')', '')
        # Handle nan
        if pd.isna(tbg_str) or tbg_str == 'nan' or tbg_str == '':
          continue
          
        tbg_data = ast.literal_eval(cleaned) if '{' in cleaned else None

        if not isinstance(tbg_data, dict):
          continue

        for test_batch, metrics in tbg_data.items():
          # Check if metrics is dict (standard) or something else
          if isinstance(metrics, dict):
            acc = metrics.get('ACC', np.nan)
          else:
            continue
            
          group_performance.append({
            'algorithm': algo,
            'train_batch': train_batch,
            'test_batch': test_batch,
            'ACC': acc
          })
      except Exception:
        continue

  if not group_performance:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No group-level generalization data found', ha='center', va='center', fontsize=14)
    return fig

  perf_df = pd.DataFrame(group_performance)
  
  # Filter algorithms present in data
  algorithms = sorted(list(perf_df['algorithm'].unique()))
  
  # Determine grid size
  n_algos = len(algorithms)
  n_cols = min(4, n_algos)
  n_rows = (n_algos + n_cols - 1) // n_cols
  
  # Adjust figure size
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5 * n_rows))
  
  # Enofe axes is always iterable array
  if n_algos == 1:
    axes = np.array([axes])
  axes = axes.flatten()

  for idx, algo in enumerate(algorithms):
    ax = axes[idx]
    algo_df = perf_df[perf_df['algorithm'] == algo]

    # Pivot to matrix form
    pivot = algo_df.pivot_table(
      values='ACC',
      index='train_batch',
      columns='test_batch',
      aggfunc='mean'
    )
    
    if pivot.empty:
      ax.text(0.5, 0.5, 'No data', ha='center', va='center')
      ax.set_title(ALGO_DISPLAY_NAMES.get(algo, algo))
      continue

    # Heatmap
    hm = sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
             ax=ax, vmin=0.4, vmax=1.0, cbar_kws={'shrink': 0.8},
             annot_kws={'size': 9})
    # Explicit color code on the colorbar for quick biological interpretation.
    try:
      cbar = hm.collections[0].colorbar
      if cbar is not None:
        cbar.set_label('ACC (Red=low, Yellow=medium, Green=high)', fontsize=9)
    except Exception:
      pass

    ax.set_title(ALGO_DISPLAY_NAMES.get(algo, algo), fontweight='bold', fontsize=12)
    ax.set_xlabel('Test Batch', fontsize=10)
    ax.set_ylabel('Train Batch', fontsize=10)
    
    # Highlight diagonal if indices match columns
    common = list(set(pivot.index) & set(pivot.columns))
    for c in common:
      try:
        i = pivot.index.get_loc(c)
        j = pivot.columns.get_loc(c)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=2))
      except:
        pass

  # Hide unused subplots
  for idx in range(len(algorithms), len(axes)):
    axes[idx].axis('off')

  fig.suptitle('Inter-Batch Generalization (Accuracy)', fontweight='bold', fontsize=16, y=1.01)
  add_figure_caption(fig, 'batch_generalization')
  plt.tight_layout()
  plt.subplots_adjust(bottom=0.16)

  return fig


def plot_test_metrics_by_batch(all_data: Dict[str, Dict], selected_algos: List[str],
                selected_conditions: List[str]) -> plt.Figure:
  """
  Create heatmaps showing test metrics (NMI, ARI, ACC) per batch for each algorithm.

  This visualization helps understand how well each algorithm performs on different
  test batches (e.g., human1, human2, human3, human4) within a split experiment.
  """
  # Collect per-batch metrics from test_by_group
  batch_metrics = []

  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

    for _, row in df.iterrows():
      algo = row.get(algo_col)
      if algo not in selected_algos:
        continue

      # Parse test_by_group
      tbg_str = str(row.get('test_by_group', ''))
      if not tbg_str or tbg_str == 'nan':
        continue

      try:
        # Clean numpy wrappers
        cleaned = tbg_str.replace('np.float64(', '').replace(')', '')
        tbg_data = ast.literal_eval(cleaned)

        if not isinstance(tbg_data, dict):
          continue

        for batch_name, metrics in tbg_data.items():
          if isinstance(metrics, dict):
            batch_metrics.append({
              'condition': condition,
              'algorithm': algo,
              'batch': batch_name,
              'NMI': metrics.get('NMI', np.nan),
              'ARI': metrics.get('ARI', np.nan),
              'ACC': metrics.get('ACC', np.nan),
              'n_samples': metrics.get('n_samples', 0)
            })
      except Exception:
        continue

  if not batch_metrics:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No test_by_group data available\n\n'
             '(Requires --stratify-by batch in benchmark mode)',
        ha='center', va='center', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  # Create DataFrame
  metrics_df = pd.DataFrame(batch_metrics)

  # Aggregate by algorithm and batch (mean across runs and conditions)
  agg_df = metrics_df.groupby(['algorithm', 'batch']).agg({
    'NMI': 'mean',
    'ARI': 'mean',
    'ACC': 'mean',
    'n_samples': 'first'
  }).reset_index()

  # Get unique algorithms and batches
  algorithms = [a for a in selected_algos if a in agg_df['algorithm'].unique()]
  batches = sorted(agg_df['batch'].unique())

  if not algorithms:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No algorithm with per-batch data',
        ha='center', va='center', fontsize=12)
    ax.axis('off')
    return fig

  # Create figure with 3 subplots (one per metric)
  fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(algorithms) * 0.8)))

  metrics_to_plot = ['NMI', 'ARI', 'ACC']
  cmaps = ['Blues', 'Greens', 'Oranges']

  for idx, (metric, cmap) in enumerate(zip(metrics_to_plot, cmaps)):
    ax = axes[idx]

    # Create pivot table
    pivot = agg_df.pivot_table(
      values=metric,
      index='algorithm',
      columns='batch',
      aggfunc='mean'
    )

    # Reorder algorithms to match selection order
    pivot = pivot.reindex([a for a in algorithms if a in pivot.index])

    if pivot.empty:
      ax.text(0.5, 0.5, 'No data', ha='center', va='center')
      ax.set_title(metric)
      continue

    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
          ax=ax, vmin=0, vmax=1, cbar_kws={'shrink': 0.8},
          annot_kws={'size': 10, 'weight': 'bold'},
          linewidths=0.5, linecolor='white')

    # Format y-axis labels with display names
    yticklabels = [ALGO_DISPLAY_NAMES.get(a, a) for a in pivot.index]
    ax.set_yticklabels(yticklabels, rotation=0, fontsize=10)
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right', fontsize=10)

    ax.set_title(f'{metric} by Test Batch', fontweight='bold', fontsize=12)
    ax.set_xlabel('Test Batch', fontsize=10)
    ax.set_ylabel('Algorithm' if idx == 0 else '', fontsize=10)

  # Add sample sizes as annotation
  sample_info = agg_df.groupby('batch')['n_samples'].first().to_dict()
  sample_text = ' | '.join([f"{b}: n={int(n)}" for b, n in sorted(sample_info.items())])
  fig.text(0.5, 0.02, f'Test batch sizes: {sample_text}',
      ha='center', fontsize=9, style='italic')

  conditions_text = ', '.join(selected_conditions)
  fig.suptitle(f'Performance by Test Batch\n({conditions_text})',
        fontweight='bold', fontsize=14, y=1.02)

  plt.tight_layout()
  plt.subplots_adjust(bottom=0.12, top=0.88)

  return fig


def plot_saved_umap_gallery(
    all_data: Dict[str, Dict],
    selected_algos: List[str],
    selected_conditions: List[str],
    split_filter: str = 'all',
    max_images: int = 18
) -> plt.Figure:
  """Display pre-rendered UMAP PNGs from each loaded result directory."""

  split_order = {'train': 0, 'test': 1, 'val': 2, 'full': 3}
  cond_order = {c: i for i, c in enumerate(selected_conditions)}
  algo_order = {a: i for i, a in enumerate(selected_algos)}

  entries: List[Dict[str, str]] = []

  for condition in selected_conditions:
    data = all_data.get(condition)
    if not data:
      continue

    load_path = data.get('path', '')
    figure_dirs: List[str] = []

    # Most benchmark outputs store plots in sibling folder: ../figures
    if load_path:
      sibling_figures = os.path.join(os.path.dirname(load_path), 'figures')
      if os.path.isdir(sibling_figures):
        figure_dirs.append(sibling_figures)
      direct_figures = os.path.join(load_path, 'figures')
      if os.path.isdir(direct_figures):
        figure_dirs.append(direct_figures)

    # De-duplicate while preserving order
    seen_dirs = set()
    unique_figure_dirs = []
    for d in figure_dirs:
      if d not in seen_dirs:
        unique_figure_dirs.append(d)
        seen_dirs.add(d)

    for figures_dir in unique_figure_dirs:
      for algo in selected_algos:
        # Standard mode naming
        if split_filter in ('all', 'full'):
          p_full = os.path.join(figures_dir, f'umap_comparison_{algo}.png')
          if os.path.isfile(p_full):
            entries.append({
              'condition': condition,
              'algorithm': algo,
              'split': 'full',
              'path': p_full,
            })

        # Benchmark split naming
        for split in ('train', 'test', 'val'):
          if split_filter not in ('all', split):
            continue
          p_split = os.path.join(figures_dir, f'umap_{algo}_{split}.png')
          if os.path.isfile(p_split):
            entries.append({
              'condition': condition,
              'algorithm': algo,
              'split': split,
              'path': p_split,
            })

  # De-duplicate by path (same file can be discovered twice with load_path variants)
  uniq_entries: List[Dict[str, str]] = []
  seen_paths = set()
  for e in entries:
    p = e['path']
    if p in seen_paths:
      continue
    uniq_entries.append(e)
    seen_paths.add(p)

  entries = sorted(
    uniq_entries,
    key=lambda e: (
      cond_order.get(e['condition'], 10_000),
      algo_order.get(e['algorithm'], 10_000),
      split_order.get(e['split'], 99),
    )
  )

  total_found = len(entries)
  if total_found == 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
      0.5, 0.5,
      "No saved UMAP found\n\n"
      "Expected: figures/umap_{algo}_{split}.png or figures/umap_comparison_{algo}.png",
      ha='center', va='center', fontsize=12
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  if max_images > 0 and total_found > max_images:
    entries = entries[:max_images]

  n = len(entries)
  n_cols = min(3, n)
  n_rows = (n + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.2 * n_rows), squeeze=False)
  flat_axes = axes.flatten()

  for ax, e in zip(flat_axes, entries):
    try:
      img = plt.imread(e['path'])
      ax.imshow(img)
      ax.axis('off')
      split_lbl = e['split']
      ax.set_title(
        f"{e['condition']}\n{ALGO_DISPLAY_NAMES.get(e['algorithm'], e['algorithm'])} [{split_lbl}]",
        fontsize=10
      )
    except Exception as err:
      ax.text(0.5, 0.5, f"Image read error\n{err}", ha='center', va='center', fontsize=9)
      ax.axis('off')

  for ax in flat_axes[n:]:
    ax.axis('off')

  if total_found > n:
    fig.suptitle(
      f"Saved UMAP Gallery ({n}/{total_found} shown)",
      fontsize=14, fontweight='bold', y=1.01
    )
  else:
    fig.suptitle(
      f"Saved UMAP Gallery ({total_found})",
      fontsize=14, fontweight='bold', y=1.01
    )

  plt.tight_layout()
  return fig


def plot_saved_umap_evolution_gallery(
    all_data: Dict[str, Dict],
    selected_algos: List[str],
    selected_conditions: List[str],
    max_images: int = 18
) -> plt.Figure:
  """Display pre-rendered UMAP evolution PNGs from loaded result directories."""

  cond_order = {c: i for i, c in enumerate(selected_conditions)}
  algo_order = {a: i for i, a in enumerate(selected_algos)}

  entries: List[Dict[str, Any]] = []

  for condition in selected_conditions:
    data = all_data.get(condition)
    if not data:
      continue

    load_path = data.get('path', '')
    figure_dirs: List[str] = []

    if load_path:
      sibling_figures = os.path.join(os.path.dirname(load_path), 'figures')
      if os.path.isdir(sibling_figures):
        figure_dirs.append(sibling_figures)
      direct_figures = os.path.join(load_path, 'figures')
      if os.path.isdir(direct_figures):
        figure_dirs.append(direct_figures)

    seen_dirs = set()
    unique_figure_dirs = []
    for d in figure_dirs:
      if d not in seen_dirs:
        unique_figure_dirs.append(d)
        seen_dirs.add(d)

    for figures_dir in unique_figure_dirs:
      for algo in selected_algos:
        pattern = os.path.join(figures_dir, f'umap_evolution_{algo}_run*.png')
        for png_path in sorted(glob.glob(pattern)):
          run_match = re.search(r'_run(\d+)\.png$', os.path.basename(png_path))
          run_id = int(run_match.group(1)) if run_match else 0
          entries.append({
            'condition': condition,
            'algorithm': algo,
            'run_id': run_id,
            'path': png_path,
          })

  uniq_entries: List[Dict[str, Any]] = []
  seen_paths = set()
  for e in entries:
    p = e['path']
    if p in seen_paths:
      continue
    uniq_entries.append(e)
    seen_paths.add(p)

  entries = sorted(
    uniq_entries,
    key=lambda e: (
      cond_order.get(e['condition'], 10_000),
      algo_order.get(e['algorithm'], 10_000),
      int(e.get('run_id', 0)),
    )
  )

  total_found = len(entries)
  if total_found == 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
      0.5, 0.5,
      "No saved UMAP evolution found\n\n"
      "Expected: figures/umap_evolution_{algo}_runX.png",
      ha='center', va='center', fontsize=12
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig

  if max_images > 0 and total_found > max_images:
    entries = entries[:max_images]

  n = len(entries)
  n_cols = min(2, n)
  n_rows = (n + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 5.6 * n_rows), squeeze=False)
  flat_axes = axes.flatten()

  for ax, e in zip(flat_axes, entries):
    try:
      img = plt.imread(e['path'])
      ax.imshow(img)
      ax.axis('off')
      ax.set_title(
        f"{e['condition']}\n{ALGO_DISPLAY_NAMES.get(e['algorithm'], e['algorithm'])} [run {int(e['run_id']) + 1}]",
        fontsize=10
      )
    except Exception as err:
      ax.text(0.5, 0.5, f"Image read error\n{err}", ha='center', va='center', fontsize=9)
      ax.axis('off')

  for ax in flat_axes[n:]:
    ax.axis('off')

  if total_found > n:
    fig.suptitle(
      f"UMAP Evolution Gallery ({n}/{total_found} shown)",
      fontsize=14, fontweight='bold', y=1.01
    )
  else:
    fig.suptitle(
      f"UMAP Evolution Gallery ({total_found})",
      fontsize=14, fontweight='bold', y=1.01
    )

  plt.tight_layout()
  return fig


def _message_figure(message: str, subtitle: str = "") -> plt.Figure:
  """Create a simple text-only figure for graceful fallbacks."""
  fig, ax = plt.subplots(figsize=(10, 6))
  full_text = message if not subtitle else f"{message}\n\n{subtitle}"
  ax.text(0.5, 0.5, full_text, ha='center', va='center', fontsize=12)
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.axis('off')
  return fig


def list_umap_diagnostic_entries(
    all_data: Dict[str, Dict],
    selected_algos: List[str],
    selected_conditions: List[str]
) -> List[Dict[str, Any]]:
  """Discover (condition, algorithm, run, split) entries that have label files."""
  entries = []
  cond_order = {c: i for i, c in enumerate(selected_conditions)}
  algo_order = {a: i for i, a in enumerate(selected_algos)}
  split_order = {'train': 0, 'test': 1, 'val': 2, 'full': 3}

  for condition in selected_conditions:
    data = all_data.get(condition, {})
    labels_by_algo = data.get('labels', {}) or {}
    for algo in selected_algos:
      runs = labels_by_algo.get(algo, {})
      for run_key in runs.keys():
        m = re.match(r'^run(\d+)_(train|test|val|full)$', str(run_key))
        if not m:
          continue
        entries.append({
          'condition': condition,
          'algorithm': algo,
          'run_id': int(m.group(1)),
          'split': m.group(2),
          'run_key': run_key
        })

  return sorted(
    entries,
    key=lambda e: (
      cond_order.get(e['condition'], 10_000),
      algo_order.get(e['algorithm'], 10_000),
      e['run_id'],
      split_order.get(e['split'], 99),
    )
  )


def _find_h5ad_for_split(results_load_path: str, split: str) -> Optional[str]:
  """Locate the relevant h5ad file for a given run split."""
  run_root = os.path.dirname(results_load_path) if results_load_path else ""
  candidates = []

  if split in ('train', 'test', 'val'):
    candidates.extend([
      os.path.join(run_root, 'data', 'benchmark', f'{split}.h5ad'),
      os.path.join(run_root, 'data', f'{split}.h5ad'),
    ])
  else:  # full / standard mode
    candidates.extend([
      os.path.join(run_root, 'data', 'processed.h5ad'),
      os.path.join(run_root, 'data', 'input.h5ad'),
      os.path.join(run_root, 'data', 'full.h5ad'),
    ])

    # Fallback: some runs save a dataset-specific filename
    # (e.g. Pancreas_RawCount_DCTCorr_Processed.h5ad) instead of processed.h5ad.
    data_dir = os.path.join(run_root, 'data')
    if os.path.isdir(data_dir):
      try:
        dynamic_h5ad = sorted(
          os.path.join(data_dir, f)
          for f in os.listdir(data_dir)
          if f.lower().endswith('.h5ad')
        )
        candidates.extend(dynamic_h5ad)
      except Exception:
        pass

  for path in candidates:
    if os.path.isfile(path):
      return path
  return None


def _select_label_columns(df_labels: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
  """Infer true/predicted column names from label CSV."""
  true_candidates = ['true_label', 'true_labels', 'label_true', 'ground_truth', 'true']
  pred_candidates = ['predicted_label', 'predicted_labels', 'label_pred', 'predicted', 'cluster', 'labels']

  true_col = next((c for c in true_candidates if c in df_labels.columns), None)
  pred_col = next((c for c in pred_candidates if c in df_labels.columns), None)
  return true_col, pred_col


def plot_umap_diagnostic_from_results(
    all_data: Dict[str, Dict],
    condition: str,
    algorithm: str,
    run_id: int,
    split: str,
    outline_mode: str = 'convex_hull',
    max_points: int = 15000,
    random_state: int = 42,
    show_centroids: bool = True
) -> plt.Figure:
  """
  Build a 2x2 UMAP diagnostic from stored labels + corresponding h5ad.
  """
  if viz_utils is None or not hasattr(viz_utils, 'plot_umap_diagnostic'):
    return _message_figure(
      "Visualization module unavailable",
      "Unable to load utils.visualization.plot_umap_diagnostic"
    )

  cond_data = all_data.get(condition)
  if not cond_data:
    return _message_figure("Condition not found", condition)

  labels_by_algo = cond_data.get('labels', {}) or {}
  algo_runs = labels_by_algo.get(algorithm, {}) or {}
  run_key = f'run{run_id}_{split}'
  if run_key not in algo_runs:
    return _message_figure(
      "Labels not found for this run/split",
      "Expected: {algorithm} / {run_key}"
    )

  labels_df = algo_runs[run_key]
  if labels_df is None or labels_df.empty:
    return _message_figure("Empty labels file", f"{algorithm} / {run_key}")

  true_col, pred_col = _select_label_columns(labels_df)
  if true_col is None or pred_col is None:
    return _message_figure(
      "Unrecognized label columns",
      f"Columns found: {', '.join(labels_df.columns)}"
    )

  h5ad_path = _find_h5ad_for_split(cond_data.get('path', ''), split)
  if h5ad_path is None:
    return _message_figure(
      "h5ad file not found for this split",
      f"split={split}, condition={condition}"
    )

  # Convert label_map keys to int for visualization decoding
  decoded_label_map = None
  raw_label_map = cond_data.get('label_map')
  if isinstance(raw_label_map, dict):
    decoded_label_map = {}
    for k, v in raw_label_map.items():
      try:
        decoded_label_map[int(k)] = str(v)
      except Exception:
        continue
    if not decoded_label_map:
      decoded_label_map = None

  try:
    import anndata as ad
    from scipy import sparse
  except Exception as e:
    return _message_figure("Missing dependencies", str(e))

  adata = None
  try:
    adata = ad.read_h5ad(h5ad_path, backed='r')

    n_labels = len(labels_df)
    n_obs = int(adata.n_obs)
    n_common = min(n_labels, n_obs)
    if n_common <= 0:
      return _message_figure("No alignable cells", f"labels={n_labels}, h5ad={n_obs}")

    # Keep rows aligned between labels CSV and h5ad; subsample only after alignment.
    idx = np.arange(n_common, dtype=int)
    if max_points > 0 and n_common > max_points:
      rng = np.random.default_rng(random_state)
      idx = np.sort(rng.choice(n_common, size=max_points, replace=False))

    # Best available source for projection:
    # 1) X_umap (already 2D), 2) X_pca (compact), 3) X (fallback, truncated dims)
    embeddings = None
    if 'X_umap' in adata.obsm and adata.obsm['X_umap'].shape[1] >= 2:
      embeddings = np.asarray(adata.obsm['X_umap'][idx, :2])
    elif 'X_pca' in adata.obsm and adata.obsm['X_pca'].shape[1] >= 2:
      n_pca = min(50, int(adata.obsm['X_pca'].shape[1]))
      embeddings = np.asarray(adata.obsm['X_pca'][idx, :n_pca])
    else:
      X_slice = adata.X[idx]
      if sparse.issparse(X_slice):
        X_slice = X_slice.toarray()
      embeddings = np.asarray(X_slice)
      if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)
      if embeddings.shape[1] > 50:
        from sklearn.decomposition import PCA
        n_comps = min(50, embeddings.shape[0] - 1, embeddings.shape[1])
        pca = PCA(n_components=n_comps, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
      return _message_figure("Unable to retrieve embeddings", h5ad_path)
    if embeddings.shape[1] < 2:
      return _message_figure("Embeddings insuffisants", f"shape={embeddings.shape}")

    true_labels = labels_df[true_col].to_numpy()[:n_common][idx]
    predicted_labels = labels_df[pred_col].to_numpy()[:n_common][idx]

    batch_labels = None
    for batch_col in ['batch', 'Batch', 'donor', 'sample']:
      if batch_col in labels_df.columns:
        batch_labels = labels_df[batch_col].to_numpy()[:n_common][idx]
        break
    if batch_labels is None:
      for batch_col in ['batch', 'Batch', 'donor', 'sample']:
        if batch_col in adata.obs.columns:
          batch_values = np.asarray(adata.obs[batch_col].to_numpy()[:n_common])
          batch_labels = batch_values[idx]
          break

    return viz_utils.plot_umap_diagnostic(
      embeddings=embeddings,
      true_labels=true_labels,
      predicted_labels=predicted_labels,
      batch_labels=batch_labels,
      algorithm_name=f"{ALGO_DISPLAY_NAMES.get(algorithm, algorithm)} | {condition} | run{run_id} [{split}]",
      outline_mode=outline_mode,
      show_cluster_centroids=show_centroids,
      max_points=max_points,
      random_state=random_state,
      label_names=decoded_label_map,
    )

  except Exception as e:
    return _message_figure("Error while rebuilding UMAP", str(e))
  finally:
    if adata is not None and getattr(adata, 'file', None) is not None:
      try:
        adata.file.close()
      except Exception:
        pass


# =============================================================================
# Main Render Function
# =============================================================================

def is_result_directory(path: str) -> bool:
  """Check if a directory contains valid result files."""
  # Direct check
  if (os.path.exists(os.path.join(path, 'benchmark_detailed.csv')) or
    os.path.exists(os.path.join(path, 'results.csv')) or
    os.path.exists(os.path.join(path, 'analysis_results.csv'))):
    return True
    
  # Subdirectory check
  sub_path = os.path.join(path, 'results')
  if os.path.isdir(sub_path):
    if (os.path.exists(os.path.join(sub_path, 'benchmark_detailed.csv')) or
      os.path.exists(os.path.join(sub_path, 'results.csv')) or
      os.path.exists(os.path.join(sub_path, 'analysis_results.csv'))):
      return True
      
  return False


@st.cache_data(show_spinner=False)
def scan_for_results(base_path: str, max_depth: int = 5) -> List[str]:
  """Recursively scan for result directories.

  When a directory is detected as a result directory (e.g. via its
  ``results/`` sub-folder), its children are pruned from the walk so
  that the same data is not reported twice.
  """
  found_dirs = []
  base_path = Path(base_path)
  if not base_path.exists():
    return []

  # Walk through directory
  for root, dirs, files in os.walk(base_path):
    try:
      current_depth = len(Path(root).relative_to(base_path).parts)
    except ValueError:
      current_depth = 0

    # Check if this is a result dir
    if is_result_directory(root):
      found_dirs.append(root)
      # Prune children to avoid duplicate detection (e.g. the
      # ``results/`` sub-folder that was already checked above).
      dirs.clear()
      continue

    # Stop if max depth reached
    if current_depth >= max_depth:
      del dirs[:] # Clear dirs to stop recursion
      continue

  return sorted(found_dirs)


_RUN_DIR_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$')


def _extract_condition_and_run_token(result_path: str, batch_root: str) -> Tuple[str, Optional[str]]:
  """Extract condition name and run timestamp token from a discovered result path."""
  result_p = Path(result_path)
  try:
    rel = result_p.relative_to(Path(batch_root))
    parts = rel.parts
  except ValueError:
    parts = ()

  if not parts:
    return result_p.name, None

  condition = parts[0]
  run_token = None
  if len(parts) > 1 and _RUN_DIR_PATTERN.match(parts[1]):
    run_token = parts[1]

  return condition, run_token


def _select_latest_result_path_per_condition(found_paths: List[str], batch_root: str) -> Dict[str, str]:
  """Keep only the most recent run per condition from discovered result paths."""
  per_condition: Dict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
  for result_path in found_paths:
    condition, run_token = _extract_condition_and_run_token(result_path, batch_root)
    per_condition[condition].append((result_path, run_token))

  selected: Dict[str, str] = {}
  for condition, candidates in per_condition.items():
    with_token = [item for item in candidates if item[1] is not None]
    if with_token:
      # Timestamp format is lexicographically sortable: YYYY-MM-DD_HH-MM-SS.
      selected[condition] = max(with_token, key=lambda item: item[1])[0]
      continue

    def _mtime(path: str) -> float:
      try:
        return Path(path).stat().st_mtime
      except OSError:
        return float('-inf')

    selected[condition] = max(candidates, key=lambda item: (_mtime(item[0]), item[0]))[0]

  return selected


@st.cache_data(show_spinner=False)
def resolve_results_root() -> str:
  """
  Resolve the default results root robustly across local/remote launches.

  Priority:
  1) SCRB_RESULTS_ROOT env var if it points to an existing directory
  2) <repo_root>/results inferred from this file location
  3) <cwd>/results
  4) cwd fallback
  """
  env_root = os.getenv('SCRB_RESULTS_ROOT', '').strip()
  if env_root:
    env_path = Path(env_root).expanduser()
    if env_path.is_dir():
      return str(env_path.resolve())

  try:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    repo_results = repo_root / 'results'
    if repo_results.is_dir():
      return str(repo_results.resolve())
  except Exception:
    pass

  cwd_results = Path(os.getcwd()) / 'results'
  if cwd_results.is_dir():
    return str(cwd_results.resolve())

  return str(Path(os.getcwd()).resolve())


def render_directory_selector(key_prefix="dir_selector"):
  """
  Optimized navigation component with automatic results detection.
  Recursively scans the 'results' folder to propose quick choices.
  """
  if f'{key_prefix}_path' not in st.session_state:
    st.session_state[f'{key_prefix}_path'] = resolve_results_root()

  current_path = st.session_state[f'{key_prefix}_path']
  results_root = resolve_results_root()

  # --- Quick Load from 'results/' ---
  col_auto_title, col_auto_btn = st.columns([4, 1])
  with col_auto_title:
    st.markdown("###### Automatic Detection")
  with col_auto_btn:
    refresh_scan = st.button(
      "Rescan",
      key=f"{key_prefix}_refresh_scan",
      use_container_width=True,
      help="Refresh detected folders list."
    )
  
  # Scan logic (cached in session state, with explicit refresh and root-change invalidation)
  scan_cache_key = f'{key_prefix}_found_results'
  scan_root_key = f'{key_prefix}_scan_root'
  needs_scan = (
    refresh_scan
    or scan_cache_key not in st.session_state
    or st.session_state.get(scan_root_key) != results_root
  )

  if needs_scan:
    with st.spinner("Scanning available results..."):
      found = scan_for_results(results_root, max_depth=3)
      options = {}
      for p in found:
        try:
          rel = os.path.relpath(p, results_root)
          if rel == '.': 
            label = "root (results)"
          else:
            # Pretty print: Replace path separators with arrows
            label = rel.replace(os.sep, ' ')
          options[label] = p
        except ValueError:
          continue
      st.session_state[scan_cache_key] = options
      st.session_state[scan_root_key] = results_root

  options = st.session_state.get(scan_cache_key, {})
  
  if options:
    selected_quick = st.selectbox(
      "Found results (click to load)",
      ["-- Select a result --"] + list(options.keys()),
      key=f"{key_prefix}_quick_select",
      label_visibility="collapsed"
    )
    
    if selected_quick != "-- Select a result --":
      target = options[selected_quick]
      if target != current_path:
        st.session_state[f'{key_prefix}_path'] = target
        st.rerun()
  else:
    st.caption("No result detected automatically.")

  st.divider()
  
  # --- Classic Navigation ---
  st.markdown("###### Manual Navigation")
  col_nav1, col_nav2 = st.columns([1, 5])
  
  with col_nav1:
    if st.button("Parent", key=f"{key_prefix}_up", use_container_width=True):
      st.session_state[f'{key_prefix}_path'] = os.path.dirname(current_path)
      st.rerun()
      
  with col_nav2:
    st.code(current_path, language=None)

  # --- Directory Content Analysis ---
  try:
    all_items = [d for d in os.listdir(current_path) if not d.startswith('.')]
    dirs = [d for d in all_items if os.path.isdir(os.path.join(current_path, d))]
    dirs.sort()
  except Exception as e:
    st.error(f"Access error : {e}")
    dirs = []

  # --- Smart Navigation Logic ---
  
  # 1. Check if CURRENT directory is a result
  is_current_valid = is_result_directory(current_path)
  
  # 2. Check subdirectories status
  subdir_status = []
  for d in dirs:
    full_p = os.path.join(current_path, d)
    if is_result_directory(full_p):
      subdir_status.append((d, "Results"))
    else:
      subdir_status.append((d, "Folder"))
      
  # --- Fast Forward for Single Folder ---
  if len(dirs) == 1 and not is_current_valid:
    single_dir = dirs[0]
    st.info(f"A single subfolder detected : **{single_dir}**")
    if st.button(f"Enter '{single_dir}'", key=f"{key_prefix}_fast_fwd", use_container_width=True, type="primary"):
      st.session_state[f'{key_prefix}_path'] = os.path.join(current_path, single_dir)
      st.rerun()

  # --- Directory Selection (Dropdown) ---
  if dirs:
    # Format options for the dropdown to show status
    options_map = {f"{status} | {name}": name for name, status in subdir_status}
    display_options = ["-- Navigate to... --"] + list(options_map.keys())
    
    selected_display = st.selectbox(
      "Folder content", 
      display_options,
      key=f"{key_prefix}_select",
      label_visibility="collapsed"
    )
    
    if selected_display != "-- Navigate to... --":
      dir_name = options_map[selected_display]
      st.session_state[f'{key_prefix}_path'] = os.path.join(current_path, dir_name)
      st.rerun()
  elif not is_current_valid:
    st.caption("No visible subfolder.")

  # --- Suggest Condition Name ---
  suggested_name = ""
  if is_current_valid:
    # Try to be smart: use folder name, but if it's 'results', use parent
    dirname = os.path.basename(current_path)
    if dirname.lower() in ['results', 'output', 'benchmarks']:
      parent = os.path.basename(os.path.dirname(current_path))
      suggested_name = f"{parent}"
    else:
      suggested_name = dirname
  
  return current_path, is_current_valid, suggested_name


def render_results_explorer_page():
  """Render the results explorer page."""
  st.header("Results Explorer")
  st.caption("Load benchmark results and generate interactive visualizations.")

  # Initialize session state
  if 'explorer_results' not in st.session_state:
    st.session_state.explorer_results = {}
  if 'explorer_agg_df' not in st.session_state:
    st.session_state.explorer_agg_df = None





  st.subheader("1. Load Results")

  with st.expander("Add Results", expanded=len(st.session_state.explorer_results) == 0):
    # Directory Selector
    selected_path, is_valid, suggested_name = render_directory_selector()
    
    st.divider()
    st.markdown("##### Configuration")
    
    col1, col2 = st.columns([3, 1])

    with col1:
      if is_valid:
        st.success(f"Valid results detected!\n\n`{selected_path}`")
      else:
        st.info(f"Selected folder : **{os.path.basename(selected_path)}**\n\n`{selected_path}`")

    with col2:
      # Auto-fill condition name logic
      # Use session state to pre-fill the name if it's empty and we have a suggestion
      if 'explorer_cond_name' not in st.session_state:
        st.session_state.explorer_cond_name = ""
      
      if is_valid and suggested_name and not st.session_state.explorer_cond_name:
        st.session_state.explorer_cond_name = suggested_name
      
      condition_name = st.text_input(
        "Condition name",
        key="explorer_cond_name",
        placeholder="ex: Split Balanced",
        help="A descriptive name to identify this condition"
      )

    # Smart Button
    if is_valid:
      if st.button("Load these results", type="primary", use_container_width=True):
         if not condition_name:
           st.warning("Please provide a condition name.")
         else:
          result = load_results_from_directory(selected_path, condition_name)
          if result:
            st.session_state.explorer_results[condition_name] = result
            st.session_state.explorer_agg_df = aggregate_metrics(st.session_state.explorer_results)
            st.success(f"Results '{condition_name}' loaded!")
            st.rerun()
          else:
            st.error("Unexpected error while loading.")
    else:
      st.button("Select a folder containing results...", disabled=True, use_container_width=True)

  with st.expander("Load a full folder", expanded=False):
    st.caption("Automatically load all detected conditions from a parent directory.")
    default_batch_path = resolve_results_root()
    batch_path = st.text_input(
      "Parent folder path",
      value=default_batch_path,
      key="explorer_batch_path",
      help="Path to the folder containing condition subfolders (e.g., results_full_benchmark_5runs/)"
    )
    if st.button("Load all conditions", type="primary", use_container_width=True):
      batch_path = batch_path.strip()
      if not batch_path or not os.path.isdir(batch_path):
        st.error("Specified path does not exist or is not a folder.")
      else:
        found = scan_for_results(batch_path)
        if not found:
          st.warning("No results folder found in this directory.")
        else:
          selected_paths = _select_latest_result_path_per_condition(found, batch_path)
          loaded_count = 0
          for condition_name, result_path in sorted(selected_paths.items()):
            if condition_name not in st.session_state.explorer_results:
              result = load_results_from_directory(result_path, condition_name)
              if result:
                st.session_state.explorer_results[condition_name] = result
                loaded_count += 1
          if loaded_count > 0:
            st.session_state.explorer_agg_df = aggregate_metrics(st.session_state.explorer_results)
          st.success(
            f"{loaded_count} condition(s) loaded out of {len(selected_paths)} condition(s) detected "
            f"({len(found)} run(s) found)."
          )
          st.rerun()

  # Display loaded results
  if st.session_state.explorer_results:
    n_loaded = len(st.session_state.explorer_results)
    with st.expander(f"**Loaded Results** ({n_loaded} condition{'s' if n_loaded > 1 else ''})", expanded=False):
      cols = st.columns(min(3, n_loaded))
      # We use list() to avoid dictionary mutation during iteration
      for i, (name, data) in enumerate(list(st.session_state.explorer_results.items())):
        with cols[i % len(cols)]:
          with st.container(border=True):
            # Header with delete
            col_header, col_del = st.columns([4, 1])
            with col_header:
              st.markdown(f"**{name}**")
            with col_del:
              if st.button("", key=f"del_{name}", help="Delete"):
                del st.session_state.explorer_results[name]
                if st.session_state.explorer_results:
                  st.session_state.explorer_agg_df = aggregate_metrics(st.session_state.explorer_results)
                else:
                  st.session_state.explorer_agg_df = None
                st.rerun()

            # Edit name expander
            with st.expander("Rename"):
              new_name = st.text_input("New name", value=name, key=f"input_ren_{name}", label_visibility="collapsed")
              if st.button("Confirm", key=f"btn_ren_{name}", use_container_width=True):
                if new_name and new_name != name:
                  if new_name in st.session_state.explorer_results:
                    st.error("This name already exists.")
                  else:
                    # Rename in the dict
                    data['condition'] = new_name
                    st.session_state.explorer_results[new_name] = st.session_state.explorer_results.pop(name)
                    # Re-aggregate because condition name is a column in the agg_df
                    st.session_state.explorer_agg_df = aggregate_metrics(st.session_state.explorer_results)
                    st.rerun()

            st.caption(f"{os.path.basename(data['path'])}")
            st.caption(f"{data['df']['algorithm'].nunique() if 'algorithm' in data['df'].columns else data['df']['algorithm_name'].nunique()} algos | {len(data['df'])} runs")

  if not st.session_state.explorer_results:
    st.info("No results loaded. Add results to get started.")
    return

  st.divider()

  # ==========================================================================
  # Section 2: Selection
  # ==========================================================================
  st.subheader("2. Selection")

  # Get available algorithms and conditions
  all_algos = set()
  for data in st.session_state.explorer_results.values():
    df = data['df']
    algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'
    all_algos.update(df[algo_col].unique())
  all_algos = sorted(all_algos)

  all_conditions = list(st.session_state.explorer_results.keys())

  col1, col2 = st.columns(2)

  with col1:
    selected_algorithms = st.multiselect(
      "Algorithms",
      options=all_algos,
      default=all_algos,
      format_func=lambda x: ALGO_DISPLAY_NAMES.get(x, x)
    )

  with col2:
    selected_conditions = st.multiselect(
      "Conditions",
      options=all_conditions,
      default=all_conditions
    )

  if not selected_algorithms or not selected_conditions:
    st.warning("Select at least one algorithm and one condition.")
    return

  st.divider()

  # ==========================================================================
  # Section 3: Visualization & Analysis
  # ==========================================================================
  st.subheader("3. Visualization & Analysis")

  col_fig, col_opts = st.columns([2, 1])

  # Enofe registry renderers are loaded before listing available figures.
  from . import figures as _figure_registrations  # noqa: F401
  available_figures = FigureRegistry.keys() or list(FIGURE_TYPES.keys())

  with col_fig:
    figure_search = st.text_input(
      "Search a figure",
      value="",
      placeholder="ex: UMAP, confusion, runtime...",
      help="Filter by key, name, or figure description."
    ).strip().lower()
    all_categories = sorted({info.category for info in FigureRegistry.list_infos()}) or sorted(
      {FIGURE_CATEGORY_MAP.get(key, "Other") for key in available_figures}
    )
    selected_categories = st.multiselect(
      "Categories",
      options=all_categories,
      default=all_categories,
    )

    filtered_infos = FigureRegistry.filter_available(
      selected_algorithms=selected_algorithms,
      selected_conditions=selected_conditions,
      search_text=figure_search,
      categories=selected_categories,
    )
    filtered_figures = [info.key for info in filtered_infos]

    if not filtered_figures:
      st.warning("No figure type matches the current filters.")
      return

    # Figure type selection
    figure_type = st.selectbox(
      "Analysis type",
      options=filtered_figures,
      format_func=lambda x: (FigureRegistry.info(x).name if FigureRegistry.info(x) else FIGURE_TYPES[x]['name'])
    )
    info = FigureRegistry.info(figure_type)
    fig_category = info.category if info else FIGURE_CATEGORY_MAP.get(figure_type, "Other")
    fig_description = info.description if info else FIGURE_TYPES[figure_type]['description']
    st.caption(f"[{fig_category}] {fig_description}")

    # Metric selection
    metric = 'NMI'
    metric_figures = ['algorithm_comparison', 'metrics_heatmap', 'generalization_gap',
             'generalization_gap_heatmap', 'statistical_test']
    if figure_type in metric_figures:
      require_train_test = figure_type in ['generalization_gap', 'generalization_gap_heatmap']
      metric_options = get_available_metrics(st.session_state.explorer_agg_df, require_train_test=require_train_test)
      if not metric_options:
        metric_options = ['NMI', 'ARI', 'ACC', 'Silhouette']
      metric = st.selectbox("Target metric", options=metric_options)

    # Additional options for statistical test
    test_mode = 'pairwise_conditions'
    if figure_type == 'statistical_test':
      test_mode = st.radio(
        "Comparison mode",
        ['pairwise_conditions', 'pairwise_algorithms'],
        format_func=lambda x: "Compare conditions (per algorithm)" if x == 'pairwise_conditions'
                   else "Compare algorithms (per condition)",
        help="Choose whether to compare conditions against each other for each algorithm, "
           "or compare algorithms against each other for each condition."
      )

    umap_split = 'all'
    umap_max_images = 18
    if figure_type in ['saved_umap_gallery', 'umap_evolution']:
      if figure_type == 'saved_umap_gallery':
        umap_split = st.selectbox(
          "UMAP split",
          options=['all', 'train', 'test', 'val', 'full'],
          format_func=lambda x: {
            'all': 'All',
            'train': 'Train',
            'test': 'Test',
            'val': 'Validation',
            'full': 'Full (standard)'
          }[x]
        )
      umap_max_images = st.slider(
        "Max number of images",
        min_value=1,
        max_value=60,
        value=18,
        step=1
      )

    diag_condition = selected_conditions[0] if selected_conditions else ''
    diag_algorithm = selected_algorithms[0] if selected_algorithms else ''
    diag_run_id = 0
    diag_split = 'test'
    diag_outline_mode = 'none'
    diag_max_points = 15000
    diag_seed = 42
    diag_show_centroids = True
    if figure_type == 'umap_diagnostic':
      diag_entries = list_umap_diagnostic_entries(
        st.session_state.explorer_results,
        selected_algorithms,
        selected_conditions
      )
      if not diag_entries:
        st.warning(
          "No run/split available for UMAP diagnostic "
          "(requires `runX_split` label files and an associated h5ad)."
        )
      else:
        cond_opts = [c for c in selected_conditions if any(e['condition'] == c for e in diag_entries)]
        diag_condition = st.selectbox("Diagnostic condition", options=cond_opts)

        cond_entries = [e for e in diag_entries if e['condition'] == diag_condition]
        algo_opts = [a for a in selected_algorithms if any(e['algorithm'] == a for e in cond_entries)]
        diag_algorithm = st.selectbox(
          "Diagnostic algorithm",
          options=algo_opts,
          format_func=lambda x: ALGO_DISPLAY_NAMES.get(x, x)
        )

        algo_entries = [e for e in cond_entries if e['algorithm'] == diag_algorithm]
        run_opts = sorted({e['run_id'] for e in algo_entries})
        diag_run_id = st.selectbox("Run", options=run_opts, format_func=lambda x: f"run{x}")

        run_entries = [e for e in algo_entries if e['run_id'] == diag_run_id]
        split_order = ['train', 'test', 'val', 'full']
        split_opts = [s for s in split_order if any(e['split'] == s for e in run_entries)]
        diag_split = st.selectbox(
          "Split",
          options=split_opts,
          format_func=lambda x: {
            'train': 'Train',
            'test': 'Test',
            'val': 'Validation',
            'full': 'Full (standard)'
          }.get(x, x)
        )

        _diag_outline_options = ['none', 'convex_hull', 'ellipse', 'density']
        diag_outline_mode = st.selectbox(
          "Cluster highlighting",
          options=_diag_outline_options,
          index=_diag_outline_options.index(diag_outline_mode) if diag_outline_mode in _diag_outline_options else 0,
          format_func=lambda x: {
            'none': 'No outline (centroid labels)',
            'convex_hull': 'Convex Hull',
            'ellipse': 'Confidence ellipse',
            'density': 'Density contours',
          }[x]
        )
        diag_show_centroids = st.checkbox("Show cluster centroids", value=True)
        diag_max_points = st.slider(
          "Max number of cells",
          min_value=2000,
          max_value=60000,
          value=15000,
          step=1000,
          help="Subsampling to speed up rendering and keep the plot readable."
        )
        diag_seed = st.number_input(
          "Subsampling seed",
          min_value=0,
          max_value=1_000_000,
          value=42,
          step=1
        )

    per_condition_compatible_figures = [
      'algorithm_comparison',
      'metrics_heatmap',
      'generalization_gap',
      'generalization_gap_heatmap',
      'batch_generalization',
      'test_metrics_by_batch',
      'saved_umap_gallery',
      'celltype_errors',
      'celltype_errors_by_batch',
      'confusion_patterns',
      'confusion_matrix_detailed',
      'error_rate_by_batch',
      'confusion_matrix_by_batch',
      'runtime_comparison',
      'train_vs_test',
      'generalization_gap_combined',
    ]
    split_figures_by_condition = False
    if len(selected_conditions) > 1 and figure_type in per_condition_compatible_figures:
      split_figures_by_condition = st.checkbox(
        "One figure per condition",
        value=(figure_type in ['celltype_errors', 'celltype_errors_by_batch', 'error_rate_by_batch']),
        help="Generate a separate figure for each selected condition (with selected algorithms)."
      )

  with col_opts:
    quick_mode = st.session_state.get("ui_mode", "quick") == "quick"
    with st.expander("Customization", expanded=not quick_mode):
      fig_context = st.selectbox("Context", ["notebook", "paper", "talk", "poster"], index=0, help="Adapt font sizes for the intended usage.")
      fig_style = st.selectbox("Style", ["whitegrid", "darkgrid", "ticks", "white"], index=0)
      export_format = st.selectbox("Export format", ["png", "svg", "pdf"], index=0)
      fig_dpi = st.number_input("DPI", value=300, min_value=72, max_value=600, step=50)
      debug_registry = st.checkbox(
        "Debug figure registry",
        value=False,
        help="Show visible/filtered figures and filtering reasons."
      )

      # CLD significance letters toggle
      cld_figures = ['algorithm_comparison', 'generalization_gap', 'generalization_gap_combined', 'runtime_comparison']
      show_cld = False
      if HAS_CLD and figure_type in cld_figures:
        show_cld = st.checkbox(
          "Significance letters (CLD)",
          value=False,
          help="Display Compact Letter Display above bars/boxes. "
               "Algorithms sharing the same letter are NOT significantly "
               "different (Kruskal-Wallis + Dunn, α=0.05)."
        )
      elif not HAS_CLD and figure_type in cld_figures:
        st.caption("CLD unavailable (missing utils.statistics import)")

    if debug_registry:
      registry_debug = debug_registry_state(
        selected_algorithms=selected_algorithms,
        selected_conditions=selected_conditions,
        search_text=figure_search,
        categories=selected_categories,
      )
      with st.expander("Registry state", expanded=True):
        st.markdown("**Visible figures**")
        st.write(registry_debug["visible"])
        st.markdown("**Filtered figures**")
        st.write(registry_debug["hidden"])

  # Apply plotting style
  sns.set_theme(context=fig_context, style=fig_style, palette="deep")

  # Generate button
  if st.button("Generate analysis", type="primary", use_container_width=True):
    agg_df = st.session_state.explorer_agg_df
    all_data = st.session_state.explorer_results

    with st.spinner("Processing..."):

      # 1. Statistical Test
      if figure_type == 'statistical_test':
        stats_df = perform_statistical_test(all_data, selected_algorithms, selected_conditions,
                          metric, test_mode)
        if not stats_df.empty:
          st.markdown(f"### Statistical test results (Mann-Whitney U) - {metric}")

          # Show explanation
          st.info(FIGURE_CAPTIONS.get('statistical_test', ''))

          # Highlight significant results
          sig_col = 'Sig. (FDR)' if 'Sig. (FDR)' in stats_df.columns else 'Sig. (brut)'
          st.dataframe(stats_df, use_container_width=True)

          # Summary
          n_sig = (stats_df[sig_col] != 'ns').sum() if sig_col in stats_df.columns else 0
          st.success(f"**{n_sig}** significant comparison(s) of {len(stats_df)} tests.")

          csv = stats_df.to_csv(index=False)
          st.download_button("Download CSV", csv, "stats_results.csv", "text/csv")
        else:
          st.info("Not enough data to run the test (minimum 3 runs per group).")
        return

      # 2. Summary Table
      if figure_type == 'summary_table':
        summary_df = create_summary_dataframe(agg_df, selected_algorithms, selected_conditions)
        page_size = st.selectbox("Page size", options=[10, 20, 50, 100], index=0, key="summary_table_page_size")
        total_rows = len(summary_df)
        n_pages = max(1, (total_rows + page_size - 1) // page_size)
        page = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1, key="summary_table_page")
        start = (int(page) - 1) * int(page_size)
        end = start + int(page_size)
        st.caption(f"Rows {start + 1}-{min(end, total_rows)} of {total_rows}")
        st.dataframe(summary_df.iloc[start:end], use_container_width=True)
        csv = summary_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "summary_table.csv", "text/csv")
        return

      # 2b. Metrics by Cell Type Table
      if figure_type == 'metrics_by_celltype_table':
        metrics_df = create_metrics_by_celltype_table(all_data, selected_algorithms, selected_conditions)

        if metrics_df.empty:
          st.warning("No label data available to compute metrics by cell type.")
          return

        st.markdown("### Metrics by Cell Type")

        # Display global statistics
        st.markdown("#### Global Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
          st.metric("Mean F1 Score", f"{metrics_df['F1_Score'].mean():.3f}")
        with col2:
          st.metric("Mean Precision", f"{metrics_df['Precision'].mean():.3f}")
        with col3:
          st.metric("Mean Recall", f"{metrics_df['Recall'].mean():.3f}")

        st.markdown("#### Detailed Table")

        # Filtres interactifs
        col_f1, col_f2 = st.columns(2)
        with col_f1:
          filter_algo = st.multiselect(
            "Filter by algorithm",
            options=metrics_df['Algorithm'].unique(),
            default=metrics_df['Algorithm'].unique()
          )
        with col_f2:
          filter_celltype = st.multiselect(
            "Filter by cell type",
            options=sorted(metrics_df['Type_Cellulaire'].unique()),
            default=sorted(metrics_df['Type_Cellulaire'].unique())
          )

        # Apply filters
        filtered_df = metrics_df[
          (metrics_df['Algorithm'].isin(filter_algo)) &
          (metrics_df['Type_Cellulaire'].isin(filter_celltype))
        ]

        # Display the styled dataframe
        st.dataframe(
          filtered_df.style.background_gradient(
            subset=['F1_Score', 'Precision', 'Recall'],
            cmap='RdYlGn',
            vmin=0,
            vmax=1
          ).format({
            'F1_Score': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}'
          }),
          use_container_width=True,
          height=600
        )

        # Export
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "metrics_by_celltype.csv", "text/csv")

        # Interpretation guide
        st.info(
          "**Interpretation guide:**\n"
          "- **F1 Score** : Harmonic mean of precision and recall (0-1, best = 1)\n"
          "- **Precision** : Among cells predicted as this type, how many are truly this type\n"
          "- **Recall** : Among true cells of this type, how many were detected\n"
          "- **Support** : Number of true cells of this type in the data"
        )
        return

      # 3. Publication Export
      if figure_type == 'publication_export':
        st.markdown("### Publication Export")

        export_fmt = st.radio("Format", ["LaTeX", "Markdown"], horizontal=True)
        fmt_key = 'latex' if export_fmt == "LaTeX" else 'markdown'

        output = export_publication_ready(agg_df, selected_algorithms, selected_conditions, fmt_key)

        st.code(output, language='latex' if fmt_key == 'latex' else 'markdown')

        st.download_button(
          f"Download ({export_fmt})",
          output,
          f"benchmark_table.{'tex' if fmt_key == 'latex' else 'md'}",
          "text/plain"
        )

        if export_fmt == "LaTeX":
          st.caption("Add `\\usepackage{booktabs}` to your LaTeX preamble.")
        return

      # 4. Figures
      def _plot_loss_curves_gallery(all_data, selected_algos, target_conditions):
        """Build a gallery figure from loss curve JSONs or PNGs found in result dirs."""
        import utils.visualization as viz_mod

        entries = []
        for condition in target_conditions:
          data = all_data.get(condition)
          if not data:
            continue
          load_path = data.get('path', '')
          if not load_path:
            continue

          # Search for loss_history dir (sibling of results/)
          search_dirs = []
          parent = os.path.dirname(load_path)
          sibling_loss = os.path.join(parent, 'loss_history')
          if os.path.isdir(sibling_loss):
            search_dirs.append(('json', sibling_loss))
          direct_loss = os.path.join(load_path, 'loss_history')
          if os.path.isdir(direct_loss):
            search_dirs.append(('json', direct_loss))

          # Also search for PNG fallback in figures/
          sibling_fig = os.path.join(parent, 'figures')
          if os.path.isdir(sibling_fig):
            search_dirs.append(('png', sibling_fig))
          direct_fig = os.path.join(load_path, 'figures')
          if os.path.isdir(direct_fig):
            search_dirs.append(('png', direct_fig))

          for algo in selected_algos:
            found = False
            for kind, search_dir in search_dirs:
              if found:
                break
              if kind == 'json':
                import glob as glob_mod
                pattern = os.path.join(search_dir, f'loss_{algo}_run*.json')
                json_files = sorted(glob_mod.glob(pattern))
                for jf in json_files:
                  try:
                    with open(jf, 'r') as f:
                      loss_data = json.load(f)
                    phases = loss_data.get('phases', [])
                    run_id = loss_data.get('run_id', 0)
                    fig = viz_mod.plot_loss_curves(phases, algo)
                    if fig:
                      entries.append({
                        'condition': condition,
                        'algorithm': algo,
                        'run_id': run_id,
                        'fig': fig,
                      })
                      found = True
                  except Exception:
                    continue
              elif kind == 'png' and not found:
                import glob as glob_mod
                pattern = os.path.join(search_dir, f'loss_curves_{algo}_run*.png')
                png_files = sorted(glob_mod.glob(pattern))
                for pf in png_files:
                  try:
                    img = plt.imread(pf)
                    fig_img, ax_img = plt.subplots(1, 1, figsize=(8, 4))
                    ax_img.imshow(img)
                    ax_img.axis('off')
                    ax_img.set_title(f'{algo} (from saved PNG)')
                    entries.append({
                      'condition': condition,
                      'algorithm': algo,
                      'run_id': 0,
                      'fig': fig_img,
                    })
                    found = True
                  except Exception:
                    continue

        if not entries:
          return None

        # Build composite figure
        n = len(entries)
        n_cols = min(2, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig_out, axes_out = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows), squeeze=False)

        for idx, entry in enumerate(entries):
          r, c = idx // n_cols, idx % n_cols
          ax = axes_out[r][c]
          # Render sub-figure into ax by extracting data from stored fig
          sub_fig = entry['fig']
          sub_axes = sub_fig.get_axes()
          for sub_ax in sub_axes:
            for line in sub_ax.get_lines():
              ax.plot(line.get_xdata(), line.get_ydata(),
                      color=line.get_color(),
                      linewidth=line.get_linewidth(),
                      linestyle=line.get_linestyle(),
                      alpha=line.get_alpha() or 1.0,
                      label=line.get_label() if not line.get_label().startswith('_') else None)
          ax.set_xlabel('Epoch')
          ax.set_ylabel('Loss')
          cond_label = entry['condition']
          algo_label = ALGO_DISPLAY_NAMES.get(entry['algorithm'], entry['algorithm'])
          ax.set_title(f"{algo_label} [{cond_label}] run{entry['run_id']}", fontsize=10, fontweight='bold')
          handles, labels = ax.get_legend_handles_labels()
          if handles:
            ax.legend(fontsize=7)
          ax.grid(True, alpha=0.3)
          plt.close(sub_fig)

        # Hide unused
        for idx in range(n, n_rows * n_cols):
          r, c = idx // n_cols, idx % n_cols
          axes_out[r][c].axis('off')

        fig_out.suptitle('Loss Curves', fontsize=13, fontweight='bold')
        plt.tight_layout()
        return fig_out

      def _plot_umap_evolution_gallery(all_data, selected_algos, target_conditions, max_images=18):
        """Build a gallery figure from saved UMAP evolution PNGs."""
        return plot_saved_umap_evolution_gallery(
          all_data=all_data,
          selected_algos=selected_algos,
          selected_conditions=target_conditions,
          max_images=int(max_images),
        )

      def _build_figure(target_conditions: List[str]):
        try:
          fig_local = FigureRegistry.render(
            figure_type,
            agg_df=agg_df,
            all_data=all_data,
            selected_algorithms=selected_algorithms,
            selected_conditions=selected_conditions,
            target_conditions=target_conditions,
            metric=metric,
            show_cld=show_cld,
            umap_split=umap_split,
            umap_max_images=umap_max_images,
            diag_condition=diag_condition,
            diag_algorithm=diag_algorithm,
            diag_run_id=diag_run_id,
            diag_split=diag_split,
            diag_outline_mode=diag_outline_mode,
            diag_max_points=diag_max_points,
            diag_seed=diag_seed,
            diag_show_centroids=diag_show_centroids,
            plot_loss_curves_gallery=_plot_loss_curves_gallery,
            plot_umap_evolution_gallery=_plot_umap_evolution_gallery,
          )
        except KeyError:
          return None, 'not_implemented'
        except Exception as exc:
          st.error(f"Renderer error '{figure_type}': {exc}")
          return None, 'renderer_error'

        if fig_local == 'no_batch_column':
          return None, 'no_batch_column'
        return fig_local, None

      def _safe_token(text: str) -> str:
        token = re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')
        return token if token else 'condition'

      condition_groups = [selected_conditions]
      if split_figures_by_condition and figure_type != 'umap_diagnostic':
        condition_groups = [[cond] for cond in selected_conditions]

      rendered_figures = []
      for cond_group in condition_groups:
        fig, fig_error = _build_figure(cond_group)
        if fig_error == 'not_implemented':
          st.error(f"Figure type not implemented: {figure_type}")
          return
        if fig_error == 'renderer_error':
          return
        if fig_error == 'no_batch_column':
          cond_text = ', '.join(cond_group)
          st.warning(
            f"**'batch' column missing from label files** for condition `{cond_text}`.\n\n"
            "Existing label CSV files do not contain batch information "
            "(format: `predicted_label, true_label` only).\n\n"
            "**Solution:** Rerun the benchmark with the current code version. "
            "The `batch` column will be automatically added to CSV files "
            "(format: `predicted_label, true_label, batch`).\n\n"
            "**Alternative:** Use **'Error Rate by Batch'**, which works "
            "with existing data (via `error_by_group` in `benchmark_detailed.csv`)."
          )
          continue
        if fig is not None:
          rendered_figures.append((cond_group, fig))

      if not rendered_figures:
        st.info("No figure generated for the current selection.")
        return

      # Display figures
      for idx, (cond_group, fig) in enumerate(rendered_figures):
        cond_text = ', '.join(cond_group)
        if split_figures_by_condition and len(rendered_figures) > 1:
          st.markdown(f"#### Condition: `{cond_text}`")

        # Determine if figure needs scrollable container
        # Check figure size or use heuristics for known large figure types
        large_figure_types = ['confusion_matrix_detailed', 'confusion_matrix_by_batch',
                              'celltype_errors', 'celltype_errors_by_batch', 'batch_generalization',
                              'saved_umap_gallery', 'umap_diagnostic']
        n_conditions_local = len(cond_group)
        n_algos = len(selected_algorithms)

        # Enable scrolling for large figures or many conditions/algorithms
        needs_scroll = (figure_type in large_figure_types or
                        n_conditions_local > 5 or
                        n_algos > 5 or
                        (hasattr(fig, 'get_size_inches') and fig.get_size_inches()[1] > 10))

        if needs_scroll:
          st.markdown(
            """
            <style>
            .scrollable-figure-container {
              overflow-x: auto;
              overflow-y: auto;
              max-height: 800px;
              max-width: 100%;
              padding: 10px;
              border: 1px solid #e0e0e0;
              border-radius: 5px;
              background: white;
            }
            </style>
            """,
            unsafe_allow_html=True
          )
          st.markdown('<div class="scrollable-figure-container">', unsafe_allow_html=True)
          st.pyplot(fig)
          st.markdown('</div>', unsafe_allow_html=True)
        else:
          st.pyplot(fig)

        # Show interpretation guide
        if figure_type in FIGURE_CAPTIONS:
          st.info(f"**Interpretation guide:** {FIGURE_CAPTIONS[figure_type]}")

        metric_suffix = metric if figure_type in metric_figures else 'all'
        condition_suffix = _safe_token(cond_text) if split_figures_by_condition else 'multi'
        file_prefix = f"{figure_type}_{metric_suffix}_{condition_suffix}"
        dataset_name = st.session_state.get("uploaded_file_name", "dataset")
        dataset_token = _safe_token(dataset_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{dataset_token}_{file_prefix}_{timestamp}"
        render_export_panel(
          fig=fig,
          base_filename=base_filename,
          formats=["PNG", "SVG", "PDF"],
          show_dpi=True,
        )

        plt.close(fig)

  # ==========================================================================
  # Section 4: Quick Stats
  # ==========================================================================
  st.divider()
  st.subheader("4. Quick Statistics")

  if st.session_state.explorer_agg_df is not None:
    agg_df = st.session_state.explorer_agg_df
    filtered_df = agg_df[
      (agg_df['algorithm'].isin(selected_algorithms)) &
      (agg_df['condition'].isin(selected_conditions))
    ]

    if not filtered_df.empty:
      col1, col2, col3, col4 = st.columns(4)

      with col1:
        if 'NMI_mean' in filtered_df.columns:
          best_nmi = filtered_df.loc[filtered_df['NMI_mean'].idxmax()]
          st.metric(
            "Best NMI",
            f"{best_nmi['NMI_mean']:.3f}",
            f"{ALGO_DISPLAY_NAMES.get(best_nmi['algorithm'], best_nmi['algorithm'])}"
          )

      with col2:
        if 'ARI_mean' in filtered_df.columns:
          best_ari = filtered_df.loc[filtered_df['ARI_mean'].idxmax()]
          st.metric(
            "Best ARI",
            f"{best_ari['ARI_mean']:.3f}",
            f"{ALGO_DISPLAY_NAMES.get(best_ari['algorithm'], best_ari['algorithm'])}"
          )

      with col3:
        if 'ACC_mean' in filtered_df.columns:
          best_acc = filtered_df.loc[filtered_df['ACC_mean'].idxmax()]
          st.metric(
            "Best ACC",
            f"{best_acc['ACC_mean']:.3f}",
            f"{ALGO_DISPLAY_NAMES.get(best_acc['algorithm'], best_acc['algorithm'])}"
          )

      with col4:
        total_runs = filtered_df['n_runs'].sum()
        st.metric("Total Runs", int(total_runs))
