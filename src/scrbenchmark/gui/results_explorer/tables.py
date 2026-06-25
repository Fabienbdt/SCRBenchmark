"""Table and publication-export helpers for Results Explorer."""

import ast
from typing import Dict, List

import numpy as np
import pandas as pd

from . import legacy
from .constants import ALGO_DISPLAY_NAMES


def create_summary_dataframe(agg_df: pd.DataFrame, selected_algos: List[str],
               selected_conditions: List[str]) -> pd.DataFrame:
  """Create summary dataframe with all metrics."""
  df = agg_df[
    (agg_df['algorithm'].isin(selected_algos)) &
    (agg_df['condition'].isin(selected_conditions))
  ]

  summary_records = []

  # Identify available metrics from aggregated columns (test metrics by default)
  metric_means = [
    c for c in df.columns
    if c.endswith('_mean')
    and not c.endswith('_train_mean')
    and not c.startswith('runtime')
  ]
  metrics = legacy._sort_metrics(list({c[:-5] for c in metric_means}))

  for algo in sorted(df['algorithm'].unique()):
    algo_df = df[df['algorithm'] == algo]

    for _, row in algo_df.iterrows():
      record = {
        'Algorithm': ALGO_DISPLAY_NAMES.get(algo, algo),
        'Condition': row['condition'],
        'N Runs': row.get('n_runs', '-'),
      }

      for metric in metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        if mean_col in row and not pd.isna(row[mean_col]):
          mean_val = row[mean_col]
          std_val = row.get(std_col, 0) if not pd.isna(row.get(std_col, np.nan)) else 0
          record[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
        else:
          record[metric] = '-'

      if 'runtime_mean' in row and not pd.isna(row.get('runtime_mean')):
        record['Runtime (s)'] = f"{row['runtime_mean']:.1f}"
      else:
        record['Runtime (s)'] = '-'

      summary_records.append(record)

  return pd.DataFrame(summary_records)


def create_metrics_by_celltype_table(all_data: Dict[str, Dict], selected_algos: List[str],
                   selected_conditions: List[str]) -> pd.DataFrame:
  """
  Creates a summary table with F1, Precision, and Recall per algorithm and cell type.

  Args:
    all_data: Dictionary of loaded results
    selected_algos: List of selected algorithms
    selected_conditions: List of selected conditions

  Returns:
    DataFrame with columns: Algorithm, Cell_Type, F1_Score, Precision, Recall, Support
  """
  from sklearn.metrics import precision_recall_fscore_support

  train_counts = legacy._collect_train_counts(all_data, selected_conditions)
  records = []

  # First try loading from label files (same approach as confusion matrix)
  for condition, data in all_data.items():
    if condition not in selected_conditions:
      continue

    labels_dict = data.get('labels', {})
    label_map = data.get('label_map')

    if labels_dict:
      for algo, runs in labels_dict.items():
        if algo not in selected_algos:
          continue
        for run_key, labels_df in runs.items():
          if '_train' in run_key or '_val' in run_key:
            continue
          if 'true_label' not in labels_df.columns or 'predicted_label' not in labels_df.columns:
            continue

          y_true = np.array(labels_df['true_label'].values, dtype=str)
          y_pred = np.array(labels_df['predicted_label'].values, dtype=str)

          # Decode numeric true labels
          if label_map:
            try:
              [int(t) for t in set(y_true)]
              y_true = np.array([label_map.get(t, t) for t in y_true])
            except ValueError:
              pass

          # Hungarian mapping for predicted labels
          true_set = set(y_true)
          pred_set = set(y_pred)
          if true_set != pred_set:
            from scipy.optimize import linear_sum_assignment
            true_unique = sorted(true_set)
            pred_unique = sorted(pred_set)
            true_to_idx = {l: i for i, l in enumerate(true_unique)}
            pred_to_idx = {l: i for i, l in enumerate(pred_unique)}
            cost = np.zeros((len(pred_unique), len(true_unique)), dtype=int)
            for t, p in zip(y_true, y_pred):
              cost[pred_to_idx[p], true_to_idx[t]] += 1
            row_ind, col_ind = linear_sum_assignment(-cost)
            pred_to_true = {}
            for r, c in zip(row_ind, col_ind):
              pred_to_true[pred_unique[r]] = true_unique[c]
            # Unmapped clusters: assign to the most frequent type
            for p in pred_unique:
              if p not in pred_to_true:
                p_idx = pred_to_idx[p]
                if cost[p_idx].sum() > 0:
                  best_true_idx = np.argmax(cost[p_idx])
                  pred_to_true[p] = true_unique[best_true_idx]
                else:
                  pred_to_true[p] = true_unique[0]
            y_pred = np.array([pred_to_true[p] for p in y_pred])

          labels_unique = sorted(set(y_true) | set(y_pred))
          precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels_unique, average=None, zero_division=0
          )
          for i, label in enumerate(labels_unique):
            record = {
              'Algorithm': ALGO_DISPLAY_NAMES.get(algo, algo),
              'Condition': condition,
              'Cell_Type': str(label),
              'F1_Score': f1[i],
              'Precision': precision[i],
              'Recall': recall[i],
              'Support_Test': int(support[i]),
            }
            n_train = train_counts.get(str(label))
            record['N_Train'] = int(n_train) if n_train is not None else ''
            records.append(record)
    else:
      # Fallback: try to get labels from dataframe columns
      df = data['df']
      algo_col = 'algorithm' if 'algorithm' in df.columns else 'algorithm_name'

      for _, row in df.iterrows():
        algo = row.get(algo_col)
        if algo not in selected_algos:
          continue

        labels_true = row.get('true_labels')
        labels_pred = row.get('predicted_labels')

        if labels_true is None or labels_pred is None:
          continue

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

        labels_unique = sorted(set(labels_true) | set(labels_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
          labels_true, labels_pred, labels=labels_unique, average=None, zero_division=0
        )
        for i, label in enumerate(labels_unique):
          record = {
            'Algorithm': ALGO_DISPLAY_NAMES.get(algo, algo),
            'Condition': condition,
            'Cell_Type': str(label),
            'F1_Score': f1[i],
            'Precision': precision[i],
            'Recall': recall[i],
            'Support_Test': int(support[i]),
          }
          n_train = train_counts.get(str(label))
          record['N_Train'] = int(n_train) if n_train is not None else ''
          records.append(record)

  if not records:
    return pd.DataFrame()

  return pd.DataFrame(records)


def export_publication_ready(agg_df: pd.DataFrame, selected_algos: List[str],
               selected_conditions: List[str],
               format: str = 'latex') -> str:
  """
  Generate a publication-ready formatted table.

  Args:
    agg_df: Aggregated DataFrame with metrics
    selected_algos: Algorithms to include
    selected_conditions: Conditions to include
    format: 'latex' or 'markdown'

  Returns:
    Formatted string (LaTeX or Markdown)
  """
  df = agg_df[
    (agg_df['algorithm'].isin(selected_algos)) &
    (agg_df['condition'].isin(selected_conditions))
  ]

  metrics = ['NMI', 'ARI', 'ACC', 'F1_Macro', 'BalancedACC']

  # Find best values per condition/metric
  best_values = {}
  for cond in selected_conditions:
    for metric in metrics:
      mean_col = f'{metric}_mean'
      cond_df = df[df['condition'] == cond]
      if mean_col in cond_df.columns and not cond_df.empty:
        best_val = cond_df[mean_col].max()
        best_values[(cond, metric)] = best_val

  # Build table rows
  rows = []
  for algo in sorted(df['algorithm'].unique()):
    algo_df = df[df['algorithm'] == algo]
    row = {'Algorithm': ALGO_DISPLAY_NAMES.get(algo, algo)}

    for cond in selected_conditions:
      cond_row = algo_df[algo_df['condition'] == cond]
      for metric in metrics:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'

        if not cond_row.empty and mean_col in cond_row.columns:
          mean = cond_row[mean_col].values[0]
          std = cond_row[std_col].values[0] if std_col in cond_row.columns else 0

          # Mark if it is the best value
          is_best = (cond, metric) in best_values and abs(mean - best_values[(cond, metric)]) < 0.001
          row[f'{cond}_{metric}'] = (mean, std, is_best)
        else:
          row[f'{cond}_{metric}'] = (None, None, False)

    rows.append(row)

  if format == 'latex':
    # Generate LaTeX
    n_cols = 1 + len(selected_conditions) * len(metrics)
    latex_lines = [
      "\\begin{table}[htbp]",
      "\\centering",
      "\\caption{Benchmark Results (mean $\\pm$ std)}",
      "\\label{tab:benchmark}",
      "\\begin{tabular}{l" + "c" * (n_cols - 1) + "}",
      "\\toprule"
    ]

    # Header row 1: condition names (multicolumn)
    header1 = "Algorithm"
    for cond in selected_conditions:
      header1 += f" & \\multicolumn{{{len(metrics)}}}{{c}}{{{cond}}}"
    latex_lines.append(header1 + " \\\\")

    # Header row 2: metric names
    header2 = ""
    for _ in selected_conditions:
      for metric in metrics:
        header2 += f" & {metric}"
    latex_lines.append(header2 + " \\\\")
    latex_lines.append("\\midrule")

    # Data rows
    for row in rows:
      line = row['Algorithm']
      for cond in selected_conditions:
        for metric in metrics:
          val = row.get(f'{cond}_{metric}', (None, None, False))
          if val[0] is not None:
            formatted = f"{val[0]:.3f}$\\pm${val[1]:.3f}"
            if val[2]: # Best value
              formatted = f"\\textbf{{{formatted}}}"
            line += f" & {formatted}"
          else:
            line += " & -"
      latex_lines.append(line + " \\\\")

    latex_lines.extend([
      "\\bottomrule",
      "\\end{tabular}",
      "\\end{table}"
    ])

    return "\n".join(latex_lines)

  else: # markdown
    md_lines = []

    # Header
    header = "| Algorithm |"
    separator = "|:---|"
    for cond in selected_conditions:
      for metric in metrics:
        header += f" {cond} {metric} |"
        separator += ":---:|"
    md_lines.append(header)
    md_lines.append(separator)

    # Data rows
    for row in rows:
      line = f"| {row['Algorithm']} |"
      for cond in selected_conditions:
        for metric in metrics:
          val = row.get(f'{cond}_{metric}', (None, None, False))
          if val[0] is not None:
            formatted = f"{val[0]:.3f}±{val[1]:.3f}"
            if val[2]: # Best value
              formatted = f"**{formatted}**"
            line += f" {formatted} |"
          else:
            line += " - |"
      md_lines.append(line)

    return "\n".join(md_lines)
