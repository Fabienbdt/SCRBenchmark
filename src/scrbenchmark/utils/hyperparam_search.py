"""
Hyperparameter search utilities for algorithm optimization.
Provides Grid Search and Random Search capabilities with multiple repetitions.
"""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
import json
import random
import numpy as np
import pandas as pd

from core.algorithm_registry import AlgorithmRegistry
from .metrics import compute_metrics, compute_error_analysis

logger = logging.getLogger(__name__)


def _generate_range_values(min_val: Union[int, float],
                           max_val: Union[int, float],
                           step: Union[int, float]) -> List[Union[int, float]]:
    """Generate deterministic values for a numeric range specification."""
    if step is None or step <= 0:
        raise ValueError(f"Invalid step value: {step}. Step must be > 0.")
    if min_val > max_val:
        raise ValueError(f"Invalid range: min ({min_val}) must be <= max ({max_val}).")

    if isinstance(min_val, int) and isinstance(max_val, int):
        int_step = int(step)
        if int_step <= 0:
            raise ValueError(f"Invalid integer step: {step}.")
        return list(range(min_val, max_val + 1, int_step))

    values = []
    current = float(min_val)
    stop = float(max_val)
    step = float(step)
    eps = abs(step) * 1e-9 + 1e-12

    while current <= stop + eps:
        values.append(round(current, 10))
        current += step
        # Hard safety guard against pathological specs.
        if len(values) > 1_000_000:
            raise ValueError("Range specification generated too many values.")

    # Ensure max value is included when step does not land exactly on it.
    if values and abs(values[-1] - stop) > eps and values[-1] < stop:
        values.append(round(stop, 10))

    if not values:
        values = [round(float(min_val), 10)]

    return values


def _infer_label_name_map(data: Any) -> Dict[str, str]:
    """
    Infer mapping from encoded label values to readable cell type names.

    Supports both orientations:
    - {"alpha": 0, "beta": 1}
    - {"0": "alpha", "1": "beta"}
    """
    if data is None or not hasattr(data, "uns"):
        return {}

    raw_map = data.uns.get("label_map")
    if not isinstance(raw_map, dict):
        return {}

    label_name_map = {}
    for key, value in raw_map.items():
        try:
            label_name_map[str(int(value))] = str(key)
        except Exception:
            pass
        try:
            label_name_map[str(int(key))] = str(value)
        except Exception:
            pass

    return label_name_map


def _decode_labels(labels: np.ndarray, label_name_map: Dict[str, str]) -> np.ndarray:
    """Decode labels to readable names when a mapping is available."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return labels.astype(str)

    if not label_name_map:
        return labels.astype(str)

    decoded = [label_name_map.get(str(lbl), str(lbl)) for lbl in labels]
    return np.asarray(decoded, dtype=object)


def _extract_group_ids(data: Any, expected_n: int) -> Tuple[np.ndarray, str]:
    """
    Extract group ids (typically batches) for error-by-group analysis.

    Returns:
      (group_ids, group_key_name)
    """
    if expected_n <= 0:
        return np.asarray([], dtype=object), "all"

    fallback = np.asarray(["all"] * expected_n, dtype=object)

    if data is None or not hasattr(data, "obs"):
        return fallback, "all"

    batch_col = None
    try:
        from .dataset_splitter import get_batch_column
        batch_col = get_batch_column(data)
    except Exception:
        batch_col = None

    if batch_col is None:
        for candidate in [
            "batch", "Batch", "tech", "dataset", "dataset_source",
            "study", "sample", "donor", "platform"
        ]:
            if candidate in data.obs.columns:
                batch_col = candidate
                break

    if batch_col is not None and batch_col in data.obs.columns:
        values = np.asarray(data.obs[batch_col].astype(str).values)
        if len(values) == expected_n:
            return values, str(batch_col)

    return fallback, "all"


def _aggregate_rate_entries(entries: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """Aggregate repeated error entries into mean/std stats."""
    if not entries:
        return None

    error_rates = []
    n_samples = []
    n_errors = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rate = entry.get("error_rate")
        if rate is not None:
            error_rates.append(float(rate))
        if entry.get("n_samples") is not None:
            n_samples.append(float(entry["n_samples"]))
        if entry.get("n_errors") is not None:
            n_errors.append(float(entry["n_errors"]))

    if not error_rates:
        return None

    return {
        "error_rate_mean": float(np.mean(error_rates)),
        "error_rate_std": float(np.std(error_rates)),
        "n_samples_mean": float(np.mean(n_samples)) if n_samples else np.nan,
        "n_errors_mean": float(np.mean(n_errors)) if n_errors else np.nan,
        "n_runs": int(len(error_rates))
    }


def _aggregate_error_analyses(
    all_error_analyses: List[Optional[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """Aggregate per-run error analyses for one parameter combination."""
    valid = [ea for ea in all_error_analyses if isinstance(ea, dict)]
    if not valid:
        return None

    summary: Dict[str, Any] = {
        "n_runs_valid": int(len(valid)),
        "group_key": next(
            (str(ea.get("group_key")) for ea in valid if ea.get("group_key") is not None),
            "all"
        )
    }

    overall = [ea.get("overall_error_rate") for ea in valid if ea.get("overall_error_rate") is not None]
    total_errors = [ea.get("total_errors") for ea in valid if ea.get("total_errors") is not None]
    total_samples = [ea.get("total_samples") for ea in valid if ea.get("total_samples") is not None]

    summary["overall_error_rate_mean"] = float(np.mean(overall)) if overall else np.nan
    summary["overall_error_rate_std"] = float(np.std(overall)) if overall else np.nan
    summary["total_errors_mean"] = float(np.mean(total_errors)) if total_errors else np.nan
    summary["total_samples_mean"] = float(np.mean(total_samples)) if total_samples else np.nan

    # Aggregate error_by_celltype and error_by_group
    for section_name in ["error_by_celltype", "error_by_group"]:
        section_values = {}
        keys = sorted({
            str(key)
            for ea in valid
            for key in ea.get(section_name, {}).keys()
        })
        for key in keys:
            entries = []
            for ea in valid:
                entry = ea.get(section_name, {}).get(key)
                if entry is not None:
                    entries.append(entry)
            aggregated = _aggregate_rate_entries(entries)
            if aggregated is not None:
                section_values[key] = aggregated
        summary[section_name] = section_values

    # Aggregate nested error_by_celltype_by_group
    nested_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    groups = sorted({
        str(group)
        for ea in valid
        for group in ea.get("error_by_celltype_by_group", {}).keys()
    })
    for group in groups:
        nested_summary[group] = {}
        cell_types = sorted({
            str(ct)
            for ea in valid
            for ct in ea.get("error_by_celltype_by_group", {}).get(group, {}).keys()
        })
        for ct in cell_types:
            entries = []
            for ea in valid:
                entry = ea.get("error_by_celltype_by_group", {}).get(group, {}).get(ct)
                if entry is not None:
                    entries.append(entry)
            aggregated = _aggregate_rate_entries(entries)
            if aggregated is not None:
                nested_summary[group][ct] = aggregated
    summary["error_by_celltype_by_group"] = nested_summary

    # Aggregate most frequent confusion pairs
    pair_stats: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for ea in valid:
        for pair in ea.get("top_confusion_pairs", []):
            true_label = pair.get("true")
            pred_label = pair.get("predicted")
            if true_label is None or pred_label is None:
                continue
            key = (str(true_label), str(pred_label))
            if key not in pair_stats:
                pair_stats[key] = {"count": [], "percentage": []}
            if pair.get("count") is not None:
                pair_stats[key]["count"].append(float(pair["count"]))
            if pair.get("percentage") is not None:
                pair_stats[key]["percentage"].append(float(pair["percentage"]))

    top_pairs = []
    for (true_label, pred_label), values in pair_stats.items():
        counts = values["count"]
        percentages = values["percentage"]
        if not counts:
            continue
        top_pairs.append({
            "true": true_label,
            "predicted": pred_label,
            "count_mean": float(np.mean(counts)),
            "count_std": float(np.std(counts)),
            "percentage_mean": float(np.mean(percentages)) if percentages else np.nan,
            "percentage_std": float(np.std(percentages)) if percentages else np.nan,
            "n_runs": int(len(counts))
        })
    top_pairs.sort(key=lambda x: x["count_mean"], reverse=True)
    summary["top_confusion_pairs"] = top_pairs[:10]

    return summary


@dataclass
class SearchResult:
    """Result of evaluating a single parameter combination."""
    params: Dict[str, Any]
    metrics: Dict[str, float]  # mean values across runs
    metrics_std: Dict[str, float]  # std values across runs
    runtime: float  # total runtime for all runs
    n_runs: int
    all_metrics: List[Dict[str, float]] = field(default_factory=list)  # individual run metrics
    error_analysis: Optional[Dict[str, Any]] = None  # aggregated error analysis across runs
    all_error_analyses: List[Dict[str, Any]] = field(default_factory=list)  # per-run error analyses

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'params': self.params,
            'metrics': self.metrics,
            'metrics_std': self.metrics_std,
            'runtime': self.runtime,
            'n_runs': self.n_runs,
            'error_analysis': self.error_analysis,
            'all_error_analyses': self.all_error_analyses
        }


@dataclass
class SearchSummary:
    """Summary of a complete hyperparameter search."""
    all_results: List[SearchResult]
    best_params: Dict[str, Any]
    best_score: float
    best_score_std: float
    metric_name: str
    search_type: str  # 'grid' or 'random'
    total_combinations: int
    total_runtime: float
    algorithm_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_name': self.algorithm_name,
            'search_type': self.search_type,
            'metric_name': self.metric_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_score_std': self.best_score_std,
            'total_combinations': self.total_combinations,
            'total_runtime': self.total_runtime,
            'timestamp': self.timestamp,
            'all_results': [r.to_dict() for r in self.all_results]
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for display."""
        rows = []
        for result in self.all_results:
            row = dict(result.params)
            for metric_name, value in result.metrics.items():
                row[f'{metric_name}_mean'] = value
                row[f'{metric_name}_std'] = result.metrics_std.get(metric_name, 0)
            if isinstance(result.error_analysis, dict):
                if result.error_analysis.get('overall_error_rate_mean') is not None:
                    row['overall_error_rate_mean'] = result.error_analysis.get('overall_error_rate_mean')
                if result.error_analysis.get('overall_error_rate_std') is not None:
                    row['overall_error_rate_std'] = result.error_analysis.get('overall_error_rate_std')
            row['runtime'] = result.runtime
            row['n_runs'] = result.n_runs
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by optimization metric
        sort_col = f'{self.metric_name}_mean'
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=False)

        return df


class HyperparameterSearcher:
    """
    Hyperparameter search engine supporting Grid Search and Random Search.
    Evaluates each combination multiple times for robust results.
    """

    def __init__(self,
                 algorithm_name: str,
                 data: Any,
                 labels: Optional[np.ndarray] = None,
                 val_data: Any = None,
                 val_labels: Optional[np.ndarray] = None,
                 base_params: Optional[Dict[str, Any]] = None,
                 metric: str = 'NMI',
                 n_repeats: int = 3,
                 random_seed: int = 42):
        """
        Initialize the hyperparameter searcher.

        Args:
            algorithm_name: Name of the registered algorithm
            data: Input data (AnnData or numpy array) - used for training
            labels: Ground truth labels (optional)
            val_data: Validation data (optional) - used for evaluation
            val_labels: Validation labels (optional)
            base_params: Base parameters to use (non-optimized params)
            metric: Metric to optimize ('NMI', 'ARI', 'ACC', 'Silhouette')
            n_repeats: Number of repetitions per combination
            random_seed: Base random seed
        """
        self.algorithm_name = algorithm_name
        self.data = data
        self.labels = labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.base_params = base_params or {}
        self.metric = metric
        self.n_repeats = n_repeats
        self.random_seed = random_seed

        # Get data matrix for metrics computation (validation or train)
        target_data = val_data if val_data is not None else data
        if hasattr(target_data, 'X'):
            self.X = target_data.X
            if hasattr(self.X, 'toarray'):
                self.X = self.X.toarray()
        else:
            self.X = target_data

    def _evaluate_params(self,
                         params: Dict[str, Any],
                         progress_callback: Optional[Callable] = None) -> SearchResult:
        """
        Evaluate a single parameter combination with multiple runs.

        Args:
            params: Parameters to evaluate
            progress_callback: Optional progress callback

        Returns:
            SearchResult with aggregated metrics
        """
        all_metrics = []
        all_error_analyses: List[Optional[Dict[str, Any]]] = []
        total_runtime = 0

        # Merge with base params
        full_params = {**self.base_params, **params}
        target_data = self.val_data if self.val_data is not None else self.data
        label_name_map = _infer_label_name_map(target_data)

        for run_id in range(self.n_repeats):
            run_params = full_params.copy()

            # Set different seed for each run
            seed = self.random_seed + run_id
            run_params['random_state'] = seed
            run_params['seed'] = seed

            try:
                # Create and run algorithm
                algorithm = AlgorithmRegistry.create(self.algorithm_name, run_params)

                start_time = time.time()
                # Fit on training data
                algorithm.fit(self.data, self.labels)
                
                # Predict on validation data if available, else training data
                if self.val_data is not None:
                    predicted_labels = algorithm.predict(self.val_data)
                    # For embeddings, some algorithms might not support encoding new data
                    # We try best effort
                    try:
                        if hasattr(algorithm, 'encode'):
                            embeddings = algorithm.encode(self.val_data)
                        elif hasattr(algorithm, 'get_embeddings_for'):
                            embeddings = algorithm.get_embeddings_for(self.val_data)
                        else:
                            embeddings = None
                    except:
                        embeddings = None
                    
                    target_labels = self.val_labels
                else:
                    predicted_labels = algorithm.predict()
                    embeddings = algorithm.get_embeddings()
                    target_labels = self.labels

                run_time = time.time() - start_time

                total_runtime += run_time

                # Compute metrics
                metrics = compute_metrics(
                    labels_true=target_labels,
                    labels_pred=predicted_labels,
                    embeddings=embeddings,
                    data=self.X
                )
                all_metrics.append(metrics)

                # Compute detailed error analysis when ground-truth labels are available.
                run_error_analysis = None
                if target_labels is not None:
                    y_true = np.asarray(target_labels)
                    y_pred = np.asarray(predicted_labels)
                    if len(y_true) == len(y_pred):
                        group_ids, group_key = _extract_group_ids(target_data, len(y_true))
                        y_true_decoded = _decode_labels(y_true, label_name_map)
                        run_error_analysis = compute_error_analysis(
                            labels_true=y_true_decoded,
                            labels_pred=y_pred,
                            group_ids=group_ids
                        )
                        run_error_analysis['group_key'] = group_key
                all_error_analyses.append(run_error_analysis)

            except Exception as e:
                logger.warning(f"Run {run_id} failed for params {params}: {e}")
                # Add placeholder metrics for failed run
                all_metrics.append({
                    'NMI': 0.0, 'ARI': 0.0, 'ACC': 0.0, 'Silhouette': 0.0
                })
                all_error_analyses.append(None)

        # Aggregate metrics
        mean_metrics = {}
        std_metrics = {}

        metric_keys = sorted({key for metric_dict in all_metrics for key in metric_dict.keys()})
        for key in metric_keys:
            if key == 'n_clusters_found':
                continue
            values = []
            for metric_dict in all_metrics:
                value = metric_dict.get(key)
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
                    values.append(float(value))
            if values:
                mean_metrics[key] = float(np.mean(values))
                std_metrics[key] = float(np.std(values))

        error_analysis_summary = _aggregate_error_analyses(all_error_analyses)
        valid_error_analyses = [ea for ea in all_error_analyses if isinstance(ea, dict)]

        return SearchResult(
            params=params,
            metrics=mean_metrics,
            metrics_std=std_metrics,
            runtime=total_runtime,
            n_runs=self.n_repeats,
            all_metrics=all_metrics,
            error_analysis=error_analysis_summary,
            all_error_analyses=valid_error_analyses
        )

    def grid_search(self,
                    param_grid: Dict[str, List[Any]],
                    progress_callback: Optional[Callable] = None) -> SearchSummary:
        """
        Perform exhaustive grid search over parameter combinations.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                Example: {'n_clusters': [5, 10, 15], 'n_pca_components': [10, 20, 30]}
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            SearchSummary with all results and best parameters
        """
        start_time = time.time()

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        total_combinations = len(combinations)

        if progress_callback:
            progress_callback(f"Grid search: {total_combinations} combinations", 0)

        all_results = []
        best_result = None
        best_score = -float('inf')

        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            if progress_callback:
                progress = (idx + 1) / total_combinations
                progress_callback(f"Evaluating {idx + 1}/{total_combinations}: {params}", progress)

            result = self._evaluate_params(params, progress_callback)
            all_results.append(result)

            # Check if best
            score = result.metrics.get(self.metric, 0)
            if score > best_score:
                best_score = score
                best_result = result

        total_runtime = time.time() - start_time

        return SearchSummary(
            all_results=all_results,
            best_params=best_result.params if best_result else {},
            best_score=best_score,
            best_score_std=best_result.metrics_std.get(self.metric, 0) if best_result else 0,
            metric_name=self.metric,
            search_type='grid',
            total_combinations=total_combinations,
            total_runtime=total_runtime,
            algorithm_name=self.algorithm_name
        )

    def random_search(self,
                      param_distributions: Dict[str, Any],
                      n_iter: int = 50,
                      progress_callback: Optional[Callable] = None) -> SearchSummary:
        """
        Perform random search over parameter distributions.

        Args:
            param_distributions: Dictionary mapping parameter names to:
                - List: Sample uniformly from list
                - Tuple (min, max): Sample uniformly from range (int if both int, else float)
                - Tuple (min, max, 'log'): Log-uniform sampling
            n_iter: Number of random combinations to try
            progress_callback: Optional callback(message, progress) for updates

        Returns:
            SearchSummary with all results and best parameters
        """
        start_time = time.time()
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        if progress_callback:
            progress_callback(f"Random search: {n_iter} iterations", 0)

        all_results = []
        best_result = None
        best_score = -float('inf')

        for idx in range(n_iter):
            # Sample parameters
            params = self._sample_random_params(param_distributions)

            if progress_callback:
                progress = (idx + 1) / n_iter
                progress_callback(f"Evaluating {idx + 1}/{n_iter}: {params}", progress)

            result = self._evaluate_params(params, progress_callback)
            all_results.append(result)

            # Check if best
            score = result.metrics.get(self.metric, 0)
            if score > best_score:
                best_score = score
                best_result = result

        total_runtime = time.time() - start_time

        return SearchSummary(
            all_results=all_results,
            best_params=best_result.params if best_result else {},
            best_score=best_score,
            best_score_std=best_result.metrics_std.get(self.metric, 0) if best_result else 0,
            metric_name=self.metric,
            search_type='random',
            total_combinations=n_iter,
            total_runtime=total_runtime,
            algorithm_name=self.algorithm_name
        )

    def _sample_random_params(self, param_distributions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample a random parameter combination from distributions.

        Args:
            param_distributions: Parameter specification
                - List: Uniform choice from list
                - Tuple (min, max): Uniform range
                - Tuple (min, max, 'log'): Log-uniform range
                - Dict with 'type': 'range', 'min', 'max', 'step'
                - Dict with 'type': 'list', 'values': [...]

        Returns:
            Sampled parameter dictionary
        """
        params = {}

        for name, dist in param_distributions.items():
            if isinstance(dist, list):
                # Uniform choice from list
                params[name] = random.choice(dist)

            elif isinstance(dist, tuple):
                if len(dist) == 2:
                    min_val, max_val = dist
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        params[name] = random.randint(min_val, max_val)
                    else:
                        params[name] = random.uniform(float(min_val), float(max_val))
                elif len(dist) == 3 and dist[2] == 'log':
                    # Log-uniform sampling
                    min_val, max_val, _ = dist
                    log_val = random.uniform(np.log(min_val), np.log(max_val))
                    params[name] = np.exp(log_val)

            elif isinstance(dist, dict):
                dist_type = dist.get('type', 'list')
                if dist_type == 'list':
                    params[name] = random.choice(dist['values'])
                elif dist_type == 'range':
                    min_val = dist['min']
                    max_val = dist['max']
                    step = dist.get('step', 1)
                    possible = _generate_range_values(min_val, max_val, step)
                    params[name] = random.choice(possible)

            else:
                # Use as-is (fixed value)
                params[name] = dist

        return params


def generate_param_grid_from_config(hyperparams: List, param_specs: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Generate parameter grid from hyperparameter configs and user specifications.

    Args:
        hyperparams: List of HyperparameterConfig objects
        param_specs: User specifications like {'n_clusters': {'type': 'list', 'values': [5,10,15]}}

    Returns:
        Parameter grid suitable for grid_search
    """
    param_grid = {}

    for spec_name, spec in param_specs.items():
        if spec['type'] == 'list':
            param_grid[spec_name] = spec['values']
        elif spec['type'] == 'range':
            min_val = spec['min']
            max_val = spec['max']
            step = spec.get('step', 1)
            param_grid[spec_name] = _generate_range_values(min_val, max_val, step)

    return param_grid


def save_search_results(summary: SearchSummary, filepath: str):
    """Save search results to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(summary.to_dict(), f, indent=2)


def load_search_results(filepath: str) -> Dict[str, Any]:
    """Load search results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)
