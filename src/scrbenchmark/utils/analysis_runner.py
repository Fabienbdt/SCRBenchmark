"""
Analysis runner for executing clustering algorithms.

Includes:
- Standard mode: fit and evaluate on the same data
- Benchmark mode: train/val/test split with generalization evaluation
"""

import time
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np

from core.algorithm_registry import AlgorithmRegistry, BaseAlgorithm
from core.config import ParamType
from .metrics import (
    compute_metrics, compute_metrics_by_group, compute_benchmark_metrics,
    compute_generalization_gap, BenchmarkMetrics
)
from .hyperparam_search import HyperparameterSearcher

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Result of a single analysis run."""
    algorithm_name: str
    run_id: int
    labels: np.ndarray
    embeddings: Optional[np.ndarray]
    metrics: Dict[str, float]
    runtime: float
    params: Dict[str, Any]
    true_labels: Optional[np.ndarray] = None  # Aligned ground truth labels
    extra_info: Dict[str, Any] = field(default_factory=dict)  # For PCA components, etc.
    loss_history: List[Dict[str, Any]] = field(default_factory=list)  # Train/validation loss curves
    weight_history: List[Dict[str, Any]] = field(default_factory=list)  # Per-epoch cell-weight stats
    embedding_snapshots: List[Dict[str, Any]] = field(default_factory=list)  # UMAP evolution snapshots
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_name': self.algorithm_name,
            'run_id': self.run_id,
            'n_samples': len(self.labels),
            'metrics': self.metrics,
            'runtime': self.runtime,
            'params': self.params,
            'extra_info': self.extra_info,
            'timestamp': self.timestamp
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple algorithms."""
    results: List[AnalysisResult]
    summary: Dict[str, Dict[str, float]]  # algo -> metric -> mean/std

    def get_best_algorithm(self, metric: str = 'NMI') -> str:
        """Get the algorithm with the best mean score for a metric."""
        best_algo = None
        best_score = -float('inf')

        for algo, stats in self.summary.items():
            if f'{metric}_mean' in stats:
                if stats[f'{metric}_mean'] > best_score:
                    best_score = stats[f'{metric}_mean']
                    best_algo = algo

        return best_algo


class AnalysisRunner:
    """
    Runner for executing and comparing clustering algorithms.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the analysis runner.

        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = Path(output_dir) if output_dir else Path('results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[AnalysisResult] = []

    def _normalize_algo_params(self, algo_name: str, algo_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize historical/aliased parameter names before algorithm execution."""
        normalized = dict(algo_params or {})



        return normalized

    @staticmethod
    def _extract_effective_params(algorithm: BaseAlgorithm) -> Dict[str, Any]:
        """Return runtime-effective params when available, else raw params."""
        if hasattr(algorithm, "get_effective_params"):
            try:
                effective = algorithm.get_effective_params()
                if isinstance(effective, dict):
                    return dict(effective)
            except Exception:
                pass
        return dict(getattr(algorithm, "params", {}) or {})

    def run_algorithm(self,
                     algorithm_name: str,
                     data: Any,
                     labels: Optional[np.ndarray] = None,
                     params: Dict[str, Any] = None,
                     run_id: int = 0,
                     compute_scib_metrics: bool = True,
                     progress_callback: Optional[Callable] = None
                     ) -> AnalysisResult:
        """
        Run a single algorithm.

        Args:
            algorithm_name: Name of the registered algorithm
            data: Input data (AnnData or numpy array)
            labels: Optional ground truth labels
            params: Algorithm parameters
            run_id: Run identifier for multiple runs
            progress_callback: Optional callback for progress updates

        Returns:
            AnalysisResult object
        """
        if progress_callback:
            progress_callback(f"Running {algorithm_name}...")

        # Create algorithm instance
        algorithm = AlgorithmRegistry.create(algorithm_name, params)

        def _as_bool(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return False

        # Time the execution
        start_time = time.time()

        # Fit and predict
        force_true_unsupervised = _as_bool((params or {}).get("force_true_unsupervised", False)) or _as_bool(
            (params or {}).get("true_unsupervised", False)
        )
        fit_labels = None if force_true_unsupervised else labels
        if force_true_unsupervised:
            logger.info(
                "%s: forcing true-unsupervised fit (labels=None) via force_true_unsupervised/true_unsupervised.",
                algorithm_name,
            )
        algorithm.fit(data, fit_labels)
        # Pass data to predict() to ensure labels are computed for all samples
        # Some algorithms (e.g., scMAE) may only store training labels internally
        predicted_labels = algorithm.predict(data)
        
        # Get embeddings: prefer encode(data) to get embeddings for the exact input data
        # This handles cases where fit() might have filtered cells but predict() processes all
        if hasattr(algorithm, 'encode'):
            try:
                embeddings = algorithm.encode(data)
            except Exception as e:
                logger.warning(f"Failed to encode data for {algorithm_name}, falling back to get_embeddings(): {e}")
                embeddings = algorithm.get_embeddings()
        else:
            embeddings = algorithm.get_embeddings()

        runtime = time.time() - start_time

        # Get data for metrics
        if hasattr(data, 'X'):
            X = data.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        # Align labels/data if algorithm filters cells internally
        valid_idx = None
        if hasattr(algorithm, "get_valid_indices"):
            try:
                valid_idx = algorithm.get_valid_indices()
            except Exception:
                valid_idx = None
        if valid_idx is not None:
            valid_idx = np.asarray(valid_idx)
            pred_len = len(predicted_labels) if predicted_labels is not None else 0
            
            # Case 1: Predictions match valid indices (algorithm already returned filtered subset)
            if len(valid_idx) == pred_len:
                if labels is not None and len(labels) != pred_len:
                    # Filter ground truth to match
                    labels = np.asarray(labels)[valid_idx]
                if embeddings is not None and len(embeddings) != pred_len:
                    # Filter embeddings if they don't match
                    if len(embeddings) == len(data): # Embeddings are full
                         embeddings = embeddings[valid_idx]
                if X is not None and len(X) != pred_len:
                    X = X[valid_idx]
            
            # Case 2: Predictions are full (algorithm returned all), but valid_idx is subset
            # We need to filter predictions, embeddings, and labels to the valid subset
            elif pred_len > len(valid_idx):
                logger.info(f"{algorithm_name}: Filtering results to valid indices ({len(valid_idx)}/{pred_len} cells).")
                
                # Filter predictions
                if predicted_labels is not None:
                    predicted_labels = predicted_labels[valid_idx]
                
                # Filter labels
                if labels is not None and len(labels) == pred_len:
                    labels = np.asarray(labels)[valid_idx]
                
                # Filter embeddings
                if embeddings is not None and len(embeddings) == pred_len:
                    embeddings = embeddings[valid_idx]
                
                # Filter data
                if X is not None and len(X) == pred_len:
                    X = X[valid_idx]
            
            else:
                logger.warning(
                    f"{algorithm_name}: valid indices length ({len(valid_idx)}) mismatches "
                    f"predictions ({pred_len}) and cannot be aligned automatically. Skipping alignment."
                )

        # Compute metrics
        # Prepare AnnData for scIB metrics (only if aligned)
        adata_for_metrics = None
        if hasattr(data, 'obs'):
            try:
                if valid_idx is not None:
                    # Use filtered view if predictions align to valid indices
                    if predicted_labels is not None and len(predicted_labels) == len(valid_idx):
                        adata_for_metrics = data[valid_idx].copy()
                    elif predicted_labels is not None and len(predicted_labels) == len(data):
                        adata_for_metrics = data.copy()
                else:
                    if predicted_labels is not None and len(predicted_labels) == len(data):
                        adata_for_metrics = data.copy()
            except Exception:
                adata_for_metrics = None

        try:
            metrics = compute_metrics(
                labels_true=labels,
                labels_pred=predicted_labels,
                embeddings=embeddings,
                data=X,
                adata=adata_for_metrics,
                compute_scib=compute_scib_metrics,
                compute_neighbor_overlap=True
            )
            logger.info(f"Metrics computed for {algorithm_name}: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Error computing metrics for {algorithm_name}: {e}")
            import traceback
            traceback.print_exc()
            metrics = {}  # Return empty metrics on failure but keep the run result

        # Collect extra info (e.g., PCA components used)
        extra_info = {}
        if hasattr(algorithm, 'get_n_components_used'):
            n_comp = algorithm.get_n_components_used()
            if n_comp is not None:
                extra_info['n_pca_components_used'] = n_comp
        if hasattr(algorithm, 'get_explained_variance'):
            exp_var = algorithm.get_explained_variance()
            if exp_var is not None:
                extra_info['cumulative_variance'] = float(np.sum(exp_var))

        # Collect aligned batch labels for batch-colored UMAP diagnostics.
        if hasattr(data, 'obs') and predicted_labels is not None:
            batch_col = None
            for candidate in ['batch', 'Batch', 'tech', 'study', 'dataset', 'donor', 'sample']:
                if candidate in data.obs.columns:
                    batch_col = candidate
                    break

            if batch_col is not None:
                try:
                    batch_values = np.asarray(data.obs[batch_col].astype(str).to_numpy())
                except Exception:
                    batch_values = None

                if batch_values is not None and len(batch_values) > 0:
                    pred_len = len(predicted_labels)
                    batch_aligned = None
                    if valid_idx is not None:
                        if len(valid_idx) == pred_len:
                            batch_aligned = batch_values[valid_idx]
                        elif pred_len == len(batch_values):
                            batch_aligned = batch_values
                    elif pred_len == len(batch_values):
                        batch_aligned = batch_values

                    if batch_aligned is not None and len(batch_aligned) == pred_len:
                        extra_info['batch_labels'] = batch_aligned
                        extra_info['batch_col'] = batch_col
        
        # Collect number of parameters for DL models
        n_params = algorithm.get_num_parameters()
        if n_params is not None:
            extra_info['num_parameters'] = n_params

        # Collect loss history
        loss_history = algorithm.get_loss_history()
        weight_history = algorithm.get_weight_history()

        # Keep training snapshots for optional UMAP-evolution diagnostics.
        embedding_snapshots = algorithm.get_embedding_snapshots()

        result = AnalysisResult(
            algorithm_name=algorithm_name,
            run_id=run_id,
            labels=predicted_labels,
            embeddings=embeddings,
            metrics=metrics,
            runtime=runtime,
            params=self._extract_effective_params(algorithm),
            true_labels=labels,
            extra_info=extra_info,
            loss_history=loss_history,
            weight_history=weight_history,
            embedding_snapshots=embedding_snapshots
        )

        self.results.append(result)

        if progress_callback:
            progress_callback(f"Completed {algorithm_name} in {runtime:.2f}s")

        return result

    def run_comparison(self,
                      algorithm_names: List[str],
                      data: Any,
                      labels: Optional[np.ndarray] = None,
                      params: Dict[str, Dict[str, Any]] = None,
                      n_repeats: int = 1,
                      compute_scib_metrics: bool = True,
                      progress_callback: Optional[Callable] = None
                      ) -> ComparisonResult:
        """
        Run multiple algorithms for comparison.

        Args:
            algorithm_names: List of algorithm names to compare
            data: Input data
            labels: Optional ground truth labels
            params: Dict mapping algorithm names to their parameters
            n_repeats: Number of repetitions per algorithm
            progress_callback: Optional callback for progress updates

        Returns:
            ComparisonResult object
        """
        params = params or {}
        all_results = []

        total_runs = len(algorithm_names) * n_repeats
        current_run = 0

        for algo_name in algorithm_names:
            algo_params = self._normalize_algo_params(algo_name, params.get(algo_name, {}))
            
            # Determine which data to use based on input_type param
            input_type = algo_params.get('input_type', None)

            # Auto-detect from algorithm info if not specified
            if input_type is None:
                algo_info = AlgorithmRegistry.get(algo_name).get_info()
                input_type = getattr(algo_info, 'recommended_data', 'processed')

            # =================================================================
            # GUARDRAIL: Input Data Validation
            # =================================================================
            self._validate_input_data(algo_name, input_type, data)

            # Default to provided data/labels
            algo_data = data
            algo_labels = labels
            
            # ENFORCE STRICT DATA CONSISTENCY
            # Algorithms must use the Common Preprocessing result (filtered cells/genes).
            # Usage of 'raw_original' (unfiltered) is forbidden to prevent index mismatch.
            
            if hasattr(data, 'get_data') and hasattr(data, 'get_original_data'):
                if input_type == 'raw_original':
                    logger.warning(f"Algorithm {algo_name} requested 'raw_original' (unfiltered) data. "
                                 f"Overriding to 'raw_filtered' to enforce common cell filtering consistency.")
                    input_type = 'raw_filtered'

                if input_type == 'raw_filtered' or input_type == 'raw':
                    fetched_data = data.get_raw_filtered_data()
                    # Labels are same as processed since filtering is same
                    algo_labels = data.get_labels(data_source='processed')
                    if fetched_data is not None:
                        algo_data = fetched_data
                    else:
                         logger.warning(f"Raw filtered data requested for {algo_name} but not available. Using default processed data.")
                         algo_data = data.get_data()
                else:
                    # processed (default)
                    algo_data = data.get_data()
                    algo_labels = data.get_labels(data_source='processed')
                    
            elif input_type != 'processed':
                 logger.warning(f"Input type '{input_type}' requested but data object is not a DataHandler. Using provided data.")

            # Log explicit data info
            if hasattr(algo_data, 'shape'):
                shape_str = f"{algo_data.shape[0]} cells x {algo_data.shape[1]} genes"
            else:
                shape_str = "unknown shape"
            
            logger.info(f"Input Data for {algo_name}: {shape_str} | Type: {input_type} | Source: Common Preprocessing")

            for run_id in range(n_repeats):
                current_run += 1

                if progress_callback:
                    progress_callback(
                        f"Running {algo_name} ({run_id + 1}/{n_repeats}) using {input_type} data",
                        current_run / total_runs
                    )

                # Set different seed for each run
                run_params = algo_params.copy()
                # Always set random_state with increment for reproducible but different runs
                base_seed = run_params.get('random_state', run_params.get('seed', 42))
                run_params['random_state'] = base_seed + run_id
                run_params['seed'] = base_seed + run_id  # Some algorithms use 'seed' instead

                try:
                    result = self.run_algorithm(
                        algorithm_name=algo_name,
                        data=algo_data,
                        labels=algo_labels,
                        params=run_params,
                        run_id=run_id,
                        compute_scib_metrics=compute_scib_metrics
                    )
                    
                    # Add input_type info to result metrics or extra info
                    result.extra_info['input_type'] = input_type
                    
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Error running {algo_name}: {e}")
                    if progress_callback:
                        progress_callback(f"Error in {algo_name}: {str(e)}")
                    # Also print to console for debugging
                    import traceback
                    traceback.print_exc()

        # Compute summary statistics
        summary = self._compute_summary(all_results)

        return ComparisonResult(results=all_results, summary=summary)

    def _compute_summary(self, results: List[AnalysisResult]
                        ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for each algorithm."""
        from collections import defaultdict

        # Group results by algorithm
        by_algo = defaultdict(list)
        for result in results:
            by_algo[result.algorithm_name].append(result)

        summary = {}
        for algo_name, algo_results in by_algo.items():
            # Collect all metrics
            metrics_lists = defaultdict(list)
            for result in algo_results:
                for metric_name, value in result.metrics.items():
                    metrics_lists[metric_name].append(value)
                metrics_lists['runtime'].append(result.runtime)

            # Compute statistics
            algo_summary = {}
            for metric_name, values in metrics_lists.items():
                # Skip dict metrics (like ClassWise) that cannot be aggregated
                if any(isinstance(v, dict) for v in values):
                    continue
                values = np.array(values)
                algo_summary[f'{metric_name}_mean'] = float(np.mean(values))
                algo_summary[f'{metric_name}_std'] = float(np.std(values))
                algo_summary[f'{metric_name}_min'] = float(np.min(values))
                algo_summary[f'{metric_name}_max'] = float(np.max(values))

            summary[algo_name] = algo_summary

        return summary

    def save_results(self, filename: str = 'results.json'):
        """Save all results to a JSON file."""
        output_path = self.output_dir / filename

        data = {
            'results': [r.to_dict() for r in self.results],
            'timestamp': datetime.now().isoformat()
        }

        # Custom encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to {output_path}")
        return output_path

    def clear_results(self):
        """Clear all stored results."""
        self.results = []

    def _validate_input_data(self, algo_name: str, input_type: str, data: Any):
        """
        SCIENTIFIC GUARDRAIL: Validate that input data matches algorithm requirements.
        Specifically ensures that RAW data requests actually get integer counts.
        """
        if input_type == 'raw' or input_type == 'raw_filtered':
            # Check if we can get raw data
            raw_available = False
            if hasattr(data, 'get_raw_filtered_data'):
                # DataHandler object
                if data.get_raw_filtered_data() is not None:
                     raw_available = True
            elif hasattr(data, 'raw') and data.raw is not None:
                raw_available = True
            elif hasattr(data, 'layers') and 'original_X' in data.layers:
                raw_available = True
            
            if not raw_available:
                # If we are falling back to .X, strict check if it is integers
                X = data.X if hasattr(data, 'X') else data
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                
                # Check for float values (indicative of normalization)
                if np.issubdtype(X.dtype, np.floating):
                    # Check if they are actually integers stored as floats
                    is_integer = np.allclose(X, np.round(X))
                    if not is_integer:
                        msg = (f"SCIENTIFIC ERROR: Algorithm '{algo_name}' requires RAW counts (integers), "
                               f"but available data appears to be normalized (floats). "
                               f"Please ensure 'original_X' layer is present or use raw .h5ad.")
                        logger.error(msg)
                        raise ValueError(msg)
            
            logger.info(f"Guardrail passed: {algo_name} requested {input_type} and valid raw data is available.")

    def _get_lr_hyperparameters(self, algorithm_name: str) -> List[Any]:
        """Return LR-like hyperparameters exposed by an algorithm."""
        algo_class = AlgorithmRegistry.get(algorithm_name)
        if algo_class is None:
            return []

        try:
            hyperparams = algo_class.get_hyperparameters()
        except Exception:
            return []

        lr_hps = []
        for hp in hyperparams:
            if hp.param_type not in {ParamType.FLOAT, ParamType.INTEGER}:
                continue
            hp_name = str(hp.name).lower()
            if (
                hp_name == "lr"
                or hp_name.endswith("_lr")
                or "learning_rate" in hp_name
            ):
                lr_hps.append(hp)
        return lr_hps

    def _build_lr_candidates(self, hp: Any, current_value: Any, scales: List[float]) -> List[Any]:
        """Build candidate LR values around the current value while respecting bounds."""
        try:
            base = float(current_value)
        except Exception:
            return []

        if not np.isfinite(base):
            return []

        min_val = hp.min_value
        max_val = hp.max_value
        step = hp.step if hp.step not in (None, 0) else None

        candidates: List[Any] = []
        for scale in scales:
            try:
                value = float(base) * float(scale)
            except Exception:
                continue

            if min_val is not None:
                value = max(float(min_val), value)
            if max_val is not None:
                value = min(float(max_val), value)

            if step is not None:
                try:
                    step_f = float(step)
                    if step_f > 0:
                        value = round(value / step_f) * step_f
                except Exception:
                    pass

            if hp.param_type == ParamType.INTEGER:
                value = int(round(value))
            else:
                value = float(value)

            if min_val is not None and value < min_val:
                value = min_val
            if max_val is not None and value > max_val:
                value = max_val

            candidates.append(value)

        # Keep order stable and remove duplicates
        dedup: List[Any] = []
        seen = set()
        for v in candidates:
            if isinstance(v, float):
                key = round(v, 12)
            else:
                key = v
            if key in seen:
                continue
            seen.add(key)
            dedup.append(v)

        return dedup

    def _optimize_learning_rates_on_validation(
        self,
        algorithm_name: str,
        data_train: Any,
        labels_train: Optional[np.ndarray],
        data_val: Any,
        labels_val: Optional[np.ndarray],
        params: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize LR-like hyperparameters using validation data.

        This is applied sequentially for each LR hyperparameter:
        optimize one LR param while keeping others fixed, then continue.
        """
        optimized_params = params.copy()
        optimization_meta: Dict[str, Any] = {}

        enabled = bool(
            optimized_params.get(
                "enable_lr_optimization",
                optimized_params.get("lr_optimization_enabled", True),
            )
        )
        if not enabled or data_val is None:
            return optimized_params, optimization_meta

        lr_hps = self._get_lr_hyperparameters(algorithm_name)
        if not lr_hps:
            return optimized_params, optimization_meta

        scales_raw = optimized_params.get("lr_optimization_scales", [100.0, 10.0, 1.0, 0.1])
        if isinstance(scales_raw, (list, tuple)):
            scales = []
            for s in scales_raw:
                try:
                    s_val = float(s)
                    if s_val > 0:
                        scales.append(s_val)
                except Exception:
                    continue
            if not scales:
                scales = [100.0, 10.0, 1.0, 0.1]
        else:
            scales = [100.0, 10.0, 1.0, 0.1]

        metric_raw = str(optimized_params.get("lr_optimization_metric", "NMI")).strip()
        metric_upper = metric_raw.upper()
        if metric_upper in {"NMI", "ARI", "ACC"}:
            metric_name = metric_upper
        elif metric_upper == "SILHOUETTE":
            metric_name = "Silhouette"
        else:
            metric_name = "NMI"

        if labels_val is None and metric_name in {"NMI", "ARI", "ACC"}:
            metric_name = "Silhouette"

        try:
            n_repeats = int(optimized_params.get("lr_optimization_repeats", 1))
        except Exception:
            n_repeats = 1
        n_repeats = max(1, min(n_repeats, 5))

        try:
            search_seed = int(optimized_params.get("seed", optimized_params.get("random_state", 42)))
        except Exception:
            search_seed = 42

        for hp in lr_hps:
            current_value = optimized_params.get(hp.name, hp.default)
            candidates = self._build_lr_candidates(hp, current_value, scales)
            if len(candidates) < 2:
                continue

            if progress_callback:
                progress_callback(
                    f"[Benchmark] Validation LR optimization ({algorithm_name}:{hp.name})..."
                )

            logger.info(
                f"[Benchmark] LR optimization for {algorithm_name}:{hp.name} "
                f"with candidates={candidates} metric={metric_name}"
            )

            try:
                searcher = HyperparameterSearcher(
                    algorithm_name=algorithm_name,
                    data=data_train,
                    labels=labels_train,
                    val_data=data_val,
                    val_labels=labels_val,
                    base_params=optimized_params,
                    metric=metric_name,
                    n_repeats=n_repeats,
                    random_seed=search_seed,
                )
                summary = searcher.grid_search({hp.name: candidates})
                best_value = summary.best_params.get(hp.name, current_value)
                optimized_params[hp.name] = best_value

                optimization_meta[hp.name] = {
                    "before": current_value,
                    "after": best_value,
                    "metric": metric_name,
                    "best_score": summary.best_score,
                    "candidates": candidates,
                }
                logger.info(
                    f"[Benchmark] Selected {algorithm_name}:{hp.name}={best_value} "
                    f"(was {current_value}, score={summary.best_score:.4f})"
                )
            except Exception as e:
                logger.warning(
                    f"[Benchmark] LR optimization failed for {algorithm_name}:{hp.name}: {e}"
                )

        if optimization_meta:
            optimized_params["_lr_optimization"] = {
                "enabled": True,
                "metric": metric_name,
                "n_repeats": n_repeats,
                "results": optimization_meta,
            }

        return optimized_params, optimization_meta

    # =========================================================================
    # Benchmark Mode Methods
    # =========================================================================

    def run_benchmark(
        self,
        algorithm_name: str,
        data_train: Any,
        data_test: Any,
        labels_train: Optional[np.ndarray] = None,
        labels_test: Optional[np.ndarray] = None,
        data_val: Optional[Any] = None,
        labels_val: Optional[np.ndarray] = None,
        params: Dict[str, Any] = None,
        batch_col: Optional[str] = None,
        run_id: int = 0,
        compute_scib_metrics: bool = True,
        progress_callback: Optional[Callable] = None,
        search_space: Optional[Dict[str, List[Any]]] = None,
        search_n_trials: int = 1
    ) -> 'BenchmarkResult':
        """
        Run algorithm in benchmark mode: fit on train, predict on test.

        This method evaluates the generalization capacity of the algorithm
        by training on one set and evaluating on a held-out test set.
        
        Hyperparameter optimization is disabled in benchmark mode; validation
        is reserved for early stopping / LR scheduling in supported algorithms.

        Args:
            algorithm_name: Name of the registered algorithm
            data_train: Training data (AnnData or numpy array)
            data_test: Test data (AnnData or numpy array)
            labels_train: Ground truth labels for training
            labels_test: Ground truth labels for test
            data_val: Optional validation data
            labels_val: Optional validation labels
            params: Algorithm parameters
            batch_col: Column name for batch info (for group metrics)
            run_id: Run identifier
            progress_callback: Optional callback for progress updates
            search_space: Optional grid for hyperparameter search (ignored in benchmark)
            search_n_trials: Number of trials for random search (ignored in benchmark)

        Returns:
            BenchmarkResult object with train/test metrics
        """
        if progress_callback:
            progress_callback(f"[Benchmark] Training {algorithm_name}...")

        # GUARDRAIL: Check for Import-Time Batch Correction Leakage
        # If batch correction was applied at import (before split), it leaks test info.
        # Check both train and test data for signs of import-stage correction
        is_leaky = False
        if hasattr(data_train, 'uns') and data_train.uns.get('batch_correction', {}).get('stage') == 'import':
            is_leaky = True
        if hasattr(data_test, 'uns') and data_test.uns.get('batch_correction', {}).get('stage') == 'import':
            is_leaky = True
        
        if is_leaky:
            msg = ("⚠️ SCIENTIFIC WARNING: Import-time batch correction detected in Benchmark Mode. "
                   "This constitutes data leakage (transductive setting) as the correction model saw the test set. "
                   "For strict benchmarking, apply batch correction AFTER splitting or use split-aware correction.")
            logger.warning(msg)
            if progress_callback:
                progress_callback(msg)
            # Store leakage flag for result metadata
            params = params.copy() if params else {}
            params['_benchmark_leakage_warning'] = True

        params = params or {}
        
        # Hyperparameter Optimization Step (disabled in benchmark)
        # Validation is reserved for early stopping / LR scheduling in algorithms that support it.
        if search_space and data_val is not None:
            logger.info("Benchmark mode: hyperparameter optimization is disabled; "
                        "validation is reserved for early stopping/LR scheduling.")
            if progress_callback:
                progress_callback("[Benchmark] Hyperparameter optimization skipped (validation reserved for early stopping/LR).")

        # Create algorithm instance
        algorithm = AlgorithmRegistry.create(algorithm_name, params)
        
        # Enforce original-method constraints for benchmark mode
        info = algorithm.get_info()
        
        # Determine if we are out-of-sample BEFORE potential get_raw_view copies
        is_transductive = not getattr(info, "supports_out_of_sample", True)
        is_out_of_sample = (data_test is not None and data_test is not data_train)

        # Preprocessing swapping
        user_input_type = params.get('input_type')
        use_raw = False
        algo_info = algorithm.get_info()
        if user_input_type in ['raw_filtered', 'raw_original']:
            use_raw = True
        elif user_input_type is None and getattr(algo_info, 'recommended_data', 'processed') == 'raw':
            use_raw = True
            
        if use_raw:
            # Validate training data
            self._validate_input_data(algorithm_name, 'raw', data_train)
            logger.info(f"[Benchmark] Using RAW counts for {algorithm_name} (recommended/requested).")
            # Helper to swap X with original_X if available
            def get_raw_view(adata):
                if hasattr(adata, 'layers') and 'original_X' in adata.layers:
                    new_adata = adata.copy()
                    new_adata.X = new_adata.layers['original_X']
                    return new_adata
                elif hasattr(adata, 'raw') and adata.raw is not None:
                     # Fallback to raw if same shape
                     if adata.raw.n_vars == adata.n_vars:
                         new_adata = adata.copy()
                         new_adata.X = adata.raw.X
                         return new_adata
                return adata

            data_train = get_raw_view(data_train)
            data_test = get_raw_view(data_test)
            if data_val is not None:
                data_val = get_raw_view(data_val)
        else:
             logger.info(f"[Benchmark] Using PROCESSED data for {algorithm_name}.")

        # Allow transductive algorithms ONLY if data_test is None or if data_test is basically data_train (full-data fallback)
        if is_transductive and is_out_of_sample:
            raise ValueError(
                f"{algorithm_name} does not support out-of-sample prediction per original methodology. "
                "Use standard mode or exclude this algorithm from benchmark mode."
            )

        # Validation-driven LR optimization for all algorithms exposing LR-like params.
        if data_val is not None:
            params, _ = self._optimize_learning_rates_on_validation(
                algorithm_name=algorithm_name,
                data_train=data_train,
                labels_train=labels_train,
                data_val=data_val,
                labels_val=labels_val,
                params=params,
                progress_callback=progress_callback,
            )

        # Time the training
        start_time = time.time()

        # Recreate algorithm with potentially optimized LR params
        algorithm = AlgorithmRegistry.create(algorithm_name, params)

        # Fit on training data
        algorithm.fit(data_train, labels_train)

        fit_time = time.time() - start_time

        # Predict on training data (for train metrics)
        train_pred = algorithm.predict(data_train)
        train_embeddings = algorithm.get_embeddings()

        # Predict on test data
        predict_start = time.time()
        test_pred = algorithm.predict(data_test)
        predict_time = time.time() - predict_start

        # Get test embeddings if possible (may need to encode test data)
        test_embeddings = self._get_test_embeddings(algorithm, data_test)

        # Predict on validation if available
        val_pred = None
        val_embeddings = None
        if data_val is not None:
            val_pred = algorithm.predict(data_val)
            val_embeddings = self._get_test_embeddings(algorithm, data_val)

        total_time = time.time() - start_time

        # Get batch IDs if available
        batch_train = self._extract_batch_ids(data_train, batch_col)
        batch_test = self._extract_batch_ids(data_test, batch_col)
        batch_val = self._extract_batch_ids(data_val, batch_col) if data_val is not None else None

        # =====================================================================
        # ALIGNMENT GUARDRAIL: Handle Internal Cell Filtering
        # =====================================================================
        # If the algorithm filtered cells internally (e.g. scDeepCluster dropping zeros),
        # we MUST align labels, predictions, and batch IDs for evaluation.
        valid_idx = None
        if hasattr(algorithm, "get_valid_indices"):
            try:
                valid_idx = algorithm.get_valid_indices()
            except Exception:
                valid_idx = None
        
        if valid_idx is not None:
            valid_idx = np.asarray(valid_idx)
            # Alignment for training results
            if len(valid_idx) != len(labels_train) and labels_train is not None:
                 logger.info(f"Aligning training labels for {algorithm_name} ({len(labels_train)} -> {len(valid_idx)})")
                 labels_train = np.asarray(labels_train)[valid_idx]
            
            if len(valid_idx) != len(train_pred) and train_pred is not None:
                 logger.info(f"Aligning training predictions for {algorithm_name} ({len(train_pred)} -> {len(valid_idx)})")
                 train_pred = np.asarray(train_pred)[valid_idx]
            
            if batch_train is not None and len(valid_idx) != len(batch_train):
                 logger.info(f"Aligning training batch IDs for {algorithm_name} ({len(batch_train)} -> {len(valid_idx)})")
                 batch_train = np.asarray(batch_train)[valid_idx]
            
            if train_embeddings is not None and len(valid_idx) != len(train_embeddings):
                 # This shouldn't happen if train_embeddings matches valid_idx already
                 logger.info(f"Aligning training embeddings for {algorithm_name} ({len(train_embeddings)} -> {len(valid_idx)})")
                 train_embeddings = np.asarray(train_embeddings)[valid_idx]

        # Extract data matrices for metrics (e.g., NNO)
        def _extract_matrix(d):
            if d is None:
                return None
            if hasattr(d, 'X'):
                X = d.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                return X
            return d

        train_data_matrix = _extract_matrix(data_train)
        test_data_matrix = _extract_matrix(data_test)
        val_data_matrix = _extract_matrix(data_val)

        # Align training data matrix if internal filtering occurred
        if valid_idx is not None and train_data_matrix is not None:
            if len(train_data_matrix) != len(valid_idx):
                if len(train_data_matrix) > len(valid_idx):
                    train_data_matrix = train_data_matrix[valid_idx]
                else:
                    logger.warning(
                        f"Train data matrix length ({len(train_data_matrix)}) < valid indices ({len(valid_idx)}); "
                        "skipping alignment for data matrix."
                    )

        # Prepare AnnData for scIB metrics (aligned)
        adata_train_metrics = None
        adata_test_metrics = None
        adata_val_metrics = None
        if hasattr(data_train, 'obs') and train_pred is not None:
            try:
                if valid_idx is not None and len(train_pred) == len(valid_idx):
                    adata_train_metrics = data_train[valid_idx].copy()
                elif len(train_pred) == len(data_train):
                    adata_train_metrics = data_train.copy()
            except Exception:
                adata_train_metrics = None
        if hasattr(data_test, 'obs') and test_pred is not None:
            try:
                if len(test_pred) == len(data_test):
                    adata_test_metrics = data_test.copy()
            except Exception:
                adata_test_metrics = None
        if data_val is not None and hasattr(data_val, 'obs') and val_pred is not None:
            try:
                if len(val_pred) == len(data_val):
                    adata_val_metrics = data_val.copy()
            except Exception:
                adata_val_metrics = None

        # Compute comprehensive metrics
        benchmark_metrics = compute_benchmark_metrics(
            labels_true_train=labels_train,
            labels_pred_train=train_pred,
            labels_true_test=labels_test,
            labels_pred_test=test_pred,
            group_ids_train=batch_train,
            group_ids_test=batch_test,
            labels_true_val=labels_val,
            labels_pred_val=val_pred,
            embeddings_train=train_embeddings,
            embeddings_test=test_embeddings,
            embeddings_val=val_embeddings,
            data_train=train_data_matrix,
            data_test=test_data_matrix,
            data_val=val_data_matrix,
            adata_train=adata_train_metrics,
            adata_test=adata_test_metrics,
            adata_val=adata_val_metrics,
            compute_scib=compute_scib_metrics,
        )

        # Collect loss history
        loss_history = algorithm.get_loss_history()
        weight_history = algorithm.get_weight_history()

        # Keep training snapshots for optional UMAP-evolution diagnostics.
        embedding_snapshots = algorithm.get_embedding_snapshots()

        # Create result object
        result = BenchmarkResult(
            algorithm_name=algorithm_name,
            run_id=run_id,
            train_labels=train_pred,
            test_labels=test_pred,
            val_labels=val_pred,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            benchmark_metrics=benchmark_metrics,
            fit_time=fit_time,
            predict_time=predict_time,
            total_time=total_time,
            params=self._extract_effective_params(algorithm),
            n_train=len(labels_train) if labels_train is not None else 0,
            n_test=len(labels_test) if labels_test is not None else 0,
            n_val=len(labels_val) if labels_val is not None else 0,
            train_true_labels=labels_train,
            test_true_labels=labels_test,
            val_true_labels=labels_val,
            train_batch_ids=batch_train,
            test_batch_ids=batch_test,
            val_batch_ids=batch_val,
            loss_history=loss_history,
            weight_history=weight_history,
            embedding_snapshots=embedding_snapshots
        )

        if progress_callback:
            train_nmi = benchmark_metrics.train_metrics.get('NMI', 0)
            test_nmi = benchmark_metrics.test_metrics.get('NMI', 0)
            progress_callback(
                f"[Benchmark] {algorithm_name} complete: "
                f"Train NMI={train_nmi:.3f}, Test NMI={test_nmi:.3f}"
            )

        return result

    def run_benchmark_comparison(
        self,
        algorithm_names: List[str],
        data_train: Any,
        data_test: Any,
        labels_train: Optional[np.ndarray] = None,
        labels_test: Optional[np.ndarray] = None,
        data_val: Optional[Any] = None,
        labels_val: Optional[np.ndarray] = None,
        params: Dict[str, Dict[str, Any]] = None,
        batch_col: Optional[str] = None,
        n_repeats: int = 1,
        compute_scib_metrics: bool = True,
        progress_callback: Optional[Callable] = None,
        search_spaces: Dict[str, Dict[str, List[Any]]] = None
    ) -> 'BenchmarkComparisonResult':
        """
        Run multiple algorithms in benchmark mode for comparison.

        Args:
            algorithm_names: List of algorithm names to compare
            data_train: Training data
            data_test: Test data
            labels_train: Training labels
            labels_test: Test labels
            data_val: Optional validation data
            labels_val: Optional validation labels
            params: Dict mapping algorithm names to their parameters
            batch_col: Column name for batch info
            n_repeats: Number of repetitions per algorithm
            progress_callback: Optional callback for progress updates
            search_spaces: Optional dict mapping algo names to search spaces

        Returns:
            BenchmarkComparisonResult with all results and summary
        """
        params = params or {}
        search_spaces = search_spaces or {}
        all_results = []

        total_runs = len(algorithm_names) * n_repeats
        current_run = 0

        for algo_name in algorithm_names:
            algo_params = self._normalize_algo_params(algo_name, params.get(algo_name, {}))
            algo_search_space = search_spaces.get(algo_name, None)

            for run_id in range(n_repeats):
                current_run += 1

                if progress_callback:
                    progress_callback(
                        f"[Benchmark] Running {algo_name} ({run_id + 1}/{n_repeats})",
                        current_run / total_runs
                    )

                # Set different seed for each run
                run_params = algo_params.copy()
                base_seed = run_params.get('random_state', run_params.get('seed', 42))
                run_params['random_state'] = base_seed + run_id
                run_params['seed'] = base_seed + run_id

                try:
                    # Pass search space only for the first run if multiple repeats? 
                    # Ideally we optimize once per repeat or once globally. 
                    # For stability, we can re-optimize or just use the same. 
                    # If random seed changes, optimization might change. Let's re-optimize per run to be robust.
                    
                    result = self.run_benchmark(
                        algorithm_name=algo_name,
                        data_train=data_train,
                        data_test=data_test,
                        labels_train=labels_train,
                        labels_test=labels_test,
                        data_val=data_val,
                        labels_val=labels_val,
                        params=run_params,
                        batch_col=batch_col,
                        run_id=run_id,
                        compute_scib_metrics=compute_scib_metrics,
                        progress_callback=None,  # Avoid nested callbacks
                        search_space=algo_search_space,
                        search_n_trials=10 # Default trial count for random search
                    )
                    all_results.append(result)

                except Exception as e:
                    logger.error(f"Error running {algo_name} in benchmark mode: {e}")
                    if progress_callback:
                        progress_callback(f"Error in {algo_name}: {str(e)}")
                    import traceback
                    traceback.print_exc()

        # Compute summary statistics
        summary = self._compute_benchmark_summary(all_results)

        return BenchmarkComparisonResult(
            results=all_results,
            summary=summary
        )

    def _get_test_embeddings(
        self,
        algorithm: BaseAlgorithm,
        data: Any
    ) -> Optional[np.ndarray]:
        """
        Get embeddings for test data if the algorithm supports it.

        Some algorithms can encode new data (autoencoders), while others
        cannot (e.g., Leiden which is based on graph structure).
        """
        # Try to get embeddings by predicting with data
        # This works if predict() also computes embeddings
        try:
            if hasattr(algorithm, 'encode') or hasattr(algorithm, 'get_embeddings_for'):
                # Algorithm has explicit encoding method
                if hasattr(algorithm, 'encode'):
                    return algorithm.encode(data)
                return algorithm.get_embeddings_for(data)
        except Exception as e:
            logger.debug(f"Could not get test embeddings: {e}")

        return None

    def _extract_batch_ids(
        self,
        data: Any,
        batch_col: Optional[str]
    ) -> Optional[np.ndarray]:
        """Extract batch IDs from data if available."""
        if batch_col is None:
            return None

        if hasattr(data, 'obs') and batch_col in data.obs.columns:
            return data.obs[batch_col].values

        return None

    def _compute_benchmark_summary(
        self,
        results: List['BenchmarkResult']
    ) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics for benchmark results."""
        from collections import defaultdict

        by_algo = defaultdict(list)
        for result in results:
            by_algo[result.algorithm_name].append(result)

        summary = {}
        for algo_name, algo_results in by_algo.items():
            metrics_lists = defaultdict(list)

            for result in algo_results:
                # Train metrics
                for m, v in result.benchmark_metrics.train_metrics.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        metrics_lists[f'train_{m}'].append(v)

                # Test metrics
                for m, v in result.benchmark_metrics.test_metrics.items():
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        metrics_lists[f'test_{m}'].append(v)

                # Generalization gap
                if result.benchmark_metrics.generalization_gap:
                    for m, v in result.benchmark_metrics.generalization_gap.items():
                        if isinstance(v, (int, float)) and not np.isnan(v):
                            metrics_lists[m].append(v)

                # Timing
                metrics_lists['fit_time'].append(result.fit_time)
                metrics_lists['predict_time'].append(result.predict_time)

            # Compute statistics
            algo_summary = {}
            for metric_name, values in metrics_lists.items():
                if values:
                    values = np.array(values)
                    algo_summary[f'{metric_name}_mean'] = float(np.mean(values))
                    algo_summary[f'{metric_name}_std'] = float(np.std(values))
                    algo_summary[f'{metric_name}_min'] = float(np.min(values))
                    algo_summary[f'{metric_name}_max'] = float(np.max(values))

            summary[algo_name] = algo_summary

        return summary


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run (train/test split)."""
    algorithm_name: str
    run_id: int
    train_labels: np.ndarray
    test_labels: np.ndarray
    val_labels: Optional[np.ndarray]
    train_embeddings: Optional[np.ndarray]
    test_embeddings: Optional[np.ndarray]
    benchmark_metrics: BenchmarkMetrics
    fit_time: float
    predict_time: float
    total_time: float
    params: Dict[str, Any]
    n_train: int
    n_test: int
    n_val: int
    # Ground truth labels for confusion matrix analysis
    train_true_labels: Optional[np.ndarray] = None
    test_true_labels: Optional[np.ndarray] = None
    val_true_labels: Optional[np.ndarray] = None
    # Batch IDs for per-batch confusion matrix analysis
    train_batch_ids: Optional[np.ndarray] = None
    test_batch_ids: Optional[np.ndarray] = None
    val_batch_ids: Optional[np.ndarray] = None
    loss_history: List[Dict[str, Any]] = field(default_factory=list)  # Train/validation loss curves
    weight_history: List[Dict[str, Any]] = field(default_factory=list)  # Per-epoch cell-weight stats
    embedding_snapshots: List[Dict[str, Any]] = field(default_factory=list)  # UMAP evolution snapshots
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'algorithm_name': self.algorithm_name,
            'run_id': self.run_id,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_val': self.n_val,
            'train_metrics': self.benchmark_metrics.train_metrics,
            'test_metrics': self.benchmark_metrics.test_metrics,
            'val_metrics': self.benchmark_metrics.val_metrics,
            'generalization_gap': self.benchmark_metrics.generalization_gap,
            'test_by_group': self.benchmark_metrics.test_by_group,
            'error_analysis': self.benchmark_metrics.error_analysis,
            'fit_time': self.fit_time,
            'predict_time': self.predict_time,
            'total_time': self.total_time,
            'params': self.params,
            'timestamp': self.timestamp
        }


@dataclass
class BenchmarkComparisonResult:
    """Result of comparing multiple algorithms in benchmark mode."""
    results: List[BenchmarkResult]
    summary: Dict[str, Dict[str, float]]

    def get_best_algorithm(
        self,
        metric: str = 'NMI',
        split: str = 'test'
    ) -> str:
        """Get the algorithm with the best mean score for a metric."""
        best_algo = None
        best_score = -float('inf')

        metric_key = f'{split}_{metric}_mean'

        for algo, stats in self.summary.items():
            if metric_key in stats:
                if stats[metric_key] > best_score:
                    best_score = stats[metric_key]
                    best_algo = algo

        return best_algo

    def get_generalization_ranking(self, metric: str = 'NMI') -> List[Tuple[str, float]]:
        """
        Rank algorithms by generalization ability (smallest gap is best).

        Returns list of (algorithm_name, gap) tuples sorted by gap (ascending).
        """
        gap_key = f'{metric}_gap_mean'
        ranking = []

        for algo, stats in self.summary.items():
            if gap_key in stats:
                ranking.append((algo, stats[gap_key]))

        # Sort by gap (ascending - smaller is better)
        ranking.sort(key=lambda x: x[1])
        return ranking

    def to_dataframe(self) -> 'pd.DataFrame':
        """Convert summary to pandas DataFrame for easy analysis."""
        import pandas as pd

        rows = []
        for algo, stats in self.summary.items():
            row = {'algorithm': algo}
            row.update(stats)
            rows.append(row)

        return pd.DataFrame(rows)
