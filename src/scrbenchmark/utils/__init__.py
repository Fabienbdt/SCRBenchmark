# Utils module
from .data_handler import DataHandler
from .analysis_runner import AnalysisRunner, BenchmarkResult, BenchmarkComparisonResult
from .metrics import (
    compute_metrics,
    compute_metrics_by_group,
    compute_benchmark_metrics,
    compute_generalization_gap,
    BenchmarkMetrics
)
from .pca_utils import (
    find_elbow_cattell,
    find_elbow_second_derivative,
    compute_optimal_pca_components,
    get_pca_with_auto_components
)
from .dataset_splitter import (
    DatasetSplitter,
    BenchmarkPreprocessor,
    SplitResult,
    get_batch_column,
    list_batches
)

__all__ = [
    'DataHandler',
    'AnalysisRunner',
    'BenchmarkResult',
    'BenchmarkComparisonResult',
    'compute_metrics',
    'compute_metrics_by_group',
    'compute_benchmark_metrics',
    'compute_generalization_gap',
    'BenchmarkMetrics',
    'find_elbow_cattell',
    'find_elbow_second_derivative',
    'compute_optimal_pca_components',
    'get_pca_with_auto_components',
    'DatasetSplitter',
    'BenchmarkPreprocessor',
    'SplitResult',
    'get_batch_column',
    'list_batches'
]
