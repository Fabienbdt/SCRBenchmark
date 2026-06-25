"""Results Explorer package.

Compatibility exports are kept here because older analysis flows import this
package as a module (`from gui import results_explorer as rex`) and call many
helpers directly on `rex.*`.
"""

from .constants import (
  ALGO_COLORS,
  ALGO_DISPLAY_NAMES,
  FIGURE_CAPTIONS,
  FIGURE_CATEGORY_MAP,
  FIGURE_TYPES,
)
from .legacy import (
  HAS_CLD,
  analyze_celltype_errors_by_batch_fig,
  analyze_celltype_errors_fig,
  get_available_metrics,
  list_umap_diagnostic_entries,
  perform_statistical_test,
  plot_algorithm_comparison,
  plot_batch_composition,
  plot_batch_generalization_fig,
  plot_confusion_matrix_by_batch,
  plot_confusion_matrix_detailed,
  plot_confusion_patterns_fig,
  plot_error_rate_by_batch,
  plot_generalization_gap_boxplot,
  plot_generalization_gap_combined,
  plot_generalization_gap_heatmap_fig,
  plot_metrics_heatmap,
  plot_runtime_comparison,
  plot_saved_umap_gallery,
  plot_test_metrics_by_batch,
  plot_train_vs_test_comparison,
  plot_umap_diagnostic_from_results,
)
from .loader import aggregate_metrics
from .page import render_results_explorer_page
from .tables import (
  create_metrics_by_celltype_table,
  create_summary_dataframe,
  export_publication_ready,
)

__all__ = [
  "ALGO_COLORS",
  "ALGO_DISPLAY_NAMES",
  "FIGURE_CAPTIONS",
  "FIGURE_CATEGORY_MAP",
  "FIGURE_TYPES",
  "HAS_CLD",
  "aggregate_metrics",
  "analyze_celltype_errors_by_batch_fig",
  "analyze_celltype_errors_fig",
  "create_metrics_by_celltype_table",
  "create_summary_dataframe",
  "export_publication_ready",
  "get_available_metrics",
  "list_umap_diagnostic_entries",
  "perform_statistical_test",
  "plot_algorithm_comparison",
  "plot_batch_composition",
  "plot_batch_generalization_fig",
  "plot_confusion_matrix_by_batch",
  "plot_confusion_matrix_detailed",
  "plot_confusion_patterns_fig",
  "plot_error_rate_by_batch",
  "plot_generalization_gap_boxplot",
  "plot_generalization_gap_combined",
  "plot_generalization_gap_heatmap_fig",
  "plot_metrics_heatmap",
  "plot_runtime_comparison",
  "plot_saved_umap_gallery",
  "plot_test_metrics_by_batch",
  "plot_train_vs_test_comparison",
  "plot_umap_diagnostic_from_results",
  "render_results_explorer_page",
]
