"""Shared constants for the Results Explorer package."""

from gui.constants import ALGO_COLORS as SHARED_ALGO_COLORS
from gui.constants import ALGO_DISPLAY_NAMES as SHARED_ALGO_DISPLAY_NAMES

ALGO_COLORS = dict(SHARED_ALGO_COLORS)
ALGO_DISPLAY_NAMES = dict(SHARED_ALGO_DISPLAY_NAMES)

FIGURE_TYPES = {
  'algorithm_comparison': {
    'name': 'Algorithm Comparison by Condition',
    'description': 'Grouped bar plot comparing metrics for each algorithm across conditions.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'metrics_heatmap': {
    'name': 'Metrics Heatmap',
    'description': 'Heatmap showing performance (NMI, ARI, ACC) by algorithm and condition.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'generalization_gap': {
    'name': 'Generalization Gap',
    'description': 'Boxplot of the train-test gap for each algorithm.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_train_test': True,
  },
  'generalization_gap_heatmap': {
    'name': 'Generalization Gap Heatmap',
    'description': 'Detailed heatmap with exact gap values by condition.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_train_test': True,
  },
  'batch_generalization': {
    'name': 'Inter-Batch Generalization',
    'description': 'Generalization matrix between training and test batches.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_batch_data': True,
  },
  'test_metrics_by_batch': {
    'name': 'Test Metrics by Batch',
    'description': 'Performance of each algorithm on each test batch individually (NMI, ARI, ACC).',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_train_test': True,
  },
  'saved_umap_gallery': {
    'name': 'Saved UMAP Gallery',
    'description': 'Display UMAP PNGs already generated in result folders (figures/umap_*).',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'umap_evolution': {
    'name': 'UMAP Evolution',
    'description': 'Display UMAP evolution PNGs generated during training (figures/umap_evolution_*).',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'umap_diagnostic': {
    'name': 'UMAP Diagnostic (raw data)',
    'description': 'Rebuild a 2x2 UMAP view (true/predicted/batch/errors) from labels and h5ad files.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'celltype_errors': {
    'name': 'Errors by Cell Type',
    'description': 'Heatmap of error rate for each cell type.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_error_analysis': True,
  },
  'celltype_errors_by_batch': {
    'name': 'Errors by Cell Type and Batch',
    'description': 'Heatmap of error rate by cell type and test batch.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_error_analysis': True,
    'requires_batch_data': True,
  },
  'confusion_patterns': {
    'name': 'Confusion Patterns',
    'description': 'Top confusions between cell types.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_error_analysis': True,
  },
  'confusion_matrix_detailed': {
    'name': 'Confusion Matrix + F1/Precision/Recall',
    'description': 'Detailed confusion matrix by algorithm with F1, precision, and recall by cell type.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'metrics_by_celltype_table': {
    'name': 'F1/Precision/Recall Table by Type',
    'description': 'Summary table of metrics (F1, precision, recall) by algorithm and cell type.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'error_rate_by_batch': {
    'name': 'Error Rate by Batch',
    'description': 'Heatmap of error rate for each test batch.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_error_analysis': True,
  },
  'confusion_matrix_by_batch': {
    'name': 'Confusion Matrix by Batch',
    'description': 'Detailed confusion matrix for each test batch.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'runtime_comparison': {
    'name': 'Runtime Comparison',
    'description': 'Bar plot comparing algorithm runtimes.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'summary_table': {
    'name': 'Summary Table',
    'description': 'Table with all mean metrics ± standard deviation.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'statistical_test': {
    'name': 'Significance Test',
    'description': 'Statistical tests (Wilcoxon/T-test) between algorithms or conditions.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'train_vs_test': {
    'name': 'Train vs Test Diagnostic',
    'description': 'Scatter plot comparing train and test performance to diagnose overfitting.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_train_test': True,
  },
  'generalization_gap_combined': {
    'name': 'Generalization Gap (3 metrics)',
    'description': 'Combined NMI/ARI/ACC boxplot showing generalization gap (train-test) by algorithm.',
    'requires_conditions': True,
    'requires_algorithms': True,
    'requires_train_test': True,
  },
  'publication_export': {
    'name': 'Publication Export (LaTeX)',
    'description': 'Generate a publication-ready formatted table.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'loss_curves': {
    'name': 'Loss Curves',
    'description': 'Training loss curves by algorithm and run.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
  'batch_composition': {
    'name': 'Train/Test Batch Composition',
    'description': 'Batch distribution in train/test splits to identify learning bias.',
    'requires_conditions': True,
    'requires_algorithms': True,
  },
}

FIGURE_CATEGORY_MAP = {
  'algorithm_comparison': 'Comparison',
  'metrics_heatmap': 'Comparison',
  'summary_table': 'Comparison',
  'train_vs_test': 'Generalization',
  'generalization_gap': 'Generalization',
  'generalization_gap_heatmap': 'Generalization',
  'generalization_gap_combined': 'Generalization',
  'batch_generalization': 'Batch',
  'test_metrics_by_batch': 'Batch',
  'batch_composition': 'Batch',
  'saved_umap_gallery': 'UMAP',
  'umap_evolution': 'UMAP',
  'umap_diagnostic': 'UMAP',
  'celltype_errors': 'Error Analysis',
  'celltype_errors_by_batch': 'Error Analysis',
  'error_rate_by_batch': 'Error Analysis',
  'confusion_patterns': 'Error Analysis',
  'confusion_matrix_detailed': 'Error Analysis',
  'confusion_matrix_by_batch': 'Error Analysis',
  'runtime_comparison': 'Runtime',
  'loss_curves': 'Runtime',
  'statistical_test': 'Statistics',
  'metrics_by_celltype_table': 'Statistics',
  'publication_export': 'Export',
}

FIGURE_CAPTIONS = {
  'algorithm_comparison': (
    "Interpretation: each bar shows the mean performance of an algorithm for one condition. "
    "Error bars show variability across repeated runs. "
    "NMI/ARI/ACC values near 1 indicate clustering that is close to biological annotations."
  ),
  'metrics_heatmap': (
    "Interpretation: each cell shows the mean performance. "
    "Green is better performance, yellow is intermediate, red is lower. "
    "Use this view to quickly spot strong algorithm-condition combinations."
  ),
  'generalization_gap': (
    "Interpretation: a positive gap means the model performs better on train than test (possible overfitting). "
    "For robust biological conclusions, prefer algorithms with gap close to 0. "
    "Definition: gap = train performance - test performance."
  ),
  'generalization_gap_heatmap': (
    "Interpretation: detailed condition-wise gap view. "
    "Red indicates stronger overfitting, near-white indicates better generalization, "
    "green means test > train (rare)."
  ),
  'saved_umap_gallery': (
    "Interpretation: gallery of UMAP figures already exported during benchmarking. "
    "Each panel corresponds to a condition, algorithm, and split (train/test/val/full). "
    "This view does not recompute UMAP; it reloads existing PNG files."
  ),
  'umap_evolution': (
    "Interpretation: evolution snapshots across training epochs, exported as static PNGs. "
    "Use this to diagnose representation stability and phase transitions."
  ),
  'umap_diagnostic': (
    "Interpretation: recomputed 2x2 diagnostic view from the selected run. "
    "Top-left: true biological labels; top-right: predicted clusters; "
    "bottom-left: batch structure; bottom-right: misclassified cells. "
    "Useful to separate biological structure from batch-driven structure."
  ),
  'celltype_errors': (
    "Interpretation: each cell shows error rate (1 - recall) for one cell type. "
    "Higher values indicate harder cell populations. "
    "Use this to identify biologically ambiguous populations. "
    "Available in standard mode (ClassWise) and benchmark mode (error_analysis)."
  ),
  'celltype_errors_by_batch': (
    "Interpretation: each heatmap shows error rate (1-accuracy) by cell type within a test batch. "
    "Rows are sorted by increasing number of cells. "
    "Each cell also reports n (mean number of cells in this cell type × batch)."
  ),
  'confusion_patterns': (
    "Interpretation: top cell-type confusion pairs. "
    "'alpha -> beta: 50' means 50 alpha cells were predicted as beta. "
    "These confusions may reflect biological similarity or model bias."
  ),
  'confusion_matrix_detailed': (
    "Interpretation: detailed confusion matrix for each algorithm. "
    "Rows are true labels, columns are predicted labels, values are cell counts. "
    "Diagonal cells are correct predictions; off-diagonal cells are errors. "
    "F1 combines precision and recall."
  ),
  'runtime_comparison': (
    "Interpretation: mean runtime per algorithm. "
    "Deep learning methods are often slower but may improve biological resolution. "
    "Use this to balance runtime vs performance."
  ),
  'train_vs_test': (
    "Interpretation: each point is one algorithm/condition pair. "
    "Points near the diagonal indicate stable generalization; points above indicate train > test."
  ),
  'generalization_gap_combined': (
    "Interpretation: combined boxplots of gap (train - test) across metrics. "
    "Positive gap suggests overfitting; near-zero gap indicates better generalization."
  ),
  'batch_generalization': (
    "Interpretation: each cell is mean ACC for a model trained on a source batch and evaluated on a target batch. "
    "Color scale summarizes generalization strength across batch pairs."
  ),
  'statistical_test': (
    "Interpretation: Mann-Whitney U test between two groups. "
    "P-values below 0.05 indicate statistically significant differences."
  ),
  'error_rate_by_batch': (
    "Interpretation: each cell shows error rate (1-accuracy) for one test batch. "
    "Use this to identify algorithms robust to batch effects."
  ),
  'confusion_matrix_by_batch': (
    "Interpretation: confusion matrices stratified by test batch. "
    "Useful to detect batch-specific misclassification patterns."
  ),
  'loss_curves': (
    "Interpretation: curves show loss evolution during training. "
    "A smooth decrease indicates stable optimization; plateaus indicate convergence."
  ),
  'batch_composition': (
    "Interpretation: stacked bars summarize global batch composition per split. "
    "Heatmaps show cell-type × batch composition and reveal potential sampling bias. "
    "Balanced train/test design generally improves robust biological conclusions."
  ),
}

__all__ = [
  "ALGO_COLORS",
  "ALGO_DISPLAY_NAMES",
  "FIGURE_TYPES",
  "FIGURE_CAPTIONS",
  "FIGURE_CATEGORY_MAP",
]
