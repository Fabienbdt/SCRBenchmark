"""Batch analysis figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="batch_generalization",
    name="Inter-Batch Generalization",
    category="Batch",
    description="Train/test matrix across batches.",
    tags=("batch", "generalization"),
  )
)
def render_batch_generalization(**ctx):
  return legacy.plot_batch_generalization_fig(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )


@FigureRegistry.register(
  FigureInfo(
    key="test_metrics_by_batch",
    name="Test Metrics by Batch",
    category="Batch",
    description="Test scores by batch.",
    tags=("batch", "test"),
  )
)
def render_test_metrics_by_batch(**ctx):
  return legacy.plot_test_metrics_by_batch(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )


@FigureRegistry.register(
  FigureInfo(
    key="error_rate_by_batch",
    name="Error Rate by Batch",
    category="Batch",
    description="Error heatmap by batch.",
    tags=("batch", "error"),
  )
)
def render_error_rate_by_batch(**ctx):
  return legacy.plot_error_rate_by_batch(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )


@FigureRegistry.register(
  FigureInfo(
    key="batch_composition",
    name="Train/Test Batch Composition",
    category="Batch",
    description="Batch distribution by split.",
    tags=("batch", "composition"),
  )
)
def render_batch_composition(**ctx):
  return legacy.plot_batch_composition(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )
