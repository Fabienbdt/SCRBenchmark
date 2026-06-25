"""Error analysis figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="celltype_errors",
    name="Cell-Type Errors",
    category="Error Analysis",
    description="Error heatmap by cell type.",
    tags=("error", "celltype"),
  )
)
def render_celltype_errors(**ctx):
  return legacy.analyze_celltype_errors_fig(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    sort_by="n_samples",
  )


@FigureRegistry.register(
  FigureInfo(
    key="celltype_errors_by_batch",
    name="Cell-Type × Batch Errors",
    category="Error Analysis",
    description="Error heatmap by cell type and batch.",
    tags=("error", "celltype", "batch"),
  )
)
def render_celltype_errors_by_batch(**ctx):
  return legacy.analyze_celltype_errors_by_batch_fig(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    sort_by="n_samples",
  )


@FigureRegistry.register(
  FigureInfo(
    key="confusion_patterns",
    name="Confusion Patterns",
    category="Error Analysis",
    description="Top confusions between cell types.",
    tags=("confusion", "error"),
  )
)
def render_confusion_patterns(**ctx):
  return legacy.plot_confusion_patterns_fig(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )
