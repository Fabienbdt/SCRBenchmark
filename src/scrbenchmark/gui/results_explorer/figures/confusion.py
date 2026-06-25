"""Confusion-matrix figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="confusion_matrix_detailed",
    name="Detailed Confusion Matrix",
    category="Error Analysis",
    description="Confusion matrix + F1/precision/recall metrics.",
    tags=("confusion", "matrix", "f1"),
  )
)
def render_confusion_matrix_detailed(**ctx):
  return legacy.plot_confusion_matrix_detailed(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    sort_by="n_samples",
  )


@FigureRegistry.register(
  FigureInfo(
    key="confusion_matrix_by_batch",
    name="Confusion by Batch",
    category="Error Analysis",
    description="Confusion matrix by batch.",
    tags=("confusion", "batch"),
  )
)
def render_confusion_matrix_by_batch(**ctx):
  return legacy.plot_confusion_matrix_by_batch(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )
