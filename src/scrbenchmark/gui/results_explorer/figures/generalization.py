"""Generalization-related figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="generalization_gap",
    name="Generalization Gap",
    category="Generalization",
    description="Train-test gap per algorithm.",
    tags=("train", "test", "gap"),
  )
)
def render_generalization_gap(**ctx):
  return legacy.plot_generalization_gap_boxplot(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    ctx.get("metric", "NMI"),
    show_cld=ctx.get("show_cld", False),
  )


@FigureRegistry.register(
  FigureInfo(
    key="generalization_gap_heatmap",
    name="Generalization Gap Heatmap",
    category="Generalization",
    description="Heatmap of train-test gap.",
    tags=("train", "test", "heatmap"),
  )
)
def render_generalization_gap_heatmap(**ctx):
  return legacy.plot_generalization_gap_heatmap_fig(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    ctx.get("metric", "NMI"),
  )


@FigureRegistry.register(
  FigureInfo(
    key="train_vs_test",
    name="Train vs Test Diagnostic",
    category="Generalization",
    description="Train/test comparison.",
    tags=("train", "test", "diagnostic"),
  )
)
def render_train_vs_test(**ctx):
  return legacy.plot_train_vs_test_comparison(
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )


@FigureRegistry.register(
  FigureInfo(
    key="generalization_gap_combined",
    name="Combined Generalization Gap",
    category="Generalization",
    description="Combined NMI/ARI/ACC gap.",
    tags=("gap", "combined"),
  )
)
def render_generalization_gap_combined(**ctx):
  return legacy.plot_generalization_gap_combined(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    show_cld=ctx.get("show_cld", False),
  )
