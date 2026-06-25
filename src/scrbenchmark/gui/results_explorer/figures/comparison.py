"""Comparison figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="algorithm_comparison",
    name="Algorithm Comparison",
    category="Comparison",
    description="Grouped bar plot comparing metrics.",
    tags=("performance", "metrics"),
  )
)
def render_algorithm_comparison(**ctx):
  return legacy.plot_algorithm_comparison(
    ctx["agg_df"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    ctx.get("metric", "NMI"),
    all_data=ctx.get("all_data"),
    show_cld=ctx.get("show_cld", False),
  )


@FigureRegistry.register(
  FigureInfo(
    key="metrics_heatmap",
    name="Metrics Heatmap",
    category="Comparison",
    description="Heatmap of performance by algorithm/condition.",
    tags=("heatmap", "metrics"),
  )
)
def render_metrics_heatmap(**ctx):
  return legacy.plot_metrics_heatmap(
    ctx["agg_df"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    ctx.get("metric", "NMI"),
  )
