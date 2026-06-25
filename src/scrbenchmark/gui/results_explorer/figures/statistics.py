"""Statistical and table figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="statistical_test",
    name="Statistical Test",
    category="Statistics",
    description="Statistical comparison across algorithms/conditions.",
    tags=("statistics", "test"),
  )
)
def render_statistical_test(**ctx):
  return legacy.perform_statistical_test(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["selected_conditions"],
    ctx.get("metric", "NMI"),
    ctx.get("test_mode", "pairwise_conditions"),
  )


@FigureRegistry.register(
  FigureInfo(
    key="summary_table",
    name="Summary Table",
    category="Statistics",
    description="Table of aggregated metrics.",
    tags=("table", "summary"),
  )
)
def render_summary_table(**ctx):
  return legacy.create_summary_dataframe(
    ctx["agg_df"], ctx["selected_algorithms"], ctx["selected_conditions"]
  )


@FigureRegistry.register(
  FigureInfo(
    key="metrics_by_celltype_table",
    name="Metrics by Cell Type",
    category="Statistics",
    description="Table of F1/precision/recall by cell type.",
    tags=("table", "celltype"),
  )
)
def render_metrics_by_celltype_table(**ctx):
  return legacy.create_metrics_by_celltype_table(
    ctx["all_data"], ctx["selected_algorithms"], ctx["selected_conditions"]
  )
