"""Runtime-related figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="runtime_comparison",
    name="Runtime Comparison",
    category="Runtime",
    description="Execution time by algorithm.",
    tags=("runtime", "time"),
  )
)
def render_runtime_comparison(**ctx):
  return legacy.plot_runtime_comparison(
    ctx["agg_df"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    all_data=ctx.get("all_data"),
    show_cld=ctx.get("show_cld", False),
  )


@FigureRegistry.register(
  FigureInfo(
    key="loss_curves",
    name="Loss Curves",
    category="Runtime",
    description="Training loss curves.",
    tags=("loss", "training"),
  )
)
def render_loss_curves(**ctx):
  return ctx["plot_loss_curves_gallery"](
    ctx["all_data"], ctx["selected_algorithms"], ctx["target_conditions"]
  )
