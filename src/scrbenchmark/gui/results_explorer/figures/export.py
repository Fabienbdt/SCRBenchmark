"""Publication/export helpers."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="publication_export",
    name="Publication Export",
    category="Export",
    description="Export table as LaTeX/Markdown.",
    tags=("export", "publication"),
  )
)
def render_publication_export(**ctx):
  return legacy.export_publication_ready(
    ctx["agg_df"],
    ctx["selected_algorithms"],
    ctx["selected_conditions"],
    ctx.get("fmt", "latex"),
  )
