"""UMAP figures."""

from .. import legacy
from ..registry import FigureInfo, FigureRegistry


@FigureRegistry.register(
  FigureInfo(
    key="saved_umap_gallery",
    name="UMAP Gallery",
    category="UMAP",
    description="Saved UMAP visualizations.",
    tags=("umap", "gallery"),
  )
)
def render_saved_umap_gallery(**ctx):
  return legacy.plot_saved_umap_gallery(
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    split_filter=ctx.get("umap_split", "all"),
    max_images=ctx.get("umap_max_images", 18),
  )


@FigureRegistry.register(
  FigureInfo(
    key="umap_evolution",
    name="UMAP Evolution",
    category="UMAP",
    description="Saved UMAP evolution snapshots.",
    tags=("umap", "evolution", "training"),
  )
)
def render_umap_evolution(**ctx):
  return ctx["plot_umap_evolution_gallery"](
    ctx["all_data"],
    ctx["selected_algorithms"],
    ctx["target_conditions"],
    max_images=ctx.get("umap_max_images", 18),
  )


@FigureRegistry.register(
  FigureInfo(
    key="umap_diagnostic",
    name="UMAP Diagnostic",
    category="UMAP",
    description="UMAP diagnostic from labels + h5ad.",
    tags=("umap", "diagnostic"),
  )
)
def render_umap_diagnostic(**ctx):
  return legacy.plot_umap_diagnostic_from_results(
    all_data=ctx["all_data"],
    condition=ctx["diag_condition"],
    algorithm=ctx["diag_algorithm"],
    run_id=int(ctx["diag_run_id"]),
    split=ctx["diag_split"],
    outline_mode=ctx.get("diag_outline_mode", "none"),
    max_points=int(ctx.get("diag_max_points", 15000)),
    random_state=int(ctx.get("diag_seed", 42)),
    show_centroids=bool(ctx.get("diag_show_centroids", True)),
  )
