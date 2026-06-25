"""Inspector wrappers for analysis outputs."""

from . import legacy


def render_group_inspector(results):
  return legacy._render_group_inspector(results)


def render_batch_gene_inspector(results):
  return legacy._render_batch_gene_inspector(results)


def render_celltype_gene_inspector(results):
  return legacy._render_celltype_gene_inspector(results)


def render_cell_inspector(results):
  return legacy._render_cell_inspector(results)


def render_umap(results):
  return legacy._render_umap(results)
