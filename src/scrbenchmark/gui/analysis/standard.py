"""Standard mode entrypoints."""

from . import legacy


def render_standard_mode(handler, info):
  # Legacy function still owns the complete UI flow for standard mode.
  # The package split is kept compatible while migration proceeds incrementally.
  return legacy.render_analysis_page()
