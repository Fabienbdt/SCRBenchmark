"""Execution engine wrappers for standard and benchmark analysis."""

from . import legacy


def run_standard_analysis(handler, settings):
  return legacy._run_analysis(
    handler,
    n_repeats=int(settings.get("n_repeats", 1)),
    random_seed=int(settings.get("random_seed", 42)),
    compute_scib_metrics=bool(settings.get("compute_scib_metrics", True)),
  )


def run_benchmark_analysis(handler):
  return legacy._run_benchmark_analysis(handler)
