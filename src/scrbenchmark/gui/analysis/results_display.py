"""Results display wrappers."""

from . import legacy


def display_standard_results():
  return legacy._display_results()


def display_benchmark_results():
  return legacy._display_benchmark_results()


def export_standard_csv(results):
  return legacy._export_csv(results)


def export_benchmark_csv(results):
  return legacy._export_benchmark_csv(results)
