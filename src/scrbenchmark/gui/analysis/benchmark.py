"""Benchmark mode wrappers."""

from . import legacy


def render_benchmark_settings(handler, info):
  return legacy._render_benchmark_settings(handler, info)


def render_benchmark_visualizations(results):
  return legacy._render_benchmark_visualizations(results)


def render_group_metrics(results, standalone: bool = False):
  return legacy._render_group_metrics(results, standalone=standalone)
