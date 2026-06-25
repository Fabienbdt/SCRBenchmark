"""
Session state helpers with explicit key contract and optional cascade invalidation.

This module is UI-only and does not alter scientific computations.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Set


KEYS = SimpleNamespace(
  DATA_HANDLER="data_handler",
  DATA_LOADED="data_loaded",
  DATA_PREPROCESSED="data_preprocessed",
  PREPROCESSING_PARAMS="preprocessing_params",
  BENCHMARK_CONFIGURED="benchmark_configured",
  BENCHMARK_SETUP="benchmark_setup",
  BENCHMARK_PROCESSED="benchmark_processed",
  BENCHMARK_RESULTS="benchmark_results",
  ANALYSIS_RESULTS="analysis_results",
  SELECTED_ALGORITHMS="selected_algorithms",
  ALGORITHM_PARAMS="algorithm_params",
  CURRENT_PAGE="current_page",
  EXPLORER_RESULTS="explorer_results",
  EXPLORER_AGG_DF="explorer_agg_df",
  HP_SEARCH_RESULTS="hp_search_results",
  HP_PARAM_SPECS="hp_param_specs",
  UI_MODE="ui_mode",
)


SESSION_DEFAULTS: Dict[str, Any] = {
  KEYS.DATA_HANDLER: None,
  KEYS.DATA_LOADED: False,
  KEYS.DATA_PREPROCESSED: False,
  KEYS.PREPROCESSING_PARAMS: {},
  KEYS.BENCHMARK_CONFIGURED: False,
  KEYS.BENCHMARK_SETUP: {},
  KEYS.BENCHMARK_RESULTS: None,
  KEYS.ANALYSIS_RESULTS: None,
  KEYS.SELECTED_ALGORITHMS: [],
  KEYS.ALGORITHM_PARAMS: {},
  KEYS.CURRENT_PAGE: "Data Upload",
  KEYS.EXPLORER_RESULTS: {},
  KEYS.EXPLORER_AGG_DF: None,
  KEYS.HP_SEARCH_RESULTS: None,
  KEYS.HP_PARAM_SPECS: {},
  KEYS.UI_MODE: "quick",
}


# Conservative dependency graph for UI invalidation.
DEPENDENCY_GRAPH: Dict[str, List[str]] = {
  KEYS.DATA_HANDLER: [
    KEYS.DATA_LOADED,
    KEYS.BENCHMARK_CONFIGURED,
    KEYS.BENCHMARK_SETUP,
    KEYS.BENCHMARK_PROCESSED,
    KEYS.DATA_PREPROCESSED,
    KEYS.PREPROCESSING_PARAMS,
    KEYS.SELECTED_ALGORITHMS,
    KEYS.ALGORITHM_PARAMS,
    KEYS.ANALYSIS_RESULTS,
    KEYS.BENCHMARK_RESULTS,
    KEYS.HP_SEARCH_RESULTS,
  ],
  KEYS.BENCHMARK_SETUP: [
    KEYS.BENCHMARK_PROCESSED,
    KEYS.DATA_PREPROCESSED,
    KEYS.ANALYSIS_RESULTS,
    KEYS.BENCHMARK_RESULTS,
    KEYS.HP_SEARCH_RESULTS,
  ],
  KEYS.PREPROCESSING_PARAMS: [
    KEYS.DATA_PREPROCESSED,
    KEYS.ANALYSIS_RESULTS,
    KEYS.BENCHMARK_RESULTS,
    KEYS.HP_SEARCH_RESULTS,
  ],
  KEYS.SELECTED_ALGORITHMS: [
    KEYS.ANALYSIS_RESULTS,
    KEYS.BENCHMARK_RESULTS,
  ],
  KEYS.ALGORITHM_PARAMS: [
    KEYS.ANALYSIS_RESULTS,
    KEYS.BENCHMARK_RESULTS,
  ],
}


def _clone_default(value: Any) -> Any:
  if isinstance(value, (dict, list, set)):
    return copy.deepcopy(value)
  return value


def init_session_defaults(session_state: MutableMapping[str, Any]) -> None:
  """Initialize session keys once using the shared key contract."""
  for key, default_value in SESSION_DEFAULTS.items():
    if key not in session_state:
      session_state[key] = _clone_default(default_value)


def _collect_downstream(start_key: str, graph: Mapping[str, Iterable[str]]) -> Set[str]:
  visited: Set[str] = set()
  stack = [start_key]
  while stack:
    current = stack.pop()
    for child in graph.get(current, []):
      if child in visited:
        continue
      visited.add(child)
      stack.append(child)
  return visited


def invalidate(
  session_state: MutableMapping[str, Any],
  key: str,
  recursive: bool = True,
  graph: Mapping[str, Iterable[str]] = DEPENDENCY_GRAPH,
) -> None:
  """Reset one key to default and optionally reset all downstream dependencies."""
  if key in SESSION_DEFAULTS:
    session_state[key] = _clone_default(SESSION_DEFAULTS[key])

  if not recursive:
    return

  for downstream_key in _collect_downstream(key, graph):
    if downstream_key in SESSION_DEFAULTS:
      session_state[downstream_key] = _clone_default(SESSION_DEFAULTS[downstream_key])


def set_with_cascade(
  session_state: MutableMapping[str, Any],
  key: str,
  value: Any,
  graph: Mapping[str, Iterable[str]] = DEPENDENCY_GRAPH,
) -> bool:
  """
  Set key only if value changed, then invalidate downstream keys.

  Returns True when the value changed.
  """
  previous = session_state.get(key)
  changed = previous != value
  session_state[key] = value
  if changed:
    for downstream_key in _collect_downstream(key, graph):
      if downstream_key in SESSION_DEFAULTS:
        session_state[downstream_key] = _clone_default(SESSION_DEFAULTS[downstream_key])
  return changed


def compute_workflow_progress(session_state: Mapping[str, Any]) -> Dict[str, Any]:
  """Return completion status for workflow steps 1->5."""
  steps = [
    ("Data Upload", bool(session_state.get(KEYS.DATA_LOADED, False))),
    (
      "Data Split",
      bool(session_state.get(KEYS.BENCHMARK_CONFIGURED, False)),
    ),
    ("Preprocessing", bool(session_state.get(KEYS.DATA_PREPROCESSED, False))),
    (
      "Algorithm Config",
      bool(session_state.get(KEYS.SELECTED_ALGORITHMS, [])),
    ),
    (
      "Run Analysis",
      session_state.get(KEYS.ANALYSIS_RESULTS) is not None
      or session_state.get(KEYS.BENCHMARK_RESULTS) is not None,
    ),
  ]
  completed = sum(1 for _, done in steps if done)
  return {"completed": completed, "total": len(steps), "steps": steps}
