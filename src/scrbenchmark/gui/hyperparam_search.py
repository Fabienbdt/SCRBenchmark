"""
Hyperparameter search page for algorithm optimization.
Provides Grid Search and Random Search with interactive parameter configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import io
import json
import copy
import re
from typing import Any, Dict, List, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.algorithm_registry import AlgorithmRegistry
from core.config import ParamType
from utils.hyperparam_search import (
  HyperparameterSearcher, SearchSummary, SearchResult,
  generate_param_grid_from_config, save_search_results
)
try:
  from core.optimization import OptunaOptimizer
  OPTUNA_AVAILABLE = True
  OPTUNA_IMPORT_ERROR = None
except Exception as exc:
  OptunaOptimizer = None
  OPTUNA_AVAILABLE = False
  OPTUNA_IMPORT_ERROR = str(exc)
from utils.dataset_splitter import DatasetSplitter
from gui.widgets import check_prerequisites
from gui.widgets import display_error


def render_hyperparam_search_page():
  """Render the hyperparameter search page."""
  st.header("Hyperparameter Search")

  # Check prerequisites
  if not check_prerequisites(require_data=True, require_preprocessing=True):
    return

  # Initialize session state
  if 'hp_search_results' not in st.session_state:
    st.session_state.hp_search_results = None
  if 'hp_param_specs' not in st.session_state:
    st.session_state.hp_param_specs = {}

  # Algorithm selection
  algo_name = _render_algorithm_selection()

  if algo_name:
    st.markdown("---")

    # Parameter grid configuration
    param_specs = _render_param_grid_config(algo_name)

    st.markdown("---")

    # Search settings
    settings = _render_search_settings(param_specs)

    st.markdown("---")

    # Run search button
    if st.button("Run Search", type="primary", width="stretch"):
      _run_search(algo_name, param_specs, settings)

  # Display results
  if st.session_state.hp_search_results is not None:
    st.markdown("---")
    _display_search_results(st.session_state.hp_search_results)


def _check_prerequisites() -> bool:
  """Check if prerequisites are met for hyperparameter search."""
  return check_prerequisites(require_data=True, require_preprocessing=True)


def _render_algorithm_selection() -> Optional[str]:
  """Render algorithm selection UI."""
  st.subheader("Algorithm Selection")

  algorithms = AlgorithmRegistry.get_all()

  if not algorithms:
    st.error("No algorithms available.")
    return None

  # Create options list
  algo_options = []
  algo_display_names = {}
  for name, algo_class in algorithms.items():
    info = algo_class.get_info()
    algo_options.append(name)
    algo_display_names[name] = f"{info.display_name} ({info.category})"

  selected = st.selectbox(
    "Select algorithm to optimize",
    options=algo_options,
    format_func=lambda x: algo_display_names.get(x, x),
    help="Choose which algorithm to perform hyperparameter search on"
  )

  if selected:
    algo_info = AlgorithmRegistry.get(selected).get_info()
    st.info(f"**{algo_info.display_name}**: {algo_info.description}")

    if algo_info.requires_gpu:
      st.warning("This algorithm requires GPU for optimal performance.")

  return selected


def _deduplicate_preserve_order(values: List[Any]) -> List[Any]:
  """Remove duplicates while preserving input order."""
  seen = set()
  unique_values = []
  for value in values:
    key = repr(value)
    if key not in seen:
      seen.add(key)
      unique_values.append(value)
  return unique_values


def _parse_numeric_list_input(values_str: str, as_integer: bool) -> List[Any]:
  """
  Parse numeric values separated by comma/semicolon/whitespace/newlines.

  Examples:
    "0, 0.25, 0.5"
    "16 32 64"
    "10;20;30"
  """
  if not values_str:
    return []

  tokens = [tok for tok in re.split(r"[,;\s]+", values_str.strip()) if tok]
  if as_integer:
    values = [int(tok) for tok in tokens]
  else:
    values = [float(tok) for tok in tokens]
  return _deduplicate_preserve_order(values)


def _sync_include_widget_state(
  algo_name: str,
  hyperparams: List[Any],
  param_specs: Dict[str, Any],
  force: bool = False
):
  """Keep include-checkbox states aligned with current parameter specs."""
  selected_names = set(param_specs.keys())
  for hp in hyperparams:
    widget_key = f"include_{algo_name}_{hp.name}"
    if force or widget_key not in st.session_state:
      st.session_state[widget_key] = hp.name in selected_names


def _filter_specs_for_hyperparams(specs: Dict[str, Any], hyperparams: List[Any]) -> Dict[str, Any]:
  """Drop any parameter specs that are not defined by the selected algorithm."""
  valid_names = {hp.name for hp in hyperparams}
  return {name: spec for name, spec in specs.items() if name in valid_names}


def _get_search_space_templates(algo_name: str) -> Dict[str, Dict[str, Any]]:
  """Return quick templates to speed up common hyperparameter searches."""
  templates = {}

  optimized_defaults = _get_optimized_search_defaults(algo_name)
  if optimized_defaults:
    templates["Optimized Defaults"] = optimized_defaults



  return templates


def _render_param_grid_config(algo_name: str) -> Dict[str, Any]:
  """Render parameter grid configuration UI."""
  st.subheader("Parameter Grid Configuration")

  algo_class = AlgorithmRegistry.get(algo_name)
  hyperparams = algo_class.get_hyperparameters()

  # Initialize specs if empty
  if algo_name not in st.session_state.hp_param_specs:
    st.session_state.hp_param_specs[algo_name] = _get_optimized_search_defaults(algo_name)

  param_specs = copy.deepcopy(st.session_state.hp_param_specs[algo_name])
  param_specs = _filter_specs_for_hyperparams(param_specs, hyperparams)

  st.markdown("Select parameters to optimize and define their search space:")

  # Quick actions and templates
  templates = _get_search_space_templates(algo_name)
  template_names = list(templates.keys())
  col_t1, col_t2, col_t3, col_t4 = st.columns([2.3, 1, 1, 1])
  with col_t1:
    template_choice = st.selectbox(
      "Quick template",
      options=template_names if template_names else ["Custom"],
      key=f"hp_template_{algo_name}",
      disabled=not template_names,
      help="Apply curated search spaces for faster setup."
    )
  with col_t2:
    if st.button("Apply Template", key=f"apply_template_{algo_name}", disabled=not template_names, width="stretch"):
      selected_specs = copy.deepcopy(templates[template_choice])
      selected_specs = _filter_specs_for_hyperparams(selected_specs, hyperparams)
      st.session_state.hp_param_specs[algo_name] = selected_specs
      _sync_include_widget_state(algo_name, hyperparams, selected_specs, force=True)
      st.rerun()
  with col_t3:
    if st.button("Reset Defaults", key=f"reset_defaults_{algo_name}", width="stretch"):
      default_specs = _filter_specs_for_hyperparams(_get_optimized_search_defaults(algo_name), hyperparams)
      st.session_state.hp_param_specs[algo_name] = default_specs
      _sync_include_widget_state(algo_name, hyperparams, default_specs, force=True)
      st.rerun()
  with col_t4:
    if st.button("Clear All", key=f"clear_specs_{algo_name}", width="stretch"):
      st.session_state.hp_param_specs[algo_name] = {}
      _sync_include_widget_state(algo_name, hyperparams, {}, force=True)
      st.rerun()

  base_params = st.session_state.get('algorithm_params', {}).get(algo_name, {})
  with st.expander("Base parameters used during search (fixed)", expanded=False):
    if base_params:
      st.json(base_params)
    else:
      st.caption("No fixed base parameters found in Algorithm Config for this algorithm.")

  show_advanced = st.toggle(
    "Show advanced hyperparameters",
    value=False,
    key=f"show_advanced_hp_{algo_name}",
    help="Advanced parameters are hidden by default to keep the search space manageable."
  )

  _sync_include_widget_state(algo_name, hyperparams, param_specs)

  hidden_advanced_selected = [
    hp.name for hp in hyperparams
    if hp.advanced and not show_advanced and hp.name in param_specs
  ]
  if hidden_advanced_selected:
    st.warning(
      "Some advanced parameters are currently selected but hidden: "
      + ", ".join(hidden_advanced_selected)
    )

  # Group params by category
  categories = {}
  for hp in hyperparams:
    if hp.advanced and not show_advanced:
      continue
    if hp.category not in categories:
      categories[hp.category] = []
    categories[hp.category].append(hp)

  for category, hps in categories.items():
    if not hps:
      continue
    with st.expander(f"**{category}**", expanded=True):
      for hp in hps:
        col1, col2, col3 = st.columns([0.15, 0.35, 0.5])

        include_key = f"include_{algo_name}_{hp.name}"
        with col1:
          # Checkbox to include this parameter in search
          include = st.checkbox(
            "Include",
            key=include_key,
            label_visibility="collapsed"
          )

        with col2:
          title = hp.display_name
          if hp.advanced:
            title = f"{title} *(advanced)*"
          st.markdown(f"**{title}**")
          if hp.description:
            st.caption(hp.description)

        with col3:
          if include:
            spec = _render_param_spec_input(hp, algo_name, param_specs.get(hp.name))
            if spec is not None:
              param_specs[hp.name] = spec
          else:
            if hp.name in param_specs:
              del param_specs[hp.name]
            st.caption(f"Default: {hp.default}")

  # Save specs
  st.session_state.hp_param_specs[algo_name] = param_specs

  # Display summary
  if param_specs:
    st.markdown("#### Selected Parameters")
    for name, spec in param_specs.items():
      if spec['type'] == 'list':
        st.markdown(f"- **{name}** ({len(spec['values'])} value(s)): {spec['values']}")
      elif spec['type'] == 'range':
        step = spec.get('step', 1)
        n_values = int(np.floor((spec['max'] - spec['min']) / step + 1e-12)) + 1 if step else 0
        st.markdown(
          f"- **{name}**: {spec['min']} to {spec['max']} "
          f"(step {step}, ~{max(0, n_values)} value(s))"
        )
  else:
    st.info("Select at least one parameter to optimize.")

  return param_specs


def _render_param_spec_input(hp, algo_name: str, current_spec: Optional[Dict]) -> Optional[Dict]:
  """Render input for parameter search specification."""
  key_prefix = f"{algo_name}_{hp.name}"

  spec_type_options = ['list']
  if hp.param_type in [ParamType.INTEGER, ParamType.FLOAT]:
    spec_type_options.append('range')

  if len(spec_type_options) == 1:
    spec_type = spec_type_options[0]
    st.caption("Type: list")
  else:
    default_index = 0
    if current_spec is not None and current_spec.get('type') == 'range':
      default_index = 1
    spec_type = st.radio(
      "Type",
      options=spec_type_options,
      index=default_index,
      key=f"{key_prefix}_type",
      horizontal=True,
      label_visibility="collapsed"
    )

  if spec_type == 'list':
    # List of values
    if hp.param_type in [ParamType.CHOICE, ParamType.MULTI_CHOICE]:
      if not hp.choices:
        st.error("No choices available for this parameter.")
        return None
      # Use available choices
      default_values = current_spec.get('values', hp.choices) if current_spec else hp.choices
      values = st.multiselect(
        "Values",
        options=hp.choices,
        default=default_values,
        key=f"{key_prefix}_list",
        label_visibility="collapsed"
      )
      if values:
        return {'type': 'list', 'values': _deduplicate_preserve_order(values)}
      st.caption("Select at least one value.")

    elif hp.param_type in [ParamType.INTEGER, ParamType.FLOAT]:
      # Text input for comma-separated values
      default_str = ""
      if current_spec and current_spec.get('type') == 'list':
        default_str = ", ".join(str(v) for v in current_spec.get('values', []))
      elif hp.default is not None:
        default_str = str(hp.default)

      values_str = st.text_input(
        "Values (comma/semicolon/space/newline separated)",
        value=default_str,
        key=f"{key_prefix}_list_str",
        placeholder="e.g., 0 0.25 0.5 0.75 1.0"
      )

      if values_str:
        try:
          values = _parse_numeric_list_input(values_str, as_integer=(hp.param_type == ParamType.INTEGER))
          return {'type': 'list', 'values': values}
        except ValueError:
          st.error("Invalid values. Use valid numbers only.")
      else:
        st.caption("Enter at least one value.")

    elif hp.param_type == ParamType.STRING:
      default_str = ""
      if current_spec and current_spec.get('type') == 'list':
        default_str = "; ".join(str(v) for v in current_spec.get('values', []))
      elif hp.default is not None:
        default_str = str(hp.default)

      values_str = st.text_input(
        "Values (semicolon-separated)",
        value=default_str,
        key=f"{key_prefix}_list_str",
        placeholder="e.g., 256; 512,256; 1024,512"
      )

      if values_str:
        values = [v.strip() for v in values_str.replace("\n", ";").split(";") if v.strip()]
        if values:
          return {'type': 'list', 'values': _deduplicate_preserve_order(values)}
      else:
        st.caption("Enter at least one value.")

    elif hp.param_type == ParamType.BOOLEAN:
      default_values = [bool(hp.default)]
      if current_spec and current_spec.get('type') == 'list':
        default_values = [bool(v) for v in current_spec.get('values', default_values)]

      values = st.multiselect(
        "Values",
        options=[True, False],
        default=_deduplicate_preserve_order(default_values),
        format_func=lambda x: "True" if x else "False",
        key=f"{key_prefix}_bool_list",
        label_visibility="collapsed"
      )
      if values:
        return {'type': 'list', 'values': _deduplicate_preserve_order(values)}
      st.caption("Select at least one value.")

  else: # range
    if hp.param_type in [ParamType.INTEGER, ParamType.FLOAT]:
      col_a, col_b, col_c = st.columns(3)

      # Defaults
      default_min = hp.min_value if hp.min_value is not None else hp.default
      default_max = hp.max_value if hp.max_value is not None else hp.default
      default_step = hp.step if hp.step is not None else 1

      if current_spec and current_spec.get('type') == 'range':
        default_min = current_spec.get('min', default_min)
        default_max = current_spec.get('max', default_max)
        default_step = current_spec.get('step', default_step)

      if default_min is None:
        default_min = 0 if hp.param_type == ParamType.INTEGER else 0.0
      if default_max is None:
        default_max = default_min + (1 if hp.param_type == ParamType.INTEGER else 1.0)
      if default_max < default_min:
        default_max = default_min

      with col_a:
        if hp.param_type == ParamType.INTEGER:
          min_val = st.number_input("Min", value=int(default_min), key=f"{key_prefix}_min")
        else:
          min_val = st.number_input("Min", value=float(default_min), key=f"{key_prefix}_min", format="%.4f")

      with col_b:
        if hp.param_type == ParamType.INTEGER:
          max_val = st.number_input("Max", value=int(default_max), key=f"{key_prefix}_max")
        else:
          max_val = st.number_input("Max", value=float(default_max), key=f"{key_prefix}_max", format="%.4f")

      with col_c:
        if hp.param_type == ParamType.INTEGER:
          step = st.number_input("Step", value=int(default_step), min_value=1, key=f"{key_prefix}_step")
        else:
          step = st.number_input("Step", value=float(default_step), min_value=0.0001, key=f"{key_prefix}_step", format="%.4f")

      if step <= 0:
        st.error("Step must be greater than 0.")
        return None

      if min_val > max_val:
        st.error("Min must be less than or equal to Max.")
        return None

      if hp.param_type == ParamType.INTEGER:
        min_val = int(min_val)
        max_val = int(max_val)
        step = int(step)

      return {'type': 'range', 'min': min_val, 'max': max_val, 'step': step}

  return None


def _render_search_settings(param_specs: Dict[str, Any]) -> Dict[str, Any]:
  """Render search settings UI."""
  st.subheader("Search Settings")

  col1, col2 = st.columns(2)

  with col1:
    search_type_options = ['grid', 'random']
    if OPTUNA_AVAILABLE:
      search_type_options.append('optuna')

    search_type = st.radio(
      "Search type",
      options=search_type_options,
      format_func=lambda x: {
        'grid': 'Grid Search',
        'random': 'Random Search',
        'optuna': 'Optuna (Bayesian)'
      }.get(x, x),
      help="Grid: exhaustive; Random: random sampling; Optuna: Bayesian optimization (smart)"
    )

    metric = st.selectbox(
      "Optimize for",
      options=['NMI', 'ARI', 'ACC', 'Silhouette'],
      index=0,
      help="Metric to maximize during search"
    )

  with col2:
    if search_type == 'optuna':
       n_repeats = st.number_input(
        "Trials (Combinations)",
        min_value=5,
        max_value=200,
        value=20,
        help="Number of trials for Optuna optimization"
      )
    else:
      n_repeats = st.number_input(
        "Repeats per combination",
        min_value=1,
        max_value=20,
        value=3,
        help="Number of runs per parameter combination for stable results"
      )

    random_seed = st.number_input(
      "Random seed",
      min_value=0,
      max_value=99999,
      value=42,
      help="Base random seed for reproducibility"
    )

  # Random search specific
  n_iter = 50
  if search_type == 'random':
    n_iter = st.number_input(
      "Number of iterations",
      min_value=10,
      max_value=1000,
      value=50,
      help="Number of random combinations to try"
    )
  elif search_type == 'optuna' and not OPTUNA_AVAILABLE:
    st.warning(f"Optuna not available: {OPTUNA_IMPORT_ERROR}")

  # Estimate search size
  if param_specs:
    total_combinations = _estimate_search_size(param_specs, search_type, n_iter if search_type == 'random' else n_repeats)
    # For Grid/Random, total_runs includes repeats. For Optuna, 'repeats' is actually 'n_trials' (one run per trial usually, or maybe internal CV? OptunaOptimizer typically does 1 run per trial unless configured otherwise).
    # OptunaOptimizer default implementation does 1 run per trial on train/val split.
    
    if search_type == 'optuna':
      total_runs = n_repeats
      st.markdown("#### Search Summary")
      col1, col2 = st.columns(2)
      with col1:
         st.metric("Total Trials", total_runs)
      with col2:
         st.metric("Parameters", len(param_specs))
    else:
      total_runs = total_combinations * n_repeats
      st.markdown("#### Search Summary")
      col1, col2, col3 = st.columns(3)
      with col1:
        st.metric("Combinations", total_combinations)
      with col2:
        st.metric("Total runs", total_runs)
      with col3:
        st.metric("Parameters", len(param_specs))

    if total_runs > 100:
      st.warning(f"This search will run {total_runs} algorithm executions. This may take a long time.")

  return {
    'search_type': search_type,
    'metric': metric,
    'n_repeats': n_repeats, # interpreted as n_trials for Optuna
    'random_seed': random_seed,
    'n_iter': n_iter
  }


def _estimate_search_size(param_specs: Dict[str, Any], search_type: str, n_iter: int) -> int:
  """Estimate the number of combinations in the search."""
  if search_type == 'random' or search_type == 'optuna':
    return n_iter

  # Grid search: product of all parameter sizes, reusing the same expansion
  # logic as the backend generator for consistency.
  try:
    param_grid = generate_param_grid_from_config([], param_specs)
  except Exception:
    return 0

  total = 1
  for values in param_grid.values():
    total *= max(1, len(values))
  return total


def _validate_param_specs(param_specs: Dict[str, Any]) -> List[str]:
  """Validate user-provided parameter specifications before launching a search."""
  errors = []
  for name, spec in param_specs.items():
    spec_type = spec.get('type')
    if spec_type == 'list':
      values = spec.get('values', [])
      if not isinstance(values, list) or len(values) == 0:
        errors.append(f"`{name}` has an empty list of values.")
    elif spec_type == 'range':
      min_val = spec.get('min')
      max_val = spec.get('max')
      step = spec.get('step')
      if step is None or step <= 0:
        errors.append(f"`{name}` has an invalid step ({step}).")
      if min_val is None or max_val is None:
        errors.append(f"`{name}` range is missing min or max.")
      elif min_val > max_val:
        errors.append(f"`{name}` has min > max ({min_val} > {max_val}).")
    else:
      errors.append(f"`{name}` has unsupported spec type: {spec_type}.")
  return errors


def _run_search(algo_name: str, param_specs: Dict[str, Any], settings: Dict[str, Any]):
  """Run the hyperparameter search."""
  if not param_specs:
    st.error("Please select at least one parameter to optimize.")
    return

  validation_errors = _validate_param_specs(param_specs)
  if validation_errors:
    st.error("Invalid parameter search space:")
    for err in validation_errors:
      st.markdown(f"- {err}")
    return

  handler = st.session_state.data_handler
  data = handler.get_data()
  labels = handler.get_labels()

  # Get base params (non-optimized parameters)
  base_params = st.session_state.get('algorithm_params', {}).get(algo_name, {})

  progress_bar = st.progress(0)
  status_text = st.empty()
  
  start_time = time.time()

  try:
    if settings['search_type'] == 'optuna':
      if not OPTUNA_AVAILABLE:
        st.error("Optuna optimizer is not available in this environment.")
        return

      # Optuna Optimization
      
      # 1. Prepare data split
      status_text.text("Splitting data for Optuna validation...")
      splitter = DatasetSplitter()
      label_col = 'Group'
      if hasattr(data, 'obs'):
        if 'Group' in data.obs.columns:
          label_col = 'Group'
        elif 'labels' in data.obs.columns:
          label_col = 'labels'
        elif 'labels_encoded' in data.obs.columns:
          label_col = 'labels_encoded'

      split_result = splitter.split_stratified(
        data,
        train_ratio=0.7,
        val_ratio=0.3,
        test_ratio=0.0,
        label_col=label_col
      )
      train_data = split_result.adata_train
      val_data = split_result.adata_val
      labels_val = split_result.get_labels('val')

      # 2. Prepare param space for Optuna
      param_space = {}
      algo_class = AlgorithmRegistry.get(algo_name)
      hyperparams_map = {hp.name: hp for hp in algo_class.get_hyperparameters()}
      
      for name, spec in param_specs.items():
        if spec['type'] == 'list':
          param_space[name] = spec['values']
        elif spec['type'] == 'range':
          hp_conf = hyperparams_map.get(name)
          p_type = 'float'
          if hp_conf and hp_conf.param_type == ParamType.INTEGER:
            p_type = 'int'
          elif isinstance(spec['min'], int) and isinstance(spec['max'], int) and isinstance(spec.get('step', 1), int):
            p_type = 'int'
          
          # Heuristic for log scale
          is_log = False
          if hp_conf:
            name_lower = name.lower()
            if any(k in name_lower for k in ['learning_rate', 'weight_decay', 'lr', 'wd', 'alpha', 'lambda', 'gamma']):
              is_log = True
              
          param_space[name] = {
            'min': spec['min'],
            'max': spec['max'],
            'step': spec.get('step'),
            'type': p_type,
            'log': is_log
          }

      # 3. Initialize Optimizer
      optimizer = OptunaOptimizer(algo_name)
      
      def optuna_callback(trial_num, params, score):
        progress = min(1.0, (trial_num + 1) / settings['n_repeats'])
        progress_bar.progress(progress)
        status_text.text(f"Trial {trial_num+1}/{settings['n_repeats']} | Score: {score:.4f} | Params: {params}")

      # 4. Run Optimization
      best_params = optimizer.optimize(
        train_data, val_data, 
        param_space, 
        n_trials=settings['n_repeats'], # reusing n_repeats as n_trials
        metric=settings['metric'].lower(),
        labels_val=labels_val,
        callback=optuna_callback
      )
      
      # 5. Convert to SearchSummary
      # We need to extract trials from study
      trials_df = optimizer.get_summary()
      all_results = []
      
      metric_col = 'value'
      
      for _, row in trials_df.iterrows():
        if row['state'] != 'COMPLETE':
          continue
        
        # Extract params from row (cols starting with params_)
        p_dict = {}
        for col in trials_df.columns:
          if col.startswith('params_'):
            p_name = col[7:]
            p_dict[p_name] = row[col]
        
        # Merge with base params? No, SearchResult usually stores optimized params
        # But to run again we might need base params. 
        # For consistency with grid/random search result structure:
        
        metrics = {settings['metric']: row[metric_col]}
        metrics_std = {settings['metric']: 0.0} # Single run per trial usually
        
        res = SearchResult(
          params=p_dict,
          metrics=metrics,
          metrics_std=metrics_std,
          runtime=row.get('duration', 0).total_seconds() if hasattr(row.get('duration'), 'total_seconds') else 0,
          n_runs=1
        )
        all_results.append(res)
      
      summary = SearchSummary(
        all_results=all_results,
        best_params=best_params,
        best_score=optimizer.study.best_value,
        best_score_std=0.0,
        metric_name=settings['metric'],
        search_type='optuna',
        total_combinations=len(all_results),
        total_runtime=time.time() - start_time,
        algorithm_name=algo_name
      )
      
    else:
      # Grid / Random Search
      searcher = HyperparameterSearcher(
        algorithm_name=algo_name,
        data=data,
        labels=labels,
        base_params=base_params,
        metric=settings['metric'],
        n_repeats=settings['n_repeats'],
        random_seed=settings['random_seed']
      )

      def progress_callback(message: str, progress: float = None):
        status_text.text(message)
        if progress is not None:
          progress_bar.progress(progress)

      if settings['search_type'] == 'grid':
        # Convert specs to grid
        param_grid = generate_param_grid_from_config([], param_specs)
        summary = searcher.grid_search(param_grid, progress_callback)
      else:
        # Convert specs to distributions
        param_distributions = {}
        for name, spec in param_specs.items():
          if spec['type'] == 'list':
            param_distributions[name] = spec['values']
          elif spec['type'] == 'range':
            param_distributions[name] = spec

        summary = searcher.random_search(
          param_distributions,
          n_iter=settings['n_iter'],
          progress_callback=progress_callback
        )

    st.session_state.hp_search_results = summary

    progress_bar.progress(1.0)
    status_text.text(f"Search completed in {summary.total_runtime:.1f}s!")
    time.sleep(1)
    st.rerun()

  except Exception as e:
    display_error(
      e,
      user_message="Hyperparameter search failed.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="hp_search_run",
    )


def _display_search_results(summary: SearchSummary):
  """Display search results with visualizations."""
  st.subheader("Search Results")

  # Best parameters
  st.markdown("### Best Parameters")

  col1, col2 = st.columns(2)

  with col1:
    st.success(f"**Best {summary.metric_name}: {summary.best_score:.4f} +/- {summary.best_score_std:.4f}**")

    st.markdown("**Optimal parameters:**")
    for name, value in summary.best_params.items():
      st.markdown(f"- `{name}`: **{value}**")

  with col2:
    st.metric("Search type", summary.search_type.title())
    st.metric("Combinations tested", summary.total_combinations)
    st.metric("Total runtime", f"{summary.total_runtime:.1f}s")

  # Results tabs
  tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Results Table",
    "Metric Heatmap",
    "Error by Cell Type",
    "Error by Batch × Cell Type",
    "Export"
  ])

  with tab1:
    _render_results_table(summary)

  with tab2:
    _render_heatmap(summary)

  with tab3:
    _render_celltype_error_view(summary)

  with tab4:
    _render_batch_celltype_error_view(summary)

  with tab5:
    _render_export_options(summary)

  # Apply best params button
  st.markdown("---")
  if st.button("Apply Best Parameters to Algorithm Config", type="primary"):
    _apply_best_params(summary)


def _render_results_table(summary: SearchSummary):
  """Render results as sortable table."""
  df = summary.to_dataframe()

  st.markdown("#### All Results (sorted by metric)")
  st.dataframe(df, width="stretch", height=400)


def _render_heatmap(summary: SearchSummary):
  """Render heatmap visualization if 2 parameters."""
  import matplotlib.pyplot as plt

  df = summary.to_dataframe()
  param_cols = [c for c in df.columns if not c.endswith('_mean') and not c.endswith('_std')
         and c not in ['runtime', 'n_runs']]

  if len(param_cols) < 2:
    st.info("Heatmap requires at least 2 parameters being optimized.")
    return

  st.markdown("#### Parameter Heatmap")

  col1, col2 = st.columns(2)
  with col1:
    x_param = st.selectbox("X-axis parameter", param_cols, index=0)
  with col2:
    y_param = st.selectbox("Y-axis parameter", param_cols, index=min(1, len(param_cols)-1))

  if x_param == y_param:
    st.warning("Please select different parameters for X and Y axes.")
    return

  metric_col = f"{summary.metric_name}_mean"

  try:
    # Create pivot table
    pivot = df.pivot_table(
      values=metric_col,
      index=y_param,
      columns=x_param,
      aggfunc='mean'
    )

    if pivot.empty:
      st.warning("Not enough data to create heatmap.")
      return

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_title(f"{summary.metric_name} by {x_param} and {y_param}")

    # Add colorbar
    plt.colorbar(im, ax=ax, label=summary.metric_name)

    # Add values in cells
    for i in range(len(pivot.index)):
      for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
          text_color = 'white' if val < pivot.values.mean() else 'black'
          ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=text_color, fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    # Download
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    st.download_button(
      label="Download Heatmap",
      data=buf.getvalue(),
      file_name=f"hyperparam_heatmap_{summary.algorithm_name}.png",
      mime="image/png"
    )
    plt.close()

  except Exception as e:
    st.warning(f"Could not create heatmap: {e}")


def _to_float(value: Any, default: float = np.nan) -> float:
  """Safely cast values to float for plotting."""
  try:
    if value is None:
      return default
    return float(value)
  except Exception:
    return default


def _extract_error_entry_stats(entry: Dict[str, Any]) -> Dict[str, float]:
  """Extract normalized error statistics from mean/std or single-run structures."""
  if not isinstance(entry, dict):
    return {
      'error_rate': np.nan,
      'error_rate_std': np.nan,
      'n_samples': np.nan,
      'n_errors': np.nan,
      'n_runs': np.nan
    }

  return {
    'error_rate': _to_float(entry.get('error_rate_mean', entry.get('error_rate'))),
    'error_rate_std': _to_float(
      entry.get('error_rate_std', 0.0 if entry.get('error_rate') is not None else np.nan)
    ),
    'n_samples': _to_float(entry.get('n_samples_mean', entry.get('n_samples'))),
    'n_errors': _to_float(entry.get('n_errors_mean', entry.get('n_errors'))),
    'n_runs': _to_float(entry.get('n_runs'))
  }


def _format_param_preview(params: Dict[str, Any], max_chars: int = 90) -> str:
  """Compact parameter preview for combo selectors."""
  text = ", ".join(f"{k}={v}" for k, v in params.items())
  if len(text) <= max_chars:
    return text
  return text[:max_chars - 3] + "..."


def _get_best_result_index(summary: SearchSummary) -> int:
  """Get index of best result based on selected optimization metric."""
  if not summary.all_results:
    return 0

  scores = []
  for result in summary.all_results:
    score = result.metrics.get(summary.metric_name, np.nan)
    scores.append(_to_float(score))

  if not np.any(np.isfinite(scores)):
    return 0
  return int(np.nanargmax(scores))


def _select_result_for_error_view(summary: SearchSummary, key_prefix: str) -> Tuple[Optional[SearchResult], int]:
  """Render selector for choosing which combination to inspect."""
  if not summary.all_results:
    return None, -1

  indices = list(range(len(summary.all_results)))
  best_idx = _get_best_result_index(summary)

  def _format_option(i: int) -> str:
    result = summary.all_results[i]
    score = _to_float(result.metrics.get(summary.metric_name))
    score_str = f"{score:.4f}" if np.isfinite(score) else "NA"
    return (
      f"#{i + 1} | {summary.metric_name}={score_str} | "
      f"{_format_param_preview(result.params)}"
    )

  selected_idx = st.selectbox(
    "Parameter combination",
    options=indices,
    index=min(best_idx, len(indices) - 1),
    format_func=_format_option,
    key=f"{key_prefix}_combo"
  )

  return summary.all_results[selected_idx], selected_idx


def _render_celltype_error_view(summary: SearchSummary):
  """Render error visualization aggregated by cell type."""
  import matplotlib.pyplot as plt

  st.markdown("#### Error by Cell Type")

  selected_result, _ = _select_result_for_error_view(summary, "hp_error_celltype")
  if selected_result is None:
    st.info("No search results available.")
    return

  error_analysis = getattr(selected_result, 'error_analysis', None)
  if not isinstance(error_analysis, dict):
    st.info(
      "No cell-type error analysis available for this result. "
      "This can happen if labels are missing or if the search was run with Optuna trials without error tracking."
    )
    return

  error_by_celltype = error_analysis.get('error_by_celltype', {}) or {}
  if not error_by_celltype:
    st.info("No per-cell-type errors found for this combination.")
    return

  overall_mean = _to_float(error_analysis.get('overall_error_rate_mean', error_analysis.get('overall_error_rate')))
  overall_std = _to_float(error_analysis.get('overall_error_rate_std'))
  n_runs_valid = int(_to_float(error_analysis.get('n_runs_valid'), 0))

  col1, col2, col3 = st.columns(3)
  col1.metric("Overall Error (mean)", f"{overall_mean:.3f}" if np.isfinite(overall_mean) else "N/A")
  col2.metric("Overall Error (std)", f"{overall_std:.3f}" if np.isfinite(overall_std) else "N/A")
  col3.metric("Runs with Error Data", n_runs_valid)

  rows = []
  for cell_type, entry in error_by_celltype.items():
    stats = _extract_error_entry_stats(entry)
    rows.append({
      'cell_type': str(cell_type),
      'error_rate': stats['error_rate'],
      'error_rate_std': stats['error_rate_std'],
      'n_samples': stats['n_samples'],
      'n_errors': stats['n_errors']
    })

  df = pd.DataFrame(rows)
  if df.empty:
    st.info("No per-cell-type errors found for this combination.")
    return

  df = df.sort_values('error_rate', ascending=False, na_position='last').reset_index(drop=True)

  max_display = max(1, min(60, len(df)))
  default_display = min(20, max_display)
  top_n = st.slider(
    "Cell types to display",
    min_value=1,
    max_value=max_display,
    value=default_display,
    key="hp_celltype_error_top_n"
  )

  plot_df = df.head(top_n).iloc[::-1].copy()
  plot_df['error_rate'] = plot_df['error_rate'].fillna(0.0)

  fig_height = max(4.0, 0.34 * len(plot_df) + 1.8)
  fig, ax = plt.subplots(figsize=(10, fig_height))
  colors = plt.cm.RdYlGn_r(np.clip(plot_df['error_rate'].values, 0, 1))
  y_positions = np.arange(len(plot_df))
  ax.barh(y_positions, plot_df['error_rate'].values, color=colors, alpha=0.95)
  ax.set_yticks(y_positions)
  ax.set_yticklabels(plot_df['cell_type'].values)

  for i, (_, row) in enumerate(plot_df.iterrows()):
    val = _to_float(row['error_rate'])
    n_samples = _to_float(row['n_samples'])
    if np.isfinite(val):
      x_pos = min(0.99, val + 0.02)
      suffix = f" (n={int(round(n_samples))})" if np.isfinite(n_samples) else ""
      ax.text(x_pos, i, f"{val:.2f}{suffix}", va='center', fontsize=8)

  ax.set_xlim(0, 1)
  ax.set_xlabel("Error rate")
  ax.set_ylabel("Cell type")
  ax.set_title("Error by cell type")
  ax.grid(axis='x', alpha=0.25)
  plt.tight_layout()
  st.pyplot(fig)
  plt.close()

  st.markdown("##### Detailed Values")
  st.dataframe(df, width="stretch", height=340)


def _render_batch_celltype_error_view(summary: SearchSummary):
  """Render heatmap of error by group (batch) and cell type."""
  import matplotlib.pyplot as plt
  from matplotlib import cm

  st.markdown("#### Error by Batch and Cell Type")

  selected_result, _ = _select_result_for_error_view(summary, "hp_error_batch_celltype")
  if selected_result is None:
    st.info("No search results available.")
    return

  error_analysis = getattr(selected_result, 'error_analysis', None)
  if not isinstance(error_analysis, dict):
    st.info("No batch × cell type error analysis available for this result.")
    return

  nested_errors = error_analysis.get('error_by_celltype_by_group', {}) or {}
  if not nested_errors:
    st.info(
      "No group-aware error data found. This usually means no batch/group column "
      "was available in the evaluated data."
    )
    return

  group_key = error_analysis.get('group_key', 'group')

  col1, col2, col3 = st.columns(3)
  with col1:
    min_samples = st.number_input(
      "Min samples per cell type × group",
      min_value=1,
      max_value=100000,
      value=1,
      step=1,
      key="hp_error_batch_min_samples"
    )
  with col2:
    sort_celltypes = st.selectbox(
      "Sort cell types",
      options=["Mean error (desc)", "Alphabetical"],
      key="hp_error_batch_sort_ct"
    )
  with col3:
    sort_groups = st.selectbox(
      "Sort groups",
      options=["Alphabetical", "Mean error (desc)"],
      key="hp_error_batch_sort_groups"
    )

  show_counts = st.checkbox(
    "Show sample counts in cells",
    value=False,
    key="hp_error_batch_show_counts"
  )

  groups = sorted(nested_errors.keys())
  cell_types = sorted({
    str(cell_type)
    for group_map in nested_errors.values()
    for cell_type in group_map.keys()
  })

  if not groups or not cell_types:
    st.info("Insufficient data to build the batch × cell type matrix.")
    return

  matrix = np.full((len(cell_types), len(groups)), np.nan, dtype=float)
  n_samples_matrix = np.full((len(cell_types), len(groups)), np.nan, dtype=float)

  for j, group in enumerate(groups):
    for i, cell_type in enumerate(cell_types):
      entry = nested_errors.get(group, {}).get(cell_type)
      stats = _extract_error_entry_stats(entry)
      if not np.isfinite(stats['error_rate']):
        continue
      if np.isfinite(stats['n_samples']) and stats['n_samples'] < min_samples:
        continue
      matrix[i, j] = stats['error_rate']
      n_samples_matrix[i, j] = stats['n_samples']

  if not np.isfinite(matrix).any():
    st.warning(
      "All matrix cells are empty after filtering. "
      "Try lowering the minimum sample threshold."
    )
    return

  if sort_celltypes == "Mean error (desc)":
    row_scores = np.nanmean(matrix, axis=1)
    row_order = np.argsort(np.nan_to_num(row_scores, nan=-1.0))[::-1]
    matrix = matrix[row_order, :]
    n_samples_matrix = n_samples_matrix[row_order, :]
    cell_types = [cell_types[i] for i in row_order]

  if sort_groups == "Mean error (desc)":
    col_scores = np.nanmean(matrix, axis=0)
    col_order = np.argsort(np.nan_to_num(col_scores, nan=-1.0))[::-1]
    matrix = matrix[:, col_order]
    n_samples_matrix = n_samples_matrix[:, col_order]
    groups = [groups[j] for j in col_order]

  cmap = cm.get_cmap('RdYlGn_r')
  if hasattr(cmap, 'copy'):
    cmap = cmap.copy()
  cmap.set_bad(color='#f0f0f0')
  masked_matrix = np.ma.masked_invalid(matrix)

  fig_width = max(10, 0.8 * len(groups) + 3)
  fig_height = max(5, 0.34 * len(cell_types) + 2)
  fig, ax = plt.subplots(figsize=(fig_width, fig_height))
  im = ax.imshow(masked_matrix, cmap=cmap, vmin=0, vmax=1, aspect='auto')

  ax.set_xticks(range(len(groups)))
  ax.set_xticklabels(groups, rotation=45, ha='right')
  ax.set_yticks(range(len(cell_types)))
  ax.set_yticklabels(cell_types)
  ax.set_xlabel(str(group_key))
  ax.set_ylabel("Cell type")
  ax.set_title(f"Error rate by {group_key} and cell type")

  cbar = plt.colorbar(im, ax=ax, label="Error rate")
  cbar.ax.set_ylabel("Error rate", rotation=90)

  for i in range(len(cell_types)):
    for j in range(len(groups)):
      val = matrix[i, j]
      if not np.isfinite(val):
        continue
      text = f"{val:.2f}"
      if show_counts and np.isfinite(n_samples_matrix[i, j]):
        text += f"\n(n={int(round(n_samples_matrix[i, j]))})"
      text_color = "white" if val >= 0.6 else "black"
      ax.text(j, i, text, ha='center', va='center', fontsize=7, color=text_color)

  plt.tight_layout()
  st.pyplot(fig)
  plt.close()

  visible_cells = int(np.isfinite(matrix).sum())
  total_cells = int(matrix.size)
  st.caption(
    f"Visible matrix cells: {visible_cells}/{total_cells} "
    f"(after min-sample filter = {int(min_samples)})"
  )

  df_matrix = pd.DataFrame(matrix, index=cell_types, columns=groups)
  st.markdown("##### Matrix Values")
  st.dataframe(df_matrix.round(3), width="stretch", height=360)


def _render_export_options(summary: SearchSummary):
  """Render export options."""
  st.markdown("#### Export Results")

  col1, col2 = st.columns(2)

  with col1:
    # CSV export
    df = summary.to_dataframe()
    csv = df.to_csv(index=False)
    st.download_button(
      label="Export as CSV",
      data=csv,
      file_name=f"hyperparam_search_{summary.algorithm_name}.csv",
      mime="text/csv",
      width="stretch"
    )

  with col2:
    # JSON export
    json_data = json.dumps(summary.to_dict(), indent=2)
    st.download_button(
      label="Export as JSON",
      data=json_data,
      file_name=f"hyperparam_search_{summary.algorithm_name}.json",
      mime="application/json",
      width="stretch"
    )


def _apply_best_params(summary: SearchSummary):
  """Apply best parameters to algorithm configuration."""
  algo_name = summary.algorithm_name

  if 'algorithm_params' not in st.session_state:
    st.session_state.algorithm_params = {}

  if algo_name not in st.session_state.algorithm_params:
    st.session_state.algorithm_params[algo_name] = {}

  # Update with best params
  st.session_state.algorithm_params[algo_name].update(summary.best_params)

  # Add algorithm to selected if not already
  if 'selected_algorithms' not in st.session_state:
    st.session_state.selected_algorithms = []

  if algo_name not in st.session_state.selected_algorithms:
    st.session_state.selected_algorithms.append(algo_name)

  st.success(f"Best parameters applied to {algo_name}! Go to Algorithm Config to review.")


def _get_optimized_search_defaults(algo_name: str) -> Dict[str, Any]:
  """
  Get optimized default search spaces for algorithms.
  These defaults are based on author recommendations and literature best practices.
  """
  defaults = {}

  if algo_name == 'sccdcg':
    # Based on analysis of train_scCDCG.py (author code)
    defaults['learning_rate'] = {'type': 'list', 'values': [0.001, 0.0005]}
    defaults['balancer'] = {'type': 'range', 'min': 0.4, 'max': 0.8, 'step': 0.05}
    defaults['factor_ort'] = {'type': 'range', 'min': 0.5, 'max': 0.9, 'step': 0.1}
    defaults['factor_KL'] = {'type': 'range', 'min': 0.1, 'max': 0.5, 'step': 0.05}
  
  elif algo_name == 'scdeepcluster':
    # Standard defaults for scDeepCluster
    defaults['gamma'] = {'type': 'range', 'min': 0.1, 'max': 2.0, 'step': 0.1} # Clustering loss weight
    defaults['learning_rate'] = {'type': 'list', 'values': [0.001, 0.0001]}
    defaults['update_interval'] = {'type': 'list', 'values': [1, 3, 5]}

  elif algo_name == 'pca':
    defaults['clustering_method'] = {'type': 'list', 'values': ['kmeans', 'louvain', 'leiden', 'hdbscan']}
    defaults['n_pca_components'] = {'type': 'list', 'values': [0, 20, 30, 50]}
    defaults['graph_resolution'] = {'type': 'range', 'min': 0.0, 'max': 2.0, 'step': 0.1}
    defaults['hdbscan_min_cluster_size'] = {'type': 'list', 'values': [5, 8, 15, 30]}

  elif algo_name == 'pca_kmeans':
    # Simple clustering
    defaults['n_clusters'] = {'type': 'range', 'min': 5, 'max': 20, 'step': 1}
    defaults['n_pca_components'] = {'type': 'list', 'values': [10, 20, 30, 50]}

  elif algo_name == 'pca_leiden':
    # Resolution is key
    defaults['leiden_resolution'] = {'type': 'range', 'min': 0.1, 'max': 2.0, 'step': 0.1}
    defaults['leiden_neighbors'] = {'type': 'list', 'values': [10, 15, 30, 50]}
  
  elif algo_name == 'sc_mae':
    defaults['mask_ratio'] = {'type': 'range', 'min': 0.2, 'max': 0.8, 'step': 0.1}
    defaults['hidden_size'] = {'type': 'list', 'values': [64, 128, 256]}



  return defaults
