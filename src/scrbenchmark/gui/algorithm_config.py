"""
Algorithm configuration page for the Streamlit interface.
"""

from __future__ import annotations

import streamlit as st
import re
from pathlib import Path
import sys
import os
import shlex
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.algorithm_registry import AlgorithmRegistry
from core.config import ParamType
from gui.widgets import (
  check_prerequisites,
  render_architecture_editor,
  render_param_input as render_param_input_widget,
)


AUTOENCODER_ALGOS = {
  'scdeepcluster',
  'sc_mae',
  'scname',
  'sccdcg',
}

ARCH_PARAMS_TO_SKIP = {
  'scdeepcluster': {'encoder_layers', 'decoder_layers', 'z_dim'},
  'sc_mae': {'hidden_size'},
  'scname': {'hidden_dims'},
  'sccdcg': {'encoder_layers', 'decoder_layers', 'embedding_dim'},
}

# Curated UI preconfigurations for specific algorithms.
# Values come from validated benchmark runs and only include explicitly selected params.
ALGORITHM_PRECONFIGS: Dict[str, Dict[str, Dict[str, Any]]] = {
}


def _parse_layer_list(text: str) -> List[int]:
  """Parse inputs like [256,64] or 256,64 into a list of ints."""
  if text is None:
    return []
  cleaned = text.strip()
  if not cleaned:
    return []

  # Strip common wrappers
  if cleaned[0] in '[({':
    cleaned = cleaned[1:]
  if cleaned and cleaned[-1] in '])}':
    cleaned = cleaned[:-1]

  cleaned = cleaned.replace(';', ',')
  parts = [p for p in re.split(r'[,\s]+', cleaned) if p]
  values = []
  for part in parts:
    try:
      values.append(int(part))
    except ValueError as exc:
      raise ValueError(f"Invalid layer size: '{part}'") from exc
  return values


def _format_layer_list(values: List[int]) -> str:
  if not values:
    return ''
  return '[' + ','.join(str(v) for v in values) + ']'


def _split_hidden_dims(hidden_dims: str) -> Tuple[List[int], Optional[int]]:
  dims = _parse_layer_list(hidden_dims)
  if not dims:
    return [], None
  return dims[:-1], dims[-1]


def _apply_architecture_inputs(
  algo_name: str,
  params: Dict[str, Any],
  encoder_list: List[int],
  latent_list: List[int],
  decoder_list: List[int]
) -> None:
  """Map standardized architecture inputs to algorithm-specific params.
  
  When lists are empty, resets to algorithm defaults to handle cleared fields.
  """
  latent_val = latent_list[0] if latent_list else None

  # Default values per algorithm (used when fields are cleared)
  DEFAULTS = {
    'scdeepcluster': {'encoder_layers': '256,64', 'decoder_layers': '64,256', 'z_dim': 32},
    'sccdcg': {'encoder_layers': '256', 'decoder_layers': '', 'embedding_dim': 16},
    'scname': {'hidden_dims': '256,64,32'},
    'sc_mae': {'hidden_size': 128, 'latent_dim': 128}, # Original uses hidden_size=128 for latent
  }

  defaults = DEFAULTS.get(algo_name, {})

  if algo_name == 'scdeepcluster':
    # Apply or reset encoder
    if encoder_list:
      params['encoder_layers'] = ','.join(str(v) for v in encoder_list)
    elif 'encoder_layers' not in params or not params['encoder_layers']:
      params['encoder_layers'] = defaults.get('encoder_layers', '256,64')
    # Apply or reset decoder
    if decoder_list:
      params['decoder_layers'] = ','.join(str(v) for v in decoder_list)
    elif 'decoder_layers' not in params or not params['decoder_layers']:
      params['decoder_layers'] = defaults.get('decoder_layers', '64,256')
    # Apply or reset latent
    if latent_val is not None:
      params['z_dim'] = latent_val
    elif 'z_dim' not in params:
      params['z_dim'] = defaults.get('z_dim', 32)
    return

  if algo_name == 'sccdcg':
    if encoder_list:
      params['encoder_layers'] = ','.join(str(v) for v in encoder_list)
    elif 'encoder_layers' not in params or not params['encoder_layers']:
      params['encoder_layers'] = defaults.get('encoder_layers', '256')
    if decoder_list:
      params['decoder_layers'] = ','.join(str(v) for v in decoder_list)
    if latent_val is not None:
      params['embedding_dim'] = latent_val
    elif 'embedding_dim' not in params:
      params['embedding_dim'] = defaults.get('embedding_dim', 16)
    return

  if algo_name == 'scname':
    dims = []
    if encoder_list:
      dims.extend(encoder_list)
    if latent_val is not None:
      dims.append(latent_val)
    if dims:
      params['hidden_dims'] = ','.join(str(v) for v in dims)
    elif 'hidden_dims' not in params or not params['hidden_dims']:
      params['hidden_dims'] = defaults.get('hidden_dims', '256,64,32')
    return

  if algo_name == 'sc_mae':
    # scMAE has hidden_size (encoder hidden layers) and latent_dim (final output)
    if latent_val is not None:
      params['latent_dim'] = latent_val
    elif 'latent_dim' not in params:
      params['latent_dim'] = defaults.get('latent_dim', 32)
    # hidden_size stays at 128 unless explicitly changed via other means
    if 'hidden_size' not in params:
      params['hidden_size'] = defaults.get('hidden_size', 128)
    return


def _render_architecture_section(algo_name: str, params: Dict[str, Any]) -> None:
  """Render a unified architecture editor for autoencoder-based algorithms."""
  st.markdown("#### Architecture")
  st.caption("Configure network layers visually (or enable advanced text mode).")

  encoder_default: List[int] = []
  latent_default = 32
  decoder_default: List[int] = []
  decoder_note = None

  if algo_name == 'scdeepcluster':
    encoder_default = _parse_layer_list(params.get('encoder_layers', '256,64'))
    latent_default = int(params.get('z_dim', 32))
    decoder_default = _parse_layer_list(params.get('decoder_layers', '64,256'))
  elif algo_name == 'sccdcg':
    encoder_default = _parse_layer_list(params.get('encoder_layers', '256'))
    latent_default = int(params.get('embedding_dim', 16))
    decoder_default = _parse_layer_list(params.get('decoder_layers', ''))
  elif algo_name == 'scname':
    enc_list, latent_val = _split_hidden_dims(params.get('hidden_dims', '256,64,32'))
    encoder_default = enc_list
    latent_default = int(latent_val if latent_val is not None else 32)
    decoder_note = "Decoder is symmetric for this algorithm (ignored if provided)."
  elif algo_name == 'sc_mae':
    # scMAE: encoder layers are fixed (256->128->128), latent=hidden_size in original
    encoder_default = [256, int(params.get('hidden_size', 128))]
    latent_default = int(params.get('latent_dim', 128))
    decoder_note = "Encoder layers are fixed (256->128->128). Original uses hidden_size (128) for latent."
  arch_min_size = 16
  arch_step = 16

  try:
    encoder_list, latent_list, decoder_list = render_architecture_editor(
      algo_name=algo_name,
      encoder_default=encoder_default,
      latent_default=latent_default,
      decoder_default=decoder_default,
      decoder_note=decoder_note,
      parse_layers=_parse_layer_list,
      format_layers=_format_layer_list,
      min_size=arch_min_size,
      max_size=2048,
      step=arch_step,
    )

    if latent_list and len(latent_list) > 1:
      st.warning("Latent space should contain a single value; using the first.")
      latent_list = [latent_list[0]]

    if algo_name == 'scname' and decoder_list:
      st.info("Decoder list ignored for this algorithm (symmetric decoder).")

    if algo_name == 'sc_mae' and (encoder_list or decoder_list):
      st.info("scMAE ignores encoder/decoder lists; only latent size is applied.")

    _apply_architecture_inputs(algo_name, params, encoder_list, latent_list, decoder_list)
  except ValueError as exc:
    st.warning(f"Invalid architecture input: {exc}")


def _on_algorithm_toggle(algo_name: str):
  """Callback for algorithm selection toggle."""
  checkbox_key = f"select_{algo_name}"
  is_selected = st.session_state.get(checkbox_key, False)

  if is_selected and algo_name not in st.session_state.selected_algorithms:
    st.session_state.selected_algorithms.append(algo_name)
  elif not is_selected and algo_name in st.session_state.selected_algorithms:
    st.session_state.selected_algorithms.remove(algo_name)


def render_algorithm_config_page():
  """Render the algorithm configuration page."""
  st.header("Algorithm Configuration")

  # Check prerequisites
  if not check_prerequisites(require_data=True, require_preprocessing=True):
    return

  # Get available algorithms
  algorithms = AlgorithmRegistry.get_all()

  if not algorithms:
    st.error("No algorithms registered. Please check the algorithms module.")
    return

  # ==========================================================================
  # GPU/CPU Settings
  # ==========================================================================
  st.subheader("Compute Device")
  
  # Check device availability
  try:
    import torch
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    
    if cuda_available:
      gpu_name = torch.cuda.get_device_name(0)
      gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) # GB
      gpu_info = f"{gpu_name} ({gpu_memory:.1f} GB)"
    else:
      gpu_info = None
      
    if mps_available:
      mps_info = "Apple Silicon GPU"
    else:
      mps_info = None
  except ImportError:
    cuda_available = False
    mps_available = False
    gpu_info = None
    mps_info = None
  
  # Initialize device preference
  if 'device_preference' not in st.session_state:
    st.session_state.device_preference = 'auto'
  
  col1, col2 = st.columns([2, 1])
  
  with col1:
    device_options = ['auto', 'cpu']
    device_labels = {
      'auto': 'Auto (best available device)',
      'cpu': 'Force CPU'
    }
    
    if cuda_available:
      device_options.append('cuda')
      device_labels['cuda'] = f'Force CUDA GPU ({gpu_info})'
    
    if mps_available:
      device_options.append('mps')
      device_labels['mps'] = f'Force MPS ({mps_info})'
    
    # Validate current preference
    if st.session_state.device_preference not in device_options:
      st.session_state.device_preference = 'auto'
    
    st.session_state.device_preference = st.selectbox(
      "Device Selection",
      options=device_options,
      index=device_options.index(st.session_state.device_preference),
      format_func=lambda x: device_labels[x],
      help="Select which compute device to use for deep learning algorithms.\n\n"
         "- **Auto**: Automatically select best available (CUDA > MPS > CPU)\n"
         "- **Force CPU**: Always use CPU (slower but most compatible)\n"
         "- **Force CUDA GPU**: Use NVIDIA GPU (fastest for large datasets)\n"
         "- **Force MPS**: Use Apple Silicon GPU (good for Mac users)"
    )
  
  with col2:
    if cuda_available:
      st.success("CUDA GPU Available")
      st.caption(gpu_info)
    elif mps_available:
      st.success("MPS Available")
      st.caption(mps_info)
    else:
      st.warning("No GPU detected")
      st.caption("Algorithms will run on CPU")
  
  # Show effective device
  effective_device = st.session_state.device_preference
  if effective_device == 'auto':
    if cuda_available:
      effective_device = 'cuda'
    elif mps_available:
      effective_device = 'mps'
    else:
      effective_device = 'cpu'
  
  if effective_device == 'cuda' and not cuda_available:
    st.error("CUDA GPU requested but not available. Falling back to CPU.")
    effective_device = 'cpu'
  
  if effective_device == 'mps' and not mps_available:
    st.error("MPS requested but not available. Falling back to CPU.")
    effective_device = 'cpu'
  
  st.session_state.effective_device = effective_device
  st.caption(f"**Effective device**: `{effective_device}`")
  
  # MPS compatibility note
  if effective_device == 'mps':
    st.info("**MPS Note**: Most algorithms support MPS, but some operations may fall back to CPU. "
        "scCDCG may have limited MPS support due to sparse matrix operations.")
  
  st.markdown("---")

  # Algorithm selection
  st.subheader("Select Algorithms")
  
  # Best Practices Info
  with st.expander("Best Practices: Input Data & Algorithms", expanded=False):
    st.markdown("""
    **Which input data should I use?**
    
    *  **Deep Learning (scDeepCluster, scCDCG, scMAE, scNAME):**
      *  **Recommendation:** Use **Raw Counts**.
      *  **Why?** These algorithms often model counts directly (ZINB loss), perform their own internal normalization (e.g. scMAE), or rely on raw data properties.
      *  *Using processed (normalized) data with these models is often theoretically incorrect (e.g. double normalization).*
      
    *  **Classical / Distance-based (PCA + K-Means, Louvain, Leiden, or HDBSCAN):**
      *  **Recommendation:** Use **Processed Data** (Normalized + Log + Scaled).
      *  **Why?** These methods rely on Euclidean distances or variance (PCA), which assume data is normalized and scaled to comparable ranges.
    
    **Note:** The interface below will attempt to select the correct default for you.
    """)


  # Initialize selected algorithms
  if 'selected_algorithms' not in st.session_state:
    st.session_state.selected_algorithms = []

  # Synchronize checkbox states with selected_algorithms on page load
  for algo_name in algorithms.keys():
    checkbox_key = f"select_{algo_name}"
    if checkbox_key not in st.session_state:
      st.session_state[checkbox_key] = algo_name in st.session_state.selected_algorithms

  # Display algorithms by category
  categories = AlgorithmRegistry.get_categories()

  for category in sorted(categories):
    with st.expander(f"{category.replace('_', ' ').title()}", expanded=True):
      category_algos = AlgorithmRegistry.get_by_category(category)
  
      for algo_name, algo_class in category_algos.items():
        info = algo_class.get_info()
  
        col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
  
        with col1:
          st.checkbox(
            "Select",
            key=f"select_{algo_name}",
            label_visibility="collapsed",
            on_change=_on_algorithm_toggle,
            args=(algo_name,)
          )
  
        with col2:
          st.markdown(f"**{info.display_name}**")
          st.caption(f"{info.description}")
          # Display preprocessing requirements if present
          if info.preprocessing_notes:
            if info.has_internal_preprocessing:
              st.warning(f"**Note**: {info.preprocessing_notes}")
            else:
              st.info(f"{info.preprocessing_notes}")
          # Display MPS warnings if MPS is active and algorithm has notes
          if (st.session_state.get('effective_device') == 'mps' and 
            hasattr(info, 'mps_notes') and info.mps_notes):
            st.caption(f"MPS: {info.mps_notes}")
  
        with col3:
          if info.requires_gpu:
            st.markdown("**GPU Required**")
          elif info.category == 'deep_learning':
            # Check MPS compatibility for deep learning algorithms
            if hasattr(info, 'mps_compatible') and not info.mps_compatible:
              st.markdown("**GPU/CPU (No MPS)**")
            else:
              st.markdown("**GPU/MPS Supported**")
          else:
            st.markdown("**CPU Only**")
        
        st.divider()

  # Hyperparameter configuration
  if st.session_state.selected_algorithms:
    st.markdown("---")

    # Check if any autoencoder-based algorithm is selected
    autoencoder_algorithms = [
      'scdeepcluster',
      'sc_mae',
      'scname',
      'sccdcg',
    ]
    selected_ae_algos = [a for a in st.session_state.selected_algorithms if a in autoencoder_algorithms]

    if selected_ae_algos:
      _render_global_network_config(selected_ae_algos)
      st.markdown("---")

    st.subheader("Hyperparameter Configuration")
    col_fast, col_fast_note = st.columns([1, 3])
    with col_fast:
      if st.button("Fast test", help="Set all epoch parameters to 10 for selected algorithms."):
        _apply_fast_test_epochs(st.session_state.selected_algorithms, fast_epochs=10)
        st.success("Fast test applied: epochs set to 10.")
    with col_fast_note:
      st.caption("Quickly reduce training time by setting all epoch-related parameters to 10.")

    # Initialize params in session state
    if 'algorithm_params' not in st.session_state:
      st.session_state.algorithm_params = {}

    # Create tabs for each selected algorithm
    tabs = st.tabs([
      AlgorithmRegistry.get(name).get_info().display_name
      for name in st.session_state.selected_algorithms
    ])

    for tab, algo_name in zip(tabs, st.session_state.selected_algorithms):
      with tab:
        _render_algorithm_params(algo_name)

  # Summary
  st.markdown("---")
  st.subheader("Configuration Summary")

  if st.session_state.selected_algorithms:
    st.success(f"Selected {len(st.session_state.selected_algorithms)} algorithm(s)")

    for algo_name in st.session_state.selected_algorithms:
      info = AlgorithmRegistry.get(algo_name).get_info()
      st.write(f"- {info.display_name}")

    # Export hyperparameters section
    st.markdown("---")
    st.subheader("Export Configuration")
    _render_export_section()
  else:
    st.info("No algorithms selected. Select at least one algorithm to proceed.")


def _render_global_network_config(selected_ae_algos: List[str]):
  """Render global network architecture configuration for autoencoder algorithms."""
  st.subheader("Global Network Architecture")
  st.caption("Configure encoder/decoder layers for all autoencoder-based algorithms. "
        "Leave empty to use algorithm-specific defaults.")

  # Initialize global network config in session state
  if 'global_network_config' not in st.session_state:
    st.session_state.global_network_config = {
      'enabled': False,
      'encoder_layers': '',
      'decoder_layers': '',
      'symmetric': True,
      'z_dim': None
    }

  config = st.session_state.global_network_config

  # Enable/disable global config
  config['enabled'] = st.checkbox(
    "Use global network architecture",
    value=config['enabled'],
    help="When enabled, the encoder/decoder layers specified here will be applied to all selected autoencoder algorithms."
  )

  if config['enabled']:
    col1, col2 = st.columns(2)

    with col1:
      config['encoder_layers'] = st.text_input(
        "Encoder Layers",
        value=config['encoder_layers'],
        placeholder="e.g., 512,256,64",
        help="Comma-separated list of layer sizes for the encoder. "
           "Example: '512,256,64' means input -> 512 -> 256 -> 64 -> latent"
      )

    with col2:
      config['symmetric'] = st.checkbox(
        "Symmetric (auto-generate decoder)",
        value=config['symmetric'],
        help="Automatically generate decoder as mirror of encoder. "
           "If encoder is '512,256,64', decoder will be '64,256,512'."
      )

    if not config['symmetric']:
      config['decoder_layers'] = st.text_input(
        "Decoder Layers",
        value=config['decoder_layers'],
        placeholder="e.g., 64,256,512",
        help="Comma-separated list of layer sizes for the decoder."
      )
    else:
      # Auto-generate decoder
      if config['encoder_layers']:
        enc_list = [x.strip() for x in config['encoder_layers'].split(',') if x.strip()]
        config['decoder_layers'] = ','.join(reversed(enc_list))
        st.info(f"Auto-generated decoder: **{config['decoder_layers']}**")
      else:
        config['decoder_layers'] = ''

    # Latent dimension
    z_dim_input = st.number_input(
      "Latent Dimension (z_dim)",
      min_value=0,
      max_value=512,
      value=config['z_dim'] if config['z_dim'] else 0,
      step=8,
      help="Dimension of the latent space. Set to 0 to use algorithm defaults. "
         "Typical values: 16, 32, 64."
    )
    config['z_dim'] = z_dim_input if z_dim_input > 0 else None

    # Preview and apply
    if config['encoder_layers']:
      st.markdown("##### Architecture Preview")
      enc_list = [x.strip() for x in config['encoder_layers'].split(',') if x.strip()]
      dec_list = [x.strip() for x in config['decoder_layers'].split(',') if x.strip()] if config['decoder_layers'] else list(reversed(enc_list))
      z = config['z_dim'] if config['z_dim'] else "default"

      st.code(f"Encoder: input -> {' -> '.join(enc_list)} -> z_dim({z})\n"
          f"Decoder: z_dim({z}) -> {' -> '.join(dec_list)} -> output")

      # Show which algorithms will be affected
      st.markdown("**Algorithms affected:**")
      affected_algos = []

      # Mapping for display
      algo_layer_info = {
        'scdeepcluster': ('encoder_layers', 'decoder_layers'),
        'sc_mae': ('hidden_size', '(fixed)'),
        'scname': ('hidden_dims', '(symmetric)'),
        'sccdcg': ('encoder_layers', 'decoder_layers'),
      }

      for algo in selected_ae_algos:
        info = algo_layer_info.get(algo, ('layers', 'layers'))
        affected_algos.append(f"- **{algo}**: {info[0]} / {info[1]}")

      st.markdown('\n'.join(affected_algos))

      # Apply to algorithm params
      _apply_global_network_config(selected_ae_algos, config)

  st.session_state.global_network_config = config


def _apply_global_network_config(selected_ae_algos: List[str], config: Dict[str, Any]):
  """Apply global network configuration to algorithm parameters."""
  if not config['enabled'] or not config['encoder_layers']:
    return

  # Initialize algorithm params if needed
  if 'algorithm_params' not in st.session_state:
    st.session_state.algorithm_params = {}

  encoder_layers = config['encoder_layers']
  decoder_layers = config['decoder_layers']
  z_dim = config['z_dim']

  # Mapping from global config to algorithm-specific parameter names
  layer_param_mapping = {
    'scdeepcluster': {
      'encoder': 'encoder_layers',
      'decoder': 'decoder_layers',
      'z_dim': 'z_dim'
    },
    'sc_mae': {
      'encoder': None,
      'decoder': None,
      'z_dim': 'latent_dim' # Original: latent_dim=hidden_size=128
    },
    'scname': {
      'encoder': 'hidden_dims',
      'decoder': None,
      'z_dim': 'hidden_dims'
    },
    'sccdcg': {
      'encoder': 'encoder_layers',
      'decoder': 'decoder_layers',
      'z_dim': 'embedding_dim'
    },
  }

  enc_list = _parse_layer_list(encoder_layers)
  dec_list = _parse_layer_list(decoder_layers)

  for algo in selected_ae_algos:
    if algo not in st.session_state.algorithm_params:
      st.session_state.algorithm_params[algo] = {}

    mapping = layer_param_mapping.get(algo, {})

    if mapping.get('encoder'):
      if algo == 'scname':
        dims = enc_list[:]
        latent_val = z_dim
        if latent_val is None:
          existing = _parse_layer_list(
            st.session_state.algorithm_params[algo].get('hidden_dims', '')
          )
          if existing:
            latent_val = existing[-1]
        if latent_val is not None:
          dims.append(latent_val)
        if dims:
          st.session_state.algorithm_params[algo][mapping['encoder']] = ','.join(
            str(v) for v in dims
          )
      elif encoder_layers:
        st.session_state.algorithm_params[algo][mapping['encoder']] = encoder_layers

    if decoder_layers and mapping.get('decoder'):
      st.session_state.algorithm_params[algo][mapping['decoder']] = decoder_layers

    if z_dim is not None and mapping.get('z_dim'):
      if algo == 'scname':
        # handled above via hidden_dims
        pass
      else:
        st.session_state.algorithm_params[algo][mapping['z_dim']] = z_dim


def _apply_fast_test_epochs(selected_algos: List[str], fast_epochs: int = 10) -> None:
  """Set all epoch-like parameters to a small value for quick testing."""
  if 'algorithm_params' not in st.session_state:
    st.session_state.algorithm_params = {}

  for algo_name in selected_algos:
    algo_class = AlgorithmRegistry.get(algo_name)
    hyperparams = algo_class.get_hyperparameters()

    if algo_name not in st.session_state.algorithm_params:
      st.session_state.algorithm_params[algo_name] = {
        hp.name: hp.default for hp in hyperparams
      }

    params = st.session_state.algorithm_params[algo_name]

    for hp in hyperparams:
      if hp.param_type != ParamType.INTEGER:
        continue
      if 'epoch' not in hp.name.lower():
        continue

      target = int(fast_epochs)
      if hp.min_value is not None:
        target = max(int(hp.min_value), target)
      if hp.max_value is not None:
        target = min(int(hp.max_value), target)

      params[hp.name] = target
      st.session_state[f"{algo_name}_{hp.name}"] = target

    st.session_state.algorithm_params[algo_name] = params


def _sync_ae_architecture_editor_state(algo_name: str, params: Dict[str, Any]) -> None:
  """Keep architecture editor widget state aligned after preset loading."""
  prefix = f"{algo_name}_arch_editor"

  encoder_raw = params.get('encoder_layers', '')
  decoder_raw = params.get('decoder_layers', '')
  latent_raw = params.get('z_dim', None)

  encoder_values: List[int] = []
  if encoder_raw not in (None, ''):
    try:
      encoder_values = _parse_layer_list(str(encoder_raw))
    except ValueError:
      encoder_values = []
  decoder_values: List[int] = []
  if decoder_raw not in (None, ''):
    try:
      decoder_values = _parse_layer_list(str(decoder_raw))
    except ValueError:
      decoder_values = []

  if encoder_values:
    st.session_state[f"{prefix}_encoder"] = encoder_values
    st.session_state[f"{prefix}_enc_txt"] = _format_layer_list(encoder_values)
  if decoder_raw in (None, ''):
    st.session_state[f"{prefix}_decoder"] = []
    st.session_state[f"{prefix}_dec_txt"] = ''
  elif decoder_values:
    st.session_state[f"{prefix}_decoder"] = decoder_values
    st.session_state[f"{prefix}_dec_txt"] = _format_layer_list(decoder_values)

  if latent_raw not in (None, ''):
    try:
      latent_value = int(latent_raw)
      st.session_state[f"{prefix}_latent"] = latent_value
      st.session_state[f"{prefix}_lat_txt"] = _format_layer_list([latent_value])
    except (TypeError, ValueError):
      pass


def _apply_algorithm_preconfiguration(
  algo_name: str,
  params: Dict[str, Any],
  hyperparams: List[Any],
  preset_name: str
) -> Dict[str, Any]:
  """Apply a named algorithm preconfiguration to current params and widget state."""
  presets_for_algo = ALGORITHM_PRECONFIGS.get(algo_name, {})
  preset_values = presets_for_algo.get(preset_name, {})
  if not preset_values:
    return params

  # Include UI-only fields and compatibility aliases in addition to real hyperparams.
  allowed_param_names = {hp.name for hp in hyperparams}
  allowed_param_names.update({
    'input_type',
    'device',
    'encoder_layers',
    'hidden_layers',
    'seed',
    'capture_embedding_snapshots',
  })

  updated_params = dict(params)
  for param_name, value in preset_values.items():
    if param_name not in allowed_param_names:
      continue
    updated_params[param_name] = value
    st.session_state[f"{algo_name}_{param_name}"] = value

  if algo_name in AUTOENCODER_ALGOS:
    _sync_ae_architecture_editor_state(algo_name, updated_params)

  st.session_state.algorithm_params[algo_name] = updated_params
  return updated_params


def _render_algorithm_preconfiguration(
  algo_name: str,
  params: Dict[str, Any],
  hyperparams: List[Any]
) -> Dict[str, Any]:
  """Render optional algorithm preconfiguration controls."""
  presets_for_algo = ALGORITHM_PRECONFIGS.get(algo_name, {})
  if not presets_for_algo:
    return params

  st.markdown("#### Preconfiguration")
  preset_names = list(presets_for_algo.keys())
  selected_preset = st.selectbox(
    "Preset",
    options=preset_names,
    key=f"{algo_name}_preconfig_name",
    help="Load a predefined parameter set for this algorithm.",
  )
  st.caption("This preconfiguration updates algorithm parameters only (not preprocessing settings).")

  if st.button("Load preconfiguration", key=f"{algo_name}_load_preconfig", width="stretch"):
    params = _apply_algorithm_preconfiguration(algo_name, params, hyperparams, selected_preset)
    st.success(f"Preconfiguration `{selected_preset}` loaded.")
    st.rerun()

  return params


def _render_algorithm_params(algo_name: str):
  """Render parameter inputs for a specific algorithm."""
  algo_class = AlgorithmRegistry.get(algo_name)
  hyperparams = algo_class.get_hyperparameters()

  # Initialize params for this algorithm
  if algo_name not in st.session_state.algorithm_params:
    st.session_state.algorithm_params[algo_name] = {
      hp.name: hp.default for hp in hyperparams
    }
    # Auto-apply the first available preset (best config) on first initialization
    presets = ALGORITHM_PRECONFIGS.get(algo_name, {})
    if presets:
      first_preset_name = next(iter(presets))
      st.session_state.algorithm_params[algo_name] = _apply_algorithm_preconfiguration(
        algo_name,
        st.session_state.algorithm_params[algo_name],
        hyperparams,
        first_preset_name,
      )

  params = st.session_state.algorithm_params[algo_name]
  params = _render_algorithm_preconfiguration(algo_name, params, hyperparams)
  
  # Add help expander
  with st.expander(f"{algo_class.get_info().display_name} Tuning Guide"):
    st.markdown(f"""
    **Algorithm**: {algo_class.get_info().display_name}
    
    **Tuning Tips**:
    - **Number of Clusters**: If unknown, try running PCA+Leiden first to estimate, or start with 10-15.
    - **Epochs**: Increase if loss hasn't converged (flatlined). Decrease if it converges quickly to save time.
    - **Latent Dimension**: 32 or 64 is usually optimal for scRNA-seq.
    
    For detailed parameter explanations, please visit the **Documentation** page.
    """)
    
 


  # =========================================================
  # INPUT DATA SELECTION
  # =========================================================
  st.markdown("#### Input Data")
  
  # Get recommendation from AlgorithmInfo
  algo_info = algo_class.get_info()
  recommended = getattr(algo_info, 'recommended_data', 'either')
  
  # Determine default based on algorithm recommendation
  if recommended == 'raw':
    default_input = 'raw_original' # Default to raw/original for ZINB algos
  elif recommended == 'preprocessed':
    default_input = 'processed'
  else:
    default_input = 'processed'
    
  # Special overrides if needed (e.g. sccdcg strongly prefers original)
  # Special overrides if needed (e.g. sccdcg strongly prefers original)
  if algo_name == 'sccdcg':
    default_input = 'raw_filtered'
    
  # User requested Strict Clarity:
  # 'raw_original' (unfiltered) is technically available in the backend but is 
  # explicitly overridden to 'raw_filtered' in analysis_runner.py to prevent 
  # index mismatches (metrics require same cell count).
  # Therefore, we remove it from the UI to avoid confusion.
  input_options = ['processed', 'raw_filtered']
  
  input_labels = {
    'processed': 'Processed (Floats: LogNorm + Scaled + HVG)',
    'raw_filtered': 'Raw Counts (Integers: QC-Filtered + HVG)',
  }
  
  # Check if we have a saved preference for this algo, otherwise use smart default
  current_input = params.get('input_type', default_input)
  
  # Handle legacy saved saves where 'raw_original' might be selected
  if current_input == 'raw_original':
    current_input = 'raw_filtered'
    
  if current_input not in input_options:
    current_input = default_input
    
  selected_input = st.selectbox(
    "Input Data Type",
    options=input_options,
    index=input_options.index(current_input),
    format_func=lambda x: input_labels[x],
    key=f"{algo_name}_input_type",
    help=f"Select which version of the data to use (Strictly based on script behavior).\n\n"
       f"**Algorithm Recommendation**: {recommended.upper()}\n"
       "- **Processed**: Floats. Normalized, Log-Transformed, Scaled. Subset to **HVGs only**.\n"
       "- **Raw Counts**: Integers. No normalization. QC-Filtered cells. Subset to **HVGs only**.\n\n"
       "Note: 'Original/Unfiltered' option is hidden because the benchmark enforces QC-filtered cells for consistent metric evaluation."
  )
  
  params['input_type'] = selected_input
  
  # Map input_type to use_raw_data for backward compatibility with algorithms
  # This ensures that selecting "Processed" forces the algorithm to use data.X (batch-corrected)
  # Check if batch correction is active in preprocessing
  preprocess_params = st.session_state.get('preprocessing_params', {})
  do_batch_correction = preprocess_params.get('do_batch_correction', False)
  
  # If not explicitly enabled in preprocessing, check if it was applied during upload
  if not do_batch_correction and 'data_handler' in st.session_state and st.session_state.data_handler is not None:
    handler_info = st.session_state.data_handler.get_info()
    if handler_info.get('batch_correction_applied', False):
      do_batch_correction = True

  if selected_input == 'processed':
    params['use_raw_data'] = False
    
    # User requested automation: if batch correction + processed, show info box
    if do_batch_correction:
      st.info(
        "**Automatic Optimization**: Internal preprocessing (normalization/scaling) has been **disabled** "
        "because Batch Correction is active and you are using **Processed data**. "
        "This ensures the batch correction effect is preserved."
      )
  else:
    params['use_raw_data'] = True
  
  # Validation Warnings
  if recommended == 'raw' and selected_input == 'processed':
    if not do_batch_correction:
      st.warning(
        f"**Warning**: {algo_info.display_name} recommends **Raw Counts** (usually for ZINB loss). "
        "Using processed data (log-transformed/scaled) is theoretically incorrect and may lead to poor performance."
      )
    else:
      # If batch correction is on, the warning is less relevant because correction is necessary
      st.caption(f"Note: {algo_info.display_name} usually prefers raw data, but Processed data is used here to leverage Batch Correction.")
  elif recommended == 'preprocessed' and 'raw' in selected_input:
    st.warning(
      f"**Warning**: {algo_info.display_name} recommends **Processed Data**. "
      "Using raw counts without normalization/scaling may cause convergence issues (PCA/Euclidean distance)."
    )
  
  st.markdown("---")

  # Unified architecture editor for autoencoder-based algorithms
  if algo_name in AUTOENCODER_ALGOS:
    _render_architecture_section(algo_name, params)
    st.markdown("---")

  # Group by category (skip architecture params if unified editor is used)
  skip_params = ARCH_PARAMS_TO_SKIP.get(algo_name, set())
  categories = {}
  for hp in hyperparams:
    if hp.name in skip_params or hp.name == 'use_raw_data':
      continue
    if hp.category not in categories:
      categories[hp.category] = []
    categories[hp.category].append(hp)

  # Show basic params first
  for category, hps in categories.items():
    basic_hps = [hp for hp in hps if not hp.advanced]
    if basic_hps:
      st.markdown(f"#### {category}")
      for hp in basic_hps:
        params[hp.name] = render_param_input_widget(
          hp=hp,
          current_value=params.get(hp.name, hp.default),
          key_prefix=algo_name,
          compact=True,
          bounded_float_mode="precision_slider",
        )

  # Show advanced params in expander
  advanced_hps = [hp for hp in hyperparams if hp.advanced]
  if advanced_hps:
    with st.expander("Advanced Parameters"):
      for hp in advanced_hps:
        params[hp.name] = render_param_input_widget(
          hp=hp,
          current_value=params.get(hp.name, hp.default),
          key_prefix=algo_name,
          compact=True,
          bounded_float_mode="precision_slider",
        )

  # ==========================================================================
  # ALGORITHM-SPECIFIC WARNINGS
  # ==========================================================================

  # scMAE: Warning about own preprocessing
  if algo_name == 'sc_mae':
    use_own_preprocessing = params.get('use_own_preprocessing', True)
    n_hvg = params.get('n_hvg', 1000)

    if use_own_preprocessing:
      st.warning(
        f"**scMAE Own Preprocessing Active**\n\n"
        f"scMAE will **ignore** the benchmark's HVG selection and apply its own preprocessing:\n"
        f"- Retrieves **raw counts** (before benchmark HVG selection)\n"
        f"- Applies: `normalize_per_cell` `log1p` `HVG selection ({n_hvg} genes)` `scale` `SVD imputation`\n\n"
        f"This ensures faithful reproduction of the original scMAE paper methodology."
      )

      # Additional warning in benchmark mode
      preprocess_params = st.session_state.get('preprocessing_params', {})
      benchmark_hvg = preprocess_params.get('n_top_genes', 2000)
      if benchmark_hvg > 0 and benchmark_hvg != n_hvg:
        st.info(
          f"**Benchmark vs scMAE HVG**: The benchmark selects **{benchmark_hvg}** HVGs for other algorithms, "
          f"but scMAE will select **{n_hvg}** HVGs from the complete gene set."
        )
    else:
      st.info(
        "**scMAE using benchmark preprocessing**: scMAE will use the benchmark's preprocessed data as-is. "
        "This may produce different results than the original paper."
      )

  # Save params
  st.session_state.algorithm_params[algo_name] = params


def _generate_cli_command(compact: bool = False, n_repetitions: int = 1, seed: int = 42, state_source: Optional[Dict[str, Any]] = None) -> str:
  """
  Generate a CLI command from the current Streamlit configuration.

  This generator targets the real run parser in `src/scrbenchmark/cli.py` and
  keeps benchmark and standard modes aligned with UI state.
  """
  if state_source is None:
    get_val = lambda k, default=None: st.session_state.get(k, default)
  else:
    get_val = lambda k, default=None: state_source.get(k, default)

  parts = ["./scrbenchmark run"]
  warnings: List[str] = []

  def _append(flag: str, value: Any = None) -> None:
    if value is None:
      parts.append(flag)
    else:
      parts.append(f"{flag} {shlex.quote(str(value))}")

  def _to_float(value: Any, default: float) -> float:
    try:
      return float(value)
    except Exception:
      return float(default)

  # ---------------------------------------------------------------------------
  # Data & algorithms
  # ---------------------------------------------------------------------------
  data_file = get_val('uploaded_file_path')
  if isinstance(data_file, str):
    data_file = data_file.strip()
  else:
    data_file = ""

  if not data_file:
    data_file = "data/your_data.h5ad"
    warnings.append("No input dataset path found in UI state; using placeholder --data path.")
  elif data_file.startswith("# Reference dataset:"):
    ref_name = data_file.split(":", 1)[1].strip() if ":" in data_file else data_file
    data_file = "data/your_data.h5ad"
    warnings.append(
      f"Reference dataset '{ref_name}' is loaded in UI memory; replace --data with a local file path for CLI."
    )
  elif not os.path.isabs(data_file) and not data_file.startswith("data/"):
    data_file = f"data/{data_file}"
  _append("--data", data_file)

  selected_algos = [str(a) for a in (get_val('selected_algorithms', []) or []) if a]
  if selected_algos:
    _append("--algorithms", ",".join(selected_algos))
  else:
    warnings.append("No algorithms selected.")

  # ---------------------------------------------------------------------------
  # Mode detection
  # ---------------------------------------------------------------------------
  benchmark_setup = get_val('benchmark_setup', {}) or {}
  benchmark_runtime = get_val('benchmark_settings', {}) or {}
  is_benchmark_mode = bool(
    get_val('benchmark_configured', False)
    and isinstance(benchmark_setup, dict)
    and benchmark_setup.get('mode') == 'benchmark'
  )
  benchmark_settings: Dict[str, Any] = {}
  if is_benchmark_mode:
    if isinstance(benchmark_setup, dict):
      benchmark_settings.update(benchmark_setup.get('original_settings', {}) or {})
    if isinstance(benchmark_runtime, dict):
      benchmark_settings.update(benchmark_runtime)
  benchmark_use_validation = bool(benchmark_settings.get('use_validation', True)) if is_benchmark_mode else False
  benchmark_val_ratio = _to_float(benchmark_settings.get('val_ratio', 0.1), 0.1) if benchmark_use_validation else 0.0
  has_validation_split = bool(is_benchmark_mode and benchmark_use_validation and benchmark_val_ratio > 0.0)

  # ---------------------------------------------------------------------------
  # Execution flags
  # ---------------------------------------------------------------------------
  try:
    n_rep = int(n_repetitions)
  except Exception:
    n_rep = 1
  if n_rep > 1:
    _append("--n-repeats", n_rep)

  try:
    seed_val = int(seed)
  except Exception:
    seed_val = 42
  if seed_val != 42:
    _append("--seed", seed_val)

  if is_benchmark_mode and isinstance(benchmark_runtime, dict):
    compute_scib_metrics = bool(
      benchmark_runtime.get(
        'compute_scib_metrics',
        get_val('analysis_compute_scib_metrics', True),
      )
    )
  else:
    compute_scib_metrics = bool(get_val('analysis_compute_scib_metrics', True))
  if not compute_scib_metrics:
    _append("--no-scib-metrics")

  # ---------------------------------------------------------------------------
  # Preprocessing flags
  # ---------------------------------------------------------------------------
  preprocess_params = get_val('preprocessing_params', {}) or {}
  if not isinstance(preprocess_params, dict):
    preprocess_params = {}
  include_preproc_defaults = bool(
    get_val('data_preprocessed', False)
    or get_val('benchmark_processed', None) is not None
  )

  if preprocess_params.get('skip', False):
    _append("--skip-preprocessing")

  toggle_map = {
    'do_cell_filtering': '--no-cell-filtering',
    'do_gene_filtering': '--no-gene-filtering',
    'do_normalization': '--no-normalization',
    'do_log_transform': '--no-log-transform',
    'do_hvg': '--no-hvg',
    'do_scaling': '--no-scaling',
  }
  for key, flag in toggle_map.items():
    if key in preprocess_params and not bool(preprocess_params.get(key, True)):
      _append(flag)

  # Keep this explicit because UI default (10000) differs from CLI parser default (None).
  max_genes = preprocess_params.get('max_genes_per_cell', None)
  if max_genes not in (None, "", 0):
    _append("--max-genes-per-cell", max_genes)

  def _add_preproc_param(name: str, cli_arg: str, default: Any) -> None:
    if name not in preprocess_params:
      return
    value = preprocess_params.get(name)
    if value is None:
      return
    if include_preproc_defaults or value != default:
      _append(f"--{cli_arg}", value)

  _add_preproc_param('n_top_genes', 'n-top-genes', 2000)
  _add_preproc_param('min_genes_per_cell', 'min-genes-per-cell', 100)
  _add_preproc_param('min_cells_per_gene', 'min-cells-per-gene', 3)
  _add_preproc_param('target_sum', 'target-sum', 20000)
  _add_preproc_param('scale_max_value', 'scale-max-value', 10.0)
  _add_preproc_param('hvg_flavor', 'hvg-flavor', 'seurat')
  _add_preproc_param('hvg_strategy', 'hvg-strategy', 'train_only')

  # Dropout simulation / noise
  dropout_method = preprocess_params.get('dropout_method', None)
  if dropout_method not in (None, "") and (include_preproc_defaults or dropout_method != 'none'):
    _append("--dropout-method", dropout_method)

  if dropout_method == 'simple':
    dropout_rate = preprocess_params.get('dropout_rate', 0.2)
    if include_preproc_defaults or dropout_rate != 0.2:
      _append("--dropout-rate", dropout_rate)
  elif dropout_method == 'logistic':
    dropout_mid = preprocess_params.get('dropout_mid', 0.0)
    dropout_shape = preprocess_params.get('dropout_shape', -1.0)
    if include_preproc_defaults or dropout_mid != 0.0:
      _append("--dropout-mid", dropout_mid)
    if include_preproc_defaults or dropout_shape != -1.0:
      _append("--dropout-shape", dropout_shape)

  noise_level = preprocess_params.get('noise_level', 0.0)
  if noise_level is not None and (include_preproc_defaults or noise_level != 0.0):
    _append("--noise-level", noise_level)

  # HVG enrichment with supervised markers
  ensure_markers = bool(preprocess_params.get('ensure_celltype_markers', False))
  if ensure_markers:
    _append("--ensure-celltype-markers")
    min_markers = preprocess_params.get('min_markers_per_celltype', 5)
    max_additional = preprocess_params.get('max_additional_hvg_genes', 500)
    if include_preproc_defaults or min_markers != 5:
      _append("--min-markers-per-celltype", min_markers)
    if include_preproc_defaults or max_additional != 500:
      _append("--max-additional-hvg-genes", max_additional)

  # ---------------------------------------------------------------------------
  # Batch correction flags
  # ---------------------------------------------------------------------------
  do_batch_correction = bool(preprocess_params.get('do_batch_correction', False))
  bc_params_source = dict(preprocess_params)

  data_handler = get_val('data_handler')
  if not do_batch_correction and data_handler is not None and hasattr(data_handler, 'get_info'):
    try:
      info = data_handler.get_info()
    except Exception:
      info = {}
    if isinstance(info, dict) and info.get('batch_correction_applied', False):
      do_batch_correction = True
      bc_info = info.get('batch_correction_info', {}) or {}
      stored_method = str(bc_info.get('method', 'scvi')).lower()
      if stored_method == 'scvi_dct':
        stored_method = 'scvi'
      bc_params_source.update({
        'batch_correction_method': stored_method,
        'batch_correction_batch_key': bc_info.get('batch_key', 'auto'),
        'batch_correction_labels_key': bc_info.get('labels_key', None),
        'batch_correction_epochs': bc_info.get('max_epochs', 100),
        'batch_correction_n_latent': bc_info.get('n_latent', 30),
        'batch_correction_batch_size': bc_info.get('batch_size', 128),
        'batch_correction_dct_weight': bc_info.get('dct_weight', 1.0),
        'batch_correction_corr_mse_weight': bc_info.get('corr_mse_weight', 0.1),
        'batch_correction_cycle_weight': bc_info.get('z_distance_cycle_weight', 5.0),
        'batch_correction_kl_weight': bc_info.get('kl_weight', 1.0),
        'batch_correction_prior': bc_info.get('prior', 'vamp'),
        'batch_correction_n_prior_components': bc_info.get('n_prior_components', 5),
        'batch_correction_embed_covariates': bc_info.get('embed_categorical_covariates', True),
      })

  if do_batch_correction:
    _append("--batch-correction")

    method = str(bc_params_source.get('batch_correction_method', 'scvi')).lower()
    if method == 'scvi_dct':
      method = 'scvi'
    if method not in (None, "") and (include_preproc_defaults or method != 'scvi'):
      _append("--batch-correction-method", method)

    batch_key = bc_params_source.get('batch_correction_batch_key', 'auto')
    if batch_key and (include_preproc_defaults or str(batch_key) != 'auto'):
      _append("--batch-key", batch_key)

    labels_key = bc_params_source.get('batch_correction_labels_key')
    if labels_key not in (None, ""):
      _append("--batch-correction-labels-key", labels_key)

    def _add_bc_param(name: str, cli_arg: str, default: Any) -> None:
      if name not in bc_params_source and not include_preproc_defaults:
        return
      val = bc_params_source.get(name, default)
      if val is None:
        return
      if include_preproc_defaults or val != default:
        _append(f"--{cli_arg}", val)

    if method == 'scvi':
      _add_bc_param('batch_correction_epochs', 'batch-correction-epochs', 100)
      _add_bc_param('batch_correction_n_latent', 'batch-correction-latent', 30)
      _add_bc_param('batch_correction_batch_size', 'batch-correction-batch-size', 128)
      _add_bc_param('batch_correction_dct_weight', 'batch-correction-dct-weight', 1.0)
      _add_bc_param('batch_correction_corr_mse_weight', 'batch-correction-corr-mse-weight', 0.1)
    elif method == 'sysvi':
      _add_bc_param('batch_correction_epochs', 'batch-correction-epochs', 200)
      _add_bc_param('batch_correction_n_latent', 'batch-correction-latent', 15)
      _add_bc_param('batch_correction_batch_size', 'batch-correction-batch-size', 128)
      _add_bc_param('batch_correction_cycle_weight', 'batch-correction-cycle-weight', 5.0)
      _add_bc_param('batch_correction_kl_weight', 'batch-correction-kl-weight', 1.0)
      _add_bc_param('batch_correction_prior', 'batch-correction-prior', 'vamp')
      _add_bc_param('batch_correction_n_prior_components', 'batch-correction-n-prior-components', 5)
      embed_covariates = bool(bc_params_source.get('batch_correction_embed_covariates', True))
      if include_preproc_defaults or embed_covariates is not True:
        _append("--batch-correction-embed-covariates" if embed_covariates else "--no-batch-correction-embed-covariates")

  # ---------------------------------------------------------------------------
  # Global network architecture
  # ---------------------------------------------------------------------------
  global_net_config = get_val('global_network_config', {}) or {}
  if isinstance(global_net_config, dict) and global_net_config.get('enabled'):
    encoder_layers = global_net_config.get('encoder_layers')
    if encoder_layers:
      _append("--encoder-layers", encoder_layers)
    if not global_net_config.get('symmetric', True):
      decoder_layers = global_net_config.get('decoder_layers')
      if decoder_layers:
        _append("--decoder-layers", decoder_layers)
    z_dim = global_net_config.get('z_dim')
    if z_dim not in (None, "", 0):
      _append("--z-dim", z_dim)

  # ---------------------------------------------------------------------------
  # Device
  # ---------------------------------------------------------------------------
  device = get_val('device_preference', 'auto')
  if device and str(device) != 'auto':
    _append("--device", device)



  # ---------------------------------------------------------------------------
  # Algorithm-specific --param
  # ---------------------------------------------------------------------------
  validation_opt = get_val('validation_optimization', {}) or {}
  enable_lr_optimization = bool(validation_opt.get('enable_lr_optimization', True))
  lr_opt_metric = str(validation_opt.get('lr_optimization_metric', 'NMI'))
  try:
    lr_opt_repeats = int(validation_opt.get('lr_optimization_repeats', 1))
  except Exception:
    lr_opt_repeats = 1

  lr_opt_scales = validation_opt.get('lr_optimization_scales', [100.0, 10.0, 1.0, 0.1])
  if not isinstance(lr_opt_scales, (list, tuple)):
    lr_opt_scales = [100.0, 10.0, 1.0, 0.1]
  normalized_scales = []
  for scale in lr_opt_scales:
    try:
      scale_val = float(scale)
      if scale_val > 0:
        normalized_scales.append(scale_val)
    except Exception:
      continue
  if not normalized_scales:
    normalized_scales = [100.0, 10.0, 1.0, 0.1]

  lr_opt_common_params = {
    'enable_lr_optimization': enable_lr_optimization,
    'lr_optimization_metric': lr_opt_metric,
    'lr_optimization_repeats': lr_opt_repeats,
    'lr_optimization_scales': normalized_scales,
  }

  global_net_enabled = bool(isinstance(global_net_config, dict) and global_net_config.get('enabled'))
  net_param_names = {
    'encoder_layers', 'decoder_layers', 'hidden_layers',
    'units_encoder', 'units_decoder', 'hidden_dims', 'z_dim',
  }

  algo_params = get_val('algorithm_params', {}) or {}
  for algo_name in selected_algos:
    params = algo_params.get(algo_name, {})
    if not isinstance(params, dict):
      params = {}

    default_params: Dict[str, Any] = {}
    try:
      algo_class = AlgorithmRegistry.get(algo_name)
      if algo_class:
        default_params = {hp.name: hp.default for hp in algo_class.get_hyperparameters()}
    except Exception:
      default_params = {}

    # CLI nuance: --seed is not automatically propagated into per-algorithm
    # params in run mode. We therefore inject random_state explicitly when no
    # non-default seed parameter will be emitted for this algorithm.
    random_state_non_default = (
      'random_state' in params
      and not (
        'random_state' in default_params
        and params.get('random_state') == default_params.get('random_state')
      )
    )
    seed_non_default = (
      'seed' in params
      and not (
        'seed' in default_params
        and params.get('seed') == default_params.get('seed')
      )
    )

    for param_name, value in params.items():
      param_name_for_cli = param_name

      if param_name == 'device':
        continue
      if global_net_enabled and param_name in net_param_names:
        continue


      if isinstance(value, bool):
        value_repr = "true" if value else "false"
      elif isinstance(value, (list, tuple, dict)):
        value_repr = json.dumps(value, separators=(",", ":"))
      elif value is None:
        value_repr = "null"
      else:
        value_repr = str(value)

      _append("--param", f"{algo_name}:{param_name_for_cli}={value_repr}")

    if not random_state_non_default and not seed_non_default:
      _append("--param", f"{algo_name}:random_state={seed_val}")

    if has_validation_split:
      _append("--param", f"{algo_name}:enable_lr_optimization={'true' if lr_opt_common_params['enable_lr_optimization'] else 'false'}")
      _append("--param", f"{algo_name}:lr_optimization_metric={lr_opt_common_params['lr_optimization_metric']}")
      _append("--param", f"{algo_name}:lr_optimization_repeats={lr_opt_common_params['lr_optimization_repeats']}")
      scales_json = json.dumps(lr_opt_common_params['lr_optimization_scales'], separators=(",", ":"))
      _append("--param", f"{algo_name}:lr_optimization_scales={scales_json}")

  # ---------------------------------------------------------------------------
  # Benchmark vs standard split options
  # ---------------------------------------------------------------------------
  if is_benchmark_mode:
    settings: Dict[str, Any] = dict(benchmark_settings)

    _append("--benchmark-mode")

    train_ratio = _to_float(settings.get('train_ratio', 0.7), 0.7)
    if not bool(settings.get('use_validation', True)):
      val_ratio = 0.0
      test_ratio = max(0.0, 1.0 - train_ratio)
    else:
      val_ratio = _to_float(settings.get('val_ratio', 0.1), 0.1)
      test_ratio = _to_float(settings.get('test_ratio', 1.0 - train_ratio - val_ratio), 1.0 - train_ratio - val_ratio)

    # Keep explicit for reproducibility.
    _append("--train-ratio", train_ratio)
    _append("--val-ratio", val_ratio)
    _append("--test-ratio", test_ratio)

    if settings.get('mode') == 'batch':
      train_batches = settings.get('train_batches', []) or []
      if train_batches:
        _append("--train-batches", ",".join(str(b) for b in train_batches))

      test_batches = settings.get('test_batches', None)
      if test_batches:
        _append("--test-batches", ",".join(str(b) for b in test_batches))

    batch_col = settings.get('batch_col')
    if batch_col not in (None, "", "auto"):
      _append("--stratify-by", batch_col)

    stratify_by_batch = bool(settings.get('stratify_by_batch', True))
    stratify_by_labels = bool(settings.get('stratify_by_labels', True))
    if stratify_by_batch and stratify_by_labels:
      stratify_mode = 'batch+labels'
    elif stratify_by_batch:
      stratify_mode = 'batch-only'
    elif stratify_by_labels:
      stratify_mode = 'labels-only'
    else:
      stratify_mode = 'none'
    _append("--stratify-mode", stratify_mode)

    label_col = settings.get('label_col')
    if label_col not in (None, ""):
      _append("--label-col", label_col)

    do_balance = bool(settings.get('balance', False) or settings.get('balance_batches', False))
    if do_balance:
      _append("--balance-batches")
      balance_target = settings.get('balance_target')
      if balance_target not in (None, "", "auto"):
        _append("--balance-target", balance_target)
      balance_mode = settings.get('balance_mode', 'eliminate')
      if balance_mode != 'eliminate':
        _append("--balance-mode", balance_mode)
  else:
    balance_settings = get_val('balance_settings_standard', {}) or {}
    if isinstance(balance_settings, dict) and balance_settings:
      _append("--balance-batches-standard")
      batch_col = balance_settings.get('batch_col')
      if batch_col not in (None, "", "auto"):
        _append("--batch-col-standard", batch_col)
      target = balance_settings.get('target')
      if target not in (None, "", "auto"):
        _append("--balance-target-standard", target)

    exclude_batches = get_val('exclude_batches')
    if isinstance(exclude_batches, list):
      exclude_batches = ",".join(str(b) for b in exclude_batches if b not in (None, ""))
    if exclude_batches:
      _append("--exclude-batches", exclude_batches)

  # ---------------------------------------------------------------------------
  # Output options
  # ---------------------------------------------------------------------------
  output_dir = get_val('output_dir', 'results')
  if not output_dir:
    output_dir = 'results'
  _append("--output", output_dir)

  if bool(get_val('save_labels', False)):
    _append("--save-labels")
  if bool(get_val('save_embeddings', False)):
    _append("--save-embeddings")
  if bool(get_val('save_csv', False) or get_val('csv', False)):
    _append("--csv")
  if bool(get_val('quiet', False)):
    _append("--quiet")

  # ---------------------------------------------------------------------------
  # Final formatting
  # ---------------------------------------------------------------------------
  if compact:
    cmd = " ".join(parts)
  else:
    if len(parts) <= 3:
      cmd = " ".join(parts)
    else:
      lines = [parts[0]]
      for part in parts[1:]:
        lines.append(f"  {part}")
      cmd = " \\\n".join(lines)

  if warnings and not compact:
    cmd += "\n# NOTE: " + "; ".join(warnings)
    cmd += "\n# You can also run: ./scrbenchmark generate-config --output config.yaml"

  return cmd


def _check_preprocessing_compatibility() -> List[str]:
  """Check if current preprocessing settings are fully compatible with CLI."""
  preprocess_params = st.session_state.get('preprocessing_params', {})
  issues = []

  if not preprocess_params.get('do_normalization', True):
    issues.append("Normalization disabled")
  if not preprocess_params.get('do_log_transform', True):
    issues.append("Log transform disabled")
  if not preprocess_params.get('do_hvg', True):
    issues.append("HVG selection disabled")
  if not preprocess_params.get('do_scaling', True):
    issues.append("Scaling disabled")

  return issues


def _render_export_section():
  """Render the export configuration section."""
  st.markdown("Export all hyperparameters for selected algorithms.")

  col1, col2, col3 = st.columns(3)

  with col1:
    json_data = _generate_json_export()
    st.download_button(
      label="Export JSON",
      data=json_data,
      file_name=f"hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
      mime="application/json",
      width="stretch"
    )

  with col2:
    yaml_data = _generate_yaml_export()
    st.download_button(
      label="Export YAML",
      data=yaml_data,
      file_name=f"hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
      mime="text/yaml",
      width="stretch"
    )

  with col3:
    txt_data = _generate_txt_export()
    st.download_button(
      label="Export TXT",
      data=txt_data,
      file_name=f"hyperparameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
      mime="text/plain",
      width="stretch"
    )

  # Preview
  with st.expander("Preview Configuration"):
    st.code(_generate_txt_export(), language=None)



def _generate_config_dict() -> Dict[str, Any]:
  """Generate configuration dictionary for export."""
  config = {
    "export_info": {
      "timestamp": datetime.now().isoformat(),
      "version": "1.0"
    },
    "preprocessing": st.session_state.get('preprocessing_params', {}),
    "algorithms": {}
  }

  for algo_name in st.session_state.selected_algorithms:
    algo_class = AlgorithmRegistry.get(algo_name)
    algo_info = algo_class.get_info()
    hyperparams = algo_class.get_hyperparameters()

    # Get current params or defaults
    current_params = st.session_state.get('algorithm_params', {}).get(algo_name, {})

    algo_config = {
      "display_name": algo_info.display_name,
      "description": algo_info.description,
      "category": algo_info.category,
      "requires_gpu": algo_info.requires_gpu,
      # Include input_type (custom UI choice not in HyperparameterConfig)
      "input_type": current_params.get('input_type', 'processed'),
      "parameters": {}
    }

    for hp in hyperparams:
      value = current_params.get(hp.name, hp.default)
      algo_config["parameters"][hp.name] = {
        "value": value,
        "display_name": hp.display_name,
        "description": hp.description,
        "type": hp.param_type.value,
        "default": hp.default,
        "category": hp.category,
        "advanced": hp.advanced
      }
      # Add constraints if available
      if hp.min_value is not None:
        algo_config["parameters"][hp.name]["min"] = hp.min_value
      if hp.max_value is not None:
        algo_config["parameters"][hp.name]["max"] = hp.max_value
      if hp.choices:
        algo_config["parameters"][hp.name]["choices"] = hp.choices

    config["algorithms"][algo_name] = algo_config

  return config


def _generate_json_export() -> str:
  """Generate JSON export of hyperparameters."""
  config = _generate_config_dict()
  return json.dumps(config, indent=2, ensure_ascii=False)


def _generate_yaml_export() -> str:
  """Generate YAML export of hyperparameters."""
  config = _generate_config_dict()

  def dict_to_yaml(d: Dict, indent: int = 0) -> str:
    """Simple YAML generator without external dependencies."""
    lines = []
    prefix = " " * indent

    for key, value in d.items():
      if isinstance(value, dict):
        lines.append(f"{prefix}{key}:")
        lines.append(dict_to_yaml(value, indent + 1))
      elif isinstance(value, list):
        lines.append(f"{prefix}{key}:")
        for item in value:
          if isinstance(item, dict):
            lines.append(f"{prefix} -")
            lines.append(dict_to_yaml(item, indent + 2))
          else:
            lines.append(f"{prefix} - {item}")
      elif isinstance(value, bool):
        lines.append(f"{prefix}{key}: {str(value).lower()}")
      elif isinstance(value, str):
        # Escape strings with special characters
        if any(c in value for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '%', '@', '`']):
          lines.append(f'{prefix}{key}: "{value}"')
        else:
          lines.append(f"{prefix}{key}: {value}")
      elif value is None:
        lines.append(f"{prefix}{key}: null")
      else:
        lines.append(f"{prefix}{key}: {value}")

    return "\n".join(lines)

  yaml_content = "# scDeepCluster Analysis Suite - Hyperparameters Configuration\n"
  yaml_content += f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
  yaml_content += dict_to_yaml(config)

  return yaml_content


def _generate_txt_export() -> str:
  """Generate human-readable TXT export of hyperparameters."""
  lines = []
  lines.append("=" * 70)
  lines.append("scDeepCluster Analysis Suite - Hyperparameters Configuration")
  lines.append("=" * 70)
  lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
  lines.append("")

  # Preprocessing params
  preprocessing = st.session_state.get('preprocessing_params', {})
  if preprocessing:
    lines.append("-" * 70)
    lines.append("PREPROCESSING PARAMETERS")
    lines.append("-" * 70)
    for key, value in preprocessing.items():
      lines.append(f" {key}: {value}")
    lines.append("")

  # Algorithm params
  for algo_name in st.session_state.selected_algorithms:
    algo_class = AlgorithmRegistry.get(algo_name)
    algo_info = algo_class.get_info()
    hyperparams = algo_class.get_hyperparameters()

    current_params = st.session_state.get('algorithm_params', {}).get(algo_name, {})

    lines.append("-" * 70)
    lines.append(f"ALGORITHM: {algo_info.display_name}")
    lines.append(f" Category: {algo_info.category}")
    lines.append(f" Description: {algo_info.description}")
    lines.append("-" * 70)

    # Group params by category
    categories = {}
    for hp in hyperparams:
      if hp.category not in categories:
        categories[hp.category] = []
      categories[hp.category].append(hp)

    for category, hps in categories.items():
      lines.append(f"\n [{category}]")
      for hp in hps:
        value = current_params.get(hp.name, hp.default)
        default_marker = "" if value == hp.default else " (modified)"
        advanced_marker = " [advanced]" if hp.advanced else ""

        lines.append(f"  {hp.display_name}:{advanced_marker}")
        lines.append(f"   Value: {value}{default_marker}")
        lines.append(f"   Default: {hp.default}")
        if hp.description:
          lines.append(f"   Description: {hp.description}")
        if hp.min_value is not None or hp.max_value is not None:
          range_str = f"[{hp.min_value}, {hp.max_value}]"
          lines.append(f"   Range: {range_str}")
        if hp.choices:
          lines.append(f"   Choices: {', '.join(map(str, hp.choices))}")

    lines.append("")

  lines.append("=" * 70)
  lines.append("END OF CONFIGURATION")
  lines.append("=" * 70)

  return "\n".join(lines)
