"""
Preprocessing page for the Streamlit interface.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import PREPROCESSING_PARAMS, ParamType, SplitConfig
from utils.dataset_splitter import get_batch_column, DatasetSplitter
from gui.widgets import (
  check_prerequisites,
  display_error,
  download_button as render_download_button,
  render_synced_number_input,
)
from gui.state_manager import KEYS, set_with_cascade

def render_split_ratios(prefix: str = "") -> dict:
  """Reusable component for inputting train/val/test ratios."""
  col1, col2, col3 = st.columns(3)
  
  with col1:
    train = st.slider(
      f"{prefix}Train Ratio", 
      min_value=SplitConfig.MIN_TRAIN_RATIO, 
      max_value=SplitConfig.MAX_TRAIN_RATIO, 
      value=SplitConfig.DEFAULT_TRAIN_RATIO, 
      step=0.05
    )
  with col2:
    val = st.slider(
      f"{prefix}Validation Ratio", 
      min_value=SplitConfig.MIN_VAL_RATIO, 
      max_value=SplitConfig.MAX_VAL_RATIO, 
      value=SplitConfig.DEFAULT_VAL_RATIO, 
      step=0.05
    )
  with col3:
    # Automatically calculate test ratio
    test = round(1.0 - train - val, 2)
    
    # Color code based on validity
    if test < SplitConfig.MIN_TEST_RATIO:
      st.metric(f"{prefix}Test Ratio", f"{test:.0%}", delta="Too Low", delta_color="inverse")
      st.error(f"Test ratio must be ≥ {SplitConfig.MIN_TEST_RATIO:.0%}")
      return None
    elif test > SplitConfig.MAX_TEST_RATIO:
       st.metric(f"{prefix}Test Ratio", f"{test:.0%}", delta="Too High", delta_color="inverse")
       st.error(f"Test ratio must be ≤ {SplitConfig.MAX_TEST_RATIO:.0%}")
       return None
    else:
      st.metric(f"{prefix}Test Ratio", f"{test:.0%}")
      
  return {"train": train, "val": val, "test": test}

def render_split_distribution_chart(split_info: dict):
  """Render a visual chart of the split distribution."""
  import matplotlib.pyplot as plt
  import pandas as pd
  import seaborn as sns
  
  # Prepare data
  data = {
    'Split': ['Train', 'Validation', 'Test'],
    'Count': [
      split_info.get('n_train', 0),
      split_info.get('n_val', 0),
      split_info.get('n_test', 0)
    ]
  }
  df = pd.DataFrame(data)
  total = df['Count'].sum()
  if total == 0: return
  
  df['Percentage'] = df['Count'] / total * 100
  
  # Plot
  fig, ax = plt.subplots(figsize=(6, 1.5))
  
  # Horizontal stacked bar
  left = 0
  colors = ['#2ecc71', '#f39c12', '#e74c3c'] # Green, Orange, Red
  
  for i, row in df.iterrows():
    ax.barh(0, row['Count'], left=left, color=colors[i], label=f"{row['Split']} ({row['Percentage']:.1f}%)")
    # Add text label in center of bar if wide enough
    if row['Percentage'] > 10:
      ax.text(left + row['Count']/2, 0, f"{row['Count']:,}", ha='center', va='center', color='white', fontweight='bold')
    left += row['Count']
    
  ax.set_yticks([])
  ax.set_xlabel("Number of Cells")
  ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
  sns.despine(left=True)
  
  st.pyplot(fig)
  plt.close()

def render_synced_slider_input(label, param_key, min_v, max_v, default, step, help_text, col_ratios=[3, 1]):
  """
  Render a synchronized slider and number input for a parameter.
  
  Args:
    label: Display label
    param_key: Key in session_state.preprocessing_params
    min_v: Minimum value
    max_v: Maximum value
    default: Default value
    step: Step size
    help_text: Tooltip text
    col_ratios: Ratios for [slider, input] columns
  """
  # Initialize preprocessing_params if not present
  if 'preprocessing_params' not in st.session_state:
    st.session_state.preprocessing_params = {}

  current_val = st.session_state.preprocessing_params.get(param_key, default)
  # Clamp to valid range to prevent stale session values from exceeding new dataset limits
  current_val = max(min_v, min(max_v, current_val))
  st.session_state.preprocessing_params[param_key] = current_val
  st.session_state[param_key] = current_val

  synced_val = render_synced_number_input(
    label=label,
    key=param_key,
    min_value=min_v,
    max_value=max_v,
    default=current_val,
    step=step,
    help_text=help_text,
    col_ratios=col_ratios,
  )

  # Backward-compatible mirrors for code paths still reading legacy keys.
  st.session_state[f"{param_key}_slider"] = synced_val
  st.session_state[f"{param_key}_input"] = synced_val
  st.session_state.preprocessing_params[param_key] = synced_val
  return synced_val

def _apply_benchmark_split(handler, settings, mode_selected):
  """Apply the benchmark split and update session state."""
  try:
    splitter = DatasetSplitter()
    
    with st.spinner(f"Splitting dataset ({mode_selected})..."):
      if settings['mode'] == 'stratified':
        # Stratified split
        split_result = splitter.split_stratified(
          handler.adata,
          train_ratio=settings['ratios'][0],
          val_ratio=settings['ratios'][1],
          test_ratio=settings['ratios'][2],
          label_col=settings['label_col']
        )
      else:
        # Batch split with optional stratified base
        split_result = splitter.split_by_batch(
          handler.adata,
          train_batches=settings['train_batches'],
          test_batches=settings['test_batches'],
          val_ratio=settings['val_ratio'],
          batch_col=settings['batch_col'],
          train_ratio=settings.get('train_ratio', 0.7),
          test_ratio=settings.get('test_ratio', 0.2),
          use_stratified_base=settings.get('use_stratified_base', True)
        )
      
      # Apply balancing if requested
      if settings.get('balance', False):
        target = settings.get('balance_target', 500)
        st.info(f"Balancing training data to {target} cells per group...")
        
        # Determine balancing key
        balance_key = settings.get('batch_col') if settings['mode'] == 'batch' else settings.get('label_col')
        
        split_result.adata_train = splitter.balance_by_batch(
          split_result.adata_train,
          batch_col=balance_key,
          target_per_batch=target
        )
        
        # Update split info manually since balancing changed counts
        split_result.split_info['n_train'] = split_result.adata_train.n_obs
      
      # Store in session state
      set_with_cascade(st.session_state, KEYS.BENCHMARK_SETUP, {
        'mode': 'benchmark',
        'adata_train': split_result.adata_train,
        'adata_test': split_result.adata_test,
        'adata_val': split_result.adata_val,
        'split_info': split_result.split_info,
        'original_settings': settings
      })
      st.session_state[KEYS.BENCHMARK_CONFIGURED] = True
      
      st.success("Split applied successfully!")
      st.rerun()
      
  except Exception as e:
    display_error(
      e,
      user_message="Failed to apply benchmark split configuration.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="preprocessing_apply_split",
    )


def render_preprocessing_page():
  """Render the preprocessing configuration page."""
  st.header("Preprocessing Configuration")

  if not check_prerequisites(require_data=True):
    return

  handler = st.session_state.data_handler
  info = handler.get_info()
  quick_mode = st.session_state.get("ui_mode", "quick") == "quick"

  st.info(f"Current data: {info['n_cells']:,} cells, {info['n_genes']:,} genes")

  # Initialize preprocessing params in session state
  if 'preprocessing_params' not in st.session_state:
    st.session_state.preprocessing_params = {
      p.name: p.default for p in PREPROCESSING_PARAMS
    }

  # ==========================================================================
  # Show Split Configuration Status (configured in Step 2: Data Split)
  # ==========================================================================
  is_configured = st.session_state.get('benchmark_configured', False)
  
  if not is_configured:
    st.warning("Please configure Data Split first in Step 2.")
    st.stop()
  
  setup = st.session_state.get('benchmark_setup', {})
  mode = setup.get('mode', 'standard')
  
  with st.container(border=True):
    if mode == 'standard':
      st.success("**All dataset Mode** (full dataset)")
    else:
      st.success("**Benchmark Mode** configured")
      split_info = setup.get('split_info', {})
      col1, col2, col3 = st.columns(3)
      with col1:
        st.metric("Train", f"{split_info.get('n_train', 0):,}")
      with col2:
        st.metric("Val", f"{split_info.get('n_val', 0):,}")
      with col3: 
        st.metric("Test Set (Eval)", f"{split_info.get('n_test', 0):,}", help="Strictly held-out for final generalization scoring")
      
      # Visual Distribution Chart
      st.caption("Class Balance Visualization:")
      render_split_distribution_chart(split_info)
  st.markdown("---")
  st.subheader("Preprocessing Parameters")

  # We work with the reference to session state params
  # Updates are done directly by callbacks, but we need to ensure consistency
  params = st.session_state.preprocessing_params

  # Information about preprocessing
  st.markdown("### Preprocessing Steps")
  
  with st.expander("Note on Deep Learning Algorithms", expanded=not quick_mode):
    st.info("""
    **Important:** Many Deep Learning algorithms (e.g., **scDeepCluster, scCDCG, scMAE, scNAME, scVI**) 
    require **Raw Counts** to function correctly (using ZINB loss).
    
    *  If you select these algorithms, they will likely **ignore** the Normalization and Scaling steps configured below 
      and use the raw data directly (performing their own internal normalization).
    *  **Quality Filtering** (Step 1) is still recommended for all methods to remove bad cells.
    *  **Classical methods** (PCA+KMeans, Leiden) **DO** require the Normalization/Scaling steps below.
    """)

  # ==========================================================================
  # SECTION 1: Quality Filtering (always independent)
  # ==========================================================================
  st.markdown("### 1. Quality Filtering")
  st.caption("Recommended for all algorithms to remove low-quality cells and genes")
  
  do_cell_filtering = st.checkbox(
    "Enable cell filtering",
    value=params.get('do_cell_filtering', True),
    help="Remove cells with too few or too many genes"
  )
  params['do_cell_filtering'] = do_cell_filtering
  
  if do_cell_filtering:
    col1, col2 = st.columns(2)
    with col1:
      params['min_genes_per_cell'] = render_synced_slider_input(
        "Min genes",
        "min_genes_per_cell",
        min_v=0, max_v=2000, default=100, step=50,
        help_text="Min genes per cell"
      )
    with col2:
      # Max genes is now optional (default None)
      current_max = params.get('max_genes_per_cell', None)
      enable_max_genes = st.checkbox(
        "Filter Max Genes (Doublets)",
        value=(current_max is not None and current_max > 0),
        help="Enable upper limit for genes per cell to filter doublets"
      )
      
      if enable_max_genes:
        # If enabling for the first time, default to 10000
        default_val = current_max if (current_max is not None and current_max > 0) else 10000
        
        # We need to manually handle the state update here because we want to pass None when disabled
        # render_synced_slider_input handles updates but assumes always active
        val = render_synced_slider_input(
          "Max genes",
          "max_genes_per_cell_val", # Use temporary key to avoid conflict when None
          min_v=1000, max_v=50000, 
          default=default_val, 
          step=500,
          help_text="Max genes per cell"
        )
        params['max_genes_per_cell'] = val
      else:
        params['max_genes_per_cell'] = None
  
  do_gene_filtering = st.checkbox(
    "Enable gene filtering",
    value=params.get('do_gene_filtering', True),
    help="Remove lowly expressed genes"
  )
  params['do_gene_filtering'] = do_gene_filtering
  
  if do_gene_filtering:
    params['min_cells_per_gene'] = render_synced_slider_input(
      "Min cells per gene",
      "min_cells_per_gene",
      min_v=1, max_v=50, default=3, step=1,
      help_text="Min cells required for a gene"
    )

  # VISUAL FEEDBACK PREVIEW
  # We estimate the effect of filtering without applying it yet to the whole dataset
  if do_cell_filtering or do_gene_filtering:
    with st.expander("Preview Filtering Impact (Estimation)", expanded=False):
      if handler.adata is not None:
        # Quick estimation
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get current data (lightweight copy ref if possible, or just use X)
        X = handler.adata.X
        
        # Counts per cell
        if hasattr(X, 'getnnz'):
           # sparse
          genes_per_cell = X.getnnz(axis=1)
        else:
          genes_per_cell = np.count_nonzero(X, axis=1)
          
        n_cells_orig = len(genes_per_cell)
        if do_cell_filtering:
          min_g = params.get('min_genes_per_cell', 0)
          max_g = params.get('max_genes_per_cell')
          if max_g is None:
            max_g = 1e9 # Effectively infinite
        else:
          min_g = 0
          max_g = 1e9
        
        cells_keep = (genes_per_cell >= min_g) & (genes_per_cell <= max_g)
        n_cells_after = np.sum(cells_keep)
        
        # Visualization
        col_chart, col_stat = st.columns([2, 1])
        
        with col_chart:
          fig, ax = plt.subplots(figsize=(6, 3))
          sns.barplot(x=['Before', 'After'], y=[n_cells_orig, n_cells_after], palette=['gray', 'steelblue'], ax=ax)
          ax.set_title("Estimated Cell Count")
          st.pyplot(fig)
          plt.close()
          
        with col_stat:
          st.metric("Original Cells", f"{n_cells_orig:,}")
          st.metric("Estimated Remaining", f"{n_cells_after:,}")
          loss = n_cells_orig - n_cells_after
          if n_cells_orig > 0:
            pct_loss = loss / n_cells_orig * 100
          else:
            pct_loss = 0
          st.metric("Removed", f"{loss:,}", delta=f"-{pct_loss:.1f}%", delta_color="inverse")

  # ==========================================================================
  # SECTION 2: Dropout Simulation (optional, applied on raw counts)
  # ==========================================================================
  st.markdown("### 2. Artificial Dropout Simulation")
  st.caption("Applied on raw counts before normalization to simulate capture failure")

  dropout_method = st.selectbox(
    "Dropout Method",
    options=['none', 'simple', 'logistic'],
    index=['none', 'simple', 'logistic'].index(params.get('dropout_method', 'none')),
    help="Method for simulating dropout"
  )
  params['dropout_method'] = dropout_method
  
  if dropout_method == 'simple':
    params['dropout_rate'] = render_synced_slider_input(
      "Dropout Rate",
      "dropout_rate",
      min_v=0.0, max_v=0.9, default=0.2, step=0.05,
      help_text="Proportion of values to drop"
    )

  elif dropout_method == 'logistic':
    c1, c2 = st.columns(2)
    with c1:
      params['dropout_mid'] = render_synced_slider_input(
        "Midpoint",
        "dropout_mid",
        min_v=-1.0, max_v=2.0, default=0.0, step=0.1,
        help_text="Logistic midpoint"
      )
    with c2:
      params['dropout_shape'] = render_synced_slider_input(
        "Shape",
        "dropout_shape",
        min_v=-3.0, max_v=0.0, default=-1.0, step=0.1,
        help_text="Logistic shape (steepness)"
      )

  # Optional Gaussian noise
  with st.expander("Additional Noise Options", expanded=not quick_mode):
    params['noise_level'] = render_synced_slider_input(
      "Gaussian Noise",
      "noise_level",
      min_v=0.0, max_v=2.0, default=0.0, step=0.1,
      help_text="Standard deviation of noise"
    )


  # ==========================================================================
  # SECTION 3: Normalization & Transformation
  # ==========================================================================
  st.markdown("### 3. Normalization & Transformation")
  st.caption("Disable for algorithms with internal normalization (scCDCG)")
  
  batch_corr_at_import = info.get('batch_correction_applied', False)

  do_normalization = st.checkbox(
    "Enable library size normalization",
    value=params.get('do_normalization', True),
    help="Normalize each cell to a fixed total count"
  )
  
  if batch_corr_at_import:
    st.success("**Data is already Batch-Corrected (Normalized + Log1p)** via scVI-DCT.")
    st.info("Standard normalization steps are automatically skipped to prevent double-normalization.")
    do_normalization = False
    # Also disable log transform if normalization is skipped due to pre-correction
    do_log_transform = False 

  params['do_normalization'] = do_normalization
  
  if do_normalization:
    # Calculate suggested target_sum based on dataset size
    n_cells = info['n_cells']
    
    # Heuristic for automatic target_sum selection:
    if n_cells < 5000:
      suggested_target = 10000
    elif n_cells < 50000:
      ratio = (n_cells - 5000) / (50000 - 5000)
      suggested_target = int(10000 + ratio * 90000)
      suggested_target = round(suggested_target / 10000) * 10000
    else:
      suggested_target = 100000
    
    # We handle target_sum differently due to "Auto" button logic interaction
    # But we can still use our synced widget with a slight modification or just use it directly
    
    c1, c2 = st.columns([4, 1])
    with c1:
      params['target_sum'] = render_synced_slider_input(
        "Target Sum",
        "target_sum",
        min_v=1000, max_v=1000000, 
        default=int(params.get('target_sum', suggested_target)), 
        step=1000,
        help_text="Normalization target sum"
      )
    with c2:
      st.markdown("<br>", unsafe_allow_html=True)
      if st.button("Auto"):
        st.session_state.preprocessing_params['target_sum'] = suggested_target
        st.session_state['target_sum'] = suggested_target
        st.session_state['target_sum__slider'] = suggested_target
        st.session_state['target_sum__input'] = suggested_target
        st.session_state['target_sum_slider'] = suggested_target
        st.session_state['target_sum_input'] = suggested_target
        st.rerun()
    
    current_target = params.get('target_sum', suggested_target)
    if current_target != suggested_target:
      st.caption(f"Suggested: **{suggested_target:,}**")

  
  do_log_transform = st.checkbox(
    "Enable log transform (log1p)",
    value=params.get('do_log_transform', True),
    help="Apply log1p transformation after normalization"
  )
  params['do_log_transform'] = do_log_transform

  # ==========================================================================
  # SECTION 4: Feature Selection
  # ==========================================================================
  st.markdown("### 4. Feature Selection")
  
  do_hvg = st.checkbox(
    "Enable HVG selection",
    value=params.get('do_hvg', True),
    help="Select highly variable genes to reduce noise"
  )
  params['do_hvg'] = do_hvg
  
  if do_hvg:
    max_hvg = min(info['n_genes'], 10000)
    current_n_top = params.get('n_top_genes', 2000)
    if current_n_top == 0: current_n_top = 2000
    
    col1, col2 = st.columns([3, 1])
    with col1:
      params['n_top_genes'] = render_synced_slider_input(
        "Number of HVGs",
        "n_top_genes",
        min_v=min(10, max_hvg), max_v=max_hvg, 
        default=current_n_top, 
        step=100,
        help_text="Number of highly variable genes"
      )
    
    with col2:
      params['hvg_flavor'] = st.selectbox(
        "Method",
        options=['seurat', 'seurat_v3', 'cell_ranger'],
        index=['seurat', 'seurat_v3', 'cell_ranger'].index(
          params.get('hvg_flavor', 'seurat')
        ),
        help="HVG selection method"
      )

    # Update params with current value
    # params['n_top_genes'] already set by render function
    
    # HVG Strategy for Benchmark Mode
    # LOGIC FIX: Check both configured state AND intent (if user is currently configuring benchmark)
    is_benchmark_mode = False
    
    # Case 1: Already configured
    if st.session_state.get('benchmark_configured', False) and \
      st.session_state.get('benchmark_setup', {}).get('mode') == 'benchmark':
      is_benchmark_mode = True
      
    # Case 2: Intent to configure (prevent chicken-and-egg problem)
    # We check if the user has selected "Benchmark" in the radio button above, even if not applied yet
    # Since we don't have direct access to the radio widget key here easily without refactoring everything,
    # we assume that if we are NOT configured for Standard, we might be in Benchmark intent.
    # A safer heuristic: If benchmark_configured is FALSE, we show options if they default to train_only anyway.
    
    if is_benchmark_mode:
      st.markdown("**HVG Selection Strategy (Benchmark Mode)**")
      
      strategy_options = {
        'train_only': 'Train Only - Select HVGs from training batches only (Strict - No Leakage)',
        'per_batch_union': 'Per-Batch Union - Select top N from EACH batch, then union'
      }
      # removed 'all_data' to strictly prevent leakage in benchmark mode
      
      # Safe handling of default value if 'all_data' was previously selected
      current_strat = params.get('hvg_strategy', 'train_only')
      if current_strat not in strategy_options:
        current_strat = 'train_only'
      
      params['hvg_strategy'] = st.selectbox(
        "HVG Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=list(strategy_options.keys()).index(current_strat),
        help="How to select HVGs when using benchmark mode with batch splits"
      )
      
      if params['hvg_strategy'] == 'all_data':
        st.info("**All Data**: Ensures consistent gene set but may introduce minor leakage.")
      elif params['hvg_strategy'] == 'per_batch_union':
        st.info("**Per-Batch Union**: Retains batch-specific markers.")
      else: # train_only
        st.info("**Train Only**: No data leakage (Strict).")
    else:
      params['hvg_strategy'] = 'train_only'

    # Option to ensure all cell types have markers in HVGs
    st.markdown("##### Advanced Feature Selection")
    if is_benchmark_mode:
       st.warning("**Benchmark Warning**: Enabling 'Ensure markers' uses cell type labels. If you aim for strict Unsupervised Benchmarking, keep this DISABLED to avoid label leakage into feature selection.")

    params['ensure_celltype_markers'] = st.checkbox(
      "Ensure all cell types have markers (Supervised)",
      value=params.get('ensure_celltype_markers', False),
      help="Add differentially expressed genes to ensure each cell type has at least N marker genes in HVGs. Uses labels."
    )

    if params['ensure_celltype_markers']:
      col_m1, col_m2 = st.columns(2)
      with col_m1:
        params['min_markers_per_celltype'] = st.number_input(
          "Min markers per cell type",
          min_value=1, max_value=50, value=params.get('min_markers_per_celltype', 5),
          help="Minimum number of marker genes required for each cell type"
        )
      with col_m2:
        params['max_additional_hvg_genes'] = st.number_input(
          "Max additional genes",
          min_value=10, max_value=2000, value=params.get('max_additional_hvg_genes', 500),
          help="Maximum number of additional genes to add for cell type markers"
        )
  else:
    params['n_top_genes'] = 0

  # ==========================================================================
  # SECTION 5: Batch Correction (Optional)
  # ==========================================================================
  st.markdown("### 5. Batch Correction (Optional)")
  
  if info.get('batch_correction_applied', False):
    st.success("Batch correction was already applied during Data Upload (Stage: Import).")
    st.info("The parameters below are disabled because the data is already corrected.")
    params['do_batch_correction'] = False
  else:
    # Check if we are in Benchmark Mode
    is_benchmark = st.session_state.get('benchmark_configured', False)
    
    if is_benchmark:
      st.info("In **Benchmark Mode**, batch correction is trained strictly on the **Training Split** to prevent data leakage.")
      
      do_batch_correction = st.checkbox(
        "Enable Batch Correction",
        value=params.get('do_batch_correction', False),
        help="Apply batch correction on training data."
      )
      params['do_batch_correction'] = do_batch_correction
      
      if do_batch_correction:
        st.markdown("#### Batch Correction Settings")
        
        # Method Selector
        bc_method_options = ['scvi', 'sysvi']
        default_bc_method = params.get('batch_correction_method', 'scvi')
        if default_bc_method not in bc_method_options:
          default_bc_method = 'scvi'
        params['batch_correction_method'] = st.selectbox(
          "Method",
          options=bc_method_options,
          index=bc_method_options.index(default_bc_method),
          help="scvi: scVI-DCT (supervised), sysvi: cycle-consistency + VampPrior"
        )

        method = params['batch_correction_method']

        c1, c2 = st.columns(2)

        if method == 'scvi':
          with c1:
            params['batch_correction_dct_weight'] = st.number_input(
              "DCT Weight",
              min_value=0.0, max_value=10.0, value=params.get('batch_correction_dct_weight', 1.0), step=0.1,
              help="Weight for Domain Class Triplet loss"
            )
            params['batch_correction_epochs'] = st.number_input(
              "Epochs",
              min_value=1, max_value=500, value=params.get('batch_correction_epochs', 100), step=50
            )
          with c2:
            params['batch_correction_corr_mse_weight'] = st.number_input(
              "Correlation Weight",
              min_value=0.0, max_value=1.0, value=params.get('batch_correction_corr_mse_weight', 0.1), step=0.05,
              help="Weight for Correlation MSE loss"
            )
            params['batch_correction_n_latent'] = st.number_input(
              "Latent Dim",
              min_value=10, max_value=100, value=params.get('batch_correction_n_latent', 30), step=5
            )

        elif method == 'sysvi':
          st.info(
            "**sysVI** (Hrovatin et al. 2023): cVAE with VampPrior and cycle-consistency loss. "
            "Designed for strong batch effects (cross-species, organoid-tissue, different technologies)."
          )
          with c1:
            params['batch_correction_cycle_weight'] = st.number_input(
              "Cycle-Consistency Weight",
              min_value=0.0, max_value=50.0, value=params.get('batch_correction_cycle_weight', 5.0), step=1.0,
              help="Increase for more batch correction (typically 2-10, up to 50)."
            )
            params['batch_correction_epochs'] = st.number_input(
              "Epochs",
              min_value=1, max_value=500, value=params.get('batch_correction_epochs', 200), step=50
            )
            params['batch_correction_prior'] = st.selectbox(
              "Prior",
              options=['vamp', 'standard_normal'],
              index=0 if params.get('batch_correction_prior', 'vamp') == 'vamp' else 1,
              help="VampPrior recommended for better biological preservation"
            )
          with c2:
            params['batch_correction_kl_weight'] = st.number_input(
              "KL Weight",
              min_value=0.0, max_value=10.0, value=params.get('batch_correction_kl_weight', 1.0), step=0.1,
              help="Decrease for better biological preservation."
            )
            params['batch_correction_n_latent'] = st.number_input(
              "Latent Dim",
              min_value=5, max_value=100, value=params.get('batch_correction_n_latent', 15), step=5
            )
            params['batch_correction_n_prior_components'] = st.number_input(
              "Prior Components",
              min_value=1, max_value=50, value=params.get('batch_correction_n_prior_components', 5), step=1,
              help="Number of VampPrior pseudo-inputs"
            )
          params['batch_correction_embed_covariates'] = st.checkbox(
            "Embed Categorical Covariates",
            value=params.get('batch_correction_embed_covariates', True),
            help="If enabled, sysVI learns embeddings for additional categorical covariates."
          )
    else:
      # Standard Mode
      st.info(
        "Batch correction (scVI/sysVI) is now configured at **Data Upload**. "
        "Preprocessing will use corrected data if already applied."
      )
      params['do_batch_correction'] = False

  # ==========================================================================
  # SECTION 6: Scaling
  # ==========================================================================
  st.markdown("### 6. Scaling")
  
  do_scaling = st.checkbox(
    "Enable Z-score scaling",
    value=params.get('do_scaling', True),
    help="Standardize data (mean=0, std=1)"
  )
  params['do_scaling'] = do_scaling
  
  if do_scaling:
    with st.expander("Advanced Scaling Options", expanded=not quick_mode):
      params['scale_max_value'] = render_synced_slider_input(
        "Scale Clip Value",
        "scale_max_value",
        min_v=1.0, max_v=50.0, default=10.0, step=1.0,
        help_text="Maximum value for clipping after scaling"
      )

  # Save params
  st.session_state.preprocessing_params = params

  # Run preprocessing button
  st.markdown("---")

  col1, col2, col3 = st.columns([1, 2, 1])
  with col2:
    if st.button("Run Preprocessing", type="primary", width="stretch"):
      _run_preprocessing(handler, params)

  # Show preprocessing status
  if st.session_state.get('data_preprocessed', False):
    st.success("Data has been preprocessed!")
    info = handler.get_info()
    batch_corr_applied, batch_corr_info = _get_batch_correction_status(handler, info)
    st.info(f"After preprocessing: {info['n_cells']:,} cells, {info['n_genes']:,} genes")

    # Show dropout info if applied
    if info.get('dropout_applied', False):
      dropout_info = info['dropout_info']
      st.markdown("#### Dropout Simulation Applied")
      col1, col2, col3 = st.columns(3)
      with col1:
        st.metric("Method", dropout_info['method'].capitalize())
      with col2:
        st.metric("Actual Dropout Rate", f"{dropout_info['actual_dropout_rate']:.1%}")
      with col3:
        st.metric("Values Dropped", f"{dropout_info['n_values_dropped']:,}")

    # Show HVG count for Per-Batch Union strategy (benchmark mode)
    if 'benchmark_processed' in st.session_state:
      processed = st.session_state.benchmark_processed
      preprocessor = processed.get('preprocessor')
      if preprocessor and preprocessor.params:
        hvg_strategy = getattr(preprocessor.params, 'hvg_strategy', None)
        hvg_genes = getattr(preprocessor.params, 'hvg_genes', None)
        hvg_per_batch = getattr(preprocessor.params, 'hvg_per_batch', None)
        requested_hvg = int(params.get('n_top_genes', 0) or 0)

        if params.get('do_hvg', True) and hvg_genes is not None:
          st.markdown("#### Final HVG Count")
          st.metric("HVGs retained", f"{len(hvg_genes):,}")
          if requested_hvg > 0 and len(hvg_genes) != requested_hvg:
            st.caption(
              f"Requested: {requested_hvg:,} | Final: {len(hvg_genes):,} "
              "(limited by available genes or strategy constraints)."
            )
        
        if hvg_strategy == 'per_batch_union' and hvg_genes is not None:
          st.markdown("#### HVG Selection Summary (Per-Batch Union)")
          st.metric("Total HVGs (Union)", f"{len(hvg_genes):,}")
          
          # Display per-batch breakdown if available
          if hvg_per_batch:
            st.markdown("**HVGs per Batch:**")
            import pandas as pd
            df_hvg = pd.DataFrame([
              {'Batch': batch, 'HVGs Selected': count}
              for batch, count in hvg_per_batch.items()
            ])
            df_hvg = df_hvg.sort_values('Batch')
            st.dataframe(df_hvg, width="stretch", hide_index=True)

          # Show updated label proportions per batch using ORIGINAL (full) data
          # Use adata_original to get ALL batches, not just filtered training data
          adata_for_props = handler.adata_original if hasattr(handler, 'adata_original') and handler.adata_original is not None else handler.adata
          if adata_for_props is not None:
            setup = st.session_state.get('benchmark_setup', {})
            settings = setup.get('original_settings', {})
            batch_col = settings.get('batch_col') or get_batch_column(adata_for_props)
            label_col = settings.get('label_col')
            if not label_col:
              for col in ['Group', 'labels', 'labels_encoded', 'celltype', 'cell_type']:
                if col in adata_for_props.obs.columns:
                  label_col = col
                  break

            if batch_col in adata_for_props.obs.columns and label_col in adata_for_props.obs.columns:
              st.markdown("**Label Distribution by Batch (Count + %):**")
              import pandas as pd
              df_obs = adata_for_props.obs[[batch_col, label_col]].copy()
              
              # Get counts
              df_counts = df_obs.groupby([batch_col, label_col]).size().reset_index(name='Count')
              
              # Get percentages per batch
              df_pct = (
                df_obs.groupby(batch_col)[label_col]
                .value_counts(normalize=True)
                .mul(100)
                .rename('Percent')
                .reset_index()
              )
              
              # Merge counts and percentages
              df_merged = df_counts.merge(df_pct, on=[batch_col, label_col])
              df_merged['Value'] = df_merged.apply(lambda row: f"{int(row['Count'])} ({row['Percent']:.1f}%)", axis=1)
              
              # Pivot for display
              df_pivot = df_merged.pivot_table(
                index=batch_col,
                columns=label_col,
                values='Value',
                aggfunc='first'
              ).fillna("0 (0.0%)")
              
              st.dataframe(df_pivot, width="stretch")

            else:
              st.info("Label proportions not shown (missing batch or label column).")

    # --- Post-Preprocessing Distribution Analysis ---
    st.markdown("---")
    from gui.shared_components import display_batch_and_distribution_info
    display_batch_and_distribution_info(
      handler.adata, 
      handler, 
      title_prefix="Post-Preprocessing "
    )

    # Show batch correction info if applied
    if batch_corr_applied:
      st.markdown("#### Batch Correction Applied")
      bc_info = batch_corr_info
      col1, col2, col3 = st.columns(3)
      with col1:
        st.metric("Method", bc_info.get('method', 'scVI').upper())
      with col2:
        st.metric("Batches", bc_info.get('n_batches', 'N/A'))
      with col3:
        st.metric("Latent Dim", bc_info.get('n_latent', 30))
      st.success("All algorithms will use batch-corrected expression data.")

    # Download Preprocessed Data
    st.markdown("---")
    st.subheader("Download Preprocessed Data")
    
    if 'benchmark_processed' in st.session_state:
      # Benchmark Mode: Split downloads
      processed = st.session_state.benchmark_processed
      c1, c2, c3 = st.columns(3)
      
      with c1:
        _render_download_button(processed.get('train'), "train_data.h5ad", "Download Train Set")
      with c2:
        _render_download_button(processed.get('test'), "test_data.h5ad", "Download Test Set")
      with c3:
        if processed.get('val') is not None:
          _render_download_button(processed.get('val'), "val_data.h5ad", "Download Val Set")
    else:
      # Standard Mode: Single download
      _render_download_button(handler.adata, "preprocessed_data.h5ad", "Download Full Dataset")

    _render_preprocessing_output_save_section(handler, info)


def _render_download_button(adata, filename, label):
  """Helper to render a download button for an AnnData object."""
  import tempfile
  import os
  
  if adata is None:
    return

  try:
    # Create a button that triggers the file preparation only when needed? 
    # Streamlit's download_button requires 'data' to be pre-computed OR a callback.
    # Computing serialization for large files on every re-run is expensive.
    # Ideally, we should cache this or use a callback.
    # But 'data' argument in download_button does not support callback for content generation directly in older Streamlit versions.
    # However, writing to temp file is relatively fast for medium datasets.
    
    # Optimization: We can't easily check if user clicked without running the logic.
    # But we can provide a "Prepare Download" button first if it's too heavy.
    # For now, let's try direct approach but handle it gracefully.
    
    # Use session state to cache the binary data to avoid re-writing on every interaction
    cache_key = f"download_cache_{filename}"
    
    # Only recompute if not cached
    if cache_key not in st.session_state:
       with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
        tmp_path = tmp.name
       
       adata.write_h5ad(tmp_path, compression='gzip')
       
       with open(tmp_path, 'rb') as f:
         st.session_state[cache_key] = f.read()
       
       os.unlink(tmp_path)
       
    render_download_button(
      data=st.session_state[cache_key],
      filename=filename,
      label=label,
      mime="application/octet-stream",
      key=f"download_{filename}",
    )
    
  except Exception as e:
    st.warning(f"Could not prepare download for {filename}: {str(e)}")


def _get_batch_correction_status(handler, info):
  """
  Detect whether batch correction has been applied and return metadata.

  This supports:
  - Standard DataHandler flag/info path
  - Benchmark mode where corrected splits are stored in session state
  """
  if info.get("batch_correction_applied", False):
    bc_info = info.get("batch_correction_info", {}) or {}
    return True, bc_info

  if getattr(handler, "adata", None) is not None:
    bc_info = handler.adata.uns.get("batch_correction", {})
    if bc_info:
      return True, bc_info

  processed = st.session_state.get("benchmark_processed")
  if processed:
    for split_name in ("train", "test", "val"):
      split_adata = processed.get(split_name)
      if split_adata is None:
        continue
      bc_info = split_adata.uns.get("batch_correction", {})
      if bc_info:
        return True, bc_info

  return False, {}


def _save_preprocessed_matrix_to_disk(
  adata,
  output_base_path,
  export_format: str = "h5ad",
  include_obs_var: bool = True
):
  """
  Save the current preprocessed matrix (adata.X) to disk.

  Returns:
    List of written file paths.
  """
  import numpy as np
  import pandas as pd
  import scipy.sparse as sp

  output_base_path = Path(output_base_path)
  saved_paths = []

  if export_format == "h5ad":
    out_path = output_base_path.with_suffix(".h5ad")
    adata.write_h5ad(out_path, compression="gzip")
    saved_paths.append(out_path)
    return saved_paths

  X = adata.X
  if sp.issparse(X):
    X = X.toarray()
  else:
    X = np.asarray(X)

  if export_format == "npy":
    out_path = output_base_path.with_suffix(".npy")
    np.save(out_path, X)
    saved_paths.append(out_path)
  elif export_format == "csv":
    out_path = output_base_path.with_suffix(".csv")
    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)
    df.to_csv(out_path)
    saved_paths.append(out_path)
  else:
    raise ValueError(f"Unsupported export format: {export_format}")

  if include_obs_var:
    obs_path = output_base_path.with_name(f"{output_base_path.name}_obs.csv")
    var_path = output_base_path.with_name(f"{output_base_path.name}_var.csv")
    adata.obs.to_csv(obs_path)
    adata.var.to_csv(var_path)
    saved_paths.extend([obs_path, var_path])

  return saved_paths


def _render_preprocessing_output_save_section(handler, info):
  """Render server-side save section for post-preprocessing matrices."""
  batch_corr_applied, bc_info = _get_batch_correction_status(handler, info)
  if not batch_corr_applied:
    return

  bc_stage = bc_info.get('stage', 'preprocessing')

  st.markdown("---")
  st.subheader("Save Matrix on Server")
  st.caption(
    "Save the matrix produced at the end of preprocessing (`adata.X`) directly on server disk."
  )

  if bc_stage == 'import':
    st.info(
      "Batch correction was applied at import stage. "
      "The saved matrix corresponds to the current post-preprocessing data."
    )
  else:
    st.info(
      "Batch correction was applied during preprocessing. "
      "The saved matrix corresponds to the corrected matrix after preprocessing."
    )

  is_benchmark = 'benchmark_processed' in st.session_state
  processed = st.session_state.get('benchmark_processed', {}) if is_benchmark else {}

  default_dir = Path("results") / "preprocessed_exports"
  output_dir = st.text_input(
    "Output directory (server path)",
    value=str(default_dir),
    key="preproc_output_save_dir",
    help="Directory where files will be written on the server."
  )
  file_prefix = st.text_input(
    "File prefix",
    value="batch_corrected_preprocessed",
    key="preproc_output_save_prefix",
    help="Prefix used for output filenames."
  )

  export_format = st.selectbox(
    "Export format",
    options=["h5ad", "npy", "csv"],
    index=0,
    key="preproc_output_export_format",
    help="h5ad preserves full AnnData structure; npy/csv export matrix values from adata.X."
  )
  include_obs_var = st.checkbox(
    "Also export obs/var CSV metadata (for npy/csv)",
    value=True,
    key="preproc_output_include_obs_var"
  )
  append_timestamp = st.checkbox(
    "Append timestamp to filenames",
    value=True,
    key="preproc_output_append_ts"
  )

  split_choice = "full"
  split_options = ["full"]
  if is_benchmark:
    split_options = ["train", "test"]
    if processed.get('val') is not None:
      split_options.append("val")
    split_options.append("all")
    split_choice = st.selectbox(
      "Split to export",
      options=split_options,
      index=split_options.index("all"),
      key="preproc_output_split_choice"
    )

  if export_format == "csv":
    st.warning(
      "CSV export may be very large and slower. Prefer h5ad or npy for big matrices."
    )

  if st.button("Save Matrix to Disk", type="secondary", key="save_preprocessed_matrix_disk"):
    try:
      from datetime import datetime

      out_dir = Path(output_dir).expanduser()
      out_dir.mkdir(parents=True, exist_ok=True)

      timestamp_suffix = ""
      if append_timestamp:
        timestamp_suffix = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

      safe_prefix = file_prefix.strip() or "batch_corrected_preprocessed"
      targets = {}

      if is_benchmark:
        if split_choice == "all":
          targets["train"] = processed.get("train")
          targets["test"] = processed.get("test")
          if processed.get("val") is not None:
            targets["val"] = processed.get("val")
        else:
          targets[split_choice] = processed.get(split_choice)
      else:
        targets["full"] = handler.adata

      saved_files = []
      for split_name, adata_obj in targets.items():
        if adata_obj is None:
          continue
        base_name = f"{safe_prefix}_{split_name}{timestamp_suffix}"
        output_base = out_dir / base_name
        saved_files.extend(
          _save_preprocessed_matrix_to_disk(
            adata_obj,
            output_base,
            export_format=export_format,
            include_obs_var=include_obs_var
          )
        )

      if not saved_files:
        st.error("No matrix was saved (empty split selection).")
        return

      st.success(f"Saved {len(saved_files)} file(s) to `{out_dir}`")
      for p in saved_files:
        st.code(str(p))

    except Exception as e:
      st.error(f"Failed to save matrix: {str(e)}")


from utils.dataset_splitter import BenchmarkPreprocessor

def _snapshot_preprocessing_params(params):
  """Return a stable subset of preprocessing params for staleness checks."""
  if not isinstance(params, dict):
    return {}
  tracked_keys = [
    'skip',
    'do_cell_filtering',
    'min_genes_per_cell',
    'max_genes_per_cell',
    'do_gene_filtering',
    'min_cells_per_gene',
    'do_normalization',
    'target_sum',
    'do_log_transform',
    'do_hvg',
    'n_top_genes',
    'hvg_flavor',
    'hvg_strategy',
    'ensure_celltype_markers',
    'min_markers_per_celltype',
    'max_additional_hvg_genes',
    'do_batch_correction',
    'batch_correction_method',
    'batch_correction_batch_key',
    'batch_correction_labels_key',
    'batch_correction_epochs',
    'batch_correction_n_latent',
    'batch_correction_batch_size',
    'batch_correction_dct_weight',
    'batch_correction_corr_mse_weight',
    'batch_correction_cycle_weight',
    'batch_correction_kl_weight',
    'batch_correction_prior',
    'batch_correction_n_prior_components',
    'batch_correction_embed_covariates',
    'do_scaling',
    'scale_max_value',
    'dropout_method',
    'dropout_rate',
    'dropout_mid',
    'dropout_shape',
    'noise_level',
    'batch_col',
  ]
  snap = {}
  for key in tracked_keys:
    if key in params:
      value = params.get(key)
      if isinstance(value, (list, tuple)):
        snap[key] = list(value)
      else:
        snap[key] = value
  return snap

def _sync_preprocessing_params_from_widgets(params):
  """Synchronize latest widget values into preprocessing params before execution."""
  if not isinstance(params, dict):
    return params

  if not params.get('do_hvg', True):
    params['n_top_genes'] = 0

  sync_specs = [
    ('min_genes_per_cell', 'min_genes_per_cell', int, 0),
    ('max_genes_per_cell', 'max_genes_per_cell_val', int, 1),
    ('min_cells_per_gene', 'min_cells_per_gene', int, 1),
    ('target_sum', 'target_sum', int, 1),
    ('n_top_genes', 'n_top_genes', int, 0),
    ('dropout_rate', 'dropout_rate', float, 0.0),
    ('dropout_mid', 'dropout_mid', float, None),
    ('dropout_shape', 'dropout_shape', float, None),
    ('noise_level', 'noise_level', float, 0.0),
    ('scale_max_value', 'scale_max_value', float, 0.0),
  ]

  for param_name, widget_key, caster, min_value in sync_specs:
    if param_name == 'n_top_genes' and not params.get('do_hvg', True):
      continue
    if param_name == 'max_genes_per_cell' and params.get('max_genes_per_cell') in (None, '', 0):
      continue

    candidates = [
      st.session_state.get(f'{widget_key}__input'),
      st.session_state.get(f'{widget_key}__slider'),
      st.session_state.get(widget_key),
      params.get(param_name),
    ]

    for value in candidates:
      if value is None:
        continue
      try:
        synced_value = caster(value)
      except (TypeError, ValueError):
        continue

      if min_value is not None and synced_value < min_value:
        continue

      params[param_name] = synced_value
      st.session_state[param_name] = synced_value
      if (
        'preprocessing_params' in st.session_state
        and isinstance(st.session_state.preprocessing_params, dict)
      ):
        st.session_state.preprocessing_params[param_name] = synced_value
      break

  return params

def _run_preprocessing(handler, params):
  """Run preprocessing with given parameters, respecting benchmark split if active."""
  try:
    with st.spinner("Preprocessing data..."):
      params_for_run = dict(params) if isinstance(params, dict) else {}
      params_for_run = _sync_preprocessing_params_from_widgets(params_for_run)
      
      # Check for Benchmark Mode
      if st.session_state.get('benchmark_configured', False):
        setup = st.session_state.get('benchmark_setup', {})
        
        if setup.get('mode') == 'benchmark':
          # --- BENCHMARK MODE: Fit on Train, Transform on all ---
          st.info("Applying split-aware preprocessing (Fit on Train, Transform on Test)...")
          
          # 1. Get raw split data
          adata_train = setup['adata_train']
          adata_test = setup['adata_test']
          adata_val = setup['adata_val']
          
          # 2. Add batch_col to params if available (for per-batch HVG strategy)
          if 'original_settings' in setup:
            params_for_run['batch_col'] = setup['original_settings'].get('batch_col')
          
          # 3. Instantiate and Fit based on HVG strategy
          preprocessor = BenchmarkPreprocessor()
          hvg_strategy = params_for_run.get('hvg_strategy', 'train_only')
          
          if hvg_strategy == 'all_data':
            # For 'all_data' strategy, fit on full dataset (before split)
            st.info("HVG Strategy: Selecting HVGs from ALL data before split...")
            # Get full data from handler
            full_data = handler.adata_original if hasattr(handler, 'adata_original') and handler.adata_original is not None else handler.adata
            preprocessor.fit(full_data, params_for_run)
          elif hvg_strategy == 'per_batch_union':
            # For 'per_batch_union' strategy, fit on full dataset to get union from ALL batches
            st.info("HVG Strategy: Computing union of HVGs across ALL batches...")
            full_data = handler.adata_original if hasattr(handler, 'adata_original') and handler.adata_original is not None else handler.adata
            preprocessor.fit(full_data, params_for_run)
          else:
            # Default (train_only): fit on training data only
            preprocessor.fit(adata_train, params_for_run)
          
          # 3. Transform
          train_proc = preprocessor.transform(adata_train)
          test_proc = preprocessor.transform(adata_test)
          val_proc = preprocessor.transform(adata_val) if adata_val is not None else None
          
          # 4. Store processed data in session state
          applied_snapshot = _snapshot_preprocessing_params(params_for_run)
          st.session_state.benchmark_processed = {
            'train': train_proc,
            'test': test_proc,
            'val': val_proc,
            'preprocessor': preprocessor,
            'preprocessing_params': applied_snapshot,
          }
          st.session_state.preprocessing_applied_params = applied_snapshot
          
          # 5. Update main handler to show TRAIN data in UI
          # This ensures UMAPs and stats reflect what the model sees during training
          handler.adata = train_proc
          # Keep labels in sync with the displayed data to avoid metric mismatches
          if 'Group' in handler.adata.obs:
            handler.labels = handler.adata.obs['Group'].values
          elif 'labels' in handler.adata.obs:
            handler.labels = handler.adata.obs['labels'].values
          elif 'labels_encoded' in handler.adata.obs:
            handler.labels = handler.adata.obs['labels_encoded'].values
          else:
            handler.labels = None
          
          st.session_state.data_preprocessed = True
          st.success("Preprocessing completed without data leakage!")
          st.rerun()
          return

      # --- STANDARD MODE: Global Preprocessing ---
      handler.preprocess(params_for_run)
      st.session_state.preprocessing_applied_params = _snapshot_preprocessing_params(params_for_run)
      st.session_state.data_preprocessed = True
      st.success("Preprocessing completed (Global)!")
      st.rerun()
      
  except Exception as e:
    st.error(f"Preprocessing failed: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
