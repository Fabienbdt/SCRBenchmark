"""
Data Split Configuration Page

Dedicated page for configuring train/val/test splits.
Supports Classical (Stratified) and Batch-Filtered modes.
"""

import streamlit as st
import numpy as np
import pandas as pd

from utils.dataset_splitter import DatasetSplitter, get_batch_column, list_batches
from gui.widgets import check_prerequisites
from gui.widgets import display_error
from gui.state_manager import KEYS, invalidate, set_with_cascade


def render_data_split_page():
  """Render the data split configuration page."""
  st.header("Data Split Configuration")
  
  if not check_prerequisites(require_data=True):
    return
  
  handler = st.session_state.data_handler
  info = handler.get_info()
  adata = handler.get_data()
  
  st.info(f"Current data: {info['n_cells']:,} cells, {info['n_genes']:,} genes")
  
  # Check if already configured
  is_configured = st.session_state.get('benchmark_configured', False)
  
  if is_configured:
    _render_split_summary()
  else:
    _render_split_configuration(handler, adata)


def _render_split_summary():
  """Display summary of configured split."""
  setup = st.session_state.get('benchmark_setup', {})
  mode = setup.get('mode', 'standard')
  
  if mode == 'standard':
    st.success(" **All dataset** (Full dataset)")
    st.caption("Using entire dataset for clustering. No train/test split.")
  else:
    split_info = setup.get('split_info', {})
    st.success(f" **Benchmark Mode** configured")
    
    # Show split metrics
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Train", f"{split_info.get('n_train', 0):,}")
    with col2:
      st.metric("Validation", f"{split_info.get('n_val', 0):,}")
    with col3:
      st.metric("Test", f"{split_info.get('n_test', 0):,}")

    settings = setup.get('original_settings', {})
    val_enabled = settings.get('use_validation', True)
    if not val_enabled or split_info.get('n_val', 0) == 0:
      st.caption("Validation: disabled (metrics computed on Train/Test only).")
    else:
      st.caption("Validation: metrics-only holdout (no hyperparameter search).")

    # Show reinjection info if applicable
    if split_info.get('test_balanced', False):
      n_reinjected = split_info.get('n_reinjected', 0)
      reinject_to = split_info.get('reinject_to', 'train')
      target = split_info.get('test_target_per_batch', 0)
      st.info(
        f"**Reinject mode**: TEST balanced to {target} cells/batch. "
        f"{n_reinjected} cells reinjected into {reinject_to.upper()}."
      )

    # Show distribution comparison
    if 'adata_train' in setup and 'adata_test' in setup:
      _render_distribution_check(setup)
  
  # Reset button
  if st.button("Reset Split Configuration"):
    invalidate(st.session_state, KEYS.BENCHMARK_SETUP, recursive=True)
    st.session_state[KEYS.BENCHMARK_CONFIGURED] = False
    if 'benchmark_processed' in st.session_state:
      del st.session_state['benchmark_processed']
    st.rerun()


def _render_distribution_check(setup):
  """Show label distribution comparison between train and test."""
  ad_train = setup['adata_train']
  ad_test = setup['adata_test']
  
  # Find label column
  label_col = None
  for col in ['labels', 'Group', 'celltype', 'cell_type']:
    if col in ad_train.obs.columns:
      label_col = col
      break
  
  if label_col:
    with st.expander(" Label Distribution (Train vs Test)", expanded=False):
      train_counts = ad_train.obs[label_col].value_counts(normalize=True).rename("Train %")
      test_counts = ad_test.obs[label_col].value_counts(normalize=True).rename("Test %")
      
      df_dist = pd.concat([train_counts, test_counts], axis=1).sort_index()
      df_dist['Diff'] = (df_dist['Train %'] - df_dist['Test %']).abs()
      
      st.dataframe(df_dist.style.format("{:.1%}"), width="stretch")
      
      max_diff = df_dist['Diff'].max()
      if max_diff < 0.05:
        st.success("Good stratification: max difference < 5%")
      else:
        st.warning(f"Max difference: {max_diff:.1%}")


def _render_split_configuration(handler, adata):
  """Render the split configuration UI."""
  
  with st.container(border=True):
    st.markdown("### Choose Analysis Mode")
    
    mode = st.radio(
      "Mode:",
      ["All dataset", "Split dataset"],
      help="All dataset: Full dataset. Split dataset: Split for generalization evaluation."
    )
    
    if mode == "All dataset":
      st.info(
        "Uses the entire dataset for clustering.\n\n"
        "**Best for:** Data exploration, visualization, biological discovery."
      )
      
      if st.button(" Confirm Standard Mode", type="primary"):
        set_with_cascade(st.session_state, KEYS.BENCHMARK_SETUP, {'mode': 'standard'})
        st.session_state[KEYS.BENCHMARK_CONFIGURED] = True
        st.rerun()
        
    else: # Benchmark Mode
      _render_benchmark_config(handler, adata)


def _render_benchmark_config(handler, adata):
  """Render benchmark mode configuration."""
  st.info("Splits data to evaluate **generalization** on unseen cells.")

  with st.expander("Validation Set: What it does (and doesn't)", expanded=False):
    st.markdown(
      "- **Held-out metrics only:** validation is used to compute NMI/ARI/ACC on unseen cells.\n"
      "- **Training uses Train only:** no hyperparameter search is run in benchmark mode.\n"
      "- **Batch-Filtered mode:** validation is drawn from the same batches as Train.\n"
      "- **Hyperparameter search is separate:** use the **Hyperparam Search** page."
    )
  
  batch_col = get_batch_column(adata)
  batches = list_batches(adata, batch_col)
  
  # Initialize settings dict
  settings = {'batch_col': batch_col}

  # =========================================================================
  # STEP 1: Dataset Balancing (Optional)
  # =========================================================================
  st.markdown("### Step 1: Dataset Balancing (Optional)")

  col_bal1, col_bal2 = st.columns([1, 2])
  with col_bal1:
    do_balance = st.checkbox(
      "Normalize batch sizes",
      help="Ensure equal representation per batch for fair evaluation."
    )
    settings['balance'] = do_balance

  with col_bal2:
    if do_balance:
      # Calculate default target (smallest batch)
      if batch_col and batch_col in adata.obs.columns:
        min_batch_size = adata.obs[batch_col].value_counts().min()
        st.caption(f"Smallest batch size: **{min_batch_size}**")
      else:
        min_batch_size = 500
        st.caption("Batch column not found or computed.")

      target_size = st.number_input(
        "Target cells per batch",
        min_value=10,
        max_value=100000,
        value=int(min_batch_size),
        step=10,
        help="Batches larger than this will be downsampled."
      )
      settings['balance_target'] = target_size

  # Balance mode selection (only shown when balancing is enabled)
  if do_balance:
    st.markdown("**Balance Mode:**")
    balance_mode = st.radio(
      "How to handle excess cells?",
      options=['eliminate', 'reinject_train', 'reinject_both'],
      format_func=lambda x: {
        'eliminate': 'Eliminate excess cells (standard)',
        'reinject_train': 'Reinject into TRAIN (no data loss)',
        'reinject_both': 'Reinject into TRAIN + VAL (no data loss)'
      }[x],
      help=(
        "**Eliminate**: Cells in excess are discarded before splitting.\n\n"
        "**Reinject into TRAIN**: Balance TEST only, excess cells go to TRAIN.\n\n"
        "**Reinject into TRAIN+VAL**: Balance TEST only, excess cells split between TRAIN and VAL."
      ),
      horizontal=True
    )
    settings['balance_mode'] = balance_mode

    if balance_mode != 'eliminate':
      st.info(
        "**Reinject mode**: Split is done first on ALL data, then TEST is balanced. "
        "Excess cells from TEST are added to TRAIN/VAL. No data is lost!"
      )

  st.markdown("---")
  
  st.markdown("### Step 2: Split Strategy")

  # Split strategy selection
  with st.expander("Understanding Split Strategies", expanded=False):
    st.markdown("""
### 1 Classical Split
- **Preserves Batch Proportions** (unless balancing is enabled above).
- **Flexible Stratification**: Choose to stratify training data by Batch (default) or Cell Type.
- Best for: Standard benchmarking (random stratified split)

### 2 Batch-Filtered (Cross-Batch Generalization)
- **Step 1**: Stratified split 70/10/20 on ALL data
- **Step 2**: Filter TRAIN and VAL to selected batch(es) only
- **TEST** remains complete with **all batches** (balanced)
- Best for: Testing if model trained on one batch generalizes to others
    """)
  
  split_mode = st.radio(
    "Split Strategy:",
    ['classical', 'batch_filtered'],
    format_func=lambda x: {
      'classical': '1 Classical Split (all batches)',
      'batch_filtered': '2 Batch-Filtered (train on one, test on all)'
    }.get(x),
    horizontal=True
  )
  
  if split_mode == 'classical':
    settings.update(_render_classical_config(adata))
  else:
    if not batches:
      st.error("No batch column detected. Use Classical Split.")
      return
    settings.update(_render_batch_filtered_config(adata, batches, batch_col))
  
  # Apply button
  st.markdown("---")
  if st.button(" Apply Split", type="primary", width="stretch"):
    _apply_split(handler, settings)


def _render_classical_config(adata):
  """Render classical stratified split configuration."""
  st.markdown("#### Split Options")

  # Stratification Method Selection
  st.markdown("**Stratification Mode**")

  with st.container(border=True):
    strat_mode = st.radio(
      "How should Train/Val/Test be balanced?",
      options=['batch+labels', 'batch-only', 'labels-only', 'none'],
      format_func=lambda x: {
        'batch+labels': 'Batch + Cell Types (Recommended)',
        'batch-only': 'Batch Only (ignore cell types)',
        'labels-only': 'Cell Types Only (ignore batches)',
        'none': 'Random (no stratification)'
      }.get(x),
      horizontal=True,
      help="Controls how samples are distributed across splits."
    )

    # Show explanation
    explanations = {
      'batch+labels': "**Preserves both batch AND cell type proportions** in each split. "
              "This prevents confounding between batch effects and biological signals.",
      'batch-only': "**Preserves batch proportions only.** Each split has similar batch composition, "
             "but cell type proportions may vary.",
      'labels-only': "**Preserves cell type proportions only.** Each split has similar cell type composition, "
             "but batch proportions may vary.",
      'none': "**Fully random split.** No stratification - proportions may vary between splits."
    }
    st.caption(explanations[strat_mode])

  # Map selection to booleans
  stratify_by_batch = strat_mode in ['batch+labels', 'batch-only']
  stratify_by_labels = strat_mode in ['batch+labels', 'labels-only']
  
  st.markdown("#### Split Ratios")

  st.markdown("#### Validation Set")
  enable_validation = st.checkbox(
    "Create validation set",
    value=True,
    help="Used only for reporting metrics on a held-out set (no hyperparameter search)."
  )
  
  col1, col2, col3 = st.columns(3)
  with col1:
    train_ratio = st.slider("Train", 0.5, 0.9, 0.7, 0.05)
  with col2:
    val_ratio_default = 0.1 if enable_validation else 0.0
    val_ratio = st.slider(
      "Validation",
      0.0,
      0.3,
      val_ratio_default,
      0.05,
      disabled=not enable_validation,
      help="Validation metrics only; set to 0 to disable."
    )
    if not enable_validation:
      val_ratio = 0.0
  with col3:
    test_ratio = 1.0 - train_ratio - val_ratio
    st.metric("Test", f"{test_ratio:.0%}")
  
  if test_ratio <= 0:
    st.error("Test ratio must be > 0")
  
  # Label column for stratification
  label_col = 'Group'
  for col in ['labels', 'Group', 'celltype']:
    if col in adata.obs.columns:
      label_col = col
      break
  
  # Build options list for selectbox
  valid_columns = [c for c in adata.obs.columns if adata.obs[c].dtype.name == 'category' or adata.obs[c].nunique() < 100]
  
  if not valid_columns:
    st.warning("No suitable columns found for stratification (need categorical or <100 unique values)")
    # Fallback to first column if none match criteria
    valid_columns = list(adata.obs.columns)[:3] if len(adata.obs.columns) > 0 else ['labels']
  
  default_index = valid_columns.index(label_col) if label_col in valid_columns else 0
  
  if stratify_by_labels:
    label_col = st.selectbox(
      "Cell Type Column:",
      options=valid_columns,
      index=default_index,
      help="Column used for cell type stratification."
    )
  
  return {
    'mode': 'stratified',
    'train_ratio': train_ratio,
    'val_ratio': val_ratio,
    'test_ratio': test_ratio,
    'label_col': label_col,
    'use_validation': enable_validation,
    'stratify_by_batch': stratify_by_batch,
    'stratify_by_labels': stratify_by_labels
  }


def _render_batch_filtered_config(adata, batches, batch_col):
  """Render batch-filtered split configuration."""
  st.info(
    "**How it works:**\n"
    "1. Stratified split 70/10/20 on **entire dataset** (all batches)\n"
    "2. Filter **TRAIN** and **VAL** to selected batch(es) only\n"
    "3. Keep **TEST** complete (all batches, balanced)"
  )
  
  st.markdown(f"**Available batches:** {', '.join(batches)}")
  
  train_batches = st.multiselect(
    "Select Batch(es) for Training:",
    options=batches,
    default=[batches[0]] if batches else [],
    help="Train and Val will contain ONLY cells from these batches."
  )
  
  if train_batches:
    other_batches = [b for b in batches if b not in train_batches]
    st.success(
      f" **Train/Val:** {', '.join(train_batches)}\n\n"
      f" **Test:** {', '.join(train_batches)} + {', '.join(other_batches) if other_batches else 'N/A'}"
    )
  
  # Ratios within selected batch
  st.markdown("#### Split Ratios (within selected batch)")
  enable_validation = st.checkbox(
    "Create validation set",
    value=True,
    key="bf_enable_val",
    help="Used only for reporting metrics on a held-out set (no hyperparameter search)."
  )
  col1, col2, col3 = st.columns(3)
  with col1:
    train_ratio = st.slider("Train", 0.5, 0.9, 0.7, 0.05, key="bf_train")
  with col2:
    val_ratio_default = 0.1 if enable_validation else 0.0
    val_ratio = st.slider(
      "Val",
      0.0,
      0.3,
      val_ratio_default,
      0.05,
      key="bf_val",
      disabled=not enable_validation,
      help="Validation metrics only; set to 0 to disable."
    )
    if not enable_validation:
      val_ratio = 0.0
  with col3:
    test_ratio = 1.0 - train_ratio - val_ratio
    st.metric("Test", f"{test_ratio:.0%}")
  
  # Stratification option
  st.markdown("#### Stratification Mode")
  with st.container(border=True):
    strat_mode = st.radio(
      "How should the base split be stratified?",
      options=['batch-only', 'batch+labels'],
      format_func=lambda x: {
        'batch-only': 'Batch Only (Recommended for cross-batch)',
        'batch+labels': 'Batch + Cell Types'
      }.get(x),
      horizontal=True,
      key="bf_strat_mode",
      help="Controls stratification of the initial 70/10/20 split before batch filtering."
    )

    if strat_mode == 'batch-only':
      st.caption("**Recommended for cross-batch generalization.** "
           "Each batch gets proportional representation, cell type balance may vary.")
    else:
      st.caption("**Preserves both batch AND cell type proportions.** "
           "May fail if some (batch, cell type) combinations have very few cells.")
      st.warning("May fail with rare cell types in small batches.")

  stratify_by_labels = strat_mode == 'batch+labels'
  
  # Label column
  label_col = 'Group'
  for col in ['labels', 'Group', 'celltype']:
    if col in adata.obs.columns:
      label_col = col
      break
  
  # Test Strategy
  st.markdown("#### Test Set Strategy")
  test_strategy = st.radio(
    "Evaluate generalization on:",
    ['all', 'unseen', 'selected'],
    format_func=lambda x: {
      'all': 'All Batches (Standard Generalization)',
      'unseen': 'Unseen Batches Only (Strict Generalization)',
      'selected': 'Same Batches as Train (In-Distribution)'
    }.get(x),
    help="Choose which batches to include in the Test set."
  )
  
  test_batches = None
  use_stratified_base = True
  
  if test_strategy == 'all':
    st.info("Test set will be a stratified sample from **ALL** batches.")
    use_stratified_base = True
    test_batches = None # Splitter interprets this as all
  elif test_strategy == 'unseen':
    st.info("Test set will contain **only batches NOT in training set**.")
    use_stratified_base = False
    other_batches = [b for b in batches if b not in train_batches]
    if not other_batches:
      st.error("No unseen batches available! Please select fewer training batches.")
    test_batches = other_batches
  else: # selected
    st.info("Test set will contain **only the same batches as training set**.")
    use_stratified_base = False
    test_batches = train_batches

  if not use_stratified_base:
    st.caption(
      "Strict batch mode: train uses all cells from selected train batch(es), "
      "test uses all cells from selected test batch(es). "
      "In this mode, Train/Test ratio sliders are informational; only Validation ratio is used "
      "to carve out a validation subset from train batches."
    )
    if enable_validation:
      st.caption("Set 'Create validation set' to OFF for 100% train-batch usage.")

  return {
    'mode': 'batch',
    'train_batches': train_batches,
    'test_batches': test_batches, # Added explicit test batches
    'train_ratio': train_ratio,
    'val_ratio': val_ratio,
    'test_ratio': test_ratio,
    'label_col': label_col,
    'use_stratified_base': use_stratified_base, # Controlled by strategy
    'stratify_by_labels': stratify_by_labels,
    'use_validation': enable_validation
  }


def _apply_split(handler, settings):
  """Apply the split and store in session state."""
  try:
    splitter = DatasetSplitter()

    if not settings.get('use_validation', True):
      settings['val_ratio'] = 0.0
      settings['test_ratio'] = 1.0 - settings['train_ratio']

    if settings.get('mode') == 'batch' and not settings.get('train_batches'):
      st.error("Select at least one training batch.")
      return
    if settings.get('train_ratio', 0) <= 0:
      st.error("Train ratio must be > 0.")
      return
    if settings.get('test_ratio', 0) <= 0:
      st.error("Test ratio must be > 0.")
      return
    if settings.get('val_ratio', 0) < 0:
      st.error("Validation ratio must be >= 0.")
      return
    
    # Determine working adata and balance mode
    working_adata = handler.adata
    balance_mode = settings.get('balance_mode', 'eliminate')
    do_balance = settings.get('balance', False)

    with st.spinner("Processing dataset..."):
      # Step 1: Pre-split balancing (only for 'eliminate' mode)
      if do_balance and balance_mode == 'eliminate':
        target = settings.get('balance_target')
        batch_col = settings.get('batch_col')
        st.info(f"Step 1: Balancing batches to max {target} cells (eliminate mode)...")

        bal_label_col = settings.get('label_col', 'Group')

        working_adata = splitter.balance_by_batch(
          handler.adata,
          batch_col=batch_col,
          target_per_batch=target,
          preserve_labels=True,
          label_col=bal_label_col
        )

      # Step 2: Splitting
      if settings['mode'] == 'stratified':
        split_result = splitter.split_stratified(
          working_adata,
          train_ratio=settings['train_ratio'],
          val_ratio=settings['val_ratio'],
          test_ratio=settings['test_ratio'],
          label_col=settings['label_col'],
          stratify_by_batch=settings.get('stratify_by_batch', True),
          stratify_by_labels=settings.get('stratify_by_labels', False),
          batch_col=settings.get('batch_col')
        )
      else:
        # Batch-filtered mode
        split_result = splitter.split_by_batch(
          working_adata,
          train_batches=settings['train_batches'],
          test_batches=settings.get('test_batches'),
          val_ratio=settings['val_ratio'],
          batch_col=settings['batch_col'],
          train_ratio=settings['train_ratio'],
          test_ratio=settings['test_ratio'],
          use_stratified_base=settings.get('use_stratified_base', True),
          stratify_by_labels=settings.get('stratify_by_labels', False)
        )

      # Step 3: Post-split balancing with reinjection (for reinject modes)
      if do_balance and balance_mode in ('reinject_train', 'reinject_both'):
        target = settings.get('balance_target')
        batch_col = settings.get('batch_col')
        bal_label_col = settings.get('label_col', 'Group')

        reinject_to = 'train' if balance_mode == 'reinject_train' else 'both'
        st.info(f"Step 3: Balancing TEST and reinjecting excess into {reinject_to.upper()}...")

        split_result = splitter.balance_test_and_reinject(
          split_result,
          batch_col=batch_col,
          target_per_batch=target,
          reinject_to=reinject_to,
          preserve_labels=True,
          label_col=bal_label_col
        )

        n_reinjected = split_result.split_info.get('n_reinjected', 0)
        st.success(f"Reinjected {n_reinjected} cells from TEST into {reinject_to.upper()}")
      
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
      user_message="Benchmark split configuration failed.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="data_split_apply",
    )
