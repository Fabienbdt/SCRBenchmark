"""
Shared GUI components for SCRBenchmark.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.dataset_splitter import get_batch_column


def _render_html_table(table, *, hide_index=False):
  """Render a small table without Streamlit's PyArrow conversion layer.

  Streamlit converts ``st.dataframe`` inputs through PyArrow.  Some macOS ARM
  combinations of pandas/PyArrow crash at native level while converting a
  ``Styler`` (there is no Python exception to catch).  The tables in this
  component are small summary tables, so HTML is both sufficient and safer.
  DataFrame values are escaped by pandas; callers must configure Styler
  instances with ``escape="html"`` before passing them here.
  """
  if isinstance(table, pd.DataFrame):
    html = table.to_html(index=not hide_index, escape=True, border=0)
  else:
    html = table.to_html()

  st.markdown(
    f'<div style="max-width:100%;overflow-x:auto">{html}</div>',
    unsafe_allow_html=True,
  )


def display_batch_and_distribution_info(adata, handler=None, title_prefix=""):
  """
  Display batch/dataset source information and distribution matrix.
  
  Args:
    adata: AnnData object to visualize.
    handler: DataHandler instance (optional, used for label key detection).
    title_prefix: Optional prefix for section titles.
  """
  if adata is None:
    return
  
  # Detect batch column
  batch_col = get_batch_column(adata)
  
  if batch_col is None:
    return
  
  st.subheader(f"{title_prefix}Batch/Dataset Distribution")
  st.caption(f"Detected batch column: `{batch_col}`")
  
  # Define suffix early to avoid UnboundLocalError in finally/except blocks
  viz_key_suffix = f"_{title_prefix.strip()}" if title_prefix else ""
  
  batches = adata.obs[batch_col]
  batch_counts = batches.value_counts()
  
  # --- Detailed Batch Statistics ---
  st.markdown("#### Batch Statistics")
  
  try:
    batch_stats = []
    total_genes = adata.n_vars  # Total genes in dataset
    
    for batch in batch_counts.index:
      # Subset data - convert mask to numpy to avoid pandas Series compatibility issues
      mask = (adata.obs[batch_col] == batch).values
      if sparse.issparse(adata.X):
        batch_X = adata.X[mask]
        # Genes per cell (nnz per row)
        genes_per_cell = batch_X.getnnz(axis=1)
        # Counts per cell
        counts_per_cell = np.array(batch_X.sum(axis=1)).flatten()
        # Sparsity
        sparsity = 1.0 - (batch_X.nnz / (batch_X.shape[0] * batch_X.shape[1]))
        # N genes detected in batch (at least 1 non-zero value across all cells)
        n_genes_detected = batch_X.getnnz(axis=0)
        n_genes_detected = int(np.sum(n_genes_detected > 0))
      else:
        # Dense matrix handling - Ensure numpy array to avoid Series/DataFrame attribute errors
        batch_X = adata.X[mask]
        batch_X = np.asarray(batch_X)
        
        genes_per_cell = np.count_nonzero(batch_X, axis=1)
        counts_per_cell = np.sum(batch_X, axis=1)
        sparsity = 1.0 - (np.count_nonzero(batch_X) / batch_X.size)
        # N genes detected in batch
        n_genes_detected = int(np.sum(np.any(batch_X > 0, axis=0)))

      batch_stats.append({
        'Batch': batch,
        'N Cells': int(np.sum(mask)),
        'N Genes Detected': n_genes_detected,
        'Mean Genes/Cell': float(np.mean(genes_per_cell)),
        'Median Genes': float(np.median(genes_per_cell)),
        'Mean Counts': float(np.mean(counts_per_cell)),
        'Median Counts': float(np.median(counts_per_cell)),
        'Sparsity (%)': float(sparsity * 100)
      })

    stats_df = pd.DataFrame(batch_stats).set_index('Batch')

    # Compute gene overlap statistics
    all_n_genes = [s['N Genes Detected'] for s in batch_stats]
    max_genes = max(all_n_genes)
    min_genes = min(all_n_genes)

    # Display gene overlap info with explanation
    if max_genes == min_genes == total_genes:
      st.success(f"All batches have the same {total_genes:,} genes expressed (intersection mode)")
    elif min_genes < total_genes * 0.9:
      st.warning(
        f"Gene coverage varies: {min_genes:,} - {max_genes:,} / {total_genes:,} total (likely **union** mode with missing genes)\n\n"
        f"**N Genes Detected** = number of genes with at least one non-zero count in the batch. "
        f"This is typically lower than the total gene count ({total_genes:,}) because in union mode, "
        f"genes absent from a study's original data are filled with zeros. "
        f"It can also differ from the raw gene count of the source study if some genes "
        f"have zero expression across all cells of a given batch."
      )
    else:
      st.info(f"Gene coverage: {min_genes:,} - {max_genes:,} / {total_genes:,} total")
    
    # Display table with formatting
    _render_html_table(
      stats_df.style.format({
        'N Cells': '{:,}',
        'N Genes Detected': '{:,}',
        'Mean Genes/Cell': '{:.1f}',
        'Median Genes': '{:.0f}',
        'Mean Counts': '{:.1f}',
        'Median Counts': '{:.0f}',
        'Sparsity (%)': '{:.2f}%'
      }, escape="html")
    )
    st.caption(
      "**N Genes Detected** = genes with at least one non-zero count across all cells of the batch. "
      "This can be lower than the source study's total gene count because: "
      "(1) in union mode, genes absent from a study are filled with 0; "
      "(2) some genes may have zero expression in all cells of a specific batch; "
      "(3) multi-batch studies (e.g. Baron inDrop1-4) split their genes across batches."
    )

    # --- Visual Comparison ---
    st.markdown("#### Comparative Distributions")
    # viz_key_suffix defined at top
    viz_tab1, viz_tab2 = st.tabs(["Genes per Cell", "Total Counts"])
    
    # Prepare data for plotting
    plot_data = []
    for batch in batch_counts.index:
      mask = (adata.obs[batch_col] == batch).values
      if sparse.issparse(adata.X):
        batch_X = adata.X[mask]
        genes = batch_X.getnnz(axis=1)
        counts = np.array(batch_X.sum(axis=1)).flatten()
      else:
        batch_X = adata.X[mask]
        batch_X = np.asarray(batch_X)
        
        genes = np.count_nonzero(batch_X, axis=1)
        counts = np.sum(batch_X, axis=1)
      
      # Subsample if too large for plotting performance
      if len(genes) > 1000:
        indices = np.random.choice(len(genes), 1000, replace=False)
        genes = genes[indices]
        counts = counts[indices]
        
      for g, c in zip(genes, counts):
        plot_data.append({'Batch': str(batch), 'Genes': g, 'Counts': c})
    
    plot_df = pd.DataFrame(plot_data)
    
    with viz_tab1:
      fig, ax = plt.subplots(figsize=(10, 5))
      sns.boxplot(data=plot_df, x='Batch', y='Genes', ax=ax, palette="Set3")
      ax.set_title("Genes Detected per Cell by Batch")
      ax.tick_params(axis='x', rotation=45)
      st.pyplot(fig)
      plt.close()
      
    with viz_tab2:
      fig, ax = plt.subplots(figsize=(10, 5))
      sns.boxplot(data=plot_df, x='Batch', y='Counts', ax=ax, palette="Set3")
      ax.set_title("Total Counts per Cell by Batch")
      ax.tick_params(axis='x', rotation=45)
      st.pyplot(fig)
      plt.close()

  except Exception as e:
    st.error(f"Could not compute detailed batch statistics: {str(e)}")
    # Fallback to simple display
    col1, col2 = st.columns([2, 3])
    with col1:
      df = pd.DataFrame({
        'Batch': batch_counts.index,
        'Cells': batch_counts.values,
        'Percentage': (batch_counts.values / batch_counts.sum() * 100).round(2)
      })
      _render_html_table(df, hide_index=True)
  
  # --- Distribution Matrix (Label vs Batch) ---
  st.markdown("---")
  # Identify labels
  label_col = None
  possible_labels = ['Group', 'labels', 'cell_type', 'celltype', 'CellType', 'cluster', 'Cluster', 'Sub_Cluster']
  
  # Try using handler if provided
  if handler and hasattr(handler, 'label_key') and handler.label_key in adata.obs.columns:
    label_col = handler.label_key
  else:
    for col in possible_labels:
      if col in adata.obs.columns:
        label_col = col
        break

  if label_col:
    st.markdown(f"### {title_prefix}Cell Distribution Matrix")
    st.caption(f"Number of cells per label across batches (Source: `{label_col}`)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
      matrix_view = st.radio(
        "Matrix View",
        options=["Per Batch", "Total"],
        horizontal=True,
        key=f"dist_matrix_view_radio{viz_key_suffix}" # Unique key
      )
    
    if matrix_view == "Per Batch" and batch_col:
      dist_matrix = pd.crosstab(
        adata.obs[label_col], 
        adata.obs[batch_col],
        margins=True,
        margins_name="Total"
      )
      # Cast to string to avoid Arrow mixed-type errors (int labels + "Total" string)
      dist_matrix.index = dist_matrix.index.astype(str)
      dist_matrix.columns = dist_matrix.columns.astype(str)
      # Use styling for better readability
      _render_html_table(
        dist_matrix.style.format(escape="html").background_gradient(
          cmap="Blues", axis=None
        )
      )
    else:
      total_dist = adata.obs[label_col].value_counts().to_frame()
      total_dist.columns = ['Cell Count']
      total_dist['Percentage (%)'] = (total_dist['Cell Count'] / total_dist['Cell Count'].sum() * 100).round(2)
      _render_html_table(total_dist)
    
    with st.expander("Matrix Details"):
      st.markdown("""
      This matrix shows the composition of your dataset in terms of biological types (labels) and technical sources (batches).
      
      *  **Per Batch:** Useful to see if certain cell types are missing in specific batches (source of bias).
      *  **Sparsity:** A sparse matrix (many zeros) indicates that batches are heterogeneous and don't share all biological states.
      *  **Margins:** Shows total cells per type and total cells per batch.
      """)
  else:
    st.info("No label column detected. Cannot display distribution matrix.")

  
  # --- Batch Inspector (Dynamic View) ---
  if title_prefix == "": # Only show inspector on upload page to avoid clutter
    st.markdown("---")
    st.subheader("Batch Inspector")
    st.caption("Explore detailed composition of each batch individually.")

    if label_col:
      col_sel, col_viz = st.columns([1, 2])
      
      with col_sel:
        selected_batch_view = st.selectbox(
          "Select a Batch", 
          options=list(batch_counts.index),
          key="batch_inspector_select"
        )
        
        # Compute specific stats for this batch
        mask = adata.obs[batch_col] == selected_batch_view
        n_cells_batch = mask.sum()
        n_classes_batch = adata.obs[mask][label_col].nunique()
        
        st.metric("Number of Cells", f"{n_cells_batch:,}")
        st.metric("Number of Classes", f"{n_classes_batch}")
        st.caption(f"Label used: `{label_col}`")

      with col_viz:
        # Prepare data
        batch_obs = adata.obs[mask]
        class_counts = batch_obs[label_col].value_counts().reset_index()
        class_counts.columns = ['Cell Type', 'Count']
        
        # Sort for better visualization
        class_counts = class_counts.sort_values('Count', ascending=False)
        
        # Plot
        fig_bi, ax_bi = plt.subplots(figsize=(8, 5))
        
        # Color based on count intensity or unique labels
        sns.barplot(
          data=class_counts, 
          y='Cell Type', 
          x='Count', 
          palette='viridis', 
          ax=ax_bi
        )
        
        ax_bi.set_title(f"Cell Type Distribution - {selected_batch_view}")
        ax_bi.set_xlabel("Number of Cells")
        ax_bi.set_ylabel("")
        sns.despine()
        
        st.pyplot(fig_bi)
        plt.close()
        
        with st.expander("Chart Details"):
          st.markdown("""
          **Distribution by Class:** 
          This chart shows how many cells of each biological type are present in the selected batch.
          
          *  **Utility:** Helps verify if a batch is "balanced" (contains all cell types) or "biased" (contains only a specific subset, e.g., immune cells only).
          *  **Imbalance:** If distributions vary significantly between batches, integration becomes harder and might require aggressive batch correction.
          """)
    else:
      st.info("No cell type column detected (e.g., 'Group', 'cell_type'). Cannot display class distribution.")

  # Store batch column in session for benchmark mode
  st.session_state['detected_batch_column'] = batch_col
  st.session_state['available_batches'] = list(batch_counts.index)
  
  if title_prefix == "":
    # Info for benchmark mode
    st.info(f"This dataset contains **{len(batch_counts)} batches**. "
        f"You can use **Benchmark Mode** in Analysis to evaluate "
        f"generalization on unseen batches.")
