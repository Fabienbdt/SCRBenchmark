"""
Data upload page for the Streamlit interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_handler import DataHandler
from utils.reference_datasets import get_available_datasets, load_reference_dataset
from utils.dataset_splitter import get_batch_column
from utils.batch_correction import check_scvi_available


from gui.shared_components import display_batch_and_distribution_info
from gui.widgets import display_error
from gui.state_manager import KEYS, invalidate, set_with_cascade

def _display_batch_info(handler):
  """Display batch/dataset source information if available."""
  display_batch_and_distribution_info(handler.get_data(), handler)


def _compute_qc_metrics(adata):
  """
  Compute quality control metrics for scRNA-seq data.

  Returns a dictionary with:
  - n_genes_by_counts: number of genes expressed per cell
  - total_counts: total UMI counts per cell
  - pct_counts_mt: percentage of mitochondrial counts per cell
  - n_cells_by_counts: number of cells expressing each gene
  """
  X = adata.X
  if sparse.issparse(X):
    X_dense = X.toarray()
  else:
    X_dense = np.asarray(X)

  # Ensure non-negative values for count-based metrics
  X_counts = np.maximum(X_dense, 0)

  # Genes per cell (number of genes with non-zero expression)
  n_genes_by_counts = np.sum(X_counts > 0, axis=1)

  # Total counts per cell
  total_counts = np.sum(X_counts, axis=1)

  # Cells per gene
  n_cells_by_counts = np.sum(X_counts > 0, axis=0)

  # Mitochondrial genes percentage
  gene_names = adata.var_names.str.upper() if hasattr(adata.var_names, 'str') else [str(g).upper() for g in adata.var_names]
  mt_mask = np.array([g.startswith('MT-') or g.startswith('MT.') for g in gene_names])

  if np.any(mt_mask):
    mt_counts = np.sum(X_counts[:, mt_mask], axis=1)
    pct_counts_mt = (mt_counts / (total_counts + 1e-10)) * 100
  else:
    pct_counts_mt = np.zeros(X_counts.shape[0])

  return {
    'n_genes_by_counts': np.asarray(n_genes_by_counts).flatten(),
    'total_counts': np.asarray(total_counts).flatten(),
    'pct_counts_mt': np.asarray(pct_counts_mt).flatten(),
    'n_cells_by_counts': np.asarray(n_cells_by_counts).flatten(),
    'has_mt_genes': np.any(mt_mask)
  }


def _render_qc_violin_plots(qc_metrics, labels=None):
  """
  Render violin plots for QC metrics to help choose preprocessing parameters.
  """
  st.subheader("Data Quality Assessment")
  st.caption("Use these distributions to choose appropriate preprocessing thresholds")

  # Create figure with subplots
  fig, axes = plt.subplots(1, 3, figsize=(14, 4))

  # Color palette
  palette = sns.color_palette("husl", 1)

  # 1. Genes per cell
  ax1 = axes[0]
  data1 = qc_metrics['n_genes_by_counts']
  sns.violinplot(y=data1, ax=ax1, color=palette[0], inner='box')
  ax1.set_ylabel('Count')
  ax1.set_title('Genes per Cell', fontweight='bold')

  # Add statistics
  median_genes = np.median(data1)
  q1_genes = np.percentile(data1, 25)
  q3_genes = np.percentile(data1, 75)
  ax1.axhline(y=median_genes, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_genes:.0f}')
  ax1.legend(loc='upper right', fontsize=8)

  # 2. Total counts per cell
  ax2 = axes[1]
  data2 = qc_metrics['total_counts']
  sns.violinplot(y=data2, ax=ax2, color=palette[0], inner='box')
  ax2.set_ylabel('Count')
  ax2.set_title('Total Counts per Cell', fontweight='bold')

  median_counts = np.median(data2)
  ax2.axhline(y=median_counts, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_counts:.0f}')
  ax2.legend(loc='upper right', fontsize=8)

  # 3. Mitochondrial percentage (or cells per gene if no MT genes)
  ax3 = axes[2]
  if qc_metrics['has_mt_genes']:
    data3 = qc_metrics['pct_counts_mt']
    sns.violinplot(y=data3, ax=ax3, color=palette[0], inner='box')
    ax3.set_ylabel('Percentage')
    ax3.set_title('Mitochondrial %', fontweight='bold')

    median_mt = np.median(data3)
    ax3.axhline(y=median_mt, color='red', linestyle='--', alpha=0.7, label=f'Median: {median_mt:.1f}%')
    ax3.legend(loc='upper right', fontsize=8)
  else:
    # Show cells per gene distribution instead
    data3 = qc_metrics['n_cells_by_counts']
    # Log transform for better visualization
    data3_log = np.log1p(data3)
    sns.violinplot(y=data3_log, ax=ax3, color=palette[0], inner='box')
    ax3.set_ylabel('log(Cells + 1)')
    ax3.set_title('Cells per Gene (log)', fontweight='bold')

    median_cells = np.median(data3)
    ax3.axhline(y=np.log1p(median_cells), color='red', linestyle='--', alpha=0.7,
          label=f'Median: {median_cells:.0f}')
    ax3.legend(loc='upper right', fontsize=8)

  plt.tight_layout()
  st.pyplot(fig)
  plt.close()

  # Statistics summary
  _render_qc_statistics(qc_metrics)


def _render_qc_statistics(qc_metrics):
  """Display QC statistics with recommendations."""

  st.markdown("#### Quality Statistics")

  col1, col2, col3, col4 = st.columns(4)

  genes_data = qc_metrics['n_genes_by_counts']
  counts_data = qc_metrics['total_counts']
  cells_data = qc_metrics['n_cells_by_counts']

  with col1:
    st.markdown("**Genes/Cell**")
    st.write(f"Min: {np.min(genes_data):.0f}")
    st.write(f"Median: {np.median(genes_data):.0f}")
    st.write(f"Max: {np.max(genes_data):.0f}")
    st.write(f"Q1: {np.percentile(genes_data, 25):.0f}")
    st.write(f"Q3: {np.percentile(genes_data, 75):.0f}")

  with col2:
    st.markdown("**Counts/Cell**")
    st.write(f"Min: {np.min(counts_data):.0f}")
    st.write(f"Median: {np.median(counts_data):.0f}")
    st.write(f"Max: {np.max(counts_data):.0f}")
    st.write(f"Q1: {np.percentile(counts_data, 25):.0f}")
    st.write(f"Q3: {np.percentile(counts_data, 75):.0f}")

  with col3:
    st.markdown("**Cells/Gene**")
    st.write(f"Min: {np.min(cells_data):.0f}")
    st.write(f"Median: {np.median(cells_data):.0f}")
    st.write(f"Max: {np.max(cells_data):.0f}")
    st.write(f"Genes in <3 cells: {np.sum(cells_data < 3)}")
    st.write(f"Genes in <10 cells: {np.sum(cells_data < 10)}")

  with col4:
    if qc_metrics['has_mt_genes']:
      mt_data = qc_metrics['pct_counts_mt']
      st.markdown("**Mito %**")
      st.write(f"Min: {np.min(mt_data):.1f}%")
      st.write(f"Median: {np.median(mt_data):.1f}%")
      st.write(f"Max: {np.max(mt_data):.1f}%")
      st.write(f"Cells >5%: {np.sum(mt_data > 5)}")
      st.write(f"Cells >10%: {np.sum(mt_data > 10)}")
    else:
      st.markdown("**Mito genes**")
      st.write("Not detected")
      st.caption("(MT-* pattern)")

  # Recommendations box
  st.markdown("---")
  st.markdown("#### Preprocessing Recommendations")

  recommendations = []

  # Min genes recommendation
  q5_genes = np.percentile(genes_data, 5)
  if q5_genes < 200:
    recommendations.append(f"Consider `min_genes_per_cell` around **{max(100, int(q5_genes))}-200** to filter low-quality cells")
  else:
    recommendations.append(f"Consider `min_genes_per_cell` around **{int(q5_genes):.0f}** (5th percentile)")

  # Max genes recommendation
  q95_genes = np.percentile(genes_data, 95)
  if q95_genes > 5000:
    recommendations.append(f"Consider `max_genes_per_cell` around **{int(q95_genes):.0f}** (95th percentile) to filter doublets")

  # Min cells recommendation
  pct_rare = (np.sum(cells_data < 3) / len(cells_data)) * 100
  if pct_rare > 50:
    recommendations.append(f"**{pct_rare:.1f}%** of genes are in <3 cells. Consider `min_cells_per_gene = 3` to filter rare genes")

  # MT recommendation
  if qc_metrics['has_mt_genes']:
    mt_data = qc_metrics['pct_counts_mt']
    high_mt_pct = (np.sum(mt_data > 10) / len(mt_data)) * 100
    if high_mt_pct > 5:
      recommendations.append(f"**{high_mt_pct:.1f}%** cells have >10% mitochondrial reads - consider filtering stressed/dead cells")

  for rec in recommendations:
    st.info(rec)


def _render_cells_per_gene_distribution(qc_metrics):
  """Render histogram for cells per gene to help choose min_cells_per_gene."""

  st.markdown("#### Cells per Gene Distribution")
  st.caption("Helps choose the `min_cells_per_gene` threshold")

  cells_data = qc_metrics['n_cells_by_counts']

  fig, ax = plt.subplots(figsize=(10, 4))

  # Use log scale for x-axis
  bins = np.logspace(0, np.log10(max(cells_data) + 1), 50)
  ax.hist(cells_data, bins=bins, color='steelblue', edgecolor='white', alpha=0.7)
  ax.set_xscale('log')
  ax.set_xlabel('Number of Cells (log scale)')
  ax.set_ylabel('Number of Genes')
  ax.set_title('Distribution of Cells per Gene')

  # Add vertical lines for common thresholds
  for threshold, color, label in [(3, 'red', 'min=3'), (5, 'orange', 'min=5'), (10, 'green', 'min=10')]:
    ax.axvline(x=threshold, color=color, linestyle='--', alpha=0.7, label=label)

  ax.legend()

  plt.tight_layout()
  st.pyplot(fig)
  plt.close()

  # Show filtering impact
  col1, col2, col3 = st.columns(3)
  with col1:
    kept = np.sum(cells_data >= 3)
    st.metric("Genes if min=3", f"{kept:,}", f"{kept/len(cells_data)*100:.1f}%")
  with col2:
    kept = np.sum(cells_data >= 5)
    st.metric("Genes if min=5", f"{kept:,}", f"{kept/len(cells_data)*100:.1f}%")
  with col3:
    kept = np.sum(cells_data >= 10)
    st.metric("Genes if min=10", f"{kept:,}", f"{kept/len(cells_data)*100:.1f}%")


def _render_batch_gene_inspector_upload(adata):
  """Render batch gene inspector for data upload page."""
  import io
  from utils.statistics import compute_highly_expressed_genes_by_group
  import utils.visualization as viz
  
  batch_col = get_batch_column(adata)
  
  if not batch_col:
    st.info("No batch column detected. This analysis requires batch annotations (e.g., 'batch', 'Batch', 'tech', 'sample').")
    return
  
  # Settings
  col1, col2, col3 = st.columns(3)
  
  with col1:
    n_genes = st.number_input(
      "Number of genes per batch",
      min_value=5, max_value=50, value=10, step=5,
      key="upload_batch_n_genes"
    )
  
  with col2:
    method = st.selectbox(
      "Ranking method",
      options=['mean', 'median', 'fraction_expressed'],
      format_func=lambda x: {'mean': 'Mean Expression', 'median': 'Median Expression', 'fraction_expressed': 'Fraction Expressing'}[x],
      key="upload_batch_method"
    )
  
  with col3:
    st.metric("Batch Column", batch_col)
    st.caption(f"{adata.obs[batch_col].nunique()} batch(es)")
  
  with st.expander("ℹ️ How are top genes determined?"):
    st.markdown("""
    **⚠️ This is NOT differential expression analysis**
    
    This identifies genes with **high expression levels** within each batch (not batch-specific markers).
    
    - **Mean Expression**: Average expression across all cells in batch
    - **Median Expression**: Median expression (robust to outliers)  
    - **Fraction Expressing**: % of cells with expression > 0
    """)
  
  if st.button("Compute Top Genes by Batch", key="upload_batch_compute", type="primary"):
    with st.spinner(f"Computing top {n_genes} genes per batch..."):
      try:
        genes_by_batch = compute_highly_expressed_genes_by_group(adata, groupby=batch_col, n_genes=n_genes, method=method)
        st.session_state['upload_batch_genes'] = genes_by_batch
        st.success(f"✓ Analysis complete for {len(genes_by_batch)} batches")
      except Exception as e:
        st.error(f"Error: {e}")
  
  # Display results
  if 'upload_batch_genes' in st.session_state:
    genes_by_batch = st.session_state['upload_batch_genes']
    
    viz_type = st.radio("View", ['Heatmap', 'Bar Charts', 'Tables'], horizontal=True, key="upload_batch_viz")
    
    if viz_type == 'Heatmap':
      fig = viz.plot_top_genes_by_group_heatmap(genes_by_batch, n_genes=n_genes, title=f"Top {n_genes} Genes by Batch")
      if fig:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download Heatmap", buf.getvalue(), f"batch_genes_heatmap.png", "image/png")
        plt.close(fig)
    elif viz_type == 'Bar Charts':
      fig = viz.plot_top_genes_by_group_bars(genes_by_batch, n_genes=n_genes, title=f"Top {n_genes} Genes by Batch")
      if fig:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download Charts", buf.getvalue(), f"batch_genes_bars.png", "image/png")
        plt.close(fig)
    else:
      batch_names = sorted(genes_by_batch.keys())
      if len(batch_names) <= 8:
        tabs = st.tabs(batch_names)
        for tab, name in zip(tabs, batch_names):
          with tab:
            st.dataframe(genes_by_batch[name], hide_index=True)
      else:
        sel = st.selectbox("Select batch", batch_names, key="upload_batch_sel")
        st.dataframe(genes_by_batch[sel], hide_index=True)


def _render_celltype_gene_inspector_upload(adata):
  """Render cell type gene inspector for data upload page."""
  import io
  from utils.statistics import compute_highly_expressed_genes_by_group
  import utils.visualization as viz
  
  # Detect cell type column
  celltype_col = None
  for col in ['Group', 'labels', 'cell_type', 'celltype', 'CellType']:
    if col in adata.obs.columns:
      celltype_col = col
      break
  
  if not celltype_col:
    st.info("No cell type column detected. This analysis requires cell type annotations (e.g., 'Group', 'labels', 'cell_type').")
    return
  
  # Settings
  col1, col2, col3 = st.columns(3)
  
  with col1:
    n_genes = st.number_input(
      "Number of genes per cell type",
      min_value=5, max_value=50, value=10, step=5,
      key="upload_celltype_n_genes"
    )
  
  with col2:
    method = st.selectbox(
      "Ranking method",
      options=['mean', 'median', 'fraction_expressed'],
      format_func=lambda x: {'mean': 'Mean Expression', 'median': 'Median Expression', 'fraction_expressed': 'Fraction Expressing'}[x],
      key="upload_celltype_method"
    )
  
  with col3:
    st.metric("Cell Type Column", celltype_col)
    st.caption(f"{adata.obs[celltype_col].nunique()} cell type(s)")
  
  # Batch filter option
  batch_col = get_batch_column(adata)
  filter_by_batch = False
  selected_batch = None
  
  if batch_col:
    col_b1, col_b2 = st.columns(2)
    with col_b1:
      filter_by_batch = st.checkbox("Filter by batch", value=False, key="upload_celltype_filter")
    if filter_by_batch:
      with col_b2:
        batches = sorted(adata.obs[batch_col].unique().astype(str))
        selected_batch = st.selectbox("Batch", batches, key="upload_celltype_batch")
  
  with st.expander("ℹ️ How are top genes determined?"):
    st.markdown("""
    **⚠️ This is NOT differential expression analysis**
    
    This identifies genes with **high expression levels** within each cell type (not cell type-specific markers).
    
    - **Mean Expression**: Average expression across all cells of this type
    - **Median Expression**: Median expression (robust to outliers)
    - **Fraction Expressing**: % of cells with expression > 0
    
    For **differential expression** (finding markers), use the "Group Gene Inspector" in the Analysis page.
    """)
  
  if st.button("Compute Top Genes by Cell Type", key="upload_celltype_compute", type="primary"):
    with st.spinner(f"Computing top {n_genes} genes per cell type..."):
      try:
        if filter_by_batch and selected_batch:
          adata_sub = adata[adata.obs[batch_col].astype(str) == selected_batch].copy()
          st.info(f"Analyzing {adata_sub.n_obs} cells from batch '{selected_batch}'")
        else:
          adata_sub = adata
        
        genes_by_ct = compute_highly_expressed_genes_by_group(adata_sub, groupby=celltype_col, n_genes=n_genes, method=method)
        st.session_state['upload_celltype_genes'] = genes_by_ct
        st.success(f"✓ Analysis complete for {len(genes_by_ct)} cell types")
      except Exception as e:
        st.error(f"Error: {e}")
  
  # Display results
  if 'upload_celltype_genes' in st.session_state:
    genes_by_ct = st.session_state['upload_celltype_genes']
    
    viz_type = st.radio("View", ['Heatmap', 'Bar Charts', 'Tables'], horizontal=True, key="upload_celltype_viz")
    
    if viz_type == 'Heatmap':
      fig = viz.plot_top_genes_by_group_heatmap(genes_by_ct, n_genes=n_genes, title=f"Top {n_genes} Genes by Cell Type")
      if fig:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download Heatmap", buf.getvalue(), f"celltype_genes_heatmap.png", "image/png")
        plt.close(fig)
    elif viz_type == 'Bar Charts':
      fig = viz.plot_top_genes_by_group_bars(genes_by_ct, n_genes=n_genes, title=f"Top {n_genes} Genes by Cell Type")
      if fig:
        st.pyplot(fig)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download Charts", buf.getvalue(), f"celltype_genes_bars.png", "image/png")
        plt.close(fig)
    else:
      ct_names = sorted(genes_by_ct.keys())
      if len(ct_names) <= 8:
        tabs = st.tabs(ct_names)
        for tab, name in zip(tabs, ct_names):
          with tab:
            st.dataframe(genes_by_ct[name], hide_index=True)
      else:
        sel = st.selectbox("Select cell type", ct_names, key="upload_celltype_sel")
        st.dataframe(genes_by_ct[sel], hide_index=True)


def _render_differential_expression_upload(adata):
  """Render differential expression analysis for data upload page."""
  import io
  from utils.statistics import compute_marker_genes
  import utils.visualization as viz
  
  st.markdown("#### Differential Expression Analysis")
  st.caption("Find marker genes that distinguish one group from all others (one-vs-rest comparison).")
  
  # Detect group column
  group_col = None
  for col in ['Group', 'labels', 'cell_type', 'celltype', 'CellType', 'cluster', 'Cluster']:
    if col in adata.obs.columns:
      group_col = col
      break
  
  if not group_col:
    st.info("No group column detected. This analysis requires cell type or cluster annotations.")
    st.caption("Common column names: 'Group', 'labels', 'cell_type', 'cluster'")
    return
  
  groups = sorted(adata.obs[group_col].unique().astype(str))
  n_groups = len(groups)
  
  with st.expander("ℹ️ What is differential expression?"):
    st.markdown("""
    **Differential expression analysis** compares one group against all other cells to find **specific marker genes**.
    
    **Method:** Wilcoxon rank-sum test (non-parametric, robust)
    
    **Output:**
    - **Log2FC**: Log2 Fold Change (positive = upregulated in target group)
    - **P-value**: Statistical significance
    - **Adj P-value**: Bonferroni-corrected p-value
    - **Score**: Combined score (Log2FC × -log10(P-value))
    
    **Example:** If you select "B cells", the analysis finds genes that are significantly MORE expressed in B cells compared to all other cell types.
    """)
  
  # Settings
  col1, col2, col3 = st.columns(3)
  
  with col1:
    target_group = st.selectbox(
      "Target group (vs rest)",
      options=groups,
      key="upload_de_target"
    )
  
  with col2:
    n_genes = st.number_input(
      "Number of genes",
      min_value=20, max_value=500, value=100, step=20,
      key="upload_de_n_genes"
    )
  
  with col3:
    st.metric("Group Column", group_col)
    st.caption(f"{n_groups} group(s)")
  
  # Compute button
  if st.button("Find Marker Genes", key="upload_de_compute", type="primary"):
    # Get labels
    labels = adata.obs[group_col].astype(str).values
    target_count = (labels == str(target_group)).sum()
    rest_count = len(labels) - target_count
    
    if target_count < 10:
      st.warning(f"Target group has only {target_count} cells. Results may be unreliable.")
    
    with st.spinner(f"Computing differential expression: {target_group} vs rest ({rest_count} cells)..."):
      try:
        df_markers = compute_marker_genes(
          adata,
          labels=labels,
          target_cluster=str(target_group),
          method='wilcoxon',
          n_genes=n_genes
        )
        st.session_state['upload_de_results'] = df_markers
        st.session_state['upload_de_target_result'] = target_group
        
        # Count significant genes
        sig_up = len(df_markers[(df_markers['Adj P-value'] < 0.05) & (df_markers['Log2FC'] > 0.5)])
        sig_down = len(df_markers[(df_markers['Adj P-value'] < 0.05) & (df_markers['Log2FC'] < -0.5)])
        
        st.success(f"✓ Found {sig_up} upregulated and {sig_down} downregulated markers (Adj P < 0.05, |Log2FC| > 0.5)")
        
      except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())
  
  if 'upload_de_results' in st.session_state:
    df_markers = st.session_state['upload_de_results']
    target = st.session_state.get('upload_de_target_result', 'Unknown')
    
    st.markdown("---")
    
    viz_de1, viz_de2, viz_de3 = st.tabs(["Volcano Plot", "Top Genes", "Full Table"])
    
    with viz_de1:
      st.markdown(f"**Volcano Plot: {target} vs Rest**")
      st.caption("Each point is a gene. Red = significant upregulation, Blue = significant downregulation")
      
      fig = viz.plot_volcano(
        df_markers,
        title=f"Differential Expression: {target} vs Rest",
        p_threshold=0.05,
        fc_threshold=0.5
      )
      st.pyplot(fig)
      
      buf = io.BytesIO()
      fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
      st.download_button("Download Volcano Plot", buf.getvalue(), f"volcano_{target}.png", "image/png")
      plt.close(fig)
    
    with viz_de2:
      st.markdown(f"**Top 20 Marker Genes for {target}**")
      
      # Filter significant upregulated
      df_sig = df_markers[(df_markers['Adj P-value'] < 0.05) & (df_markers['Log2FC'] > 0)].head(20)
      
      if not df_sig.empty:
        fig_bar = viz.plot_top_genes(
          df_sig.rename(columns={'Score': 'Mean Expression'}),
          title=f"Top Markers for {target}"
        )
        st.pyplot(fig_bar)
        plt.close(fig_bar)
      else:
        st.info("No significant upregulated genes found.")
      
      st.dataframe(
        df_markers[df_markers['Adj P-value'] < 0.05].head(20),
        column_config={
          "Log2FC": st.column_config.NumberColumn(format="%.2f"),
          "P-value": st.column_config.NumberColumn(format="%.2e"),
          "Adj P-value": st.column_config.NumberColumn(format="%.2e"),
          "Score": st.column_config.NumberColumn(format="%.2f"),
        },
        hide_index=True
      )
    
    with viz_de3:
      st.markdown("**All Results**")
      st.dataframe(
        df_markers,
        column_config={
          "Log2FC": st.column_config.NumberColumn(format="%.3f"),
          "P-value": st.column_config.NumberColumn(format="%.2e"),
          "Adj P-value": st.column_config.NumberColumn(format="%.2e"),
          "Score": st.column_config.NumberColumn(format="%.3f"),
          "Mean Expr": st.column_config.NumberColumn(format="%.3f"),
        },
        hide_index=True
      )
      
      # Export
      csv = df_markers.to_csv(index=False)
      st.download_button("Download CSV", csv, f"markers_{target}.csv", "text/csv")


def _render_label_harmonization(adata, label_key, handler):
  """Render label harmonization section to fix duplicate/inconsistent labels."""
  labels = adata.obs[label_key].astype(str)
  unique_labels = labels.unique()
  
  # Detect potential duplicates (case-insensitive)
  label_lower_map = {}
  for lbl in unique_labels:
    lower = lbl.lower().strip()
    if lower not in label_lower_map:
      label_lower_map[lower] = []
    label_lower_map[lower].append(lbl)
  
  # Find groups with multiple variants
  duplicates = {k: v for k, v in label_lower_map.items() if len(v) > 1}
  
  # AUTO-APPLY: If duplicates found and not already harmonized this session
  harmonize_key = f"harmonized_{label_key}"
  if duplicates and harmonize_key not in st.session_state:
    mapping = {}
    for lower_name, variants in duplicates.items():
      # Use the version with capital letter as target
      target = variants[0]
      for v in variants:
        if v[0].isupper():
          target = v
          break
      for v in variants:
        if v != target:
          mapping[v] = target
    
    if mapping:
      _apply_label_mapping(adata, label_key, mapping, handler)
      st.session_state[harmonize_key] = True
      st.toast(f"✓ Auto-harmonized {len(mapping)} duplicate labels", icon="🔄")
      st.rerun()
  
  with st.expander("🔧 Label Harmonization", expanded=False):
    st.success("✓ Labels are harmonized.")
    
    st.markdown("---")
    st.markdown("**Manual label rename:**")
    
    col1, col2 = st.columns(2)
    with col1:
      old_label = st.selectbox("From", options=sorted(unique_labels), key="harmonize_from")
    with col2:
      new_label = st.text_input("To", value=old_label or "", key="harmonize_to")
    
    if st.button("Rename Label", key="harmonize_rename"):
      if old_label and new_label and old_label != new_label:
        _apply_label_mapping(adata, label_key, {old_label: new_label}, handler)
        st.success(f"✓ Renamed '{old_label}' → '{new_label}'")
        st.rerun()
      else:
        st.warning("Please select different labels.")


def _apply_label_mapping(adata, label_key, mapping, handler):
  """Apply a label mapping to the adata object and update handler."""
  # Apply mapping
  new_labels = adata.obs[label_key].astype(str).replace(mapping)
  adata.obs[label_key] = new_labels
  
  # Update handler's adata reference
  if handler and hasattr(handler, 'adata'):
    handler.adata = adata


def render_data_upload_page():
  """Render the data upload page."""
  st.header("Data Loading")

  # Create tabs for different loading methods
  tab_upload, tab_reference = st.tabs(["Upload File", "Reference Datasets"])
  
  with tab_upload:
    _render_upload_tab()
  
  with tab_reference:
    _render_reference_datasets_tab()

  # Display loaded data info
  if 'data_handler' in st.session_state and st.session_state.data_handler is not None:
    _display_data_info()


def _render_upload_tab():
  """Render the file upload tab."""
  # File upload section
  st.subheader("Upload Data File")

  uploaded_file = st.file_uploader(
    "Choose a file",
    type=['h5', 'h5ad', 'csv', 'tsv', 'txt'],
    help="Supported formats: H5AD, H5, CSV, TSV, TXT"
  )

  # Or load from path
  st.subheader("Or Load from Path")
  file_path = st.text_input(
    "File path",
    placeholder="/path/to/your/data.h5ad",
    help="Enter the full path to your data file"
  )

  col1, col2 = st.columns(2)
  with col1:
    load_uploaded = st.button("Load Uploaded File", disabled=uploaded_file is None)
  with col2:
    load_from_path = st.button("Load from Path", disabled=not file_path)

  # Handle loading
  if load_uploaded and uploaded_file is not None:
    _load_data(uploaded_file, is_uploaded=True)

  if load_from_path and file_path:
    _load_data(file_path, is_uploaded=False)


def _render_reference_datasets_tab():
  """Render the reference datasets tab."""
  st.subheader("Reference Datasets")
  st.markdown("""
  Load standard scRNA-seq benchmark datasets directly from **scanpy.datasets**.
  These datasets are commonly used in the literature for method evaluation.
  """)
  
  available_datasets = get_available_datasets()
  
  # Display dataset cards
  for dataset_name, info in available_datasets.items():
    with st.container():
      col1, col2 = st.columns([3, 1])
      
      with col1:
        st.markdown(f"### {info.display_name}")
        st.markdown(info.description)
        st.caption(f"**{info.n_cells:,}** cells **{info.n_genes:,}** genes **{info.n_clusters}** clusters")
        st.caption(f"{info.citation}")
      
      with col2:
        if st.button(f"Load", key=f"load_{dataset_name}", width="stretch"):
          _load_reference_dataset(dataset_name)
      
      st.divider()


def _load_reference_dataset(dataset_name: str):
  """Load a reference dataset."""
  try:
    with st.spinner(f"Downloading and processing {dataset_name}... This may take a moment."):
      adata, labels, info = load_reference_dataset(dataset_name)
      
      # Create DataHandler and set data
      handler = DataHandler()
      handler.adata = adata
      handler.adata_original = adata.copy() # Match DataHandler attribute name
      handler.labels = labels
      handler.label_key = info.label_key
      
      # Store label mapping if available
      if labels is not None:
        unique_labels = np.unique(labels)
        handler.adata.uns['label_map'] = {str(l): i for i, l in enumerate(unique_labels)}
      
      set_with_cascade(st.session_state, KEYS.DATA_HANDLER, handler)
      st.session_state[KEYS.DATA_LOADED] = True
      st.session_state[KEYS.DATA_PREPROCESSED] = False
      st.session_state.loaded_dataset_name = info.display_name
      # Store reference dataset name for CLI (user needs to download it separately)
      st.session_state.uploaded_file_path = f"# Reference dataset: {info.display_name}"
      st.session_state.uploaded_file_name = info.display_name

      st.success(f"**{info.display_name}** loaded successfully!")
      st.rerun()

  except Exception as e:
    display_error(
      e,
      user_message="Failed to load the reference dataset.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="data_upload_reference_load",
    )


def _load_data(file_source, is_uploaded: bool = True):
  """Load data from file source."""
  try:
    with st.spinner("Loading data..."):
      handler = DataHandler()

      if is_uploaded:
        # Save uploaded file temporarily
        import tempfile
        suffix = Path(file_source.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
          tmp.write(file_source.read())
          tmp_path = tmp.name
        handler.load(tmp_path)
        # Store the original filename for CLI command generation
        st.session_state.uploaded_file_path = file_source.name
        st.session_state.uploaded_file_name = file_source.name
        # Clean up
        Path(tmp_path).unlink()
      else:
        handler.load(file_source)
        # Store the file path for CLI command generation
        st.session_state.uploaded_file_path = str(file_source)
        st.session_state.uploaded_file_name = Path(file_source).name

      set_with_cascade(st.session_state, KEYS.DATA_HANDLER, handler)
      st.session_state[KEYS.DATA_LOADED] = True
      st.session_state[KEYS.DATA_PREPROCESSED] = False

      st.success(f"Data loaded successfully!")
      st.rerun()

  except Exception as e:
    display_error(
      e,
      user_message="Failed to load data. Please check file format and content.",
      show_traceback=True,
      show_retry=False,
      show_reset=False,
      key_prefix="data_upload_file_load",
    )


def _display_data_info():
  """Display information about loaded data and allow benchmark configuration."""
  handler = st.session_state.data_handler
  info = handler.get_info()
  adata = handler.get_data()

  # -------------------------------------------------------------------------
  # 1. KPI Dashboard (Enhanced for Biologists)
  # -------------------------------------------------------------------------
  st.subheader("Dataset Overview")
  
  # Pre-calculations
  n_cells = adata.n_obs
  n_genes = adata.n_vars
  
  if adata is not None and adata.X is not None:
    X = adata.X
    # Handle sparse matrices
    is_sparse = sparse.issparse(X)
    
    # 1. Sparsity
    if is_sparse:
      sparsity = 1.0 - (X.nnz / (n_cells * n_genes))
      total_counts = X.sum()
      mean_counts = total_counts / n_cells
      # Median genes per cell (nnz per row)
      genes_per_cell = X.getnnz(axis=1)
      median_genes = np.median(genes_per_cell)
      # Data type check (Raw vs Norm)
      sample_data = X.data[:100] if X.nnz > 0 else []
      is_integers = np.all(np.mod(sample_data, 1) == 0)
      max_val = X.max()
    else:
      # Ensure numpy array for dense data
      if hasattr(X, 'to_numpy'):
        X = X.to_numpy()
      else:
        X = np.asarray(X)
        
      sparsity = 1.0 - (np.count_nonzero(X) / X.size)
      total_counts = np.sum(X)
      mean_counts = total_counts / n_cells
      # Median genes per cell
      genes_per_cell = np.count_nonzero(X, axis=1)
      median_genes = np.median(genes_per_cell)
      # Data type check
      sample_data = X.flatten()[:100]
      is_integers = np.all(np.mod(sample_data, 1) == 0)
      max_val = np.max(X)
  else:
    sparsity = 0.0
    mean_counts = 0
    median_genes = 0
    is_integers = True
    max_val = 0

  # Determine Data Status
  if is_integers:
    data_status = "Raw Counts (Integers)"
    status_color = "off" # grey
  elif max_val < 20: # Heuristic for log-transformed data
    data_status = "Log-Normalized"
    status_color = "normal" 
  else:
    data_status = "Normalized (Linear)"
    status_color = "off"

  # Row 1: Basic Dimensions
  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Cells", f"{n_cells:,}")
  with c2:
    st.metric("Genes", f"{n_genes:,}")
  with c3:
    n_batches = len(adata.obs[get_batch_column(adata)].unique()) if get_batch_column(adata) else 1
    st.metric("Batches", n_batches)
  with c4:
    st.metric("Data Type", data_status, help=f"Max value: {max_val:.2f}")

  # Row 2: Biological Depth
  c1, c2, c3, c4 = st.columns(4)
  with c1:
    st.metric("Median Genes/Cell", f"{int(median_genes):,}", help="Complexity: Median number of genes detected per cell")
  with c2:
    st.metric("Mean Reads/Cell", f"{int(mean_counts):,}", help="Depth: Average number of UMIs/reads per cell")
  with c3:
    st.metric("Sparsity", f"{sparsity*100:.1f}%", help="Percentage of zeros in the matrix")
  with c4:
    # Check for embeddings
    n_embeddings = len(adata.obsm.keys()) if adata.obsm else 0
    st.metric("Embeddings", n_embeddings, help=f"Keys: {', '.join(adata.obsm.keys()) if adata.obsm else 'None'}")

  # -------------------------------------------------------------------------
  # Batch Correction at Import (scVI-DCT)
  # -------------------------------------------------------------------------
  st.markdown("---")
  st.subheader("Batch Correction (scVI-DCT / sysVI)")

  scvi_available = check_scvi_available()
  if not scvi_available:
    st.warning("scVI not installed. Batch correction requires: `pip install scvi-tools`")
  else:
    if info.get('batch_correction_applied', False):
      st.success("Batch correction already applied at import.")
      bc_info = info.get('batch_correction_info', {})
      if bc_info:
        st.caption(f"Batch key: {bc_info.get('batch_key', 'n/a')}")
    else:
      if not is_integers:
        st.warning(
          "This data does not look like raw counts. scVI expects raw counts "
          "(integers). Batch correction with scVI may be unreliable. "
          "sysVI works on normalized+log-transformed data."
        )

      enable_bc = st.checkbox(
        "Apply batch correction now (raw counts)",
        value=False,
        help="Runs batch correction (scVI/sysVI) before any preprocessing."
      )

      if enable_bc:
        st.warning(
          "**BENCHMARK MODE WARNING**: Applying batch correction here uses the **ENTIRE** dataset. "
          "If you plan to use **Benchmark Mode** (which strictly isolates the Test set), "
          "you should **NOT** enable this here. Instead, apply batch correction in the **Preprocessing** step "
          "after the Train/Test split is configured to prevent data leakage."
        )

        detected_batch_col = get_batch_column(adata)
        batch_column_options = []
        if detected_batch_col and detected_batch_col in adata.obs.columns:
          batch_column_options.append(detected_batch_col)
        for col in ['batch', 'Batch', 'dataset', 'dataset_source', 'study', 'sample', 'donor', 'platform', 'tech']:
          if col in adata.obs.columns and col not in batch_column_options:
            if adata.obs[col].nunique() > 1:
              batch_column_options.append(col)
        for col in adata.obs.columns:
          if col not in batch_column_options and adata.obs[col].dtype.name == 'category':
            n_cats = adata.obs[col].nunique()
            if 1 < n_cats <= 50:
              batch_column_options.append(col)

        if not batch_column_options:
          st.info("No batch column detected. Batch correction requires multiple batches.")
        else:
          scvi_mode = st.selectbox(
            "Batch Correction Method",
            options=['scvi', 'sysvi'],
            index=0,
            help="`scvi` uses label-aware DCT. `sysvi` uses cycle-consistency + VampPrior (Hrovatin et al. 2023)."
          )

          col1, col2 = st.columns(2)
          with col1:
            selected_batch_col = st.selectbox(
              "Batch Column",
              options=batch_column_options,
              index=0,
              help=f"Detected: {detected_batch_col}"
            )
          with col2:
            detected_label_col = None
            for col in ['cell_type', 'celltype', 'Group', 'labels', 'CellType', 'label']:
              if col in adata.obs.columns and adata.obs[col].nunique() > 1:
                detected_label_col = col
                break

            label_options = ['None (unsupervised)']
            if 'Group' in adata.obs.columns:
              label_options.append('Group')
            if 'labels' in adata.obs.columns:
              label_options.append('labels')
            for col in ['celltype', 'cell_type', 'CellType']:
              if col in adata.obs.columns and col not in label_options:
                label_options.append(col)

            default_label_option = 'None (unsupervised)'
            if scvi_mode == 'scvi' and detected_label_col in label_options:
              default_label_option = detected_label_col

            selected_labels = st.selectbox(
              "Cell Type Labels (optional)",
              options=label_options,
              index=label_options.index(default_label_option),
              help="For scvi (DCT), a real cell-type label column should be selected."
            )

          with st.expander("Advanced Settings", expanded=False):
            if scvi_mode == 'scvi':
              col1, col2, col3 = st.columns(3)
              with col1:
                n_latent = st.number_input(
                  "Latent Dimensions",
                  min_value=10,
                  max_value=100,
                  value=30,
                  step=5
                )
              with col2:
                max_epochs = st.number_input(
                  "Training Epochs",
                  max_value=500,
                  value=100,
                  step=25
                )
              with col3:
                batch_size = st.number_input(
                  "Batch Size",
                  min_value=32,
                  max_value=512,
                  value=128,
                  step=32
                )

              col4, col5 = st.columns(2)
              with col4:
                weight_label = "DCT Weight"
                dct_weight = st.number_input(
                  weight_label,
                  min_value=0.0,
                  max_value=10.0,
                  value=1.0,
                  step=0.1
                )
              with col5:
                corr_mse_weight = st.number_input(
                  "Corr MSE Weight",
                  min_value=0.0,
                  max_value=1.0,
                  value=0.1,
                  step=0.05
                )

            elif scvi_mode == 'sysvi':
              st.info(
                "**sysVI** (Hrovatin et al. 2023): cVAE with VampPrior and cycle-consistency loss. "
                "Designed for strong batch effects."
              )
              col1, col2, col3 = st.columns(3)
              with col1:
                n_latent = st.number_input(
                  "Latent Dimensions",
                  min_value=5,
                  max_value=100,
                  value=15,
                  step=5
                )
              with col2:
                max_epochs = st.number_input(
                  "Training Epochs",
                  max_value=500,
                  value=200,
                  step=25
                )
              with col3:
                batch_size = st.number_input(
                  "Batch Size",
                  min_value=32,
                  max_value=512,
                  value=128,
                  step=32
                )

              col4, col5 = st.columns(2)
              with col4:
                sysvi_cycle_weight = st.number_input(
                  "Cycle-Consistency Weight",
                  min_value=0.0,
                  max_value=50.0,
                  value=5.0,
                  step=1.0,
                  help="Increase for more batch correction (typically 2-10)."
                )
              with col5:
                sysvi_kl_weight = st.number_input(
                  "KL Weight",
                  min_value=0.0,
                  max_value=10.0,
                  value=1.0,
                  step=0.1,
                  help="Decrease for better biological preservation."
                )

              col6, col7 = st.columns(2)
              with col6:
                sysvi_prior = st.selectbox(
                  "Prior",
                  options=['vamp', 'standard_normal'],
                  index=0,
                  help="VampPrior recommended"
                )
              with col7:
                sysvi_n_prior_components = st.number_input(
                  "Prior Components",
                  min_value=1,
                  max_value=50,
                  value=5,
                  step=1,
                  help="Number of VampPrior pseudo-inputs"
                )
              sysvi_embed_covariates = st.checkbox(
                "Embed Categorical Covariates",
                value=True,
                help="Enables embeddings for additional categorical covariates."
              )

          if st.button("Run Batch Correction", type="primary"):
            params = {
              'batch_correction_method': scvi_mode,
              'batch_correction_batch_key': selected_batch_col,
              'batch_correction_labels_key': None if selected_labels == 'None (unsupervised)' else selected_labels,
              'batch_correction_n_latent': int(n_latent),
              'batch_correction_epochs': int(max_epochs),
              'batch_correction_batch_size': int(batch_size),
            }
            if scvi_mode == 'scvi':
              params.update({
                'batch_correction_dct_weight': float(dct_weight),
                'batch_correction_corr_mse_weight': float(corr_mse_weight),
              })
            if scvi_mode == 'sysvi':
              params.update({
                'batch_correction_cycle_weight': float(sysvi_cycle_weight),
                'batch_correction_kl_weight': float(sysvi_kl_weight),
                'batch_correction_prior': sysvi_prior,
                'batch_correction_n_prior_components': int(sysvi_n_prior_components),
                'batch_correction_embed_covariates': bool(sysvi_embed_covariates),
              })
            try:
              with st.spinner(f"Running {scvi_mode} batch correction..."):
                handler.apply_batch_correction(params)
              st.session_state.data_preprocessed = False
              if 'benchmark_processed' in st.session_state:
                del st.session_state['benchmark_processed']
              st.success("Batch correction applied at import.")
              st.rerun()
            except Exception as e:
              st.error(f"Batch correction failed: {e}")

  # -------------------------------------------------------------------------
  # Batch Exclusion Option (for RPKM batches like smarter/Xin)
  # -------------------------------------------------------------------------
  batch_col = get_batch_column(adata)
  if batch_col and batch_col in adata.obs.columns:
    batches_in_data = sorted(adata.obs[batch_col].unique().astype(str).tolist())
    if len(batches_in_data) > 1:
      st.markdown("---")
      st.subheader("Batch Selection")
      st.caption(
        "Exclude specific batches from analysis. Useful for removing RPKM-only batches "
        "(e.g., 'smarter'/Xin dataset) when raw counts are required for ZINB models."
      )
      
      # Detect smarter batch
      smarter_batches = [b for b in batches_in_data if 'smarter' in b.lower()]
      default_exclude = smarter_batches if smarter_batches else []
      
      excluded_batches = st.multiselect(
        "Exclude batches",
        options=batches_in_data,
        default=default_exclude,
        help="Select batches to exclude from the analysis. The 'smarter' batch contains RPKM values, not raw counts."
      )
      
      if excluded_batches:
        n_excluded_cells = adata.obs[adata.obs[batch_col].astype(str).isin(excluded_batches)].shape[0]
        st.info(f"⚠️ {n_excluded_cells} cells from {len(excluded_batches)} batch(es) will be excluded.")
        
        if st.button("Apply Batch Exclusion", type="secondary", key="apply_batch_exclusion"):
          try:
            mask = ~adata.obs[batch_col].astype(str).isin(excluded_batches)
            handler.adata = adata[mask].copy()
            handler.adata_original = handler.adata_original[handler.adata_original.obs[batch_col].astype(str).isin(adata.obs[batch_col].unique())].copy() if hasattr(handler, 'adata_original') else handler.adata.copy()
            st.session_state.data_preprocessed = False
            st.success(f"✓ Excluded {n_excluded_cells} cells. Remaining: {handler.adata.n_obs} cells.")
            st.rerun()
          except Exception as e:
            st.error(f"Error excluding batches: {e}")

  # -------------------------------------------------------------------------
  # 1.5 Biological Context (Top Genes & Embeddings)
  # -------------------------------------------------------------------------
  st.markdown("### Biological Context")

  
  bc1, bc2 = st.columns([1, 1])
  
  # Top Expressed Genes (Tissue Identity)
  with bc1:
    st.markdown("**Top 10 Highly Expressed Genes**")
    st.caption("Helps identify tissue type or dominant contamination.")
    
    try:
      # Efficiently sum columns
      if is_sparse:
        # Sum over axis 0 (genes)
        gene_sums = np.array(X.sum(axis=0)).flatten()
      else:
        gene_sums = np.sum(X, axis=0)
      
      # Create series
      gene_series = pd.Series(gene_sums, index=adata.var_names)
      top_genes = gene_series.sort_values(ascending=False).head(10)
      
      # Normalize for percentage
      top_genes_pct = (top_genes / total_counts) * 100
      
      # Plot horizontal bar
      fig_genes, ax_genes = plt.subplots(figsize=(6, 4))
      sns.barplot(x=top_genes_pct.values, y=top_genes_pct.index, palette="viridis", ax=ax_genes)
      ax_genes.set_xlabel("% of Total Counts")
      ax_genes.set_title("Highest Expression (Identity Check)")
      st.pyplot(fig_genes)
      plt.close()

      with st.expander("Calculation Details"):
        st.markdown("""
        **Method:** Raw sum of counts (reads or UMIs) for each gene divided by total matrix counts.
        
        *  **UMI** = Unique Molecular Identifier, a barcode used to count unique transcripts and reduce PCR bias.
        *  Shows the share of sequencing "bandwidth" captured by these genes.
        *  Excessive dominance (>5-10%) of a single gene (e.g. MALAT1, MT-*) can indicate poor quality or cellular stress.
        """)
      
    except Exception as e:
      st.info("Could not calculate top genes (memory or format issue).")

  # Existing Embeddings (Visual Structure)
  with bc2:
    st.markdown("**Existing Embeddings**")
    
    # Check for common keys
    emb_keys = [k for k in adata.obsm.keys() if 'umap' in k.lower() or 'tsne' in k.lower()]
    
    if emb_keys:
      selected_emb = st.selectbox("Show Embedding", options=emb_keys, key="upload_preview_emb")
      st.caption(f"Visualizing `{selected_emb}` from uploaded file.")
      
      coords = adata.obsm[selected_emb]
      if coords.shape[1] >= 2:
        fig_emb, ax_emb = plt.subplots(figsize=(5, 4))
        # Plot
        ax_emb.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.6, c='steelblue', edgecolors='none')
        ax_emb.set_xticks([])
        ax_emb.set_yticks([])
        ax_emb.set_title(f"Pre-computed {selected_emb}")
        # Remove box
        sns.despine(left=True, bottom=True)
        st.pyplot(fig_emb)
        plt.close()
      else:
        st.warning("Embedding has fewer than 2 dimensions.")
    else:
      st.info("No pre-computed UMAP or t-SNE found in this file.")
      st.caption("Embeddings will be generated in the Analysis step.")

  # -------------------------------------------------------------------------
  # 1.6 Gene Expression Analysis (NEW - Available at Data Upload)
  # -------------------------------------------------------------------------
  st.markdown("---")
  st.markdown("### Gene Expression Analysis")
  st.caption("Explore gene expression patterns in your dataset without running a clustering algorithm.")
  
  gene_tab1, gene_tab2, gene_tab3 = st.tabs(["By Batch", "By Cell Type", "Differential Expression"])
  
  with gene_tab1:
    _render_batch_gene_inspector_upload(adata)
  
  with gene_tab2:
    _render_celltype_gene_inspector_upload(adata)
  
  with gene_tab3:
    _render_differential_expression_upload(adata)

  # -------------------------------------------------------------------------
  # 1.7 Expression Matrix Preview
  # -------------------------------------------------------------------------
  st.markdown("### Expression Matrix Preview")
  
  with st.expander("View Gene x Cell Count Matrix", expanded=False):
    # Subset for visualization (first 50 cells, top 50 variable genes)
    n_view = 50
    
    # Select top genes by mean expression if no HVG info
    if sparse.issparse(X):
      mean_expr = np.array(X.mean(axis=0)).flatten()
    else:
      mean_expr = np.mean(X, axis=0)
      
    top_indices = np.argsort(mean_expr)[::-1][:n_view]
    cell_indices = np.arange(min(n_cells, n_view))
    
    # Extract subset
    if sparse.issparse(X):
      subset = X[cell_indices, :][:, top_indices].toarray()
    else:
      subset = X[cell_indices, :][:, top_indices]
      
    # Plot
    fig_mat, ax_mat = plt.subplots(figsize=(10, 6))
    sns.heatmap(subset, cmap="viridis", xticklabels=adata.var_names[top_indices], 
          yticklabels=False, ax=ax_mat, cbar_kws={'label': 'Expression'})
    ax_mat.set_xlabel(f"Top {n_view} Expressed Genes")
    ax_mat.set_ylabel(f"First {n_view} Cells")
    ax_mat.set_title("Expression Heatmap (Subset)")
    st.pyplot(fig_mat)
    plt.close()

  # -------------------------------------------------------------------------
  # 1.7 Label / Cell Type Information
  # -------------------------------------------------------------------------
  st.markdown("### Cell Type Labels")
  
  # Try to detect label column from handler or common names
  label_key = getattr(handler, 'label_key', None)
  if not label_key:
    potential_keys = ['Group', 'labels', 'cell_type', 'celltype', 'Cluster', 'Sub_Cluster']
    for key in potential_keys:
      if key in adata.obs.columns:
        label_key = key
        break
  
  if label_key and label_key in adata.obs:
    labels = adata.obs[label_key]
    n_labels = labels.nunique()
    
    lc1, lc2 = st.columns([1, 2])
    
    with lc1:
      st.metric("Unique Labels", n_labels)
      st.caption(f"Detected column: `{label_key}`")
      if n_labels > 50:
        st.warning("High number of labels detected!")
      
    with lc2:
      # Bar chart of label distribution
      # Limit to top 20 if too many
      label_counts = labels.value_counts()
      if n_labels > 20:
        original_n = n_labels
        label_counts = label_counts.head(20)
        st.caption(f"Showing top 20 of {original_n} labels")
      
      fig_labels, ax_labels = plt.subplots(figsize=(8, max(3, len(label_counts)*0.3)))
      sns.barplot(x=label_counts.values, y=label_counts.index.astype(str), palette="pastel", ax=ax_labels)
      ax_labels.set_xlabel("Number of Cells")
      ax_labels.set_ylabel("")
      ax_labels.set_title("Label Distribution", fontsize=10)
      sns.despine()
      st.pyplot(fig_labels)
      plt.close()
    
    # Label Harmonization Section
    _render_label_harmonization(adata, label_key, handler)
    
  else:
    st.info("No obvious label column detected (checked: Group, labels, cell_type, etc).")
    st.caption("The analysis will proceed in unsupervised mode.")

  # -------------------------------------------------------------------------
  # 2. Metadata Explorer
  # -------------------------------------------------------------------------
  with st.expander("Explore Metadata (obs)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
      st.markdown("**Metadata (obs - Cells)**")
      st.write(f"Count: {adata.n_obs} cells")
      st.write(f"Columns: {', '.join(adata.obs.columns.tolist())}")
      
      if adata.obsm:
        st.markdown("**Embeddings (obsm)**")
        st.write(f"Keys: {', '.join(adata.obsm.keys())}")
    
    with c2:
      st.markdown("**Features (var - Genes)**")
      st.write(f"Count: {adata.n_vars} genes")
      if len(adata.var.columns) > 0:
        st.write(f"Columns: {', '.join(adata.var.columns.tolist())}")
      
      if adata.layers:
        st.markdown("**Alternative Matrices (layers)**")
        st.write(f"Keys: {', '.join(adata.layers.keys())}")
    
    if adata.uns:
      st.markdown("**Global Metadata (uns)**")
      # Filter some long/internal keys for better display
      uns_keys = [k for k in adata.uns.keys()]
      st.write(f"Keys: {', '.join(uns_keys)}")

  # -------------------------------------------------------------------------
  # 3. Batch Information
  # -------------------------------------------------------------------------
  _display_batch_info(handler)

  # -------------------------------------------------------------------------
  # 4. Quality Control
  # -------------------------------------------------------------------------
  st.markdown("---")
  with st.expander("Quality Control (on-demand)", expanded=False):
    qc_cache_key = f"upload_qc_metrics_{adata.n_obs}_{adata.n_vars}"
    if qc_cache_key not in st.session_state:
      st.info("Click the button to compute QC metrics.")
    if st.button("Compute QC metrics", key=f"{qc_cache_key}_btn", type="secondary"):
      with st.spinner("Computing quality metrics..."):
        st.session_state[qc_cache_key] = _compute_qc_metrics(adata)

    qc_metrics = st.session_state.get(qc_cache_key)
    if qc_metrics is not None:
      _render_qc_violin_plots(qc_metrics)
      _render_cells_per_gene_distribution(qc_metrics)

  # Warning if X is None
  if not info.get('has_X', True):
    st.error("Data matrix (adata.X) is empty! The data cannot be processed.")
    if info.get('has_raw'):
      st.info("Data found in adata.raw - it will be used automatically.")
    if info.get('layers'):
      st.info(f"Available layers: {', '.join(info['layers'])}")

  # Clear data button (at the end of the page)
  st.markdown("---")
  if st.button("Clear Data"):
    invalidate(st.session_state, KEYS.DATA_HANDLER, recursive=True)
    st.rerun()
