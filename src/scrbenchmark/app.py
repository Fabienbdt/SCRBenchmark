"""
scDeepCluster Analysis Suite - Main Streamlit Application

A graphical interface for single-cell RNA-seq clustering analysis.
Supports multiple algorithms including scDeepCluster, scCDCG, scMAE, scNAME,
and PCA followed by K-Means, Louvain, Leiden, or HDBSCAN.

Run with: streamlit run src/scrbenchmark/app.py
"""

import os
import tempfile
import streamlit as st
from pathlib import Path
import sys

# Runtime safety for Numba-based libraries (UMAP/pynndescent/scIB) in Streamlit.
# Keep defaults overrideable, but avoid parallel thread-layer crashes in local runs.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
_numba_cache_dir = Path(tempfile.gettempdir()) / "scrbenchmark_numba_cache"
_numba_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache_dir))

# Matplotlib also wants a writable cache directory in managed environments.
_mpl_cache_dir = Path(tempfile.gettempdir()) / "scrbenchmark_mpl_cache"
_mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache_dir))

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent)) # For accessing src package

# Import pages
from gui.data_upload import render_data_upload_page
from gui.data_split import render_data_split_page
from gui.preprocessing import render_preprocessing_page
from gui.algorithm_config import render_algorithm_config_page
from gui.analysis import render_analysis_page
from gui.latent_reclustering import render_latent_reclustering_page
from gui.hyperparam_search import render_hyperparam_search_page
from gui.customize_benchmark import render_customize_benchmark_page
from gui.report_reproduction import render_report_reproduction_page
from gui.results_explorer import render_results_explorer_page
from gui.documentation import render_documentation_page
from gui.i18n import t
from gui.state_manager import KEYS, compute_workflow_progress, init_session_defaults

# Import algorithms to register them
import algorithms # noqa: F401


def main():
  """Main application entry point."""

  # Page configuration
  st.set_page_config(
    page_title="SCRBenchmark Platform",
    layout="wide",
    initial_sidebar_state="expanded"
  )

  # Custom CSS for a professional look
  st.markdown("""
    <style>
      /* Reduce top padding */
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      
      /* Professional Header */
      h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
      }
      .subtitle {
        font-size: 1.2rem;
        color: #7f8c8d;
        font-weight: 300;
        margin-bottom: 2rem;
      }
      
      /* Metric Cards styling */
      div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
      }
      
      /* Alert/Info box styling */
      .stAlert {
        padding: 0.75rem;
        border-radius: 0.5rem;
      }
      
      /* Sidebar refinements */
      section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
      }
    </style>
  """, unsafe_allow_html=True)

  # Initialize session state
  _init_session_state()

  # Sidebar
  _render_sidebar()

  # Main content
  _render_main_content()


def _init_session_state():
  """Initialize session state variables."""
  init_session_defaults(st.session_state)


def _is_page_disabled(page_name: str) -> bool:
  if page_name == "Data Split" and not st.session_state.get(KEYS.DATA_LOADED, False):
    return True
  if page_name == "Preprocessing" and not st.session_state.get(KEYS.BENCHMARK_CONFIGURED, False):
    return True
  if page_name == "Algorithm Config" and not st.session_state.get(KEYS.DATA_PREPROCESSED, False):
    return True
  if page_name == "Analysis" and not st.session_state.get(KEYS.SELECTED_ALGORITHMS, []):
    return True
  if page_name == "Hyperparam Search" and not st.session_state.get(KEYS.DATA_PREPROCESSED, False):
    return True
  return False


def _render_sidebar():
  """Render the sidebar navigation."""
  with st.sidebar:
    toggle_widget = getattr(st, "toggle", None)
    if callable(toggle_widget):
      is_quick_mode = toggle_widget(
        t("ui.mode.quick"),
        value=st.session_state.get(KEYS.UI_MODE, "quick") == "quick",
        help="Switch to advanced mode to show expert options.",
        key="global_quick_mode_toggle",
      )
    else:
      is_quick_mode = st.checkbox(
        t("ui.mode.quick"),
        value=st.session_state.get(KEYS.UI_MODE, "quick") == "quick",
        help="Switch to advanced mode to show expert options.",
        key="global_quick_mode_toggle",
      )
    st.session_state[KEYS.UI_MODE] = "quick" if is_quick_mode else "advanced"
    st.caption(f"Active mode: {t('ui.mode.quick') if is_quick_mode else t('ui.mode.advanced')}")

    progress = compute_workflow_progress(st.session_state)
    completed = progress["completed"]
    total = progress["total"]
    try:
      st.progress(completed / max(1, total), text=f"Progression Workflow: {completed}/{total}")
    except TypeError:
      st.progress(completed / max(1, total))
      st.caption(f"Progression Workflow: {completed}/{total}")

    workflow_pages = [
      ("Step 1: Data Upload", "Data Upload"),
      ("Step 2: Data Split", "Data Split"),
      ("Step 3: Preprocessing", "Preprocessing"),
      ("Step 4: Algorithm Config", "Algorithm Config"),
      ("Step 5: Run Analysis", "Analysis"),
    ]
    tool_pages = [
      ("Latent Re-clustering", "Latent Re-clustering"),
      ("Customize Benchmark", "Customize Benchmark"),
      ("Report Reproduction", "Report Reproduction"),
      ("Results Explorer", "Results Explorer"),
      ("Hyperparam Search", "Hyperparam Search"),
      ("Documentation", "Documentation"),
    ]

    st.markdown(f"### {t('sidebar.workflow')}")
    for idx, (label, page_name) in enumerate(workflow_pages, start=1):
      disabled = _is_page_disabled(page_name)
      is_active = st.session_state.get(KEYS.CURRENT_PAGE) == page_name
      marker = "● " if is_active else ""
      done_icon = "✓ " if progress["steps"][idx - 1][1] else ""
      btn_label = f"{marker}{done_icon}{label}"
      if st.button(
        btn_label,
        key=f"nav_workflow_{page_name}",
        width="stretch",
        disabled=disabled,
      ):
        st.session_state[KEYS.CURRENT_PAGE] = page_name

    st.markdown(f"### {t('sidebar.tools')}")
    for label, page_name in tool_pages:
      disabled = _is_page_disabled(page_name)
      is_active = st.session_state.get(KEYS.CURRENT_PAGE) == page_name
      marker = "● " if is_active else ""
      if st.button(
        f"{marker}{label}",
        key=f"nav_tool_{page_name}",
        width="stretch",
        disabled=disabled,
      ):
        st.session_state[KEYS.CURRENT_PAGE] = page_name

    st.markdown("---")

    # Status indicators
    st.markdown(f"### {t('sidebar.status')}")

    # Data loaded
    if st.session_state.get(KEYS.DATA_LOADED, False):
      st.success("Data loaded")
      info = st.session_state[KEYS.DATA_HANDLER].get_info()
      st.caption(f"{info['n_cells']:,} cells, {info['n_genes']:,} genes")
    else:
      st.info("Data not loaded")

    # Preprocessed
    if st.session_state.get(KEYS.DATA_PREPROCESSED, False):
      st.success("Data preprocessed")
    else:
      st.info("Not preprocessed")

    # Algorithms selected
    n_algos = len(st.session_state.get(KEYS.SELECTED_ALGORITHMS, []))
    if n_algos > 0:
      st.success(f"{n_algos} algorithm(s) selected")
    else:
      st.info("No algorithms selected")

    # Results available
    if (
      st.session_state.get(KEYS.ANALYSIS_RESULTS) is not None
      or st.session_state.get(KEYS.BENCHMARK_RESULTS) is not None
    ):
      st.success("Results available")

    st.markdown("---")

    # About
    with st.expander("About"):
      st.markdown("""
      **scDeepCluster Analysis Suite v1.1**

      A graphical interface for single-cell RNA-seq
      clustering analysis.

      **Features:**
      - Multiple clustering algorithms
      - Statistical comparison (CLD)
      - Hyperparameter optimization
      - UMAP visualization

      **Supported Formats:**
      - H5AD (AnnData)
      - H5 (HDF5)
      - CSV/TSV
      - MTX (10X Genomics)

      [Documentation](https://github.com/your-repo)
      """)


def _render_main_content():
  """Render the main content area."""

  # Render current page
  current_page = st.session_state[KEYS.CURRENT_PAGE]
  st.caption(f"Navigation: Workflow > {current_page}")

  # Header (only on main pages or if needed, but here we can keep it cleaner)
  # Most pages render their own title, but we can add a global breadcrumb or similar if desired.
  # For now, we just ensure the landing page logic is clean.
  
  if current_page == "Data Upload":
    st.markdown("# SCRBenchmark Platform")
    st.markdown('<div class="subtitle">Single-Cell RNA-seq Clustering Benchmark Suite</div>', unsafe_allow_html=True)
    render_data_upload_page()

  elif current_page == "Data Split":
    st.markdown("# Data Split Configuration")
    render_data_split_page()

  elif current_page == "Preprocessing":
    st.markdown("# Preprocessing Pipeline")
    render_preprocessing_page()

  elif current_page == "Algorithm Config":
    st.markdown("# Algorithm Configuration")
    render_algorithm_config_page()

  elif current_page == "Analysis":
    st.markdown("# Benchmark Analysis")
    render_analysis_page()

  elif current_page == "Latent Re-clustering":
    render_latent_reclustering_page()

  elif current_page == "Customize Benchmark":
    render_customize_benchmark_page()

  elif current_page == "Report Reproduction":
    render_report_reproduction_page()

  elif current_page == "Results Explorer":
    render_results_explorer_page()

  elif current_page == "Hyperparam Search":
    st.markdown("# Hyperparameter Optimization")
    render_hyperparam_search_page()

  elif current_page == "Documentation":
    render_documentation_page()

  # Footer
  st.markdown("---")
  col1, col2 = st.columns([1, 1])
  with col1:
    st.caption("SCRBenchmark v1.1 | Developed for M2 Internship")
  with col2:
    st.caption("Based on: scDeepCluster, scVI, scMAE, scNAME, scCDCG")

if __name__ == "__main__":
  main()
