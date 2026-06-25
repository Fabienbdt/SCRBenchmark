"""
Documentation page for the scDeepCluster Analysis Suite.
"""

import streamlit as st

def render_documentation_page():
  """Render the documentation and help page."""
  
  st.title("Documentation & User Guide")
  
  st.markdown("""
  Welcome to the **scDeepCluster Analysis Suite** documentation!
  
  This guide provides detailed information about the algorithms, parameters, and workflows 
  integrated into this platform. Whether you are looking to understand how a specific 
  algorithm works, how to adjust parameters for your dataset, or interpret the results, 
  you will find the answers here.
  """)
  
  # Table of Contents using Streamlit tabs
  tab_overview, tab_workflow, tab_preprocessing, tab_algorithms, tab_report, tab_tuning = st.tabs([
    "Overview", "Workflow", "Preprocessing", "Algorithms", "Report Reproduction", "Parameter Tuning"
  ])
  
  # =========================================================================
  # 1. OVERVIEW
  # =========================================================================
  with tab_overview:
    st.header("1. Overview")
    st.markdown("""
    The **scDeepCluster Analysis Suite** is a unified platform for benchmarking and running 
    state-of-the-art deep learning clustering algorithms for single-cell RNA-seq (scRNA-seq) data.
    
    ### Key Features
    - **Multi-Algorithm Support**: Run and compare classic methods (PCA+Leiden) and advanced deep learning methods (scDeepCluster, scCDCG, scMAE, scNAME).
    - **Unified Interface**: Use a consistent interface for configuration, execution, and visualization.
    - **Robust Preprocessing**: Standardized preprocessing pipeline including cell/gene filtering, normalization, and HVG selection.
    - **Robustness Testing**: Built-in tools to simulate data dropouts and noise to test algorithm stability.
    
    ### Supported Algorithms
    | Algorithm | Type | Description |
    |-----------|------|-------------|
    | **scDeepCluster** | Deep Learning | Combines autoencoder feature learning with DEC (Deep Embedded Clustering). |
    | **scCDCG** | Deep Learning | Deep Cut-informed Graph Embedding. Uses attention and optimal transport. |
    | **scMAE** | Deep Learning | Masked Autoencoder for robust feature learning from sparse data. |
    | **scNAME** | Deep Learning | Neighborhood contrastive clustering with mask estimation for denoising. |
    | **PCA + clustering** | Baseline | PCA followed by K-Means, Louvain, Leiden, or HDBSCAN. |
    
    ### New Features
    - **GPU/CPU Control**: Select your compute device (Auto/CPU/GPU) in Algorithm Configuration.
    - **Adaptive Target Sum**: Normalization target automatically suggested based on dataset size.
    - **Report Reproduction**: Rerun internship-report experiments from guided launchers instead of copying long CLI commands.
    - **+Harmony Variants**: Add post-hoc Harmony correction runs from the Algorithms tab when a registered method supports it.
    """)
    
  # =========================================================================
  # 2. WORKFLOW GUIDE
  # =========================================================================
  with tab_workflow:
    st.header("2. Workflow Guide")
    st.markdown("""
    ### Which page should I use?

    | Page | Main purpose |
    |------|--------------|
    | **Customize Benchmark** | Build a new experiment: choose data, split, preprocessing, algorithms, hyperparameters, and optional report-method reruns. |
    | **Report Reproduction** | Reproduce the internship-report experiments with predefined, rerunnable launchers. |
    | **Results Explorer** | Inspect output folders, metrics, logs, and generated artifacts after a run. |
    | **Latent Re-clustering** | Recompute clustering from existing embeddings without retraining the whole model. |
    | **Documentation** | Understand the interface, parameters, and reproduction workflow. |

    **Quick Start** keeps the interface compact and guides the standard benchmark path. Disable it when you need advanced configuration panels, manual protocol generation, or more detailed customization.

    Follow these steps to perform a complete analysis:
    
    **1. Data Upload**
    - Upload your scRNA-seq data.
    - Supported formats: `.h5ad` (AnnData), `.h5` (10x), `.csv`/`.tsv` (Count matrix), `.mtx` (Matrix Market).
    - **Tip**: Detailed file info will be shown after upload.
    
    **2. Preprocessing & Experimental Design**
    - **Experimental Design (New):** Define your evaluation protocol (All dataset vs Split dataset).
     - **Stratified Split (I.I.D.):** Ensures balanced cell types in Train/Test. Best for evaluating model capacity.
     - **Split by Batch (O.O.D.):** Tests robustness to unseen batches.
     - **Split Inspector:** Visually verify that your Train/Test sets are balanced before proceeding.
    - **Data Processing:**
     - Filter low-quality cells and genes.
     - Normalize library sizes (Adaptive Target Sum).
     - Select Highly Variable Genes (HVGs) - *Strategies available to prevent Data Leakage*.
     - **Optional**: Batch Correction (scVI-DCT) or Robustness testing (Dropout).
    
    **3. Algorithm Configuration**
    - Select one or more algorithms to run.
    - Tweak hyperparameters (number of clusters, latent dimension, epochs).
    - To compare a method with batch correction on its final representation, enable **Also evaluate + Harmony variants when available** in the Algorithms tab.
    - **Note**: Deep learning methods often require GPU for reasonable speed, though CPU is supported.
    
    **4. Analysis (Execution)**
    - Launch the training process.
    - Watch real-time logs for progress.
    - Models are trained sequentially.
    
    **5. Results & Visualization**
    - View dimensionality reduction plots (UMAP/t-SNE).
    - Compare clustering metrics (ARI, NMI, Accuracy).
    - Download results and models.
    """)

  # =========================================================================
  # 3. PREPROCESSING
  # =========================================================================
  with tab_preprocessing:
    st.header("3. Preprocessing Details")
    st.markdown("""
    Proper preprocessing is critical for scRNA-seq analysis.
    """)
    
    with st.expander("Cell & Gene Filtering"):
      st.markdown("""
      - **Min genes per cell**: Removes cells with poor library complexity (likely dead cells or empty droplets). Default: 200.
      - **Max genes per cell**: Removes potential doublets (two cells captured together). Default: varies (e.g., 20,000).
      - **Min cells per gene**: Removes genes that are barely expressed and provide little statistical power. Default: 3.
      """)
      
    with st.expander("Normalization & Log Transformation"):
      st.markdown("""
      - **Target Sum**: Library size normalization steps scale counts so that every cell has the same total count (e.g., 10,000). This corrects for technical differences in sequencing depth.
       - **Adaptive Suggestion**: The application now automatically suggests a target sum based on dataset size:
        - Small datasets (<5,000 cells): 10,000 (standard Scanpy default)
        - Medium datasets (5,000-50,000 cells): 10,000 to 100,000 (interpolated)
        - Large datasets (>50,000 cells): 100,000 (standard for large-scale workflows)
      - **Log1p**: Computes `log(1 + x)`. This stabilizes variance and makes the data distribution more normal-like, which is assumed by many algorithms.
      """)
      
    with st.expander("Highly Variable Genes (HVG)"):
      st.markdown("""
      - **Why?** scRNA-seq datasets have ~20,000 genes, but most don't vary between cell types. Selecting the top ~2,000-5,000 variable genes reduces noise and computational cost.
      - **Relevance**: All deep learning models in this suite perform better on HVGs than on the full transcriptome.
      """)

    with st.expander("Batch Correction (scVI-DCT & DCT-Corr)"):
      st.markdown("""
      - **Method**: The platform uses an enhanced version of **scVI** (Single-cell Variational Inference) incorporating **DCT-Corr**.
      - **Components**:
       - **DCT (Domain Class Triplet Loss)**: A loss function that actively minimizes the distance between cells of the same type across different batches, enforcing better mixing.
       - **Corr (Correlation Regularization)**: Preserves the biological gene-gene correlation structure during the integration process.
      - **Reference**: *Benchmarking deep learning methods for biologically conserved single-cell integration, Chenxin Yi et al. 2025* (Benchmark of deep learning batch correction).
      - **Usage**: Enable this in the Preprocessing tab if your dataset is composed of multiple sources (batches) to prevent the algorithms from clustering by batch instead of biology.
      """)

    with st.expander("Preprocessing correction vs. +Harmony variants"):
      st.markdown("""
      These two options answer different questions:

      | Option | Where to configure it | What it changes | When to use it |
      |--------|-----------------------|-----------------|----------------|
      | **Preprocessing batch correction** | Customize Benchmark > Preprocessing | Corrects the input representation before the selected algorithms are fitted. | Use it when every downstream method should start from a corrected dataset. |
      | **+Harmony variant** | Customize Benchmark > Algorithms | Adds an extra run such as `scMAE+Harmony` or `scNAME+Harmony`. For deep learning methods, the model is trained first, Harmony is then applied on the learned latent space, and final clustering is recomputed. | Use it when you want to compare a method against the same method with post-hoc latent batch correction. |

      In short: preprocessing correction changes the data entering the model; `+Harmony` variants evaluate an additional corrected version of a method output.
      """)

    st.subheader("Dropout Simulation & Robustness")
    st.info("Relevance: Single-cell data is notoriously sparse ('dropout' events). Use this feature to test if an algorithm can still identify cell types when data quality degrades.")
    
    st.markdown("""
    You can artificially add dropout to your data to simulate lower sequencing depth or lower capture efficiency.
    
    - **Simple Dropout**: Randomly sets non-zero entries to zero with probability `p`.
    - **Logistic Dropout**: Probability of dropout depends on expression level (lower expression = higher dropout probability). This is more biologically realistic.
      - **Midpoint**: The expression value where dropout probability is 50%.
      - **Shape**: Controls how sharply the dropout probability changes with expression.
    """)
    
  # =========================================================================
  # 4. ALGORITHMS REFERENCE
  # =========================================================================
  with tab_algorithms:
    st.header("4. Algorithm Reference")
    
    # --- scDeepCluster ---
    with st.expander("scDeepCluster"):
      st.markdown("""
      **Type**: Deep Learning (autoencoder + DEC) 
      **Reference**: Tian et al. (Nature Machine Intelligence, 2019)
      
      **How it works**:
      1. **Pretraining**: A standard ZINB (Zero-Inflated Negative Binomial) autoencoder learns a low-dimensional representation of the data that captures the count distribution.
      2. **Clustering**: The standard DEC (Deep Embedded Clustering) loss is added. This iteratively refines the representation to push cells towards cluster centers.
      
      **Key Parameters**:
      - **Embedding Dimension**: Size of the latent space (usually 32 or 64). Smaller = more compression.
      - **Gamma**: Weight of the clustering loss. Higher = forced tighter clusters (risk of distortion).
      - **Sigma**: Updates the soft cluster assignment.
      
      **When to use**: Good all-rounder for scRNA-seq. The ZINB loss specifically handles the sparsity and overdispersion of count data.
      """)

    # --- scCDCG ---
    with st.expander("scCDCG (Deep Cut-informed Graph Embedding)"):
      st.markdown("""
      **Type**: Deep Learning (Graph + Transport) 
      **Reference**: DASFAA 2024
      
      **How it works**:
      It goes beyond simple autoencoders by incorporating structural information:
      - **Graph Regularization**: Preserves local neighborhood structure in the latent space.
      - **Optimal Transport (Sinkhorn)**: Uses the Sinkhorn-Knopp algorithm to enforce a balanced cluster assignment, preventing the "trivial solution" where all cells go to one cluster.
      
      **Relevance**:
      Very effective when cell types are not well separated in simple PCA space but form a manifold structure.
      
      **Key Adjustments**:
      - **Balancer**: Controls weight between self-loop graph and feature graph.
      - **Lambdas (Sinkhorn)**: Temperature for the optimal transport. Lower values = softer assignment.
      """)

    # --- scMAE ---
    with st.expander("scMAE (Masked Autoencoder)"):
      st.markdown("""
      **Type**: Deep Learning (Generative)
      
      **How it works**:
      Inspired by MAE in computer vision. It randomly masks a large percentage of the input genes and forces the network to reconstruct them.
      
      **Relevance**:
      Forces the model to learn really strong "gene programs" or co-expression patterns, because it has to predict missing genes from the visible ones. Extremely robust to noise.
      
      **Key Parameters**:
      - **Mask Ratio**: % of genes to hide during training (e.g., 0.5 or 0.75).
      """)

    # --- scNAME ---
    with st.expander("scNAME (Neighborhood Contrastive + Mask Estimation)"):
      st.markdown("""
      **Type**: Deep Learning (Contrastive + Denoising) 
      **Reference**: Wan et al. (Bioinformatics, 2022)
      
      **How it works**:
      Combines three powerful ideas:
      1. **Mask Estimation**: Tries to figure out which zeros are "real" vs "dropout".
      2. **Contrastive Learning**: Pulls a cell's representation closer to its neighbors.
      3. **Memory Bank**: Keeps track of cell representations across batches.
      
      **Relevance**:
      State-of-the-art for denoising data while clustering.
      """)

    with st.expander("How to add +Harmony variants"):
      st.markdown("""
      Use this when you want to evaluate both a base method and its Harmony-corrected counterpart.

      1. Open **Customize Benchmark > Algorithms**.
      2. Select the report methods to rerun, for example `scMAE`, `scNAME`, or `scvi`.
      3. Enable **Also evaluate + Harmony variants when available**.
      4. Keep all compatible methods selected, or restrict the list under **Apply +Harmony to these selected methods**.
      5. Generate the commands. The output will include the base methods and the additional variants, for example `scMAE` plus `scMAE+Harmony`.

      For deep learning methods, Harmony is not applied before training. It is applied after training on the learned latent embedding, then clustering and metrics are recomputed on the corrected representation.
      """)



  # =========================================================================
  # 5. REPORT REPRODUCTION GUIDE
  # =========================================================================
  with tab_report:
    st.header("5. Report Reproduction Guide")
    st.markdown("""
    The **Report Reproduction** page is the recommended entry point for rerunning experiments from the M2 internship report. It avoids putting long command blocks in the documentation and keeps the mapping inside the software.

    ### How to choose the right tab

    | Goal | Use this tab |
    |------|--------------|
    | Check which report figure or table can be rerun | **Traceability** |
    | Reproduce the stable generalist benchmark | **Stable Generalist** |
    | Reproduce complementary report experiments such as loss-transfer, inductive splits, or DEG-related analyses | **Report Complements** |
    | Design a close variant of a report experiment | **Custom Protocols** |

    ### Recommended workflow

    1. Open **Report Reproduction**.
    2. Start with **Traceability** to identify the figure or table you want to reproduce. Only rerunnable entries are shown.
    3. Move to the corresponding launcher tab.
    4. Review datasets, seeds, output directory, and runtime options.
    5. Generate or launch the script from the interface.
    6. Inspect the produced artifacts in **Results Explorer**.

    The launcher generation step is lightweight. Running the generated scripts can be long, especially for deep learning methods, multi-seed evaluations, and scIB metrics.

    ### When to use Customize Benchmark instead

    Use **Customize Benchmark** when you are designing a new experiment rather than reproducing a report figure. This is also where you should add `+Harmony` variants to selected methods, tune hyperparameters, or change the preprocessing strategy.

    ### Harmony in report experiments

    - Use **Preprocessing > Batch Correction** when you want a corrected input representation before model fitting.
    - Use **Algorithms > +Harmony variants** when you want an additional method run where Harmony is applied post-hoc.
    - For deep learning methods, the `+Harmony` variant trains the model first, applies Harmony on the learned latent embedding, then recomputes the final clustering.
    """)

  # =========================================================================
  # 6. PARAMETER TUNING GUIDE
  # =========================================================================
  with tab_tuning:
    st.header("6. How to Choose Parameters")
    
    st.markdown("""
    Choosing the right hyperparameters is crucial for deep learning models. This guide helps you select values based on your data quality and hyperparameter search results.
    
    ### 1. Based on Data Quality (from Upload Report)
    
    **If your dataset is Small (< 2,000 cells):**
    - **Latent Dimension**: Use smaller values (16 or 32) to prevent overfitting.
    - **Model Complexity**: Avoid very deep networks (use default layers).
    - **Batch Size**: 64 or 128 is sufficient.
    
    **If your dataset is Large (> 20,000 cells):**
    - **Latent Dimension**: Increase to 64 or 128 to capture more complex biology.
    - **Batch Size**: Increase to 256 or 512 for faster training.
    - **Epochs**: You might need fewer epochs as the model sees more data per epoch.
    
    **If your dataset is Sparse (> 90% zeros):**
    - **Algorithm Choice**: Prefer scMAE or scNAME which are designed for sparsity.
    - **Dropout**: Do NOT add extra dropout in preprocessing.
    - **Reconstruction Weight**: In scCIDCG, increase `factor_construct` slightly.
    
    ### 2. Based on Hyperparameter Search
    
    When running the **Hyperparameter Search** module:
    - **Range Selection**: Start with a wide range (e.g., learning rate 0.0001 to 0.01).
    - **Interpretation**:
      - If the best performing value is at the *edge* of your range, shift the range in that direction and search again.
      - If performance varies wildly with small changes, the model is unstable. Lower the learning rate.
      - If performance is flat across a parameter, choose the value that is computationally cheapest (e.g., fewer epochs, smaller dimension).

    ### 3. Common Parameters Guide
    
    #### Learning Rate (`lr`)
    - **Too High**: Loss oscillates or explodes (NaN).
    - **Too Low**: Loss decreases extremely slowly.
    - **Tip**: 0.001 is a safe starting point for Adam optimizer.
    
    #### Latent Dimension
    - **Trade-off**: Information preservation vs. Clustering compactness.
    - **Visual check**: If UMAP looks like a fuzzy ball, dimension might be too low (collapsed) or too high (noise).
    
    #### Epochs
    - **Monitoring**: Watch the training loss curve.
    - **Stop**: When the curve flattens out (plateau).
    - **Tip**: Pretraining needs more epochs (200-400) than the clustering phase (100-200).
    """)

  # =========================================================================
  # 7. REFERENCES
  # =========================================================================
  with st.expander("Scientific References"):
    st.markdown("""
    The parameter tuning guides and impact assessments in this application are derived from the following rigorous sources:

    1. **scDeepCluster**:
      *  Tian, T. et al. (2019). "Clustering Single-Cell RNA-seq Data with a Recurrent Neural Network". *Nature Machine Intelligence*.
      *  *Key Insights*: Importance of ZINB loss for count data; Pretraining necessity.

    2. **scMAE**:
      *  Chung, H. et al. (2024). "Delineating the Effective Use of Self-Supervised Learning in Single-Cell Genomics". *bioRxiv* 10.1101/2024.02.16.580624.
      *  *Key Insights*: High masking rates (0.75) are crucial for learning meaningful gene correlations; Necessity of long training schedules.

    3. **scCDCG**:
      *  "Deep Cut-informed Graph Embedding for Single-Cell Clustering". *DASFAA 2024*.
      *  *Key Insights*: Sinkhorn temperature impact on cluster sharpness; Balancing local (KNN) vs global regularization.

    4. **scNAME**:
      *  Wan, et al. (2022). "scNAME: Neighborhood Contrastive Clustering with Mask Estimation...". *Bioinformatics*.
      *  *Key Insights*: Critical role of mask probability (`p_m`) in denoising; Neighborhood contrastive loss stability.

    5. **Scanpy & Seurat Best Practices**:
      *  Wolf, F. et al. (2018). "SCANPY: large-scale single-cell gene expression data analysis". *Genome Biology*.
      *  Luecken, M.D. & Theis, F.J. (2019). "Current best practices in single-cell RNA-seq analysis: a tutorial". *Molecular Systems Biology*.
      *  *Key Insights*: Standard parameters for PCA (10-50 components), Leiden resolution effects, and KNN graph construction.


    """)

if __name__ == "__main__":
  # Test standalone
  st.set_page_config(page_title="Documentation", layout="wide")
  render_documentation_page()
