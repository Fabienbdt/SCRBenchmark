"""
Shared GUI constants.

UI-only constants must live here to avoid duplicated literals across pages.
"""

ALGO_COLORS = {
  "scdeepcluster": "#1f77b4",
  "pca": "#17becf",
  "pca_kmeans": "#ff7f0e",
  "pca_leiden": "#2ca02c",
  "scname": "#9467bd",
  "sccdcg": "#8c564b",
  "sc_mae": "#e377c2",
  "scvi": "#aec7e8",
}

ALGO_DISPLAY_NAMES = {
  "scdeepcluster": "scDeepCluster",
  "pca": "PCA + clustering",
  "pca_kmeans": "PCA + K-Means",
  "pca_leiden": "PCA + Leiden",
  "scname": "scNAME",
  "sccdcg": "scCDCG",
  "sc_mae": "scMAE",
  "scvi": "scVI",
}

BATCH_COLUMN_CANDIDATES = [
  "batch",
  "Batch",
  "tech",
  "sample",
  "donor",
  "patient",
]

LABEL_COLUMN_CANDIDATES = [
  "Group",
  "labels",
  "cell_type",
  "celltype",
  "CellType",
  "label",
  "labels_encoded",
  "cluster",
  "Cluster",
  "Sub_Cluster",
]

METRIC_NAMES = [
  "NMI",
  "ARI",
  "ACC",
  "Silhouette",
  "F1_Macro",
  "BalancedACC",
]
