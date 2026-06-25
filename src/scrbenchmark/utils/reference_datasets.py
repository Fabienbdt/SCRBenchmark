"""
Reference datasets for scRNA-seq benchmarking.
Provides easy access to standard datasets used in the literature.
"""

import scanpy as sc
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a reference dataset."""
    name: str
    display_name: str
    description: str
    n_cells: int
    n_genes: int
    n_clusters: int
    source: str
    citation: str
    label_key: str  # Key in adata.obs for ground truth labels


# Registry of available reference datasets
REFERENCE_DATASETS: Dict[str, DatasetInfo] = {
    'pbmc3k': DatasetInfo(
        name='pbmc3k',
        display_name='PBMC 3k (10X Genomics)',
        description='3k Peripheral Blood Mononuclear Cells from a healthy donor. '
                   'Standard benchmark dataset for scRNA-seq methods.',
        n_cells=2700,
        n_genes=32738,
        n_clusters=8,
        source='10X Genomics',
        citation='10X Genomics (2017)',
        label_key='louvain'  # Will be computed after loading
    ),
    'paul15': DatasetInfo(
        name='paul15',
        display_name='Paul15 (Bone Marrow)',
        description='Mouse bone marrow myeloid progenitors. '
                   'Contains 19 annotated cell types along differentiation trajectory.',
        n_cells=2730,
        n_genes=3451,
        n_clusters=19,
        source='Paul et al. Cell 2015',
        citation='Paul et al. (2015) Cell',
        label_key='paul15_clusters'
    ),
    'pbmc68k': DatasetInfo(
        name='pbmc68k',
        display_name='PBMC 68k (Fresh)',
        description='68k fresh PBMCs from a healthy donor. '
                   'Large dataset for performance testing.',
        n_cells=68579,
        n_genes=32738,
        n_clusters=11,
        source='10X Genomics',
        citation='Zheng et al. (2017) Nature Communications',
        label_key='bulk_labels'
    ),
}


def get_available_datasets() -> Dict[str, DatasetInfo]:
    """Return all available reference datasets."""
    return REFERENCE_DATASETS.copy()


def load_reference_dataset(dataset_name: str, cache_dir: Optional[Path] = None) -> Tuple:
    """
    Load a reference dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'pbmc3k', 'paul15')
        cache_dir: Directory to cache downloaded data (default: ~/.cache/scanpy)
    
    Returns:
        Tuple of (adata, labels, dataset_info)
    """
    if dataset_name not in REFERENCE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Available: {list(REFERENCE_DATASETS.keys())}")
    
    info = REFERENCE_DATASETS[dataset_name]
    logger.info(f"Loading reference dataset: {info.display_name}")
    
    if dataset_name == 'pbmc3k':
        adata = _load_pbmc3k()
    elif dataset_name == 'paul15':
        adata = _load_paul15()
    elif dataset_name == 'pbmc68k':
        adata = _load_pbmc68k()
    else:
        raise ValueError(f"Dataset loader not implemented: {dataset_name}")
    
    # Extract labels
    labels = adata.obs[info.label_key].values if info.label_key in adata.obs else None
    
    return adata, labels, info


def _load_pbmc3k():
    """Load PBMC 3k dataset with preprocessing for labels."""
    # Download raw data
    adata = sc.datasets.pbmc3k()
    
    # Basic preprocessing to generate Leiden clusters as "ground truth"
    # This is standard practice for PBMC 3k which doesn't have true labels
    adata_processed = adata.copy()
    
    # Standard preprocessing
    sc.pp.filter_cells(adata_processed, min_genes=200)
    sc.pp.filter_genes(adata_processed, min_cells=3)
    sc.pp.normalize_total(adata_processed, target_sum=1e4)
    sc.pp.log1p(adata_processed)
    sc.pp.highly_variable_genes(adata_processed, n_top_genes=2000)
    adata_processed = adata_processed[:, adata_processed.var.highly_variable]
    sc.pp.scale(adata_processed, max_value=10)
    sc.tl.pca(adata_processed, n_comps=50)
    sc.pp.neighbors(adata_processed, n_neighbors=10, n_pcs=40)
    
    # Use Leiden instead of Louvain (more common, usually available)
    sc.tl.leiden(adata_processed, resolution=0.8)
    
    # Transfer labels back to original data
    # Use intersection of cell indices
    common_cells = adata.obs_names.intersection(adata_processed.obs_names)
    adata = adata[common_cells].copy()
    adata.obs['louvain'] = adata_processed[common_cells].obs['leiden']
    
    return adata


def _load_paul15():
    """Load Paul15 bone marrow dataset."""
    adata = sc.datasets.paul15()
    
    # The dataset has 'paul15_clusters' annotations
    return adata


def _load_pbmc68k():
    """Load PBMC 68k dataset."""
    adata = sc.datasets.pbmc68k_reduced()
    
    return adata


def get_dataset_summary_table() -> str:
    """Generate a markdown table of available datasets."""
    lines = [
        "| Dataset | Cells | Genes | Clusters | Source |",
        "|---------|-------|-------|----------|--------|"
    ]
    
    for name, info in REFERENCE_DATASETS.items():
        lines.append(
            f"| {info.display_name} | {info.n_cells:,} | {info.n_genes:,} | "
            f"{info.n_clusters} | {info.source} |"
        )
    
    return "\n".join(lines)
