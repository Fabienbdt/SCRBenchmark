"""
Pytest fixtures for algorithm comparison tests.
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "scrbenchmark"))


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return PROJECT_ROOT / "tests" / "comparisons" / "data"


@pytest.fixture
def results_dir():
    """Return path to results directory."""
    results_path = PROJECT_ROOT / "tests" / "comparisons" / "results"
    results_path.mkdir(exist_ok=True)
    return results_path


@pytest.fixture
def authors_code_dir():
    """Return path to authors' original code directory."""
    return PROJECT_ROOT / "Authors__original_code"


@pytest.fixture
def random_seed():
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture
def small_synthetic_data(random_seed):
    """
    Generate small synthetic scRNA-seq-like data for testing.

    Creates a dataset with clear cluster structure for validation.
    """
    np.random.seed(random_seed)

    n_cells_per_cluster = 100
    n_clusters = 5
    n_genes = 500

    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_genes) * 3

    # Generate cells around each center
    X = []
    labels = []

    for i in range(n_clusters):
        cluster_data = centers[i] + np.random.randn(n_cells_per_cluster, n_genes) * 0.5
        X.append(cluster_data)
        labels.extend([i] * n_cells_per_cluster)

    X = np.vstack(X)
    labels = np.array(labels)

    # Make non-negative (like count data)
    X = np.maximum(X, 0)

    # Shuffle
    indices = np.random.permutation(len(labels))
    X = X[indices]
    labels = labels[indices]

    return {
        'X': X.astype(np.float32),
        'labels': labels,
        'n_clusters': n_clusters,
        'gene_names': np.array([f'gene_{i}' for i in range(n_genes)])
    }


@pytest.fixture
def pbmc3k_data():
    """
    Load PBMC 3k reference dataset if available.
    """
    import scanpy as sc
    import anndata as ad

    data_path = PROJECT_ROOT / "data" / "pbmc3k_raw.h5ad"

    if data_path.exists():
        adata = sc.read_h5ad(data_path)
        return adata
    else:
        pytest.skip("PBMC 3k dataset not available")


def compare_metrics(labels1, labels2, threshold=0.8):
    """
    Compare two clustering results using multiple metrics.

    Args:
        labels1: First clustering labels
        labels2: Second clustering labels
        threshold: Minimum ARI threshold for "similar" results

    Returns:
        dict with comparison metrics and similarity assessment
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)

    return {
        'ari': ari,
        'nmi': nmi,
        'are_similar': ari >= threshold,
        'threshold': threshold
    }


def compute_embedding_correlation(emb1, emb2):
    """
    Compute correlation between two embedding matrices.

    Since the latent spaces may be rotated/scaled differently,
    we compare the pairwise distance matrices.

    Args:
        emb1, emb2: Embedding matrices (N x D)

    Returns:
        Correlation coefficient between distance matrices
    """
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr

    # Compute pairwise distances
    dist1 = pdist(emb1)
    dist2 = pdist(emb2)

    # Compute correlation
    corr, p_value = pearsonr(dist1, dist2)

    return {
        'correlation': corr,
        'p_value': p_value,
        'are_similar': corr > 0.7
    }
