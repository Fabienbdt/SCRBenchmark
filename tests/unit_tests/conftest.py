"""
Pytest fixtures for SCRBenchmark unit tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))


@pytest.fixture
def sample_labels_perfect():
    """Two identical label arrays for testing perfect match."""
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return labels, labels.copy()


@pytest.fixture
def sample_labels_permuted():
    """Labels with different numbering but same clustering."""
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # Permuted
    return labels_true, labels_pred


@pytest.fixture
def sample_labels_random():
    """Random labels for testing non-perfect match."""
    np.random.seed(42)
    labels_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    labels_pred = np.random.randint(0, 3, size=9)
    return labels_true, labels_pred


@pytest.fixture
def sample_embeddings_well_separated():
    """Well separated clusters in 2D for silhouette testing."""
    np.random.seed(42)
    # 3 clusters, well separated
    cluster_0 = np.random.randn(30, 2) + np.array([0, 0])
    cluster_1 = np.random.randn(30, 2) + np.array([10, 0])
    cluster_2 = np.random.randn(30, 2) + np.array([5, 10])
    
    embeddings = np.vstack([cluster_0, cluster_1, cluster_2])
    labels = np.array([0]*30 + [1]*30 + [2]*30)
    
    return embeddings, labels


@pytest.fixture
def synthetic_data_path():
    """Path to synthetic test data."""
    path = Path(__file__).parent.parent / "synthetic_test_data.h5ad"
    if not path.exists():
        pytest.skip(f"Synthetic data not found at {path}")
    return path
