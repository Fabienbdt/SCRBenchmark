"""
Unit tests for benchmark preprocessing quality-filter propagation.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

ad = pytest.importorskip("anndata")
pytest.importorskip("scanpy")

from utils.dataset_splitter import BenchmarkPreprocessor


def _make_adata(matrix: np.ndarray):
    obs_names = [f"cell_{i}" for i in range(matrix.shape[0])]
    var_names = [f"g{i}" for i in range(matrix.shape[1])]
    adata = ad.AnnData(X=matrix.astype(np.float32))
    adata.obs_names = obs_names
    adata.var_names = var_names
    adata.obs["Group"] = ["A"] * matrix.shape[0]
    return adata


def test_benchmark_preprocessor_applies_cell_and_gene_filters():
    """Cell/gene filtering configured in UI must be applied in benchmark fit/transform."""
    train_matrix = np.array(
        [
            [1, 1, 0, 0, 0, 0],  # keep (2 genes)
            [1, 0, 0, 0, 0, 0],  # drop (min_genes=2)
            [1, 1, 1, 0, 0, 0],  # keep (3 genes)
            [1, 1, 1, 1, 1, 0],  # drop (max_genes=4)
            [0, 1, 0, 0, 0, 0],  # drop (min_genes=2)
            [0, 1, 1, 0, 0, 0],  # keep (2 genes)
        ],
        dtype=np.float32,
    )
    test_matrix = np.array(
        [
            [1, 1, 0, 0, 0, 0],  # keep
            [1, 0, 0, 0, 0, 0],  # drop (min_genes=2)
            [1, 1, 1, 1, 0, 0],  # keep (4 genes)
        ],
        dtype=np.float32,
    )

    adata_train = _make_adata(train_matrix)
    adata_test = _make_adata(test_matrix)

    params = {
        "do_cell_filtering": True,
        "min_genes_per_cell": 2,
        "max_genes_per_cell": 4,
        "do_gene_filtering": True,
        "min_cells_per_gene": 2,
        "do_normalization": False,
        "do_log_transform": False,
        "do_hvg": False,
        "do_scaling": False,
        "dropout_method": "none",
        "do_batch_correction": False,
    }

    preprocessor = BenchmarkPreprocessor().fit(adata_train, params)
    train_processed = preprocessor.transform(adata_train, params)
    test_processed = preprocessor.transform(adata_test, params)

    # Expected after train-based filtering: keep cells [0,2,5] and genes [g0,g1,g2]
    assert train_processed.n_obs == 3
    assert train_processed.n_vars == 3
    assert list(train_processed.var_names) == ["g0", "g1", "g2"]

    # Same filtering rules must apply to test split (no silent bypass)
    assert test_processed.n_obs == 2
    assert test_processed.n_vars == 3
    assert list(test_processed.var_names) == ["g0", "g1", "g2"]

    # Raw-count layer must stay aligned with transformed matrix.
    assert "original_X" in train_processed.layers
    assert train_processed.layers["original_X"].shape == train_processed.X.shape
