"""Regression tests for warning-free, legacy-equivalent algorithm helpers."""

from pathlib import Path
import sys
import warnings

import anndata as ad
import numpy as np
import pytest
import torch




def _normalization_warnings(captured):
    return [
        warning
        for warning in captured
        if issubclass(warning.category, FutureWarning)
        and "normalize_total" in str(warning.message)
    ]


@pytest.mark.parametrize(
    ("algorithm_factory", "method_name"),
    [
        (
            lambda: __import__(
                "scrbenchmark.algorithms.sc_mae", fromlist=["ScMaeAlgorithm"]
            ).ScMaeAlgorithm({"n_hvg": 0}),
            "_scanpy_preprocess",
        ),
        (
            lambda: __import__(
                "scrbenchmark.algorithms.scname", fromlist=["ScNAMEAlgorithm"]
            ).ScNAMEAlgorithm(),
            "_normalize_scname",
        ),
    ],
)
def test_internal_normalizers_preserve_legacy_size_factors(
    algorithm_factory,
    method_name,
):
    """Modern Scanpy normalization should preserve legacy median size factors."""
    counts = np.array(
        [
            [1, 1, 1, 1],
            [2, 1, 1, 1],
            [3, 2, 1, 1],
            [4, 2, 2, 1],
        ],
        dtype=np.float32,
    )
    totals = counts.sum(axis=1)
    algorithm = algorithm_factory()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        processed = getattr(algorithm, method_name)(ad.AnnData(counts.copy()))

    assert not _normalization_warnings(captured)
    np.testing.assert_allclose(processed.obs["n_counts"].to_numpy(), totals)
    np.testing.assert_allclose(
        processed.obs["size_factors"].to_numpy(),
        totals / np.median(totals),
    )
    assert algorithm._train_norm_target == pytest.approx(float(np.median(totals)))


def test_scdeepcluster_fit_preserves_legacy_size_factors(monkeypatch):
    """scDeepCluster should use the same counts while avoiding deprecated Scanpy APIs."""
    from scrbenchmark.algorithms import scdeepcluster as module

    counts = np.arange(1, 73, dtype=np.float64).reshape(12, 6) % 5 + 1
    totals = counts.sum(axis=1)
    captured_inputs = {}

    def fake_pretrain(self, X, X_raw, size_factor, **kwargs):
        captured_inputs["size_factor"] = np.asarray(size_factor).copy()

    def fake_cluster_fit(self, X, X_raw, size_factor, **kwargs):
        return np.zeros(X.shape[0], dtype=int), np.zeros((X.shape[0], self.z_dim))

    monkeypatch.setattr(module.scDeepCluster, "pretrain_autoencoder", fake_pretrain)
    monkeypatch.setattr(module.scDeepCluster, "fit", fake_cluster_fit)
    algorithm = module.ScDeepClusterAlgorithm(
        {
            "n_clusters": 2,
            "pretrain_epochs": 0,
            "maxiter": 0,
            "use_ground_truth_k": False,
        }
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        algorithm.fit(counts, np.repeat([0, 1], 6))

    assert not _normalization_warnings(captured)
    np.testing.assert_allclose(
        captured_inputs["size_factor"],
        totals / np.median(totals),
    )


def test_apply_noise_copies_tensor_probabilities_without_warning():
    """Tensor probabilities should retain the legacy detached-copy behavior."""
    from scrbenchmark.algorithms.sc_mae import apply_noise

    X = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    probabilities = torch.full((3,), 0.25, requires_grad=True)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        corrupted, mask = apply_noise(X, probabilities)

    assert corrupted.shape == X.shape
    assert mask.shape == X.shape
    assert not any("copy construct from a tensor" in str(item.message) for item in captured)


def test_sccdcg_laplacian_handles_nonpositive_degrees_without_warning():
    """Invalid graph degrees should map to zero rows instead of NaN/Inf."""
    from scrbenchmark.algorithms.sccdcg import get_laplace_matrix

    adjacency = np.array(
        [
            [0.0, -2.0, 1.0],
            [-2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    with np.errstate(all="raise"):
        laplacian = get_laplace_matrix(adjacency)

    assert torch.isfinite(laplacian).all()
    assert torch.count_nonzero(laplacian) == 0
