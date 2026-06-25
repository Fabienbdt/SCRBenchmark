import numpy as np

from algorithms.sc_mae_scraw_weighted import ScMaeScrawWeightedAlgorithm
from algorithms.scdeepcluster_scraw_weighted import ScDeepClusterScrawWeighted


def test_scmae_density_only_skips_reconstruction_pseudo_labels(monkeypatch):
    alg = ScMaeScrawWeightedAlgorithm({"weight_component_mode": "density_only"})
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ],
        dtype=np.float32,
    )

    def fail_cluster_weights(*args, **kwargs):
        raise AssertionError("density_only must not compute cluster-frequency weights")

    monkeypatch.setattr(alg, "_compute_cluster_frequency_weights", fail_cluster_weights)
    weights, labels = alg._compute_combined_cell_weights(embeddings, n_clusters=2)

    assert weights.shape == (4,)
    assert np.all(np.isfinite(weights))
    assert np.all(labels == 0)


def test_scmae_density_only_triplet_kmeans_skips_leiden(monkeypatch):
    alg = ScMaeScrawWeightedAlgorithm(
        {
            "weight_component_mode": "density_only",
            "triplet_pseudo_label_method": "kmeans",
            "random_state": 0,
        }
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.2, 10.0],
        ],
        dtype=np.float32,
    )

    def fail_leiden(*args, **kwargs):
        raise AssertionError("density_only_triplet_kmeans must not call Leiden")

    monkeypatch.setattr(alg, "_leiden_pseudo_labels", fail_leiden)
    labels = alg._compute_triplet_pseudo_labels(embeddings, n_clusters=2)

    assert labels.shape == (6,)
    assert len(np.unique(labels)) == 2


def test_scdeepcluster_density_only_skips_reconstruction_pseudo_labels(monkeypatch):
    model = ScDeepClusterScrawWeighted(
        input_dim=2,
        z_dim=2,
        encodeLayer=[4],
        decodeLayer=[4],
        device="cpu",
        weight_params={"weight_component_mode": "density_only"},
    )
    embeddings = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ],
        dtype=np.float32,
    )

    def fail_cluster_weights(*args, **kwargs):
        raise AssertionError("density_only must not compute cluster-frequency weights")

    monkeypatch.setattr(model, "_compute_cluster_frequency_weights", fail_cluster_weights)
    weights, labels = model._compute_combined_cell_weights(embeddings, n_clusters=2)

    assert weights.shape == (4,)
    assert np.all(np.isfinite(weights))
    assert np.all(labels == 0)
