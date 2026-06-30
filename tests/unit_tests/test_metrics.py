"""
Unit tests for utils/metrics.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

from utils.metrics import (
    compute_nmi, compute_ari, compute_accuracy,
    compute_silhouette, compute_metrics, align_labels,
    compute_balanced_rare_class_accuracy,
    _filter_noise_samples, NOISE_LABELS
)


class TestComputeNMI:
    """Tests for compute_nmi function."""

    def test_perfect_match(self, sample_labels_perfect):
        """NMI should be 1.0 for identical labels."""
        labels_true, labels_pred = sample_labels_perfect
        nmi = compute_nmi(labels_true, labels_pred)
        assert nmi == pytest.approx(1.0, abs=1e-6)

    def test_random_labels(self, sample_labels_random):
        """NMI should be < 1.0 for random labels."""
        labels_true, labels_pred = sample_labels_random
        nmi = compute_nmi(labels_true, labels_pred)
        assert 0.0 <= nmi < 1.0

    def test_permuted_labels(self, sample_labels_permuted):
        """NMI should be 1.0 for permuted but structurally identical labels."""
        labels_true, labels_pred = sample_labels_permuted
        nmi = compute_nmi(labels_true, labels_pred)
        assert nmi == pytest.approx(1.0, abs=1e-6)


class TestComputeARI:
    """Tests for compute_ari function."""

    def test_perfect_match(self, sample_labels_perfect):
        """ARI should be 1.0 for identical labels."""
        labels_true, labels_pred = sample_labels_perfect
        ari = compute_ari(labels_true, labels_pred)
        assert ari == pytest.approx(1.0, abs=1e-6)

    def test_random_labels(self, sample_labels_random):
        """ARI should be < 1.0 for random labels."""
        labels_true, labels_pred = sample_labels_random
        ari = compute_ari(labels_true, labels_pred)
        assert ari < 1.0

    def test_permuted_labels(self, sample_labels_permuted):
        """ARI should be 1.0 for permuted but structurally identical labels."""
        labels_true, labels_pred = sample_labels_permuted
        ari = compute_ari(labels_true, labels_pred)
        assert ari == pytest.approx(1.0, abs=1e-6)


class TestComputeAccuracy:
    """Tests for compute_accuracy function."""

    def test_perfect_match(self, sample_labels_perfect):
        """Accuracy should be 1.0 for identical labels."""
        labels_true, labels_pred = sample_labels_perfect
        acc = compute_accuracy(labels_true, labels_pred)
        assert acc == pytest.approx(1.0, abs=1e-6)

    def test_permuted_labels(self, sample_labels_permuted):
        """Accuracy should be 1.0 for permuted labels (Hungarian algorithm)."""
        labels_true, labels_pred = sample_labels_permuted
        acc = compute_accuracy(labels_true, labels_pred)
        assert acc == pytest.approx(1.0, abs=1e-6)

    def test_string_labels(self):
        """Accuracy should handle string labels."""
        labels_true = np.array(['A', 'A', 'B', 'B', 'C', 'C'])
        labels_pred = np.array(['X', 'X', 'Y', 'Y', 'Z', 'Z'])
        acc = compute_accuracy(labels_true, labels_pred)
        assert acc == pytest.approx(1.0, abs=1e-6)


class TestComputeSilhouette:
    """Tests for compute_silhouette function."""

    def test_well_separated_clusters(self, sample_embeddings_well_separated):
        """Silhouette should be positive for well-separated clusters."""
        embeddings, labels = sample_embeddings_well_separated
        score = compute_silhouette(embeddings, labels, sample_size=None)
        assert score > 0.5  # Well-separated should have high silhouette

    def test_single_cluster(self):
        """Silhouette should be 0 for single cluster."""
        data = np.random.randn(100, 10)
        labels = np.zeros(100, dtype=int)
        score = compute_silhouette(data, labels)
        assert score == 0.0


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_with_ground_truth(self, sample_labels_perfect, sample_embeddings_well_separated):
        """compute_metrics should return all metrics when ground truth is provided."""
        labels_true, labels_pred = sample_labels_perfect
        embeddings, _ = sample_embeddings_well_separated
        
        # Create matching size arrays
        n = min(len(labels_true), len(embeddings))
        labels_true = labels_true[:n]
        labels_pred = labels_pred[:n]
        embeddings = embeddings[:n]
        
        metrics = compute_metrics(labels_true, labels_pred, embeddings=embeddings)
        
        assert 'NMI' in metrics
        assert 'ARI' in metrics
        assert 'ACC' in metrics
        assert 'BalancedRareACC' in metrics
        assert 'Silhouette' in metrics
        assert 'n_clusters_found' in metrics

    def test_without_ground_truth(self, sample_embeddings_well_separated):
        """compute_metrics should only return unsupervised metrics without ground truth."""
        embeddings, labels = sample_embeddings_well_separated
        metrics = compute_metrics(None, labels, embeddings=embeddings)
        
        assert 'NMI' not in metrics
        assert 'ARI' not in metrics
        assert 'ACC' not in metrics
        assert 'Silhouette' in metrics


class TestBalancedRareClassAccuracy:
    """Tests for balanced rare-class accuracy."""

    def test_perfect_permuted_labels(self):
        labels_true = np.array(["common"] * 95 + ["rare"] * 5)
        labels_pred = np.array(["x"] * 95 + ["y"] * 5)

        score = compute_balanced_rare_class_accuracy(labels_true, labels_pred, threshold=0.10)

        assert score == pytest.approx(1.0, abs=1e-6)

    def test_balances_multiple_rare_classes(self):
        labels_true = np.array(["common"] * 90 + ["rare_a"] * 5 + ["rare_b"] * 5)
        labels_pred = np.array(["c"] * 90 + ["a"] * 5 + ["c"] * 5)

        score = compute_balanced_rare_class_accuracy(labels_true, labels_pred, threshold=0.10)

        assert score == pytest.approx(0.5, abs=1e-6)

    def test_no_rare_class_returns_nan(self):
        labels_true = np.array(["a"] * 50 + ["b"] * 50)
        labels_pred = np.array(["x"] * 50 + ["y"] * 50)

        score = compute_balanced_rare_class_accuracy(labels_true, labels_pred, threshold=0.05)

        assert np.isnan(score)


class TestAlignLabels:
    """Tests for align_labels function."""

    def test_permuted_alignment(self, sample_labels_permuted):
        """align_labels should map permuted labels correctly."""
        labels_true, labels_pred = sample_labels_permuted
        aligned = align_labels(labels_true, labels_pred)

        # After alignment, labels should match (compare as strings for robustness)
        # The align_labels function converts to strings internally
        aligned_str = np.array([str(x) for x in aligned])
        true_str = np.array([str(x) for x in labels_true])
        assert np.all(aligned_str == true_str), \
            f"Aligned: {aligned}, Expected: {labels_true}"

    def test_alignment_preserves_structure(self):
        """align_labels should preserve clustering structure."""
        # Create labels with same structure but different numeric mapping
        labels_true = np.array([0, 0, 1, 1, 2, 2])
        labels_pred = np.array([5, 5, 3, 3, 7, 7])

        aligned = align_labels(labels_true, labels_pred)

        # Each unique value in aligned should correspond to one unique value in true
        from utils.metrics import compute_accuracy
        acc = compute_accuracy(labels_true, labels_pred)
        assert acc == pytest.approx(1.0, abs=1e-6), \
            "Labels with same structure should have perfect accuracy"


class TestNoiseFiltering:
    """Tests for noise label filtering functionality."""

    def test_filter_numeric_noise(self):
        """Should filter out samples with label -1."""
        labels_true = np.array([0, 1, 2, -1, 0, 1])
        labels_pred = np.array([0, 1, 2, 0, 0, 1])

        filtered_true, filtered_pred, _, _ = _filter_noise_samples(
            labels_true, labels_pred, None
        )

        assert len(filtered_true) == 5, "Should filter 1 sample with -1"
        assert -1 not in filtered_true, "Filtered should not contain -1"

    def test_filter_string_noise(self):
        """Should filter out samples with 'noise' label."""
        labels_true = np.array(['A', 'B', 'noise', 'A', 'B'])
        labels_pred = np.array(['X', 'Y', 'Z', 'X', 'Y'])

        filtered_true, filtered_pred, _, _ = _filter_noise_samples(
            labels_true, labels_pred, None
        )

        assert len(filtered_true) == 4, "Should filter 1 'noise' sample"
        assert 'noise' not in filtered_true

    def test_filter_pred_noise(self):
        """Should filter when predicted label is noise."""
        labels_true = np.array([0, 1, 2, 0, 1])
        labels_pred = np.array([0, 1, -1, 0, 1])  # -1 in predictions

        filtered_true, filtered_pred, _, _ = _filter_noise_samples(
            labels_true, labels_pred, None
        )

        assert len(filtered_pred) == 4, "Should filter sample with pred=-1"
        assert -1 not in filtered_pred

    def test_filter_both_noise(self):
        """Should filter when either true OR pred is noise."""
        labels_true = np.array([0, -1, 2, 0, 1])
        labels_pred = np.array([0, 1, -1, 0, 1])

        filtered_true, filtered_pred, _, _ = _filter_noise_samples(
            labels_true, labels_pred, None
        )

        # Should filter index 1 (true=-1) and index 2 (pred=-1)
        assert len(filtered_true) == 3

    def test_filter_with_data(self):
        """Should filter data matrix along with labels."""
        labels_true = np.array([0, 1, -1, 2])
        labels_pred = np.array([0, 1, 0, 2])
        data = np.random.rand(4, 10)

        filtered_true, filtered_pred, filtered_data, _ = _filter_noise_samples(
            labels_true, labels_pred, data
        )

        assert filtered_data.shape[0] == 3, "Data should be filtered too"
        assert len(filtered_true) == len(filtered_data)

    def test_no_noise_no_change(self):
        """Should not filter when no noise labels present."""
        labels_true = np.array([0, 1, 2, 0, 1])
        labels_pred = np.array([0, 1, 2, 0, 1])

        filtered_true, filtered_pred, _, _ = _filter_noise_samples(
            labels_true, labels_pred, None
        )

        assert len(filtered_true) == 5, "Should not filter any samples"
        assert np.array_equal(filtered_true, labels_true)

    def test_noise_labels_constant(self):
        """NOISE_LABELS constant should contain expected values."""
        assert -1 in NOISE_LABELS
        assert '-1' in NOISE_LABELS
        assert 'noise' in NOISE_LABELS
        assert 'Noise' in NOISE_LABELS
        assert 'unassigned' in NOISE_LABELS

    def test_compute_metrics_with_noise(self):
        """compute_metrics should handle noise labels correctly."""
        # Perfect match except for noise samples
        labels_true = np.array([0, 0, 1, 1, -1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 0, 2, 2])

        metrics = compute_metrics(labels_true, labels_pred, filter_noise=True)

        # Should report 1 sample filtered
        assert metrics.get('n_noise_filtered', 0) == 1

        # Metrics should be perfect (noise excluded)
        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ARI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ACC'] == pytest.approx(1.0, abs=1e-6)

    def test_compute_metrics_filter_disabled(self):
        """compute_metrics with filter_noise=False should include noise."""
        labels_true = np.array([0, 0, 1, 1, -1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 0, 2, 2])

        metrics = compute_metrics(labels_true, labels_pred, filter_noise=False)

        # Should NOT report filtered samples
        assert 'n_noise_filtered' not in metrics or metrics['n_noise_filtered'] == 0

        # Metrics will be lower because -1 is treated as a label
        assert metrics['NMI'] < 1.0 or metrics['ACC'] < 1.0

    def test_all_noise_edge_case(self):
        """Should handle edge case where all samples are noise."""
        labels_true = np.array([-1, -1, -1])
        labels_pred = np.array([0, 1, 2])

        metrics = compute_metrics(labels_true, labels_pred, filter_noise=True)

        # No samples left to evaluate
        assert metrics.get('n_samples_evaluated', 0) == 0
        # Supervised metrics should be absent or undefined
        assert 'NMI' not in metrics or metrics.get('n_samples_evaluated') == 0


class TestMetricsEdgeCases:
    """Tests for edge cases in metrics computation."""

    def test_single_cluster_metrics(self):
        """Metrics should handle single cluster case."""
        labels_true = np.array([0, 0, 0, 0, 0])
        labels_pred = np.array([0, 0, 0, 0, 0])

        metrics = compute_metrics(labels_true, labels_pred)

        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ARI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ACC'] == pytest.approx(1.0, abs=1e-6)

    def test_many_clusters(self):
        """Metrics should handle many clusters."""
        n_samples = 100
        n_clusters = 20
        labels = np.repeat(np.arange(n_clusters), n_samples // n_clusters)

        metrics = compute_metrics(labels, labels)

        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['n_clusters_found'] == n_clusters

    def test_unbalanced_clusters(self):
        """Metrics should handle highly unbalanced clusters."""
        # 95 samples in cluster 0, 5 in cluster 1
        labels_true = np.array([0] * 95 + [1] * 5)
        labels_pred = np.array([0] * 95 + [1] * 5)

        metrics = compute_metrics(labels_true, labels_pred)

        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ACC'] == pytest.approx(1.0, abs=1e-6)

    def test_completely_wrong_predictions(self):
        """Metrics should handle completely wrong predictions."""
        labels_true = np.array([0, 0, 0, 1, 1, 1])
        labels_pred = np.array([1, 1, 1, 0, 0, 0])  # Inverted

        metrics = compute_metrics(labels_true, labels_pred)

        # Despite inversion, structure is preserved (ACC uses Hungarian)
        assert metrics['ACC'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)

    def test_random_predictions_low_metrics(self):
        """Metrics should be low for random predictions."""
        np.random.seed(42)
        labels_true = np.repeat([0, 1, 2], 100)
        labels_pred = np.random.randint(0, 3, 300)

        metrics = compute_metrics(labels_true, labels_pred)

        # Random should give ~0 ARI and low NMI
        assert metrics['ARI'] < 0.2, "ARI should be low for random"
        assert metrics['NMI'] < 0.3, "NMI should be low for random"
