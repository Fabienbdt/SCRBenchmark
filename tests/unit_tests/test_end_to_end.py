"""
End-to-end integration tests for SCRBenchmark.

Tests cover:
- Complete pipeline execution from data to clustering
- Multiple algorithm execution
- Metrics computation correctness
- Reproducibility across runs
- Oracle mode vs non-Oracle mode behavior
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))


@pytest.fixture
def synthetic_scrnaseq_data():
    """Generate synthetic scRNA-seq-like data with known clusters."""
    np.random.seed(42)

    n_cells = 300
    n_genes = 100
    n_clusters = 3
    cells_per_cluster = n_cells // n_clusters

    # Create expression matrix with cluster structure
    X = np.zeros((n_cells, n_genes))
    labels = np.zeros(n_cells, dtype=int)

    for i in range(n_clusters):
        start_idx = i * cells_per_cluster
        end_idx = (i + 1) * cells_per_cluster

        # Each cluster has different gene expression patterns
        # Cluster i has high expression in genes [i*20 : (i+1)*20]
        base_expression = np.random.poisson(5, (cells_per_cluster, n_genes))
        cluster_genes = slice(i * 20, (i + 1) * 20)
        base_expression[:, cluster_genes] += np.random.poisson(20, (cells_per_cluster, 20))

        X[start_idx:end_idx] = base_expression
        labels[start_idx:end_idx] = i

    # Add some dropout (zeros)
    dropout_mask = np.random.random(X.shape) < 0.3
    X[dropout_mask] = 0

    return X.astype(np.float64), labels


@pytest.fixture
def minimal_training_params():
    """Minimal parameters for fast testing."""
    return {
        'pretrain_epochs': 2,
        'maxiter': 5,
        'max_epochs': 5,
        'n_epochs': 5,
        'epochs': 5,
        'n_neighbors': 10,
        'random_state': 42
    }


class TestDataHandler:
    """Tests for DataHandler data generation and loading."""

    def test_load_h5ad(self, tmp_path):
        """DataHandler should load H5AD files correctly."""
        from utils.data_handler import DataHandler
        import anndata

        # Create test data
        X = np.random.poisson(5, (100, 50)).astype(np.float32)
        adata = anndata.AnnData(X)
        adata.obs['labels'] = np.repeat([0, 1, 2, 3], 25)

        path = tmp_path / "test.h5ad"
        adata.write(path)

        # Load with DataHandler
        dh = DataHandler()
        dh.load(str(path))

        assert dh.adata is not None, "Should load data"
        assert dh.adata.n_obs == 100, "Should have 100 cells"

    def test_save_and_reload_h5ad(self, tmp_path, synthetic_scrnaseq_data):
        """DataHandler should save and reload data correctly."""
        import anndata

        X, labels = synthetic_scrnaseq_data

        # Create AnnData and save
        adata = anndata.AnnData(X)
        adata.obs['labels'] = labels

        path = tmp_path / "test_save.h5ad"
        adata.write(path)

        # Reload and verify
        loaded = anndata.read_h5ad(path)

        assert loaded.n_obs == len(labels), "Should preserve cell count"
        assert 'labels' in loaded.obs.columns, "Should preserve labels"
        assert np.array_equal(loaded.obs['labels'].values, labels), "Labels should match"


class TestAlgorithmExecution:
    """Tests for algorithm execution with synthetic data."""

    def test_pca_kmeans_execution(self, synthetic_scrnaseq_data):
        """PCA + K-Means should run and produce valid results."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm

        X, labels = synthetic_scrnaseq_data

        algo = PCAKMeansAlgorithm({
            'n_clusters': 3,
            'n_components': 10,
            'random_state': 42,
            'use_ground_truth_k': False
        })

        algo.fit(X, labels)
        pred_labels = algo.get_labels()

        assert pred_labels is not None, "Should produce labels"
        assert len(pred_labels) == len(labels), "Should have same number of labels"
        assert set(pred_labels).issubset({0, 1, 2}), "Should have 3 clusters"

    def test_pca_kmeans_oracle_mode(self, synthetic_scrnaseq_data):
        """PCA + K-Means Oracle mode should use ground truth k."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm

        X, labels = synthetic_scrnaseq_data

        # Without oracle mode, use wrong k
        algo_no_oracle = PCAKMeansAlgorithm({
            'n_clusters': 5,  # Wrong k
            'n_components': 10,
            'random_state': 42,
            'use_ground_truth_k': False
        })
        algo_no_oracle.fit(X, labels)
        pred_no_oracle = algo_no_oracle.get_labels()

        # With oracle mode, should use k=3 from ground truth
        algo_oracle = PCAKMeansAlgorithm({
            'n_clusters': 5,  # Will be overridden
            'n_components': 10,
            'random_state': 42,
            'use_ground_truth_k': True
        })
        algo_oracle.fit(X, labels)
        pred_oracle = algo_oracle.get_labels()

        n_clusters_no_oracle = len(np.unique(pred_no_oracle))
        n_clusters_oracle = len(np.unique(pred_oracle))

        assert n_clusters_no_oracle == 5, "Without oracle should use specified k"
        assert n_clusters_oracle == 3, "Oracle mode should use ground truth k"

    @pytest.mark.slow
    def test_scdeepcluster_execution(self, synthetic_scrnaseq_data, minimal_training_params):
        """scDeepCluster should run and produce valid results."""
        from algorithms.scdeepcluster import ScDeepClusterAlgorithm

        X, labels = synthetic_scrnaseq_data

        params = {
            'n_clusters': 3,
            'random_state': 42,
            'use_ground_truth_k': False,
            **minimal_training_params
        }

        algo = ScDeepClusterAlgorithm(params)
        algo.fit(X, labels)
        pred_labels = algo.get_labels()

        assert pred_labels is not None, "Should produce labels"
        assert len(pred_labels) == len(labels), "Should have same number of labels"

    @pytest.mark.slow
    def test_scdeepcluster_reproducibility(self, synthetic_scrnaseq_data, minimal_training_params):
        """scDeepCluster should be reproducible with same seed."""
        from algorithms.scdeepcluster import ScDeepClusterAlgorithm

        X, labels = synthetic_scrnaseq_data

        results = []
        for _ in range(2):
            params = {
                'n_clusters': 3,
                'random_state': 42,
                **minimal_training_params
            }

            algo = ScDeepClusterAlgorithm(params)
            algo.fit(X, labels)
            results.append(algo.get_labels().copy())

        assert np.array_equal(results[0], results[1]), \
            "scDeepCluster should be reproducible with same seed"


class TestMetricsIntegration:
    """Tests for metrics computation in full pipeline."""

    def test_metrics_on_perfect_clustering(self, synthetic_scrnaseq_data):
        """Metrics should be 1.0 for perfect clustering."""
        from utils.metrics import compute_metrics

        X, labels = synthetic_scrnaseq_data

        # Perfect prediction
        metrics = compute_metrics(labels, labels, embeddings=X)

        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ARI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['ACC'] == pytest.approx(1.0, abs=1e-6)

    def test_metrics_with_noise_filtering(self):
        """Metrics should correctly filter noise labels."""
        from utils.metrics import compute_metrics

        # Labels with noise (-1)
        labels_true = np.array([0, 0, 1, 1, -1, 2, 2])
        labels_pred = np.array([0, 0, 1, 1, 0, 2, 2])  # -1 assigned to cluster 0

        metrics_filtered = compute_metrics(labels_true, labels_pred, filter_noise=True)
        metrics_unfiltered = compute_metrics(labels_true, labels_pred, filter_noise=False)

        # With filtering, noise sample excluded -> perfect match
        assert metrics_filtered['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics_filtered['n_noise_filtered'] == 1

        # Without filtering, -1 treated as label -> imperfect match
        assert metrics_unfiltered['NMI'] < 1.0

    def test_metrics_on_algorithm_output(self, synthetic_scrnaseq_data):
        """Metrics should work correctly on actual algorithm output."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm
        from utils.metrics import compute_metrics

        X, labels = synthetic_scrnaseq_data

        algo = PCAKMeansAlgorithm({
            'n_clusters': 3,
            'n_components': 10,
            'random_state': 42
        })

        algo.fit(X, labels)
        pred_labels = algo.get_labels()
        embeddings = algo.get_embeddings()

        metrics = compute_metrics(labels, pred_labels, embeddings=embeddings)

        # On well-structured synthetic data, clustering should be decent
        assert 'NMI' in metrics
        assert 'ARI' in metrics
        assert 'ACC' in metrics
        assert 'Silhouette' in metrics
        assert metrics['n_clusters_found'] == 3

        # With clear cluster structure, should get reasonable metrics
        assert metrics['NMI'] > 0.3, "NMI should be reasonable on structured data"


class TestAnalysisRunner:
    """Tests for AnalysisRunner integration."""

    def test_analysis_runner_single_algorithm(self, synthetic_scrnaseq_data, tmp_path):
        """AnalysisRunner should execute single algorithm correctly."""
        from utils.analysis_runner import AnalysisRunner

        X, labels = synthetic_scrnaseq_data

        runner = AnalysisRunner(output_dir=tmp_path / "results")

        # Run algorithm
        result = runner.run_algorithm(
            algorithm_name='pca_kmeans',
            data=X,
            labels=labels,
            params={
                'n_clusters': 3,
                'n_components': 10,
                'random_state': 42
            }
        )

        assert result is not None, "Should return result"
        assert result.labels is not None, "Should have labels"
        assert result.metrics is not None, "Should have metrics"
        assert len(result.labels) == len(labels), "Should have same number of labels"

    def test_analysis_runner_multiple_algorithms(self, synthetic_scrnaseq_data, tmp_path):
        """AnalysisRunner should execute multiple algorithms."""
        from utils.analysis_runner import AnalysisRunner

        X, labels = synthetic_scrnaseq_data

        runner = AnalysisRunner(output_dir=tmp_path / "results")

        results = []
        for algo_name, params in [
            ('pca_kmeans', {'n_clusters': 3, 'random_state': 42}),
            ('pca_leiden', {'resolution': 0.5, 'random_state': 42})
        ]:
            result = runner.run_algorithm(
                algorithm_name=algo_name,
                data=X,
                labels=labels,
                params=params
            )
            results.append(result)

        assert len(results) == 2, "Should have 2 results"
        assert all(r.labels is not None for r in results), "All should have labels"


class TestPreprocessingIntegration:
    """Tests for preprocessing integration."""

    def test_preprocessing_pipeline(self, synthetic_scrnaseq_data):
        """Preprocessing should work correctly end-to-end."""
        from utils.data_handler import DataHandler
        import anndata

        X, labels = synthetic_scrnaseq_data

        # Create AnnData
        adata = anndata.AnnData(X)
        adata.obs['labels'] = labels

        dh = DataHandler()
        dh.adata = adata

        # Apply preprocessing steps
        params = {
            'do_cell_filtering': False,  # Skip filtering for speed
            'do_gene_filtering': False,
            'do_normalization': True,
            'target_sum': 10000,
            'do_log_transform': True,
            'do_hvg': False,  # Skip HVG for speed
            'do_scaling': True,
            'scale_max_value': 10.0,
            'dropout_method': 'none'
        }
        dh.preprocess(params)

        processed = dh.get_data()

        assert processed is not None, "Should return processed data"
        assert processed.n_obs == X.shape[0], "Should preserve cell count"

    def test_labels_retrievable_after_preprocessing(self, synthetic_scrnaseq_data, tmp_path):
        """Labels should be retrievable after preprocessing workflow."""
        from utils.data_handler import DataHandler
        import anndata

        X, labels = synthetic_scrnaseq_data

        # Create AnnData and save it (simulating real workflow)
        adata = anndata.AnnData(X)
        adata.obs['labels'] = labels
        path = tmp_path / "test_preprocess.h5ad"
        adata.write(path)

        # Load via DataHandler (this properly detects labels)
        dh = DataHandler()
        dh.load(str(path))

        # Check that labels were detected on load
        original_labels = dh.get_labels()
        assert original_labels is not None, "Labels should be detected on load"
        assert len(original_labels) == len(labels), "Should have all labels"

        # After preprocessing, labels should still be available via get_labels()
        params = {
            'do_cell_filtering': False,
            'do_gene_filtering': False,
            'do_normalization': True,
            'target_sum': 10000,
            'do_log_transform': True,
            'do_hvg': False,
            'do_scaling': False,
            'dropout_method': 'none'
        }
        dh.preprocess(params)

        # Labels accessible via get_labels (stored separately)
        retrieved_labels = dh.get_labels()
        assert retrieved_labels is not None, "Labels should be retrievable after preprocessing"
        assert len(retrieved_labels) == len(labels), "Should preserve all labels"


class TestOracleModeConsistency:
    """Tests for Oracle mode consistency across algorithms."""

    def test_oracle_mode_explicit_vs_implicit(self, synthetic_scrnaseq_data):
        """Oracle mode should only activate when explicitly enabled."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm

        X, labels = synthetic_scrnaseq_data  # 3 clusters

        # Implicit n_clusters=0 should NOT use oracle mode (default behavior)
        algo_default = PCAKMeansAlgorithm({
            'n_clusters': 8,  # Specified k
            'n_components': 10,
            'random_state': 42
            # use_ground_truth_k defaults to False
        })
        algo_default.fit(X, labels)
        pred_default = algo_default.get_labels()

        # Explicit oracle mode
        algo_oracle = PCAKMeansAlgorithm({
            'n_clusters': 8,  # Will be overridden
            'n_components': 10,
            'random_state': 42,
            'use_ground_truth_k': True
        })
        algo_oracle.fit(X, labels)
        pred_oracle = algo_oracle.get_labels()

        n_clusters_default = len(np.unique(pred_default))
        n_clusters_oracle = len(np.unique(pred_oracle))

        assert n_clusters_default == 8, "Default should use specified k"
        assert n_clusters_oracle == 3, "Oracle should use ground truth k"


class TestEdgeCases:
    """Tests for edge cases in end-to-end pipeline."""

    def test_single_cluster_data(self):
        """Pipeline should handle single cluster data."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm
        from utils.metrics import compute_metrics

        np.random.seed(42)
        X = np.random.poisson(5, (100, 50)).astype(np.float64)
        labels = np.zeros(100, dtype=int)  # Single cluster

        algo = PCAKMeansAlgorithm({
            'n_clusters': 1,
            'n_components': 10,
            'random_state': 42
        })

        algo.fit(X, labels)
        pred = algo.get_labels()

        metrics = compute_metrics(labels, pred)

        assert metrics['NMI'] == pytest.approx(1.0, abs=1e-6)
        assert metrics['n_clusters_found'] == 1

    def test_many_small_clusters(self):
        """Pipeline should handle many small clusters."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm
        from utils.metrics import compute_metrics

        np.random.seed(42)
        n_clusters = 10
        cells_per_cluster = 20
        n_cells = n_clusters * cells_per_cluster

        X = np.random.poisson(5, (n_cells, 50)).astype(np.float64)
        labels = np.repeat(np.arange(n_clusters), cells_per_cluster)

        algo = PCAKMeansAlgorithm({
            'n_clusters': n_clusters,
            'n_components': 10,
            'random_state': 42
        })

        algo.fit(X, labels)
        pred = algo.get_labels()

        assert len(np.unique(pred)) == n_clusters, f"Should find {n_clusters} clusters"

    def test_unbalanced_clusters(self):
        """Pipeline should handle highly unbalanced clusters."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm
        from utils.metrics import compute_metrics

        np.random.seed(42)

        # Very unbalanced: 90% in cluster 0, 10% in cluster 1
        X = np.random.poisson(5, (100, 50)).astype(np.float64)
        labels = np.array([0] * 90 + [1] * 10)

        algo = PCAKMeansAlgorithm({
            'n_clusters': 2,
            'n_components': 10,
            'random_state': 42
        })

        algo.fit(X, labels)
        pred = algo.get_labels()

        metrics = compute_metrics(labels, pred)

        # Should at least find 2 clusters
        assert len(np.unique(pred)) == 2


class TestCrossValidation:
    """Tests for cross-validation and stability."""

    def test_multiple_runs_variance(self, synthetic_scrnaseq_data):
        """Multiple runs with different seeds should show variance."""
        from algorithms.pca_kmeans import PCAKMeansAlgorithm
        from utils.metrics import compute_metrics

        X, labels = synthetic_scrnaseq_data

        nmis = []
        for seed in [42, 123, 456]:
            algo = PCAKMeansAlgorithm({
                'n_clusters': 3,
                'n_components': 10,
                'random_state': seed
            })

            algo.fit(X, labels)
            pred = algo.get_labels()
            metrics = compute_metrics(labels, pred)
            nmis.append(metrics['NMI'])

        # With different seeds, there might be some variance
        # but on well-structured data, results should be similar
        assert max(nmis) - min(nmis) < 0.3, \
            "NMI variance across seeds should be reasonable"
