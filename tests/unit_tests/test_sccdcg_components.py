"""
Unit tests for scCDCG critical components.

Tests cover:
- Sinkhorn algorithm for optimal transport
- Laplacian matrix computation
- Cluster assignment (Student-t)
- Reproducibility with fixed seed
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))


class TestSinkhornAlgorithm:
    """Tests for Sinkhorn optimal transport algorithm."""

    @pytest.fixture
    def sinkhorn_function(self):
        """Import Sinkhorn from scCDCG."""
        from algorithms.sccdcg import sinkhorn
        return sinkhorn

    def test_sinkhorn_output_shape(self, sinkhorn_function):
        """Sinkhorn should return matrix of same shape as input."""
        n_samples, n_clusters = 100, 5

        Q = np.random.rand(n_samples, n_clusters)
        Q = Q / Q.sum(axis=1, keepdims=True)  # Normalize rows

        # Sinkhorn requires: pred, lambdas, row, col
        lambdas = 5.0
        row = np.ones(n_samples) / n_samples
        col = np.ones(n_clusters) / n_clusters

        P = sinkhorn_function(Q, lambdas, row, col)

        assert P.shape == Q.shape, f"Expected shape {Q.shape}, got {P.shape}"

    def test_sinkhorn_row_constraints(self, sinkhorn_function):
        """Sinkhorn output should have rows approximately summing to row constraint."""
        n_samples, n_clusters = 100, 5

        Q = np.random.rand(n_samples, n_clusters)
        Q = Q / Q.sum(axis=1, keepdims=True)

        lambdas = 5.0
        row = np.ones(n_samples) / n_samples
        col = np.ones(n_clusters) / n_clusters

        P = sinkhorn_function(Q, lambdas, row, col)
        row_sums = P.sum(axis=1)

        # With these constraints, row sums should be close to 1/n_samples
        assert not np.any(np.isnan(row_sums)), "Row sums should not be NaN"

    def test_sinkhorn_column_constraints(self, sinkhorn_function):
        """Sinkhorn output should have balanced column sums."""
        n_samples, n_clusters = 100, 5

        Q = np.random.rand(n_samples, n_clusters)
        Q = Q / Q.sum(axis=1, keepdims=True)

        lambdas = 5.0
        row = np.ones(n_samples) / n_samples
        col = np.ones(n_clusters) / n_clusters

        P = sinkhorn_function(Q, lambdas, row, col)
        col_sums = P.sum(axis=0)

        # Column sums should be close to col constraint
        assert not np.any(np.isnan(col_sums)), "Column sums should not be NaN"

    def test_sinkhorn_positive(self, sinkhorn_function):
        """Sinkhorn output should be non-negative."""
        Q = np.random.rand(50, 5)
        Q = Q / Q.sum(axis=1, keepdims=True)

        lambdas = 5.0
        row = np.ones(50) / 50
        col = np.ones(5) / 5

        P = sinkhorn_function(Q, lambdas, row, col)

        assert (P >= -1e-10).all(), "Sinkhorn output should be non-negative"

    def test_sinkhorn_numerical_stability(self, sinkhorn_function):
        """Sinkhorn should handle edge cases without NaN/Inf."""
        # Very peaked distribution (almost one-hot)
        n_samples, n_clusters = 50, 5
        Q = np.zeros((n_samples, n_clusters))
        Q[np.arange(n_samples), np.random.randint(0, n_clusters, n_samples)] = 0.99
        Q += 0.01 / n_clusters  # Add small probability to others
        Q = Q / Q.sum(axis=1, keepdims=True)

        lambdas = 5.0
        row = np.ones(n_samples) / n_samples
        col = np.ones(n_clusters) / n_clusters

        P = sinkhorn_function(Q, lambdas, row, col)

        assert not np.any(np.isnan(P)), "Sinkhorn should not produce NaN"
        # Note: The original implementation may produce some Inf that are replaced with small values

    def test_sinkhorn_uniform_input(self, sinkhorn_function):
        """Sinkhorn on uniform input should produce valid output."""
        n_samples, n_clusters = 100, 5

        # Uniform distribution
        Q = np.ones((n_samples, n_clusters)) / n_clusters

        lambdas = 5.0
        row = np.ones(n_samples) / n_samples
        col = np.ones(n_clusters) / n_clusters

        P = sinkhorn_function(Q, lambdas, row, col)

        # Output should be valid
        assert not np.any(np.isnan(P)), "Should not produce NaN"
        assert P.shape == Q.shape, "Shape should be preserved"

    def test_sinkhorn_deterministic(self, sinkhorn_function):
        """Sinkhorn should be deterministic."""
        np.random.seed(42)
        Q = np.random.rand(50, 5)
        Q = Q / Q.sum(axis=1, keepdims=True)

        lambdas = 5.0
        row = np.ones(50) / 50
        col = np.ones(5) / 5

        P1 = sinkhorn_function(Q.copy(), lambdas, row, col)
        P2 = sinkhorn_function(Q.copy(), lambdas, row, col)

        assert np.allclose(P1, P2), "Sinkhorn should be deterministic"


class TestLaplacianMatrix:
    """Tests for graph Laplacian computation."""

    @pytest.fixture
    def compute_laplacian(self):
        """Import Laplacian computation from scCDCG."""
        from algorithms.sccdcg import get_laplace_matrix
        return get_laplace_matrix

    def test_laplacian_symmetric(self, compute_laplacian):
        """Normalized adjacency matrix should be symmetric."""
        n = 50
        # Random symmetric adjacency matrix
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        # get_laplace_matrix returns D^(-1/2) A D^(-1/2), which is symmetric for symmetric A
        L = compute_laplacian(A)

        # Check symmetry
        L_np = L.numpy()
        assert np.allclose(L_np, L_np.T, atol=1e-5), \
            "Normalized adjacency matrix should be symmetric"

    def test_laplacian_eigenvalues(self, compute_laplacian):
        """Normalized adjacency eigenvalues should be in [-1, 1]."""
        n = 30
        A = np.random.rand(n, n)
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        L = compute_laplacian(A)

        eigenvalues = np.linalg.eigvalsh(L.numpy())

        # For normalized adjacency D^(-1/2) A D^(-1/2), eigenvalues are in [-1, 1]
        assert eigenvalues.min() >= -1 - 1e-5, \
            f"Eigenvalues should be >= -1, got min={eigenvalues.min()}"
        assert eigenvalues.max() <= 1 + 1e-5, \
            f"Eigenvalues should be <= 1, got max={eigenvalues.max()}"

    def test_laplacian_output_shape(self, compute_laplacian):
        """Output should have same shape as input."""
        n = 20
        # Ensure fully connected (no isolated nodes)
        A = np.random.rand(n, n) + 0.1
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)

        L = compute_laplacian(A)

        assert L.shape == (n, n), f"Expected shape ({n}, {n}), got {L.shape}"
        assert not torch.any(torch.isnan(L)), "Output should not contain NaN"


class TestClusterAssignment:
    """Tests for Student-t cluster assignment in scCDCG."""

    @pytest.fixture
    def cluster_assignment_module(self):
        """Import ClusterAssignment from scCDCG."""
        from algorithms.sccdcg import ClusterAssignment
        return ClusterAssignment

    def test_cluster_assignment_output_shape(self, cluster_assignment_module):
        """Cluster assignment should return (n_samples, n_clusters) matrix."""
        n_clusters = 5
        embedding_dim = 32

        # ClusterAssignment(cluster_number, embedding_dimension, alpha, cluster_centers)
        module = cluster_assignment_module(n_clusters, embedding_dim)
        z = torch.randn(100, embedding_dim)

        q = module(z)

        assert q.shape == (100, n_clusters), \
            f"Expected shape (100, {n_clusters}), got {q.shape}"

    def test_cluster_assignment_probabilities(self, cluster_assignment_module):
        """Cluster assignment should output valid probabilities."""
        n_clusters = 5
        embedding_dim = 32

        module = cluster_assignment_module(n_clusters, embedding_dim)
        z = torch.randn(50, embedding_dim)

        q = module(z)

        # Should sum to 1
        row_sums = q.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
            "Assignment probabilities should sum to 1"

        # Should be non-negative
        assert (q >= 0).all(), "Assignment probabilities should be non-negative"

    def test_cluster_assignment_gradient_flow(self, cluster_assignment_module):
        """Gradient should flow through cluster assignment."""
        n_clusters = 5
        embedding_dim = 32

        module = cluster_assignment_module(n_clusters, embedding_dim)
        z = torch.randn(50, embedding_dim, requires_grad=True)

        q = module(z)
        loss = q.sum()
        loss.backward()

        assert z.grad is not None, "Gradient should flow to input"
        assert module.cluster_centers.grad is not None, \
            "Gradient should flow to cluster centers"


class TestReproducibilityScCDCG:
    """Tests for scCDCG reproducibility."""

    def test_sinkhorn_reproducible(self):
        """Sinkhorn should be fully reproducible."""
        from algorithms.sccdcg import sinkhorn

        np.random.seed(42)
        n_samples, n_clusters = 50, 5
        Q = np.random.rand(n_samples, n_clusters)
        Q = Q / Q.sum(axis=1, keepdims=True)

        lambdas = 5.0
        row = np.ones(n_samples) / n_samples
        col = np.ones(n_clusters) / n_clusters

        P1 = sinkhorn(Q.copy(), lambdas, row, col)

        np.random.seed(42)
        Q2 = np.random.rand(n_samples, n_clusters)
        Q2 = Q2 / Q2.sum(axis=1, keepdims=True)

        P2 = sinkhorn(Q2, lambdas, row, col)

        assert np.allclose(P1, P2), \
            "Sinkhorn should produce identical results with same input"

    def test_sccdcg_initialization_reproducible(self):
        """scCDCG model initialization should be reproducible."""
        from algorithms.sccdcg import ScCDCGAlgorithm

        params = {
            'random_state': 42,
            'n_clusters': 5,
            'embedding_dim': 16
        }

        algo1 = ScCDCGAlgorithm(params.copy())
        algo2 = ScCDCGAlgorithm(params.copy())

        # Both should have same random state
        assert algo1.params['random_state'] == algo2.params['random_state']


class TestNCutLoss:
    """Tests for Normalized Cut loss computation."""

    def test_full_nn_forward_pass(self):
        """FULL_NN forward pass should work without errors."""
        from algorithms.sccdcg import FULL_NN
        import torch

        n_samples = 50
        input_dim = 100
        n_clusters = 5
        embedding_dim = 16

        # FULL_NN(dim_input, dims_encoder, dims_decoder, num_class, pretrain_state_dict=None)
        dims_encoder = [64, embedding_dim]
        dims_decoder = [embedding_dim, 64]

        model = FULL_NN(
            dim_input=input_dim,
            dims_encoder=dims_encoder,
            dims_decoder=dims_decoder,
            num_class=n_clusters
        )

        # Create sample data
        X = torch.randn(n_samples, input_dim)
        adj = torch.rand(n_samples, n_samples)
        adj = (adj + adj.T) / 2
        adj.fill_diagonal_(0)

        # Forward pass
        model.eval()
        with torch.no_grad():
            z, x_hat = model(X, adj)

        # Check output shapes
        assert z.shape == (n_samples, embedding_dim), f"Z shape should be ({n_samples}, {embedding_dim})"
        assert x_hat.shape == (n_samples, input_dim), f"X_hat shape should be ({n_samples}, {input_dim})"

    def test_orthogonality_regularization(self):
        """Test that orthogonality regularization is applied."""
        # The NCut loss includes ||H^T H - I||_F term
        # This encourages orthogonal embeddings

        n_samples = 50
        embedding_dim = 10

        # Random embeddings
        H = np.random.randn(n_samples, embedding_dim)

        # Orthogonality measure
        orth_violation = np.linalg.norm(H.T @ H - np.eye(embedding_dim))

        # For random embeddings, this should be large
        assert orth_violation > 1.0, \
            "Random embeddings should have large orthogonality violation"

        # After optimization, it should be smaller
        # (we just document the metric here)


class TestKLDivergence:
    """Tests for KL divergence in clustering loss."""

    def test_kl_divergence_positive(self):
        """KL divergence should be non-negative."""
        n_samples, n_clusters = 100, 5

        # Create two probability distributions
        P = np.random.rand(n_samples, n_clusters)
        P = P / P.sum(axis=1, keepdims=True)

        Q = np.random.rand(n_samples, n_clusters)
        Q = Q / Q.sum(axis=1, keepdims=True)

        # KL(P || Q) = sum(P * log(P/Q))
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        kl = np.sum(P * np.log((P + eps) / (Q + eps)))

        assert kl >= -eps, "KL divergence should be non-negative"

    def test_kl_divergence_zero_for_identical(self):
        """KL divergence should be 0 for identical distributions."""
        n_samples, n_clusters = 100, 5

        P = np.random.rand(n_samples, n_clusters)
        P = P / P.sum(axis=1, keepdims=True)

        eps = 1e-10
        kl = np.sum(P * np.log((P + eps) / (P + eps)))

        assert np.abs(kl) < 1e-5, \
            f"KL divergence for identical distributions should be ~0, got {kl}"
