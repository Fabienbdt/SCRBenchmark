#!/usr/bin/env python3
"""
Quick comparison tests with minimal epochs to verify implementations produce similar results.
"""

import sys
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "scrbenchmark"))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def generate_synthetic_data(n_cells=300, n_genes=200, n_clusters=5, seed=42):
    """Generate synthetic scRNA-seq data with clear cluster structure."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    centers = np.random.randn(n_clusters, n_genes) * 3
    X = []
    labels = []

    for i in range(n_clusters):
        n_per = n_cells // n_clusters
        data = centers[i] + np.random.randn(n_per, n_genes) * 0.5
        X.append(data)
        labels.extend([i] * n_per)

    X = np.vstack(X)
    labels = np.array(labels)

    # Shuffle
    idx = np.random.permutation(len(labels))
    X = X[idx]
    labels = labels[idx]

    # Make non-negative (count-like)
    X = np.maximum(X, 0)
    X = (X * 10).astype(np.float32)

    return X, labels, n_clusters


class SimpleData:
    """Simple data wrapper that mimics AnnData interface."""
    def __init__(self, X, gene_names=None):
        self.X = X
        if gene_names is None:
            gene_names = np.array([f'gene_{i}' for i in range(X.shape[1])])
        self.var_names = type('obj', (object,), {'values': gene_names})()
        self.layers = {'original_X': X.copy()}


def test_scdeepcluster(X, labels, n_clusters):
    """Test scDeepCluster with minimal epochs."""
    print("\n" + "="*60)
    print("Testing scDeepCluster")
    print("="*60)

    try:
        from algorithms.scdeepcluster import ScDeepClusterAlgorithm

        algo = ScDeepClusterAlgorithm({
            'n_clusters': n_clusters,
            'pretrain_epochs': 5,  # Minimal
            'maxiter': 10,         # Minimal
            'batch_size': 64,
            'z_dim': 32,
            'random_state': 42
        })

        data = SimpleData(X)
        pred = algo.fit_predict(data, labels)

        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)

        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  Status: {'PASS' if ari > 0.1 else 'LOW (expected with few epochs)'}")

        return {'ari': ari, 'nmi': nmi, 'status': 'success'}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'status': 'error', 'error': str(e)}


def test_sccdcg(X, labels, n_clusters):
    """Test scCDCG with minimal epochs."""
    print("\n" + "="*60)
    print("Testing scCDCG")
    print("="*60)

    try:
        from algorithms.sccdcg import ScCDCGAlgorithm

        algo = ScCDCGAlgorithm({
            'n_clusters': n_clusters,
            'pretrain_epochs': 5,  # Minimal
            'train_epochs': 10,    # Minimal
            'random_state': 42
        })

        data = SimpleData(X)
        pred = algo.fit_predict(data, labels)

        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)

        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  Status: {'PASS' if ari > 0.1 else 'LOW (expected with few epochs)'}")

        return {'ari': ari, 'nmi': nmi, 'status': 'success'}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'status': 'error', 'error': str(e)}


def test_scmae(X, labels, n_clusters):
    """Test scMAE with minimal epochs."""
    print("\n" + "="*60)
    print("Testing scMAE")
    print("="*60)

    try:
        from algorithms.sc_mae import ScMAEAlgorithm

        algo = ScMAEAlgorithm({
            'n_clusters': n_clusters,
            'max_epochs': 10,      # Minimal
            'masking_rate': 0.5,
            'random_state': 42
        })

        data = SimpleData(X)
        pred = algo.fit_predict(data, labels)

        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)

        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  Status: {'PASS' if ari > 0.1 else 'LOW (expected with few epochs)'}")

        return {'ari': ari, 'nmi': nmi, 'status': 'success'}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'status': 'error', 'error': str(e)}


def test_scname(X, labels, n_clusters):
    """Test scNAME with minimal epochs."""
    print("\n" + "="*60)
    print("Testing scNAME")
    print("="*60)

    try:
        from algorithms.scname import ScNAMEAlgorithm

        algo = ScNAMEAlgorithm({
            'n_clusters': n_clusters,
            'pretrain_epochs': 5,   # Minimal
            'finetrain_epochs': 10, # Minimal
            'batch_size': 64,
            'random_state': 42
        })

        data = SimpleData(X)
        pred = algo.fit_predict(data, labels)

        ari = adjusted_rand_score(labels, pred)
        nmi = normalized_mutual_info_score(labels, pred)

        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  Status: {'PASS' if ari > 0.1 else 'LOW (expected with few epochs)'}")

        return {'ari': ari, 'nmi': nmi, 'status': 'success'}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'status': 'error', 'error': str(e)}



def main():
    print("="*60)
    print("QUICK COMPARISON TESTS - MINIMAL EPOCHS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Generate test data
    print("\nGenerating synthetic data...")
    X, labels, n_clusters = generate_synthetic_data(
        n_cells=300,
        n_genes=200,
        n_clusters=5,
        seed=42
    )
    print(f"  Data shape: {X.shape}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Labels distribution: {np.bincount(labels)}")

    results = {}

    # Run all tests
    results['scdeepcluster'] = test_scdeepcluster(X, labels, n_clusters)
    results['sccdcg'] = test_sccdcg(X, labels, n_clusters)
    results['scmae'] = test_scmae(X, labels, n_clusters)
    results['scname'] = test_scname(X, labels, n_clusters)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n| Algorithm     | Status  | ARI    | NMI    |")
    print("|---------------|---------|--------|--------|")

    for algo, res in results.items():
        if res['status'] == 'success':
            print(f"| {algo:13} | SUCCESS | {res['ari']:.4f} | {res['nmi']:.4f} |")
        else:
            print(f"| {algo:13} | ERROR   | -      | -      |")

    print("\nNote: Low ARI/NMI values are expected with minimal epochs.")
    print("The goal is to verify that the code runs without errors.")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
