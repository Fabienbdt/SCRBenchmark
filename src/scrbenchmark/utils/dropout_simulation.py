"""
Dropout simulation for scRNA-seq data.

This module implements dropout simulation methods as described in the
scDeepCluster paper (Nature Machine Intelligence, 2019).
"""

import numpy as np
from scipy.special import expit


def simulate_dropout_logistic(X, dropout_mid=0, dropout_shape=-1):
    """
    Simulate dropout events using a logistic function.

    This method is based on the Splatter simulation approach used in the
    scDeepCluster paper.

    Args:
        X: True count matrix (cells x genes)
        dropout_mid: Midpoint parameter of the dropout logistic function.
            Controls the overall dropout rate.
            - dropout_mid = -0.5 -> ~12% dropout
            - dropout_mid = 0    -> ~17% dropout (default in Splatter)
            - dropout_mid = 0.5  -> ~23% dropout
            - dropout_mid = 1    -> ~30% dropout
        dropout_shape: Shape parameter of the dropout logistic function (default: -1)
            Controls the steepness of the dropout probability curve

    Returns:
        X_dropout: Count matrix with dropout events applied
        dropout_mask: Boolean mask indicating which values were dropped out
        dropout_rate: Actual dropout rate achieved
    """
    X_dropout = X.copy()

    # Calculate the mean expression for each gene
    gene_means = np.mean(X, axis=0)

    # Avoid log(0) by adding a small constant
    log_means = np.log(gene_means + 1e-10)

    # Calculate dropout probability for each gene using logistic function
    dropout_prob = expit(dropout_shape * (log_means - dropout_mid))

    # Initialize dropout mask
    dropout_mask = np.zeros_like(X, dtype=bool)

    # Vectorized dropout application for efficiency
    random_vals = np.random.random(X.shape)
    for j in range(X.shape[1]):
        mask_col = (X[:, j] > 0) & (random_vals[:, j] < dropout_prob[j])
        dropout_mask[:, j] = mask_col
        X_dropout[mask_col, j] = 0

    # Calculate actual dropout rate
    true_nonzero = np.sum(X > 0)
    new_zeros = np.sum(dropout_mask)
    dropout_rate = new_zeros / true_nonzero if true_nonzero > 0 else 0

    return X_dropout, dropout_mask, dropout_rate


def simulate_dropout_simple(X, dropout_rate=0.2):
    """
    Simple dropout simulation: randomly set non-zero values to zero.

    Args:
        X: Count matrix (cells x genes)
        dropout_rate: Target proportion of non-zero values to drop (0 to 1)

    Returns:
        X_dropout: Count matrix with dropout events applied
        dropout_mask: Boolean mask indicating which values were dropped out
        actual_rate: Actual dropout rate achieved
    """
    X_dropout = X.copy()
    dropout_mask = np.zeros_like(X, dtype=bool)

    # Find non-zero entries
    nonzero_mask = X > 0
    nonzero_indices = np.where(nonzero_mask)
    n_nonzero = len(nonzero_indices[0])

    if n_nonzero == 0:
        return X_dropout, dropout_mask, 0.0

    # Randomly select entries to dropout
    n_dropout = int(n_nonzero * dropout_rate)
    dropout_indices = np.random.choice(n_nonzero, size=n_dropout, replace=False)

    # Apply dropout
    for idx in dropout_indices:
        i, j = nonzero_indices[0][idx], nonzero_indices[1][idx]
        X_dropout[i, j] = 0
        dropout_mask[i, j] = True

    actual_rate = n_dropout / n_nonzero if n_nonzero > 0 else 0

    return X_dropout, dropout_mask, actual_rate


def add_gaussian_noise(X, noise_level=1.0):
    """
    Add Gaussian noise to the data.

    Args:
        X: Input data matrix
        noise_level: Standard deviation of the Gaussian noise (default: 1.0)

    Returns:
        X_noisy: Data with Gaussian noise added
    """
    noise = np.random.normal(0, noise_level, size=X.shape)
    X_noisy = X + noise
    # Ensure non-negative values for count data
    X_noisy = np.maximum(X_noisy, 0)
    return X_noisy


def apply_dropout(X, method='none', dropout_rate=0.2, dropout_mid=0.0,
                  dropout_shape=-1, noise_level=0.0, random_state=None):
    """
    Apply dropout and/or noise to data.

    This is the main entry point for dropout simulation in the preprocessing pipeline.

    Args:
        X: Input count matrix (cells x genes)
        method: Dropout method - 'none', 'simple', 'logistic'
        dropout_rate: For 'simple' method, target dropout rate (0 to 1)
        dropout_mid: For 'logistic' method, midpoint parameter
        dropout_shape: For 'logistic' method, shape parameter
        noise_level: Gaussian noise standard deviation (0 = no noise)
        random_state: Random seed for reproducibility

    Returns:
        X_modified: Modified data matrix
        info: Dictionary with dropout statistics
    """
    if random_state is not None:
        np.random.seed(random_state)

    info = {
        'method': method,
        'dropout_rate_target': dropout_rate if method == 'simple' else None,
        'dropout_mid': dropout_mid if method == 'logistic' else None,
        'dropout_shape': dropout_shape if method == 'logistic' else None,
        'noise_level': noise_level,
        'actual_dropout_rate': 0.0,
        'n_values_dropped': 0,
        'n_nonzero_original': int(np.sum(X > 0))
    }

    X_modified = X.copy()

    # Apply dropout
    if method == 'simple' and dropout_rate > 0:
        X_modified, dropout_mask, actual_rate = simulate_dropout_simple(
            X_modified, dropout_rate=dropout_rate
        )
        info['actual_dropout_rate'] = actual_rate
        info['n_values_dropped'] = int(np.sum(dropout_mask))

    elif method == 'logistic':
        X_modified, dropout_mask, actual_rate = simulate_dropout_logistic(
            X_modified, dropout_mid=dropout_mid, dropout_shape=dropout_shape
        )
        info['actual_dropout_rate'] = actual_rate
        info['n_values_dropped'] = int(np.sum(dropout_mask))

    # Apply Gaussian noise if requested
    if noise_level > 0:
        X_modified = add_gaussian_noise(X_modified, noise_level=noise_level)

    info['n_nonzero_final'] = int(np.sum(X_modified > 0))
    info['sparsity_original'] = 1.0 - (info['n_nonzero_original'] / X.size)
    info['sparsity_final'] = 1.0 - (info['n_nonzero_final'] / X_modified.size)

    return X_modified, info
