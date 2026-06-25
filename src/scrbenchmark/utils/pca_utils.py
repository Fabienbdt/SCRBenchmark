"""
PCA utilities including Cattell's scree test for automatic component selection.

Reference:
    Cattell, R. B. (1966). The scree test for the number of factors.
    Multivariate Behavioral Research, 1(2), 245-276.
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.decomposition import PCA


def find_elbow_cattell(explained_variance: np.ndarray) -> int:
    """
    Find the elbow point using Cattell's scree test method.

    This implements the geometric method: find the point that maximizes
    the distance from the line connecting the first and last points.

    Args:
        explained_variance: Array of explained variance ratios

    Returns:
        elbow_index: Index of the elbow point (0-based)
    """
    n_points = len(explained_variance)
    if n_points < 3:
        return 0

    # Coordinates of all points
    coords = np.column_stack([
        np.arange(n_points),
        explained_variance
    ])

    # Normalize coordinates to [0, 1] for fair distance calculation
    coords_norm = coords.copy()
    coords_norm[:, 0] = coords[:, 0] / (n_points - 1)
    var_range = coords[:, 1].max() - coords[:, 1].min()
    if var_range > 1e-10:
        coords_norm[:, 1] = (coords[:, 1] - coords[:, 1].min()) / var_range
    else:
        coords_norm[:, 1] = 0

    # Line from first to last point
    p1 = coords_norm[0]
    p2 = coords_norm[-1]

    # Vector from p1 to p2
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)

    if line_len < 1e-10:
        return 0

    line_unit = line_vec / line_len

    # Calculate perpendicular distance from each point to the line
    distances = np.zeros(n_points)
    for i in range(n_points):
        point_vec = coords_norm[i] - p1
        # Project point onto line
        proj_length = np.dot(point_vec, line_unit)
        proj_point = p1 + proj_length * line_unit
        # Distance from point to projection
        distances[i] = np.linalg.norm(coords_norm[i] - proj_point)

    # The elbow is the point with maximum distance
    elbow_index = np.argmax(distances)

    return elbow_index


def find_elbow_second_derivative(explained_variance: np.ndarray) -> int:
    """
    Find the elbow point using second derivative method.

    The elbow is where the rate of decrease changes most significantly.

    Args:
        explained_variance: Array of explained variance ratios

    Returns:
        elbow_index: Index of the elbow point (0-based)
    """
    if len(explained_variance) < 4:
        return 0

    # First derivative (rate of change)
    first_derivative = np.diff(explained_variance)

    # Second derivative (acceleration)
    second_derivative = np.diff(first_derivative)

    # The elbow is where second derivative is maximum (most positive change in slope)
    # Add 1 because we lost one point with diff
    elbow_index = np.argmax(second_derivative) + 1

    return elbow_index


def compute_optimal_pca_components(
    X: np.ndarray,
    max_components: int = 100,
    method: str = 'cattell',
    min_components: int = 2,
    max_components_limit: Optional[int] = None
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Compute the optimal number of PCA components using Cattell's scree test.

    Args:
        X: Data matrix (n_samples, n_features)
        max_components: Maximum number of components to compute for analysis
        method: Method for elbow detection ('cattell' or 'derivative')
        min_components: Minimum number of components to return
        max_components_limit: Maximum number of components to return

    Returns:
        n_components: Optimal number of components
        explained_variance: Explained variance ratio for each component
        cumulative_variance: Cumulative explained variance
    """
    # Limit max_components to data dimensions
    n_components = min(max_components, X.shape[0] - 1, X.shape[1])

    # Compute PCA
    pca = PCA(n_components=n_components)
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Find elbow
    if method == 'cattell':
        elbow_index = find_elbow_cattell(explained_variance)
    elif method == 'derivative':
        elbow_index = find_elbow_second_derivative(explained_variance)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cattell' or 'derivative'.")

    # Convert to 1-based and apply limits
    optimal_n = elbow_index + 1
    optimal_n = max(optimal_n, min_components)

    if max_components_limit is not None:
        optimal_n = min(optimal_n, max_components_limit)

    return optimal_n, explained_variance, cumulative_variance


def get_pca_with_auto_components(
    X: np.ndarray,
    n_components: int = 0,
    random_state: int = 42,
    max_components_for_elbow: int = 100
) -> Tuple[np.ndarray, int, Optional[np.ndarray]]:
    """
    Perform PCA with automatic component selection if n_components is 0 or 'auto'.

    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components (0 or negative = auto)
        random_state: Random seed for reproducibility
        max_components_for_elbow: Max components to analyze for elbow detection

    Returns:
        X_pca: Transformed data
        n_components_used: Actual number of components used
        explained_variance: Explained variance ratios (if auto mode)
    """
    explained_variance = None

    if n_components <= 0:
        # Auto mode: use Cattell's scree test
        optimal_n, explained_variance, _ = compute_optimal_pca_components(
            X,
            max_components=max_components_for_elbow,
            method='cattell',
            min_components=2
        )
        n_components = optimal_n

    # Ensure n_components is within valid range
    # PCA requires at least 1 component and cannot exceed min(n_samples, n_features)
    # Note: rank is typically min(n_samples-1, n_features) after centering
    max_possible = min(X.shape[0], X.shape[1])
    if max_possible < 1:
         raise ValueError(f"Data has invalid dimensions for PCA: {X.shape}")
         
    # Sklearn PCA handles min(n_samples, n_features), but for meaningful variance 
    # we usually want min(n_samples-1, n_features).
    # Being safe:
    upper_bound = min(X.shape[0], X.shape[1])
    n_components = min(n_components, upper_bound)
    
    # Force at least 1 component
    n_components = max(1, n_components)

    # Perform PCA with selected number of components
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    if explained_variance is None:
        explained_variance = pca.explained_variance_ratio_

    return X_pca, n_components, explained_variance
