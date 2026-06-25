"""
PCA + K-Means algorithm implementation.
Classical dimensionality reduction followed by centroid-based clustering.
Supports automatic PCA component selection using Cattell's scree test.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType


@AlgorithmRegistry.register
class PCAKMeansAlgorithm(BaseAlgorithm):
    """
    PCA + K-Means: Classical clustering approach.
    Performs PCA dimensionality reduction followed by K-Means clustering.
    Standard approach used in Seurat and Scanpy.

    Supports automatic PCA component selection using Cattell's scree test
    when n_pca_components is set to 0 (auto).
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name='pca_kmeans',
            display_name='PCA + K-Means',
            description='Classical approach: PCA dimensionality reduction followed by '
                       'K-Means clustering. Supports automatic component selection '
                       'using Cattell\'s scree test (set PCA components to 0 for auto).',
            category='classical',
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes='Expects preprocessed data (normalized, log-transformed, scaled). '
                               'Runs on Highly Variable Genes (HVGs).',
            has_internal_preprocessing=False,
            recommended_data='preprocessed'
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        return [
            HyperparameterConfig(
                name='n_pca_components',
                display_name='PCA Components',
                param_type=ParamType.INTEGER,
                default=0,
                description='Number of principal components (0 = auto using Cattell\'s scree test)',
                min_value=0,
                max_value=100,
                step=1,
                category='PCA',
                tuning_guide='**Impact**: Dimensionality Reduction.\n**Details**: Keeps the top N principal components capturing the most variance. 0 uses Cattell\'s scree test to automatically find the "elbow" where variance drops off (separating signal from noise). \n**Recommendation**: 0 (Auto) or 10-50 based on Scree plot.'
            ),
            HyperparameterConfig(
                name='pca_elbow_method',
                display_name='Elbow Detection Method',
                param_type=ParamType.CHOICE,
                default='cattell',
                description='Method for automatic PCA component detection (only used when auto)',
                choices=['cattell', 'derivative'],
                category='PCA',
                advanced=True
            ),
            HyperparameterConfig(
                name='n_clusters',
                display_name='Number of Clusters',
                param_type=ParamType.INTEGER,
                default=8,
                description='Number of clusters (required for K-Means)',
                min_value=2,
                max_value=100,
                step=1,
                category='Clustering'
            ),
            HyperparameterConfig(
                name='use_ground_truth_k',
                display_name='Use Ground Truth K (Oracle)',
                param_type=ParamType.BOOLEAN,
                default=False,
                description='ORACLE MODE: Use number of clusters from ground truth labels. '
                           'This leaks test information and should NOT be used for fair benchmarking.',
                category='Clustering',
                tuning_guide='**WARNING**: This option uses information from ground truth labels. '
                            'For fair benchmarking, set n_clusters manually based on prior knowledge.'
            ),
            HyperparameterConfig(
                name='kmeans_n_init',
                display_name='K-Means Initializations',
                param_type=ParamType.INTEGER,
                default=20,
                description='Number of K-Means initializations',
                min_value=1,
                max_value=50,
                step=1,
                category='Clustering',
                advanced=True,
                tuning_guide='**Impact**: Clustering robustness.\n**Details**: K-Means is sensitive to initialization seeds. Running multiple times (n_init) and keeping the best inertia avoids bad local optima.\n**Recommendation**: 20 is safe.'
            ),
            HyperparameterConfig(
                name='kmeans_max_iter',
                display_name='K-Means Max Iterations',
                param_type=ParamType.INTEGER,
                default=300,
                description='Maximum K-Means iterations',
                min_value=100,
                max_value=1000,
                step=100,
                category='Clustering',
                advanced=True
            ),
            HyperparameterConfig(
                name='random_state',
                display_name='Random State',
                param_type=ParamType.INTEGER,
                default=42,
                description='Random seed for reproducibility',
                min_value=0,
                max_value=99999,
                step=1,
                category='General'
            ),
            HyperparameterConfig(
                name='use_raw_data',
                display_name='Use Raw Data',
                param_type=ParamType.BOOLEAN,
                default=False,
                description='Use original unprocessed data instead of preprocessed data.',
                category='Data'
            ),
        ]

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.pca = None
        self.kmeans = None
        self.n_components_used = None
        self._explained_variance = None

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'PCAKMeansAlgorithm':
        """
        Fit PCA + K-Means model.

        Args:
            data: AnnData object or numpy array with preprocessed data
            labels: Optional ground truth labels
        """
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from utils.pca_utils import get_pca_with_auto_components

        # Get data matrix (optionally use raw data)
        use_raw_data = self.params.get('use_raw_data', False)
        if hasattr(data, 'X'):
            if use_raw_data and hasattr(data, 'layers') and 'original_X' in data.layers:
                X = data.layers['original_X']
            else:
                X = data.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        # Determine number of clusters
        n_clusters = self.params.get('n_clusters', 8)
        use_ground_truth_k = self.params.get('use_ground_truth_k', False)

        if use_ground_truth_k:
            if labels is not None:
                n_clusters = len(np.unique(labels))
                logger.warning(f"ORACLE MODE ENABLED: PCA+KMeans using ground truth k={n_clusters}. "
                              f"Results are NOT comparable to methods that estimate k independently.")
            else:
                raise ValueError("use_ground_truth_k=True but no labels provided. "
                               "Either provide labels or set use_ground_truth_k=False.")

        # Get PCA components parameter
        n_pca_components = self.params.get('n_pca_components', 0)

        # PCA with auto or manual component selection
        if n_pca_components == 0:
            # Auto mode: use Cattell's scree test
            self._embeddings, self.n_components_used, self._explained_variance = \
                get_pca_with_auto_components(
                    X,
                    n_components=0,  # Auto
                    random_state=self.params.get('random_state', 42),
                    max_components_for_elbow=100
                )
            # Store PCA for later use
            self.pca = PCA(
                n_components=self.n_components_used,
                random_state=self.params.get('random_state', 42)
            )
            self.pca.fit(X)
        else:
            # Manual mode
            n_components = min(n_pca_components, X.shape[0] - 1, X.shape[1])
            self.pca = PCA(
                n_components=n_components,
                random_state=self.params.get('random_state', 42)
            )
            self._embeddings = self.pca.fit_transform(X)
            self.n_components_used = n_components
            self._explained_variance = self.pca.explained_variance_ratio_

        # K-Means
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=self.params.get('kmeans_n_init', 10),
            max_iter=self.params.get('kmeans_max_iter', 300),
            random_state=self.params.get('random_state', 42)
        )
        self._labels = self.kmeans.fit_predict(self._embeddings)

        self._fitted = True
        return self

    def predict(self, data: Any = None) -> Any:
        """Return predicted cluster labels."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if data is not None:
            # Transform new data
            if hasattr(data, 'X'):
                X = data.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
            else:
                X = data

            embeddings = self.pca.transform(X)
            return self.kmeans.predict(embeddings)

        return self._labels

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """Get explained variance ratio from PCA."""
        return self._explained_variance

    def get_n_components_used(self) -> Optional[int]:
        """Get the actual number of PCA components used."""
        return self.n_components_used

    def encode(self, data: Any) -> Optional[np.ndarray]:
        """
        Encode data using the fitted PCA model.

        This method enables Silhouette score computation for test data
        by providing embeddings in the PCA-transformed space.

        Args:
            data: AnnData object or numpy array

        Returns:
            PCA-transformed embeddings or None if not fitted
        """
        if not self._fitted or self.pca is None:
            return None

        # Get data matrix
        use_raw_data = self.params.get('use_raw_data', False)
        if hasattr(data, 'X'):
            if use_raw_data and hasattr(data, 'layers') and 'original_X' in data.layers:
                X = data.layers['original_X']
            else:
                X = data.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        # Check if this is the training data (same number of samples)
        if X.shape[0] == len(self._embeddings):
            return self._embeddings

        # Transform new data
        return self.pca.transform(X)
