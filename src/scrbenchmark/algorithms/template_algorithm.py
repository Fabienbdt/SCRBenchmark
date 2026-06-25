"""
Template for creating new algorithms.

To add a new algorithm:
1. Copy this file and rename it (e.g., my_algorithm.py)
2. Implement all abstract methods
3. Register the algorithm using the @AlgorithmRegistry.register decorator

The framework will automatically discover and load your algorithm class at runtime.
The module filename must not start with "template"; those files are skipped.
If an optional dependency is heavy or may be absent, import it inside fit() so a
missing dependency does not hide every other algorithm from the CLI/UI.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType


# Uncomment the decorator below to register this algorithm
# @AlgorithmRegistry.register
class TemplateAlgorithm(BaseAlgorithm):
    """
    Template algorithm for clustering.

    Replace this docstring with a description of your algorithm.
    Include key features, theoretical background, and references.
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        """
        Return information about this algorithm.

        Modify the values below to match your algorithm.
        """
        return AlgorithmInfo(
            name='template',  # Unique identifier (lowercase, no spaces)
            display_name='Template Algorithm',  # Display name in UI
            description='A template algorithm for demonstration. '
                       'Replace with your algorithm description.',
            category='classical',  # Options: 'deep_learning', 'classical', 'graph_based', 'other'
            requires_gpu=False,  # Set True if GPU is required
            supports_labels=True,  # Set False if labels are not used
            preprocessing_notes='Describe expected input data for the UI.',
            has_internal_preprocessing=False,
            recommended_data='preprocessed',  # 'raw', 'preprocessed', or 'either'
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        """
        Define the hyperparameters for this algorithm.

        Add all configurable parameters here.
        """
        return [
            # Integer parameter example
            HyperparameterConfig(
                name='n_clusters',
                display_name='Number of Clusters',
                param_type=ParamType.INTEGER,
                default=8,
                description='Number of clusters to find',
                min_value=2,
                max_value=100,
                step=1,
                category='Clustering'  # Group in UI
            ),

            # Float parameter example
            HyperparameterConfig(
                name='learning_rate',
                display_name='Learning Rate',
                param_type=ParamType.FLOAT,
                default=0.001,
                description='Learning rate for optimization',
                min_value=0.0001,
                max_value=1.0,
                step=0.0001,
                category='Training'
            ),

            # Choice parameter example
            HyperparameterConfig(
                name='method',
                display_name='Method',
                param_type=ParamType.CHOICE,
                default='auto',
                description='Method for clustering',
                choices=['auto', 'fast', 'accurate'],
                category='Algorithm'
            ),

            # Boolean parameter example
            HyperparameterConfig(
                name='use_preprocessing',
                display_name='Use Preprocessing',
                param_type=ParamType.BOOLEAN,
                default=True,
                description='Whether to apply preprocessing',
                category='Preprocessing'
            ),

            # String parameter example
            HyperparameterConfig(
                name='custom_param',
                display_name='Custom Parameter',
                param_type=ParamType.STRING,
                default='default_value',
                description='A custom string parameter',
                category='Advanced',
                advanced=True  # Hidden in advanced section
            ),

            # Random seed (recommended for reproducibility)
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
        ]

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the algorithm.

        Args:
            params: Dictionary of hyperparameters
        """
        super().__init__(params)
        # Initialize any additional attributes here
        self.model = None

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'TemplateAlgorithm':
        """
        Fit the algorithm to the data.

        Args:
            data: Input data (AnnData object or numpy array)
            labels: Optional ground truth labels

        Returns:
            self for method chaining
        """
        # Extract data matrix
        if hasattr(data, 'X'):
            X = data.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        # Get parameters
        n_clusters = self.params.get('n_clusters', 8)
        random_state = self.params.get('random_state', 42)
        self._set_seed(int(random_state))

        # If n_clusters is 0, try to infer from labels
        if n_clusters == 0 and labels is not None:
            n_clusters = len(np.unique(labels))

        # ===========================================
        # YOUR ALGORITHM IMPLEMENTATION HERE
        # ===========================================

        # Example: Simple K-Means
        from sklearn.cluster import KMeans

        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )

        self._labels = self.model.fit_predict(X)
        self._embeddings = X  # Or your algorithm's embeddings

        # ===========================================
        # END OF IMPLEMENTATION
        # ===========================================

        self._fitted = True
        return self

    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels.

        Args:
            data: Optional new data to predict

        Returns:
            Predicted cluster labels
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if data is not None:
            # Handle new data prediction
            if hasattr(data, 'X'):
                X = data.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
            else:
                X = data

            return self.model.predict(X)

        return self._labels

    # Optional: Add any additional methods specific to your algorithm
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers if available.

        Returns:
            Cluster centers or None
        """
        if self.model is not None and hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None
