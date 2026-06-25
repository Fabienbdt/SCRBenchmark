"""
PCA + Leiden algorithm implementation.
Dimensionality reduction followed by graph-based community detection.
Supports automatic PCA component selection using Cattell's scree test.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import logging

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType

logger = logging.getLogger(__name__)


@AlgorithmRegistry.register
class PCALeidenAlgorithm(BaseAlgorithm):
    """
    PCA + Leiden: Graph-based clustering approach.
    Performs PCA dimensionality reduction, constructs a KNN graph,
    then applies Leiden community detection algorithm.
    Modern standard approach in scRNA-seq analysis (Seurat v3+, Scanpy).

    Supports automatic PCA component selection using Cattell's scree test
    when n_pca_components is set to 0 (auto).
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name='pca_leiden',
            display_name='PCA + Leiden',
            description='Graph-based clustering: PCA dimensionality reduction, KNN graph '
                       'construction, and Leiden community detection. Supports automatic '
                       'component selection using Cattell\'s scree test (set to 0 for auto).',
            category='graph_based',
            is_graph_based=True,
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
                tuning_guide='**Impact**: Dimensionality Reduction.\n**Details**: Keeps the top N principal components capturing the most variance. 0 uses Cattell\'s scree test to automatically find the "elbow".\n**Recommendation**: 0 (Auto) or 10-50 based on Scree plot.'
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
                name='leiden_resolution',
                display_name='Leiden Resolution',
                param_type=ParamType.FLOAT,
                default=0.0,
                description='Resolution parameter (higher = more clusters). Set to 0.0 for automatic search.',
                min_value=0.0,
                max_value=3.0,
                step=0.1,
                category='Leiden',
                tuning_guide='**Impact**: Cluster granularity.\n**Details**: Controls how coarse or fine the clusters are. Higher values (1.0-2.0) find smaller sub-populations. Lower values (0.1-0.5) find broad cell types.\n**Recommendation**: Keep at 0.0 to auto-search a silhouette-optimal resolution, or set manually.'
            ),
            HyperparameterConfig(
                name='leiden_neighbors',
                display_name='KNN Neighbors',
                param_type=ParamType.INTEGER,
                default=19,
                description='Number of neighbors for KNN graph construction',
                min_value=5,
                max_value=100,
                step=5,
                category='Leiden',
                tuning_guide='**Impact**: Graph connectivity.\n**Details**: Number of neighbors used to construct the graph. Larger values (30-50) result in a more connected graph (global view) but might merge distinct small populations. Small values (5-10) capture local structure.\n**Recommendation**: 15 is standard in Scanpy.'
            ),
            HyperparameterConfig(
                name='leiden_resolution_min',
                display_name='Leiden Res Min',
                param_type=ParamType.FLOAT,
                default=0.1,
                description='Minimum resolution explored during automatic search.',
                min_value=0.01,
                max_value=3.0,
                step=0.01,
                category='Leiden',
                advanced=True
            ),
            HyperparameterConfig(
                name='leiden_resolution_max',
                display_name='Leiden Res Max',
                param_type=ParamType.FLOAT,
                default=2.5,
                description='Maximum resolution explored during automatic search.',
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                category='Leiden',
                advanced=True
            ),
            HyperparameterConfig(
                name='leiden_resolution_step',
                display_name='Leiden Res Step',
                param_type=ParamType.FLOAT,
                default=0.1,
                description='Step size for automatic resolution search.',
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                category='Leiden',
                advanced=True
            ),
            HyperparameterConfig(
                name='leiden_silhouette_max_cells',
                display_name='Leiden Silhouette Max Cells',
                param_type=ParamType.INTEGER,
                default=5000,
                description='Maximum cells used to evaluate silhouette during auto-search.',
                min_value=500,
                max_value=50000,
                step=500,
                category='Leiden',
                advanced=True
            ),
            HyperparameterConfig(
                name='n_iterations',
                display_name='Leiden Iterations',
                param_type=ParamType.INTEGER,
                default=-1,
                description='Number of Leiden iterations (-1 = until convergence)',
                min_value=-1,
                max_value=100,
                step=1,
                category='Leiden',
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
        self.n_components_used = None
        self._explained_variance = None

    def _search_optimal_leiden_resolution(self, adata, embeddings: np.ndarray) -> float:
        """Search Leiden resolution maximizing silhouette score on PCA embeddings."""
        import scanpy as sc
        from sklearn.metrics import silhouette_score

        seed = int(self.params.get('random_state', 42))
        res_min = float(self.params.get('leiden_resolution_min', 0.1))
        res_max = float(self.params.get('leiden_resolution_max', 2.5))
        res_step = float(self.params.get('leiden_resolution_step', 0.1))
        max_cells = int(self.params.get('leiden_silhouette_max_cells', 5000))

        if res_step <= 0:
            logger.warning("PCA+Leiden: invalid leiden_resolution_step=%s. Falling back to 0.1.", res_step)
            res_step = 0.1
        if res_max < res_min:
            res_min, res_max = res_max, res_min

        resolutions = np.arange(res_min, res_max + 1e-12, res_step)
        if len(resolutions) == 0:
            logger.warning("PCA+Leiden: empty resolution grid. Falling back to 1.0.")
            return 1.0

        X = np.asarray(embeddings)
        if X.ndim != 2 or X.shape[0] < 3:
            logger.warning("PCA+Leiden: not enough samples for silhouette search. Falling back to 1.0.")
            return 1.0

        eval_idx = np.arange(X.shape[0])
        if max_cells > 0 and X.shape[0] > max_cells:
            rng = np.random.default_rng(seed)
            eval_idx = np.sort(rng.choice(X.shape[0], size=max_cells, replace=False))

        n_iterations = 2 if self.params.get('n_iterations', -1) == -1 else self.params.get('n_iterations', -1)
        best_res = None
        best_score = -np.inf
        best_clusters = 0

        for res in resolutions:
            sc.tl.leiden(
                adata,
                resolution=float(res),
                n_iterations=n_iterations,
                flavor="igraph",
                directed=False,
                random_state=seed,
                key_added="_leiden_search",
            )
            labels_full = adata.obs["_leiden_search"].astype(int).to_numpy()
            labels_eval = labels_full[eval_idx]
            if np.unique(labels_eval).size < 2:
                continue
            try:
                score = float(silhouette_score(X[eval_idx], labels_eval, metric="euclidean"))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_res = float(res)
                best_clusters = int(np.unique(labels_full).size)

        if "_leiden_search" in adata.obs:
            del adata.obs["_leiden_search"]

        if best_res is None:
            logger.warning("PCA+Leiden: no valid silhouette score found. Falling back to 1.0.")
            return 1.0

        logger.info(
            "PCA+Leiden: selected resolution %.3f (silhouette=%.4f, clusters=%d).",
            best_res,
            best_score,
            best_clusters,
        )
        return best_res

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'PCALeidenAlgorithm':
        """
        Fit PCA + Leiden model.

        Args:
            data: AnnData object or numpy array with preprocessed data
            labels: Optional ground truth labels (not used by Leiden)
        """
        from sklearn.decomposition import PCA
        import scanpy as sc
        import anndata
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
            adata = data.copy() if isinstance(data, anndata.AnnData) else None
        else:
            X = data
            adata = None

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
            # Store PCA for reference
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

        # Create or update AnnData for scanpy
        if adata is None:
            adata = anndata.AnnData(X)

        adata.obsm['X_pca'] = self._embeddings

        # Compute neighbors
        # Use method='gauss' to match original script behavior
        sc.pp.neighbors(
            adata,
            n_neighbors=self.params.get('leiden_neighbors', 19),
            use_rep='X_pca',
            method='gauss',
            random_state=self.params.get('random_state', 42)
        )

        # Leiden clustering
        # Use flavor='igraph' to avoid FutureWarning and n_iterations=2 as recommended.
        # If resolution is 0.0, search for a silhouette-optimal resolution.
        n_iterations = 2 if self.params.get('n_iterations', -1) == -1 else self.params.get('n_iterations', -1)
        leiden_resolution = float(self.params.get('leiden_resolution', 0.0))
        if leiden_resolution > 0:
            resolution_used = leiden_resolution
            selection_mode = "manual"
        else:
            resolution_used = float(self._search_optimal_leiden_resolution(adata, self._embeddings))
            selection_mode = "auto_silhouette"

        sc.tl.leiden(
            adata,
            resolution=resolution_used,
            n_iterations=n_iterations,
            flavor="igraph",
            directed=False,
            random_state=self.params.get('random_state', 42)
        )

        self.params['leiden_resolution_selected'] = float(resolution_used)
        self.params['leiden_resolution_selection_mode'] = selection_mode
        self._labels = adata.obs['leiden'].astype(int).values

        self._fitted = True
        return self

    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels.

        For new data, uses KNN to assign clusters based on training embeddings.
        This is necessary because Leiden is graph-based and doesn't natively
        support prediction on new data.

        Args:
            data: Optional new data to predict on

        Returns:
            Predicted cluster labels
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if data is None:
            return self._labels

        # For new data, use KNN based on PCA embeddings
        from sklearn.neighbors import KNeighborsClassifier

        # Transform new data with PCA
        if hasattr(data, 'X'):
            X = data.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        new_embeddings = self.pca.transform(X)

        # Use KNN to assign clusters based on training labels
        # Fit KNN on training embeddings if not already done
        if not hasattr(self, '_knn') or self._knn is None:
            n_neighbors = min(15, len(self._labels))
            if n_neighbors < 1:
                raise RuntimeError("Not enough training samples to run KNN prediction")
            self._knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            self._knn.fit(self._embeddings, self._labels)

        return self._knn.predict(new_embeddings)

    def encode(self, data: Any) -> np.ndarray:
        """
        Encode data into PCA space.

        Args:
            data: Data to encode (AnnData or numpy array)

        Returns:
            PCA embeddings
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before encoding")

        if hasattr(data, 'X'):
            X = data.X
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        return self.pca.transform(X)

    def get_explained_variance(self) -> Optional[np.ndarray]:
        """Get explained variance ratio from PCA."""
        return self._explained_variance

    def get_n_components_used(self) -> Optional[int]:
        """Get the actual number of PCA components used."""
        return self.n_components_used
