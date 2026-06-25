"""
PCA followed by a configurable clustering method.

This covers the classical methods cited in the report Table 1 with a single
representation step: PCA, then K-Means, Louvain, Leiden, or HDBSCAN.
"""

from typing import Any, Dict, List, Optional
import logging

import numpy as np

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType

logger = logging.getLogger(__name__)


@AlgorithmRegistry.register
class PCAClusteringAlgorithm(BaseAlgorithm):
    """PCA representation with selectable downstream clustering."""

    CLUSTERING_METHODS = ["kmeans", "louvain", "leiden", "hdbscan"]

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name="pca",
            display_name="PCA + clustering",
            description=(
                "Linear dimensionality reduction followed by a configurable "
                "clustering method: K-Means, Louvain, Leiden, or HDBSCAN."
            ),
            category="classical",
            is_graph_based=True,
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes=(
                "Expects preprocessed data (normalized, log-transformed, scaled). "
                "Use the clustering_method parameter to reproduce the classical "
                "clustering rows from the report."
            ),
            has_internal_preprocessing=False,
            recommended_data="preprocessed",
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        return [
            HyperparameterConfig(
                name="clustering_method",
                display_name="Clustering Method",
                param_type=ParamType.CHOICE,
                default="leiden",
                description="Downstream clustering tool applied to the PCA embedding.",
                choices=cls.CLUSTERING_METHODS,
                category="Clustering",
            ),
            HyperparameterConfig(
                name="n_pca_components",
                display_name="PCA Components",
                param_type=ParamType.INTEGER,
                default=0,
                description="Number of principal components (0 = auto using scree elbow).",
                min_value=0,
                max_value=100,
                step=1,
                category="PCA",
            ),
            HyperparameterConfig(
                name="pca_elbow_method",
                display_name="Elbow Detection Method",
                param_type=ParamType.CHOICE,
                default="cattell",
                description="Automatic PCA component detection method.",
                choices=["cattell", "derivative"],
                category="PCA",
                advanced=True,
            ),
            HyperparameterConfig(
                name="n_clusters",
                display_name="Number of Clusters",
                param_type=ParamType.INTEGER,
                default=8,
                description="Number of clusters for K-Means.",
                min_value=2,
                max_value=100,
                step=1,
                category="K-Means",
            ),
            HyperparameterConfig(
                name="use_ground_truth_k",
                display_name="Use Ground Truth K (Oracle)",
                param_type=ParamType.BOOLEAN,
                default=False,
                description=(
                    "Oracle mode for K-Means: use the number of ground-truth labels. "
                    "This leaks labels and should not be used for fair benchmarks."
                ),
                category="K-Means",
                advanced=True,
            ),
            HyperparameterConfig(
                name="kmeans_n_init",
                display_name="K-Means Initializations",
                param_type=ParamType.INTEGER,
                default=20,
                description="Number of K-Means initializations.",
                min_value=1,
                max_value=50,
                step=1,
                category="K-Means",
                advanced=True,
            ),
            HyperparameterConfig(
                name="kmeans_max_iter",
                display_name="K-Means Max Iterations",
                param_type=ParamType.INTEGER,
                default=300,
                description="Maximum K-Means iterations.",
                min_value=100,
                max_value=1000,
                step=100,
                category="K-Means",
                advanced=True,
            ),
            HyperparameterConfig(
                name="graph_neighbors",
                display_name="Graph Neighbors",
                param_type=ParamType.INTEGER,
                default=19,
                description="Number of neighbors for Louvain/Leiden graph construction.",
                min_value=5,
                max_value=100,
                step=5,
                category="Graph",
            ),
            HyperparameterConfig(
                name="graph_resolution",
                display_name="Graph Resolution",
                param_type=ParamType.FLOAT,
                default=0.0,
                description="Leiden resolution. Use 0.0 for automatic silhouette search.",
                min_value=0.0,
                max_value=3.0,
                step=0.1,
                category="Graph",
            ),
            HyperparameterConfig(
                name="graph_resolution_min",
                display_name="Graph Resolution Min",
                param_type=ParamType.FLOAT,
                default=0.1,
                description="Minimum Leiden resolution explored during automatic search.",
                min_value=0.01,
                max_value=3.0,
                step=0.01,
                category="Graph",
                advanced=True,
            ),
            HyperparameterConfig(
                name="graph_resolution_max",
                display_name="Graph Resolution Max",
                param_type=ParamType.FLOAT,
                default=2.5,
                description="Maximum Leiden resolution explored during automatic search.",
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                category="Graph",
                advanced=True,
            ),
            HyperparameterConfig(
                name="graph_resolution_step",
                display_name="Graph Resolution Step",
                param_type=ParamType.FLOAT,
                default=0.1,
                description="Step size for automatic Leiden resolution search.",
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                category="Graph",
                advanced=True,
            ),
            HyperparameterConfig(
                name="graph_silhouette_max_cells",
                display_name="Graph Silhouette Max Cells",
                param_type=ParamType.INTEGER,
                default=5000,
                description="Maximum cells used during automatic Leiden resolution search.",
                min_value=500,
                max_value=50000,
                step=500,
                category="Graph",
                advanced=True,
            ),
            HyperparameterConfig(
                name="n_iterations",
                display_name="Leiden Iterations",
                param_type=ParamType.INTEGER,
                default=-1,
                description="Number of Leiden iterations (-1 = until convergence).",
                min_value=-1,
                max_value=100,
                step=1,
                category="Graph",
                advanced=True,
            ),
            HyperparameterConfig(
                name="hdbscan_min_cluster_size",
                display_name="HDBSCAN Min Cluster Size",
                param_type=ParamType.INTEGER,
                default=8,
                description="Minimum size of an HDBSCAN cluster.",
                min_value=2,
                max_value=200,
                step=1,
                category="HDBSCAN",
            ),
            HyperparameterConfig(
                name="hdbscan_min_samples",
                display_name="HDBSCAN Min Samples",
                param_type=ParamType.INTEGER,
                default=6,
                description="Minimum samples for HDBSCAN core points.",
                min_value=1,
                max_value=100,
                step=1,
                category="HDBSCAN",
            ),
            HyperparameterConfig(
                name="hdbscan_cluster_selection_method",
                display_name="HDBSCAN Selection Method",
                param_type=ParamType.CHOICE,
                default="eom",
                description="HDBSCAN cluster selection strategy.",
                choices=["eom", "leaf"],
                category="HDBSCAN",
                advanced=True,
            ),
            HyperparameterConfig(
                name="hdbscan_reassign_noise",
                display_name="Reassign HDBSCAN Noise",
                param_type=ParamType.BOOLEAN,
                default=False,
                description="Assign HDBSCAN noise points (-1) to the nearest non-noise cluster.",
                category="HDBSCAN",
                advanced=True,
            ),
            HyperparameterConfig(
                name="random_state",
                display_name="Random State",
                param_type=ParamType.INTEGER,
                default=42,
                description="Random seed for reproducibility.",
                min_value=0,
                max_value=99999,
                step=1,
                category="General",
            ),
            HyperparameterConfig(
                name="use_raw_data",
                display_name="Use Raw Data",
                param_type=ParamType.BOOLEAN,
                default=False,
                description="Use original unprocessed data instead of preprocessed data.",
                category="Data",
            ),
        ]

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.pca = None
        self.n_components_used = None
        self._explained_variance = None
        self._predictor = None

    def _get_matrix(self, data: Any) -> np.ndarray:
        use_raw_data = self.params.get("use_raw_data", False)
        if hasattr(data, "X"):
            if use_raw_data and hasattr(data, "layers") and "original_X" in data.layers:
                X = data.layers["original_X"]
            else:
                X = data.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            return np.asarray(X)
        return np.asarray(data)

    def _fit_pca(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA
        from utils.pca_utils import compute_optimal_pca_components

        seed = int(self.params.get("random_state", 42))
        n_pca_components = int(self.params.get("n_pca_components", 0))

        if n_pca_components <= 0:
            method = str(self.params.get("pca_elbow_method", "cattell"))
            optimal_n, _, _ = compute_optimal_pca_components(
                X,
                max_components=100,
                method=method,
                min_components=2,
            )
            n_pca_components = optimal_n

        n_components = min(max(1, n_pca_components), X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_components, random_state=seed)
        embeddings = self.pca.fit_transform(X)
        self.n_components_used = int(n_components)
        self._explained_variance = self.pca.explained_variance_ratio_
        return embeddings

    def _cluster_kmeans(self, embeddings: np.ndarray, labels: Optional[Any]) -> np.ndarray:
        from sklearn.cluster import KMeans

        n_clusters = int(self.params.get("n_clusters", 8))
        if self.params.get("use_ground_truth_k", False):
            if labels is None:
                raise ValueError("use_ground_truth_k=True but no labels were provided.")
            n_clusters = len(np.unique(labels))
            logger.warning(
                "ORACLE MODE ENABLED: PCA+KMeans using ground truth k=%s.",
                n_clusters,
            )
        if n_clusters < 2:
            raise ValueError("K-Means requires n_clusters >= 2.")

        self._predictor = KMeans(
            n_clusters=n_clusters,
            n_init=int(self.params.get("kmeans_n_init", 20)),
            max_iter=int(self.params.get("kmeans_max_iter", 300)),
            random_state=int(self.params.get("random_state", 42)),
        )
        return self._predictor.fit_predict(embeddings)

    def _make_scanpy_adata(self, X: np.ndarray, embeddings: np.ndarray):
        import anndata as ad

        adata = ad.AnnData(X)
        adata.obsm["X_pca"] = embeddings
        return adata

    def _search_optimal_leiden_resolution(self, adata, embeddings: np.ndarray) -> float:
        import scanpy as sc
        from sklearn.metrics import silhouette_score

        seed = int(self.params.get("random_state", 42))
        res_min = float(self.params.get("graph_resolution_min", 0.1))
        res_max = float(self.params.get("graph_resolution_max", 2.5))
        res_step = float(self.params.get("graph_resolution_step", 0.1))
        max_cells = int(self.params.get("graph_silhouette_max_cells", 5000))

        if res_step <= 0:
            res_step = 0.1
        if res_max < res_min:
            res_min, res_max = res_max, res_min

        resolutions = np.arange(res_min, res_max + 1e-12, res_step)
        eval_idx = np.arange(embeddings.shape[0])
        if max_cells > 0 and embeddings.shape[0] > max_cells:
            rng = np.random.default_rng(seed)
            eval_idx = np.sort(rng.choice(embeddings.shape[0], size=max_cells, replace=False))

        n_iterations = 2 if self.params.get("n_iterations", -1) == -1 else self.params.get("n_iterations", -1)
        best_res = None
        best_score = -np.inf

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
                score = float(silhouette_score(embeddings[eval_idx], labels_eval, metric="euclidean"))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_res = float(res)

        if "_leiden_search" in adata.obs:
            del adata.obs["_leiden_search"]

        if best_res is None:
            logger.warning("PCA+Leiden: no valid silhouette score found. Falling back to 1.0.")
            return 1.0
        return best_res

    def _cluster_leiden(self, X: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        import scanpy as sc

        adata = self._make_scanpy_adata(X, embeddings)
        sc.pp.neighbors(
            adata,
            n_neighbors=int(self.params.get("graph_neighbors", 19)),
            use_rep="X_pca",
            method="gauss",
            random_state=int(self.params.get("random_state", 42)),
        )

        resolution = float(self.params.get("graph_resolution", 0.0))
        selection_mode = "manual"
        if resolution <= 0:
            resolution = self._search_optimal_leiden_resolution(adata, embeddings)
            selection_mode = "auto_silhouette"

        n_iterations = 2 if self.params.get("n_iterations", -1) == -1 else self.params.get("n_iterations", -1)
        sc.tl.leiden(
            adata,
            resolution=resolution,
            n_iterations=n_iterations,
            flavor="igraph",
            directed=False,
            random_state=int(self.params.get("random_state", 42)),
        )
        self.params["graph_resolution_selected"] = float(resolution)
        self.params["graph_resolution_selection_mode"] = selection_mode
        return adata.obs["leiden"].astype(int).to_numpy()

    def _cluster_louvain(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.neighbors import kneighbors_graph
        import igraph as ig

        if embeddings.shape[0] < 2:
            return np.zeros(embeddings.shape[0], dtype=int)

        n_neighbors = int(self.params.get("graph_neighbors", 19))
        n_neighbors = max(1, min(n_neighbors, embeddings.shape[0] - 1))
        graph_sparse = kneighbors_graph(
            embeddings,
            n_neighbors=n_neighbors,
            mode="distance",
            include_self=False,
        )
        graph_sparse = graph_sparse.maximum(graph_sparse.T).tocoo()

        graph = ig.Graph(n=embeddings.shape[0], directed=False)
        edges = list(zip(graph_sparse.row.tolist(), graph_sparse.col.tolist()))
        graph.add_edges(edges)
        if edges:
            weights = 1.0 / (1.0 + graph_sparse.data.astype(float))
            graph.es["weight"] = weights.tolist()
            partition = graph.community_multilevel(weights=graph.es["weight"])
            return np.asarray(partition.membership, dtype=int)

        return np.zeros(embeddings.shape[0], dtype=int)

    def _reassign_hdbscan_noise(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        noise_mask = labels < 0
        if not np.any(noise_mask):
            return labels
        non_noise_mask = ~noise_mask
        if not np.any(non_noise_mask):
            return np.zeros_like(labels)

        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(embeddings[non_noise_mask], labels[non_noise_mask])
        reassigned = labels.copy()
        reassigned[noise_mask] = knn.predict(embeddings[noise_mask])
        return reassigned

    def _cluster_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        import hdbscan

        if embeddings.shape[0] < 2:
            return np.zeros(embeddings.shape[0], dtype=int)

        min_cluster_size = int(self.params.get("hdbscan_min_cluster_size", 8))
        min_cluster_size = max(2, min(min_cluster_size, embeddings.shape[0]))
        min_samples = int(self.params.get("hdbscan_min_samples", 6))
        min_samples = max(1, min(min_samples, embeddings.shape[0]))

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=str(self.params.get("hdbscan_cluster_selection_method", "eom")),
        )
        labels = clusterer.fit_predict(embeddings).astype(int)
        if self.params.get("hdbscan_reassign_noise", False):
            labels = self._reassign_hdbscan_noise(embeddings, labels)
        return labels

    def _fit_predictor_for_nonparametric(self, labels: np.ndarray) -> None:
        from sklearn.neighbors import KNeighborsClassifier

        n_neighbors = min(15, len(labels))
        if n_neighbors < 1:
            return
        self._predictor = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._predictor.fit(self._embeddings, labels)

    def fit(self, data: Any, labels: Optional[Any] = None) -> "PCAClusteringAlgorithm":
        X = self._get_matrix(data)
        self._embeddings = self._fit_pca(X)

        method = str(self.params.get("clustering_method", "leiden")).lower()
        if method not in self.CLUSTERING_METHODS:
            raise ValueError(
                f"Unknown clustering_method={method!r}. "
                f"Expected one of {', '.join(self.CLUSTERING_METHODS)}."
            )

        if method == "kmeans":
            self._labels = self._cluster_kmeans(self._embeddings, labels)
        elif method == "louvain":
            self._labels = self._cluster_louvain(self._embeddings)
            self._fit_predictor_for_nonparametric(self._labels)
        elif method == "leiden":
            self._labels = self._cluster_leiden(X, self._embeddings)
            self._fit_predictor_for_nonparametric(self._labels)
        elif method == "hdbscan":
            self._labels = self._cluster_hdbscan(self._embeddings)
            self._fit_predictor_for_nonparametric(self._labels)

        self.set_effective_params(
            {
                "clustering_method": method,
                "n_pca_components_used": self.n_components_used,
            }
        )
        self._fitted = True
        return self

    def predict(self, data: Any = None) -> Any:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        if data is None:
            return self._labels

        method = str(self.params.get("clustering_method", "leiden")).lower()
        embeddings = self.encode(data)
        if method == "kmeans" and self._predictor is not None:
            return self._predictor.predict(embeddings)
        if self._predictor is None:
            raise RuntimeError(f"No predictor is available for PCA+{method}.")
        return self._predictor.predict(embeddings)

    def encode(self, data: Any) -> np.ndarray:
        if not self._fitted or self.pca is None:
            raise RuntimeError("Model must be fitted before encoding.")

        X = self._get_matrix(data)
        if self._embeddings is not None and X.shape[0] == self._embeddings.shape[0]:
            return self._embeddings
        return self.pca.transform(X)

    def get_explained_variance(self) -> Optional[np.ndarray]:
        return self._explained_variance

    def get_n_components_used(self) -> Optional[int]:
        return self.n_components_used
