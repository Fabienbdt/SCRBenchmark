"""
scMAE variant with scRAW-inspired weighted reconstruction loss.

This keeps the original scMAE architecture, preprocessing, masking strategy,
and clustering behavior, while replacing the uniform reconstruction averaging
with dynamic per-cell weights estimated from the latent space as in scRAW:

1. inverse pseudo-cluster frequency,
2. sparse-region density weighting via kNN distance,
3. additive or multiplicative fusion of both signals.
"""

from typing import Any, Dict, List, Optional, Tuple
import gc
import logging

import numpy as np

from core.algorithm_registry import AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType
from .sc_mae import ScMaeAlgorithm, apply_noise


logger = logging.getLogger(__name__)


@AlgorithmRegistry.register
class ScMaeScrawWeightedAlgorithm(ScMaeAlgorithm):
    """scMAE with scRAW-style dynamic weighted reconstruction loss."""

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name="sc_mae_scraw_weighted",
            display_name="scMAE + scRAW Weighted Loss",
            description=(
                "scMAE architecture and preprocessing with scRAW-inspired dynamic "
                "per-cell weighting on the reconstruction term."
            ),
            category="deep_learning",
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes=(
                "Same preprocessing path as scMAE. The weighted phase estimates "
                "latent rare-cell weights from pseudo-cluster frequency and local density."
            ),
            has_internal_preprocessing=True,
            recommended_data="raw",
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        params = super().get_hyperparameters()
        params.extend(
            [
                HyperparameterConfig(
                    name="warmup_epochs",
                    display_name="Warm-up Epochs",
                    param_type=ParamType.INTEGER,
                    default=30,
                    description=(
                        "Epochs trained exactly like baseline scMAE before enabling "
                        "scRAW-style cell weighting."
                    ),
                    min_value=0,
                    max_value=200,
                    step=5,
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="dynamic_weight_update_interval",
                    display_name="Weight Update Interval",
                    param_type=ParamType.INTEGER,
                    default=10,
                    description="Recompute latent cell weights every N weighted epochs (0 disables refresh).",
                    min_value=0,
                    max_value=100,
                    step=1,
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="dynamic_weight_momentum",
                    display_name="Weight Momentum",
                    param_type=ParamType.FLOAT,
                    default=0.7,
                    description="Blend factor for weight refreshes: new = m*old + (1-m)*fresh.",
                    min_value=0.0,
                    max_value=0.95,
                    step=0.05,
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="pseudo_label_method",
                    display_name="Pseudo-label Method",
                    param_type=ParamType.CHOICE,
                    default="leiden",
                    choices=["kmeans", "leiden"],
                    description="Pseudo-label generator used for cluster-frequency weighting.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="weight_component_mode",
                    display_name="Weight Component Mode",
                    param_type=ParamType.CHOICE,
                    default="full",
                    choices=["full", "density_only"],
                    description=(
                        "Use full scRAW-style cluster+density weighting, or only the "
                        "density kNN component without computing reconstruction pseudo-labels."
                    ),
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="weight_exponent",
                    display_name="Cluster Weight Exponent",
                    param_type=ParamType.FLOAT,
                    default=0.2,
                    description="Exponent for inverse-frequency cluster weighting.",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="density_knn_k",
                    display_name="Density kNN (k)",
                    param_type=ParamType.INTEGER,
                    default=15,
                    description="k used for local latent density estimation.",
                    min_value=2,
                    max_value=100,
                    step=1,
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="density_weight_exponent",
                    display_name="Density Exponent",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    description="Exponent applied to normalized kNN-distance weights.",
                    min_value=0.0,
                    max_value=3.0,
                    step=0.1,
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="density_weight_clip",
                    display_name="Density Weight Clip",
                    param_type=ParamType.FLOAT,
                    default=5.0,
                    description="Upper clip applied to density-derived weights before fusion.",
                    min_value=1.0,
                    max_value=100.0,
                    step=0.5,
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="weight_fusion_mode",
                    display_name="Weight Fusion Mode",
                    param_type=ParamType.CHOICE,
                    default="additive",
                    choices=["additive", "multiplicative"],
                    description="How cluster-frequency and density weights are fused.",
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="cluster_density_alpha",
                    display_name="Cluster/Density Alpha",
                    param_type=ParamType.FLOAT,
                    default=0.6,
                    description="Alpha used in additive fusion: alpha*cluster + (1-alpha)*density.",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    category="Weighting",
                ),
                HyperparameterConfig(
                    name="cluster_weight_power",
                    display_name="Cluster Weight Power",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    description="Power applied to cluster-frequency weights before fusion.",
                    min_value=0.0,
                    max_value=3.0,
                    step=0.1,
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="density_weight_power",
                    display_name="Density Weight Power",
                    param_type=ParamType.FLOAT,
                    default=1.0,
                    description="Power applied to density-derived weights before fusion.",
                    min_value=0.0,
                    max_value=3.0,
                    step=0.1,
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="min_cell_weight",
                    display_name="Min Cell Weight",
                    param_type=ParamType.FLOAT,
                    default=0.25,
                    description="Lower clip for the final per-cell reconstruction weights.",
                    min_value=0.01,
                    max_value=5.0,
                    step=0.05,
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="max_cell_weight",
                    display_name="Max Cell Weight",
                    param_type=ParamType.FLOAT,
                    default=10.0,
                    description="Upper clip for the final per-cell reconstruction weights.",
                    min_value=1.0,
                    max_value=100.0,
                    step=0.5,
                    category="Weighting",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_weight",
                    display_name="Rare Triplet Weight",
                    param_type=ParamType.FLOAT,
                    default=0.0,
                    description="Weight applied to the optional scRAW rare-cell triplet loss.",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.01,
                    category="Triplet",
                ),
                HyperparameterConfig(
                    name="rare_triplet_margin",
                    display_name="Rare Triplet Margin",
                    param_type=ParamType.FLOAT,
                    default=0.4,
                    description="Margin for the rare-cell triplet loss.",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    category="Triplet",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_min_weight",
                    display_name="Rare Triplet Min Weight",
                    param_type=ParamType.FLOAT,
                    default=1.2,
                    description="Minimum cell weight required for a cell to be a triplet anchor.",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.1,
                    category="Triplet",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="rare_triplet_start_epoch",
                    display_name="Rare Triplet Start Epoch",
                    param_type=ParamType.INTEGER,
                    default=60,
                    description="Epoch at which the optional rare-cell triplet loss starts.",
                    min_value=0,
                    max_value=500,
                    step=5,
                    category="Triplet",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="max_triplet_anchors_per_batch",
                    display_name="Max Triplet Anchors per Batch",
                    param_type=ParamType.INTEGER,
                    default=64,
                    description="Maximum number of rare anchors sampled per mini-batch.",
                    min_value=0,
                    max_value=1024,
                    step=16,
                    category="Triplet",
                    advanced=True,
                ),
                HyperparameterConfig(
                    name="triplet_pseudo_label_method",
                    display_name="Triplet Pseudo-label Method",
                    param_type=ParamType.CHOICE,
                    default="kmeans",
                    choices=["kmeans", "leiden"],
                    description=(
                        "Pseudo-label generator used only for the optional triplet loss. "
                        "This is separate from reconstruction weighting."
                    ),
                    category="Triplet",
                    advanced=True,
                ),
            ]
        )
        return params

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self._cell_weights: Optional[np.ndarray] = None
        self._pseudo_labels: Optional[np.ndarray] = None
        self._weight_history: List[Dict[str, Any]] = []

    @staticmethod
    def _summarize_cell_weights(
        weights: np.ndarray,
        *,
        epoch: int,
        phase: str,
        refreshed: bool,
    ) -> Dict[str, Any]:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if w.size == 0:
            return {
                "epoch": int(epoch),
                "phase": str(phase),
                "refreshed": bool(refreshed),
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "max": float("nan"),
            }
        return {
            "epoch": int(epoch),
            "phase": str(phase),
            "refreshed": bool(refreshed),
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
            "min": float(np.min(w)),
            "p05": float(np.percentile(w, 5)),
            "p50": float(np.percentile(w, 50)),
            "p95": float(np.percentile(w, 95)),
            "max": float(np.max(w)),
        }

    def _sanitize_embeddings(self, values: np.ndarray, context: str) -> np.ndarray:
        """Ensure finite arrays before clustering and distance computations."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        non_finite = ~np.isfinite(arr)
        if np.any(non_finite):
            logger.warning(
                "scMAE+scRAW weights: %d non-finite values detected in %s. Applying nan_to_num.",
                int(np.sum(non_finite)),
                context,
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        return np.clip(arr, -1e4, 1e4).astype(np.float32, copy=False)

    def _leiden_pseudo_labels(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Compute pseudo-labels with Leiden by matching the requested cluster count."""
        import anndata as ad
        import scanpy as sc

        emb = self._sanitize_embeddings(embeddings, "leiden pseudo-label input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.zeros(n_cells, dtype=np.int64)

        target_k = int(max(2, min(int(n_clusters), n_cells)))
        adata = ad.AnnData(X=emb)
        sc.pp.neighbors(adata, n_neighbors=min(15, max(2, n_cells - 1)), use_rep="X")

        best_res = 1.0
        best_diff = n_cells
        random_state = int(self.params.get("random_state", 42))
        for res in np.arange(0.05, 3.0, 0.05):
            sc.tl.leiden(adata, resolution=float(res), random_state=random_state)
            n_found = len(np.unique(adata.obs["leiden"].astype(int).values))
            diff = abs(n_found - target_k)
            if diff < best_diff:
                best_diff = diff
                best_res = float(res)
            if n_found == target_k:
                break

        sc.tl.leiden(adata, resolution=best_res, random_state=random_state)
        return adata.obs["leiden"].astype(int).values.astype(np.int64)

    def _pseudo_labels_for_method(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        method: str,
    ) -> np.ndarray:
        """Compute pseudo-labels for the requested method."""
        emb = self._sanitize_embeddings(embeddings, f"{method} pseudo-label input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.zeros(n_cells, dtype=np.int64)

        upper_k = n_cells if n_cells <= 2 else (n_cells - 1)
        k = int(max(2, min(int(n_clusters), upper_k)))
        method = str(method).strip().lower()

        if method == "leiden":
            return self._leiden_pseudo_labels(emb, k)
        if method != "kmeans":
            logger.warning(
                "scMAE+scRAW weighted: unknown pseudo-label method '%s'; falling back to kmeans.",
                method,
            )

        from sklearn.cluster import KMeans

        random_state = int(self.params.get("random_state", 42))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        return kmeans.fit_predict(emb).astype(np.int64)

    def _weight_component_mode(self) -> str:
        mode = str(self.params.get("weight_component_mode", "full")).strip().lower()
        if mode not in {"full", "density_only"}:
            logger.warning(
                "scMAE+scRAW weighted: unknown weight_component_mode='%s'; using full.",
                mode,
            )
            mode = "full"
        return mode

    def _compute_cluster_frequency_weights(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute inverse-frequency weights from pseudo-clusters."""
        emb = self._sanitize_embeddings(embeddings, "cluster-frequency input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.ones(n_cells, dtype=np.float32), np.zeros(n_cells, dtype=np.int64)

        upper_k = n_cells if n_cells <= 2 else (n_cells - 1)
        k = int(max(2, min(int(n_clusters), upper_k)))
        method = str(self.params.get("pseudo_label_method", "leiden")).strip().lower()
        exponent = float(self.params.get("weight_exponent", 0.2))
        pseudo_labels = self._pseudo_labels_for_method(emb, k, method)

        unique_labels, counts = np.unique(pseudo_labels, return_counts=True)
        freqs = counts / np.sum(counts)
        label_to_weight = {
            int(label): float((1.0 / max(freq, 1e-8)) ** exponent)
            for label, freq in zip(unique_labels, freqs)
        }

        weights = np.array([label_to_weight[int(label)] for label in pseudo_labels], dtype=np.float32)
        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights = weights / mean_weight
        return weights.astype(np.float32), pseudo_labels.astype(np.int64)

    def _compute_density_weights(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute density-aware weights from latent kNN distances."""
        from sklearn.neighbors import NearestNeighbors

        emb = self._sanitize_embeddings(embeddings, "density input")
        n_cells = emb.shape[0]
        if n_cells <= 2:
            return np.ones(n_cells, dtype=np.float32)

        k_param = int(self.params.get("density_knn_k", 15))
        k = int(max(2, min(k_param, n_cells - 1)))
        exponent = float(self.params.get("density_weight_exponent", 1.0))
        density_clip = float(self.params.get("density_weight_clip", 5.0))

        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nn.fit(emb)
        distances, _ = nn.kneighbors(emb)

        kth_dist = distances[:, -1].astype(np.float32)
        scale = float(np.median(kth_dist)) + 1e-8
        normalized = kth_dist / scale

        weights = np.power(np.maximum(normalized, 1e-8), exponent)
        weights = np.clip(weights, 0.05, density_clip).astype(np.float32)
        mean_weight = float(np.mean(weights))
        if mean_weight > 0:
            weights = weights / mean_weight
        return weights.astype(np.float32)

    def _compute_combined_cell_weights(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combine cluster-frequency and density weights like scRAW."""
        density_weights = self._compute_density_weights(embeddings)
        if self._weight_component_mode() == "density_only":
            pseudo_labels = np.zeros(np.asarray(embeddings).shape[0], dtype=np.int64)
            min_cell_weight = float(self.params.get("min_cell_weight", 0.25))
            max_cell_weight = float(self.params.get("max_cell_weight", 10.0))
            min_cell_weight = min(min_cell_weight, max_cell_weight)
            weights = np.clip(density_weights, min_cell_weight, max_cell_weight).astype(np.float32)
            mean_weight = float(np.mean(weights))
            if mean_weight > 0:
                weights = weights / mean_weight
            return weights.astype(np.float32), pseudo_labels

        cluster_weights, pseudo_labels = self._compute_cluster_frequency_weights(
            embeddings=embeddings,
            n_clusters=n_clusters,
        )

        fusion_mode = str(self.params.get("weight_fusion_mode", "additive")).strip().lower()
        cluster_power = float(self.params.get("cluster_weight_power", 1.0))
        density_power = float(self.params.get("density_weight_power", 1.0))
        min_cell_weight = float(self.params.get("min_cell_weight", 0.25))
        max_cell_weight = float(self.params.get("max_cell_weight", 10.0))
        min_cell_weight = min(min_cell_weight, max_cell_weight)

        cw = np.power(np.maximum(cluster_weights, 1e-8), cluster_power)
        dw = np.power(np.maximum(density_weights, 1e-8), density_power)

        if fusion_mode == "multiplicative":
            combined = (cw * dw).astype(np.float32)
        else:
            alpha = float(self.params.get("cluster_density_alpha", 0.6))
            combined = (alpha * cw + (1.0 - alpha) * dw).astype(np.float32)

        mean_weight = float(np.mean(combined))
        if mean_weight > 0:
            combined = combined / mean_weight
        combined = np.clip(combined, min_cell_weight, max_cell_weight).astype(np.float32)
        return combined, pseudo_labels

    def _compute_triplet_pseudo_labels(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        fallback_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute pseudo-labels for triplet mining, separately from reconstruction weights."""
        method = str(
            self.params.get(
                "triplet_pseudo_label_method",
                self.params.get("pseudo_label_method", "kmeans"),
            )
        ).strip().lower()
        if (
            fallback_labels is not None
            and self._weight_component_mode() == "full"
            and method == str(self.params.get("pseudo_label_method", "leiden")).strip().lower()
        ):
            return np.asarray(fallback_labels, dtype=np.int64)
        return self._pseudo_labels_for_method(embeddings, n_clusters, method)

    def _rare_triplet_loss(
        self,
        z: Any,
        batch_indices: Any,
        batch_weights: Any,
        pseudo_labels_tensor: Any,
        *,
        margin: float,
        min_anchor_weight: float,
        max_anchors: int,
    ) -> Tuple[Any, int]:
        """Semi-hard pseudo-label triplet loss on high-weight anchors."""
        import torch

        if pseudo_labels_tensor is None or z.size(0) < 3:
            return z.new_tensor(0.0), 0

        labels = pseudo_labels_tensor[batch_indices]
        candidate_mask = batch_weights >= float(min_anchor_weight)
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        if candidate_idx.numel() == 0:
            return z.new_tensor(0.0), 0

        if max_anchors > 0 and candidate_idx.numel() > max_anchors:
            perm = torch.randperm(candidate_idx.numel(), device=candidate_idx.device)
            candidate_idx = candidate_idx[perm[:max_anchors]]

        dists = torch.cdist(z, z, p=2)
        losses = []
        valid_anchors = 0
        for anchor in candidate_idx:
            same = labels == labels[anchor]
            same[anchor] = False
            diff = ~same
            diff[anchor] = False
            if not bool(torch.any(same)) or not bool(torch.any(diff)):
                continue

            d_pos = dists[anchor, same].max()
            neg_dists = dists[anchor, diff]
            semi_hard_mask = neg_dists > d_pos
            if bool(torch.any(semi_hard_mask)):
                d_neg = neg_dists[semi_hard_mask].min()
            else:
                d_neg = neg_dists.min()
            losses.append(torch.relu(d_pos - d_neg + float(margin)))
            valid_anchors += 1

        if not losses:
            return z.new_tensor(0.0), 0
        return torch.stack(losses).mean(), valid_anchors

    def _weighted_loss_mask(self, x: Any, y: Any, mask: Any, batch_weights: Any) -> Tuple[Any, Any, Any, Any]:
        """scMAE loss with scRAW-style per-cell weighting on the reconstruction term."""
        import torch
        from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
        from torch.nn.functional import mse_loss as mse

        latent, predicted_mask, reconstruction = self.model.forward_mask(x)

        masked_data_weight = float(self.model.masked_data_weight)
        mask_loss_weight = float(self.model.mask_loss_weight)
        w_nums = mask * masked_data_weight + (1.0 - mask) * (1.0 - masked_data_weight)

        per_sample_recon = torch.mul(
            w_nums,
            mse(reconstruction, y, reduction="none"),
        ).mean(dim=1)

        reconstruction_loss = (1.0 - mask_loss_weight) * (
            per_sample_recon * batch_weights
        ).mean()
        mask_loss = mask_loss_weight * bce_logits(predicted_mask, mask, reduction="mean")
        total_loss = reconstruction_loss + mask_loss
        return latent, total_loss, reconstruction_loss, mask_loss

    def fit(self, data: Any, labels: Optional[Any] = None) -> "ScMaeScrawWeightedAlgorithm":
        """
        Fit scMAE with a scRAW-style weighted reconstruction phase.

        The first `warmup_epochs` reproduce baseline scMAE exactly.
        Then latent weights are refreshed periodically and injected into the
        reconstruction term as per-cell multipliers.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X, labels = self._prepare_scmae_inputs(data, labels)
        X = self._apply_internal_preprocessing(X)

        n_clusters = self.params.get("n_clusters", 0)
        if n_clusters == 0 and labels is not None:
            n_clusters = len(np.unique(labels))
            logger.warning(
                "scMAE+scRAW weighted: n_clusters=0 (Auto). Using ground truth labels to determine k=%s (Oracle Mode).",
                n_clusters,
            )
        elif n_clusters == 0:
            n_clusters = 8
            logger.warning(
                "scMAE+scRAW weighted: n_clusters=0 and no labels. Defaulting to k=%s.",
                n_clusters,
            )

        random_state = int(self.params.get("random_state", 42))
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        num_genes = X.shape[1]
        self.model = self._build_model(num_genes)

        device_name = self.get_device()
        device = torch.device(device_name)
        if device_name == "mps":
            torch.set_default_dtype(torch.float32)
        self.model = self.model.float().to(device)

        batch_size = int(self.params.get("batch_size", 256))
        n_samples = int(X.shape[0])
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.arange(n_samples, dtype=torch.long),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=n_samples > batch_size,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params.get("learning_rate", 0.001)),
        )

        epochs = int(self.params.get("epochs", 100))
        warmup_epochs = int(max(0, min(self.params.get("warmup_epochs", 30), epochs)))
        eval_epoch = int(self.params.get("eval_epoch", 80))
        masking_rate = float(self.params.get("masking_rate", 0.4))
        weight_update_interval = int(max(0, self.params.get("dynamic_weight_update_interval", 10)))
        weight_momentum = float(np.clip(self.params.get("dynamic_weight_momentum", 0.7), 0.0, 0.95))
        rare_triplet_weight = float(self.params.get("rare_triplet_weight", 0.0))
        rare_triplet_margin = float(self.params.get("rare_triplet_margin", 0.4))
        rare_triplet_min_weight = float(self.params.get("rare_triplet_min_weight", 1.2))
        rare_triplet_start_epoch = int(self.params.get("rare_triplet_start_epoch", warmup_epochs))
        max_triplet_anchors = int(self.params.get("max_triplet_anchors_per_batch", 64))
        mask_probas = [masking_rate] * num_genes
        full_X_tensor = torch.FloatTensor(X)

        cell_weights = np.ones(n_samples, dtype=np.float32)
        pseudo_labels = np.zeros(n_samples, dtype=np.int64)
        triplet_pseudo_labels = np.zeros(n_samples, dtype=np.int64)
        weights_tensor = torch.ones(n_samples, dtype=torch.float32, device=device)
        triplet_labels_tensor = torch.zeros(n_samples, dtype=torch.long, device=device)

        self.train_losses = []
        self.train_recon_losses = []
        self.train_mask_losses = []
        self.train_triplet_losses = []
        self.train_triplet_valid_anchors = []
        self._eval_epoch_embeddings = None
        self._weight_history = []

        def _refresh_global_weights(previous_weights: Optional[np.ndarray], epoch: int) -> None:
            nonlocal cell_weights
            nonlocal pseudo_labels
            nonlocal triplet_pseudo_labels
            nonlocal weights_tensor
            nonlocal triplet_labels_tensor

            self.model.eval()
            with torch.no_grad():
                latent = self.model.feature(full_X_tensor.to(device)).cpu().numpy()
            latent = self._sanitize_embeddings(latent, f"weight refresh epoch={epoch}")

            fresh_weights, fresh_labels = self._compute_combined_cell_weights(
                embeddings=latent,
                n_clusters=n_clusters,
            )

            if previous_weights is not None:
                mixed = weight_momentum * previous_weights + (1.0 - weight_momentum) * fresh_weights
                mean_weight = float(np.mean(mixed))
                if mean_weight > 0:
                    mixed = mixed / mean_weight
                fresh_weights = mixed.astype(np.float32)

            min_cell_weight = float(self.params.get("min_cell_weight", 0.25))
            max_cell_weight = float(self.params.get("max_cell_weight", 10.0))
            min_cell_weight = min(min_cell_weight, max_cell_weight)
            fresh_weights = np.clip(fresh_weights, min_cell_weight, max_cell_weight).astype(np.float32)

            cell_weights = fresh_weights
            pseudo_labels = fresh_labels.astype(np.int64)
            if rare_triplet_weight > 0:
                triplet_pseudo_labels = self._compute_triplet_pseudo_labels(
                    latent,
                    n_clusters,
                    fallback_labels=pseudo_labels,
                ).astype(np.int64)
            weights_tensor = torch.from_numpy(cell_weights).to(device=device, dtype=torch.float32)
            triplet_labels_tensor = torch.from_numpy(triplet_pseudo_labels).to(device=device, dtype=torch.long)

            logger.info(
                "scMAE+scRAW weighted: refreshed weights at epoch=%d | min=%.3f p50=%.3f p95=%.3f max=%.3f",
                epoch,
                float(np.min(cell_weights)),
                float(np.percentile(cell_weights, 50)),
                float(np.percentile(cell_weights, 95)),
                float(np.max(cell_weights)),
            )
            self.model.train()

        print(
            "scMAE+scRAW weighted: Training for "
            f"{epochs} epochs (warm-up={warmup_epochs}, eval at epoch {eval_epoch if eval_epoch > 0 else epochs})..."
        )

        for epoch in range(epochs):
            weights_refreshed = False
            if epoch == warmup_epochs and epoch < epochs:
                _refresh_global_weights(previous_weights=None, epoch=epoch)
                weights_refreshed = True
            elif (
                epoch > warmup_epochs
                and weight_update_interval > 0
                and ((epoch - warmup_epochs) % weight_update_interval == 0)
            ):
                _refresh_global_weights(previous_weights=cell_weights.copy(), epoch=epoch)
                weights_refreshed = True

            self.model.train()
            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_mask = 0.0
            epoch_triplet = 0.0
            epoch_valid_anchors = 0.0
            epoch_samples = 0

            for batch, batch_idx in loader:
                batch = batch.to(device)
                batch_idx = batch_idx.to(device)
                batch_weights = weights_tensor[batch_idx]

                corrupted_batch, mask = apply_noise(batch, mask_probas)

                optimizer.zero_grad()
                latent, loss, recon_loss, mask_loss = self._weighted_loss_mask(
                    corrupted_batch,
                    batch,
                    mask,
                    batch_weights,
                )
                triplet_loss = latent.new_tensor(0.0)
                valid_anchors = 0
                if (
                    rare_triplet_weight > 0
                    and epoch >= warmup_epochs
                    and epoch >= rare_triplet_start_epoch
                ):
                    ramp_epochs = max(1, min(20, epochs - rare_triplet_start_epoch))
                    rare_loss_ramp = min(1.0, (epoch - rare_triplet_start_epoch) / ramp_epochs)
                    triplet_loss, valid_anchors = self._rare_triplet_loss(
                        latent,
                        batch_idx,
                        batch_weights,
                        triplet_labels_tensor,
                        margin=rare_triplet_margin,
                        min_anchor_weight=rare_triplet_min_weight,
                        max_anchors=max_triplet_anchors,
                    )
                    loss = loss + (rare_loss_ramp * rare_triplet_weight * triplet_loss)
                loss.backward()
                optimizer.step()

                batch_size_eff = int(batch.size(0))
                epoch_total += float(loss.item()) * batch_size_eff
                epoch_recon += float(recon_loss.item()) * batch_size_eff
                epoch_mask += float(mask_loss.item()) * batch_size_eff
                epoch_triplet += float(triplet_loss.item()) * batch_size_eff
                epoch_valid_anchors += float(valid_anchors)
                epoch_samples += batch_size_eff

            avg_total = epoch_total / max(1, epoch_samples)
            avg_recon = epoch_recon / max(1, epoch_samples)
            avg_mask = epoch_mask / max(1, epoch_samples)
            avg_triplet = epoch_triplet / max(1, epoch_samples)
            self.train_losses.append(avg_total)
            self.train_recon_losses.append(avg_recon)
            self.train_mask_losses.append(avg_mask)
            self.train_triplet_losses.append(avg_triplet)
            self.train_triplet_valid_anchors.append(epoch_valid_anchors)
            self._weight_history.append(
                self._summarize_cell_weights(
                    cell_weights,
                    epoch=epoch,
                    phase="warmup" if epoch < warmup_epochs else "weighted",
                    refreshed=weights_refreshed,
                )
            )

            if eval_epoch > 0 and epoch == eval_epoch:
                self.model.eval()
                with torch.no_grad():
                    self._eval_epoch_embeddings = self.model.feature(
                        full_X_tensor.to(device)
                    ).cpu().numpy()
                print(f"  Epoch {epoch}: Saved embeddings for clustering")

            if epoch % 20 == 0 or epoch == epochs - 1 or epoch == warmup_epochs:
                phase = "warmup" if epoch < warmup_epochs else "weighted"
                print(
                    f"  Epoch {epoch}/{epochs} [{phase}]: "
                    f"loss={avg_total:.6f} recon={avg_recon:.6f} "
                    f"mask={avg_mask:.6f} triplet={avg_triplet:.6f}"
                )

        if self._eval_epoch_embeddings is not None and eval_epoch > 0:
            self._embeddings = self._eval_epoch_embeddings
        else:
            self.model.eval()
            with torch.no_grad():
                self._embeddings = self.model.feature(full_X_tensor.to(device)).cpu().numpy()

        clustering_method = self.params.get("clustering_method", "auto")
        print(
            f"scMAE+scRAW weighted: Starting clustering ({clustering_method}) "
            f"for {n_clusters} clusters on {len(X)} cells..."
        )

        if clustering_method == "auto":
            if len(X) < 10000:
                print("scMAE+scRAW weighted: Using KMeans (n_cells < 10000)")
                self._labels = self._kmeans_clustering(self._embeddings, n_clusters)
            else:
                print("scMAE+scRAW weighted: Using Leiden with resolution search (n_cells >= 10000)")
                self._labels = self._leiden_clustering_with_search(self._embeddings, n_clusters)
        elif clustering_method == "leiden":
            print("scMAE+scRAW weighted: Using Leiden with resolution search (forced)")
            self._labels = self._leiden_clustering_with_search(self._embeddings, n_clusters)
        else:
            print("scMAE+scRAW weighted: Using KMeans (forced)")
            self._labels = self._kmeans_clustering(self._embeddings, n_clusters)

        print(
            f"scMAE+scRAW weighted: Clustering complete. Found {len(np.unique(self._labels))} clusters."
        )

        self._train_n_cells = len(self._labels)
        self._cell_weights = cell_weights.copy()
        self._pseudo_labels = pseudo_labels.copy()
        self._triplet_pseudo_labels = triplet_pseudo_labels.copy()
        self._loss_history = [
            {
                "name": "training",
                "epochs": list(range(len(self.train_losses))),
                "train_loss": list(self.train_losses),
                "components": {
                    "weighted_reconstruction": list(self.train_recon_losses),
                    "mask_bce": list(self.train_mask_losses),
                    "rare_triplet": list(self.train_triplet_losses),
                    "triplet_valid_anchors": list(self.train_triplet_valid_anchors),
                    "weight_mean": [row["mean"] for row in self._weight_history],
                    "weight_std": [row["std"] for row in self._weight_history],
                    "weight_min": [row["min"] for row in self._weight_history],
                    "weight_max": [row["max"] for row in self._weight_history],
                },
            }
        ]

        self._fitted = True
        gc.collect()
        return self
