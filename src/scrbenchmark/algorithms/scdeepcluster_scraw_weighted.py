"""
scDeepCluster variant with scRAW-inspired weighted reconstruction loss.

This keeps the original scDeepCluster preprocessing, architecture, and
clustering objective intact while replacing the uniform reconstruction averaging
with dynamic per-cell weights estimated from the latent space:

1. inverse pseudo-cluster frequency,
2. sparse-region density weighting via kNN distance,
3. additive or multiplicative fusion of both signals.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset

from core.algorithm_registry import AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType
from .scdeepcluster import (
    ScDeepClusterAlgorithm,
    geneSelection,
    read_dataset,
    scDeepCluster,
)


logger = logging.getLogger(__name__)


class ScDeepClusterScrawWeighted(scDeepCluster):
    """scDeepCluster model with scRAW-style dynamic reconstruction weighting."""

    def __init__(self, *args, weight_params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_params = dict(weight_params or {})
        self.pretrain_recon_history: List[float] = []
        self.pretrain_rare_history: List[float] = []
        self.pretrain_valid_anchor_history: List[float] = []
        self.cluster_recon_history: List[float] = []
        self.cluster_kl_history: List[float] = []
        self.cluster_rare_history: List[float] = []
        self.cluster_valid_anchor_history: List[float] = []
        self.pretrain_weight_history: List[Dict[str, float]] = []
        self.cluster_weight_history: List[Dict[str, float]] = []
        self.current_cell_weights: Optional[np.ndarray] = None
        self.current_pseudo_labels: Optional[np.ndarray] = None
        self.current_triplet_pseudo_labels: Optional[np.ndarray] = None

    @staticmethod
    def _summarize_cell_weights(
        weights: np.ndarray,
        *,
        epoch: int,
        phase: str,
        refreshed: bool,
        phase_epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        row: Dict[str, Any] = {
            "epoch": int(epoch),
            "phase": str(phase),
            "refreshed": bool(refreshed),
        }
        if phase_epoch is not None:
            row["phase_epoch"] = int(phase_epoch)
        if w.size == 0:
            row.update(
                {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "p05": float("nan"),
                    "p50": float("nan"),
                    "p95": float("nan"),
                    "max": float("nan"),
                }
            )
            return row
        row.update(
            {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "p05": float(np.percentile(w, 5)),
                "p50": float(np.percentile(w, 50)),
                "p95": float(np.percentile(w, 95)),
                "max": float(np.max(w)),
            }
        )
        return row

    def _sanitize_embeddings(self, values: np.ndarray, context: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        non_finite = ~np.isfinite(arr)
        if np.any(non_finite):
            logger.warning(
                "scDeepCluster+scRAW weights: %d non-finite values detected in %s. Applying nan_to_num.",
                int(np.sum(non_finite)),
                context,
            )
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4)
        return np.clip(arr, -1e4, 1e4).astype(np.float32, copy=False)

    def _leiden_pseudo_labels(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
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
        random_state = int(self.weight_params.get("random_state", 42))
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
                "scDeepCluster+scRAW weights: unknown pseudo-label method '%s'; falling back to kmeans.",
                method,
            )
        random_state = int(self.weight_params.get("random_state", 42))
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        return kmeans.fit_predict(emb).astype(np.int64)

    def _weight_component_mode(self) -> str:
        mode = str(self.weight_params.get("weight_component_mode", "full")).strip().lower()
        if mode not in {"full", "density_only"}:
            logger.warning(
                "scDeepCluster+scRAW weights: unknown weight_component_mode='%s'; using full.",
                mode,
            )
            mode = "full"
        return mode

    def _compute_cluster_frequency_weights(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        emb = self._sanitize_embeddings(embeddings, "cluster-frequency input")
        n_cells = emb.shape[0]
        if n_cells <= 1:
            return np.ones(n_cells, dtype=np.float32), np.zeros(n_cells, dtype=np.int64)

        upper_k = n_cells if n_cells <= 2 else (n_cells - 1)
        k = int(max(2, min(int(n_clusters), upper_k)))
        method = str(self.weight_params.get("pseudo_label_method", "leiden")).strip().lower()
        exponent = float(self.weight_params.get("weight_exponent", 0.2))
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
        from sklearn.neighbors import NearestNeighbors

        emb = self._sanitize_embeddings(embeddings, "density input")
        n_cells = emb.shape[0]
        if n_cells <= 2:
            return np.ones(n_cells, dtype=np.float32)

        k_param = int(self.weight_params.get("density_knn_k", 15))
        k = int(max(2, min(k_param, n_cells - 1)))
        exponent = float(self.weight_params.get("density_weight_exponent", 1.0))
        density_clip = float(self.weight_params.get("density_weight_clip", 5.0))

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
        density_weights = self._compute_density_weights(embeddings)
        if self._weight_component_mode() == "density_only":
            pseudo_labels = np.zeros(np.asarray(embeddings).shape[0], dtype=np.int64)
            min_cell_weight = float(self.weight_params.get("min_cell_weight", 0.25))
            max_cell_weight = float(self.weight_params.get("max_cell_weight", 10.0))
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

        fusion_mode = str(self.weight_params.get("weight_fusion_mode", "additive")).strip().lower()
        cluster_power = float(self.weight_params.get("cluster_weight_power", 1.0))
        density_power = float(self.weight_params.get("density_weight_power", 1.0))
        min_cell_weight = float(self.weight_params.get("min_cell_weight", 0.25))
        max_cell_weight = float(self.weight_params.get("max_cell_weight", 10.0))
        min_cell_weight = min(min_cell_weight, max_cell_weight)

        cw = np.power(np.maximum(cluster_weights, 1e-8), cluster_power)
        dw = np.power(np.maximum(density_weights, 1e-8), density_power)

        if fusion_mode == "multiplicative":
            combined = (cw * dw).astype(np.float32)
        else:
            alpha = float(self.weight_params.get("cluster_density_alpha", 0.6))
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
        method = str(
            self.weight_params.get(
                "triplet_pseudo_label_method",
                self.weight_params.get("pseudo_label_method", "kmeans"),
            )
        ).strip().lower()
        if (
            fallback_labels is not None
            and self._weight_component_mode() == "full"
            and method == str(self.weight_params.get("pseudo_label_method", "leiden")).strip().lower()
        ):
            return np.asarray(fallback_labels, dtype=np.int64)
        return self._pseudo_labels_for_method(embeddings, n_clusters, method)

    def _rare_triplet_loss(
        self,
        z: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_weights: torch.Tensor,
        pseudo_labels_tensor: Optional[torch.Tensor],
        *,
        margin: float,
        min_anchor_weight: float,
        max_anchors: int,
    ) -> Tuple[torch.Tensor, int]:
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

    def _refresh_cell_weights(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        previous_weights: Optional[np.ndarray],
        epoch: int,
        phase: str,
    ) -> np.ndarray:
        fresh_weights, fresh_labels = self._compute_combined_cell_weights(
            embeddings=embeddings,
            n_clusters=n_clusters,
        )

        if previous_weights is not None:
            momentum = float(np.clip(self.weight_params.get("dynamic_weight_momentum", 0.7), 0.0, 0.95))
            mixed = momentum * previous_weights + (1.0 - momentum) * fresh_weights
            mean_weight = float(np.mean(mixed))
            if mean_weight > 0:
                mixed = mixed / mean_weight
            fresh_weights = mixed.astype(np.float32)

        min_cell_weight = float(self.weight_params.get("min_cell_weight", 0.25))
        max_cell_weight = float(self.weight_params.get("max_cell_weight", 10.0))
        min_cell_weight = min(min_cell_weight, max_cell_weight)
        fresh_weights = np.clip(fresh_weights, min_cell_weight, max_cell_weight).astype(np.float32)

        self.current_cell_weights = fresh_weights
        self.current_pseudo_labels = fresh_labels.astype(np.int64)

        stats = self._summarize_cell_weights(
            fresh_weights,
            epoch=int(epoch),
            phase=phase,
            refreshed=True,
            phase_epoch=int(epoch),
        )

        logger.info(
            "scDeepCluster+scRAW weights refreshed [%s] epoch=%d | min=%.3f p50=%.3f p95=%.3f max=%.3f",
            phase,
            int(epoch),
            stats["min"],
            stats["p50"],
            stats["p95"],
            stats["max"],
        )
        return fresh_weights

    def weighted_zinb_loss(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        disp: torch.Tensor,
        pi: torch.Tensor,
        scale_factor: torch.Tensor,
        cell_weights: Optional[torch.Tensor] = None,
        ridge_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
            x * (torch.log(disp + eps) - torch.log(mean + eps))
        )
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result = result + ridge

        per_cell = torch.mean(result, dim=1)
        if cell_weights is not None:
            weights = cell_weights.reshape(-1).to(device=per_cell.device, dtype=per_cell.dtype)
            loss = torch.mean(per_cell * weights)
        else:
            loss = torch.mean(per_cell)
        return loss, per_cell

    def pretrain_autoencoder(
        self,
        X,
        X_raw,
        size_factor,
        batch_size=256,
        lr=0.001,
        epochs=400,
        ae_save=True,
        ae_weights="AE_weights.pth.tar",
        n_clusters_for_weights=8,
    ):
        self.train()
        dataset = TensorDataset(
            torch.tensor(X, dtype=self.dtype),
            torch.tensor(X_raw, dtype=self.dtype),
            torch.tensor(size_factor, dtype=self.dtype),
            torch.arange(X.shape[0], dtype=torch.long),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        full_X = torch.tensor(X, dtype=self.dtype)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)

        warmup_epochs = int(max(0, min(int(self.weight_params.get("warmup_epochs", 30)), int(epochs))))
        update_interval = int(max(0, self.weight_params.get("dynamic_weight_update_interval", 10)))
        cell_weights = np.ones(X.shape[0], dtype=np.float32)
        weights_tensor = torch.ones(X.shape[0], dtype=torch.float32, device=self.device)
        triplet_labels_tensor = torch.zeros(X.shape[0], dtype=torch.long, device=self.device)
        rare_triplet_weight = float(self.weight_params.get("rare_triplet_weight", 0.0))
        rare_triplet_start_epoch = int(self.weight_params.get("rare_triplet_start_epoch", warmup_epochs))
        rare_triplet_margin = float(self.weight_params.get("rare_triplet_margin", 0.4))
        rare_triplet_min_weight = float(self.weight_params.get("rare_triplet_min_weight", 1.2))
        max_triplet_anchors = int(self.weight_params.get("max_triplet_anchors_per_batch", 64))

        print("Pretraining stage (scRAW-weighted)")
        for epoch in range(int(epochs)):
            weights_refreshed = False
            if epoch == warmup_epochs and epoch < int(epochs):
                with torch.no_grad():
                    latent = self.encodeBatch(full_X.to(self.device)).cpu().numpy()
                cell_weights = self._refresh_cell_weights(
                    embeddings=latent,
                    n_clusters=n_clusters_for_weights,
                    previous_weights=None,
                    epoch=epoch,
                    phase="pretrain",
                )
                weights_refreshed = True
                weights_tensor = torch.from_numpy(cell_weights).to(device=self.device, dtype=torch.float32)
                if rare_triplet_weight > 0:
                    triplet_labels = self._compute_triplet_pseudo_labels(
                        latent,
                        n_clusters_for_weights,
                        fallback_labels=self.current_pseudo_labels,
                    )
                    self.current_triplet_pseudo_labels = triplet_labels.astype(np.int64)
                    triplet_labels_tensor = torch.from_numpy(self.current_triplet_pseudo_labels).to(
                        device=self.device,
                        dtype=torch.long,
                    )
            elif (
                epoch > warmup_epochs
                and update_interval > 0
                and ((epoch - warmup_epochs) % update_interval == 0)
            ):
                with torch.no_grad():
                    latent = self.encodeBatch(full_X.to(self.device)).cpu().numpy()
                cell_weights = self._refresh_cell_weights(
                    embeddings=latent,
                    n_clusters=n_clusters_for_weights,
                    previous_weights=cell_weights.copy(),
                    epoch=epoch,
                    phase="pretrain",
                )
                weights_refreshed = True
                weights_tensor = torch.from_numpy(cell_weights).to(device=self.device, dtype=torch.float32)
                if rare_triplet_weight > 0:
                    triplet_labels = self._compute_triplet_pseudo_labels(
                        latent,
                        n_clusters_for_weights,
                        fallback_labels=self.current_pseudo_labels,
                    )
                    self.current_triplet_pseudo_labels = triplet_labels.astype(np.int64)
                    triplet_labels_tensor = torch.from_numpy(self.current_triplet_pseudo_labels).to(
                        device=self.device,
                        dtype=torch.long,
                    )

            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_rare = 0.0
            epoch_valid = 0.0
            n_samples = 0
            for x_batch, x_raw_batch, sf_batch, batch_indices in dataloader:
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                batch_indices = batch_indices.to(self.device)

                z_tensor, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                batch_weights = None
                if epoch >= warmup_epochs:
                    batch_weights = weights_tensor[batch_indices]
                loss, _ = self.weighted_zinb_loss(
                    x=x_raw_tensor,
                    mean=mean_tensor,
                    disp=disp_tensor,
                    pi=pi_tensor,
                    scale_factor=sf_tensor,
                    cell_weights=batch_weights,
                )
                recon_loss = loss
                rare_loss = z_tensor.new_tensor(0.0)
                valid_anchors = 0
                if (
                    rare_triplet_weight > 0
                    and batch_weights is not None
                    and epoch >= warmup_epochs
                    and epoch >= rare_triplet_start_epoch
                ):
                    ramp_epochs = max(1, min(20, int(epochs) - rare_triplet_start_epoch))
                    rare_loss_ramp = min(1.0, (epoch - rare_triplet_start_epoch) / ramp_epochs)
                    rare_loss, valid_anchors = self._rare_triplet_loss(
                        z_tensor,
                        batch_indices,
                        batch_weights,
                        triplet_labels_tensor,
                        margin=rare_triplet_margin,
                        min_anchor_weight=rare_triplet_min_weight,
                        max_anchors=max_triplet_anchors,
                    )
                    loss = loss + (rare_loss_ramp * rare_triplet_weight * rare_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_size_eff = len(x_batch)
                epoch_total += float(loss.item()) * batch_size_eff
                epoch_recon += float(recon_loss.item()) * batch_size_eff
                epoch_rare += float(rare_loss.item()) * batch_size_eff
                epoch_valid += float(valid_anchors)
                n_samples += batch_size_eff

            avg_total = epoch_total / max(1, n_samples)
            avg_recon = epoch_recon / max(1, n_samples)
            avg_rare = epoch_rare / max(1, n_samples)
            self.pretrain_loss_history.append(avg_total)
            self.pretrain_recon_history.append(avg_recon)
            self.pretrain_rare_history.append(avg_rare)
            self.pretrain_valid_anchor_history.append(epoch_valid)
            self.pretrain_weight_history.append(
                self._summarize_cell_weights(
                    cell_weights,
                    epoch=epoch,
                    phase="pretrain_warmup" if epoch < warmup_epochs else "pretrain_weighted",
                    refreshed=weights_refreshed,
                    phase_epoch=epoch,
                )
            )
            print(
                "Pretrain epoch %3d, weighted ZINB loss: %.8f rare_triplet: %.8f"
                % (epoch + 1, avg_total, avg_rare)
            )

        if ae_save:
            torch.save(
                {
                    "ae_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ae_weights,
            )

    def fit(
        self,
        X,
        X_raw,
        size_factor,
        n_clusters,
        init_centroid=None,
        y=None,
        y_pred_init=None,
        lr=1.0,
        batch_size=256,
        num_epochs=10,
        update_interval=1,
        tol=1e-3,
    ):
        self.train()
        print("Clustering stage (scRAW-weighted)")
        X = torch.tensor(X, dtype=self.dtype)
        X_raw = torch.tensor(X_raw, dtype=self.dtype)
        size_factor = torch.tensor(size_factor, dtype=self.dtype)

        self.mu = Parameter(torch.zeros(n_clusters, self.z_dim, dtype=self.dtype, device=self.device))
        optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=0.95)

        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=self.dtype))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=self.dtype))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred

        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        weight_update_interval = int(max(0, self.weight_params.get("dynamic_weight_update_interval", 10)))
        cell_weights = np.ones(num, dtype=np.float32)
        weights_tensor = torch.ones(num, dtype=torch.float32, device=self.device)
        triplet_labels_tensor = torch.zeros(num, dtype=torch.long, device=self.device)
        rare_triplet_weight = float(self.weight_params.get("rare_triplet_weight", 0.0))
        pretrain_epochs_total = int(self.weight_params.get("pretrain_epochs", 0))
        rare_triplet_start_epoch = int(self.weight_params.get("rare_triplet_start_epoch", pretrain_epochs_total))
        rare_triplet_margin = float(self.weight_params.get("rare_triplet_margin", 0.4))
        rare_triplet_min_weight = float(self.weight_params.get("rare_triplet_min_weight", 1.2))
        max_triplet_anchors = int(self.weight_params.get("max_triplet_anchors_per_batch", 64))

        for epoch in range(int(num_epochs)):
            weights_refreshed = False
            if epoch % int(update_interval) == 0:
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                latent_np = latent.detach().cpu().numpy()
                if epoch == 0 or (
                    weight_update_interval > 0 and (epoch % weight_update_interval == 0)
                ):
                    cell_weights = self._refresh_cell_weights(
                        embeddings=latent_np,
                        n_clusters=n_clusters,
                        previous_weights=cell_weights.copy() if epoch > 0 else None,
                        epoch=epoch,
                        phase="cluster",
                    )
                    weights_refreshed = True
                    weights_tensor = torch.from_numpy(cell_weights).to(
                        device=self.device,
                        dtype=torch.float32,
                    )
                    if rare_triplet_weight > 0:
                        triplet_labels = self._compute_triplet_pseudo_labels(
                            latent_np,
                            n_clusters,
                            fallback_labels=self.current_pseudo_labels,
                        )
                        self.current_triplet_pseudo_labels = triplet_labels.astype(np.int64)
                        triplet_labels_tensor = torch.from_numpy(self.current_triplet_pseudo_labels).to(
                            device=self.device,
                            dtype=torch.long,
                        )

                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred

                if epoch > 0 and delta_label < tol:
                    print(f"Clustering converged at epoch {epoch}.")
                    break

            if epoch % 10 == 0:
                print(f"Clustering epoch {epoch + 1}/{num_epochs} (Update interval: {update_interval})")

            epoch_total = 0.0
            epoch_recon = 0.0
            epoch_kl = 0.0
            epoch_rare = 0.0
            epoch_valid = 0.0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, num)
                xbatch = X[start:end]
                xrawbatch = X_raw[start:end]
                sfbatch = size_factor[start:end]
                pbatch = p[start:end]
                batch_weights = weights_tensor[start:end]
                batch_indices = torch.arange(start, end, dtype=torch.long, device=self.device)

                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)
                cluster_loss = self.cluster_loss(target, qbatch)
                effective_cluster = cluster_loss * self.gamma
                recon_loss, _ = self.weighted_zinb_loss(
                    rawinputs,
                    meanbatch,
                    dispbatch,
                    pibatch,
                    sfinputs,
                    cell_weights=batch_weights,
                )

                loss = effective_cluster + recon_loss
                rare_loss = qbatch.new_tensor(0.0)
                valid_anchors = 0
                global_epoch = pretrain_epochs_total + epoch
                if rare_triplet_weight > 0 and global_epoch >= rare_triplet_start_epoch:
                    ramp_epochs = max(1, min(20, pretrain_epochs_total + int(num_epochs) - rare_triplet_start_epoch))
                    rare_loss_ramp = min(1.0, (global_epoch - rare_triplet_start_epoch) / ramp_epochs)
                    rare_loss, valid_anchors = self._rare_triplet_loss(
                        zbatch,
                        batch_indices,
                        batch_weights,
                        triplet_labels_tensor,
                        margin=rare_triplet_margin,
                        min_anchor_weight=rare_triplet_min_weight,
                        max_anchors=max_triplet_anchors,
                    )
                    loss = loss + (rare_loss_ramp * rare_triplet_weight * rare_loss)
                loss.backward()
                optimizer.step()

                batch_size_eff = len(xbatch)
                epoch_total += float(loss.item()) * batch_size_eff
                epoch_recon += float(recon_loss.item()) * batch_size_eff
                epoch_kl += float(effective_cluster.item()) * batch_size_eff
                epoch_rare += float(rare_loss.item()) * batch_size_eff
                epoch_valid += float(valid_anchors)

            self.cluster_loss_history.append(epoch_total / num)
            self.cluster_recon_history.append(epoch_recon / num)
            self.cluster_kl_history.append(epoch_kl / num)
            self.cluster_rare_history.append(epoch_rare / num)
            self.cluster_valid_anchor_history.append(epoch_valid)
            self.cluster_weight_history.append(
                self._summarize_cell_weights(
                    cell_weights,
                    epoch=pretrain_epochs_total + epoch,
                    phase="cluster",
                    refreshed=weights_refreshed,
                    phase_epoch=epoch,
                )
            )

        self.eval()
        with torch.no_grad():
            latent = self.encodeBatch(X.to(self.device))

        return self.y_pred, latent.cpu().numpy()


@AlgorithmRegistry.register
class ScDeepClusterScrawWeightedAlgorithm(ScDeepClusterAlgorithm):
    """scDeepCluster with scRAW-style dynamic reconstruction weighting."""

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name="scdeepcluster_scraw_weighted",
            display_name="scDeepCluster + scRAW Weighted Loss",
            description=(
                "Original scDeepCluster architecture and preprocessing with "
                "scRAW-inspired dynamic per-cell weighting on the ZINB reconstruction term."
            ),
            category="deep_learning",
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes=(
                "Same preprocessing path as scDeepCluster. The weighted phase estimates "
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
                    display_name="Weighted Warm-up Epochs",
                    param_type=ParamType.INTEGER,
                    default=30,
                    description=(
                        "Pretraining epochs run exactly like baseline scDeepCluster "
                        "before enabling scRAW-style cell weighting."
                    ),
                    min_value=0,
                    max_value=400,
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
                        "density kNN component without reconstruction pseudo-labels."
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
                    name="weight_n_clusters",
                    display_name="Weight Pseudo K",
                    param_type=ParamType.INTEGER,
                    default=0,
                    description=(
                        "Pseudo-label cluster count used during weighted pretraining "
                        "(0 = use requested/oracle k when available, else fallback to 8)."
                    ),
                    min_value=0,
                    max_value=100,
                    step=1,
                    category="Weighting",
                    advanced=True,
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
                    max_value=1000,
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
                    description="Pseudo-label generator used only for the optional triplet loss.",
                    category="Triplet",
                    advanced=True,
                ),
            ]
        )
        return params

    def _resolve_weight_k(self, requested_n_clusters: int, labels: Optional[np.ndarray]) -> int:
        manual_k = int(self.params.get("weight_n_clusters", 0))
        if manual_k > 1:
            return manual_k
        if int(requested_n_clusters) > 1:
            return int(requested_n_clusters)
        if bool(self.params.get("use_ground_truth_k", True)) and labels is not None:
            return int(len(np.unique(labels)))
        return 8

    def fit(self, data: Any, labels: Optional[Any] = None) -> "ScDeepClusterScrawWeightedAlgorithm":
        seed = self.params.get("random_state", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device(self.get_device())
        force_float32 = self.params.get("force_float32", False)
        np_dtype = np.float32 if device.type == "mps" or force_float32 else np.float64
        if force_float32 and device.type != "mps":
            logger.info("force_float32 enabled: using float32 for cross-platform reproducibility")

        enc_layers = [int(x) for x in self.params.get("encoder_layers", "256,64").split(",")]
        dec_layers = [int(x) for x in self.params.get("decoder_layers", "64,256").split(",")]

        use_raw_data = self.params.get("use_raw_data", True)
        if hasattr(data, "X"):
            if use_raw_data and hasattr(data, "layers") and "original_X" in data.layers:
                X_counts = data.layers["original_X"]
            elif use_raw_data and hasattr(data, "raw") and data.raw is not None:
                X_counts = data.raw.X
            else:
                X_counts = data.X
            if hasattr(X_counts, "toarray"):
                X_counts = X_counts.toarray()
        else:
            X_counts = data

        if np.issubdtype(X_counts.dtype, np.floating) and np.allclose(X_counts, np.round(X_counts)):
            X_counts = np.round(X_counts)
        X_counts = X_counts.astype(np_dtype)

        select_genes = self.params.get("select_genes", 0)
        if select_genes is not None and select_genes > 0:
            gene_mask = geneSelection(X_counts, n=select_genes, verbose=0)
            X_counts = X_counts[:, gene_mask]

        import scanpy as sc

        adata = sc.AnnData(X_counts, dtype=np_dtype)
        adata.obs_names = [str(i) for i in range(adata.n_obs)]
        if labels is not None:
            adata.obs["Group"] = labels

        adata = read_dataset(adata, transpose=False, test_split=False, copy=True)

        if use_raw_data:
            adata.raw = adata.copy()
            sc.pp.normalize_per_cell(adata)
            n_counts = adata.obs.n_counts if "n_counts" in adata.obs.columns else adata.obs["n_counts"]
            adata.obs["size_factors"] = n_counts / np.median(n_counts)
            sc.pp.log1p(adata)
            self.scaler = StandardScaler()
            X_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            adata.X = self.scaler.fit_transform(X_mat)
        else:
            if adata.raw is None:
                adata.raw = adata
            adata.obs["size_factors"] = 1.0
            self.scaler = StandardScaler()
            X_mat = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            adata.X = self.scaler.fit_transform(X_mat)

        self._valid_indices = adata.obs_names.values.astype(int)

        X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        X = X.astype(np_dtype)

        X_raw = adata.raw.X.toarray() if hasattr(adata.raw.X, "toarray") else adata.raw.X
        X_raw = X_raw.astype(np_dtype)

        if labels is not None:
            labels = adata.obs["Group"].values

        size_factor = np.array(adata.obs["size_factors"]).astype(np_dtype)
        input_dim = X.shape[1]
        self._n_features = input_dim

        n_clusters = int(self.params.get("n_clusters", 0))
        weight_params = dict(self.params)
        weight_params["random_state"] = int(seed)

        self.model = ScDeepClusterScrawWeighted(
            input_dim=input_dim,
            z_dim=self.params.get("z_dim", 32),
            encodeLayer=enc_layers,
            decodeLayer=dec_layers,
            activation=self.params.get("activation", "relu"),
            sigma=self.params.get("sigma", 1.0),
            alpha=self.params.get("alpha", 1.0),
            gamma=self.params.get("gamma", 1.0),
            device=device.type,
            force_float32=force_float32,
            weight_params=weight_params,
        )

        pretrain_weight_k = self._resolve_weight_k(n_clusters, labels)
        self.model.pretrain_autoencoder(
            X,
            X_raw,
            size_factor,
            batch_size=self.params.get("batch_size", 256),
            lr=self.params.get("pretrain_lr", 0.001),
            epochs=self.params.get("pretrain_epochs", 400),
            n_clusters_for_weights=pretrain_weight_k,
        )

        use_ground_truth_k = self.params.get("use_ground_truth_k", True)

        if use_ground_truth_k and labels is not None:
            n_clusters = len(np.unique(labels))
            logger.warning(
                "⚠️ ORACLE MODE ENABLED: scDeepCluster+scRAW weighted using ground truth k=%s. "
                "Results are NOT comparable to methods that estimate k independently.",
                n_clusters,
            )
            self._labels, self._embeddings = self.model.fit(
                X,
                X_raw,
                size_factor,
                n_clusters=n_clusters,
                y=labels,
                batch_size=self.params.get("batch_size", 256),
                num_epochs=self.params.get("maxiter", 2000),
                lr=self.params.get("lr", 1.0),
                update_interval=self.params.get("update_interval", 1),
                tol=self.params.get("tol", 0.001),
            )
        elif use_ground_truth_k and labels is None:
            raise ValueError(
                "use_ground_truth_k=True but no labels provided. "
                "Either provide labels or set use_ground_truth_k=False."
            )
        elif n_clusters == 0:
            import pandas as pd

            pretrain_latent = self.model.encodeBatch(torch.tensor(X, dtype=self.model.dtype)).cpu().numpy()
            adata_latent = sc.AnnData(pretrain_latent)
            sc.pp.neighbors(
                adata_latent,
                n_neighbors=self.params.get("knn", 20),
                use_rep="X",
            )
            resolution_param = float(self.params.get("resolution", 0.0))
            if resolution_param > 0:
                resolution_used = resolution_param
                selection_mode = "manual"
            else:
                resolution_used = float(self._search_optimal_leiden_resolution(adata_latent, pretrain_latent))
                selection_mode = "auto_silhouette"

            sc.tl.leiden(
                adata_latent,
                random_state=seed,
                resolution=resolution_used,
            )
            y_pred_init = adata_latent.obs["leiden"].astype(int).values
            self.params["leiden_resolution_selected"] = float(resolution_used)
            self.params["leiden_resolution_selection_mode"] = selection_mode

            features = pd.DataFrame(adata_latent.X, index=np.arange(adata_latent.n_obs))
            group = pd.Series(y_pred_init, index=np.arange(adata_latent.n_obs), name="Group")
            cluster_centers = pd.concat([features, group], axis=1).groupby("Group").mean().to_numpy()

            n_clusters = cluster_centers.shape[0]
            print("Estimated number of clusters: ", n_clusters)

            self._labels, self._embeddings = self.model.fit(
                X,
                X_raw,
                size_factor,
                n_clusters=n_clusters,
                init_centroid=cluster_centers,
                y_pred_init=y_pred_init,
                y=labels,
                batch_size=self.params.get("batch_size", 256),
                num_epochs=self.params.get("maxiter", 2000),
                lr=self.params.get("lr", 1.0),
                update_interval=self.params.get("update_interval", 1),
                tol=self.params.get("tol", 0.001),
            )
        else:
            self._labels, self._embeddings = self.model.fit(
                X,
                X_raw,
                size_factor,
                n_clusters=n_clusters,
                y=labels,
                batch_size=self.params.get("batch_size", 256),
                num_epochs=self.params.get("maxiter", 2000),
                lr=self.params.get("lr", 1.0),
                update_interval=self.params.get("update_interval", 1),
                tol=self.params.get("tol", 0.001),
            )

        self._loss_history = []
        if getattr(self.model, "pretrain_loss_history", None):
            self._loss_history.append(
                {
                    "name": "pretrain (weighted ZINB)",
                    "epochs": list(range(len(self.model.pretrain_loss_history))),
                    "train_loss": list(self.model.pretrain_loss_history),
                    "components": {
                        "zinb": list(self.model.pretrain_recon_history or self.model.pretrain_loss_history),
                        "rare_triplet": list(getattr(self.model, "pretrain_rare_history", [])),
                        "triplet_valid_anchors": list(getattr(self.model, "pretrain_valid_anchor_history", [])),
                        "weight_mean": [row["mean"] for row in self.model.pretrain_weight_history],
                        "weight_std": [row["std"] for row in self.model.pretrain_weight_history],
                        "weight_min": [row["min"] for row in self.model.pretrain_weight_history],
                        "weight_max": [row["max"] for row in self.model.pretrain_weight_history],
                    },
                }
            )
        if getattr(self.model, "cluster_loss_history", None):
            n_pretrain = len(self.model.pretrain_loss_history)
            self._loss_history.append(
                {
                    "name": "clustering (KL+weighted ZINB)",
                    "epochs": list(range(n_pretrain, n_pretrain + len(self.model.cluster_loss_history))),
                    "train_loss": list(self.model.cluster_loss_history),
                    "components": {
                        "zinb": list(self.model.cluster_recon_history),
                        "effective_kl": list(self.model.cluster_kl_history),
                        "rare_triplet": list(getattr(self.model, "cluster_rare_history", [])),
                        "triplet_valid_anchors": list(getattr(self.model, "cluster_valid_anchor_history", [])),
                        "weight_mean": [row["mean"] for row in self.model.cluster_weight_history],
                        "weight_std": [row["std"] for row in self.model.cluster_weight_history],
                        "weight_min": [row["min"] for row in self.model.cluster_weight_history],
                        "weight_max": [row["max"] for row in self.model.cluster_weight_history],
                    },
                }
            )

        self._weight_history = list(self.model.pretrain_weight_history) + list(self.model.cluster_weight_history)

        self._fitted = True
        return self
