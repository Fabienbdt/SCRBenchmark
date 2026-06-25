"""
Batch correction utilities using scVI-DCT (Domain Class Triplet Loss).

This module provides functions to apply batch correction to scRNA-seq data
using scVI with the custom Domain Class Triplet Loss training plan.

The DCT loss encourages:
- Cells of the same type from different batches to have similar representations
- Cells of different types from the same batch to be separable

Reference:
- Domain-aware triplet loss: Kaiyu Guo, Brian C. Lovell
  https://github.com/workerbcd/DCT/blob/main/domainbed/losses/tripletloss.py
"""

import numpy as np
import inspect
import logging
import random
import scipy.sparse as sp
from typing import Optional, Dict, Any, Tuple, Literal

logger = logging.getLogger(__name__)

# Check if scVI is available
try:
    import scvi
    from scvi.train._trainingplans import TrainingPlan
    from scvi.module.base import BaseModuleClass
    from scvi import REGISTRY_KEYS
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch
    from torch import nn
    SCVI_AVAILABLE = True
except ImportError as e:
    SCVI_AVAILABLE = False
    TrainingPlan = object
    BaseModuleClass = Any  # Type alias for when scVI not available
    REGISTRY_KEYS = None
    torch = None
    nn = None
    ReduceLROnPlateau = None
    logger.warning(f"scVI not installed or import failed. Error: {e}. Batch correction will not be available. "
                  "Install with: pip install scvi-tools")


# Type alias for optimizer creator
TorchOptimizerCreator = Any

# Number of PCA components to use for correlation MSE loss
N_PCA_COMPONENTS_FOR_CORR = 50


def compute_per_batch_pca(
    adata,
    batch_key: str = 'batch',
    n_comps: int = N_PCA_COMPONENTS_FOR_CORR,
) -> np.ndarray:
    """Compute PCA independently for each batch, then concatenate.

    This follows the scIB-E preprocessing strategy (data_preprocess.py:23-30)
    where PCA is computed per-batch so that the correlation structure reflects
    batch-specific gene relationships rather than global (batch-mixed) ones.

    Args:
        adata: AnnData object (preprocessed, normalized, log-transformed).
        batch_key: Column in ``adata.obs`` identifying batches.
        n_comps: Number of principal components per batch.

    Returns:
        np.ndarray of shape ``(n_cells, n_comps)`` with per-batch PCA
        embeddings concatenated in the original cell order.
    """
    import scanpy as sc

    n_comps_safe = min(n_comps, adata.n_vars - 1)
    X_pca_batch = np.zeros((adata.n_obs, n_comps_safe), dtype=np.float32)
    batches = adata.obs[batch_key].unique()

    for batch_id in batches:
        mask = (adata.obs[batch_key] == batch_id).values
        n_cells_batch = int(mask.sum())
        # Ensure n_comps does not exceed batch dimensions
        n_comps_batch = min(n_comps_safe, n_cells_batch - 1)
        if n_comps_batch < 1:
            logger.warning(
                f"Batch '{batch_id}' has only {n_cells_batch} cell(s). "
                "Skipping PCA for this batch (zeros will be used)."
            )
            continue

        adata_batch = adata[mask].copy()
        sc.pp.pca(adata_batch, n_comps=n_comps_batch, svd_solver='arpack')
        pca_result = adata_batch.obsm['X_pca'].astype(np.float32)

        # If this batch yielded fewer components, pad with zeros
        if pca_result.shape[1] < n_comps_safe:
            padded = np.zeros((n_cells_batch, n_comps_safe), dtype=np.float32)
            padded[:, :pca_result.shape[1]] = pca_result
            pca_result = padded

        X_pca_batch[mask] = pca_result

    logger.info(
        f"Computed per-batch PCA ({len(batches)} batches, {n_comps_safe} PCs) "
        f"following scIB-E methodology"
    )
    return X_pca_batch


def check_scvi_available() -> bool:
    """Check if scVI is available for batch correction."""
    return SCVI_AVAILABLE


def _scvi_supports_plan_class(model) -> bool:
    """Return True if scvi-tools train() accepts plan_class (older versions)."""
    try:
        return "plan_class" in inspect.signature(model.train).parameters
    except (TypeError, ValueError):
        return False


def _set_scvi_training_plan(model, plan_class) -> bool:
    """Set the training plan class on the model for scvi-tools >= 1.0."""
    if hasattr(model, "_training_plan_cls"):
        model._training_plan_cls = plan_class
        return True
    if hasattr(model, "training_plan_class"):
        model.training_plan_class = plan_class
        return True
    return False


def _train_scvi_model(
    model,
    *,
    max_epochs: int,
    batch_size: int,
    accelerator: str,
    devices: Any,
    plan_class=None,
    plan_kwargs: Optional[Dict[str, Any]] = None,
    **trainer_kwargs,
) -> bool:
    """Train scvi-tools model with optional custom training plan."""
    train_kwargs: Dict[str, Any] = {
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "accelerator": accelerator,
        "devices": devices,
    }
    if plan_kwargs is not None:
        train_kwargs["plan_kwargs"] = plan_kwargs
    if plan_class is not None:
        if _scvi_supports_plan_class(model):
            train_kwargs["plan_class"] = plan_class
        elif not _set_scvi_training_plan(model, plan_class):
            return False
    train_kwargs.update(trainer_kwargs)
    model.train(**train_kwargs)
    return True


_DCT_LABEL_CANDIDATES = [
    "cell_type",
    "celltype",
    "cell_types",
    "Group",
    "labels",
    "labels_encoded",
    "label",
    "CellType",
    "original_celltype",
    "cluster",
    "clusters",
    "assigned_cluster",
]


def _is_plausible_label_series(series, n_obs: int) -> bool:
    """Heuristic guard to avoid using continuous numeric columns as labels."""
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1:
        return False

    # Numeric labels are allowed only if cardinality looks like class labels.
    try:
        is_numeric = np.issubdtype(series.dtype, np.number)
    except TypeError:
        is_numeric = False
    if is_numeric:
        numeric_unique_limit = max(20, min(500, int(np.sqrt(max(n_obs, 1)) * 4)))
        return n_unique <= numeric_unique_limit

    return True


def _resolve_dct_labels_key(
    adata,
    batch_key: str,
    labels_key: Optional[str],
) -> Optional[str]:
    """Resolve the labels column used by DCT with safe auto-detection."""
    if not hasattr(adata, "obs"):
        return None

    obs = adata.obs

    if labels_key is not None:
        if labels_key in obs.columns and _is_plausible_label_series(obs[labels_key], adata.n_obs):
            return labels_key
        logger.warning(
            f"Provided labels_key '{labels_key}' is missing or not a plausible label column. "
            "Attempting automatic detection."
        )

    for key in _DCT_LABEL_CANDIDATES:
        if key == batch_key:
            continue
        if key in obs.columns and _is_plausible_label_series(obs[key], adata.n_obs):
            return key

    for key in obs.columns:
        key_str = str(key)
        if key_str == batch_key:
            continue
        key_lower = key_str.lower()
        if not any(tok in key_lower for tok in ("cell", "label", "cluster", "annot", "type")):
            continue
        if _is_plausible_label_series(obs[key], adata.n_obs):
            return key

    return None


def _torch_corrcoef_fallback(matrix: 'torch.Tensor') -> 'torch.Tensor':
    """
    Fallback implementation of torch.corrcoef for PyTorch < 1.10.

    Computes Pearson correlation coefficient matrix between rows (samples).
    Matches torch.corrcoef behavior where each row is a variable.

    Args:
        matrix: Input tensor of shape (n_samples, n_features)

    Returns:
        Correlation matrix of shape (n_samples, n_samples)
    """
    if matrix.dim() == 1:
        return torch.tensor(1.0, device=matrix.device)

    # Center the data along features (dim 1)
    # matrix is (N, F), mean is (N, 1)
    matrix_centered = matrix - matrix.mean(dim=1, keepdim=True)

    # Compute covariance (N, N)
    n_features = matrix.size(1)
    if n_features <= 1:
        return torch.ones(matrix.size(0), matrix.size(0), device=matrix.device)
        
    cov = torch.mm(matrix_centered, matrix_centered.T) / (n_features - 1)

    # Compute standard deviations
    std = torch.sqrt(torch.diag(cov))
    std = std.unsqueeze(1)

    # Compute correlation with numerical stability
    # Using a slightly larger epsilon for denominator stability
    denominator = std @ std.T + 1e-8
    corr = cov / denominator

    # Clamp to [-1, 1] range to handle precision issues
    corr = torch.clamp(corr, -1.0, 1.0)

    return corr


def safe_corrcoef(matrix: 'torch.Tensor') -> 'torch.Tensor':
    """
    Compute correlation coefficient matrix with fallback and stability fixes.

    Always computes row-wise (cell-cell) correlation.

    Args:
        matrix: Input tensor (N samples x F features)

    Returns:
        Correlation matrix (N x N)
    """
    # Force float32 for correlation calculation to prevent NaNs on MPS/stable numericals
    # In scVI, we might be in float32 already but let's be explicit
    original_dtype = matrix.dtype
    if original_dtype != torch.float32:
        matrix = matrix.to(torch.float32)

    if hasattr(torch, 'corrcoef'):
        # torch.corrcoef computes correlation between rows
        res = torch.corrcoef(matrix)
    else:
        res = _torch_corrcoef_fallback(matrix)
        
    if res.dtype != original_dtype:
        res = res.to(original_dtype)
        
    return res


# =============================================================================
# Domain Class Triplet Loss Training Plan (from user's implementation)
# =============================================================================

class DomainClassTripletLoss_TrainingPlan(TrainingPlan if SCVI_AVAILABLE else object):
    """Training plan for VAEs with Domain class triplet loss to encourage latent space mixing.

    Parameters
    ----------
    module
        A module instance from class ``BaseModuleClass``.
    optimizer
        One of "Adam", "AdamW", or "Custom".
    optimizer_creator
        A callable taking in parameters and returning an optimizer.
    lr
        Learning rate used for optimization.
    weight_decay
        Weight decay used in optimization.
    n_steps_kl_warmup
        Number of training steps to scale KL weight from 0 to 1.
    n_epochs_kl_warmup
        Number of epochs to scale KL weight from 0 to 1.
    reduce_lr_on_plateau
        Whether to reduce learning rate on plateau.
    lr_factor
        Factor to reduce learning rate.
    lr_patience
        Number of epochs with no improvement before reducing lr.
    lr_threshold
        Threshold for measuring improvement.
    lr_scheduler_metric
        Which metric to track for lr reduction.
    lr_min
        Minimum learning rate allowed.
    batch_correction_weight
        Weights dict with 'batch_celltype_constraint_weight' and 'corr_mse_weight'.
    **loss_kwargs
        Additional kwargs for the loss method.
    """

    def __init__(
        self,
        module,  # BaseModuleClass when scVI available
        *,
        optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        n_steps_kl_warmup: int = None,
        n_epochs_kl_warmup: int = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        lr_min: float = 0,
        batch_correction_weight: dict = None,
        **loss_kwargs,
    ):
        if not SCVI_AVAILABLE:
            raise ImportError("scVI is required for DomainClassTripletLoss_TrainingPlan.")

        super().__init__(
            module=module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )
        self.automatic_optimization = False

        # Batch correction weights
        if batch_correction_weight is None:
            batch_correction_weight = {
                "batch_celltype_constraint_weight": 1.0,
                "corr_mse_weight": 0.1
            }

        self.batch_correction_weight = batch_correction_weight.get(
            "batch_celltype_constraint_weight", 1.0
        )
        self.corr_mse_weight = batch_correction_weight.get("corr_mse_weight", 0.1)

        # Triplet loss parameters
        self.hard_factor = 0
        self.margin = None
        self.leri = False

        if self.margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

        self.mse_loss = torch.nn.MSELoss()

    def euclidean_dist(self, x, y):
        """Compute euclidean distance between two tensors."""
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = dist - 2 * torch.matmul(x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def domain_hard_sample_mining(self, dist_mat, labels, domains, leri=None):
        """Mine hard positive and negative samples based on domain and label.

        This method finds hard positive/negative pairs for triplet loss:
        - Hard positive: same label, different domain (should be close)
        - Hard negative: different label, same domain (should be far)

        Args:
            dist_mat: Distance matrix (N x N)
            labels: Cell type labels (N,)
            domains: Batch/domain labels (N,)
            leri: Legacy parameter for domain mixing

        Returns:
            Tuple of (dist_dnlp, dist_dpln) for triplet loss computation
        """
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # Use device from input tensor (supports cuda, mps, cpu)
        device = dist_mat.device

        is_domain_pos = domains.expand(N, N).eq(domains.expand(N, N).t())
        is_domain_neg = domains.expand(N, N).ne(domains.expand(N, N).t())
        is_2 = domains.expand(N, N).eq(torch.ones(N, N, device=device))
        is_label_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_label_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        domainpos_labelneg = is_label_neg & is_domain_pos
        domainneg_labelpos = is_label_pos & is_domain_neg

        if leri:
            is_domain_pos = is_domain_pos | is_2
            is_domain_neg = is_domain_neg | is_2
            domainpos_labelneg = is_label_neg & is_domain_pos
            domainneg_labelpos = is_label_pos & is_domain_neg

        dist_dpln, dist_dnlp = [], []
        for i in range(N):
            if dist_mat[i][domainneg_labelpos[i]].shape[0] != 0:
                dist_dnlp.append(
                    torch.max(dist_mat[i][domainneg_labelpos[i]].contiguous(), 0, keepdim=True)[0]
                )
            else:
                # Use device from dist_mat instead of hardcoded .cuda()
                dist_dnlp.append(torch.zeros(1, device=device))

            if dist_mat[i][domainpos_labelneg[i]].shape[0] != 0:
                dist_dpln.append(
                    torch.min(dist_mat[i][domainpos_labelneg[i]].contiguous(), 0, keepdim=True)[0]
                )
            else:
                # Use device from dist_mat instead of hardcoded .cuda()
                dist_dpln.append(torch.zeros(1, device=device))

        dist_dnlp = torch.cat(dist_dnlp).clamp(min=1e-12)
        dist_dpln = torch.cat(dist_dpln).clamp(min=1e-12)

        return dist_dnlp, dist_dpln

    def domain_triplet_loss(self, global_feat, labels, batch_labels):
        """Compute domain class triplet loss."""
        dist_mat = self.euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = self.domain_hard_sample_mining(
            dist_mat, labels, batch_labels, leri=self.leri
        )

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss

    def calculate_corr_mse(self, matrix1, matrix2, mask=None):
        """Calculate MSE between correlation matrices.

        This preserves gene-gene correlation structure during batch correction.
        Uses safe_corrcoef for PyTorch version compatibility.

        Args:
            matrix1: Original PCA embeddings (n_cells x n_pcs)
            matrix2: Latent embeddings z (n_cells x n_latent)
            mask: Optional mask for subset of correlations

        Returns:
            MSE loss between correlation matrices
        """
        corr_matrix1 = safe_corrcoef(matrix1)
        corr_matrix2 = safe_corrcoef(matrix2)
        if mask is not None:
            corr_matrix1 = corr_matrix1[mask]
            corr_matrix2 = corr_matrix2[mask]
        corr_mse_loss = self.mse_loss(corr_matrix2, corr_matrix1)
        return corr_mse_loss

    def training_step(self, batch, batch_idx):
        """Training step with Domain Class Triplet Loss.

        Uses CONT_COVS_KEY to access per-batch PCA data registered as continuous
        covariates (equivalent to scIB-E's PCA_KEY via ObsmField).
        """
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})

        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]
        celltype_labels = batch[REGISTRY_KEYS.LABELS_KEY]
        # Per-batch PCA registered as continuous covariates, access via CONT_COVS_KEY
        X_pca = batch[REGISTRY_KEYS.CONT_COVS_KEY]

        opt1 = self.optimizers()
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)

        loss = scvi_loss.loss
        z = inference_outputs["z"]

        # Domain Class Triplet Loss
        dct_loss = self.batch_correction_weight * self.domain_triplet_loss(
            z, celltype_labels.squeeze(1), batch_tensor.squeeze(1)
        )
        loss += dct_loss

        # Correlation MSE Loss (per batch)
        total_corr_mse_loss = 0.0
        unique_batch_labels = torch.unique(batch_tensor)

        for lbl in unique_batch_labels:
            mask = batch_tensor == lbl
            mask = mask.squeeze()
            X_pca_batch = X_pca[mask]
            z_batch = z[mask]
            corr_mse_loss = self.corr_mse_weight * self.calculate_corr_mse(X_pca_batch, z_batch)
            total_corr_mse_loss += corr_mse_loss

        loss += total_corr_mse_loss

        # Logging
        self.log("corr_mse_loss", total_corr_mse_loss, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("DomainClassTripletLoss_loss", dct_loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")

        # Manual optimization
        opt1.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        # Manually clip gradients since automatic_optimization=False
        self.clip_gradients(opt1, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt1.step()

    def validation_step(self, batch, batch_idx):
        """Validation step for the model.

        Uses CONT_COVS_KEY to access per-batch PCA data registered as continuous
        covariates (equivalent to scIB-E's PCA_KEY via ObsmField).
        """
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]
        celltype_labels = batch[REGISTRY_KEYS.LABELS_KEY]
        # Per-batch PCA registered as continuous covariates, access via CONT_COVS_KEY
        X_pca = batch[REGISTRY_KEYS.CONT_COVS_KEY]

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        loss = scvi_loss.loss

        z = inference_outputs["z"]
        dct_loss = self.batch_correction_weight * self.domain_triplet_loss(
            z, celltype_labels.squeeze(1), batch_tensor.squeeze(1)
        )
        loss += dct_loss

        corr_mse_loss = self.corr_mse_weight * self.calculate_corr_mse(X_pca, z)
        loss += corr_mse_loss

        self.log(
            "validation_loss",
            loss,
            on_epoch=True,
            sync_dist=self.use_sync_dist,
        )
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")

    def on_train_epoch_end(self):
        """Update learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        """Update learning rate via scheduler steps."""
        if not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for training."""
        parameters_to_optimize = list(self.module.parameters())
        params1 = filter(lambda p: p.requires_grad, parameters_to_optimize)
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}

        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update({
                "lr_scheduler": {
                    "scheduler": scheduler1,
                    "monitor": self.lr_scheduler_metric,
                },
            })

        return config1


# =============================================================================
# Batch Correction Function
# =============================================================================

def apply_scvi_batch_correction(
    adata,
    batch_key: str = 'batch',
    labels_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    device: str = 'auto'
) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Apply scVI-DCT batch correction to data.
    
    Uses the Domain Class Triplet Loss training plan to learn batch-corrected
    representations. This requires cell type labels for optimal performance.
    
    Args:
        adata: AnnData object (preprocessed, normalized, log-transformed)
        batch_key: Column in obs with batch information
        labels_key: Column with cell type labels (required for DCT loss)
        params: Dict with training parameters:
            - n_latent: Latent dimension (default: 30)
            - n_hidden: Hidden layer size (default: 128)
            - n_layers: Number of hidden layers (default: 2)
            - max_epochs: Training epochs (default: 100)
            - batch_size: Training batch size (default: 128)
            - learning_rate: Learning rate (default: 1e-3)
            - dct_weight: Weight for DCT loss (default: 1.0)
            - corr_mse_weight: Weight for correlation MSE loss (default: 0.1)
        device: Compute device ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        Tuple of:
            - adata: AnnData with X replaced by corrected expression
            - embeddings: Latent space embeddings (n_cells x n_latent)
    
    Raises:
        ImportError: If scVI is not installed
        ValueError: If batch_key not found in adata.obs
    """
    if not SCVI_AVAILABLE:
        raise ImportError(
            "scVI is required for batch correction. "
            "Install with: pip install scvi-tools"
        )

    import scanpy as sc

    params = params or {}

    # IMPORTANT: Work on a copy to avoid modifying the original adata
    # This is critical for Streamlit where users may want to go back/forth
    adata = adata.copy()

    # Validate batch key
    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"Batch column '{batch_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    # Check if there are actually multiple batches
    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        logger.warning(
            f"Only {n_batches} batch(es) found. Batch correction may have no effect."
        )
    
    # Determine labels key - DCT loss requires labels.
    # If not explicitly provided, auto-detect a plausible cell-type column.
    resolved_labels_key = _resolve_dct_labels_key(
        adata=adata,
        batch_key=batch_key,
        labels_key=labels_key,
    )
    if resolved_labels_key is None:
        logger.info("No valid labels_key found. Using dummy labels (Unsupervised Mode).")
        adata.obs['_dummy_labels'] = 0
        labels_key = '_dummy_labels'
    else:
        labels_key = resolved_labels_key
        logger.info(f"Using labels_key '{labels_key}' for DCT supervision.")
    
    logger.info(f"Starting scVI-DCT batch correction with {n_batches} batches")
    
    # Get device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    # Model parameters
    n_latent = params.get('n_latent', 30)
    n_hidden = params.get('n_hidden', 128)
    n_layers = params.get('n_layers', 2)
    max_epochs = params.get('max_epochs', 100)
    batch_size = params.get('batch_size', 128)
    lr = params.get('learning_rate', 1e-3)
    dct_weight = params.get('dct_weight', 1.0)
    corr_mse_weight = params.get('corr_mse_weight', 0.1)
    
    # Prepare data for scVI
    # IMPORTANT: scVI is a generative model assuming raw count data (Negative Binomial)
    # Using log-normalized data violates model assumptions. We extract raw counts here.
    adata_scvi = adata.copy()
    
    # Extract raw counts for scVI (critical for theoretical correctness)
    raw_counts_found = False
    if 'original_X' in adata.layers:
        # Preferred: use original_X layer which contains raw counts
        X_raw = adata.layers['original_X']
        if sp.issparse(X_raw):
            X_raw = X_raw.toarray()
        adata_scvi.X = X_raw.copy()
        raw_counts_found = True
        logger.info("Using raw counts from layers['original_X'] for scVI (correct for NB model)")
    elif adata.raw is not None:
        # Fallback: use adata.raw
        try:
            X_raw = adata.raw.X
            if sp.issparse(X_raw):
                X_raw = X_raw.toarray()
            # Check if raw has same genes as current adata
            if X_raw.shape[1] == adata.n_vars:
                adata_scvi.X = X_raw.copy()
                raw_counts_found = True
                logger.info("Using raw counts from adata.raw for scVI (correct for NB model)")
            else:
                logger.warning(f"adata.raw has different gene count ({X_raw.shape[1]} vs {adata.n_vars}). "
                              "Using preprocessed data (suboptimal for scVI).")
        except Exception as e:
            logger.warning(f"Could not extract raw counts from adata.raw: {e}")
    
    if not raw_counts_found:
        # Check if current X looks like raw counts (integers)
        X_current = adata_scvi.X
        if sp.issparse(X_current):
            X_check = X_current.data[:1000] if len(X_current.data) > 1000 else X_current.data
        else:
            X_check = X_current.ravel()[:1000]

        is_likely_raw = np.allclose(X_check, np.round(X_check), rtol=1e-5)

        if is_likely_raw:
            logger.info("Current adata.X appears to contain integer counts. Using as raw counts for scVI.")
        else:
            logger.error(
                "CRITICAL: Raw counts not available and current data appears normalized (floats). "
                "scVI requires raw counts for its Negative Binomial model. "
                "Results may be scientifically invalid. "
                "Fix: Ensure raw counts are stored in layers['original_X'] before batch correction."
            )
            raise ValueError(
                "scVI batch correction requires raw counts but only normalized data is available. "
                "Please ensure raw counts are preserved in layers['original_X'] or adata.raw."
            )
    
    # Compute per-batch PCA on preprocessed data for correlation MSE loss (required by DCT).
    # Following the scIB-E methodology (data_preprocess.py:23-30), PCA is computed
    # independently for each batch so that correlation structures reflect batch-specific
    # gene relationships rather than global (batch-mixed) ones.
    X_pca_batch = compute_per_batch_pca(adata, batch_key=batch_key, n_comps=N_PCA_COMPONENTS_FOR_CORR)
    adata.obsm['X_pca_batch'] = X_pca_batch

    # Copy per-batch PCA to scvi adata
    adata_scvi.obsm['X_pca_batch'] = X_pca_batch.copy()

    # CRITICAL FIX: Ensure all data is float32 to prevent MPS NaN errors
    # MPS (Metal Performance Shaders) often produces NaNs with float64 or mixed precision
    if hasattr(adata_scvi.X, 'astype'):
        adata_scvi.X = adata_scvi.X.astype(np.float32)
    
    # Check for NaNs/Infs in input data
    if sp.issparse(adata_scvi.X):
        if np.isnan(adata_scvi.X.data).any() or np.isinf(adata_scvi.X.data).any():
             logger.warning("Input data contains NaNs or Infs! This will cause training failure. Fixing...")
             adata_scvi.X.data = np.nan_to_num(adata_scvi.X.data)
    else:
        if np.isnan(adata_scvi.X).any() or np.isinf(adata_scvi.X).any():
             logger.warning("Input data contains NaNs or Infs! This will cause training failure. Fixing...")
             adata_scvi.X = np.nan_to_num(adata_scvi.X)

    # =========================================================================
    # Register per-batch PCA as continuous covariates for DCT training plan.
    # The original scIB-E adds a custom PCA_KEY to REGISTRY_KEYS and uses
    # ObsmField to register it. Since we cannot modify the installed scvi-tools
    # library, we register the per-batch PCA components as continuous covariates
    # and access them via CONT_COVS_KEY in the training plan.
    # =========================================================================
    n_pcs = min(N_PCA_COMPONENTS_FOR_CORR, adata_scvi.obsm['X_pca_batch'].shape[1])

    # Check PCA for NaNs
    X_pca = adata_scvi.obsm['X_pca_batch'].astype(np.float32)
    if np.isnan(X_pca).any() or np.isinf(X_pca).any():
        logger.warning("Per-batch PCA contains NaNs or Infs. Filling with zeros.")
        X_pca = np.nan_to_num(X_pca)

    pca_column_names = []
    for i in range(n_pcs):
        col_name = f'_pca_cov_{i}'
        adata_scvi.obs[col_name] = X_pca[:, i]
        pca_column_names.append(col_name)

    logger.info(f"Registered {n_pcs} per-batch PCA components as continuous covariates for DCT loss")

    # Setup scVI anndata with raw counts and PCA as continuous covariates
    scvi.model.SCVI.setup_anndata(
        adata_scvi,
        batch_key=batch_key,
        labels_key=labels_key,
        continuous_covariate_keys=pca_column_names,
    )
    
    # Create model
    model = scvi.model.SCVI(
        adata_scvi,
        n_latent=n_latent,
        n_hidden=n_hidden,
        n_layers=n_layers,
    )
    
    logger.info(f"Training scVI-DCT for {max_epochs} epochs on {device}...")
    
    # Batch correction weights for DCT training plan
    batch_correction_weight = {
        "batch_celltype_constraint_weight": dct_weight,
        "corr_mse_weight": corr_mse_weight
    }
    
    # ===========================================================================
    # Determine if we can use full DCT Training Plan
    # DCT requires: labels_key (for triplet loss) and PCA data (for corr_mse)
    # ===========================================================================
    use_dct_loss = True
    
    # Check if labels are available (required for triplet loss)
    if labels_key == '_dummy_labels':
        logger.warning(
            "No cell type labels available. DCT triplet loss will be less effective. "
            "Falling back to standard scVI training."
        )
        use_dct_loss = False
    
    # Check if per-batch PCA is available
    if 'X_pca_batch' not in adata_scvi.obsm:
        logger.warning("Per-batch PCA not available. Falling back to standard scVI training.")
        use_dct_loss = False
    
    # Determine accelerator for training
    if device == 'cuda':
        accelerator = "gpu"
        devices = 1
    elif device == 'mps':
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = "auto"
    
    if use_dct_loss:
        # Use the custom DCT Training Plan
        logger.info("Using DomainClassTripletLoss training plan for enhanced batch correction")

        try:
            plan_ok = _train_scvi_model(
                model,
                max_epochs=max_epochs,
                batch_size=batch_size,
                plan_class=DomainClassTripletLoss_TrainingPlan,
                plan_kwargs={
                    'lr': lr,
                    'batch_correction_weight': batch_correction_weight,
                },
                accelerator=accelerator,
                devices=devices,
                # gradient_clip_val=0.5, # Removed: Not supported with manual optimization
            )
            if plan_ok:
                logger.info("DCT training completed successfully with DomainClassTripletLoss")
            else:
                logger.warning(
                    "scvi-tools version does not support custom training plans. "
                    "Falling back to standard scVI."
                )
                use_dct_loss = False
                _train_scvi_model(
                    model,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    plan_kwargs={'lr': lr},
                    accelerator=accelerator,
                    devices=devices,
                    gradient_clip_val=0.5, # Keep for standard scVI (auto optimization)
                )
            
        except Exception as e:
            if accelerator != "cpu":
                logger.warning(f"DCT training failed ({e}). Retrying on CPU for stability.")
                model = scvi.model.SCVI(
                    adata_scvi,
                    n_latent=n_latent,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                )
                try:
                    plan_ok = _train_scvi_model(
                        model,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        plan_class=DomainClassTripletLoss_TrainingPlan,
                        plan_kwargs={
                            'lr': lr,
                            'batch_correction_weight': batch_correction_weight,
                        },
                        accelerator="cpu",
                        devices="auto",
                    )
                    if not plan_ok:
                        logger.warning(
                            "scvi-tools version does not support custom training plans. "
                            "Falling back to standard scVI on CPU."
                        )
                        use_dct_loss = False
                        _train_scvi_model(
                            model,
                            max_epochs=max_epochs,
                            batch_size=batch_size,
                            plan_kwargs={'lr': lr},
                            accelerator="cpu",
                            devices="auto",
                        )
                except Exception as cpu_e:
                    logger.warning(f"DCT training failed on CPU ({cpu_e}). Falling back to standard scVI.")
                    use_dct_loss = False
                    _train_scvi_model(
                        model,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        plan_kwargs={'lr': lr},
                        accelerator="cpu",
                        devices="auto",
                    )
            else:
                logger.warning(f"DCT training plan failed ({e}). Falling back to standard scVI.")
                use_dct_loss = False
                _train_scvi_model(
                    model,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    plan_kwargs={'lr': lr},
                    accelerator=accelerator,
                    devices=devices,
                    gradient_clip_val=0.5,
                )
    else:
        # Standard scVI training (still provides batch correction via batch_key)
        try:
            _train_scvi_model(
                model,
                max_epochs=max_epochs,
                batch_size=batch_size,
                plan_kwargs={'lr': lr},
                accelerator=accelerator,
                devices=devices,
                gradient_clip_val=0.5,
            )
        except Exception as e:
            if accelerator != "cpu":
                error_msg = str(e)
                if "invalid values" in error_msg or "nan" in error_msg.lower():
                    logger.warning(f"scVI training unstable on {accelerator.upper()} (NaNs detected). This is a known issue with some PyTorch/MPS versions. Automatically retrying on CPU...")
                else:
                    logger.warning(f"Standard scVI training failed on {accelerator.upper()} ({error_msg}). Retrying on CPU.")
                
                model = scvi.model.SCVI(
                    adata_scvi,
                    n_latent=n_latent,
                    n_hidden=n_hidden,
                    n_layers=n_layers,
                )
                _train_scvi_model(
                    model,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    plan_kwargs={'lr': lr},
                    accelerator="cpu",
                    devices="auto",
                    gradient_clip_val=0.5,
                )
            else:
                raise
    
    logger.info("scVI-DCT training complete. Getting corrected expression...")
    
    # Get batch-corrected normalized expression
    # Select reference batch (first category)
    ref_batch = adata.obs[batch_key].unique()[0]
    logger.info(f"Using reference batch '{ref_batch}' for output correction")
    
    corrected_expression = model.get_normalized_expression(
        adata=adata_scvi,
        library_size=1e4,
        return_numpy=True,
        transform_batch=str(ref_batch)
    )
    
    corrected_expression = np.asarray(corrected_expression, dtype=np.float64)

    if not np.isfinite(corrected_expression).all():
        non_finite_count = int((~np.isfinite(corrected_expression)).sum())
        logger.warning(
            "sysVI corrected expression contains %d non-finite values; "
            "replacing them with 0 before log1p.",
            non_finite_count,
        )
        corrected_expression = np.nan_to_num(
            corrected_expression, nan=0.0, posinf=0.0, neginf=0.0
        )

    min_expr = float(np.min(corrected_expression))
    if min_expr < 0.0:
        logger.warning(
            "sysVI corrected expression contains negative values (min=%.4f); "
            "clipping to 0 before log1p.",
            min_expr,
        )
        corrected_expression = np.clip(corrected_expression, a_min=0.0, a_max=None)

    # Apply log1p to match the preprocessing pipeline
    corrected_expression = np.log1p(corrected_expression)
    
    # Replace X in the original adata with corrected expression
    adata.X = corrected_expression
    
    # Store batch correction metadata
    adata.uns['batch_correction'] = {
        'method': 'scvi_dct',
        'batch_key': batch_key,
        'labels_key': labels_key,
        'n_latent': n_latent,
        'max_epochs': max_epochs,
        'n_batches': n_batches,
        'dct_weight': dct_weight,
        'corr_mse_weight': corr_mse_weight,
        'dct_loss_used': use_dct_loss,  # Track if DCT training plan was actually used
    }
    
    # Get latent embeddings
    embeddings = model.get_latent_representation()
    
    # Clean up dummy labels if created
    if '_dummy_labels' in adata.obs.columns:
        del adata.obs['_dummy_labels']
    
    logger.info(
        f"Batch correction complete. "
        f"Corrected expression shape: {corrected_expression.shape}"
    )
    
    if params.get('return_model', False):
        return adata, embeddings, model
    return adata, embeddings


def get_batch_correction_default_params() -> Dict[str, Any]:
    """Get default parameters for batch correction."""
    return {
        'n_latent': 30,
        'n_hidden': 128,
        'n_layers': 2,
        'max_epochs': 100,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'dct_weight': 1.0,
        'corr_mse_weight': 0.1,
    }


# =============================================================================
# sysVI Batch Correction (Hrovatin et al. 2023)
# =============================================================================

def apply_sysvi_batch_correction(
    adata,
    batch_key: str = 'batch',
    labels_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    device: str = 'auto'
) -> Tuple[Any, Optional[np.ndarray]]:
    """Apply sysVI batch correction (Hrovatin et al. 2023).

    sysVI uses a conditional VAE with VampPrior and cycle-consistency loss,
    designed for datasets with substantial batch effects (cross-species,
    organoid-tissue, different sequencing technologies).

    Unlike standard scVI, sysVI works on normalized+log-transformed data.

    Reference:
        Hrovatin et al. "Integrating single-cell RNA-seq datasets with
        substantial batch effects" (2023).

    Args:
        adata: AnnData object (normalized, log-transformed).
        batch_key: Column in obs with batch information (used as system key).
        labels_key: Not used by sysVI (accepted for API compatibility).
        params: Dict with training parameters:
            - prior: Prior type ('vamp' or 'standard_normal', default: 'vamp')
            - n_prior_components: Number of VampPrior pseudoinputs (default: 5)
            - n_latent: Latent dimension (default: 15)
            - n_hidden: Hidden layer size (default: 256)
            - embed_categorical_covariates: Embed covariates (default: True)
            - max_epochs: Training epochs (default: 200)
            - batch_size: Training batch size (default: 128)
            - z_distance_cycle_weight: Cycle-consistency loss weight (default: 5.0)
            - kl_weight: KL divergence weight (default: 1.0)
            - categorical_covariate_keys: Optional list of additional covariate columns
        device: Compute device ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Tuple of (adata, embeddings)
    """
    if not SCVI_AVAILABLE:
        raise ImportError(
            "scvi-tools is required for sysVI batch correction. "
            "Install with: pip install scvi-tools>=1.3"
        )

    try:
        from scvi.external import SysVI
    except ImportError:
        raise ImportError(
            "SysVI not found in scvi.external. "
            "Please upgrade scvi-tools: pip install scvi-tools>=1.3"
        )

    params = params or {}
    adata = adata.copy()

    # Explicitly seed python/numpy/torch/scvi for reproducible sysVI runs.
    seed = params.get('seed', params.get('random_seed', params.get('random_state', None)))
    if seed is not None:
        try:
            seed = int(seed)
            random.seed(seed)
            np.random.seed(seed)
            if torch is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            try:
                scvi.settings.seed = seed
            except Exception:
                pass
            logger.info(f"Using sysVI random seed: {seed}")
        except Exception as seed_error:
            logger.warning(f"Could not apply sysVI seed '{seed}': {seed_error}")
            seed = None

    # Validate batch key
    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"Batch column '{batch_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    n_batches = adata.obs[batch_key].nunique()
    if n_batches < 2:
        logger.warning(
            f"Only {n_batches} batch(es) found. Batch correction may have no effect."
        )

    logger.info(f"Starting sysVI batch correction with {n_batches} batches")

    # sysVI works on normalized+log-transformed data (NOT raw counts)
    # Ensure data is float32 for stability
    if hasattr(adata.X, 'astype'):
        adata.X = adata.X.astype(np.float32)

    # Handle sparse matrices - sysVI may need dense
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()

    # Optional additional categorical covariates
    categorical_covariates = params.get('categorical_covariate_keys', None)

    # Setup anndata for sysVI
    setup_kwargs = {
        'batch_key': batch_key,
    }
    if categorical_covariates:
        setup_kwargs['categorical_covariate_keys'] = categorical_covariates

    SysVI.setup_anndata(adata, **setup_kwargs)

    # Model parameters
    model_kwargs = {}
    n_latent = params.get('n_latent', 15)
    n_hidden = params.get('n_hidden', 256)
    if n_latent != 15:
        model_kwargs['n_latent'] = n_latent
    if n_hidden != 256:
        model_kwargs['n_hidden'] = n_hidden

    model = SysVI(
        adata,
        prior=params.get('prior', 'vamp'),
        n_prior_components=params.get('n_prior_components', 5),
        embed_categorical_covariates=params.get('embed_categorical_covariates', True),
        **model_kwargs,
    )

    # Training parameters
    max_epochs = params.get('max_epochs', 200)
    batch_size = params.get('batch_size', 128)
    z_distance_cycle_weight = params.get('z_distance_cycle_weight', 5.0)
    kl_weight = params.get('kl_weight', 1.0)

    # Determine accelerator
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    if device == 'cuda':
        accelerator = "gpu"
        devices_arg = 1
    elif device == 'mps':
        accelerator = "mps"
        devices_arg = 1
    else:
        accelerator = "cpu"
        devices_arg = "auto"

    logger.info(
        f"Training sysVI for {max_epochs} epochs on {device} "
        f"(cycle_weight={z_distance_cycle_weight}, kl_weight={kl_weight})..."
    )

    try:
        model.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            accelerator=accelerator,
            devices=devices_arg,
            plan_kwargs={
                'z_distance_cycle_weight': z_distance_cycle_weight,
                'kl_weight': kl_weight,
            },
        )
    except Exception as e:
        if accelerator != "cpu":
            logger.warning(
                f"sysVI training failed on {accelerator.upper()} ({e}). "
                "Retrying on CPU."
            )
            model = SysVI(
                adata,
                prior=params.get('prior', 'vamp'),
                n_prior_components=params.get('n_prior_components', 5),
                embed_categorical_covariates=params.get('embed_categorical_covariates', True),
                **model_kwargs,
            )
            model.train(
                max_epochs=max_epochs,
                batch_size=batch_size,
                accelerator="cpu",
                devices="auto",
                plan_kwargs={
                    'z_distance_cycle_weight': z_distance_cycle_weight,
                    'kl_weight': kl_weight,
                },
            )
        else:
            raise

    logger.info("sysVI training complete. Getting embeddings and corrected expression...")

    # Get latent embeddings
    embeddings = model.get_latent_representation(adata)

    # Get batch-corrected normalized expression
    ref_batch = adata.obs[batch_key].unique()[0]
    logger.info(f"Using reference batch '{ref_batch}' for output correction")

    try:
        corrected_expression = model.get_normalized_expression(
            adata=adata,
            library_size=1e4,
            return_numpy=True,
            transform_batch=str(ref_batch),
        )
    except TypeError as e:
        # sysVI on some scvi-tools versions can return a px object where "scale" is None.
        # Fall back to latent-scale normalized expression to keep correction usable.
        if "NoneType" not in str(e):
            raise
        logger.warning(
            "sysVI get_normalized_expression(library_size=1e4) failed (%s). "
            "Retrying with library_size='latent'.",
            e,
        )
        corrected_expression = model.get_normalized_expression(
            adata=adata,
            library_size="latent",
            return_numpy=True,
            transform_batch=str(ref_batch),
        )

    corrected_expression = np.asarray(corrected_expression, dtype=np.float64)

    if not np.isfinite(corrected_expression).all():
        non_finite_count = int((~np.isfinite(corrected_expression)).sum())
        logger.warning(
            "sysVI corrected expression contains %d non-finite values; "
            "replacing them with 0 before log1p.",
            non_finite_count,
        )
        corrected_expression = np.nan_to_num(
            corrected_expression, nan=0.0, posinf=0.0, neginf=0.0
        )

    min_expr = float(np.min(corrected_expression))
    if min_expr < 0.0:
        logger.warning(
            "sysVI corrected expression contains negative values (min=%.4f); "
            "clipping to 0 before log1p.",
            min_expr,
        )
        corrected_expression = np.clip(corrected_expression, a_min=0.0, a_max=None)

    # Apply log1p to match the preprocessing pipeline
    corrected_expression = np.log1p(corrected_expression)

    # Replace X in the adata with corrected expression
    adata.X = corrected_expression

    # Store batch correction metadata
    adata.uns['batch_correction'] = {
        'method': 'sysvi',
        'batch_key': batch_key,
        'n_latent': n_latent,
        'max_epochs': max_epochs,
        'n_batches': n_batches,
        'prior': params.get('prior', 'vamp'),
        'n_prior_components': params.get('n_prior_components', 5),
        'z_distance_cycle_weight': z_distance_cycle_weight,
        'kl_weight': kl_weight,
        'embed_categorical_covariates': params.get('embed_categorical_covariates', True),
        'seed': seed,
    }

    logger.info(
        f"sysVI batch correction complete. "
        f"Corrected expression shape: {corrected_expression.shape}"
    )

    if params.get('return_model', False):
        return adata, embeddings, model
    return adata, embeddings


def get_sysvi_default_params() -> Dict[str, Any]:
    """Get default parameters for sysVI batch correction."""
    return {
        'prior': 'vamp',
        'n_prior_components': 5,
        'n_latent': 15,
        'n_hidden': 256,
        'embed_categorical_covariates': True,
        'max_epochs': 200,
        'batch_size': 128,
        'z_distance_cycle_weight': 5.0,
        'kl_weight': 1.0,
    }


# =============================================================================
# Main Dispatch
# =============================================================================

def apply_batch_correction(
    adata,
    method: str = 'scvi',
    batch_key: str = 'batch',
    labels_key: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    device: str = 'auto'
) -> Tuple[Any, Optional[np.ndarray]]:
    """
    Main entry point for applying batch correction.

    Args:
        adata: AnnData object
        method: 'scvi', 'scvi_dct', 'sysvi'
        batch_key: Batch column name
        labels_key: Labels column name (for supervised methods like scVI-DCT)
        params: Algorithm-specific parameters
        device: 'cpu', 'cuda', 'mps', 'auto'

    Returns:
        (adata, embeddings)
    """
    params = params or {}
    # Handle both 'method' arg and 'method' in params
    method = params.get('method', method).lower()

    logger.info(f"Requested batch correction method: {method}")

    if method == 'scvi' or method == 'scvi_dct':
        return apply_scvi_batch_correction(
            adata, batch_key=batch_key, labels_key=labels_key, params=params, device=device
        )
    elif method == 'sysvi':
        return apply_sysvi_batch_correction(
            adata, batch_key=batch_key, labels_key=labels_key, params=params, device=device
        )
    else:
        raise ValueError(f"Unknown batch correction method: {method}")
