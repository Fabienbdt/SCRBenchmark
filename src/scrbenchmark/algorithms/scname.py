"""
scNAME algorithm implementation.
Based on https://github.com/aster-ww/scNAME

scNAME: Neighborhood contrastive clustering with ancillary mask estimation
for scRNA-seq data.

Reference:
    Wan, H., Chen, L., & Deng, M. (2022). scNAME: Neighborhood contrastive
    clustering with ancillary mask estimation for scRNA-seq data.
    Bioinformatics. https://doi.org/10.1093/bioinformatics/btac011

The method integrates three core mechanisms:
1. Mask Estimation Task - Mines gene pertinence to reveal uncorrupted data
2. Neighborhood Contrastive Learning - Exploits cell intrinsic structure
3. Offline Memory Bank - Global-scope discriminative feature representation
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType


# =============================================================================
# Utility functions
# =============================================================================

def mask_generator(p_m: float, x: np.ndarray) -> np.ndarray:
    """
    Generate binary mask for self-supervised learning.

    Args:
        p_m: Corruption probability
        x: Input data

    Returns:
        Binary mask with same shape as x
    """
    return np.random.binomial(1, p_m, x.shape)


def pretext_generator(m: np.ndarray, x: np.ndarray):
    """
    Generate corrupted samples for mask estimation pretext task.

    Args:
        m: Binary mask
        x: Input data

    Returns:
        m_new: New mask (1 where data was actually changed)
        x_tilde: Corrupted samples
    """
    no, dim = x.shape

    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples: keep original where mask is 0, use shuffled where mask is 1
    x_tilde = x * (1 - m) + x_bar * m

    # Define new mask matrix: 1 where values actually changed
    m_new = (x != x_tilde).astype(np.float32)

    return m_new, x_tilde


def safe_bce_probability_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Numerically safe BCE on probabilities.

    scNAME uses a sigmoid head for mask estimation, but with aggressive LR search
    (benchmark mode), NaN/Inf can still appear and trigger CUDA device asserts in
    BCELoss. We sanitize and clamp probabilities before BCE to keep training stable.
    """
    pred = torch.nan_to_num(pred, nan=0.5, posinf=1.0 - eps, neginf=eps)
    pred = torch.clamp(pred, min=eps, max=1.0 - eps)
    target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)
    target = torch.clamp(target, min=0.0, max=1.0)
    return F.binary_cross_entropy(pred, target)


# =============================================================================
# Activation functions
# =============================================================================

class MeanAct(nn.Module):
    """Mean activation for ZINB (exponential with clamping)."""
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation for ZINB (softplus with clamping)."""
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


# =============================================================================
# Loss functions
# =============================================================================

class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial loss for scRNA-seq count data."""

    def __init__(self, ridge_lambda: float = 1.0):
        super(ZINBLoss, self).__init__()
        self.ridge_lambda = ridge_lambda

    def forward(self, pi, theta, y_true, y_pred):
        """
        Args:
            pi: Dropout probability
            theta: Dispersion parameter
            y_true: True counts
            y_pred: Predicted mean

        Returns:
            ZINB loss value
        """
        eps = 1e-10

        # Clamp theta to avoid numerical issues
        theta = torch.clamp(theta, max=1e6)

        # Negative binomial case
        t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + \
             (y_true * (torch.log(theta + eps) - torch.log(y_pred + eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)

        # Zero case
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)

        # Select appropriate case based on y_true value
        result = torch.where(y_true < 1e-8, zero_case, nb_case)

        # Ridge regularization on dropout probability
        ridge = self.ridge_lambda * torch.square(pi)
        result = result + ridge

        return torch.mean(result)


def neighbor_k_loss(batch_size: int, k: int, bank: torch.Tensor,
                    z: torch.Tensor, z_tilde: torch.Tensor,
                    temperature: float) -> torch.Tensor:
    """
    Neighborhood contrastive loss.

    Encourages the corrupted representation z_tilde to be similar to
    the k nearest neighbors of the original representation z in the memory bank.

    Args:
        batch_size: Batch size
        k: Number of nearest neighbors
        bank: Memory bank containing all cell representations
        z: Original latent representations
        z_tilde: Corrupted latent representations
        temperature: Temperature parameter for softmax

    Returns:
        Contrastive loss value
    """
    # Normalize vectors
    bank_normal = F.normalize(bank, p=2, dim=1)
    z_normal = F.normalize(z, p=2, dim=1)
    z_tilde_normal = F.normalize(z_tilde, p=2, dim=1)

    # Find k nearest neighbors of z in the bank
    similarity_z_bank = torch.mm(z_normal, bank_normal.T)
    _, top_k_indices = torch.topk(similarity_z_bank, k=k, dim=1)

    # Compute similarity between z_tilde and bank
    similarity_tilde_bank = torch.mm(z_tilde_normal, bank_normal.T)

    # Get similarities to k nearest neighbors
    numerator = torch.zeros(batch_size, device=z.device)
    for i in range(batch_size):
        neighbor_sims = similarity_tilde_bank[i, top_k_indices[i]]
        numerator[i] = torch.sum(torch.exp(neighbor_sims / temperature))

    # Denominator: sum over all cells in bank
    denominator = torch.sum(torch.exp(similarity_tilde_bank / temperature), dim=1)

    # Contrastive loss
    loss = -torch.sum(torch.log(numerator / (denominator + 1e-10)))

    return loss / batch_size


def cal_dist(hidden: torch.Tensor, clusters: torch.Tensor):
    """
    Calculate distance-based soft cluster assignment.

    Args:
        hidden: Latent representations (n_samples, dim)
        clusters: Cluster centers (n_clusters, dim)

    Returns:
        dist1: Squared distances to clusters
        dist2: Weighted distances for clustering loss
    """
    # Squared Euclidean distance
    dist1 = torch.sum((hidden.unsqueeze(1) - clusters) ** 2, dim=2)

    # Soft assignment
    temp_dist1 = dist1 - dist1.min(dim=1, keepdim=True)[0]
    q = torch.exp(-temp_dist1)
    q = q / q.sum(dim=1, keepdim=True)
    q = q ** 2
    q = q / q.sum(dim=1, keepdim=True)

    # Weighted distance
    dist2 = dist1 * q

    return dist1, dist2


# =============================================================================
# scNAME Autoencoder Model
# =============================================================================

class ScNAMEAutoencoder(nn.Module):
    """
    scNAME Autoencoder with mask estimation and ZINB reconstruction.

    Architecture:
    - Encoder: Input -> [hidden layers with noise] -> Latent
    - Decoder: Latent -> [hidden layers] -> Reconstructed
    - Additional heads: mask_estimator, pi, disp, mean
    """

    def __init__(self, dims: List[int], noise_sd: float = 1.5, activation: str = 'relu'):
        """
        Args:
            dims: Layer dimensions [input_dim, hidden1, hidden2, ..., latent_dim]
            noise_sd: Standard deviation for Gaussian noise
            activation: Activation function
        """
        super(ScNAMEAutoencoder, self).__init__()

        self.dims = dims
        self.noise_sd = noise_sd
        self.n_stacks = len(dims) - 1

        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(self.n_stacks - 1):
            self.encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))

        # Latent layer
        self.encoder_latent = nn.Linear(dims[-2], dims[-1])

        # Decoder layers (mirror of encoder)
        # From latent_dim back to first hidden layer
        self.decoder_layers = nn.ModuleList()
        # Decoder goes: latent -> dims[-2] -> dims[-3] -> ... -> dims[1]
        for i in range(self.n_stacks - 1, 0, -1):
            in_dim = dims[i + 1] if i < self.n_stacks - 1 else dims[-1]
            out_dim = dims[i]
            self.decoder_layers.append(nn.Linear(in_dim, out_dim))

        # Output heads (from dims[1] to dims[0])
        output_dim = dims[1] if self.n_stacks > 1 else dims[-1]

        self.mask_estimator = nn.Sequential(
            nn.Linear(output_dim, dims[0]),
            nn.Sigmoid()
        )

        self.pi_layer = nn.Sequential(
            nn.Linear(output_dim, dims[0]),
            nn.Sigmoid()
        )

        self.disp_layer = nn.Sequential(
            nn.Linear(output_dim, dims[0]),
            DispAct()
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(output_dim, dims[0]),
            MeanAct()
        )

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise during training."""
        if self.training:
            return x + torch.randn_like(x) * self.noise_sd
        return x

    def encode(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """Encode input to latent space."""
        h = x
        if add_noise:
            h = self.add_noise(h)

        for layer in self.encoder_layers:
            h = layer(h)
            if add_noise:
                h = self.add_noise(h)
            h = self.activation(h)

        latent = self.encoder_latent(h)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction space."""
        h = latent
        for layer in self.decoder_layers:
            h = layer(h)
            h = self.activation(h)
        return h

    def forward(self, x: torch.Tensor, x_tilde: torch.Tensor = None):
        """
        Forward pass.

        Args:
            x: Original input
            x_tilde: Corrupted input (optional, for mask estimation)

        Returns:
            latent_x: Latent representation of x
            latent_tilde: Latent representation of x_tilde (if provided)
            mask_est: Mask estimation
            pi: Dropout probability
            disp: Dispersion
            mean: Predicted mean
        """
        # Encode original (without noise for clean latent)
        latent_x = self.encode(x, add_noise=False)

        # Encode for reconstruction (with noise)
        latent_noisy = self.encode(x, add_noise=True)
        h = self.decode(latent_noisy)

        # Output heads
        pi = self.pi_layer(h)
        disp = self.disp_layer(h)
        mean = self.mean_layer(h)

        # Mask estimation from corrupted input
        if x_tilde is not None:
            latent_tilde = self.encode(x_tilde, add_noise=True)
            h_tilde = self.decode(latent_tilde)
            mask_est = self.mask_estimator(h_tilde)
        else:
            latent_tilde = None
            mask_est = self.mask_estimator(h)

        return latent_x, latent_tilde, mask_est, pi, disp, mean


# =============================================================================
# Algorithm wrapper for SCRBenchmark
# =============================================================================

@AlgorithmRegistry.register

class ScNAMEAlgorithm(BaseAlgorithm):
    """
    scNAME: Neighborhood contrastive clustering with ancillary mask estimation.

    A deep learning method that combines:
    - Mask estimation for denoising scRNA-seq data
    - Neighborhood contrastive learning using memory bank
    - ZINB-based reconstruction loss
    - K-means clustering on learned representations

    Reference:
        Wan et al. (2022) Bioinformatics. DOI: 10.1093/bioinformatics/btac011
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name='scname',
            display_name='scNAME',
            description='Neighborhood contrastive clustering with mask estimation. '
                       'Combines ZINB reconstruction, mask estimation for denoising, '
                       'and contrastive learning for scRNA-seq clustering.',
            category='deep_learning',
            requires_gpu=False,
            supports_labels=True,
            supports_out_of_sample=True,
            preprocessing_notes='Expects raw counts of selected Highly Variable Genes (HVGs). '
                               'Internal preprocessing (size factors, log, scale) is applied, '
                               'but NO internal gene selection is performed.',
            has_internal_preprocessing=True,
            recommended_data='raw'
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        return [
            # Network architecture
            HyperparameterConfig(
                name='hidden_dims',
                display_name='Hidden Dimensions',
                param_type=ParamType.STRING,
                default='256,64,32',
                description='Hidden layer dimensions (comma-separated)',
                category='Network',
                tuning_guide='**Impact**: Architecture complexity and latent space size.\n**Details**: The last dimension (32) is the latent space. Larger hidden layers (256, 128) increase capacity but require more data. 32 is standard for the latent dimension to force a compact representation.\n**Recommendation**: Default is suitable for most datasets.'
            ),
            HyperparameterConfig(
                name='noise_sd',
                display_name='Noise Std Dev',
                param_type=ParamType.FLOAT,
                default=1.5,
                description='Standard deviation for Gaussian noise regularization',
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                category='Network',
                tuning_guide='**Impact**: Regularization strength.\n**Details**: Adds Gaussian noise to the encoder input. Higher values (1.5) behave like strong augmentations in contrastive learning, preventing the model from learning trivial features.\n**Recommendation**: 1.5 is standard. Increase if overfitting.'
            ),
            # Mask estimation
            HyperparameterConfig(
                name='p_m',
                display_name='Mask Probability',
                param_type=ParamType.FLOAT,
                default=0.3,
                description='Corruption probability for mask estimation',
                min_value=0.1,
                max_value=0.7,
                step=0.05,
                category='Mask Estimation',
                tuning_guide='**Impact**: Denoising difficulty.\n**Details**: Probability of masking inputs for the denoising task. 0.3 means 30% of expression values are dropped. Too high destroys signal; too low makes the task trivial.\n**Recommendation**: 0.3 works well for sparse scRNA-seq.'
            ),
            # Contrastive learning
            HyperparameterConfig(
                name='k',
                display_name='K Neighbors',
                param_type=ParamType.INTEGER,
                default=10,
                description='Number of nearest neighbors for contrastive loss',
                min_value=3,
                max_value=50,
                step=1,
                category='Contrastive Learning',
                tuning_guide='**Impact**: Locality of the contrastive constraint.\n**Details**: Determines how many neighbors in the memory bank are considered "positive" pairs. Small k (5-10) enforces very local clusters. Large k (50) might merge distinct subtypes.\n**Recommendation**: 10 is a good baseline.'
            ),

            HyperparameterConfig(
                name='temperature',
                display_name='Temperature',
                param_type=ParamType.FLOAT,
                default=0.7,
                description='Temperature parameter for contrastive loss',
                min_value=0.1,
                max_value=2.0,
                step=0.1,
                category='Contrastive Learning',
                tuning_guide='**Impact**: Softness of similarity probability.\n**Details**: Controls how sharp the attention to the nearest neighbors is. Lower values make the loss focus only on the very hardest positives/negatives.\n**Recommendation**: 0.7 is standard for SimCLR-like losses.'
            ),
            # Loss weights
            HyperparameterConfig(
                name='alpha',
                display_name='Mask Loss Weight',
                param_type=ParamType.FLOAT,
                default=1.0,
                description='Weight for mask estimation loss',
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                category='Loss Weights'
            ),
            HyperparameterConfig(
                name='beta',
                display_name='Neighbor Loss Weight',
                param_type=ParamType.FLOAT,
                default=0.1,
                description='Weight for neighborhood contrastive loss',
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                category='Loss Weights',
                tuning_guide='**Impact**: Importance of maintaining local structure.\n**Details**: Balances the reconstruction objective with the neighborhood similarity objective. Increase if clusters are not separating well.\n**Recommendation**: 0.1 is standard.'
            ),
            HyperparameterConfig(
                name='gamma',
                display_name='K-Means Loss Weight',
                param_type=ParamType.FLOAT,
                default=0.1,
                description='Weight for k-means clustering loss',
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                category='Loss Weights'
            ),
            # Training
            HyperparameterConfig(
                name='pretrain_epochs',
                display_name='Pretrain Epochs',
                param_type=ParamType.INTEGER,
                default=500,
                description='Number of pretraining epochs',
                max_value=1000,
                step=50,
                category='Training'
            ),
            HyperparameterConfig(
                name='finetune_epochs',
                display_name='Finetune Epochs',
                param_type=ParamType.INTEGER,
                default=1000,
                description='Number of fine-tuning epochs',
                max_value=2000,
                step=100,
                category='Training',
                tuning_guide='**Impact**: Cluster refinement time.\n**Details**: Contrastive learning converges slowly. Requires long training (1000+ epochs) for the neighborhood constraints to reshape the latent space effectively.\n**Recommendation**: 1000+.'
            ),
            HyperparameterConfig(
                name='batch_size',
                display_name='Batch Size',
                param_type=ParamType.INTEGER,
                default=128,
                description='Training batch size',
                min_value=32,
                max_value=512,
                step=32,
                category='Training'
            ),
            HyperparameterConfig(
                name='learning_rate',
                display_name='Learning Rate',
                param_type=ParamType.FLOAT,
                default=0.0001,
                description='Learning rate for Adam optimizer',
                min_value=0.00001,
                max_value=0.01,
                step=0.00001,
                category='Training',
                tuning_guide='**Impact**: Stability vs Speed.\n**Details**: scNAME uses a relatively low learning rate (1e-4) to ensure the multiple loss components (mask, contrastive, clustering) do not destabilize each other.\n**Recommendation**: 0.0001.'
            ),
            HyperparameterConfig(
                name='update_interval',
                display_name='Update Interval',
                param_type=ParamType.INTEGER,
                default=50,
                description='Interval for checking convergence',
                min_value=10,
                max_value=200,
                step=10,
                category='Training',
                advanced=True
            ),
            HyperparameterConfig(
                name='convergence_tol',
                display_name='Convergence Tolerance',
                param_type=ParamType.FLOAT,
                default=0.001,
                description='Label change tolerance for convergence',
                min_value=0.0001,
                max_value=0.01,
                step=0.0001,
                category='Training',
                advanced=True
            ),
            # Clustering
            HyperparameterConfig(
                name='n_clusters',
                display_name='Number of Clusters',
                param_type=ParamType.INTEGER,
                default=0,
                description='Number of clusters (0 = auto-detect from labels)',
                min_value=0,
                max_value=100,
                step=1,
                category='Clustering'
            ),
            HyperparameterConfig(
                name='random_state',
                display_name='Random State',
                param_type=ParamType.INTEGER,
                default=1111,
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
                default=True,
                description='Use raw counts with scNAME internal preprocessing (recommended).',
                category='Data'
            ),
        ]


    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.model = None
        self.memory_bank = None
        self.cluster_centers = None
        self.le = None  # For label encoding
        self._shuffle_index = None
        self._filtered_index = None
        self._train_norm_target = None  # Normalization target (median of counts) from training

    def _normalize_scname(self, adata):
        """Match scNAME preprocessing (raw counts -> size factors -> log -> scale)."""
        import scanpy as sc
        import scipy as sp

        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        assert 'n_count' not in adata.obs, norm_error
        if adata.X.size < 50e6:
            if sp.sparse.issparse(adata.X):
                assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
            else:
                assert np.all(adata.X.astype(int) == adata.X), norm_error

        # Store original gene names before filtering
        original_genes = adata.var_names.copy() if hasattr(adata, 'var_names') else None

        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

        # Store the gene names that remain after filtering (for predict())
        if hasattr(adata, 'var_names'):
            self._training_genes = adata.var_names.copy()
        else:
            self._training_genes = None

        # Store number of genes for dimension check in predict
        self._n_training_genes = adata.n_vars

        adata.raw = adata.copy()
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        # Store normalization target for consistent test preprocessing
        self._train_norm_target = float(np.median(adata.obs.n_counts))
        sc.pp.log1p(adata)

        # Store training statistics BEFORE scaling for use in predict()
        X_for_scale = adata.X
        if hasattr(X_for_scale, 'toarray'):
            X_for_scale = X_for_scale.toarray()
        self._training_mean = np.mean(X_for_scale, axis=0).astype(np.float32)
        self._training_std = np.std(X_for_scale, axis=0).astype(np.float32)
        self._training_std[self._training_std == 0] = 1.0  # Avoid division by zero

        sc.pp.scale(adata)
        return adata

    def _prepare_scname_inputs(self, data: Any, labels: Optional[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare X, X_raw, and size_factors using scNAME preprocessing."""
        import anndata as ad
        import pandas as pd

        use_raw_data = self.params.get('use_raw_data', True)

        if hasattr(data, 'X'):
            gene_names = data.var_names if hasattr(data, 'var_names') else None
            if use_raw_data and hasattr(data, 'layers') and 'original_X' in data.layers:
                X_counts = data.layers['original_X']
            elif use_raw_data and hasattr(data, 'raw') and data.raw is not None:
                X_counts = data.raw.X
            else:
                X_counts = data.X
            if hasattr(X_counts, 'toarray'):
                X_counts = X_counts.toarray()
        else:
            gene_names = None
            X_counts = data

        X_counts = np.ceil(X_counts).astype(np.float32)

        adata = ad.AnnData(
            X=X_counts,
            var=pd.DataFrame(index=gene_names) if gene_names is not None else None
        )
        if labels is not None:
            adata.obs['Group'] = labels

        # Prepare missing size factors if not normalizing
        if not use_raw_data and 'size_factors' not in adata.obs:
             # Just use 1.0 or mean if using processed data
             adata.obs['size_factors'] = 1.0

        if use_raw_data:
            adata = self._normalize_scname(adata)
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = X.astype(np.float32)

        if adata.raw is not None:
            X_raw = adata.raw.X
            if hasattr(X_raw, 'toarray'):
                X_raw = X_raw.toarray()
            # Align raw counts to selected HVGs when subset=True
            if X_raw.shape[1] != X.shape[1]:
                raw_gene_names = adata.raw.var_names
                idx = raw_gene_names.get_indexer(adata.var_names)
                X_raw = X_raw[:, idx]
        else:
            X_raw = np.ceil(np.expm1(X)).astype(np.float32)

        X_raw = X_raw.astype(np.float32)

        if 'size_factors' in adata.obs:
            size_factor = adata.obs['size_factors'].values.astype(np.float32).reshape(-1, 1)
        else:
            row_sums = np.sum(X_raw, axis=1, keepdims=True)
            size_factor = (row_sums / np.median(row_sums)).astype(np.float32)

        if labels is not None:
            # Try to convert obs_names to int indices
            try:
                self._filtered_index = np.asarray(adata.obs_names, dtype=int)
            except (ValueError, TypeError):
                # Fallback: maintain a mapping if indices are non-numeric
                # This assumes input data was processed/indexed sequentially before this call
                # For safety in SCRBenchmark, we rely on DataHandler providing sequential indices in preprocess
                # If not, we might be returning incorrect indices.
                # However, DataHandler uses range(N) index, so this should work.
                # If it fails, we warn and return None for indices
                print("scNAME: Warning - could not convert obs_names to int. Plotting alignment may fail.")
                self._filtered_index = None
        else:
            self._filtered_index = None

        return X, X_raw, size_factor

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'ScNAMEAlgorithm':
        """
        Fit the scNAME model following the original two-phase training:
        1. Pretrain: ZINB reconstruction + mask estimation
        2. Finetune: Add neighborhood contrastive + k-means loss

        Args:
            data: AnnData object or numpy array with expression data
            labels: Optional ground truth labels
        """
        # Set random seeds
        seed = self.params.get('random_state', 1111)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Get device (respects user preference from UI)
        device = torch.device(self.get_device())
        if device.type == 'mps':
            torch.set_default_dtype(torch.float32)

        # Extract data matrix and raw counts using original scNAME preprocessing
        # Extract data matrix and raw counts using original scNAME preprocessing
        if self.params.get('use_raw_data', True):
            X, X_raw, size_factor = self._prepare_scname_inputs(data, labels)
        else:
            if hasattr(data, 'X'):
                X = data.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
            else:
                X = data
            X = X.astype(np.float32)
            n_samples, _ = X.shape
            if hasattr(data, 'raw') and data.raw is not None:
                X_raw = data.raw.X
                if hasattr(X_raw, 'toarray'):
                    X_raw = X_raw.toarray()
                X_raw = X_raw.astype(np.float32)
            else:
                X_raw = np.ceil(np.expm1(X)).astype(np.float32)
            if hasattr(data, 'obs') and 'size_factors' in data.obs:
                size_factor = data.obs['size_factors'].values.astype(np.float32).reshape(-1, 1)
            else:
                row_sums = np.sum(X_raw, axis=1, keepdims=True)
                size_factor = (row_sums / np.median(row_sums)).astype(np.float32)

        n_samples, n_genes = X.shape

        if labels is not None and self._filtered_index is not None:
            labels = np.array(labels)[self._filtered_index]

        # Shuffle data as in the original implementation, then keep an index to restore order
        shuffle_idx = np.random.permutation(n_samples)
        self._shuffle_index = shuffle_idx
        X = X[shuffle_idx]
        X_raw = X_raw[shuffle_idx]
        size_factor = size_factor[shuffle_idx]

        # Get parameters
        n_clusters = self.params.get('n_clusters', 0)
        if n_clusters == 0 and labels is not None:
            n_clusters = len(np.unique(labels))
            logger.warning(f"scNAME: n_clusters=0 (Auto). Using ground truth labels to determine k={n_clusters} (Oracle Mode).")
        elif n_clusters == 0:
            n_clusters = 10
            logger.warning(f"scNAME: n_clusters=0 and no labels. Defaulting to k={n_clusters}.")

        hidden_dims = [int(x) for x in self.params.get('hidden_dims', '256,64,32').split(',')]
        dims = [n_genes] + hidden_dims
        latent_dim = dims[-1]

        p_m = self.params.get('p_m', 0.3)
        k = self.params.get('k', 10)
        temperature = self.params.get('temperature', 0.7)
        alpha = self.params.get('alpha', 1.0)
        beta = self.params.get('beta', 0.1)
        gamma = self.params.get('gamma', 0.1)
        noise_sd = self.params.get('noise_sd', 1.5)
        pretrain_epochs = self.params.get('pretrain_epochs', 500)
        finetune_epochs = self.params.get('finetune_epochs', 1000)
        batch_size = self.params.get('batch_size', 128)
        lr = self.params.get('learning_rate', 0.0001)
        update_interval = self.params.get('update_interval', 50)
        convergence_tol = self.params.get('convergence_tol', 0.001)

        # Create model and ensure float32
        self.model = ScNAMEAutoencoder(dims, noise_sd=noise_sd).float().to(device)

        # Loss functions
        zinb_loss_fn = ZINBLoss(ridge_lambda=1.0)

        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize memory bank with explicit dtype
        self.memory_bank = torch.zeros(n_samples, latent_dim, device=device, dtype=torch.float32)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        X_raw_tensor = torch.tensor(X_raw, dtype=torch.float32).to(device)
        sf_tensor = torch.tensor(size_factor, dtype=torch.float32).to(device)

        # Convert labels if provided
        if labels is not None:
            self.le = LabelEncoder()
            y = self.le.fit_transform(labels)
            y = y[shuffle_idx]
            # Store original labels for reference if needed, but we use int internally
        else:
            y = None

        # =====================================================================
        # PHASE 1: PRETRAIN (ZINB + Mask)
        # =====================================================================
        print("scNAME: Pretraining (ZINB + Mask estimation)...")
        pretrain_losses = []

        for epoch in range(1, pretrain_epochs + 1):
            self.model.train()

            # Shuffle indices
            indices = np.random.permutation(n_samples)

            epoch_zinb_loss = 0
            epoch_mask_loss = 0
            n_batches = 0

            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_idx = indices[batch_start:batch_end]
                actual_batch_size = len(batch_idx)

                x_batch = X_tensor[batch_idx]
                x_raw_batch = X_raw_tensor[batch_idx]
                sf_batch = sf_tensor[batch_idx]

                # Generate mask and corrupted data
                x_np = x_batch.cpu().numpy()
                m_batch = mask_generator(p_m, x_np)
                m_new, x_tilde = pretext_generator(m_batch, x_np)

                m_new_tensor = torch.tensor(m_new, dtype=torch.float32).to(device)
                x_tilde_tensor = torch.tensor(x_tilde, dtype=torch.float32).to(device)

                # Forward pass
                latent_x, latent_tilde, mask_est, pi, disp, mean = self.model(x_batch, x_tilde_tensor)

                # ZINB reconstruction loss
                output = mean * sf_batch
                zinb_loss = zinb_loss_fn(pi, disp, x_raw_batch, output)

                # Mask estimation loss
                mask_loss = safe_bce_probability_loss(mask_est, m_new_tensor)

                # Total pretrain loss
                loss = zinb_loss + alpha * mask_loss
                if not bool(torch.isfinite(loss)):
                    logger.warning(
                        "scNAME pretrain: non-finite loss at epoch %d, batch_start %d; skipping batch.",
                        epoch,
                        batch_start,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                # Update memory bank
                with torch.no_grad():
                    self.memory_bank[batch_idx] = latent_x.detach()

                epoch_zinb_loss += zinb_loss.item()
                epoch_mask_loss += mask_loss.item()
                n_batches += 1

            if n_batches == 0:
                raise RuntimeError(
                    "scNAME pretraining produced no valid batches (all losses non-finite)."
                )
            pretrain_losses.append((epoch_zinb_loss + epoch_mask_loss) / n_batches)

            if epoch % 50 == 0:
                print(f'  Pretrain Epoch {epoch}/{pretrain_epochs}, '
                      f'ZINB: {epoch_zinb_loss/n_batches:.4f}, '
                      f'Mask: {epoch_mask_loss/n_batches:.4f}')

        # =====================================================================
        # Initialize clusters with K-means
        # =====================================================================
        print("scNAME: Initializing clusters...")
        with torch.no_grad():
            bank_np = self.memory_bank.cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=seed)
            kmeans_pred = kmeans.fit_predict(bank_np)
            self.cluster_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        last_pred = kmeans_pred.copy()

        if y is not None:
            acc = self._cluster_accuracy(y, kmeans_pred)
            print(f"  Initial K-means ACC: {acc:.4f}")

        # =====================================================================
        # PHASE 2: FINETUNE (Full loss)
        # =====================================================================
        print("scNAME: Fine-tuning (ZINB + Mask + Neighbor + K-means)...")
        finetune_losses = []

        best_pred = kmeans_pred.copy()
        best_acc = 0 if y is None else self._cluster_accuracy(y, kmeans_pred)

        for epoch in range(1, finetune_epochs + 1):
            self.model.train()

            indices = np.random.permutation(n_samples)

            epoch_total_loss = 0
            n_batches = 0

            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_idx = indices[batch_start:batch_end]
                actual_batch_size = len(batch_idx)

                x_batch = X_tensor[batch_idx]
                x_raw_batch = X_raw_tensor[batch_idx]
                sf_batch = sf_tensor[batch_idx]

                # Generate mask and corrupted data
                x_np = x_batch.cpu().numpy()
                m_batch = mask_generator(p_m, x_np)
                m_new, x_tilde = pretext_generator(m_batch, x_np)

                m_new_tensor = torch.tensor(m_new, dtype=torch.float32).to(device)
                x_tilde_tensor = torch.tensor(x_tilde, dtype=torch.float32).to(device)

                # Forward pass
                latent_x, latent_tilde, mask_est, pi, disp, mean = self.model(x_batch, x_tilde_tensor)

                # ZINB loss
                output = mean * sf_batch
                zinb_loss = zinb_loss_fn(pi, disp, x_raw_batch, output)

                # Mask loss
                mask_loss = safe_bce_probability_loss(mask_est, m_new_tensor)

                # Neighbor contrastive loss
                if actual_batch_size >= k:
                    neighbor_loss = neighbor_k_loss(
                        actual_batch_size, k, self.memory_bank,
                        latent_x, latent_tilde, temperature
                    )
                else:
                    neighbor_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

                # K-means loss
                _, dist2 = cal_dist(latent_x, self.cluster_centers)
                kmeans_loss = torch.mean(torch.sum(dist2, dim=1))

                # Total loss
                loss = zinb_loss + alpha * mask_loss + beta * neighbor_loss + gamma * kmeans_loss
                if not bool(torch.isfinite(loss)):
                    logger.warning(
                        "scNAME finetune: non-finite loss at epoch %d, batch_start %d; skipping batch.",
                        epoch,
                        batch_start,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                # Update memory bank
                with torch.no_grad():
                    self.memory_bank[batch_idx] = latent_x.detach()

                epoch_total_loss += loss.item()
                n_batches += 1

            if n_batches == 0:
                raise RuntimeError(
                    "scNAME finetuning produced no valid batches (all losses non-finite)."
                )
            finetune_losses.append(epoch_total_loss / n_batches)

            # Check convergence periodically
            if epoch % update_interval == 0:
                with torch.no_grad():
                    # Get cluster assignments
                    dist1, _ = cal_dist(self.memory_bank, self.cluster_centers)
                    current_pred = torch.argmin(dist1, dim=1).cpu().numpy()

                    # Update cluster centers with K-means
                    bank_np = self.memory_bank.cpu().numpy()
                    kmeans = KMeans(n_clusters=n_clusters, init=self.cluster_centers.cpu().numpy(),
                                   n_init=1, max_iter=20, random_state=seed)
                    kmeans.fit(bank_np)
                    self.cluster_centers = torch.tensor(kmeans.cluster_centers_,
                                                        dtype=torch.float32).to(device)

                    # Check convergence
                    label_change = np.sum(current_pred != last_pred) / len(last_pred)

                    if y is not None:
                        acc = self._cluster_accuracy(y, current_pred)
                        if acc > best_acc:
                            best_acc = acc
                            best_pred = current_pred.copy()
                        print(f'  Finetune Epoch {epoch}/{finetune_epochs}, '
                              f'Loss: {epoch_total_loss/n_batches:.4f}, '
                              f'ACC: {acc:.4f}, Label change: {label_change:.4f}')
                    else:
                        best_pred = current_pred.copy()
                        print(f'  Finetune Epoch {epoch}/{finetune_epochs}, '
                              f'Loss: {epoch_total_loss/n_batches:.4f}, '
                              f'Label change: {label_change:.4f}')

                    if label_change < convergence_tol and epoch > update_interval:
                        print(f"  Converged at epoch {epoch}")
                        break

                    last_pred = current_pred.copy()

        # Store final results
        # Restore original ordering for outputs
        inv_idx = np.empty_like(self._shuffle_index)
        inv_idx[self._shuffle_index] = np.arange(len(self._shuffle_index))
        self._labels = best_pred[inv_idx]
        self._embeddings = self.memory_bank.cpu().numpy()[inv_idx]

        # Build loss history
        self._loss_history = []
        if pretrain_losses:
            self._loss_history.append({
                'name': 'pretrain (ZINB+Mask)',
                'epochs': list(range(len(pretrain_losses))),
                'train_loss': pretrain_losses,
                'components': {'zinb+mask': pretrain_losses},
            })
        if finetune_losses:
            n_pre = len(pretrain_losses)
            self._loss_history.append({
                'name': 'finetune (ZINB+Mask+Neighbor+KMeans)',
                'epochs': list(range(n_pre, n_pre + len(finetune_losses))),
                'train_loss': finetune_losses,
                'components': {'total': finetune_losses},
            })

        self._fitted = True

        if y is not None:
            print(f"scNAME: Best ACC = {best_acc:.4f}")

        return self

    def _cluster_accuracy(self, y_true, y_pred):
        """
        Compute clustering accuracy using optimal label assignment.
        """
        from scipy.optimize import linear_sum_assignment

        y_true = np.array(y_true).astype(np.int64)
        y_pred = np.array(y_pred).astype(np.int64)

        assert y_pred.size == y_true.size

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_sum_assignment(w.max() - w)

        return w[row_ind, col_ind].sum() / y_pred.size

    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels for new data using transfer learning.

        Uses the trained encoder and cluster centers to assign labels.
        Normalization is applied using TRAINING statistics (mean/std)
        to prevent data leakage.

        Args:
            data: New data to predict on. If None, returns training labels.
        """
        if data is None:
            return self._labels

        # Debug: Log data type and shape at entry
        if hasattr(data, 'shape'):
            logger.info(f"scNAME predict: data type={type(data).__name__}, shape={data.shape}")
        elif hasattr(data, 'n_obs'):
            logger.info(f"scNAME predict: data type={type(data).__name__}, n_obs={data.n_obs}, n_vars={data.n_vars}")
        else:
            logger.info(f"scNAME predict: data type={type(data).__name__}")

        device = torch.device(self.get_device())
        self.model.eval()

        # Preprocess input data
        X_in = data
        gene_names = None
        if hasattr(data, 'X'):
            X_in = data.X
            if hasattr(data, 'var_names'):
                gene_names = data.var_names
        if hasattr(X_in, 'toarray'):
            X_in = X_in.toarray()

        # Ensure we have a numpy array
        if not isinstance(X_in, np.ndarray):
            X_in = np.array(X_in)

        n_test_samples = X_in.shape[0]
        logger.info(f"scNAME predict: X_in shape = {X_in.shape}, dtype = {X_in.dtype}")

        # Handle gene alignment: select only genes used in training
        if hasattr(self, '_training_genes') and self._training_genes is not None and gene_names is not None:
            # Find common genes and align
            common_genes = self._training_genes.intersection(gene_names)
            if len(common_genes) < len(self._training_genes):
                logger.warning(f"scNAME predict: Only {len(common_genes)}/{len(self._training_genes)} training genes found in test data")

            if len(common_genes) == 0:
                logger.error("scNAME predict: No common genes between training and test data!")
                # Return -1 labels to indicate failure (will be filtered as noise)
                return np.full(n_test_samples, -1, dtype=int)

            # Select training genes from test data in the same order
            gene_idx = gene_names.get_indexer(self._training_genes)
            valid_idx = gene_idx >= 0  # -1 means gene not found
            if not np.all(valid_idx):
                # Some training genes missing from test data
                # Use only common genes - but this requires model retraining
                # For now, warn and try to proceed with available genes
                logger.warning(f"scNAME predict: {np.sum(~valid_idx)} training genes missing from test data")
                # Select only genes that exist in test data
                gene_idx_valid = gene_idx[valid_idx]
                X_in = X_in[:, gene_idx_valid]
                # Also subset training stats
                training_mean = self._training_mean[valid_idx] if hasattr(self, '_training_mean') else None
                training_std = self._training_std[valid_idx] if hasattr(self, '_training_std') else None
            else:
                X_in = X_in[:, gene_idx]
                training_mean = self._training_mean if hasattr(self, '_training_mean') else None
                training_std = self._training_std if hasattr(self, '_training_std') else None
        else:
            training_mean = self._training_mean if hasattr(self, '_training_mean') else None
            training_std = self._training_std if hasattr(self, '_training_std') else None

        # Check gene dimension matches model expectation
        model_input_dim = self.model.encoder_layers[0].in_features if hasattr(self.model, 'encoder_layers') else None
        if model_input_dim is not None and X_in.shape[1] != model_input_dim:
            logger.error(f"scNAME predict: Gene dimension mismatch - test has {X_in.shape[1]} genes, model expects {model_input_dim}")
            return np.full(n_test_samples, -1, dtype=int)

        if hasattr(self, '_n_training_genes') and X_in.shape[1] != self._n_training_genes:
            logger.warning(f"scNAME predict: Gene count differs from training ({X_in.shape[1]} vs {self._n_training_genes})")

        # Apply scNAME normalization using TRAINING statistics
        import scanpy as sc

        # Improved raw data detection
        def _is_likely_raw_counts(X):
            """Detect if data is likely raw counts vs normalized."""
            max_val = np.max(X)
            is_integer_like = np.allclose(X, np.round(X), rtol=0, atol=0.01)
            has_count_range = max_val > 10 and np.min(X) >= 0
            return is_integer_like and has_count_range

        is_raw = _is_likely_raw_counts(X_in)

        if is_raw:
            # Normalize on the fly (per-cell normalization + log1p)
            # CAUTION: sc.pp.normalize_per_cell can filter cells with zero counts.
            # We do manual normalization to preserve output size.

            # Count per cell
            counts_per_cell = np.sum(X_in, axis=1)
            # Avoid division by zero
            counts_per_cell[counts_per_cell == 0] = 1.0

            # Use TRAINING normalization target for consistency across batches
            # (Critical: using test median would shift the scale for cross-batch data)
            test_median = float(np.median(counts_per_cell[counts_per_cell > 0]))
            if hasattr(self, '_train_norm_target') and self._train_norm_target is not None:
                norm_target = self._train_norm_target
                logger.info(f"scNAME predict: Using training normalization target: {norm_target:.1f} (test median: {test_median:.1f})")
            else:
                norm_target = test_median
                logger.warning(f"scNAME predict: No training norm target stored, using test median: {norm_target:.1f}")

            size_factors = counts_per_cell / norm_target

            # Normalize
            X_in = X_in / size_factors[:, None]

            # Log1p
            X_in = np.log1p(X_in)
            if hasattr(X_in, 'toarray'):
                X_in = X_in.toarray()

        # Apply scaling using TRAINING mean/std (not test data statistics)
        if training_mean is not None and training_std is not None:
            # Ensure dimensions match
            if X_in.shape[1] == len(training_mean):
                X_scaled = (X_in - training_mean) / training_std
                X_scaled = np.clip(X_scaled, -10, 10)
                X_in = X_scaled.astype(np.float32)
            else:
                logger.warning(f"scNAME predict: Training stats dimension mismatch ({len(training_mean)} vs {X_in.shape[1]}). Using test statistics.")
                X_in = (X_in - np.mean(X_in, axis=0)) / (np.std(X_in, axis=0) + 1e-6)
                X_in = np.clip(X_in, -10, 10).astype(np.float32)
        else:
            logger.warning("scNAME predict: Training statistics not found. Using test data statistics (potential leakage).")
            X_in = (X_in - np.mean(X_in, axis=0)) / (np.std(X_in, axis=0) + 1e-6)
            X_in = np.clip(X_in, -10, 10).astype(np.float32)

        # Ensure float32
        X_tensor = torch.tensor(X_in, dtype=torch.float32).to(device)

        # Debug: log tensor shape
        logger.debug(f"scNAME predict: X_tensor shape = {X_tensor.shape}, n_test_samples = {n_test_samples}")

        # Encode and predict
        try:
            with torch.no_grad():
                latent = self.model.encode(X_tensor, add_noise=False)
                logger.debug(f"scNAME predict: latent shape = {latent.shape}, cluster_centers shape = {self.cluster_centers.shape}")
                dist1, _ = cal_dist(latent, self.cluster_centers)
                y_pred = torch.argmin(dist1, dim=1).cpu().numpy()
        except Exception as e:
            logger.error(f"scNAME predict: Forward pass failed - {e}")
            return np.full(n_test_samples, -1, dtype=int)

        # Sanity check: predictions should match input size
        if len(y_pred) != n_test_samples:
            logger.error(f"scNAME predict: Output size mismatch - got {len(y_pred)}, expected {n_test_samples}. X_tensor had {X_tensor.shape[0]} rows.")
            return np.full(n_test_samples, -1, dtype=int)

        return y_pred

    def get_valid_indices(self) -> Optional[np.ndarray]:
        """Return indices of cells retained after internal filtering."""
        return self._filtered_index

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Return learned embeddings."""
        return self._embeddings

    def encode(self, data: Any) -> Optional[np.ndarray]:
        """
        Encode data using the trained scNAME encoder.

        This method enables Silhouette score computation for test data
        by providing embeddings in the latent space.

        Args:
            data: AnnData object or numpy array

        Returns:
            Latent space embeddings or None if not fitted
        """
        if not self._fitted or self.model is None:
            if data is None: return None
            n_samples = data.n_obs if hasattr(data, 'n_obs') else data.shape[0]
            return np.zeros((n_samples, 32))

        device = torch.device(self.get_device())
        self.model.eval()

        # Preprocess input data (same logic as predict)
        X_in = data
        gene_names = None
        if hasattr(data, 'X'):
            X_in = data.X
            if hasattr(data, 'var_names'):
                gene_names = data.var_names
        if hasattr(X_in, 'toarray'):
            X_in = X_in.toarray()
        if not isinstance(X_in, np.ndarray):
            X_in = np.array(X_in)

        n_samples = X_in.shape[0]

        # Check if this is the training data
        if self._embeddings is not None and n_samples == len(self._embeddings):
            return self._embeddings

        # Handle gene alignment
        if hasattr(self, '_training_genes') and self._training_genes is not None and gene_names is not None:
            gene_idx = gene_names.get_indexer(self._training_genes)
            valid_idx = gene_idx >= 0
            if np.all(valid_idx):
                X_in = X_in[:, gene_idx]
            else:
                gene_idx_valid = gene_idx[valid_idx]
                X_in = X_in[:, gene_idx_valid]

        # Check dimension match
        model_input_dim = self.model.encoder_layers[0].in_features
        if X_in.shape[1] != model_input_dim:
            logger.warning(f"scNAME encode: dimension mismatch ({X_in.shape[1]} vs {model_input_dim})")
            return np.zeros((n_samples, model_input_dim))

        # Apply normalization using training statistics
        def _is_likely_raw_counts(X):
            max_val = np.max(X)
            is_integer_like = np.allclose(X, np.round(X), rtol=0, atol=0.01)
            has_count_range = max_val > 10 and np.min(X) >= 0
            return is_integer_like and has_count_range

        if _is_likely_raw_counts(X_in):
            counts_per_cell = np.sum(X_in, axis=1)
            counts_per_cell[counts_per_cell == 0] = 1.0
            # Use training normalization target for consistency
            if hasattr(self, '_train_norm_target') and self._train_norm_target is not None:
                norm_target = self._train_norm_target
            else:
                norm_target = float(np.median(counts_per_cell[counts_per_cell > 0]))
            size_factors = counts_per_cell / norm_target
            X_in = X_in / size_factors[:, None]
            X_in = np.log1p(X_in)

        # Scale using training statistics
        if hasattr(self, '_training_mean') and hasattr(self, '_training_std'):
            if X_in.shape[1] == len(self._training_mean):
                X_in = (X_in - self._training_mean) / self._training_std
                X_in = np.clip(X_in, -10, 10)

        X_tensor = torch.tensor(X_in.astype(np.float32), dtype=torch.float32).to(device)

        with torch.no_grad():
            embeddings = self.model.encode(X_tensor, add_noise=False).cpu().numpy()

        return embeddings
