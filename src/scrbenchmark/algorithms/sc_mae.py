"""
scMAE: Single-Cell Masked Autoencoder implementation.
Based on https://github.com/scMAE/scMAE (scMAE-main)

Masked Autoencoder for single-cell RNA-seq data with mask prediction and
shuffle corruption strategy.

Reference: scMAE paper - Masked Autoencoder for single-cell genomics

This implementation is strictly faithful to the original authors' code from:
- scMAE-main/model.py (AutoEncoder architecture)
- scMAE-main/main.py (training loop with epoch 80 evaluation)
- scMAE-main/datasets.py (IterativeSVDImputator, apply_noise)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gc
import logging

logger = logging.getLogger(__name__)

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType


# ============================================================================
# IterativeSVDImputator: Exact copy from scMAE-main/datasets.py
# ============================================================================

default_svd_params = {
    "n_components": 128,
    "random_state": 42,
    "n_oversamples": 20,
    "n_iter": 7,
}


class IterativeSVDImputator:
    """
    Iterative SVD imputation for missing values (zeros).
    Exact implementation from scMAE-main/datasets.py.
    """
    def __init__(self, svd_params=None, iters=2):
        self.missing_values = 0.0
        self.svd_params = svd_params if svd_params is not None else default_svd_params
        self.iters = iters
        self.svd_decomposers = [None for _ in range(self.iters)]

    def fit(self, X):
        from sklearn.decomposition import TruncatedSVD
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            self.svd_decomposers[i] = TruncatedSVD(**self.svd_params)
            self.svd_decomposers[i].fit(transformed_X)
            new_X = self.svd_decomposers[i].inverse_transform(
                self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]

    def transform(self, X):
        mask = X == self.missing_values
        transformed_X = X.copy()
        for i in range(self.iters):
            new_X = self.svd_decomposers[i].inverse_transform(
                self.svd_decomposers[i].transform(transformed_X))
            transformed_X[mask] = new_X[mask]
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ============================================================================
# apply_noise: Exact copy from scMAE-main/datasets.py
# ============================================================================

def apply_noise(X, p):
    """
    Apply shuffle corruption exactly as in scMAE-main/datasets.py.

    For each gene position, with probability p, replace the value with
    the value from a random cell (shuffle corruption).

    Args:
        X: Input tensor (batch_size, num_genes) on device
        p: List/tensor of masking probabilities per gene position

    Returns:
        corrupted_X: Corrupted input
        masked: Binary mask (1 = corrupted, 0 = unchanged)
    """
    import torch

    p = torch.tensor(p)
    should_swap = torch.bernoulli(
        p.to(X.device) * torch.ones((X.shape)).to(X.device)
    )
    corrupted_X = torch.where(
        should_swap == 1, X[torch.randperm(X.shape[0]).to(X.device)], X
    )
    masked = (corrupted_X != X).float()
    return corrupted_X, masked


# ============================================================================
# res_search_fixed_clus: Exact copy from scMAE-main/main.py
# ============================================================================

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02, verbose=True):
    """
    Search for Leiden resolution that produces target cluster count.
    Exact implementation from scMAE-main/main.py.

    Args:
        adata: AnnData matrix with computed neighbors
        fixed_clus_count: Target number of clusters
        increment: Resolution search increment (default 0.02)
        verbose: Print progress (default True)

    Returns:
        resolution: Best resolution value
    """
    import scanpy as sc
    import pandas as pd

    dis = []
    resolutions = sorted(list(np.arange(0.01, 2.5, increment)), reverse=True)
    n_resolutions = len(resolutions)
    res_new = []

    if verbose:
        print(f"       Searching resolution for {fixed_clus_count} clusters (max {n_resolutions} iterations)...")

    for i, res in enumerate(resolutions):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(
            adata.obs['leiden']).leiden.unique())
        dis.append(abs(count_unique_leiden - fixed_clus_count))
        res_new.append(res)

        if verbose and (i % 20 == 0 or count_unique_leiden == fixed_clus_count):
            print(f"       Resolution {res:.2f}: {count_unique_leiden} clusters (target: {fixed_clus_count})")

        if count_unique_leiden == fixed_clus_count:
            if verbose:
                print(f"       Found exact match at resolution {res:.2f}")
            break

    reso = resolutions[np.argmin(dis)]
    if verbose and reso != res:
        print(f"       Best resolution: {reso:.2f} (closest to {fixed_clus_count} clusters)")
    return reso


@AlgorithmRegistry.register
class ScMaeAlgorithm(BaseAlgorithm):
    """
    scMAE: Single-Cell Masked Autoencoder.

    Implements the original scMAE architecture with:
    - Encoder with LayerNorm + Mish activation
    - Separate mask predictor head
    - Decoder that takes concatenated [latent, predicted_mask]
    - Combined loss: weighted MSE reconstruction + BCE mask prediction
    - IterativeSVD imputation (optional, for internal preprocessing)

    Reference: scMAE-main original implementation
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name='sc_mae',
            display_name='scMAE (Masked Autoencoder)',
            description='Single-Cell Masked Autoencoder with mask prediction. '
                       'Uses shuffle corruption strategy and learns to predict '
                       'both the mask and reconstruct the original expression.',
            category='deep_learning',
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes='Uses scMAE preprocessing (raw counts → size factors → log → HVG → scale) '
                               'followed by SVD imputation.',
            has_internal_preprocessing=True,
            recommended_data='raw'
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        return [
            # Network architecture (matching paper: hidden_size=128, latent_dim=32)
            HyperparameterConfig(
                name='hidden_size',
                display_name='Hidden Size',
                param_type=ParamType.INTEGER,
                default=128,
                description='Hidden layer size in encoder (paper: 128)',
                min_value=32,
                max_value=512,
                step=32,
                category='Network',
                tuning_guide='**Impact**: Intermediate representation capacity.\\n'
                            '**Details**: Size of hidden layers before latent space. 128 is the paper default.\\n'
                            '**Recommendation**: Keep at 128 unless you have specific reasons to change.'
            ),
            HyperparameterConfig(
                name='latent_dim',
                display_name='Latent Dimension',
                param_type=ParamType.INTEGER,
                default=128,
                description='Latent space dimension (original code: hidden_size=128)',
                min_value=32,
                max_value=256,
                step=32,
                category='Network',
                tuning_guide='**Impact**: Final representation capacity for clustering.\\n'
                            '**Details**: In original code, latent = hidden_size = 128. We default to 128 to match.\\n'
                            '**Recommendation**: Keep at 128 for strict reproduction. Lower (e.g. 32) can improve silhouette.'
            ),
            HyperparameterConfig(
                name='dropout',
                display_name='Dropout Rate',
                param_type=ParamType.FLOAT,
                default=0.0,
                description='Dropout probability at encoder input (original: 0)',
                min_value=0.0,
                max_value=0.5,
                step=0.05,
                category='Network'
            ),
            # Masking parameters (matching original: p=0.4 for all genes)
            HyperparameterConfig(
                name='masking_rate',
                display_name='Masking Rate',
                param_type=ParamType.FLOAT,
                default=0.4,
                description='Probability of masking each gene (original: 0.4)',
                min_value=0.1,
                max_value=0.9,
                step=0.1,
                category='Masking',
                tuning_guide='**Impact**: Difficulty of the self-supervised task.\\n'
                            '**Details**: Each gene has this probability of being corrupted via shuffle.\\n'
                            '**Recommendation**: 0.4 is the original default.'
            ),
            HyperparameterConfig(
                name='masked_data_weight',
                display_name='Masked Data Weight',
                param_type=ParamType.FLOAT,
                default=0.75,
                description='Weight for masked positions in reconstruction loss (original: 0.75)',
                min_value=0.5,
                max_value=1.0,
                step=0.05,
                category='Loss',
                tuning_guide='**Impact**: Focus on masked vs unmasked reconstruction.\\n'
                            '**Details**: Higher values emphasize reconstructing masked positions.\\n'
                            '**Recommendation**: 0.75 is the original default.'
            ),
            HyperparameterConfig(
                name='mask_loss_weight',
                display_name='Mask Loss Weight',
                param_type=ParamType.FLOAT,
                default=0.7,
                description='Weight for mask prediction loss (original: 0.7)',
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                category='Loss',
                tuning_guide='**Impact**: Balance between mask prediction and reconstruction.\\n'
                            '**Details**: Total loss = (1-weight)*recon_loss + weight*mask_loss.\\n'
                            '**Recommendation**: 0.7 is the original default.'
            ),
            # Training parameters (matching original: epochs=100, batch_size=256, lr=1e-3)
            HyperparameterConfig(
                name='epochs',
                display_name='Training Epochs',
                param_type=ParamType.INTEGER,
                default=100,
                description='Number of training epochs (original: 100)',
                max_value=500,
                step=10,
                category='Training',
                tuning_guide='**Impact**: Model convergence.\\n'
                            '**Details**: Original uses 100 epochs with evaluation at epoch 80.\\n'
                            '**Recommendation**: 100 is sufficient for most datasets.'
            ),
            HyperparameterConfig(
                name='eval_epoch',
                display_name='Evaluation Epoch',
                param_type=ParamType.INTEGER,
                default=80,
                description='Epoch at which to extract embeddings for clustering (original: 80)',
                max_value=500,
                step=10,
                category='Training',
                tuning_guide='**Impact**: When to extract final embeddings.\\n'
                            '**Details**: Original evaluates at epoch 80. Set to -1 to use final epoch.\\n'
                            '**Recommendation**: 80 is the original default.'
            ),
            HyperparameterConfig(
                name='batch_size',
                display_name='Batch Size',
                param_type=ParamType.INTEGER,
                default=256,
                description='Training batch size (original: 256)',
                min_value=32,
                max_value=1024,
                step=32,
                category='Training'
            ),
            HyperparameterConfig(
                name='learning_rate',
                display_name='Learning Rate',
                param_type=ParamType.FLOAT,
                default=0.001,
                description='Learning rate for Adam optimizer (original: 1e-3)',
                min_value=0.0001,
                max_value=0.01,
                step=0.0001,
                category='Training'
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
                name='clustering_method',
                display_name='Clustering Method',
                param_type=ParamType.CHOICE,
                default='auto',
                description='Clustering: auto (KMeans if <10k cells, Leiden otherwise), kmeans, or leiden',
                choices=['auto', 'kmeans', 'leiden'],
                category='Clustering',
                tuning_guide='**Impact**: Clustering algorithm selection.\\n'
                            '**Details**: Original uses KMeans for <10k cells, Leiden for larger datasets.\\n'
                            '**Recommendation**: Use "auto" to match original behavior.'
            ),
            # Internal preprocessing (SVD imputation)
            HyperparameterConfig(
                name='apply_svd_imputation',
                display_name='Apply SVD Imputation',
                param_type=ParamType.BOOLEAN,
                default=True,
                description='Apply IterativeSVD imputation as in original (handles zeros)',
                category='Preprocessing',
                tuning_guide='**Impact**: Missing value handling.\\n'
                            '**Details**: Original applies SVD imputation after scanpy preprocessing.\\n'
                            '**Recommendation**: Keep True for strict faithfulness to original.'
            ),
            HyperparameterConfig(
                name='svd_n_components',
                display_name='SVD Components',
                param_type=ParamType.INTEGER,
                default=128,
                description='Number of SVD components for imputation (original: 128)',
                min_value=32,
                max_value=256,
                step=32,
                category='Preprocessing',
                advanced=True
            ),
            HyperparameterConfig(
                name='svd_iters',
                display_name='SVD Iterations',
                param_type=ParamType.INTEGER,
                default=2,
                description='Number of SVD imputation iterations (original: 2)',
                min_value=1,
                max_value=5,
                step=1,
                category='Preprocessing',
                advanced=True
            ),
            # HVG Selection parameters (as in original datasets.py)
            HyperparameterConfig(
                name='n_hvg',
                display_name='Number of HVGs',
                param_type=ParamType.INTEGER,
                default=1000,
                description='Number of highly variable genes to select (original: 1000). Set to 0 to skip.',
                min_value=0,
                max_value=5000,
                step=100,
                category='Preprocessing',
                tuning_guide='**Impact**: Gene selection for model input.\\n'
                            '**Details**: Original scMAE uses 1000 HVGs with specific filtering.\\n'
                            '**Recommendation**: Keep at 1000 to match original. Set to 0 to use pre-selected genes.'
            ),
            HyperparameterConfig(
                name='hvg_min_mean',
                display_name='HVG Min Mean',
                param_type=ParamType.FLOAT,
                default=0.0125,
                description='Minimum mean expression for HVG selection (original: 0.0125)',
                min_value=0.0,
                max_value=1.0,
                step=0.0025,
                category='Preprocessing',
                advanced=True
            ),
            HyperparameterConfig(
                name='hvg_max_mean',
                display_name='HVG Max Mean',
                param_type=ParamType.FLOAT,
                default=3.0,
                description='Maximum mean expression for HVG selection (original: 3)',
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                category='Preprocessing',
                advanced=True
            ),
            HyperparameterConfig(
                name='hvg_min_disp',
                display_name='HVG Min Dispersion',
                param_type=ParamType.FLOAT,
                default=0.5,
                description='Minimum dispersion for HVG selection (original: 0.5)',
                min_value=0.0,
                max_value=2.0,
                step=0.1,
                category='Preprocessing',
                advanced=True
            ),
            HyperparameterConfig(
                name='sort_genes_by_mean',
                display_name='Sort Genes by Mean',
                param_type=ParamType.BOOLEAN,
                default=True,
                description='Sort genes by mean expression as in original preprocessing (original: True)',
                category='Preprocessing',
                advanced=True
            ),
            # General
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
                name='use_own_preprocessing',
                display_name='Use Own Preprocessing',
                param_type=ParamType.BOOLEAN,
                default=True,
                description='Use scMAE\'s complete preprocessing pipeline (normalize, log1p, HVG selection, scale) '
                           'from raw counts. When enabled, scMAE ignores the benchmark\'s HVG selection and '
                           'performs its own preprocessing exactly as in the original paper.',
                category='Data',
                tuning_guide='**Impact**: Controls whether scMAE uses its own preprocessing or the benchmark\'s.\\n'
                            '**When True**: scMAE retrieves raw counts (pre-HVG) and applies its full pipeline:\\n'
                            '  - normalize_per_cell → log1p → HVG selection (1000 genes) → scale → SVD imputation\\n'
                            '**When False**: scMAE uses the benchmark\'s preprocessed data as-is.\\n'
                            '**Recommendation**: Keep True to match the original scMAE implementation.'
            ),
        ]

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.model = None
        self.clusterer = None
        self.svd_imputator = None
        self._scmae_sorted = False
        self._valid_indices = None
        # For benchmark mode: store preprocessing params to apply on test data
        self._selected_hvg_genes = None
        self._scale_mean = None
        self._scale_std = None
        self._train_n_cells = None
        self._train_norm_target = None  # Normalization target (median of counts) from training

    def get_valid_indices(self) -> Optional[np.ndarray]:
        """Return indices of samples used in the final model."""
        return self._valid_indices

    def get_num_parameters(self) -> Optional[int]:
        """Get the number of trainable parameters."""
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _build_model(self, num_genes: int):
        """
        Build the scMAE model EXACTLY as in original model.py.

        Original architecture (model.py):
        - Encoder: Dropout -> Linear(256) -> LayerNorm -> Mish -> Linear(hidden_size) -> LayerNorm -> Mish -> Linear(hidden_size)
        - Mask Predictor: Linear(hidden_size -> num_genes)
        - Decoder: Linear(hidden_size + num_genes -> num_genes)

        NOTE: The original code uses hidden_size=128 for the latent space, NOT latent_dim=32.
        The latent_dim parameter in main.py is never actually used in the model!
        """
        import torch
        import torch.nn as nn
        from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
        from torch.nn.functional import mse_loss as mse

        hidden_size = self.params.get('hidden_size', 128)
        # Note: latent_dim is set to hidden_size to match original code where latent = hidden_size = 128
        latent_dim = self.params.get('latent_dim', 128)
        dropout = self.params.get('dropout', 0.0)
        masked_data_weight = self.params.get('masked_data_weight', 0.75)
        mask_loss_weight = self.params.get('mask_loss_weight', 0.7)

        class AutoEncoder(nn.Module):
            """
            AutoEncoder EXACTLY as in scMAE-main/model.py.
            Original uses hidden_size=128 for latent dimension (NOT latent_dim=32).
            """
            def __init__(self, num_genes, hidden_size=128, latent_dim=128, dropout=0,
                        masked_data_weight=0.75, mask_loss_weight=0.7):
                super().__init__()
                self.num_genes = num_genes
                self.latent_dim = latent_dim
                self.masked_data_weight = masked_data_weight
                self.mask_loss_weight = mask_loss_weight

                # Encoder: num_genes -> 256 -> hidden_size -> hidden_size (latent)
                # Original model.py:
                # 1. Input -> 256 (Dropout, Linear, LayerNorm, Mish)
                # 2. 256 -> hidden_size (Linear, LayerNorm, Mish)
                # 3. hidden_size -> hidden_size (Linear) - This is the bottleneck/latent
                
                # NOTE: The last layer in original code uses 'hidden_size' as output size.
                # We use self.latent_dim to keep flexibility, but defaults will be hidden_size=128, latent_dim=128.
                
                self.encoder = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_genes, 256),
                    nn.LayerNorm(256),
                    nn.Mish(inplace=True),
                    nn.Linear(256, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.Mish(inplace=True),
                    nn.Linear(hidden_size, latent_dim)  # Output latent representation
                )

                # Mask predictor: Linear(latent_dim, num_genes)
                self.mask_predictor = nn.Linear(latent_dim, num_genes)

                # Decoder: Linear(latent_dim + num_genes, num_genes)
                self.decoder = nn.Linear(
                    in_features=latent_dim + num_genes,
                    out_features=num_genes
                )

            def forward_mask(self, x):
                """Forward pass with mask prediction."""
                latent = self.encoder(x)
                predicted_mask = self.mask_predictor(latent)
                reconstruction = self.decoder(
                    torch.cat([latent, predicted_mask], dim=1)
                )
                return latent, predicted_mask, reconstruction

            def loss_mask(self, x, y, mask, return_components=False):
                """
                Compute combined loss.

                Args:
                    x: Corrupted input
                    y: Original target
                    mask: Binary mask (1 = corrupted, 0 = unchanged)

                Returns:
                    latent: Latent representation
                    loss: Combined reconstruction + mask prediction loss
                """
                latent, predicted_mask, reconstruction = self.forward_mask(x)

                # Weighted reconstruction loss
                w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)
                reconstruction_loss = (1 - self.mask_loss_weight) * torch.mul(
                    w_nums, mse(reconstruction, y, reduction='none')
                )

                # Mask prediction loss (BCE with logits)
                mask_loss = self.mask_loss_weight * \
                    bce_logits(predicted_mask, mask, reduction="mean")

                reconstruction_loss = reconstruction_loss.mean()

                loss = reconstruction_loss + mask_loss
                if return_components:
                    return latent, loss, reconstruction_loss, mask_loss
                return latent, loss

            def feature(self, x):
                """Get latent embeddings (for inference)."""
                latent = self.encoder(x)
                return latent

        return AutoEncoder(
            num_genes=num_genes,
            hidden_size=hidden_size,
            latent_dim=latent_dim,
            dropout=dropout,
            masked_data_weight=masked_data_weight,
            mask_loss_weight=mask_loss_weight
        )

    def _apply_internal_preprocessing(self, X: np.ndarray) -> np.ndarray:
        """
        Apply internal preprocessing as in original scMAE-main/datasets.py.

        This includes:
        1. (Optional) Sort genes by mean expression
        2. IterativeSVD imputation for zero values
        """
        # Sort genes by mean expression only if not already sorted
        if self.params.get('sort_genes_by_mean', False) and not self._scmae_sorted:
            # ORIGINAL BUG FIX:
            # Original code sorted by mean of the CURRENT matrix 'X'.
            # If X is already scaled (mean~0), this sort is random/unstable.
            # If X is raw/log1p, it's fine for training, BUT...
            # For test data, we must use the EXACT SAME gene order as training.
            # Sorting test data by its own means would result in different order!
            
            # Solution:
            # 1. Sort based on the UN-SCALED means (stored in self._scale_mean during preprocessing)
            # 2. Store the permutation indices
            # 3. Apply this permutation to X
            
            if self._scale_mean is not None:
                # Robust sort using the stored means from preprocessing (log1p space)
                print("scMAE: Sorting genes based on stored training means (stable)")
                sorted_indices = np.argsort(self._scale_mean)
            else:
                # Fallback (should not happen if _scanpy_preprocess called)
                print("scMAE: Sorting genes based on current X (less stable)")
                sorted_indices = np.argsort(np.mean(X, axis=0))
            
            # Store permutation for use in predict()
            self._gene_permutation = sorted_indices
            
            # Apply permutation
            X = X[:, sorted_indices]
            self._scmae_sorted = True
            print(f"scMAE: Genes sorted. stored permutation size: {len(self._gene_permutation)}")

        # SVD imputation (as in original)
        if self.params.get('apply_svd_imputation', True):
            svd_params = {
                "n_components": min(self.params.get('svd_n_components', 128), X.shape[1] - 1),
                "random_state": self.params.get('random_state', 42),
                "n_oversamples": 20,
                "n_iter": 7,
            }
            self.svd_imputator = IterativeSVDImputator(
                svd_params=svd_params,
                iters=self.params.get('svd_iters', 2)
            )
            X = self.svd_imputator.fit_transform(X)

        return X

    def _scanpy_preprocess(self, adata):
        """
        Match scMAE scanpy preprocessing EXACTLY as in original datasets.py normalize() function.

        Pipeline (from datasets.py lines 187-243):
        1. filter_genes(min_counts=1) - optional, disabled for benchmark consistency
        2. filter_cells(min_counts=1) - optional, disabled for benchmark consistency
        3. normalize_per_cell
        4. size_factors = n_counts / median(n_counts)
        5. log1p
        6. highly_variable_genes(min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=1000)
        7. scale
        """
        import scanpy as sc
        import scipy as sp

        norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
        assert 'n_count' not in adata.obs, norm_error
        if adata.X.size < 50e6:
            if sp.sparse.issparse(adata.X):
                assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
            else:
                assert np.all(adata.X.astype(int) == adata.X), norm_error

        # Original does filter_genes/filter_cells but we skip for benchmark consistency
        # sc.pp.filter_genes(adata, min_counts=1)
        # sc.pp.filter_cells(adata, min_counts=1)
        adata.raw = adata.copy()

        # Convert to float to avoid casting error during normalization (int64 -> float64)
        if adata.X.dtype != np.float32 and adata.X.dtype != np.float64:
             adata.X = adata.X.astype(np.float32)

        # Step 3-4: Normalize per cell and compute size factors (original lines 230-233)
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
        # Store normalization target for consistent test preprocessing
        self._train_norm_target = float(np.median(adata.obs.n_counts))

        # Step 5: Log transform (original line 237)
        sc.pp.log1p(adata)

        # Step 6: HVG selection - EXACTLY as in original datasets.py lines 239-240
        n_hvg = self.params.get('n_hvg', 1000)
        if n_hvg > 0 and adata.n_vars > n_hvg:
            hvg_min_mean = self.params.get('hvg_min_mean', 0.0125)
            hvg_max_mean = self.params.get('hvg_max_mean', 3.0)
            hvg_min_disp = self.params.get('hvg_min_disp', 0.5)

            sc.pp.highly_variable_genes(
                adata,
                min_mean=hvg_min_mean,
                max_mean=hvg_max_mean,
                min_disp=hvg_min_disp,
                n_top_genes=n_hvg,
                subset=True  # Keep only HVGs
            )
            # Store selected HVG names for use in predict()
            self._selected_hvg_genes = list(adata.var_names)
            print(f"scMAE: Selected {adata.n_vars} HVGs (requested {n_hvg}) with "
                  f"min_mean={hvg_min_mean}, max_mean={hvg_max_mean}, min_disp={hvg_min_disp}")
        else:
            self._selected_hvg_genes = list(adata.var_names)

        # Step 7: Scale (original line 242)
        # Store mean and std for transform on new data
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        self._scale_mean = np.mean(X, axis=0)
        self._scale_std = np.std(X, axis=0)
        self._scale_std[self._scale_std == 0] = 1.0  # Avoid division by zero

        sc.pp.scale(adata)

        return adata

    def _prepare_scmae_inputs(self, data: Any, labels: Optional[Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare X and labels using scMAE preprocessing from the authors' code.

        When use_own_preprocessing=True (default):
        - Retrieves raw counts BEFORE the benchmark's HVG selection (from adata.uns['pre_hvg_counts'])
        - Applies scMAE's complete preprocessing pipeline (normalize, log1p, HVG selection, scale)
        - This ensures faithful reproduction of the original scMAE methodology

        When use_own_preprocessing=False:
        - Uses the benchmark's preprocessed data as-is
        - Skips scMAE's internal preprocessing
        """
        import anndata as ad
        import pandas as pd

        use_own_preprocessing = self.params.get('use_own_preprocessing', True)
        n_hvg = self.params.get('n_hvg', 1000)

        X_counts = None
        gene_names = None

        if hasattr(data, 'X'):
            n_benchmark_genes = data.n_vars if hasattr(data, 'n_vars') else data.X.shape[1]

            if use_own_preprocessing:
                # Priority 1: Use pre-HVG counts (complete gene set before benchmark's HVG selection)
                # Priority 1: Use pre-HVG counts (complete gene set before benchmark's HVG selection)
                found_raw = False
                
                # Priority 1: Use pre-HVG counts (complete gene set) IF dimensions match
                if hasattr(data, 'uns') and 'pre_hvg_counts' in data.uns:
                    X_candidate = data.uns['pre_hvg_counts']
                    if X_candidate.shape[0] == data.n_obs:
                        X_counts = X_candidate
                        gene_names = data.uns.get('pre_hvg_var_names', None)
                        n_pre_hvg_genes = data.uns.get('pre_hvg_n_genes', X_counts.shape[1])
                        print(f"scMAE: ✓ Using pre-HVG raw counts ({X_counts.shape[0]} cells x {n_pre_hvg_genes} genes)")
                        found_raw = True
                    else:
                        print(f"scMAE: ⚠ pre_hvg_counts shape {X_candidate.shape} != data cells {data.n_obs}. Skipping.")

                # Priority 2: Use original_X layer (may already be HVG-filtered)
                if not found_raw and hasattr(data, 'layers') and 'original_X' in data.layers:
                    X_counts = data.layers['original_X']
                    gene_names = list(data.var_names) if hasattr(data, 'var_names') else None
                    print(f"scMAE: ⚠ Using layers['original_X'] ({n_benchmark_genes} genes)")
                    found_raw = True

                # Priority 3: Use adata.raw
                if not found_raw and hasattr(data, 'raw') and data.raw is not None:
                    X_counts = data.raw.X
                    gene_names = list(data.var_names) if hasattr(data, 'var_names') else None
                    print(f"scMAE: ⚠ Using adata.raw ({n_benchmark_genes} genes)")
                    found_raw = True
                
                # Fallback
                if not found_raw:
                    X_counts = data.X
                    gene_names = list(data.var_names) if hasattr(data, 'var_names') else None
                    print(f"scMAE: ⚠ Using data.X ({n_benchmark_genes} genes)")

            else:
                # use_own_preprocessing=False: Use benchmark's preprocessed data
                X_counts = data.X
                gene_names = list(data.var_names) if hasattr(data, 'var_names') else None
                print(f"scMAE: Using benchmark's preprocessed data ({n_benchmark_genes} genes)")
                print(f"       Skipping scMAE's internal preprocessing.")

            if hasattr(X_counts, 'toarray'):
                X_counts = X_counts.toarray()

        else:
            # data is a numpy array
            X_counts = data
            print(f"scMAE: Input is numpy array ({X_counts.shape[1]} genes)")

        # Convert to integer counts for preprocessing
        X_counts = np.ceil(X_counts).astype(np.int64)
        n_cells = X_counts.shape[0]
        obs_names = pd.Index([str(i) for i in range(n_cells)])

        adata = ad.AnnData(
            X=X_counts,
            var=pd.DataFrame(index=gene_names) if gene_names is not None else None,
            obs=pd.DataFrame(index=obs_names)
        )

        print(f"scMAE: Starting with {adata.n_obs} cells x {adata.n_vars} genes")

        # Apply scMAE's preprocessing if enabled
        if use_own_preprocessing:
            print(f"scMAE: Applying own preprocessing pipeline (normalize → log1p → HVG({n_hvg}) → scale)")
            adata = self._scanpy_preprocess(adata)
        else:
            print(f"scMAE: Skipping own preprocessing (using benchmark data as-is)")

        # Sort genes by mean expression (as in original datasets.py lines 177-178)
        # DISABLE sorting here to allow _apply_internal_preprocessing to handle it
        # with proper permutation tracking for test set consistency.
        #
        # if self.params.get('sort_genes_by_mean', True):
        #     mean_key = 'mean' if 'mean' in adata.var else 'means' if 'means' in adata.var else None
        #     if mean_key is not None:
        #         sorted_genes = adata.var_names[np.argsort(adata.var[mean_key])]
        #         adata = adata[:, sorted_genes]
        #     else:
        #         gene_means = np.asarray(adata.X.mean(axis=0)).ravel()
        #         sorted_idx = np.argsort(gene_means)
        #         adata = adata[:, sorted_idx]
        #     self._scmae_sorted = True
        #     print(f"scMAE: Sorted {adata.n_vars} genes by mean expression")

        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = X.astype(np.float32)

        filtered_labels = None
        if hasattr(adata, 'obs_names'):
            try:
                self._valid_indices = adata.obs_names.astype(int)
            except (ValueError, TypeError):
                # Fallback if names are not integers
                print("scMAE: Warning - could not convert obs_names to int. Plotting alignment may fail.")
                self._valid_indices = None
        
        if labels is not None:
            label_array = np.array(labels)
            if self._valid_indices is not None:
                filtered_labels = label_array[self._valid_indices]
            else:
                 # Should not happen if labels provided and size changed
                 if len(labels) != len(X):
                     print("scMAE: Warning - labels length mismatch and valid indices not available.")
                     filtered_labels = label_array[:len(X)] # Fallback
                 else:
                     filtered_labels = label_array

        return X, filtered_labels

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'ScMaeAlgorithm':
        """
        Fit scMAE model following the original scMAE-main implementation.

        Training loop matches scMAE-main/main.py:
        - Uses mask_probas = [0.4] * num_genes
        - Evaluates at epoch 80 by default (configurable)
        - Uses KMeans for <10k cells, Leiden with resolution search otherwise

        Args:
            data: AnnData object or numpy array with preprocessed data
            labels: Optional ground truth labels
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        # Get data matrix (apply original preprocessing when using raw counts)
        X, labels = self._prepare_scmae_inputs(data, labels)

        # Apply internal preprocessing (SVD imputation as in original)
        X = self._apply_internal_preprocessing(X)

        # Determine number of clusters
        n_clusters = self.params.get('n_clusters', 0)
        if n_clusters == 0 and labels is not None:
            n_clusters = len(np.unique(labels))
            logger.warning(f"scMAE: n_clusters=0 (Auto). Using ground truth labels to determine k={n_clusters} (Oracle Mode).")
        elif n_clusters == 0:
            n_clusters = 8
            logger.warning(f"scMAE: n_clusters=0 and no labels. Defaulting to k={n_clusters}.")

        # Set random seed (as in original main.py)
        random_state = self.params.get('random_state', 42)
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
            torch.cuda.manual_seed_all(random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Build model
        num_genes = X.shape[1]
        self.model = self._build_model(num_genes)

        # Device selection using standardized get_device
        device_name = self.get_device()
        device = torch.device(device_name)
        
        # Handle MPS float32 requirement
        if device_name == 'mps':
            torch.set_default_dtype(torch.float32)
            
        self.model = self.model.float().to(device)

        # Data loader (original uses batch_size=256, drop_last=True)
        batch_size = self.params.get('batch_size', 256)
        # Create dataset on CPU first, move to device in loop if dataset is large, 
        # or here if small. Standard practice is to keep loader on CPU and move batch to device.
        train_dataset = TensorDataset(torch.FloatTensor(X))
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(X) > batch_size
        )

        # Optimizer (Adam with lr=1e-3 as in original)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.params.get('learning_rate', 0.001)
        )

        # Training parameters
        epochs = self.params.get('epochs', 100)
        eval_epoch = self.params.get('eval_epoch', 80)
        masking_rate = self.params.get('masking_rate', 0.4)

        # mask_probas = [0.4] * num_genes (as in original main.py line 75)
        mask_probas = torch.tensor([masking_rate] * num_genes).to(device)

        self.train_losses = []
        self.train_recon_losses = []
        self.train_mask_losses = []
        self._eval_epoch_embeddings = None

        print(f"scMAE: Training for {epochs} epochs (eval at epoch {eval_epoch if eval_epoch > 0 else epochs})...")

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_recon = 0
            epoch_mask = 0
            epoch_samples = 0

            for batch, in train_loader:
                batch = batch.to(device)

                # Apply shuffle corruption (exactly as in original)
                corrupted_batch, mask = apply_noise(batch, mask_probas)

                optimizer.zero_grad()

                # Forward pass with mask loss (as in original main.py line 96)
                _, loss, recon_loss, mask_loss = self.model.loss_mask(
                    corrupted_batch,
                    batch,
                    mask,
                    return_components=True,
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.size(0)
                epoch_recon += recon_loss.item() * batch.size(0)
                epoch_mask += mask_loss.item() * batch.size(0)
                epoch_samples += batch.size(0)

            avg_loss = epoch_loss / epoch_samples
            avg_recon = epoch_recon / epoch_samples
            avg_mask = epoch_mask / epoch_samples
            self.train_losses.append(avg_loss)
            self.train_recon_losses.append(avg_recon)
            self.train_mask_losses.append(avg_mask)

            # Evaluation at specified epoch (original evaluates at epoch 80)
            if eval_epoch > 0 and epoch == eval_epoch:
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    self._eval_epoch_embeddings = self.model.feature(X_tensor).cpu().numpy()
                print(f"  Epoch {epoch}: Saved embeddings for clustering")

            # Progress logging
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(
                    f"  Epoch {epoch}/{epochs}: loss={avg_loss:.6f} "
                    f"recon={avg_recon:.6f} mask={avg_mask:.6f}"
                )

        # Get final embeddings (use eval_epoch embeddings if available)
        if self._eval_epoch_embeddings is not None and eval_epoch > 0:
            self._embeddings = self._eval_epoch_embeddings
        else:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                self._embeddings = self.model.feature(X_tensor).cpu().numpy()

        # Clustering (original: KMeans for <10k cells, Leiden with resolution search for larger)
        clustering_method = self.params.get('clustering_method', 'auto')

        print(f"scMAE: Starting clustering ({clustering_method}) for {n_clusters} clusters on {len(X)} cells...")

        if clustering_method == 'auto':
            if len(X) < 10000:
                print(f"scMAE: Using KMeans (n_cells < 10000)")
                self._labels = self._kmeans_clustering(self._embeddings, n_clusters)
            else:
                print(f"scMAE: Using Leiden with resolution search (n_cells >= 10000)")
                print(f"       This can take several minutes for {n_clusters} clusters...")
                self._labels = self._leiden_clustering_with_search(self._embeddings, n_clusters)
        elif clustering_method == 'leiden':
            print(f"scMAE: Using Leiden with resolution search (forced)")
            self._labels = self._leiden_clustering_with_search(self._embeddings, n_clusters)
        else:
            print(f"scMAE: Using KMeans (forced)")
            self._labels = self._kmeans_clustering(self._embeddings, n_clusters)

        print(f"scMAE: Clustering complete. Found {len(np.unique(self._labels))} clusters.")

        # Store number of training cells for predict()
        self._train_n_cells = len(self._labels)

        # Convert existing train_losses to standard loss_history format
        if hasattr(self, 'train_losses') and self.train_losses:
            self._loss_history = [{
                'name': 'training',
                'epochs': list(range(len(self.train_losses))),
                'train_loss': list(self.train_losses),
                'components': {
                    'reconstruction': list(self.train_recon_losses),
                    'mask_bce': list(self.train_mask_losses),
                },
            }]

        self._fitted = True
        gc.collect()
        return self

    def _kmeans_clustering(self, embeddings: np.ndarray, n_clusters: int,
                          fit: bool = True) -> np.ndarray:
        """
        Apply K-Means clustering on embeddings.
        Exactly as in original main.py (line 105-107).
        """
        from sklearn.cluster import KMeans

        if fit:
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=self.params.get('random_state', 42)
            )
            return self.clusterer.fit_predict(embeddings)
        else:
            if self.clusterer is None:
                raise RuntimeError("Clusterer not fitted yet")
            return self.clusterer.predict(embeddings)

    def _leiden_clustering_with_search(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Apply Leiden clustering with resolution search.
        Exactly as in original main.py (lines 108-115).
        Uses res_search_fixed_clus to find optimal resolution.
        """
        import scanpy as sc
        import anndata as ad

        adata = ad.AnnData(embeddings)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')

        # Search for resolution that gives target cluster count
        reso = res_search_fixed_clus(adata, n_clusters)
        sc.tl.leiden(adata, resolution=reso)

        pred = adata.obs['leiden'].to_list()
        pred_label = [int(x) for x in pred]

        return np.array(pred_label)

    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels.

        In benchmark mode (different number of cells than training):
        - Retrieves pre-HVG counts from test data
        - Filters to the same HVG genes selected during training
        - Applies the same preprocessing (normalize, log, scale with training stats)
        - Encodes with the trained model
        - Clusters with the trained KMeans model
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # If no data provided, return stored labels
        if data is None:
            return self._labels

        # Get number of cells in input data
        if hasattr(data, 'X'):
            n_cells_input = data.n_obs if hasattr(data, 'n_obs') else data.X.shape[0]
        else:
            n_cells_input = data.shape[0]

        # If same number of cells as training data, assume it's the same data
        if n_cells_input == self._train_n_cells:
            return self._labels

        # =====================================================================
        # BENCHMARK MODE: Process new data (test set)
        # =====================================================================
        print(f"scMAE: predict() on new data ({n_cells_input} cells, training had {self._train_n_cells})")

        use_own_preprocessing = self.params.get('use_own_preprocessing', True)

        if use_own_preprocessing and self._selected_hvg_genes is not None:
            # Get pre-HVG counts from test data
            X_test = self._get_preprocessed_test_data(data)
            if X_test is None:
                logger.warning("scMAE: Could not preprocess test data. Returning empty labels.")
                return np.zeros(n_cells_input, dtype=int)
        else:
            # use_own_preprocessing=False: use benchmark data directly
            if hasattr(data, 'X'):
                X_test = data.X
                if hasattr(X_test, 'toarray'):
                    X_test = X_test.toarray()
            else:
                X_test = data
            X_test = X_test.astype(np.float32)

            # Check dimensions
            expected_genes = self.model.encoder[1].in_features
            if X_test.shape[1] != expected_genes:
                logger.warning(f"scMAE: Gene dimension mismatch ({X_test.shape[1]} vs {expected_genes})")
                return np.zeros(n_cells_input, dtype=int)

        # Encode with trained model
        import torch
        device = next(self.model.parameters()).device

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(device)
            embeddings = self.model.feature(X_tensor).cpu().numpy()

        # Cluster with trained KMeans (don't refit)
        if self.clusterer is not None:
            return self.clusterer.predict(embeddings)
        else:
            # Fallback: use KMeans with same n_clusters
            n_clusters = len(np.unique(self._labels))
            return self._kmeans_clustering(embeddings, n_clusters, fit=True)

    def _get_preprocessed_test_data(self, data: Any) -> Optional[np.ndarray]:
        """
        Apply scMAE preprocessing to test data using training parameters.

        Steps:
        1. Get pre-HVG counts from test data
        2. Filter to same HVG genes as training
        3. Normalize, log, scale using training statistics
        4. Apply SVD imputation
        5. Sort genes by mean (same order as training)
        """
        import scanpy as sc

        # Get pre-HVG counts
        if hasattr(data, 'uns') and 'pre_hvg_counts' in data.uns:
            X_counts = data.uns['pre_hvg_counts']
            gene_names = data.uns.get('pre_hvg_var_names', None)
            print(f"scMAE: Using pre-HVG counts for test data ({X_counts.shape[1]} genes)")
        elif hasattr(data, 'X'):
            X_counts = data.X
            gene_names = list(data.var_names) if hasattr(data, 'var_names') else None
            print(f"scMAE: Using data.X for test data ({X_counts.shape[1]} genes)")
        else:
            logger.warning("scMAE: No count data found in test data")
            return None

        if hasattr(X_counts, 'toarray'):
            X_counts = X_counts.toarray()
        X_counts = X_counts.astype(np.float32)

        # Normalize per cell using TRAINING normalization target for consistency
        # (Critical: using test median would shift the scale for cross-batch data)
        row_sums = np.sum(X_counts, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        if self._train_norm_target is not None:
            norm_target = self._train_norm_target
            print(f"scMAE: Using training normalization target: {norm_target:.1f} (test median: {float(np.median(row_sums)):.1f})")
        else:
            norm_target = float(np.median(row_sums))
            print(f"scMAE: WARNING - No training norm target stored, using test median: {norm_target:.1f}")
        X_norm = X_counts / row_sums * norm_target

        # Log transform
        X_log = np.log1p(X_norm)

        # Filter to same HVG genes as training (moved AFTER normalization to preserve library size)
        if gene_names is not None and self._selected_hvg_genes is not None:
            gene_mask = np.isin(gene_names, self._selected_hvg_genes)
            if np.sum(gene_mask) < len(self._selected_hvg_genes):
                logger.warning(f"scMAE: Only {np.sum(gene_mask)}/{len(self._selected_hvg_genes)} HVG genes found in test data")

            # Reorder to match training gene order
            gene_to_idx = {g: i for i, g in enumerate(gene_names) if g in self._selected_hvg_genes}
            ordered_indices = [gene_to_idx[g] for g in self._selected_hvg_genes if g in gene_to_idx]
            
            # Apply filtering to X_log
            X_log = X_log[:, ordered_indices]
            print(f"scMAE: Filtered test data to {X_log.shape[1]} HVG genes (after normalization)")

        # Scale using training mean/std
        if self._scale_mean is not None and self._scale_std is not None:
            if X_log.shape[1] == len(self._scale_mean):
                X_scaled = (X_log - self._scale_mean) / self._scale_std
                X_scaled = np.clip(X_scaled, -10, 10)
            else:
                logger.warning(f"scMAE: Scaling dimension mismatch, skipping scaling")
                X_scaled = X_log
        else:
            X_scaled = X_log

        # Apply same gene permutation as training (CRITICAL FIX)
        if hasattr(self, '_gene_permutation') and self._gene_permutation is not None:
             print(f"scMAE: Applying stored gene permutation to test data ({len(self._gene_permutation)} features)")
             if X_scaled.shape[1] == len(self._gene_permutation):
                 X_scaled = X_scaled[:, self._gene_permutation]
             else:
                 logger.warning(f"scMAE: Permutation dimension mismatch ({X_scaled.shape[1]} vs {len(self._gene_permutation)})")

        # SVD imputation
        if self.svd_imputator is not None:
            try:
                X_final = self.svd_imputator.transform(X_scaled)
            except Exception as e:
                logger.warning(f"scMAE: SVD transform failed: {e}. Using scaled data.")
                X_final = X_scaled
        else:
            X_final = X_scaled

        return X_final.astype(np.float32)

    def encode(self, data: Any) -> np.ndarray:
        """
        Encode data into latent space.

        Args:
            data: AnnData or numpy array

        Returns:
            Latent embeddings
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before encoding")

        # Get number of cells in input data
        if hasattr(data, 'X'):
            n_cells_input = data.n_obs if hasattr(data, 'n_obs') else data.X.shape[0]
        else:
            n_cells_input = data.shape[0]

        # If same number of cells as training, return stored embeddings
        if n_cells_input == self._train_n_cells and self._embeddings is not None:
            return self._embeddings

        # For new data, preprocess and encode
        use_own_preprocessing = self.params.get('use_own_preprocessing', True)

        if use_own_preprocessing and self._selected_hvg_genes is not None:
            # Use the same preprocessing as predict()
            X = self._get_preprocessed_test_data(data)
            if X is None:
                logger.warning("scMAE encode(): Could not preprocess data. Returning zeros.")
                return np.zeros((n_cells_input, self.model.latent_dim if self.model else 128))
        else:
            # use_own_preprocessing=False: use data directly
            if hasattr(data, 'X'):
                X = data.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
            else:
                X = data
            X = X.astype(np.float32)

            # Check dimensions
            expected_genes = self.model.num_genes
            if X.shape[1] != expected_genes:
                logger.warning(f"scMAE encode(): Gene dimension mismatch ({X.shape[1]} vs {expected_genes}). Returning training embeddings.")
                return self._embeddings if self._embeddings is not None else np.zeros((n_cells_input, 128))

        import torch
        device = next(self.model.parameters()).device

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            embeddings = self.model.feature(X_tensor).cpu().numpy()

        return embeddings
