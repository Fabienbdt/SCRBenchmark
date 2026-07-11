"""
scDeepCluster algorithm implementation.
Based on https://github.com/ttgump/scDeepCluster_pytorch
Deep learning-based clustering with ZINB loss for scRNA-seq data.
"""

import math
import os
from typing import Any, Dict, List, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Layers from original scDeepCluster implementation
# =============================================================================

class ZINBLoss(nn.Module):
    """Zero-Inflated Negative Binomial loss for scRNA-seq count data."""

    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class MeanAct(nn.Module):
    """Mean activation with exponential and clamping."""

    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """Dispersion activation with softplus and clamping."""

    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


# =============================================================================
# Network builder utility
# =============================================================================

def buildNetwork(layers, type, activation="relu"):
    """Build a sequential network with specified layers and activation."""
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "elu":
            net.append(nn.ELU())
        elif activation == "leaky_relu":
            net.append(nn.LeakyReLU())
    return nn.Sequential(*net)


# =============================================================================
# scDeepCluster Model (from original implementation)
# =============================================================================

class scDeepCluster(nn.Module):
    """
    scDeepCluster: Deep Embedding Clustering for scRNA-seq data.

    Uses ZINB loss for reconstruction and KL divergence for soft clustering
    with Student's t-distribution.

    Reference:
        Tian et al. "Clustering single-cell RNA-seq data with a model-based
        deep learning approach" Nature Machine Intelligence, 2019
    """

    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[],
                 activation="relu", sigma=1., alpha=1., gamma=1., device="cuda",
                 force_float32=False):
        super(scDeepCluster, self).__init__()
        # Match original implementation: use float64 for CPU/CUDA, float32 for MPS
        # MPS (Apple Silicon) does not support float64
        # force_float32=True enables cross-platform reproducibility by using float32 everywhere
        # Note: We use explicit dtype in tensor operations instead of torch.set_default_dtype()
        # to avoid affecting other algorithms in the same session
        if device == "mps" or force_float32:
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        
        # Encoder network
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        # Decoder network
        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)

        # Latent space projection
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        # ZINB output heads
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        # ZINB loss - convert to correct dtype and device
        self.zinb_loss = ZINBLoss().to(dtype=self.dtype, device=self.device)

        # Loss history tracking
        self.pretrain_loss_history = []
        self.cluster_loss_history = []
        self.cluster_recon_history = []
        self.cluster_kl_history = []

        # Convert model to the correct dtype and move to device
        self.to(dtype=self.dtype, device=device)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        return z0, _mean, _disp, _pi

    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        """Encode data in batches."""
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _ = self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)

    def cluster_loss(self, p, q):
        """Compute KL divergence clustering loss."""
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma * kldloss

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400, ae_save=True, ae_weights='AE_weights.pth.tar'):
        self.train()
        # Use appropriate dtype (float32 for MPS, float64 otherwise)
        dataset = TensorDataset(
            torch.tensor(X, dtype=self.dtype),
            torch.tensor(X_raw, dtype=self.dtype),
            torch.tensor(size_factor, dtype=self.dtype)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)

                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor,
                                     pi=pi_tensor, scale_factor=sf_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            self.pretrain_loss_history.append(loss_val / X.shape[0])
            print('Pretrain epoch %3d, ZINB loss: %.8f' % (epoch+1, loss_val/X.shape[0]))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None,
            y_pred_init=None, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3):
        """
        Fit the clustering model.

        Args:
            X: Normalized expression matrix
            X_raw: Raw count matrix
            size_factor: Size factors for each cell
            n_clusters: Number of clusters
            init_centroid: Initial cluster centroids (optional)
            y: Ground truth labels (optional, for evaluation)
            y_pred_init: Initial predictions (optional)
            lr: Learning rate
            batch_size: Batch size
            num_epochs: Maximum number of epochs
            update_interval: Interval for updating target distribution
            tol: Convergence tolerance

        Returns:
            y_pred: Predicted cluster labels
            latent: Latent embeddings
        """
        self.train()
        print("Clustering stage")
        # Use appropriate dtype (float32 for MPS, float64 otherwise)
        X = torch.tensor(X, dtype=self.dtype)
        X_raw = torch.tensor(X_raw, dtype=self.dtype)
        size_factor = torch.tensor(size_factor, dtype=self.dtype)

        # Initialize cluster centers
        self.mu = Parameter(torch.zeros(n_clusters, self.z_dim, dtype=self.dtype, device=self.device))

        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        # K-means initialization
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

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                # Check convergence
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred

                if epoch > 0 and delta_label < tol:
                    print(f'Clustering converged at epoch {epoch}.')
                    break

            if epoch % 10 == 0:
                 # Calculate loss for monitoring (approximation)
                 print(f"Clustering epoch {epoch+1}/{num_epochs} (Update interval: {update_interval})")

            # Training step
            epoch_loss_accum = 0
            epoch_recon_accum = 0
            epoch_kl_accum = 0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                sfbatch = size_factor[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]

                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                # NOTE: gamma is applied twice (in cluster_loss() and here) matching
                # the original scDeepCluster implementation. Effective weight = gamma².
                # See: external/original_code/scDeepCluster_pytorch/scDeepCluster.py lines 112 & 231
                effective_cluster = cluster_loss * self.gamma
                loss = effective_cluster + recon_loss
                loss.backward()
                optimizer.step()
                batch_size_eff = len(xbatch)
                epoch_loss_accum += loss.item() * batch_size_eff
                epoch_recon_accum += recon_loss.item() * batch_size_eff
                epoch_kl_accum += effective_cluster.item() * batch_size_eff
            self.cluster_loss_history.append(epoch_loss_accum / num)
            self.cluster_recon_history.append(epoch_recon_accum / num)
            self.cluster_kl_history.append(epoch_kl_accum / num)

        # Get final embeddings
        self.eval()
        with torch.no_grad():
            latent = self.encodeBatch(X.to(self.device))

        return self.y_pred, latent.cpu().numpy()


# =============================================================================
# Preprocessing helpers from original scDeepCluster implementation
# =============================================================================

def geneSelection(data, threshold=0, atleast=10, yoffset=.02, xoffset=5,
                  decay=1.5, n=None, verbose=1):
    """Select informative genes using the original zero-rate heuristic."""
    from scipy import sparse

    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data > threshold).mean(axis=0)))
        A = data.multiply(data > threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:, detected].mean(axis=0))) / (1 - zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data > threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:, detected] > threshold
        logs = np.zeros_like(data[:, detected]) * np.nan
        logs[mask] = np.log2(data[:, detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data > threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan

    if n is not None:
        up = 10
        low = 0
        for _ in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            if np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low) / 2
            else:
                low = xoffset
                xoffset = (xoffset + up) / 2
        if verbose > 0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay * (meanExpr[nonan] - xoffset)) + yoffset

    return selected


def read_dataset(adata, transpose=False, test_split=False, copy=False):
    import scanpy as sc
    import scipy as sp
    import pandas as pd
    from sklearn.model_selection import train_test_split

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6:
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose:
        adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))
    return adata


def normalize(adata, filter_min_counts=False, size_factors=True, normalize_input=True, logtrans_input=True):
    import scanpy as sc

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        # Handle both old and new scanpy n_counts location
        n_counts = adata.obs.n_counts if 'n_counts' in adata.obs.columns else adata.obs['n_counts']
        adata.obs['size_factors'] = n_counts / np.median(n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


# =============================================================================
# Algorithm wrapper for SCRBenchmark
# =============================================================================

@AlgorithmRegistry.register
class ScDeepClusterAlgorithm(BaseAlgorithm):
    """
    scDeepCluster: Deep Embedding Clustering for scRNA-seq data.
    Uses ZINB loss for reconstruction and KL divergence for clustering.
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name='scdeepcluster',
            display_name='scDeepCluster',
            description='Deep learning autoencoder with ZINB loss and soft clustering '
                       'using Student\'s t-distribution. Specifically designed for '
                       'scRNA-seq count data with dropout.',
            category='deep_learning',
            requires_gpu=False,
            supports_labels=True,
            preprocessing_notes='Expects raw counts of selected Highly Variable Genes (HVGs). '
                               'Internal preprocessing (size factors, log, scale) is applied, '
                               'but NO internal gene selection is performed by default.',
            has_internal_preprocessing=True,
            recommended_data='raw'
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        return [
            # Network architecture
            HyperparameterConfig(
                name='z_dim',
                display_name='Latent Dimension',
                param_type=ParamType.INTEGER,
                default=32,
                description='Dimension of the latent space',
                min_value=8,
                max_value=256,
                step=8,
                category='Network',
                tuning_guide='**Impact**: Controls the information bottleneck capacity.\n**Details**: Higher values (64+) capture fine-grained details but risk overfitting and encoding noise. Lower values (16-32) force generalized feature learning but may miss subtle cell subtypes.\n**Recommendation**: Start with 32. Increase to 64 for complex datasets (>10k cells).'
            ),
            HyperparameterConfig(
                name='encoder_layers',
                display_name='Encoder Layers',
                param_type=ParamType.STRING,
                default='256,64',
                description='Encoder hidden layer sizes (comma-separated)',
                category='Network',
                advanced=True
            ),
            HyperparameterConfig(
                name='decoder_layers',
                display_name='Decoder Layers',
                param_type=ParamType.STRING,
                default='64,256',
                description='Decoder hidden layer sizes (comma-separated)',
                category='Network',
                advanced=True
            ),
            HyperparameterConfig(
                name='activation',
                display_name='Activation Function',
                param_type=ParamType.CHOICE,
                default='relu',
                description='Activation function for hidden layers',
                choices=['relu', 'sigmoid', 'elu', 'leaky_relu'],
                category='Network'
            ),
            # Training
            HyperparameterConfig(
                name='batch_size',
                display_name='Batch Size',
                param_type=ParamType.INTEGER,
                default=256,
                description='Training batch size',
                min_value=16,
                max_value=1024,
                step=16,
                category='Training',
                tuning_guide='**Impact**: Affects gradient estimation stability and memory usage.\n**Details**: Larger batches (256+) provide stable gradients for ZINB loss. Smaller batches introduce noise which can help generalization but increase training time.\n**Recommendation**: 256 is standard. Reduce to 64 only if memory is limited.'
            ),
            HyperparameterConfig(
                name='pretrain_epochs',
                display_name='Pretraining Epochs',
                param_type=ParamType.INTEGER,
                default=300,
                description='Number of autoencoder pretraining epochs',
                max_value=1000,
                step=10,
                category='Training',
                tuning_guide='**Impact**: Critical for initialization.\n**Details**: The clustering phase relies heavily on a well-initialized latent space. Insufficient pretraining leads to poor cluster convergence.\n**Recommendation**: Keep >= 400. Do not reduce unless for debugging.'
            ),
            HyperparameterConfig(
                name='maxiter',
                display_name='Max Iterations',
                param_type=ParamType.INTEGER,
                default=2000,
                description='Maximum clustering iterations',
                min_value=100,
                max_value=10000,
                step=100,
                category='Training',
                tuning_guide='**Impact**: Duration of clustering refinement.\n**Details**: The algorithm stops early if the convergence tolerance is met. This limit prevents infinite loops.\n**Recommendation**: 2000 is usually sufficient.'
            ),
            HyperparameterConfig(
                name='pretrain_lr',
                display_name='Pretraining Learning Rate',
                param_type=ParamType.FLOAT,
                default=0.001,
                description='Learning rate for pretraining (Adam)',
                min_value=0.0001,
                # max_value=0.1,  # Removed to force st.number_input
                step=0.0001,
                category='Training',
                tuning_guide='**Impact**: Speed of convergence during pretraining.\n**Details**: Too high (>0.01) causes divergence (NaN loss). Too low (<0.0001) results in slow learning.\n**Recommendation**: 0.001 is optimal for Adam optimizer in this context.'
            ),
            HyperparameterConfig(
                name='lr',
                display_name='Clustering Learning Rate',
                param_type=ParamType.FLOAT,
                default=1.0,
                description='Learning rate for clustering phase (Adadelta)',
                min_value=0.01,
                max_value=10.0,
                step=0.1,
                category='Training',
                tuning_guide='Adadelta requires high LR (1.0). Do not lower unless divergence occurs.'
            ),
            HyperparameterConfig(
                name='tol',
                display_name='Convergence Tolerance',
                param_type=ParamType.FLOAT,
                default=0.001,
                description='Convergence tolerance (label change percentage)',
                min_value=0.0001,
                max_value=0.1,
                step=0.0001,
                category='Training',
                advanced=True
            ),
            # Model parameters
            HyperparameterConfig(
                name='sigma',
                display_name='Noise Sigma',
                param_type=ParamType.FLOAT,
                default=2.5,
                description='Standard deviation for Gaussian noise (denoising)',
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                category='Model'
            ),
            HyperparameterConfig(
                name='gamma',
                display_name='Clustering Loss Weight',
                param_type=ParamType.FLOAT,
                default=1.0,
                description='Weight for clustering loss',
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                category='Model',
                advanced=True
            ),
            HyperparameterConfig(
                name='alpha',
                display_name='Student-t DOF',
                param_type=ParamType.FLOAT,
                default=1.0,
                description="Degrees of freedom for Student's t-distribution",
                min_value=0.1,
                max_value=10.0,
                step=0.1,
                category='Model',
                advanced=True
            ),
            # Clustering
            HyperparameterConfig(
                name='n_clusters',
                display_name='Number of Clusters',
                param_type=ParamType.INTEGER,
                default=0,
                description='Number of clusters (0 = auto-estimate with Leiden on AE latent space)',
                min_value=0,
                max_value=100,
                step=1,
                category='Clustering'
            ),
            HyperparameterConfig(
                name='use_ground_truth_k',
                display_name='Use Ground Truth K (Oracle)',
                param_type=ParamType.BOOLEAN,
                default=False,
                description='⚠️ ORACLE MODE: Use number of clusters from ground truth labels. '
                           'This leaks test information and should NOT be used for fair benchmarking. '
                           'Only use for debugging or when comparing with published results that used this approach.',
                category='Clustering',
                tuning_guide='**WARNING**: This option uses information from ground truth labels to determine k. '
                            'Results are NOT comparable to methods that estimate k independently. '
                            'For fair benchmarking, set to False and specify n_clusters manually or use auto-estimation.'
            ),
            HyperparameterConfig(
                name='knn',
                display_name='KNN (Leiden)',
                param_type=ParamType.INTEGER,
                default=20,
                description='Number of neighbors for Leiden graph when n_clusters=0',
                min_value=5,
                max_value=100,
                step=1,
                category='Clustering'
            ),
            HyperparameterConfig(
                name='resolution',
                display_name='Leiden Resolution',
                param_type=ParamType.FLOAT,
                default=0.0,
                description='Resolution parameter for Leiden when n_clusters=0 (0.0 = automatic silhouette search).',
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                category='Clustering'
            ),
            HyperparameterConfig(
                name='resolution_min',
                display_name='Leiden Res Min',
                param_type=ParamType.FLOAT,
                default=0.1,
                description='Minimum Leiden resolution explored during automatic search.',
                min_value=0.01,
                max_value=3.0,
                step=0.01,
                category='Clustering',
                advanced=True
            ),
            HyperparameterConfig(
                name='resolution_max',
                display_name='Leiden Res Max',
                param_type=ParamType.FLOAT,
                default=2.5,
                description='Maximum Leiden resolution explored during automatic search.',
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                category='Clustering',
                advanced=True
            ),
            HyperparameterConfig(
                name='resolution_step',
                display_name='Leiden Res Step',
                param_type=ParamType.FLOAT,
                default=0.1,
                description='Step size for Leiden resolution automatic search.',
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                category='Clustering',
                advanced=True
            ),
            HyperparameterConfig(
                name='resolution_silhouette_max_cells',
                display_name='Leiden Silhouette Max Cells',
                param_type=ParamType.INTEGER,
                default=5000,
                description='Maximum number of cells used for silhouette scoring in automatic resolution search.',
                min_value=500,
                max_value=50000,
                step=500,
                category='Clustering',
                advanced=True
            ),
            HyperparameterConfig(
                name='select_genes',
                display_name='Select Genes',
                param_type=ParamType.INTEGER,
                default=0,
                description='Number of genes to select before preprocessing (0 = use all genes)',
                min_value=0,
                max_value=20000,
                step=100,
                category='Preprocessing'
            ),
            HyperparameterConfig(
                name='update_interval',
                display_name='Update Interval',
                param_type=ParamType.INTEGER,
                default=1,
                description='Target distribution update interval',
                min_value=1,
                max_value=10,
                step=1,
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
                default=True,
                description='Use raw counts with scDeepCluster preprocessing (recommended).',
                category='Data'
            ),
            HyperparameterConfig(
                name='force_float32',
                display_name='Force Float32 (Cross-platform)',
                param_type=ParamType.BOOLEAN,
                default=False,
                description='Force float32 precision on all platforms for cross-platform reproducibility. '
                           'By default, CPU/CUDA use float64 for accuracy while MPS uses float32. '
                           'Enable this for consistent results across different hardware.',
                category='General',
                advanced=True,
                tuning_guide='Enable this if you need to compare results between different platforms '
                            '(e.g., training on GPU server and inference on Mac). '
                            'May slightly reduce numerical precision on CPU/CUDA.'
            ),
        ]

    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.model = None
        self._valid_indices = None
        self.scaler = None  # StandardScaler for consistent scaling
        self._n_features = None  # Number of features at training time

    def get_valid_indices(self) -> Optional[np.ndarray]:
        """Return indices of samples used in the final model."""
        return self._valid_indices

    def get_num_parameters(self) -> Optional[int]:
        """Get the number of trainable parameters."""
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _search_optimal_leiden_resolution(self, adata_latent, latent_embeddings: np.ndarray) -> float:
        """Search Leiden resolution maximizing silhouette score on AE latent embeddings."""
        import scanpy as sc

        seed = int(self.params.get('random_state', 42))
        res_min = float(self.params.get('resolution_min', 0.1))
        res_max = float(self.params.get('resolution_max', 2.5))
        res_step = float(self.params.get('resolution_step', 0.1))
        max_cells = int(self.params.get('resolution_silhouette_max_cells', 5000))

        if res_step <= 0:
            logger.warning("scDeepCluster: invalid resolution_step=%s. Falling back to 0.1.", res_step)
            res_step = 0.1
        if res_max < res_min:
            res_min, res_max = res_max, res_min

        resolutions = np.arange(res_min, res_max + 1e-12, res_step)
        if len(resolutions) == 0:
            logger.warning("scDeepCluster: empty resolution grid. Falling back to 0.8.")
            return 0.8

        X = np.asarray(latent_embeddings)
        if X.ndim != 2 or X.shape[0] < 3:
            logger.warning("scDeepCluster: not enough samples for silhouette search. Falling back to 0.8.")
            return 0.8

        eval_idx = np.arange(X.shape[0])
        if max_cells > 0 and X.shape[0] > max_cells:
            rng = np.random.default_rng(seed)
            eval_idx = np.sort(rng.choice(X.shape[0], size=max_cells, replace=False))

        best_res = None
        best_score = -np.inf
        best_clusters = 0

        for res in resolutions:
            sc.tl.leiden(
                adata_latent,
                random_state=seed,
                resolution=float(res),
                key_added="_leiden_search",
            )
            labels_full = adata_latent.obs["_leiden_search"].astype(int).to_numpy()
            labels_eval = labels_full[eval_idx]
            if np.unique(labels_eval).size < 2:
                continue
            try:
                score = float(metrics.silhouette_score(X[eval_idx], labels_eval, metric='euclidean'))
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_res = float(res)
                best_clusters = int(np.unique(labels_full).size)

        if "_leiden_search" in adata_latent.obs:
            del adata_latent.obs["_leiden_search"]

        if best_res is None:
            logger.warning("scDeepCluster: no valid silhouette score found. Falling back to 0.8.")
            return 0.8

        logger.info(
            "scDeepCluster: selected Leiden resolution %.3f (silhouette=%.4f, clusters=%d).",
            best_res,
            best_score,
            best_clusters,
        )
        return best_res

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'ScDeepClusterAlgorithm':
        """
        Fit the scDeepCluster model.

        Args:
            data: AnnData object with preprocessed data
            labels: Optional ground truth labels
        """
        # Set random seed
        seed = self.params.get('random_state', 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Get device (respects user preference from UI)
        device = torch.device(self.get_device())

        # Determine numpy dtype based on device and force_float32 option
        # MPS only supports float32; CPU/CUDA default to float64 for accuracy
        # force_float32=True ensures cross-platform reproducibility
        force_float32 = self.params.get('force_float32', False)
        if device.type == "mps" or force_float32:
            np_dtype = np.float32
            if force_float32 and device.type != "mps":
                logger.info("force_float32 enabled: using float32 for cross-platform reproducibility")
        else:
            np_dtype = np.float64

        # Parse encoder/decoder layers
        enc_layers = [int(x) for x in self.params.get('encoder_layers', '256,64').split(',')]
        dec_layers = [int(x) for x in self.params.get('decoder_layers', '64,256').split(',')]

        # Get input data (raw counts) and apply original preprocessing
        use_raw_data = self.params.get('use_raw_data', True)
        if hasattr(data, 'X'):
            if use_raw_data and hasattr(data, 'layers') and 'original_X' in data.layers:
                X_counts = data.layers['original_X']
            elif use_raw_data and hasattr(data, 'raw') and data.raw is not None:
                X_counts = data.raw.X
            else:
                X_counts = data.X
            if hasattr(X_counts, 'toarray'):
                X_counts = X_counts.toarray()
        else:
            X_counts = data

        # Round to integers before dtype conversion: some datasets store raw counts
        # as floats (e.g., 1.0 instead of 1), which causes read_dataset's integer
        # assertion to fail after float64 conversion.
        if np.issubdtype(X_counts.dtype, np.floating) and np.allclose(X_counts, np.round(X_counts)):
            X_counts = np.round(X_counts)
        X_counts = X_counts.astype(np_dtype)

        # Optional gene selection (as in original CLI)
        select_genes = self.params.get('select_genes', 0)
        if select_genes is not None and select_genes > 0:
            gene_mask = geneSelection(X_counts, n=select_genes, verbose=0)
            X_counts = X_counts[:, gene_mask]

        import scanpy as sc
        # Create AnnData and ensure obs_names are the original 0-indexed strings
        # This is CRITICAL for tracking which cells were filtered out by 'normalize'
        adata = sc.AnnData(X_counts, dtype=np_dtype)
        adata.obs_names = [str(i) for i in range(adata.n_obs)]
        
        if labels is not None:
            adata.obs['Group'] = labels
        
        # Original preprocessing sequence
        adata = read_dataset(adata, transpose=False, test_split=False, copy=True)
        
        # filter_min_counts=False because we want to honor the benchmark's preprocessing
        # as requested by the user. Original implementation used True.
        # Custom preprocessing with StandardScaler capture
        if use_raw_data:
            # SAVE RAW COUNTS specifically for ZINB loss
            adata.raw = adata.copy()

            # 1. Normalize and get size factors
            sc.pp.normalize_per_cell(adata)
            # Handle both old and new scanpy n_counts location
            n_counts = adata.obs.n_counts if 'n_counts' in adata.obs.columns else adata.obs['n_counts']
            adata.obs['size_factors'] = n_counts / np.median(n_counts)
            
            # 2. Log transform
            sc.pp.log1p(adata)
            
            # 3. Scale with StandardScaler and store
            self.scaler = StandardScaler()
            # Ensure dense for scaler
            X_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            # Fit on training data
            X_scaled = self.scaler.fit_transform(X_mat)
            adata.X = X_scaled
        else:
            # If not using internal preprocessing, we assume input X is what we want for ZINB target too
            # (or user should have set adata.raw beforehand).
            # To be safe and avoid crash:
            if adata.raw is None:
                adata.raw = adata

            # No internal normalization requested

            adata.obs['size_factors'] = 1.0
            # If data is not raw, we assume it's preprocessed.
            # However, we should still fit a scaler to capture statistics for consistency,
            # or if the user passed unscaled log-transformed data.
            # Safe bet: Fit scaler on the input X.
            self.scaler = StandardScaler()
            X_mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
            X_scaled = self.scaler.fit_transform(X_mat)
            adata.X = X_scaled

        # Capture valid indices after filtering for alignment in analysis_runner
        # We use the obs_names which we set to original integer strings
        self._valid_indices = adata.obs_names.values.astype(int)
        
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = X.astype(np_dtype)

        X_raw = adata.raw.X
        if hasattr(X_raw, 'toarray'):
            X_raw = X_raw.toarray()
        X_raw = X_raw.astype(np_dtype)

        if labels is not None:
            # Important: Get the filtered labels from adata.obs
            labels = adata.obs['Group'].values

        size_factor = np.array(adata.obs['size_factors']).astype(np_dtype)
        input_dim = X.shape[1]

        # Store number of features for validation in predict/encode
        self._n_features = input_dim

        # Determine number of clusters (0 = Louvain estimate on AE latent)
        n_clusters = self.params.get('n_clusters', 0)

        # Create model (pass device type and force_float32 for proper dtype detection)
        self.model = scDeepCluster(
            input_dim=input_dim,
            z_dim=self.params.get('z_dim', 32),
            encodeLayer=enc_layers,
            decodeLayer=dec_layers,
            activation=self.params.get('activation', 'relu'),
            sigma=self.params.get('sigma', 1.0),
            alpha=self.params.get('alpha', 1.0),
            gamma=self.params.get('gamma', 1.0),
            device=device.type,
            force_float32=force_float32
        )

        # Pretrain autoencoder
        self.model.pretrain_autoencoder(
            X, X_raw, size_factor,
            batch_size=self.params.get('batch_size', 256),
            lr=self.params.get('pretrain_lr', 0.001),
            epochs=self.params.get('pretrain_epochs', 400)
        )

        # Determine number of clusters
        n_clusters = self.params.get('n_clusters', 0)
        # Oracle mode must be explicitly enabled; the fair default is unsupervised.
        use_ground_truth_k = self.params.get('use_ground_truth_k', False)

        # Oracle Mode: ONLY if explicitly requested via use_ground_truth_k
        if use_ground_truth_k and labels is not None:
            n_clusters = len(np.unique(labels))
            logger.warning(f"⚠️ ORACLE MODE ENABLED: scDeepCluster using ground truth k={n_clusters}. "
                           f"Results are NOT comparable to methods that estimate k independently.")

            self._labels, self._embeddings = self.model.fit(
                X, X_raw, size_factor,
                n_clusters=n_clusters,
                y=labels,
                batch_size=self.params.get('batch_size', 256),
                num_epochs=self.params.get('maxiter', 2000),
                lr=self.params.get('lr', 1.0),
                update_interval=self.params.get('update_interval', 1),
                tol=self.params.get('tol', 0.001)
            )
        elif use_ground_truth_k and labels is None:
            raise ValueError("use_ground_truth_k=True but no labels provided. "
                           "Either provide labels or set use_ground_truth_k=False.")
        elif n_clusters == 0:
            import pandas as pd

            pretrain_latent = self.model.encodeBatch(
                torch.tensor(X, dtype=self.model.dtype)
            ).cpu().numpy()
            adata_latent = sc.AnnData(pretrain_latent)
            sc.pp.neighbors(
                adata_latent,
                n_neighbors=self.params.get('knn', 20),
                use_rep="X"
            )
            resolution_param = float(self.params.get('resolution', 0.0))
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
            y_pred_init = adata_latent.obs['leiden'].astype(int).values
            self.params['leiden_resolution_selected'] = float(resolution_used)
            self.params['leiden_resolution_selection_mode'] = selection_mode

            features = pd.DataFrame(adata_latent.X, index=np.arange(adata_latent.n_obs))
            group = pd.Series(y_pred_init, index=np.arange(adata_latent.n_obs), name="Group")
            cluster_centers = pd.concat([features, group], axis=1).groupby("Group").mean().to_numpy()

            n_clusters = cluster_centers.shape[0]
            print('Estimated number of clusters: ', n_clusters)

            self._labels, self._embeddings = self.model.fit(
                X, X_raw, size_factor,
                n_clusters=n_clusters,
                init_centroid=cluster_centers,
                y_pred_init=y_pred_init,
                y=labels,
                batch_size=self.params.get('batch_size', 256),
                num_epochs=self.params.get('maxiter', 2000),
                lr=self.params.get('lr', 1.0),
                update_interval=self.params.get('update_interval', 1),
                tol=self.params.get('tol', 0.001)
            )
        else:
            self._labels, self._embeddings = self.model.fit(
                X, X_raw, size_factor,
                n_clusters=n_clusters,
                y=labels,
                batch_size=self.params.get('batch_size', 256),
                num_epochs=self.params.get('maxiter', 2000),
                lr=self.params.get('lr', 1.0),
                update_interval=self.params.get('update_interval', 1),
                tol=self.params.get('tol', 0.001)
            )

        # Extract loss history from model
        self._loss_history = []
        if hasattr(self.model, 'pretrain_loss_history') and self.model.pretrain_loss_history:
            self._loss_history.append({
                'name': 'pretrain (ZINB)',
                'epochs': list(range(len(self.model.pretrain_loss_history))),
                'train_loss': list(self.model.pretrain_loss_history),
                'components': {'zinb': list(self.model.pretrain_loss_history)},
            })
        if hasattr(self.model, 'cluster_loss_history') and self.model.cluster_loss_history:
            n_pretrain = len(self.model.pretrain_loss_history) if hasattr(self.model, 'pretrain_loss_history') else 0
            cluster_components = {'kl+zinb': list(self.model.cluster_loss_history)}
            if hasattr(self.model, 'cluster_recon_history') and self.model.cluster_recon_history:
                cluster_components['zinb'] = list(self.model.cluster_recon_history)
                cluster_components['effective_kl'] = list(getattr(self.model, 'cluster_kl_history', []))
            self._loss_history.append({
                'name': 'clustering (KL+ZINB)',
                'epochs': list(range(n_pretrain, n_pretrain + len(self.model.cluster_loss_history))),
                'train_loss': list(self.model.cluster_loss_history),
                'components': cluster_components,
            })

        self._fitted = True
        return self

    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels.

        If data is provided, encodes new data and assigns to clusters
        using the learned cluster centroids (transfer learning mode).
        If data is None, returns the labels from training.

        Args:
            data: Optional new data to predict on

        Returns:
            Predicted cluster labels
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if data is None:
            return self._labels

        # Transfer learning mode: encode new data and assign to clusters
        import torch

        # Get device from model
        device = self.model.device

        # Determine numpy dtype based on device (MPS only supports float32)
        if device == "mps":
            np_dtype = np.float32
        else:
            np_dtype = np.float64

        # Extract data matrix
        if hasattr(data, 'X'):
            X_in = data.X
        else:
            X_in = data
            
        if hasattr(X_in, 'toarray'):
            X_in = X_in.toarray()
            
        # Determine if we should treat input as raw
        # If use_raw_data is True, we check for raw layers first
        is_raw_source = False
        if self.params.get('use_raw_data', False):
            if hasattr(data, 'layers') and 'original_X' in data.layers:
                X_in = data.layers['original_X']
                if hasattr(X_in, 'toarray'): X_in = X_in.toarray()
                is_raw_source = True
            elif hasattr(data, 'raw') and data.raw is not None:
                X_in = data.raw.X
                if hasattr(X_in, 'toarray'): X_in = X_in.toarray()
                is_raw_source = True
            elif np.max(X_in) > 20 and np.allclose(X_in, np.round(X_in)):
                # Heuristic: high values AND approximately integer = likely raw counts
                is_raw_source = True

        # Apply preprocessing if raw
        if is_raw_source:
             import scanpy as sc
             import anndata as ad
             # Create temp AnnData
             adata_tmp = ad.AnnData(X=X_in)
             # Normalize -> Log
             sc.pp.normalize_per_cell(adata_tmp)
             sc.pp.log1p(adata_tmp)
             X_in = adata_tmp.X
             if hasattr(X_in, 'toarray'):
                 X_in = X_in.toarray()

        # Validate feature dimensions
        if self._n_features is not None and X_in.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: model was trained with {self._n_features} features, "
                f"but input has {X_in.shape[1]} features. Ensure the same genes/HVGs are used."
            )

        # Apply Scaling (if scaler exists)
        if self.scaler:
             X = self.scaler.transform(X_in)
        else:
             X = X_in

        X = X.astype(np_dtype)
        X_tensor = torch.tensor(X, dtype=self.model.dtype)

        # Encode new data
        self.model.eval()
        with torch.no_grad():
            z = self.model.encodeBatch(X_tensor)

            # Soft assign using learned cluster centers
            q = self.model.soft_assign(z)

            # Get hard assignments
            labels = torch.argmax(q, dim=1).cpu().numpy()

        return labels

    def encode(self, data: Any) -> np.ndarray:
        """
        Encode data into latent space.

        Args:
            data: Data to encode (AnnData or numpy array)

        Returns:
            Latent embeddings
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before encoding")

        import torch

        # Get device from model
        device = self.model.device

        # Determine numpy dtype
        if device == "mps":
            np_dtype = np.float32
        else:
            np_dtype = np.float64

        # Extract data matrix
        if hasattr(data, 'X'):
            X_in = data.X
        else:
            X_in = data
            
        if hasattr(X_in, 'toarray'):
            X_in = X_in.toarray()
            
        # Determine if we should treat input as raw (Logic matches predict)
        is_raw_source = False
        if self.params.get('use_raw_data', False):
            if hasattr(data, 'layers') and 'original_X' in data.layers:
                X_in = data.layers['original_X']
                if hasattr(X_in, 'toarray'): X_in = X_in.toarray()
                is_raw_source = True
            elif hasattr(data, 'raw') and data.raw is not None:
                X_in = data.raw.X
                if hasattr(X_in, 'toarray'): X_in = X_in.toarray()
                is_raw_source = True
            elif np.max(X_in) > 20 and np.allclose(X_in, np.round(X_in)):
                # Heuristic: high values AND approximately integer = likely raw counts
                is_raw_source = True

        # Apply preprocessing if raw
        if is_raw_source:
             import scanpy as sc
             import anndata as ad
             adata_tmp = ad.AnnData(X=X_in)
             sc.pp.normalize_per_cell(adata_tmp)
             sc.pp.log1p(adata_tmp)
             X_in = adata_tmp.X
             if hasattr(X_in, 'toarray'):
                 X_in = X_in.toarray()

        # Validate feature dimensions
        if self._n_features is not None and X_in.shape[1] != self._n_features:
            raise ValueError(
                f"Feature dimension mismatch: model was trained with {self._n_features} features, "
                f"but input has {X_in.shape[1]} features. Ensure the same genes/HVGs are used."
            )

        # Apply Scaling
        if self.scaler:
             X = self.scaler.transform(X_in)
        else:
             X = X_in

        X = X.astype(np_dtype)
        X_tensor = torch.tensor(X, dtype=self.model.dtype)

        # Encode
        self.model.eval()
        with torch.no_grad():
            z = self.model.encodeBatch(X_tensor)

        return z.cpu().numpy()
