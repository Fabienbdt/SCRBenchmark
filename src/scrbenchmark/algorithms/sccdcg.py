"""
scCDCG algorithm implementation.
Based on https://github.com/XPgogogo/scCDCG

scCDCG: Single-cell RNA-seq Clustering via Deep Cut-informed Graph Embedding
Published at DASFAA 2024

This implementation is a direct port of the authors' original code.
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import LabelEncoder
from typing import Any, Dict, List, Optional
from time import time
import logging

logger = logging.getLogger(__name__)

from core.algorithm_registry import BaseAlgorithm, AlgorithmInfo, AlgorithmRegistry
from core.config import HyperparameterConfig, ParamType


# =============================================================================
# Utility functions from original scCDCG implementation
# =============================================================================

def get_laplace_matrix(tensor_matrix):
    """
    Generate normalized adjacency matrix: L = D^(-1/2) * A * D^(-1/2)
    This is the EXACT implementation from the authors.
    """
    A = np.array(tensor_matrix)
    D = A.sum(axis=1)

    L_matrix = np.diag(D**(-0.5)).dot(A.dot(np.diag(D**(-0.5))))
    L_matrix = torch.tensor(L_matrix, dtype=torch.float)
    return torch.nan_to_num(L_matrix)


def sinkhorn(pred, lambdas, row, col):
    """
    Sinkhorn-Knopp algorithm for optimal transport.
    EXACT implementation from the authors.
    """
    num_node = pred.shape[0]
    num_class = pred.shape[1]
    p = np.power(pred, lambdas)
    
    u = np.ones(num_node)
    v = np.ones(num_class)

    for index in range(1000):
        u = row * np.power(np.dot(p, v), -1)
        u[np.isinf(u)] = -9e-15
        v = col * np.power(np.dot(u, p), -1)
        v[np.isinf(v)] = -9e-15
    u = row * np.power(np.dot(p, v), -1)
    target = np.dot(np.dot(np.diag(u), p), np.diag(v))
    return target


def best_map(y_true, y_pred):
    """
    Permute labels of y_pred to match y_true as much as possible.
    EXACT implementation from the authors.
    """
    if len(y_true) != len(y_pred):
        print("y_true.shape must == y_pred.shape")
        raise SystemExit(0)

    label_set = np.unique(y_true)
    num_class = len(label_set)

    G = np.zeros((num_class, num_class))
    for i in range(0, num_class):
        for j in range(0, num_class):
            s = y_true == label_set[i]
            t = y_pred == label_set[j]
            G[i, j] = np.count_nonzero(s & t)

    A = linear_sum_assignment(-G)
    new_y_pred = np.zeros(y_pred.shape)
    for i in range(0, num_class):
        new_y_pred[y_pred == label_set[A[1][i]]] = label_set[A[0][i]]
    return new_y_pred.astype(int), label_set[A[1]], label_set[A[0]]


def evaluation(y_true, y_pred):
    """
    Compute clustering metrics.
    EXACT implementation from the authors.
    """
    y_pred_, label_original, label_truth = best_map(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred_)
    f1_macro = f1_score(y_true, y_pred_, average='macro')
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    return acc, nmi, ari, f1_macro


# =============================================================================
# Neural Network Models - EXACT copy from authors' model.py
# =============================================================================

class AE_NN(nn.Module):
    """
    Neural Network Autoencoder.
    EXACT implementation from the authors.
    """
    def __init__(self, dim_input, dims_encoder, dims_decoder):
        super(AE_NN, self).__init__()
        self.dims_en = [dim_input] + dims_encoder
        self.dims_de = dims_decoder + [dim_input]

        self.num_layer = len(self.dims_en) - 1
        
        self.Encoder = nn.ModuleList()
        self.Decoder = nn.ModuleList()
        self.leakyrelu = nn.LeakyReLU(0.2)
        
        for index in range(self.num_layer):
            self.Encoder.append(nn.Linear(self.dims_en[index], self.dims_en[index + 1]))
            self.Decoder.append(nn.Linear(self.dims_de[index], self.dims_de[index + 1]))

    def forward(self, x, adj):
        for index in range(self.num_layer):
            x = self.Encoder[index].forward(x)
        h = x
        
        for index in range(self.num_layer):
            x = self.Decoder[index].forward(x)
        x_hat = x
  
        return h, x_hat


class FULL_NN(nn.Module):
    """
    Full model for scCDCG clustering with pretrained autoencoder.
    EXACT implementation from the authors.
    """
    def __init__(self, dim_input, dims_encoder, dims_decoder, num_class, pretrain_state_dict=None):
        super(FULL_NN, self).__init__()
        self.dims_encoder = dims_encoder
        self.num_class = num_class

        self.AE = AE_NN(dim_input, dims_encoder, dims_decoder)
        
        # Load pretrained weights if provided (modified to accept state_dict directly)
        if pretrain_state_dict is not None:
            self.AE.load_state_dict(pretrain_state_dict)
  
    def forward(self, x, adj):
        h, x_hat = self.AE.forward(x, adj)
        self.z = F.normalize(h, p=2, dim=1)
        return self.z, x_hat

    def prediction(self, kappas, centers, normalize_constants, mixture_cofficences):
        cos_similarity = torch.mul(kappas, torch.mm(self.z, centers.T))
        pdf_component = torch.mul(normalize_constants, torch.exp(cos_similarity))
        p = F.normalize(torch.mul(mixture_cofficences, pdf_component), p=1, dim=1)
        return p


class ClusterAssignment(nn.Module):
    """
    Soft cluster assignment module using Student's t-distribution.
    EXACT implementation from the authors.
    """
    def __init__(self, cluster_number: int, embedding_dimension: int,
                 alpha: float = 1.0, cluster_centers: Optional[torch.Tensor] = None):
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)


# =============================================================================
# Algorithm wrapper for SCRBenchmark
# =============================================================================

@AlgorithmRegistry.register
class ScCDCGAlgorithm(BaseAlgorithm):
    """
    scCDCG: Single-cell RNA-seq Clustering via Deep Cut-informed Graph Embedding.

    This is a DIRECT PORT of the authors' original implementation.
    Reference: DASFAA 2024
    """

    @classmethod
    def get_info(cls) -> AlgorithmInfo:
        return AlgorithmInfo(
            name='sccdcg',
            display_name='scCDCG',
            description='Deep Cut-informed Graph Clustering for scRNA-seq. '
                       'Combines autoencoder feature learning with graph-based '
                       'regularization and optimal transport for clustering. '
                       'Supports inductive inference on new cells via learned encoder.',
            category='deep_learning',
            is_graph_based=True,
            requires_gpu=False,
            supports_labels=True,
            supports_out_of_sample=True,
            preprocessing_notes='Uses raw data with internal L2 normalization. '
                               'Disable normalization and log transform in preprocessing.',
            has_internal_preprocessing=True,
            recommended_data='raw',
            mps_compatible=True,
            mps_notes='Some sparse matrix operations may fall back to CPU.'
        )

    @classmethod
    def get_hyperparameters(cls) -> List[HyperparameterConfig]:
        return [
            # Network architecture
            HyperparameterConfig(
                name='embedding_dim',
                display_name='Embedding Dimension (z_dim)',
                param_type=ParamType.INTEGER,
                default=16,
                description='Dimension of the latent embedding space',
                min_value=8,
                max_value=128,
                step=8,
                category='Network',
                tuning_guide='**Impact**: Dimensionality of the learned node representations.\n**Details**: scCDCG uses graph convolutional networks. A smaller dimension (16) focuses on preserving local structural information (neighborhoods). Larger dimensions might capture more global variation usually handled by the clustering.\n**Recommendation**: 16 matches the original paper setup.'
            ),
            HyperparameterConfig(
                name='encoder_layers',
                display_name='Encoder Layers',
                param_type=ParamType.STRING,
                default='256',
                description='Encoder hidden layer sizes (comma-separated). Example: "256" or "512,256"',
                category='Network',
                tuning_guide='**Impact**: Network capacity and feature extraction.\n**Details**: Defines the hidden layers before the embedding dimension. "256" gives input->256->embedding_dim. "512,256" gives input->512->256->embedding_dim.\n**Recommendation**: "256" (default) for most datasets.'
            ),
            HyperparameterConfig(
                name='decoder_layers',
                display_name='Decoder Layers',
                param_type=ParamType.STRING,
                default='',
                description='Decoder hidden layer sizes (comma-separated). Leave empty for symmetric decoder (mirror of encoder).',
                category='Network',
                advanced=True,
                tuning_guide='**Impact**: Reconstruction quality.\n**Details**: If empty, decoder mirrors encoder (e.g., encoder "512,256" -> decoder "256,512"). Specify custom layers if asymmetric decoder needed.'
            ),
            # Training
            HyperparameterConfig(
                name='epochs',
                display_name='Epochs',
                param_type=ParamType.INTEGER,
                default=200,
                description='Number of training epochs for both pretrain and train phases',
                max_value=1000,
                step=50,
                category='Training',
                tuning_guide='**Impact**: Model convergence.\n**Details**: Deep Graph Clustering can be unstable if trained too long (over-smoothing). 200 epochs is empirically determined to peak NMI (Normalized Mutual Information).\n**Recommendation**: 200-400 is the safe range.'
            ),
            HyperparameterConfig(
                name='learning_rate',
                display_name='Learning Rate',
                param_type=ParamType.FLOAT,
                default=0.001,
                description='Learning rate for Adam optimizer (authors: 1e-3)',
                min_value=0.0001,
                max_value=0.01,
                step=0.0001,
                category='Training',
                tuning_guide='**Impact**: Optimization stability (CRITICAL).\n**Details**: Authors vary this between 1e-4 and 5e-3 depending on dataset.\n**Recommendation**: Search [1e-4, 1e-2] using LOG scale. Start with 0.001 or 0.005.'
            ),
            HyperparameterConfig(
                name='weight_decay',
                display_name='Weight Decay',
                param_type=ParamType.FLOAT,
                default=0.0005,
                description='L2 regularization weight (authors: 5e-4)',
                min_value=0.0001,
                max_value=0.01,
                step=0.0001,
                category='Training',
                advanced=False,
                tuning_guide='**Impact**: Regularization strength.\n**Recommendation**: Search [1e-4, 1e-2] on LOG scale. Authors often use 5e-4 or 1e-3.'
            ),
            # Loss weights (defaults from Meuro_human_Pancreas_cell)
            HyperparameterConfig(
                name='balancer',
                display_name='Laplacian Balancer',
                param_type=ParamType.FLOAT,
                default=0.5,
                description='Balance between two Laplacian matrices (0-1)',
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                category='Loss Weights',
                tuning_guide='**Impact**: Balances local vs global graph structure.\n**Details**: 0.0=Local (KNN), 1.0=Global (Hypergraph). Authors range: 0.1 to 0.9.\n**Recommendation**: Search [0.1, 0.9].'
            ),
            HyperparameterConfig(
                name='factor_construct',
                display_name='Reconstruction Weight',
                param_type=ParamType.FLOAT,
                default=0.259,
                description='Reconstruction loss weight (authors: 0.259)',
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                category='Loss Weights',
                tuning_guide='**Author Range**: [0.0, 1.0]. Controls feature reconstruction.'
            ),
            HyperparameterConfig(
                name='factor_ort',
                display_name='Orthogonality Weight',
                param_type=ParamType.FLOAT,
                default=0.9,
                description='Orthogonality loss weight',
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                category='Loss Weights',
                tuning_guide='**Author Range**: [0.3, 0.95]. Enforces embedding orthogonality.'
            ),
            HyperparameterConfig(
                name='factor_corvar',
                display_name='Covariance Weight',
                param_type=ParamType.FLOAT,
                default=0.16,
                description='Covariance loss weight (authors: 0.16)',
                min_value=0.01,
                max_value=0.6,
                step=0.01,
                category='Loss Weights',
                tuning_guide='**Author Range**: [0.01, 0.6]. Regularizes covariance.'
            ),
            HyperparameterConfig(
                name='factor_KL',
                display_name='KL Divergence Weight',
                param_type=ParamType.FLOAT,
                default=0.5,
                description='KL divergence loss weight',
                min_value=0.01,
                max_value=0.7,
                step=0.01,
                category='Loss Weights',
                tuning_guide='**Author Range**: [0.02, 0.65]. Controls clustering strength.'
            ),
            # Sinkhorn parameters
            HyperparameterConfig(
                name='lambdas',
                display_name='Sinkhorn Temperature',
                param_type=ParamType.FLOAT,
                default=5.0,
                description='Simulated annealing temperature for Sinkhorn algorithm',
                min_value=1.0,
                max_value=20.0,
                step=0.5,
                category='Sinkhorn',
                advanced=True,
                tuning_guide='**Impact**: Softness of cluster assignment.\n**Details**: Higher values (20) make the assignment matrix closer to uniform (doubly stochastic). Lower values make it sharper (closer to hard clustering).\n**Recommendation**: 20.0 facilitates gradient flow during training.'
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
                name='alpha_pre',
                display_name='Alpha Pre (t-dist)',
                param_type=ParamType.FLOAT,
                default=0.58,
                description='Degrees of freedom for Student kernel (authors default)',
                min_value=0.1,
                max_value=5.0,
                step=0.01,
                category='Clustering'
            ),
            HyperparameterConfig(
                name='random_state',
                display_name='Random State',
                param_type=ParamType.INTEGER,
                default=2021,
                description='Random seed for reproducibility',
                min_value=0,
                max_value=99999,
                step=1,
                category='General'
            ),
            # Data options
            HyperparameterConfig(
                name='use_raw_data',
                display_name='Use Raw Data',
                param_type=ParamType.BOOLEAN,
                default=True,
                description='Use original unprocessed data (as per authors implementation). '
                           'If True, ignores preprocessing and uses raw counts with L2 normalization.',
                category='Data'
            ),
            # Inductive inference options
            HyperparameterConfig(
                name='inference_mode',
                display_name='Inference Mode',
                param_type=ParamType.CHOICE,
                default='centroid',
                choices=['centroid', 'knn'],
                description='Method for assigning new cells to clusters. '
                           '"centroid": assign to nearest cluster center. '
                           '"knn": K-nearest neighbor vote in embedding space.',
                category='Inference',
                tuning_guide='**centroid**: Fast, assumes spherical clusters. '
                            '**knn**: More robust for non-spherical clusters but slower.'
            ),
            HyperparameterConfig(
                name='knn_k',
                display_name='KNN K (for inference)',
                param_type=ParamType.INTEGER,
                default=15,
                min_value=1,
                max_value=100,
                step=1,
                description='Number of neighbors for KNN-based inference (only used if inference_mode="knn").',
                category='Inference',
                advanced=True
            ),
        ]


    def __init__(self, params: Dict[str, Any] = None):
        super().__init__(params)
        self.pretrain_model = None
        self.model = None
        self.centers = None
        self.le = None
        # For inductive inference
        self._cluster_centers = None  # Cluster centers in embedding space
        self._train_embeddings = None  # Training embeddings for KNN
        self._train_labels = None  # Training labels for KNN
        self._knn_index = None  # KNN index for fast lookup
        self._device = None  # Store device for inference
        self._n_clusters = None  # Number of clusters

    def fit(self, data: Any, labels: Optional[Any] = None) -> 'ScCDCGAlgorithm':
        """
        Fit the scCDCG model following the EXACT original two-phase training.
        This is a direct port of the authors' train_scCDCG.py
        """
        seed_list = [self.params.get('random_state', 2021)]

        # Get device (respects user preference from UI)
        device = torch.device(self.get_device())
        if device.type == 'mps':
            torch.set_default_dtype(torch.float32)
        print(f"scCDCG: Using device: {device}")

        # Check if we should use raw data (as per authors' implementation)
        use_raw_data = self.params.get('use_raw_data', True)

        # Extract data matrix
        # Extract data matrix
        if hasattr(data, 'X'):
            # Logic to handle raw vs processed data selection
            if use_raw_data:
                if hasattr(data, 'layers') and 'original_X' in data.layers:
                    print("scCDCG: Using raw counts from layers['original_X']")
                    X = data.layers['original_X']
                elif hasattr(data, 'raw') and data.raw is not None:
                    print("scCDCG: Using raw counts from adata.raw")
                    X = data.raw.X
                else:
                    X = data.X
            else:
                X = data.X
                
            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = data

        # Determine number of clusters
        n_clusters = self.params.get('n_clusters', 0)
        if n_clusters == 0 and labels is not None:
            n_clusters = len(np.unique(labels))
            logger.warning(f"scCDCG: n_clusters=0 (Auto). Using ground truth labels to determine k={n_clusters} (Oracle Mode).")
        elif n_clusters == 0:
            # Default if no labels and not specified
            n_clusters = 8
            logger.warning(f"scCDCG: n_clusters=0 and no labels. Defaulting to k={n_clusters}.")

        X = X.astype(np.float32)
        x = torch.tensor(X, dtype=torch.float)
        
        # Process labels
        y = None
        y_np = None


        # Get parameters
        n_clusters_param = self.params.get('n_clusters', 0)
        
        if labels is not None:
            self.le = LabelEncoder()
            y_np = self.le.fit_transform(labels)
            y = torch.tensor(y_np, dtype=torch.float)
            if n_clusters_param == 0:
                n_clusters = len(np.unique(y_np))
            else:
                n_clusters = n_clusters_param
        else:
            y_np = None
            y = None
            n_clusters = n_clusters_param if n_clusters_param > 0 else 10
        embedding_dim = self.params.get('embedding_dim', 16)
        encoder_layers_str = self.params.get('encoder_layers', '256')
        decoder_layers_str = self.params.get('decoder_layers', '')
        epochs = self.params.get('epochs', 200)
        lr = self.params.get('learning_rate', 0.005)
        weight_decay = self.params.get('weight_decay', 0.005)
        balancer = self.params.get('balancer', 0.42)
        factor_construct = self.params.get('factor_construct', 0.93)
        factor_ort = self.params.get('factor_ort', 0.87)
        factor_corvar = self.params.get('factor_corvar', 0.2)
        factor_KL = self.params.get('factor_KL', 0.45)
        lambdas = self.params.get('lambdas', 5.0)
        alpha_pre = self.params.get('alpha_pre', 0.58)

        # Parse encoder layers from string
        encoder_hidden = [int(x.strip()) for x in encoder_layers_str.split(',') if x.strip()]
        if not encoder_hidden:
            encoder_hidden = [256]  # Default fallback

        # Build dims_encoder: hidden layers + embedding_dim
        dims_encoder = encoder_hidden + [embedding_dim]

        # Build dims_decoder: mirror of encoder or custom
        if decoder_layers_str.strip():
            decoder_hidden = [int(x.strip()) for x in decoder_layers_str.split(',') if x.strip()]
            dims_decoder = [embedding_dim] + decoder_hidden
        else:
            # Symmetric: mirror the encoder hidden layers
            dims_decoder = [embedding_dim] + list(reversed(encoder_hidden))

        dim_input = X.shape[1]
        print(f"scCDCG: Network architecture - Encoder: {[dim_input] + dims_encoder}, Decoder: {dims_decoder + [dim_input]}")

        # Build adjacency matrices (EXACT as authors)
        print("scCDCG: Building adjacency matrices...")
        x_ = F.normalize(x, p=2, dim=1)
        adj_self_loop = torch.mm(x_, x_.T)

        try:
            from torchmetrics.functional import pairwise_cosine_similarity
            adj_f = torch.abs(pairwise_cosine_similarity(x_, x_))
        except Exception:
            adj_f = torch.abs(torch.mm(x_, x_.T))
        adj_f = torch.mm(adj_f, adj_f.T)
        
        L_1 = get_laplace_matrix(adj_self_loop)
        L_2 = get_laplace_matrix(adj_f)

        # Move to device and ensure float32
        x = x.float().to(device)
        adj_self_loop = adj_self_loop.float().to(device)
        L_1 = L_1.float().to(device)
        L_2 = L_2.float().to(device)
        if y is not None:
            y = y.to(device)

        best_global = None

        for seed in seed_list:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # =================================================================
            # PHASE 1: PRETRAIN AUTOENCODER (EXACT as authors)
            # =================================================================
            print(f"scCDCG: Phase 1 - Pretraining for {epochs} epochs (seed {seed})...")

            Model = AE_NN(dim_input=dim_input, dims_encoder=dims_encoder,
                          dims_decoder=dims_decoder).float().to(device)
            optimizer = torch.optim.Adam(Model.parameters(), lr=lr, weight_decay=weight_decay)

            acc_max = 0
            best_state_dict = None
            best_centers = None
            best_pseudo_labels = None

            pretrain_losses = []
            for epoch in range(1, epochs + 1):
                Model.train()

                h, x_hat = Model.forward(x, adj_self_loop)
                z = F.normalize(h, p=2, dim=0)
                adj_pred = torch.mm(z, z.T)

                loss_x = F.mse_loss(x_hat, x)
                loss_corvariates = -torch.mm(torch.mm(z.T, (balancer * L_1 + (1 - balancer) * L_2)), z).trace() / len(z.T)
                loss_ort = F.mse_loss(torch.mm(z.T, z).view(-1),
                                      torch.eye(len(z.T), device=device).view(-1))
                loss = factor_construct * loss_x + factor_ort * loss_ort + factor_corvar * loss_corvariates

                pretrain_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    z_np = z.cpu().numpy()
                    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_np)

                    if y_np is not None:
                        acc, nmi, ari, f1 = evaluation(y_np, kmeans.labels_)
                        if acc > acc_max:
                            acc_max = acc
                            best_state_dict = {k: v.clone() for k, v in Model.state_dict().items()}
                            best_centers = torch.tensor(kmeans.cluster_centers_)
                            best_pseudo_labels = torch.LongTensor(kmeans.labels_)
                    else:
                        best_state_dict = {k: v.clone() for k, v in Model.state_dict().items()}
                        best_centers = torch.tensor(kmeans.cluster_centers_)
                        best_pseudo_labels = torch.LongTensor(kmeans.labels_)

            print(f"scCDCG: Best Pretrain ACC (seed {seed}): {acc_max:.4f}")

            # =================================================================
            # PHASE 2: CLUSTERING WITH SINKHORN (EXACT as authors)
            # =================================================================
            print(f"scCDCG: Phase 2 - Clustering for {epochs} epochs (seed {seed})...")

            Model = FULL_NN(dim_input=dim_input, dims_encoder=dims_encoder,
                           dims_decoder=dims_decoder, num_class=n_clusters,
                           pretrain_state_dict=best_state_dict).float().to(device)

            optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
            centers = best_centers.to(device)
            pseudo_labels = best_pseudo_labels.to(device)

            acc_max_final = 0
            nmi_max, ari_max, f1_max = 0, 0, 0
            best_labels_final = None
            best_embeddings_final = None
            p_distribution = None

            cluster_losses = []
            for epoch in range(1, epochs + 1):
                Model.train()

                z, x_hat = Model.forward(x, adj_self_loop)
                z = F.normalize(z, p=2, dim=0)
                centers = centers.detach()
                adj_pred = torch.mm(z, z.T)

                loss_x = F.mse_loss(x_hat, x)
                loss_corvariates = -torch.mm(torch.mm(z.T, (balancer * L_1 + (1 - balancer) * L_2)), z).trace() / len(z.T)
                loss_ort = F.mse_loss(torch.mm(z.T, z).view(-1),
                                      torch.eye(len(z.T), device=device).view(-1))
                loss_adj_graph = F.mse_loss(adj_pred.view(-1), adj_self_loop.view(-1))

                class_assign_model = ClusterAssignment(n_clusters, len(z.T), 1, centers).float().to(device)
                temp_class = class_assign_model(z)

                if epoch == 1 or epoch // 10 == 0:
                    p_distribution = torch.tensor(
                        sinkhorn(
                            temp_class.cpu().detach().numpy(),
                            lambdas,
                            torch.ones(X.shape[0]).numpy(),
                            torch.tensor([torch.sum(pseudo_labels.cpu() == i) for i in range(n_clusters)]).numpy()
                        )
                    ).float().to(device).detach()
                    q_max, q_max_index = torch.max(p_distribution, dim=1)

                KL_loss_function = nn.KLDivLoss(reduction='sum')
                loss_KL = KL_loss_function(temp_class, p_distribution)

                loss = factor_construct * loss_x + factor_ort * loss_ort + factor_corvar * loss_corvariates + factor_KL * loss_KL

                cluster_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    z_np = z.cpu().numpy()
                    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_np)

                    if y_np is not None:
                        acc, nmi, ari, f1 = evaluation(y_np, kmeans.labels_)
                        if acc > acc_max_final:
                            acc_max_final = acc
                            nmi_max, ari_max, f1_max = nmi, ari, f1
                            best_labels_final = kmeans.labels_.copy()
                            best_embeddings_final = z_np.copy()
                    else:
                        best_labels_final = kmeans.labels_.copy()
                        best_embeddings_final = z_np.copy()

                    pseudo_labels = torch.LongTensor(kmeans.labels_).to(device)
                    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

            print(f"scCDCG: Final MAX ACC (seed {seed}): {acc_max_final:.4f}, "
                  f"NMI: {nmi_max:.4f}, ARI: {ari_max:.4f}, F1: {f1_max:.4f}")

            seed_result = {
                'acc': acc_max_final,
                'nmi': nmi_max,
                'ari': ari_max,
                'f1': f1_max,
                'labels': best_labels_final,
                'embeddings': best_embeddings_final
            }

            best_global = seed_result

        if best_global is not None:
            self._labels = best_global['labels']
            self._embeddings = best_global['embeddings']

        # Store model for parameter counting and inductive inference
        try:
            self.model = Model
            self.model.eval()  # Set to evaluation mode
        except UnboundLocalError:
            pass

        # Store elements for inductive inference
        self._device = device
        self._n_clusters = n_clusters
        self._n_train_cells = X.shape[0]  # Store training set size for mode detection

        # Store cluster centers from final KMeans (in embedding space)
        if 'centers' in dir() and centers is not None:
            self._cluster_centers = centers.cpu().numpy()
        elif best_global is not None and best_global['embeddings'] is not None:
            # Recompute centers from best embeddings
            kmeans_final = KMeans(n_clusters=n_clusters, random_state=self.params.get('random_state', 2021), n_init=20)
            kmeans_final.fit(best_global['embeddings'])
            self._cluster_centers = kmeans_final.cluster_centers_

        # Store training embeddings and labels for KNN-based inference
        if best_global is not None:
            self._train_embeddings = best_global['embeddings']
            self._train_labels = best_global['labels']

            # Build KNN index for fast inference
            knn_k = self.params.get('knn_k', 15)
            self._knn_index = NearestNeighbors(n_neighbors=min(knn_k, len(self._train_labels)), metric='cosine')
            self._knn_index.fit(self._train_embeddings)

        print(f"scCDCG: Model ready for inductive inference (mode: {self.params.get('inference_mode', 'centroid')})")

        # Build loss history from last seed run
        self._loss_history = []
        if pretrain_losses:
            self._loss_history.append({
                'name': 'pretrain (MSE+Ort+Cov)',
                'epochs': list(range(len(pretrain_losses))),
                'train_loss': pretrain_losses,
                'components': {'mse+ort+cov': pretrain_losses},
            })
        if cluster_losses:
            n_pre = len(pretrain_losses) if pretrain_losses else 0
            self._loss_history.append({
                'name': 'clustering (+Adj+KL)',
                'epochs': list(range(n_pre, n_pre + len(cluster_losses))),
                'train_loss': cluster_losses,
                'components': {'mse+ort+cov+kl': cluster_losses},
            })

        self._fitted = True

        return self

    def predict(self, data: Any = None) -> Any:
        """
        Predict cluster labels.

        Behavior:
        - If data is None: returns the labels from training (original behavior)
        - If data has the same number of cells as training: returns training labels
          (assumes same data, preserves original transductive behavior)
        - If data has different number of cells: performs inductive inference

        Parameters
        ----------
        data : AnnData, np.ndarray, or None
            New data to predict labels for. If None, returns training labels.

        Returns
        -------
        np.ndarray
            Predicted cluster labels.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # If no new data provided, return training labels
        if data is None:
            return self._labels

        # Extract data matrix from new data
        use_raw_data = self.params.get('use_raw_data', True)

        if hasattr(data, 'X'):
            n_cells = data.n_obs if hasattr(data, 'n_obs') else data.X.shape[0]
            if use_raw_data:
                if hasattr(data, 'layers') and 'original_X' in data.layers:
                    X_new = data.layers['original_X']
                elif hasattr(data, 'raw') and data.raw is not None:
                    X_new = data.raw.X
                else:
                    X_new = data.X
            else:
                X_new = data.X

            if hasattr(X_new, 'toarray'):
                X_new = X_new.toarray()
        else:
            X_new = np.array(data)
            n_cells = X_new.shape[0]

        # Check if this is the same data as training (standard mode)
        # If same number of cells, assume it's the training data and return cached labels
        # This preserves the original transductive behavior for standard mode
        if hasattr(self, '_n_train_cells') and n_cells == self._n_train_cells:
            logger.debug(f"scCDCG: predict() called with same number of cells as training ({n_cells}). "
                        f"Returning cached training labels (transductive mode).")
            return self._labels

        # Different number of cells -> inductive inference on new data
        logger.info(f"scCDCG: Inductive inference on {n_cells} new cells "
                   f"(training had {getattr(self, '_n_train_cells', 'unknown')} cells)")

        X_new = X_new.astype(np.float32)

        # Encode new data using the trained model
        embeddings_new = self._encode_new_data(X_new)

        # Assign to clusters based on inference mode
        inference_mode = self.params.get('inference_mode', 'centroid')

        if inference_mode == 'knn':
            labels_new = self._predict_knn(embeddings_new)
        else:  # centroid
            labels_new = self._predict_centroid(embeddings_new)

        return labels_new

    def _encode_new_data(self, X_new: np.ndarray) -> np.ndarray:
        """
        Encode new data using the trained encoder.

        Parameters
        ----------
        X_new : np.ndarray
            New data matrix (n_cells x n_genes).

        Returns
        -------
        np.ndarray
            Embeddings for new data (n_cells x embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("Model not available for encoding. Was it properly trained?")

        # Check dimension match with model's expected input
        expected_dim = self.model.AE.Encoder[0].in_features
        if X_new.shape[1] != expected_dim:
            logger.warning(f"scCDCG encode: Dimension mismatch - input has {X_new.shape[1]} features, "
                          f"model expects {expected_dim}. Cannot encode, returning None.")
            return None

        self.model.eval()
        device = self._device if self._device is not None else torch.device('cpu')

        with torch.no_grad():
            x_tensor = torch.tensor(X_new, dtype=torch.float32).to(device)

            # The model expects an adjacency matrix, but for inference we don't need it
            # since the encoder (AE part) doesn't actually use it in forward pass
            # We pass a dummy adjacency matrix
            dummy_adj = torch.eye(X_new.shape[0], dtype=torch.float32).to(device)

            # Get embeddings from the model
            z, _ = self.model.forward(x_tensor, dummy_adj)

            # Normalize embeddings (as done during training)
            z = F.normalize(z, p=2, dim=1)

            embeddings = z.cpu().numpy()

        return embeddings

    def _predict_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels by assigning to nearest centroid.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings of new data.

        Returns
        -------
        np.ndarray
            Predicted cluster labels.
        """
        if self._cluster_centers is None:
            raise RuntimeError("Cluster centers not available. Was the model properly trained?")

        # Compute distances to all cluster centers
        # Using cosine distance since embeddings are L2-normalized
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        centers_norm = self._cluster_centers / (np.linalg.norm(self._cluster_centers, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity (higher is closer)
        similarities = np.dot(embeddings_norm, centers_norm.T)

        # Assign to cluster with highest similarity
        labels = np.argmax(similarities, axis=1)

        return labels

    def _predict_knn(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels using K-nearest neighbor voting.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings of new data.

        Returns
        -------
        np.ndarray
            Predicted cluster labels.
        """
        if self._knn_index is None or self._train_labels is None:
            raise RuntimeError("KNN index not available. Was the model properly trained?")

        knn_k = self.params.get('knn_k', 15)
        k = min(knn_k, len(self._train_labels))

        # Find k nearest neighbors
        distances, indices = self._knn_index.kneighbors(embeddings, n_neighbors=k)

        # Majority vote among neighbors
        labels = np.zeros(len(embeddings), dtype=np.int32)
        for i in range(len(embeddings)):
            neighbor_labels = self._train_labels[indices[i]]
            # Count votes for each cluster
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            labels[i] = unique_labels[np.argmax(counts)]

        return labels

    def predict_with_embeddings(self, data: Any) -> tuple:
        """
        Predict cluster labels and return embeddings for new data.

        Parameters
        ----------
        data : AnnData or np.ndarray
            New data to predict labels for.

        Returns
        -------
        tuple
            (labels, embeddings) for the new data.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Extract data matrix
        use_raw_data = self.params.get('use_raw_data', True)

        if hasattr(data, 'X'):
            if use_raw_data:
                if hasattr(data, 'layers') and 'original_X' in data.layers:
                    X_new = data.layers['original_X']
                elif hasattr(data, 'raw') and data.raw is not None:
                    X_new = data.raw.X
                else:
                    X_new = data.X
            else:
                X_new = data.X

            if hasattr(X_new, 'toarray'):
                X_new = X_new.toarray()
        else:
            X_new = np.array(data)

        X_new = X_new.astype(np.float32)

        # Encode new data
        embeddings = self._encode_new_data(X_new)

        # Predict labels
        inference_mode = self.params.get('inference_mode', 'centroid')
        if inference_mode == 'knn':
            labels = self._predict_knn(embeddings)
        else:
            labels = self._predict_centroid(embeddings)

        return labels, embeddings

    def get_embeddings(self) -> Optional[np.ndarray]:
        """Return learned embeddings."""
        return self._embeddings

    def encode(self, data: Any) -> Optional[np.ndarray]:
        """
        Encode data to get embeddings (for inductive inference).

        This method is used by analysis_runner to get test embeddings.

        Parameters
        ----------
        data : AnnData or np.ndarray
            Data to encode.

        Returns
        -------
        np.ndarray or None
            Embeddings for the input data.
        """
        if not self._fitted or self.model is None:
            return None

        # Extract data matrix
        use_raw_data = self.params.get('use_raw_data', True)

        if hasattr(data, 'X'):
            n_cells = data.n_obs if hasattr(data, 'n_obs') else data.X.shape[0]
            if use_raw_data:
                if hasattr(data, 'layers') and 'original_X' in data.layers:
                    X = data.layers['original_X']
                elif hasattr(data, 'raw') and data.raw is not None:
                    X = data.raw.X
                else:
                    X = data.X
            else:
                X = data.X

            if hasattr(X, 'toarray'):
                X = X.toarray()
        else:
            X = np.array(data)
            n_cells = X.shape[0]

        # If same size as training, return cached embeddings
        if hasattr(self, '_n_train_cells') and n_cells == self._n_train_cells:
            return self._embeddings

        # Check dimension before encoding
        expected_dim = self.model.AE.Encoder[0].in_features if self.model else None
        if expected_dim and X.shape[1] != expected_dim:
            logger.warning(f"scCDCG encode: Test data has {X.shape[1]} genes, model trained with {expected_dim}. "
                          f"Silhouette cannot be computed for test set.")
            return None

        # Otherwise, encode new data
        X = X.astype(np.float32)
        return self._encode_new_data(X)

    def get_num_parameters(self) -> Optional[int]:
        """Get the number of trainable parameters."""
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save_model(self, path: str) -> None:
        """
        Save the trained model for later inductive inference.

        Parameters
        ----------
        path : str
            Path to save the model checkpoint.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before saving")

        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model is not None else None,
            'cluster_centers': self._cluster_centers,
            'train_embeddings': self._train_embeddings,
            'train_labels': self._train_labels,
            'labels': self._labels,
            'embeddings': self._embeddings,
            'n_clusters': self._n_clusters,
            'n_train_cells': self._n_train_cells,
            'params': self.params,
            'fitted': self._fitted,
        }

        torch.save(checkpoint, path)
        print(f"scCDCG: Model saved to {path}")

    def load_model(self, path: str, device: str = None) -> 'ScCDCGAlgorithm':
        """
        Load a trained model for inductive inference.

        Parameters
        ----------
        path : str
            Path to the saved model checkpoint.
        device : str, optional
            Device to load the model on ('cuda', 'cpu', 'mps').
            If None, uses the configured device.

        Returns
        -------
        ScCDCGAlgorithm
            Self, with loaded model.
        """
        checkpoint = torch.load(path, map_location='cpu')

        # Restore parameters
        self.params = checkpoint['params']
        self._fitted = checkpoint['fitted']
        self._labels = checkpoint['labels']
        self._embeddings = checkpoint['embeddings']
        self._cluster_centers = checkpoint['cluster_centers']
        self._train_embeddings = checkpoint['train_embeddings']
        self._train_labels = checkpoint['train_labels']
        self._n_clusters = checkpoint['n_clusters']
        self._n_train_cells = checkpoint.get('n_train_cells', len(self._labels) if self._labels is not None else 0)

        # Determine device
        if device is None:
            device = self.get_device()
        self._device = torch.device(device)

        # Rebuild model architecture from params
        if checkpoint['model_state_dict'] is not None:
            embedding_dim = self.params.get('embedding_dim', 16)
            encoder_layers_str = self.params.get('encoder_layers', '256')
            decoder_layers_str = self.params.get('decoder_layers', '')

            encoder_hidden = [int(x.strip()) for x in encoder_layers_str.split(',') if x.strip()]
            if not encoder_hidden:
                encoder_hidden = [256]

            dims_encoder = encoder_hidden + [embedding_dim]

            if decoder_layers_str.strip():
                decoder_hidden = [int(x.strip()) for x in decoder_layers_str.split(',') if x.strip()]
                dims_decoder = [embedding_dim] + decoder_hidden
            else:
                dims_decoder = [embedding_dim] + list(reversed(encoder_hidden))

            # Infer input dimension from state dict
            first_layer_key = 'AE.Encoder.0.weight'
            dim_input = checkpoint['model_state_dict'][first_layer_key].shape[1]

            # Rebuild model
            self.model = FULL_NN(
                dim_input=dim_input,
                dims_encoder=dims_encoder,
                dims_decoder=dims_decoder,
                num_class=self._n_clusters
            ).float().to(self._device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

        # Rebuild KNN index
        if self._train_embeddings is not None and self._train_labels is not None:
            knn_k = self.params.get('knn_k', 15)
            self._knn_index = NearestNeighbors(
                n_neighbors=min(knn_k, len(self._train_labels)),
                metric='cosine'
            )
            self._knn_index.fit(self._train_embeddings)

        print(f"scCDCG: Model loaded from {path}")
        print(f"scCDCG: Ready for inductive inference (mode: {self.params.get('inference_mode', 'centroid')})")

        return self
