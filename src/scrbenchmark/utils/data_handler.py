"""
Data handling utilities for loading and preprocessing scRNA-seq data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import logging

from .dropout_simulation import apply_dropout

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Handler for loading and preprocessing single-cell RNA-seq data.
    Supports multiple file formats: H5, H5AD, CSV, TSV, MTX.
    """

    SUPPORTED_FORMATS = ['.h5', '.h5ad', '.csv', '.tsv', '.txt', '.mtx']
    LABEL_COLUMNS = ['Group', 'label', 'cell_type', 'cluster', 'Groupe', 'Y',
                     'celltype', 'CellType', 'cell_types', 'labels', 'clusters',
                     'assigned_cluster']

    def __init__(self):
        self.adata_original = None  # To store the completely original, unfiltered data
        self.adata = None
        self.raw_data = None
        self.labels = None
        self.gene_names = None
        self.cell_names = None
        self._dropout_info = None
        self._batch_correction_applied = False

    def load(self, file_path: Union[str, Path]) -> 'DataHandler':
        """
        Load data from file.

        Args:
            file_path: Path to the data file

        Returns:
            self for method chaining
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # specific handling for compressed files
        is_compressed = file_path.suffix.lower() == '.gz'
        suffixes = file_path.suffixes
        
        # Check standard extensions
        if file_path.suffix.lower() == '.h5ad':
            self._load_h5ad(file_path)
        elif file_path.suffix.lower() == '.h5':
            self._load_h5(file_path)
        elif file_path.suffix.lower() == '.mtx':
            self._load_mtx(file_path)
        elif file_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            self._load_tabular(file_path)
        elif is_compressed:
            # Check inner extension for compressed files
            if len(suffixes) >= 2:
                inner_ext = suffixes[-2].lower()
                if inner_ext in ['.csv', '.tsv', '.txt']:
                    self._load_tabular(file_path)
                else:
                    raise ValueError(f"Unsupported compressed file format: {''.join(suffixes)}. "
                                   "Expected .csv.gz, .tsv.gz, or .txt.gz")
            else:
                 raise ValueError(f"Cannot determine format of compressed file: {file_path.name}")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}. "
                           f"Supported: {self.SUPPORTED_FORMATS}")

        # Store a copy of the original data immediately after loading
        if self.adata is not None:
             self.adata_original = self.adata.copy()

        logger.info(f"Loaded data: {self.adata.shape[0]} cells, "
                   f"{self.adata.shape[1]} genes")
        return self

    # ... existing methods ...

    def _load_tabular(self, file_path: Path):
        """Load CSV/TSV/TXT file (supports .gz compression)."""
        import pandas as pd
        import anndata as ad

        # Determine separator based on extensions (handling .csv.gz etc)
        suffixes = [s.lower() for s in file_path.suffixes]
        sep = ',' # Default to comma
        if '.tsv' in suffixes or '.txt' in suffixes:
            sep = '\t'
            
        df = pd.read_csv(file_path, sep=sep, index_col=0)

        # Separate numeric data (counts) from metadata (labels/obs)
        obs_data = {}
        numeric_cols = []
        
        # Check for known label columns also, just in case they are actually numeric but meant as labels
        # But generally, we want to move strings to obs
        
        for col in df.columns:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            is_known_label = col in self.LABEL_COLUMNS or 'cluster' in col.lower() or 'type' in col.lower()
            
            if not is_numeric or is_known_label:
                obs_data[col] = df[col]
            else:
                numeric_cols.append(col)
        
        # If no numeric columns found, something is wrong, or maybe transposed?
        if not numeric_cols:
            # Fallback: try to convert everything to numeric, coercing errors
            logger.warning("No numeric columns detected in CSV. Attempting to force conversion.")
            # This is risky, but better than empty X
            X_df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            obs_df = pd.DataFrame(index=df.index.astype(str))
        else:
            X_df = df[numeric_cols]
            obs_df = pd.DataFrame(obs_data, index=df.index.astype(str))

        self.adata = ad.AnnData(X_df.values, obs=obs_df)
        self.adata.obs_names = X_df.index.astype(str)
        self.adata.var_names = X_df.columns.astype(str)

        # Check for labels in the extracted obs
        self._extract_labels()

    def apply_batch_correction(self, params: Dict[str, Any]) -> None:
        """Apply batch correction immediately after import."""
        if self.adata is None:
            raise ValueError("No data loaded. Load data before batch correction.")

        from .batch_correction import apply_batch_correction

        if self._batch_correction_applied:
            logger.info("Batch correction already applied. Skipping.")
            return

        # Preserve raw counts before correction if not already stored
        if 'original_X' not in self.adata.layers:
            self.adata.layers['original_X'] = self.adata.X.copy()

        batch_key = params.get('batch_correction_batch_key', 'batch')
        labels_key = params.get('batch_correction_labels_key', None)
        method = params.get('batch_correction_method', 'scvi').lower()
        is_sysvi = method == 'sysvi'
        default_n_latent = 15 if is_sysvi else 30
        default_n_hidden = 256 if is_sysvi else 128
        default_epochs = 200 if is_sysvi else 100
        bc_seed = params.get(
            'batch_correction_seed',
            params.get('seed', params.get('random_state', None)),
        )

        def _or_default(key: str, default: Any) -> Any:
            value = params.get(key, None)
            return default if value is None else value

        # Generic params + scVI/sysVI specific params
        bc_params = {
            'method': method,
            'n_latent': _or_default('batch_correction_n_latent', default_n_latent),
            'n_hidden': _or_default('batch_correction_n_hidden', default_n_hidden),
            'n_layers': _or_default('batch_correction_n_layers', 2),
            'max_epochs': _or_default('batch_correction_epochs', default_epochs),
            'batch_size': _or_default('batch_correction_batch_size', 128),
            'learning_rate': _or_default('batch_correction_learning_rate', 1e-3),
            'dct_weight': _or_default('batch_correction_dct_weight', 1.0),
            'corr_mse_weight': _or_default('batch_correction_corr_mse_weight', 0.1),
            # sysVI params
            'prior': _or_default('batch_correction_prior', 'vamp'),
            'n_prior_components': _or_default('batch_correction_n_prior_components', 5),
            'embed_categorical_covariates': _or_default('batch_correction_embed_covariates', True),
            'z_distance_cycle_weight': _or_default('batch_correction_cycle_weight', 5.0),
            'kl_weight': _or_default('batch_correction_kl_weight', 1.0),
            'categorical_covariate_keys': params.get('batch_correction_categorical_covariate_keys', None),
            'seed': bc_seed,
            'random_state': bc_seed,
        }

        device = params.get('batch_correction_device', 'auto')

        logger.info(f"Applying {method} batch correction at import (batch_key='{batch_key}')...")
        
        try:
            self.adata, _ = apply_batch_correction(
                self.adata,
                method=method,
                batch_key=batch_key,
                labels_key=labels_key,
                params=bc_params,
                device=device,
            )
        except ImportError as e:
            logger.error(f"Batch correction failed due to missing dependency: {e}")
            raise
        except Exception as e:
            logger.error(f"Batch correction failed: {e}")
            raise

        # Mark stage to skip re-normalization in preprocessing
        if 'batch_correction' in self.adata.uns:
            self.adata.uns['batch_correction']['stage'] = 'import'

        self._batch_correction_applied = True

    def _load_h5ad(self, file_path: Path):
        """Load AnnData H5AD file."""
        import anndata as ad

        self.adata = ad.read_h5ad(file_path)

        # If X is None, try to get data from layers or raw
        if self.adata.X is None:
            self._resolve_missing_x()

        self._extract_labels()
    
    def _resolve_missing_x(self):
        """
        Resolve missing X matrix by using data from layers or raw.
        Prioritizes raw counts over normalized data.
        """
        import anndata as ad
        
        # Priority order: raw counts first, then normalized
        priority_layers = [
            'counts',           # OpenProblems format
            'raw_counts', 
            'spliced',          # RNA velocity format
            'X_raw',
            'normalized',       # OpenProblems normalized
            'log_normalized',   # OpenProblems log-normalized
            'X'                 # Some formats
        ]
        
        # First, try layers
        if self.adata.layers:
            available = list(self.adata.layers.keys())
            logger.info(f"adata.X is None. Available layers: {available}")
            
            for layer_name in priority_layers:
                if layer_name in self.adata.layers:
                    layer_data = self.adata.layers[layer_name]
                    
                    # Check if this looks like raw counts
                    is_counts = layer_name in ['counts', 'raw_counts', 'spliced']
                    
                    logger.info(f"Using adata.layers['{layer_name}'] as X "
                               f"({'raw counts' if is_counts else 'normalized data'})")
                    
                    self.adata.X = layer_data.copy()
                    
                    # Store original layer info for reference
                    self.adata.uns['x_source'] = {
                        'layer': layer_name,
                        'is_raw_counts': is_counts
                    }
                    return
            
            # If no priority layer found, use first available
            first_layer = available[0]
            logger.warning(f"No standard layer found, using first available: '{first_layer}'")
            self.adata.X = self.adata.layers[first_layer].copy()
            self.adata.uns['x_source'] = {'layer': first_layer, 'is_raw_counts': False}
            return
        
        # Try adata.raw as last resort
        if self.adata.raw is not None and self.adata.raw.X is not None:
            logger.info("adata.X is None, using adata.raw.X")
            self.adata = ad.AnnData(
                X=self.adata.raw.X,
                obs=self.adata.obs,
                var=self.adata.raw.var if hasattr(self.adata.raw, 'var') else None
            )
            self.adata.uns['x_source'] = {'layer': 'raw', 'is_raw_counts': True}
            return
        
        raise ValueError("Could not find any data matrix in H5AD file. "
                        "X is None and no suitable layers or raw attribute found.")

    def _load_h5(self, file_path: Path):
        """Load HDF5 file, with support for 10X Genomics Cell Ranger format."""
        import h5py
        import anndata as ad
        import scipy.sparse as sp
        import scanpy as sc

        # First, check if this is a 10X Genomics HDF5 file
        with h5py.File(file_path, 'r') as f:
            is_10x_format = self._is_10x_h5(f)

        if is_10x_format:
            # Use scanpy's dedicated 10X reader
            logger.info("Detected 10X Genomics HDF5 format, using scanpy.read_10x_h5()")
            self.adata = sc.read_10x_h5(file_path)
            self._extract_labels()
            return

        # Fallback to generic H5 loading for other formats
        with h5py.File(file_path, 'r') as f:
            # Try different common structures
            if 'X' in f:
                X = f['X'][:]
            elif 'matrix' in f:
                X = f['matrix'][:]
            elif 'data' in f:
                X = f['data'][:]
            elif 'exprs' in f and isinstance(f['exprs'], h5py.Group):
                # Handle sparse matrix in 'exprs' group
                g = f['exprs']
                if all(k in g for k in ['data', 'indices', 'indptr', 'shape']):
                    data = g['data'][:]
                    indices = g['indices'][:]
                    indptr = g['indptr'][:]
                    shape = tuple(g['shape'][:])
                    # Create sparse matrix
                    X = sp.csr_matrix((data, indices, indptr), shape=shape)
                else:
                    # Fallback if not sparse components
                    logger.warning("'exprs' group found but missing sparse components")
                    X = None
            else:
                # Try to find a suitable dataset
                X = None
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 2:
                        X = f[key][:]
                        break
                
            if X is None:
                raise ValueError("Could not find data matrix in H5 file (checked X, matrix, data, exprs)")

            # Try to get labels
            labels = None
            # Check root level
            for label_key in self.LABEL_COLUMNS + ['Y']:
                if label_key in f:
                    labels = f[label_key][:]
                    break
            
            # Check inside 'obs' group if not found
            if labels is None and 'obs' in f and isinstance(f['obs'], h5py.Group):
                obs_group = f['obs']
                for label_key in self.LABEL_COLUMNS + ['Y', 'cell_type1', 'cell_ontology_class']:
                    if label_key in obs_group:
                        labels = obs_group[label_key][:]
                        break

        self.adata = ad.AnnData(X)

        if labels is not None:
            self.adata.obs['labels'] = labels
            self._encode_labels()

    def _is_10x_h5(self, f) -> bool:
        """
        Check if an open HDF5 file is in 10X Genomics Cell Ranger format.
        
        10X format has a structure like:
        - GRCh38/ (or other genome name)
            - barcodes
            - data
            - gene_names (or features for v3)
            - genes (or features for v3)
            - indices
            - indptr
            - shape
        
        Or for newer format:
        - matrix/
            - barcodes
            - data
            - features/
            - indices
            - indptr
            - shape
        """
        import h5py
        
        # Check for 10X v3 format (has 'matrix' group at root)
        if 'matrix' in f and isinstance(f['matrix'], h5py.Group):
            matrix_group = f['matrix']
            required_keys = ['data', 'indices', 'indptr', 'shape', 'barcodes']
            if all(k in matrix_group for k in required_keys):
                return True
        
        # Check for 10X v2 format (has genome-named group like 'GRCh38' or 'mm10')
        for key in f.keys():
            if isinstance(f[key], h5py.Group):
                group = f[key]
                required_keys = ['data', 'indices', 'indptr', 'shape', 'barcodes']
                optional_keys = ['gene_names', 'genes']
                if all(k in group for k in required_keys) and any(k in group for k in optional_keys):
                    return True
        
        return False



    def _load_mtx(self, file_path: Path):
        """Load Matrix Market file (10X Genomics format)."""
        import scanpy as sc

        # Assume standard 10X structure
        parent = file_path.parent
        self.adata = sc.read_10x_mtx(parent)
        self._extract_labels()

    def _extract_labels(self):
        """Extract labels from loaded AnnData."""
        for col in self.LABEL_COLUMNS:
            if col in self.adata.obs.columns:
                self.adata.obs['labels'] = self.adata.obs[col]
                self._encode_labels()
                return

    def _encode_labels(self):
        """Encode string labels to integers."""
        labels = self.adata.obs['labels']

        # Convert bytes to string if necessary
        if len(labels) > 0 and isinstance(labels.iloc[0], bytes):
            self.adata.obs['labels'] = labels.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
            labels = self.adata.obs['labels']

        if labels.dtype == object or isinstance(labels.iloc[0], str):
            # String labels - encode
            unique_labels = labels.unique()
            label_map = {label: i for i, label in enumerate(unique_labels)}
            self.adata.obs['labels_encoded'] = labels.map(label_map)
            self.adata.uns['label_map'] = label_map
            self.labels = self.adata.obs['labels_encoded'].values
        else:
            try:
                self.labels = labels.values.astype(int)
                self.adata.obs['labels_encoded'] = self.labels
            except (ValueError, TypeError):
                # Fallback for mixed types or non-convertible types
                unique_labels = labels.unique()
                label_map = {label: i for i, label in enumerate(unique_labels)}
                self.adata.obs['labels_encoded'] = labels.map(label_map)
                self.adata.uns['label_map'] = label_map
                self.labels = self.adata.obs['labels_encoded'].values

    def preprocess(self, params: Dict[str, Any] = None) -> 'DataHandler':
        """
        Preprocess the data using the same pipeline as the original scripts.

        Args:
            params: Preprocessing parameters

        Returns:
            self for method chaining
        """
        import sys
        from pathlib import Path

        params = params or {}

        if params.get('skip', False):
            logger.info("Skipping preprocessing as requested.")
            # Ensure labels are set correctly even if skipping
            if 'Group' in self.adata.obs:
                self.labels = self.adata.obs['Group'].values
            elif 'labels' in self.adata.obs:
                self.labels = self.adata.obs['labels'].values
            
            # Store current X as original_X for algorithms that need raw data
            # ONLY if not already present. This is CRITICAL if batch correction was applied at import,
            # as self.adata.X would be the corrected data, and overwriting original_X would lose the raw counts.
            if 'original_X' not in self.adata.layers:
                import scipy.sparse as sp
                X = self.adata.X
                if hasattr(X, 'toarray'):
                    X = X.toarray()
                
                # CRITICAL CHECK: Ensure we are not saving normalized floats as 'raw'
                is_integer = False
                if np.issubdtype(X.dtype, np.integer):
                    is_integer = True
                elif np.issubdtype(X.dtype, np.floating):
                    # Check tolerance
                    is_integer = np.allclose(X, np.round(X))
                
                if is_integer:
                    self.adata.layers['original_X'] = X
                else:
                    logger.warning(
                        "Skipping 'original_X' creation: Data in .X appears to be normalized (floats). "
                        "Algorithms needing raw counts (scVI, etc.) may fail."
                    )
            else:
                logger.info("Preserving existing 'original_X' layer (likely from import-time batch correction).")
                
            return self

        # Use built-in preprocessing with Scanpy
        self._preprocess_builtin(params)

        return self

    def _preprocess_builtin(self, params: Dict[str, Any] = None) -> None:
        """
        Built-in preprocessing with granular step control.
        Each step can be individually enabled/disabled via params.
        """
        import scanpy as sc
        import scipy.sparse as sp

        params = params or {}

        # Get step toggles (default all True for backward compatibility)
        do_normalization = params.get('do_normalization', True)
        do_log_transform = params.get('do_log_transform', True)
        do_hvg = params.get('do_hvg', True)
        do_scaling = params.get('do_scaling', True)
        do_batch_correction = params.get('do_batch_correction', False)
        batch_correction_method = params.get('batch_correction_method', 'scvi').lower()
        batch_corr_stage = None
        if self._batch_correction_applied and self.adata is not None:
            batch_corr_stage = self.adata.uns.get('batch_correction', {}).get('stage')
        batch_corr_at_import = batch_corr_stage == 'import'
        if batch_corr_at_import:
            logger.info("Batch correction applied at import; skipping normalization/log1p and dropout.")
            do_normalization = False
            do_log_transform = False

        # sysVI expects normalized + log1p input. Force these steps when sysVI
        # correction is requested through this preprocessing pipeline.
        if (
            do_batch_correction
            and batch_correction_method == 'sysvi'
            and not batch_corr_at_import
        ):
            if not do_normalization or not do_log_transform:
                logger.warning(
                    "Forcing do_normalization=True and do_log_transform=True "
                    "because batch_correction_method=sysvi."
                )
            do_normalization = True
            do_log_transform = True

        # Get raw X
        X = self.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Store labels
        y = self.labels

        # Store dropout parameters for later application (after filtering and HVG)
        dropout_method = params.get('dropout_method', 'none')
        dropout_params = {
            'method': dropout_method,
            'dropout_rate': params.get('dropout_rate', 0.2),
            'dropout_mid': params.get('dropout_mid', 0.0),
            'dropout_shape': params.get('dropout_shape', -1.0),
            'noise_level': params.get('noise_level', 0.0)
        }
        if batch_corr_at_import:
            dropout_params['method'] = 'none'
            dropout_params['noise_level'] = 0.0

        # Create fresh AnnData preserving important metadata columns
        import pandas as pd
        
        # Columns to preserve (batch, cell type, sample info)
        preserve_cols = ['tech', 'batch', 'Batch', 'dataset', 'dataset_source', 
                         'study', 'sample', 'donor', 'platform',
                         'celltype', 'cell_type', 'CellType', 'cell_types']
        
        obs_df = pd.DataFrame(index=range(X.shape[0]))
        if y is not None:
            obs_df['Group'] = y
        
        # Preserve existing important columns from original adata
        if self.adata_original is not None:
            for col in preserve_cols:
                if col in self.adata_original.obs.columns:
                    # Handle potential index mismatch due to cell filtering
                    if len(self.adata_original.obs) == X.shape[0]:
                        obs_df[col] = self.adata_original.obs[col].values
                    else:
                        logger.warning(f"Cannot preserve '{col}': cell count mismatch after filtering")
        
        var_df = None
        if self.adata is not None and getattr(self.adata, 'var', None) is not None:
            if self.adata.var.shape[0] == X.shape[1]:
                var_df = self.adata.var.copy()
            else:
                logger.warning("Cannot preserve var: gene count mismatch after preprocessing input")

        if var_df is not None:
            self.adata = sc.AnnData(X, obs=obs_df, var=var_df)
        else:
            self.adata = sc.AnnData(X, obs=obs_df)

        # Cell filtering (if enabled)
        do_cell_filtering = params.get('do_cell_filtering', True)
        if do_cell_filtering:
            min_genes = params.get('min_genes_per_cell', 200)
            max_genes = params.get('max_genes_per_cell', None)

            if min_genes > 0:
                sc.pp.filter_cells(self.adata, min_genes=min_genes)
            if max_genes is not None and max_genes > 0:
                sc.pp.filter_cells(self.adata, max_genes=max_genes)
        
        # Gene filtering (if enabled)
        do_gene_filtering = params.get('do_gene_filtering', True)
        if do_gene_filtering:
            min_cells = params.get('min_cells_per_gene', 3)
            if min_cells > 0:
                sc.pp.filter_genes(self.adata, min_cells=min_cells)

        # Store raw counts before any modification
        X_raw = self.adata.X.copy()
        if sp.issparse(X_raw):
            X_raw = X_raw.toarray()

        # =====================================================================
        # DROPOUT SIMULATION: Apply to RAW COUNTS before normalization
        # Scientific rationale: Dropout simulates technical capture failure,
        # which occurs at the count level (molecules not captured)
        # =====================================================================
        if dropout_params['method'] != 'none':
            X_raw_dropout, dropout_info = apply_dropout(
                X_raw,
                method=dropout_params['method'],
                dropout_rate=dropout_params['dropout_rate'],
                dropout_mid=dropout_params['dropout_mid'],
                dropout_shape=dropout_params['dropout_shape'],
                noise_level=dropout_params['noise_level']
            )
            
            # Update adata.X with dropout-affected raw counts
            self.adata.X = X_raw_dropout
            
            logger.info(f"Applied {dropout_params['method']} dropout (on RAW counts): "
                       f"actual rate = {dropout_info['actual_dropout_rate']:.2%}, "
                       f"values dropped = {dropout_info['n_values_dropped']}")
            
            self._dropout_info = dropout_info
        else:
            X_raw_dropout = X_raw  # No dropout applied

        # =====================================================================
        # SIZE FACTORS: Computed AFTER dropout to reflect observed depth
        # This is important because models using size_factors should see
        # the depth of the data they actually receive (including dropout)
        # =====================================================================
        if batch_corr_at_import and 'original_X' in self.adata.layers:
            counts_source = self.adata.layers['original_X']
            counts_per_cell = np.asarray(counts_source.sum(axis=1)).ravel()
        else:
            counts_per_cell = np.asarray(self.adata.X.sum(axis=1)).ravel()
        median_counts = np.median(counts_per_cell) if len(counts_per_cell) > 0 else 1
        if median_counts > 0:
            self.adata.obs['size_factors'] = counts_per_cell / median_counts
        else:
            self.adata.obs['size_factors'] = np.ones(len(counts_per_cell))

        # Track raw counts versions:
        # - X_raw_dropout: raw counts WITH dropout (for adata.raw and original_X layer)
        # - X_raw: original raw counts WITHOUT dropout (kept for reference)
        # NOTE: When dropout is enabled, models should see the dropout-affected data
        # for fair comparison, so we store X_raw_dropout in layers['original_X']

        # =====================================================================
        # NORMALIZATION AND LOG TRANSFORM
        # =====================================================================
        if do_normalization:
            target_sum = params.get('target_sum', 10000)
            sc.pp.normalize_total(self.adata, target_sum=target_sum)
            logger.info(f"Applied library size normalization (target_sum={target_sum})")

        if do_log_transform:
            sc.pp.log1p(self.adata)
            logger.info("Applied log1p transformation")

        # =====================================================================
        # HVG SELECTION
        # =====================================================================
        n_top_genes = params.get('n_top_genes', 5000)
        hvg_flavor = params.get('hvg_flavor', 'seurat')

        # Store FULL raw counts BEFORE HVG selection for algorithms with internal preprocessing
        # (e.g., scMAE which needs to do its own HVG selection from complete data)
        if sp.issparse(X_raw_dropout):
            pre_hvg_counts = X_raw_dropout.toarray().astype(np.float32)
        else:
            pre_hvg_counts = X_raw_dropout.astype(np.float32)
        self.adata.uns['pre_hvg_counts'] = pre_hvg_counts
        self.adata.uns['pre_hvg_var_names'] = list(self.adata.var_names)
        self.adata.uns['pre_hvg_n_genes'] = self.adata.n_vars
        logger.info(f"Stored pre-HVG raw counts: {pre_hvg_counts.shape} for algorithms with internal preprocessing")

        if do_hvg and n_top_genes > 0 and n_top_genes < self.adata.shape[1]:
            sc.pp.highly_variable_genes(
                self.adata,
                n_top_genes=min(n_top_genes, self.adata.n_vars),
                flavor=hvg_flavor
            )
            gene_mask = self.adata.var['highly_variable'].to_numpy()
            self.adata = self.adata[:, gene_mask].copy()

            # Apply same mask to raw count matrices for consistent gene set
            X_raw = X_raw[:, gene_mask]
            X_raw_dropout = X_raw_dropout[:, gene_mask]

            logger.info(f"Selected {n_top_genes} HVGs")

        # =====================================================================
        # STORE RAW COUNTS BEFORE BATCH CORRECTION
        # scVI and other methods need raw counts in layers['original_X']
        # This must be done BEFORE batch correction, not after!
        # =====================================================================
        if sp.issparse(X_raw_dropout):
            X_raw_dropout = X_raw_dropout.toarray()
        self.adata.layers['original_X'] = X_raw_dropout.astype(np.float32)
        logger.info(f"Stored HVG-filtered raw counts in layers['original_X']: {X_raw_dropout.shape}")

        # =====================================================================
        # BATCH CORRECTION (Optional)
        # NOTE: Batch correction must come BEFORE scaling!
        # scVI will use layers['original_X'] for raw counts
        # =====================================================================
        if do_batch_correction:
            strict_bc = bool(params.get('batch_correction_strict', False))
            try:
                from utils.batch_correction import apply_batch_correction
                
                method = batch_correction_method
                is_sysvi = method == 'sysvi'
                default_n_latent = 15 if is_sysvi else 30
                default_n_hidden = 256 if is_sysvi else 128
                default_epochs = 200 if is_sysvi else 100
                bc_seed = params.get(
                    'batch_correction_seed',
                    params.get('seed', params.get('random_state', None)),
                )

                def _or_default(key: str, default: Any) -> Any:
                    value = params.get(key, None)
                    return default if value is None else value

                logger.info(f"Applying batch correction ({method})...")
                
                # Prepare parameters
                bc_params = {
                    'method': method,
                    'max_epochs': _or_default('batch_correction_epochs', default_epochs),
                    'n_latent': _or_default('batch_correction_n_latent', default_n_latent),
                    'n_hidden': _or_default('batch_correction_n_hidden', default_n_hidden),
                    'batch_size': _or_default('batch_correction_batch_size', 128),
                    'dct_weight': _or_default('batch_correction_dct_weight', 1.0),
                    'corr_mse_weight': _or_default('batch_correction_corr_mse_weight', 0.1),
                    # sysVI
                    'prior': _or_default('batch_correction_prior', 'vamp'),
                    'n_prior_components': _or_default('batch_correction_n_prior_components', 5),
                    'embed_categorical_covariates': _or_default('batch_correction_embed_covariates', True),
                    'z_distance_cycle_weight': _or_default('batch_correction_cycle_weight', 5.0),
                    'kl_weight': _or_default('batch_correction_kl_weight', 1.0),
                    'categorical_covariate_keys': params.get('batch_correction_categorical_covariate_keys', None),
                    'seed': bc_seed,
                    'random_state': bc_seed,
                    'return_model': False
                }
                
                # Apply correction
                # Note: This updates adata.X and adds embeddings to obsm
                # Apply correction
                # Note: This updates adata.X and adds embeddings to obsm
                self.adata, _ = apply_batch_correction(
                    self.adata,
                    method=method,
                    batch_key=params.get('batch_correction_batch_key', 'batch'),
                    labels_key=params.get('batch_correction_labels_key', None),
                    params=bc_params
                )
                self._batch_correction_applied = True
                
            except Exception as e:
                logger.error(f"Batch correction failed: {e}")
                import traceback
                traceback.print_exc()
                if strict_bc:
                    raise RuntimeError(
                        "Batch correction failed and strict mode is enabled "
                        "(--batch-correction-strict)."
                    ) from e
                logger.warning("Skipping batch correction due to error.")

        # =====================================================================
        # SCALING: Applied AFTER batch correction to ensure consistency
        # =====================================================================
        if do_scaling:
            scale_max = params.get('scale_max_value', 10.0)

            if sp.issparse(self.adata.X):
                self.adata.X = self.adata.X.toarray()

            mean = np.mean(self.adata.X, axis=0)
            std = np.std(self.adata.X, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            self.adata.X = (self.adata.X - mean) / std
            self.adata.X = np.clip(self.adata.X, -scale_max, scale_max)
            logger.info(f"Applied Z-score scaling (clip={scale_max})")

        # =====================================================================
        # FINALIZE RAW DATA STORAGE
        # layers['original_X'] was already stored before batch correction
        # Now create adata.raw for compatibility with some tools
        # =====================================================================
        raw_counts = self.adata.layers.get('original_X', X_raw_dropout)
        if sp.issparse(raw_counts):
            raw_counts = raw_counts.toarray()

        # adata.raw stores the HVG-filtered raw counts (for tools expecting adata.raw)
        self.adata.raw = sc.AnnData(raw_counts, var=self.adata.var.copy(), obs=self.adata.obs.copy())

        # Update labels (preserve label_map from initial _set_labels)
        saved_label_map = self.adata.uns.get('label_map', None)
        if 'Group' in self.adata.obs:
            self.labels = self.adata.obs['Group'].values
            try:
                self.labels = self.labels.astype(int)
            except (ValueError, TypeError):
                unique_labels = np.unique(self.labels)
                label_map = {label: i for i, label in enumerate(unique_labels)}
                self.labels = np.array([label_map[l] for l in self.labels])
        # Restore label_map if it was lost during preprocessing
        if saved_label_map is not None and 'label_map' not in self.adata.uns:
            self.adata.uns['label_map'] = saved_label_map

        logger.info(f"Preprocessed data: {self.adata.shape[0]} cells, "
                   f"{self.adata.shape[1]} genes")

    def get_data(self) -> Any:
        """Get the AnnData object (processed)."""
        return self.adata

    def get_original_data(self) -> Any:
        """Get the original, unfiltered, unprocessed AnnData object."""
        return self.adata_original

    def get_raw_filtered_data(self) -> Any:
        """
        Get the current (filtered) AnnData but with raw counts in X.
        Useful for algorithms that need filtered cells but raw counts.
        """
        if self.adata is None:
            return None
            
        import scanpy as sc
        adata_raw = self.adata.copy()
        
        # Try to restore raw counts to X
        if 'original_X' in adata_raw.layers:
             adata_raw.X = adata_raw.layers['original_X']
        elif adata_raw.raw is not None:
             # Check if raw has same shape
             if adata_raw.raw.n_vars == adata_raw.n_vars:
                  adata_raw.X = adata_raw.raw.X
             else:
                  # If raw has more genes (due to HVG), we can't easily put it back into X
                  # without restoring genes. But algorithms usually just need counts for the *selected* genes.
                  # If 'original_X' layer was saved during preprocessing, it matches current genes.
                  # If not, we might fail or return normalized X with warning.
                  logger.warning("Could not find raw counts matching current genes (HVG selection applied?). Using current X.")
        
        return adata_raw

    def get_labels(self, data_source: str = 'processed') -> Optional[np.ndarray]:
        """
        Get the labels if available.
        
        Args:
            data_source: 'processed' or 'original' to get labels matching that data
        """
        if data_source == 'original' and self.adata_original is not None:
            if 'labels_encoded' in self.adata_original.obs:
                return self.adata_original.obs['labels_encoded'].values
            elif 'labels' in self.adata_original.obs:
                 return self.adata_original.obs['labels'].values
            elif 'Group' in self.adata_original.obs:
                 return self.adata_original.obs['Group'].values

        # If current labels are out of sync, fall back to the current adata.obs
        if self.adata is not None and self.labels is not None:
            if len(self.labels) != self.adata.n_obs:
                if 'labels_encoded' in self.adata.obs:
                    return self.adata.obs['labels_encoded'].values
                if 'labels' in self.adata.obs:
                    return self.adata.obs['labels'].values
                if 'Group' in self.adata.obs:
                    return self.adata.obs['Group'].values
                 
        return self.labels

    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        info = {
            'n_cells': self.adata.shape[0] if self.adata is not None else 0,
            'n_genes': self.adata.shape[1] if self.adata is not None else 0,
            'has_labels': self.labels is not None,
            'n_clusters': len(np.unique(self.labels)) if self.labels is not None else None,
            'has_X': self.adata is not None and self.adata.X is not None,
        }

        if self.adata is not None:
            info['obs_columns'] = list(self.adata.obs.columns)
            info['var_columns'] = list(self.adata.var.columns)
            info['has_raw'] = self.adata.raw is not None
            info['layers'] = list(self.adata.layers.keys()) if self.adata.layers else []

            if 'label_map' in self.adata.uns:
                info['label_names'] = list(self.adata.uns['label_map'].keys())

        # Add dropout info if available
        if self._dropout_info is not None:
            info['dropout_applied'] = True
            info['dropout_info'] = self._dropout_info
        else:
            info['dropout_applied'] = False

        # Add batch correction info
        info['batch_correction_applied'] = self._batch_correction_applied
        if self._batch_correction_applied and self.adata is not None:
            if 'batch_correction' in self.adata.uns:
                info['batch_correction_info'] = self.adata.uns['batch_correction']

        return info

    def get_dropout_info(self) -> Optional[Dict[str, Any]]:
        """Get information about applied dropout simulation."""
        return self._dropout_info

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported file formats."""
        return DataHandler.SUPPORTED_FORMATS.copy()
