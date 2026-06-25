"""
Dataset splitter for benchmark mode with train/val/test splits.

Provides stratified splitting that maintains:
1. Proportional representation of each cell type (label)
2. Proportional representation of each batch/dataset source
3. Prevention of data leakage (preprocessing fit on train only)

Reference: Implements the methodology requested for evaluating generalization
capacity of clustering algorithms on unseen data.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from core.config import SplitConfig

logger = logging.getLogger(__name__)


@dataclass
class SplitIndices:
    """Container for train/val/test indices."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        """Validate indices don't overlap."""
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)

        assert len(train_set & val_set) == 0, "Train and val indices overlap!"
        assert len(train_set & test_set) == 0, "Train and test indices overlap!"
        assert len(val_set & test_set) == 0, "Val and test indices overlap!"


@dataclass
class SplitResult:
    """Result of splitting a dataset."""
    adata_train: Any  # AnnData
    adata_val: Optional[Any]  # AnnData (None if val_ratio=0)
    adata_test: Any  # AnnData
    indices: SplitIndices
    split_info: Dict[str, Any] = field(default_factory=dict)

    def get_labels(self, split: str = 'train') -> Optional[np.ndarray]:
        """Get labels for a specific split."""
        adata = getattr(self, f'adata_{split}')
        if adata is None:
            return None
        # Prefer the label column used during split
        label_col = self.split_info.get('label_col') if isinstance(self.split_info, dict) else None
        if label_col and label_col in adata.obs:
            return adata.obs[label_col].values
        if 'Group' in adata.obs:
            return adata.obs['Group'].values
        if 'labels' in adata.obs:
            return adata.obs['labels'].values
        if 'labels_encoded' in adata.obs:
            return adata.obs['labels_encoded'].values
        return None


@dataclass
class PreprocessingParams:
    """
    Stores preprocessing parameters fit on training data.
    Used to transform validation/test data without data leakage.
    """
    # Scaling parameters
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    scale_max: float = 10.0

    # HVG selection (genes selected on train)
    hvg_genes: Optional[np.ndarray] = None
    hvg_strategy: str = 'train_only'
    hvg_per_batch: Optional[Dict[str, int]] = None  # HVG counts per batch (for per_batch_union strategy)

    # Normalization
    target_sum: float = 10000

    # Size factors (computed per sample, but median from train)
    median_counts: float = 1.0

    # Quality filtering (fit on train, applied consistently to all splits)
    do_cell_filtering: bool = True
    min_genes_per_cell: int = 200
    max_genes_per_cell: Optional[int] = None
    do_gene_filtering: bool = True
    min_cells_per_gene: int = 3
    filtered_gene_names: Optional[np.ndarray] = None

    # Dropout simulation (applied on raw counts before normalization)
    dropout_method: str = 'none'
    dropout_rate: float = 0.2
    dropout_mid: float = 0.0
    dropout_shape: float = -1.0
    noise_level: float = 0.0

    # Batch correction parameters
    do_batch_correction: bool = False
    batch_correction_batch_key: Optional[str] = None
    batch_correction_labels_key: Optional[str] = None
    batch_correction_method: str = 'scvi'


class DatasetSplitter:
    """
    Splits scRNA-seq datasets into train/val/test sets with stratification.

    Key features:
    - Stratified by cell type labels
    - Stratified by batch/dataset source (if available)
    - Supports two experimental schemes:
        a) Merged learning: train on merged dataset
        b) Individual learning: train on single dataset, test on all
    """

    # Common column names for batch/dataset source
    BATCH_COLUMNS = ['batch', 'dataset', 'dataset_source', 'study',
                     'sample', 'donor', 'tech', 'platform']

    def __init__(self, random_state: int = 42):
        """
        Initialize splitter.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def _normalize_ratios(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[float, float, float]:
        """Validate and normalize split ratios."""
        ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
        if np.any(np.isnan(ratios)):
            raise ValueError("Split ratios must be numeric values.")
        if np.any(ratios < 0):
            raise ValueError("Split ratios must be non-negative.")
        total = float(ratios.sum())
        if total <= 0:
            raise ValueError("Split ratios must sum to a positive value.")
        if train_ratio <= 0:
            raise ValueError("train_ratio must be > 0.")
        if not np.isclose(total, 1.0):
            logger.warning(f"Ratios sum to {total}, normalizing...")
            ratios = ratios / total
        return float(ratios[0]), float(ratios[1]), float(ratios[2])

    def split_stratified(
        self,
        adata: Any,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        label_col: str = 'Group',
        batch_col: Optional[str] = None,
        stratify_by_batch: bool = True,
        stratify_by_labels: bool = True
    ) -> SplitResult:
        """
        Split dataset with stratification by labels and/or batch.

        Args:
            adata: AnnData object to split
            train_ratio: Proportion for training (default 0.8)
            val_ratio: Proportion for validation (default 0.1, set to 0 to skip)
            test_ratio: Proportion for testing (default 0.1)
            label_col: Column name for cell type labels
            batch_col: Column name for batch/dataset source (auto-detected if None)
            stratify_by_batch: Whether to stratify by batch source
            stratify_by_labels: Whether to stratify by cell type labels

        Returns:
            SplitResult with train/val/test AnnData objects and indices
        """
        import anndata as ad

        train_ratio, val_ratio, test_ratio = self._normalize_ratios(
            train_ratio, val_ratio, test_ratio
        )

        n_cells = adata.n_obs
        if n_cells == 0:
            raise ValueError("Cannot split an empty dataset.")

        # Get labels (if needed for stratification)
        labels = None
        if stratify_by_labels:
            if label_col in adata.obs.columns:
                labels = adata.obs[label_col].values
            elif 'labels' in adata.obs.columns:
                labels = adata.obs['labels'].values
            elif 'labels_encoded' in adata.obs.columns:
                labels = adata.obs['labels_encoded'].values
            else:
                logger.warning("No label column found, splitting without label stratification")
                stratify_by_labels = False

        # Auto-detect batch column
        if batch_col is None and stratify_by_batch:
            batch_col = self._detect_batch_column(adata)

        # Get batch labels (if needed for stratification)
        batches = None
        if stratify_by_batch:
            if batch_col and batch_col in adata.obs.columns:
                batches = adata.obs[batch_col].values
                logger.info(f"Stratifying by batch column: {batch_col}")
                logger.info(f"Found {len(np.unique(batches))} unique batches")
            else:
                logger.warning("No batch column found, splitting without batch stratification")
                stratify_by_batch = False

        # Create stratification groups based on user choice
        if stratify_by_labels and stratify_by_batch and batches is not None:
            # Both: combine labels and batches
            strat_groups = np.array([f"{l}_{b}" for l, b in zip(labels, batches)])
            logger.info("Stratifying by BOTH cell types AND batches")
        elif stratify_by_labels and labels is not None:
            # Labels only
            strat_groups = labels
            logger.info("Stratifying by cell types only")
        elif stratify_by_batch and batches is not None:
            # Batch only
            strat_groups = batches
            logger.info("Stratifying by batches only")
        else:
            # No stratification - random split
            strat_groups = np.zeros(n_cells)
            logger.warning("No stratification applied - using random split")

        # Perform stratified split
        indices = self._stratified_split(
            n_cells, strat_groups, train_ratio, val_ratio, test_ratio
        )

        if len(indices.train) == 0:
            raise ValueError("Train split is empty; adjust split ratios.")
        if test_ratio > 0:
            if len(indices.test) == 0:
                raise ValueError("Test split is empty; adjust split ratios.")
            if len(indices.test) < SplitConfig.MIN_TEST_CELLS:
                logger.warning(
                    f"Test split has only {len(indices.test)} cells; "
                    f"recommended minimum is {SplitConfig.MIN_TEST_CELLS}."
                )

        # Create split AnnData objects
        adata_train = adata[indices.train].copy()
        adata_test = adata[indices.test].copy()
        adata_val = adata[indices.val].copy() if len(indices.val) > 0 else None

        # =====================================================================
        # PROPAGATE pre_hvg_counts to each split subset
        # pre_hvg_counts contains raw counts for ALL genes before HVG selection,
        # needed by algorithms like scMAE that do their own internal HVG selection
        # =====================================================================
        if 'pre_hvg_counts' in adata.uns and adata.uns['pre_hvg_counts'] is not None:
            pre_hvg_counts = adata.uns['pre_hvg_counts']
            pre_hvg_var_names = adata.uns.get('pre_hvg_var_names', None)
            
            # Index by cell position for each split
            adata_train.uns['pre_hvg_counts'] = pre_hvg_counts[indices.train]
            adata_train.uns['pre_hvg_var_names'] = pre_hvg_var_names
            
            adata_test.uns['pre_hvg_counts'] = pre_hvg_counts[indices.test]
            adata_test.uns['pre_hvg_var_names'] = pre_hvg_var_names
            
            if adata_val is not None:
                adata_val.uns['pre_hvg_counts'] = pre_hvg_counts[indices.val]
                adata_val.uns['pre_hvg_var_names'] = pre_hvg_var_names
            
            logger.info(f"Propagated pre_hvg_counts ({pre_hvg_counts.shape[1]} genes) to train/val/test splits")

        # Compute split statistics
        split_info = self._compute_split_info(
            adata, indices, labels, batches, label_col, batch_col
        )

        logger.info(f"Split complete: Train={len(indices.train)}, "
                   f"Val={len(indices.val)}, Test={len(indices.test)}")

        return SplitResult(
            adata_train=adata_train,
            adata_val=adata_val,
            adata_test=adata_test,
            indices=indices,
            split_info=split_info
        )

    def split_by_batch(
        self,
        adata: Any,
        train_batches: List[str],
        test_batches: Optional[List[str]] = None,
        val_ratio: float = 0.1,
        batch_col: Optional[str] = None,
        label_col: str = 'Group',
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        use_stratified_base: bool = True,
        stratify_by_labels: bool = False
    ) -> SplitResult:
        """
        Split dataset by batch for cross-dataset generalization experiments.

        When use_stratified_base=True (default):
            First performs a stratified split on the full dataset, then filters
            the TRAIN set by batch. The TEST set remains IDENTICAL to the
            stratified split (contains ALL batches).

            This allows comparing:
            - Training on 1 batch vs training on all batches
            - With the EXACT SAME test set for fair comparison

        When use_stratified_base=False:
            Legacy behavior - takes all cells from selected batches for both
            train and test sets.

        Args:
            adata: AnnData object to split
            train_batches: List of batch names to use for training
            test_batches: List of batch names for testing (ignored when
                          use_stratified_base=True, as test set is kept intact)
            val_ratio: Proportion for validation (default 0.1)
            batch_col: Column name for batch (auto-detected if None)
            label_col: Column name for labels
            train_ratio: Proportion for training in stratified base (default 0.7)
            test_ratio: Proportion for testing in stratified base (default 0.2)
            use_stratified_base: If True, use stratified split as base (default True)

        Returns:
            SplitResult with train/val/test AnnData objects
        """
        import anndata as ad

        # Auto-detect batch column
        if batch_col is None:
            batch_col = self._detect_batch_column(adata)
            if batch_col is None:
                raise ValueError("No batch column found. Specify batch_col parameter.")

        batches = adata.obs[batch_col].values
        unique_batches = np.unique(batches)

        if not train_batches:
            raise ValueError("train_batches must contain at least one batch.")

        if val_ratio < 0 or val_ratio >= 1:
            raise ValueError("val_ratio must be in the range [0, 1).")

        # Validate train batches
        for b in train_batches:
            if b not in unique_batches:
                raise ValueError(f"Batch '{b}' not found. Available: {unique_batches}")

        # Determine test batches (only used in legacy mode)
        if test_batches is None:
            test_batches = [b for b in unique_batches if b not in train_batches]
            if len(test_batches) == 0:
                # If no other batches, use all batches for testing
                test_batches = list(unique_batches)
                logger.warning("No separate test batches, testing on all batches")

        if use_stratified_base:
            train_ratio, val_ratio, test_ratio = self._normalize_ratios(
                train_ratio, val_ratio, test_ratio
            )
            if test_ratio <= 0:
                raise ValueError("test_ratio must be > 0 for batch-filtered splits.")
            # =====================================================================
            # BATCH-FILTERED MODE (Strategy 2) - Correct Implementation:
            #
            # Step 1: Perform stratified split 70/10/20 on ENTIRE dataset (all batches)
            # Step 2: Filter TRAIN to keep only cells from selected batch(es)
            # Step 3: Filter VAL to keep only cells from selected batch(es)
            # Step 4: Keep TEST complete (all batches, balanced)
            #
            # This allows testing cross-domain generalization:
            # - Train on D1 only
            # - Validate on D1 only (hyperparameter tuning on same domain)
            # - Test on D1 ∪ D2 ∪ D3 ∪ D4 (complete balanced test set)
            # =====================================================================
            logger.info("Batch-Filtered Mode: Stratified split first, then filter train/val by batch")
            
            # Step 1: Perform stratified split on ENTIRE dataset
            logger.info("Step 1: Stratified split on entire dataset (70/10/20)")
            base_split = self.split_stratified(
                adata,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                label_col=label_col,
                batch_col=batch_col,
                stratify_by_batch=True,
                stratify_by_labels=stratify_by_labels  # User choice: batch-only (False) or batch+labels (True)
            )
            
            # Step 2: Filter TRAIN to keep only selected batch(es)
            train_batches_arr = adata.obs[batch_col].values[base_split.indices.train]
            train_batch_mask = np.isin(train_batches_arr, train_batches)
            train_indices = base_split.indices.train[train_batch_mask]
            
            logger.info(f"Step 2: Filter train to {train_batches}")
            logger.info(f"  Original train: {len(base_split.indices.train)} → Filtered: {len(train_indices)}")
            
            # Step 3: Filter VAL to keep only selected batch(es)
            val_batches_arr = adata.obs[batch_col].values[base_split.indices.val]
            val_batch_mask = np.isin(val_batches_arr, train_batches)
            val_indices = base_split.indices.val[val_batch_mask]
            
            logger.info(f"Step 3: Filter val to {train_batches}")
            logger.info(f"  Original val: {len(base_split.indices.val)} → Filtered: {len(val_indices)}")
            
            # Step 4: Keep TEST complete (all batches)
            test_indices = base_split.indices.test
            
            logger.info(f"Step 4: Keep test complete (all batches)")
            logger.info(f"  Test: {len(test_indices)} cells (all batches)")
            
            # Log batch composition of test set
            test_batches_arr = adata.obs[batch_col].values[test_indices]
            test_batch_counts = {b: np.sum(test_batches_arr == b) for b in unique_batches}
            logger.info(f"  Test batch composition: {test_batch_counts}")
            
            logger.info(f"Batch-Filtered split complete:")
            logger.info(f"  Train: {len(train_indices)} cells (from {train_batches} only)")
            logger.info(f"  Val: {len(val_indices)} cells (from {train_batches} only)")
            logger.info(f"  Test: {len(test_indices)} cells (all batches, balanced)")

        else:
            # =====================================================================
            # LEGACY BEHAVIOR: Take all cells from selected batches
            # =====================================================================
            logger.info("Using legacy batch split (all cells from selected batches)")

            # Get indices for each set
            train_mask = np.isin(batches, train_batches)
            test_mask = np.isin(batches, test_batches)

            train_indices_all = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            # Split train into train/val
            if val_ratio > 0 and len(train_indices_all) > 0:
                # Get labels for stratification within train
                if label_col in adata.obs.columns:
                    train_labels = adata.obs[label_col].values[train_indices_all]
                else:
                    train_labels = np.zeros(len(train_indices_all))

                n_val = int(len(train_indices_all) * val_ratio)
                val_indices_local, train_indices_local = self._stratified_sample(
                    len(train_indices_all), train_labels, n_val
                )

                train_indices = train_indices_all[train_indices_local]
                val_indices = train_indices_all[val_indices_local]
            else:
                train_indices = train_indices_all
                val_indices = np.array([], dtype=int)

        if len(train_indices) == 0:
            raise ValueError("Train split is empty; check selected batches and split ratios.")
        if len(test_indices) == 0:
            raise ValueError("Test split is empty; check split ratios.")
        if len(test_indices) < SplitConfig.MIN_TEST_CELLS:
            logger.warning(
                f"Test split has only {len(test_indices)} cells; "
                f"recommended minimum is {SplitConfig.MIN_TEST_CELLS}."
            )
        if len(val_indices) == 0 and val_ratio > 0:
            logger.warning("Validation split is empty; consider adjusting val_ratio.")

        indices = SplitIndices(
            train=train_indices,
            val=val_indices,
            test=test_indices
        )

        # Create split AnnData objects
        adata_train = adata[indices.train].copy()
        adata_test = adata[indices.test].copy()
        adata_val = adata[indices.val].copy() if len(indices.val) > 0 else None

        # =====================================================================
        # PROPAGATE pre_hvg_counts to each split subset
        # Same as split_stratified for consistency
        # =====================================================================
        if 'pre_hvg_counts' in adata.uns and adata.uns['pre_hvg_counts'] is not None:
            pre_hvg_counts = adata.uns['pre_hvg_counts']
            pre_hvg_var_names = adata.uns.get('pre_hvg_var_names', None)
            
            adata_train.uns['pre_hvg_counts'] = pre_hvg_counts[indices.train]
            adata_train.uns['pre_hvg_var_names'] = pre_hvg_var_names
            
            adata_test.uns['pre_hvg_counts'] = pre_hvg_counts[indices.test]
            adata_test.uns['pre_hvg_var_names'] = pre_hvg_var_names
            
            if adata_val is not None:
                adata_val.uns['pre_hvg_counts'] = pre_hvg_counts[indices.val]
                adata_val.uns['pre_hvg_var_names'] = pre_hvg_var_names
            
            logger.info(f"Propagated pre_hvg_counts ({pre_hvg_counts.shape[1]} genes) to batch-split train/val/test")

        # Get labels array
        if label_col in adata.obs.columns:
            labels = adata.obs[label_col].values
        else:
            labels = None

        split_info = self._compute_split_info(
            adata, indices, labels, batches, label_col, batch_col
        )
        split_info['train_batches'] = train_batches
        split_info['split_mode'] = 'by_batch'
        split_info['use_stratified_base'] = use_stratified_base

        if use_stratified_base:
            # In stratified base mode, test set contains ALL batches (unchanged)
            split_info['test_batches'] = list(unique_batches)
            split_info['base_train_ratio'] = train_ratio
            split_info['base_val_ratio'] = val_ratio
            split_info['base_test_ratio'] = test_ratio
            split_info['test_set_unchanged'] = True
            logger.info(f"Split by batch complete (stratified base mode):")
            logger.info(f"  Train batches: {train_batches}")
            logger.info(f"  Test batches: ALL ({list(unique_batches)})")
        else:
            # In legacy mode, test set is filtered by test_batches
            split_info['test_batches'] = test_batches
            split_info['test_set_unchanged'] = False
            logger.info(f"Split by batch complete (legacy mode):")
            logger.info(f"  Train batches: {train_batches}")
            logger.info(f"  Test batches: {test_batches}")

        logger.info(f"  Train={len(train_indices)}, Val={len(val_indices)}, "
                   f"Test={len(test_indices)}")

        return SplitResult(
            adata_train=adata_train,
            adata_val=adata_val,
            adata_test=adata_test,
            indices=indices,
            split_info=split_info
        )

    def balance_by_batch(
        self,
        adata: Any,
        batch_col: Optional[str] = None,
        target_per_batch: Optional[int] = None,
        preserve_labels: bool = True,
        label_col: str = 'Group'
    ) -> Any:
        """
        Balance dataset by subsampling larger batches to match batch sizes.
        
        Args:
            adata: AnnData object to balance
            batch_col: Column name for batch (auto-detected if None)
            target_per_batch: Target cells per batch (None = use smallest batch size)
            preserve_labels: If True, maintain label proportions within each batch
            label_col: Column for stratification if preserve_labels=True
            
        Returns:
            Balanced AnnData object
        """
        import anndata as ad
        
        # Auto-detect batch column
        if batch_col is None:
            batch_col = self._detect_batch_column(adata)
            if batch_col is None:
                logger.warning("No batch column found. Returning original data.")
                return adata
        
        batches = adata.obs[batch_col]
        batch_counts = batches.value_counts()
        
        # Determine target size
        if target_per_batch is None:
            target_per_batch = batch_counts.min()
        
        logger.info(f"Balancing batches to {target_per_batch} cells each")
        
        selected_indices = []
        
        for batch_name, count in batch_counts.items():
            batch_mask = batches == batch_name
            batch_indices = np.where(batch_mask)[0]
            
            if count <= target_per_batch:
                # Keep all cells from this batch
                selected_indices.extend(batch_indices)
            else:
                # Subsample
                if preserve_labels and label_col in adata.obs.columns:
                    # Stratified subsampling by labels
                    batch_labels = adata.obs[label_col].values[batch_indices]
                    selected, _ = self._stratified_sample(
                        len(batch_indices), batch_labels, target_per_batch
                    )
                    selected_indices.extend(batch_indices[selected])
                else:
                    # Random subsampling
                    selected = np.random.choice(batch_indices, target_per_batch, replace=False)
                    selected_indices.extend(selected)
        
        selected_indices = np.array(selected_indices)
        balanced_adata = adata[selected_indices].copy()
        
        # Log the result
        new_batch_counts = balanced_adata.obs[batch_col].value_counts()
        logger.info(f"Balanced dataset: {len(balanced_adata)} cells, {len(new_batch_counts)} batches")
        for batch_name, count in new_batch_counts.items():
            logger.info(f"  {batch_name}: {count} cells")
        
        return balanced_adata

    def balance_with_surplus(
        self,
        adata: Any,
        batch_col: Optional[str] = None,
        target_per_batch: Optional[int] = None,
        preserve_labels: bool = True,
        label_col: str = 'Group'
    ) -> Tuple[Any, Any]:
        """
        Balance dataset by subsampling larger batches to match batch sizes,
        AND return the surplus (eliminated) cells as a separate AnnData object.

        This is used for the "Reinject" strategy:
        1. Balance to 1303 cells (Core)
        2. Keep the surplus (Reinjectable)
        3. Split the Core
        4. Add Reinjectable to Train

        Args:
            adata: AnnData object to balance
            batch_col: Column name for batch (auto-detected if None)
            target_per_batch: Target cells per batch (None = use smallest batch size)
            preserve_labels: If True, maintain label proportions within each batch
            label_col: Column for stratification if preserve_labels=True

        Returns:
            (balanced_adata, surplus_adata)
        """
        import anndata as ad

        # Auto-detect batch column
        if batch_col is None:
            batch_col = self._detect_batch_column(adata)
            if batch_col is None:
                logger.warning("No batch column found. Returning original data and empty surplus.")
                return adata, None

        batches = adata.obs[batch_col]
        batch_counts = batches.value_counts()

        # Determine target size
        if target_per_batch is None:
            target_per_batch = batch_counts.min()

        logger.info(f"Balancing batches to {target_per_batch} cells each (with surplus extraction)")

        selected_indices = []
        surplus_indices = []

        for batch_name, count in batch_counts.items():
            batch_mask = batches == batch_name
            batch_indices = np.where(batch_mask)[0]

            if count <= target_per_batch:
                # Keep all cells from this batch
                selected_indices.extend(batch_indices)
            else:
                # Subsample
                if preserve_labels and label_col in adata.obs.columns:
                    # Stratified subsampling by labels
                    batch_labels = adata.obs[label_col].values[batch_indices]
                    selected, surplus = self._stratified_sample(
                        len(batch_indices), batch_labels, target_per_batch
                    )
                    selected_indices.extend(batch_indices[selected])
                    surplus_indices.extend(batch_indices[surplus])
                else:
                    # Random subsampling
                    np.random.seed(self.random_state)
                    # Use permutation to separate selected and surplus
                    perm = np.random.permutation(len(batch_indices))
                    selected_local = perm[:target_per_batch]
                    surplus_local = perm[target_per_batch:]
                    
                    selected_indices.extend(batch_indices[selected_local])
                    surplus_indices.extend(batch_indices[surplus_local])

        selected_indices = np.array(selected_indices)
        surplus_indices = np.array(surplus_indices)

        balanced_adata = adata[selected_indices].copy()
        
        if len(surplus_indices) > 0:
            surplus_adata = adata[surplus_indices].copy()
            logger.info(f"Extracted {len(surplus_adata)} surplus cells")
        else:
            surplus_adata = None
            logger.info("No surplus cells found")

        # Log the result
        new_batch_counts = balanced_adata.obs[batch_col].value_counts()
        logger.info(f"Balanced Core: {len(balanced_adata)} cells")
        
        return balanced_adata, surplus_adata

    def balance_test_and_reinject(
        self,
        split_result: SplitResult,
        batch_col: str,
        target_per_batch: Optional[int] = None,
        reinject_to: str = 'train',  # 'train', 'val', or 'both'
        preserve_labels: bool = True,
        label_col: str = 'Group'
    ) -> SplitResult:
        """
        Balance only the TEST set and reinject excess cells into TRAIN and/or VAL.

        This method ensures:
        1. TEST set has equal representation per batch (for fair evaluation)
        2. No cells are lost - excess cells are added to TRAIN/VAL
        3. All original data is preserved

        Args:
            split_result: Original SplitResult from split_stratified()
            batch_col: Column name for batch identification
            target_per_batch: Target cells per batch in TEST (None = use smallest batch)
            reinject_to: Where to reinject excess cells: 'train', 'val', or 'both'
            preserve_labels: Maintain label proportions when selecting cells to keep in TEST
            label_col: Column name for labels (used if preserve_labels=True)

        Returns:
            New SplitResult with balanced TEST and augmented TRAIN/VAL
        """
        import anndata as ad

        adata_test = split_result.adata_test
        adata_train = split_result.adata_train
        adata_val = split_result.adata_val

        if batch_col not in adata_test.obs.columns:
            logger.warning(f"Batch column '{batch_col}' not found in test data. Returning unchanged.")
            return split_result

        # Get batch counts in TEST
        test_batches = adata_test.obs[batch_col]
        batch_counts = test_batches.value_counts()

        # Determine target size (smallest batch in TEST)
        if target_per_batch is None:
            target_per_batch = batch_counts.min()

        logger.info(f"Balancing TEST to {target_per_batch} cells per batch")
        logger.info(f"Original TEST batch sizes: {dict(batch_counts)}")

        # Identify cells to keep in TEST vs reinject
        keep_in_test_indices = []  # Local indices within adata_test
        reinject_indices = []       # Local indices within adata_test

        for batch_name, count in batch_counts.items():
            batch_mask = test_batches == batch_name
            batch_local_indices = np.where(batch_mask)[0]

            if count <= target_per_batch:
                # Keep all cells from this batch in TEST
                keep_in_test_indices.extend(batch_local_indices)
            else:
                # Select cells to keep, rest will be reinjected
                if preserve_labels and label_col in adata_test.obs.columns:
                    # Stratified selection to maintain label proportions
                    batch_labels = adata_test.obs[label_col].values[batch_local_indices]
                    selected_local, not_selected_local = self._stratified_sample(
                        len(batch_local_indices), batch_labels, target_per_batch
                    )
                    keep_in_test_indices.extend(batch_local_indices[selected_local])
                    reinject_indices.extend(batch_local_indices[not_selected_local])
                else:
                    # Random selection
                    np.random.seed(self.random_state)
                    perm = np.random.permutation(len(batch_local_indices))
                    keep_in_test_indices.extend(batch_local_indices[perm[:target_per_batch]])
                    reinject_indices.extend(batch_local_indices[perm[target_per_batch:]])

        keep_in_test_indices = np.array(keep_in_test_indices)
        reinject_indices = np.array(reinject_indices)

        n_reinjected = len(reinject_indices)
        logger.info(f"Cells to reinject from TEST: {n_reinjected}")

        if n_reinjected == 0:
            logger.info("No cells to reinject, TEST already balanced.")
            return split_result

        # Create new balanced TEST
        new_adata_test = adata_test[keep_in_test_indices].copy()

        # Get cells to reinject
        cells_to_reinject = adata_test[reinject_indices].copy()

        # Reinject into TRAIN and/or VAL
        if reinject_to == 'train':
            new_adata_train = ad.concat([adata_train, cells_to_reinject], join='outer')
            new_adata_val = adata_val
            logger.info(f"Reinjected {n_reinjected} cells into TRAIN")
        elif reinject_to == 'val':
            if adata_val is None:
                logger.warning("No VAL set exists, reinjecting into TRAIN instead")
                new_adata_train = ad.concat([adata_train, cells_to_reinject], join='outer')
                new_adata_val = None
            else:
                new_adata_train = adata_train
                new_adata_val = ad.concat([adata_val, cells_to_reinject], join='outer')
                logger.info(f"Reinjected {n_reinjected} cells into VAL")
        elif reinject_to == 'both':
            if adata_val is None:
                logger.warning("No VAL set exists, reinjecting all into TRAIN")
                new_adata_train = ad.concat([adata_train, cells_to_reinject], join='outer')
                new_adata_val = None
            else:
                # Split reinjected cells proportionally between TRAIN and VAL
                n_train = len(adata_train)
                n_val = len(adata_val)
                train_ratio = n_train / (n_train + n_val)

                n_to_train = int(n_reinjected * train_ratio)
                n_to_val = n_reinjected - n_to_train

                np.random.seed(self.random_state)
                perm = np.random.permutation(n_reinjected)

                train_reinject = cells_to_reinject[perm[:n_to_train]].copy()
                val_reinject = cells_to_reinject[perm[n_to_train:]].copy()

                new_adata_train = ad.concat([adata_train, train_reinject], join='outer')
                new_adata_val = ad.concat([adata_val, val_reinject], join='outer')
                logger.info(f"Reinjected {n_to_train} cells into TRAIN, {n_to_val} into VAL")
        else:
            raise ValueError(f"Invalid reinject_to value: {reinject_to}. Use 'train', 'val', or 'both'.")

        # Propagate pre_hvg_counts if present
        # Note: The concatenated data already has these from the original splits

        # Update indices (use non-overlapping ranges for SplitIndices validation)
        # These indices are symbolic - actual data is in the AnnData objects
        n_train = len(new_adata_train)
        n_val = len(new_adata_val) if new_adata_val is not None else 0
        n_test = len(new_adata_test)

        new_indices = SplitIndices(
            train=np.arange(0, n_train),
            val=np.arange(n_train, n_train + n_val) if n_val > 0 else np.array([], dtype=int),
            test=np.arange(n_train + n_val, n_train + n_val + n_test)
        )

        # Update split_info
        new_split_info = split_result.split_info.copy()
        new_split_info['n_train'] = len(new_adata_train)
        new_split_info['n_val'] = len(new_adata_val) if new_adata_val is not None else 0
        new_split_info['n_test'] = len(new_adata_test)
        new_split_info['n_reinjected'] = n_reinjected
        new_split_info['reinject_to'] = reinject_to
        new_split_info['test_balanced'] = True
        new_split_info['test_target_per_batch'] = target_per_batch

        # Log final sizes
        logger.info(f"Final sizes: TRAIN={len(new_adata_train)}, "
                   f"VAL={len(new_adata_val) if new_adata_val else 0}, "
                   f"TEST={len(new_adata_test)}")

        new_test_batch_counts = new_adata_test.obs[batch_col].value_counts()
        logger.info(f"Balanced TEST batch sizes: {dict(new_test_batch_counts)}")

        return SplitResult(
            adata_train=new_adata_train,
            adata_val=new_adata_val,
            adata_test=new_adata_test,
            indices=new_indices,
            split_info=new_split_info
        )

    def _detect_batch_column(self, adata: Any) -> Optional[str]:
        """Auto-detect the batch/dataset source column."""
        for col in self.BATCH_COLUMNS:
            if col in adata.obs.columns:
                return col
        return None

    def _stratified_split(
        self,
        n_samples: int,
        groups: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> SplitIndices:
        """
        Perform stratified split maintaining group proportions.
        Falls back to random split if stratification fails (e.g., rare classes).
        """
        from sklearn.model_selection import train_test_split

        indices = np.arange(n_samples)

        try:
            # First split: train+val vs test
            if test_ratio > 0:
                train_val_idx, test_idx = self._robust_train_test_split(
                    indices,
                    groups,
                    test_size=test_ratio,
                    random_state=self.random_state
                )
            else:
                train_val_idx = indices
                test_idx = np.array([], dtype=int)

            # Second split: train vs val
            if val_ratio > 0 and len(train_val_idx) > 0:
                # Adjust val_ratio relative to train+val
                relative_val_ratio = val_ratio / (train_ratio + val_ratio)

                train_val_groups = groups[train_val_idx]
                
                train_idx_local, val_idx_local = self._robust_train_test_split(
                    np.arange(len(train_val_idx)),
                    train_val_groups,
                    test_size=relative_val_ratio,
                    random_state=self.random_state
                )

                train_idx = train_val_idx[train_idx_local]
                val_idx = train_val_idx[val_idx_local]
            else:
                train_idx = train_val_idx
                val_idx = np.array([], dtype=int)
                
        except Exception as e:
            # Ultimate fallback if even robust split fails
            logger.error(f"Robust stratification failed: {e}")
            logger.warning("Falling back to fully random split")
            
            np.random.shuffle(indices)
            n_test = int(n_samples * test_ratio)
            n_val = int(n_samples * val_ratio)
            
            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]

        return SplitIndices(train=train_idx, val=val_idx, test=test_idx)

    def _robust_train_test_split(
        self,
        indices: np.ndarray,
        groups: np.ndarray,
        test_size: float,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Robust wrapper for train_test_split that handles rare classes.
        
        Strategy:
        1. Try standard stratified split.
        2. If it fails (rare classes), identify classes with count < 2.
        3. Split data into 'common' (stratifiable) and 'rare' (random split).
        4. Perform stratified split on 'common' and random split on 'rare'.
        5. Combine results.
        """
        from sklearn.model_selection import train_test_split
        
        # Immediate fallback for very small N
        if len(indices) < 2:
            return (indices, np.array([], dtype=int)) if test_size < 1 else (np.array([], dtype=int), indices)

        try:
            # Try standard stratification first
            return train_test_split(
                indices,
                test_size=test_size,
                stratify=groups,
                random_state=random_state
            )
        except ValueError:
            # Fallback for rare groups
            unique_groups, counts = np.unique(groups, return_counts=True)
            rare_groups = unique_groups[counts < 2]
            
            if len(rare_groups) > 0:
                logger.warning(
                    f"Stratification warning: {len(rare_groups)} groups have < 2 samples. "
                    "Using hybrid split (stratified for common, random for rare)."
                )
                
            # Mask for rare groups
            is_rare = np.isin(groups, rare_groups)
            
            # Split indices
            indices_rare = indices[is_rare]
            indices_common = indices[~is_rare]
            groups_common = groups[~is_rare]
            
            # Split common (stratified)
            if len(indices_common) > 0:
                # Recursively call robust split in case 'common' also has issues 
                # (though <2 check should catch most, edge cases with exact test_size might exist)
                try:
                    train_c, test_c = train_test_split(
                        indices_common,
                        test_size=test_size,
                        stratify=groups_common,
                        random_state=random_state
                    )
                except ValueError:
                     # Deep fallback if even filtered groups fail (e.g. very small N)
                     if len(indices_common) < 2:
                         train_c, test_c = (indices_common, np.array([], dtype=int)) if test_size < 1 else (np.array([], dtype=int), indices_common)
                     else:
                         train_c, test_c = train_test_split(
                            indices_common,
                            test_size=test_size,
                            random_state=random_state
                        )
            else:
                train_c, test_c = np.array([], dtype=int), np.array([], dtype=int)
                
            # Split rare (random)
            if len(indices_rare) > 0:
                if len(indices_rare) < 2:
                    train_r, test_r = (indices_rare, np.array([], dtype=int)) if test_size < 1 else (np.array([], dtype=int), indices_rare)
                else:
                    train_r, test_r = train_test_split(
                        indices_rare,
                        test_size=test_size,
                        random_state=random_state
                    )
            else:
                train_r, test_r = np.array([], dtype=int), np.array([], dtype=int)
            
            # Combine
            return (
                np.concatenate([train_c, train_r]),
                np.concatenate([test_c, test_r])
            )

    def _stratified_sample(
        self,
        n_samples: int,
        groups: np.ndarray,
        n_select: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample n_select indices maintaining group proportions.
        Returns: (selected_indices, remaining_indices)
        """
        from sklearn.model_selection import train_test_split

        indices = np.arange(n_samples)
        ratio = n_select / n_samples

        try:
            selected, remaining = train_test_split(
                indices,
                test_size=1-ratio,
                stratify=groups,
                random_state=self.random_state
            )
        except ValueError:
            # Fallback for very small groups
            np.random.shuffle(indices)
            selected = indices[:n_select]
            remaining = indices[n_select:]

        return selected, remaining

    def _compute_split_info(
        self,
        adata: Any,
        indices: SplitIndices,
        labels: Optional[np.ndarray],
        batches: Optional[np.ndarray],
        label_col: str,
        batch_col: Optional[str]
    ) -> Dict[str, Any]:
        """Compute statistics about the split."""
        info = {
            'n_train': len(indices.train),
            'n_val': len(indices.val),
            'n_test': len(indices.test),
            'n_total': len(indices.train) + len(indices.val) + len(indices.test),
            'train_ratio': len(indices.train) / adata.n_obs,
            'val_ratio': len(indices.val) / adata.n_obs if len(indices.val) > 0 else 0,
            'test_ratio': len(indices.test) / adata.n_obs,
            'label_col': label_col,
            'batch_col': batch_col,
            'split_mode': 'stratified'
        }

        # Label distribution per split
        if labels is not None:
            info['label_distribution'] = {
                'train': self._count_distribution(labels[indices.train]),
                'test': self._count_distribution(labels[indices.test])
            }
            if len(indices.val) > 0:
                info['label_distribution']['val'] = self._count_distribution(
                    labels[indices.val]
                )

        # Batch distribution per split
        if batches is not None:
            info['batch_distribution'] = {
                'train': self._count_distribution(batches[indices.train]),
                'test': self._count_distribution(batches[indices.test])
            }
            if len(indices.val) > 0:
                info['batch_distribution']['val'] = self._count_distribution(
                    batches[indices.val]
                )

        return info

    def _count_distribution(self, arr: np.ndarray) -> Dict[str, int]:
        """Count occurrences of each unique value."""
        unique, counts = np.unique(arr, return_counts=True)
        return {str(u): int(c) for u, c in zip(unique, counts)}


class BenchmarkPreprocessor:
    """
    Preprocessor that fits on training data and transforms train/val/test
    without data leakage.

    Usage:
        preprocessor = BenchmarkPreprocessor()
        preprocessor.fit(adata_train, params)
        adata_train_processed = preprocessor.transform(adata_train)
        adata_test_processed = preprocessor.transform(adata_test)
    """

    def __init__(self):
        self.params: Optional[PreprocessingParams] = None
        self._fitted = False
        self._scvi_model = None  # Store scVI model for batch correction

    def fit(self, adata: Any, params: Dict[str, Any] = None) -> 'BenchmarkPreprocessor':
        """
        Fit preprocessing parameters on training data.

        Args:
            adata: Training AnnData object
            params: Preprocessing parameters (same as DataHandler.preprocess)

        Returns:
            self for method chaining
        """
        import scanpy as sc
        import scipy.sparse as sp

        params = params or {}
        self.params = PreprocessingParams()

        adata = adata.copy()

        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = X.astype(np.float32)

        # ---------------------------------------------------------------------
        # QUALITY FILTERING (fit on train only, then reused in transform())
        # ---------------------------------------------------------------------
        self.params.do_cell_filtering = bool(params.get('do_cell_filtering', True))
        self.params.min_genes_per_cell = int(params.get('min_genes_per_cell', 200))
        max_genes_raw = params.get('max_genes_per_cell', None)
        if max_genes_raw in (None, "", 0):
            self.params.max_genes_per_cell = None
        else:
            self.params.max_genes_per_cell = int(max_genes_raw)

        if self.params.do_cell_filtering:
            genes_per_cell = np.count_nonzero(X, axis=1)
            cell_mask = genes_per_cell >= self.params.min_genes_per_cell
            if self.params.max_genes_per_cell is not None:
                cell_mask &= genes_per_cell <= self.params.max_genes_per_cell

            kept_cells = int(np.sum(cell_mask))
            if kept_cells == 0:
                raise ValueError(
                    "Cell filtering removed all training cells. "
                    "Relax min/max genes per cell thresholds."
                )

            if kept_cells != adata.n_obs:
                logger.info(
                    "BenchmarkPreprocessor fit: cell filtering kept %d/%d cells.",
                    kept_cells,
                    adata.n_obs,
                )
                adata = adata[cell_mask].copy()
                X = X[cell_mask]

        self.params.do_gene_filtering = bool(params.get('do_gene_filtering', True))
        self.params.min_cells_per_gene = int(params.get('min_cells_per_gene', 3))
        if self.params.do_gene_filtering:
            cells_per_gene = np.count_nonzero(X, axis=0)
            gene_mask = cells_per_gene >= self.params.min_cells_per_gene
            kept_genes = int(np.sum(gene_mask))
            if kept_genes == 0:
                raise ValueError(
                    "Gene filtering removed all genes in training split. "
                    "Relax min cells per gene threshold."
                )
            if kept_genes != adata.n_vars:
                logger.info(
                    "BenchmarkPreprocessor fit: gene filtering kept %d/%d genes.",
                    kept_genes,
                    adata.n_vars,
                )
                adata = adata[:, gene_mask].copy()
                X = X[:, gene_mask]

        self.params.filtered_gene_names = np.asarray(adata.var_names)

        batch_corr_stage = adata.uns.get('batch_correction', {}).get('stage') if hasattr(adata, 'uns') else None
        batch_corr_at_import = batch_corr_stage == 'import'
        if batch_corr_at_import:
            logger.info("Batch correction applied at import; skipping dropout and normalization/log1p.")

        # =====================================================================
        # DROPOUT PARAMETERS: Stored for transform() and applied here for HVG/median
        # Applied on raw counts before normalization (same as DataHandler)
        # =====================================================================
        self.params.dropout_method = params.get('dropout_method', 'none')
        self.params.dropout_rate = params.get('dropout_rate', 0.2)
        self.params.dropout_mid = params.get('dropout_mid', 0.0)
        self.params.dropout_shape = params.get('dropout_shape', -1.0)
        self.params.noise_level = params.get('noise_level', 0.0)
        if batch_corr_at_import:
            self.params.dropout_method = 'none'
            self.params.noise_level = 0.0

        if self.params.dropout_method != 'none':
            from .dropout_simulation import apply_dropout
            X, _ = apply_dropout(
                X,
                method=self.params.dropout_method,
                dropout_rate=self.params.dropout_rate,
                dropout_mid=self.params.dropout_mid,
                dropout_shape=self.params.dropout_shape,
                noise_level=self.params.noise_level
            )

        # Store size factor median from train (after dropout)
        if batch_corr_at_import and hasattr(adata, 'layers') and 'original_X' in adata.layers:
            counts_per_cell = np.asarray(np.sum(adata.layers['original_X'], axis=1)).ravel()
        else:
            counts_per_cell = np.asarray(np.sum(X, axis=1)).ravel()
        self.params.median_counts = float(np.median(counts_per_cell))
        if self.params.median_counts <= 0:
            logger.warning("Training median counts is non-positive; using 1.0 for size factors.")
            self.params.median_counts = 1.0

        # Store target sum
        self.params.target_sum = params.get('target_sum', 10000)
        self.params.scale_max = params.get('scale_max_value', 10.0)

        # Apply normalization and log1p to get HVGs
        do_normalization = params.get('do_normalization', True)
        do_log_transform = params.get('do_log_transform', True)
        do_hvg = params.get('do_hvg', True)
        do_scaling = params.get('do_scaling', True)
        if batch_corr_at_import:
            do_normalization = False
            do_log_transform = False

        # Work on a copy for HVG selection (use dropout-affected counts)
        adata_temp = adata.copy()
        adata_temp.X = X

        if do_normalization:
            sc.pp.normalize_total(adata_temp, target_sum=self.params.target_sum)
        if do_log_transform:
            sc.pp.log1p(adata_temp)

        # Select HVGs based on strategy
        n_top_genes = params.get('n_top_genes', 5000)
        hvg_flavor = params.get('hvg_flavor', 'seurat')
        hvg_strategy = params.get('hvg_strategy', 'train_only')
        self.params.hvg_strategy = hvg_strategy

        # Store FULL raw counts BEFORE HVG selection for algorithms with internal preprocessing
        # (e.g., scMAE which needs to do its own HVG selection from complete data)
        self.params.pre_hvg_counts = X.astype(np.float32)
        self.params.pre_hvg_var_names = list(adata_temp.var_names)
        self.params.pre_hvg_n_genes = adata_temp.n_vars
        logger.info(f"Stored pre-HVG raw counts: {X.shape} for algorithms with internal preprocessing")

        if do_hvg and n_top_genes > 0 and n_top_genes < adata_temp.n_vars:
            if hvg_strategy == 'per_batch_union':
                # Strategy: Select top N HVGs from each batch, then take union
                hvg_genes, hvg_per_batch = self._select_hvg_per_batch_union(
                    adata_temp, n_top_genes, hvg_flavor, params.get('batch_col')
                )
                self.params.hvg_genes = hvg_genes
                self.params.hvg_per_batch = hvg_per_batch  # Store per-batch counts
                gene_mask = np.isin(adata_temp.var_names, hvg_genes)
                adata_temp = adata_temp[:, gene_mask].copy()
                logger.info(f"Per-batch HVG union: {len(hvg_genes)} genes selected")
            else:
                # Default strategy: HVGs from training data only
                sc.pp.highly_variable_genes(
                    adata_temp,
                    n_top_genes=min(n_top_genes, adata_temp.n_vars),
                    flavor=hvg_flavor
                )
                self.params.hvg_genes = adata_temp.var_names[
                    adata_temp.var['highly_variable']
                ].values
                adata_temp = adata_temp[:, adata_temp.var['highly_variable']].copy()

            # Enrich HVGs with cell type markers if option enabled
            ensure_celltype_markers = params.get('ensure_celltype_markers', False)
            if ensure_celltype_markers:
                label_col = params.get('label_col', 'Group')
                min_markers = params.get('min_markers_per_celltype', 5)
                max_additional = params.get('max_additional_hvg_genes', 500)

                # Use adata_temp which has normalized+log data for DE analysis
                self.params.hvg_genes = self._enrich_hvg_with_celltype_markers(
                    adata_temp,
                    self.params.hvg_genes,
                    label_col=label_col,
                    min_markers_per_celltype=min_markers,
                    max_additional_genes=max_additional
                )

                # Update adata_temp to include new genes
                gene_mask = np.isin(adata.var_names, self.params.hvg_genes)
                # Need to re-normalize+log the full adata with new genes
                adata_temp = adata[:, gene_mask].copy()
                if do_normalization:
                    sc.pp.normalize_total(adata_temp, target_sum=self.params.target_sum)
                if do_log_transform:
                    sc.pp.log1p(adata_temp)

            # Guardrail: keep the final number of genes aligned with the UI request.
            if self.params.hvg_genes is not None and len(self.params.hvg_genes) > n_top_genes:
                logger.warning(
                    "HVG selection produced %d genes (> n_top_genes=%d). "
                    "Capping to the top requested genes by variance.",
                    len(self.params.hvg_genes),
                    n_top_genes,
                )
                X_temp = adata_temp.X.toarray() if sp.issparse(adata_temp.X) else np.asarray(adata_temp.X)
                gene_vars = np.var(X_temp, axis=0)
                if not np.any(np.isfinite(gene_vars)):
                    selected_idx = np.arange(min(n_top_genes, adata_temp.n_vars))
                else:
                    selected_idx = np.argsort(gene_vars)[::-1][:n_top_genes]
                selected_genes = np.asarray(adata_temp.var_names)[selected_idx]
                self.params.hvg_genes = selected_genes
                adata_temp = adata_temp[:, selected_genes].copy()
                logger.info("HVG cap applied: final number of genes = %d", adata_temp.n_vars)
        else:
            self.params.hvg_genes = adata_temp.var_names.values



        # =====================================================================
        # Batch Correction: Train scVI model on TRAIN data only
        # =====================================================================
        do_batch_correction = params.get('do_batch_correction', False)
        self.params.do_batch_correction = do_batch_correction
        
        if do_batch_correction:
            from .batch_correction import (
                check_scvi_available,
                DomainClassTripletLoss_TrainingPlan,
                N_PCA_COMPONENTS_FOR_CORR,
                _train_scvi_model,
                _resolve_dct_labels_key,
            )
            
            if not check_scvi_available():
                logger.warning("scVI not available. Skipping batch correction.")
                self.params.do_batch_correction = False
            else:
                import scvi
                import scanpy as sc
                from scvi import REGISTRY_KEYS
                
                # Get method
                method = params.get('batch_correction_method', 'scvi').lower()
                self.params.batch_correction_method = method
                
                batch_key = params.get('batch_correction_batch_key', 'auto')
                labels_key = params.get('batch_correction_labels_key', None)
                
                # Auto-detect batch key if set to 'auto'
                if batch_key == 'auto':
                    detected_key = get_batch_column(adata_temp)
                    if detected_key:
                        batch_key = detected_key
                        logger.info(f"Auto-detected batch column for correction: '{batch_key}'")
                    else:
                        batch_key = 'batch'  # Default fallback
                
                # Resolve labels for DCT supervision.
                # Priority: explicit batch_correction_labels_key -> benchmark label_col -> auto-detection.
                if labels_key is None:
                    labels_key = params.get('label_col', None)
                labels_key = _resolve_dct_labels_key(
                    adata=adata_temp,
                    batch_key=batch_key,
                    labels_key=labels_key,
                )
                if labels_key is None:
                    logger.warning(
                        "No valid cell-type labels found for DCT. "
                        "Benchmark scVI correction will run without DCT supervision."
                    )
                else:
                    logger.info(f"Using labels_key '{labels_key}' for benchmark DCT supervision.")
                
                self.params.batch_correction_batch_key = batch_key
                self.params.batch_correction_labels_key = labels_key
                
                # Check if batch column exists
                if batch_key not in adata_temp.obs.columns:
                    logger.warning(f"Batch column '{batch_key}' not found. Skipping batch correction.")
                    self.params.do_batch_correction = False
                elif method in {'scvi', 'scvi_dct'}:
                    # Only train scVI model if method is scvi
                    n_batches = adata_temp.obs[batch_key].nunique()
                    logger.info(f"Training scVI for batch correction on TRAIN data ({n_batches} batches)...")
                    
                    # Prepare PCA for DCT correlation loss (uses preprocessed data)
                    pca_matrix = None
                    if 'X_pca' not in adata_temp.obsm:
                        n_comps = min(
                            N_PCA_COMPONENTS_FOR_CORR,
                            adata_temp.n_vars - 1,
                            adata_temp.n_obs - 1,
                        )
                        if n_comps > 0:
                            sc.pp.pca(adata_temp, n_comps=n_comps)
                    if 'X_pca' in adata_temp.obsm:
                        pca_matrix = adata_temp.obsm['X_pca'].astype(np.float32, copy=False)

                    # Prepare adata for scVI
                    # IMPORTANT: scVI assumes raw count data (Negative Binomial model)
                    # We need to use raw counts, not log-normalized data
                    
                    # Get raw counts (dropout already applied if enabled)
                    # ROBUSTNESS: Check for original_X or raw layers
                    X_raw = X.copy()
                    if hasattr(adata, 'layers') and 'original_X' in adata.layers:
                        logger.info("Using original_X layer for scVI training (ensuring raw counts)")
                        X_raw = adata.layers['original_X'].copy()
                        if hasattr(X_raw, 'toarray'):
                            X_raw = X_raw.toarray()
                    elif hasattr(adata, 'raw') and adata.raw is not None:
                        # Only use raw if shape matches (HVG filtering might have happened earlier?)
                        # Actually fit is usually called before HVG filtering in the class logic
                        # But verifying shape is safe
                        raw_X = adata.raw.X
                        if raw_X.shape == X.shape:
                             logger.info("Using adata.raw for scVI training")
                             X_raw = raw_X if not hasattr(raw_X, 'toarray') else raw_X.toarray()

                    
                    # Create a new AnnData with raw counts for scVI
                    import anndata as ad
                    adata_scvi = ad.AnnData(
                        X=X_raw.copy(),
                        obs=adata.obs.copy(),
                        var=adata.var.copy() if hasattr(adata, 'var') else None
                    )
                    
                    # Filter to HVG genes if already selected
                    if self.params.hvg_genes is not None:
                        gene_mask = np.isin(adata_scvi.var_names if adata_scvi.var is not None 
                                           else np.arange(adata_scvi.n_vars), 
                                           self.params.hvg_genes)
                        if np.sum(gene_mask) > 0:
                            adata_scvi = adata_scvi[:, gene_mask].copy()
                    
                    logger.info("Using raw counts for scVI training (correct for NB model)")

                    # Ensure float32 + sanitize input to avoid NaNs on MPS
                    if hasattr(adata_scvi.X, 'astype'):
                        adata_scvi.X = adata_scvi.X.astype(np.float32)
                    if sp.issparse(adata_scvi.X):
                        if np.isnan(adata_scvi.X.data).any() or np.isinf(adata_scvi.X.data).any():
                            logger.warning("Input data contains NaNs or Infs! Fixing for scVI training...")
                            adata_scvi.X.data = np.nan_to_num(adata_scvi.X.data)
                    else:
                        if np.isnan(adata_scvi.X).any() or np.isinf(adata_scvi.X).any():
                            logger.warning("Input data contains NaNs or Infs! Fixing for scVI training...")
                            adata_scvi.X = np.nan_to_num(adata_scvi.X)
                        adata_scvi.X = np.maximum(adata_scvi.X, 0)
                    
                    # Handle labels key
                    if labels_key and labels_key not in adata_scvi.obs.columns:
                        labels_key = None

                    use_dct_loss = labels_key is not None and pca_matrix is not None

                    use_pca_indices = hasattr(REGISTRY_KEYS, "INDICES_KEY")
                    pca_column_names = None
                    if use_dct_loss and not use_pca_indices:
                        n_pcs = min(N_PCA_COMPONENTS_FOR_CORR, pca_matrix.shape[1])
                        pca_column_names = []
                        for i in range(n_pcs):
                            col_name = f'_pca_cov_{i}'
                            adata_scvi.obs[col_name] = pca_matrix[:, i].astype(np.float32)
                            pca_column_names.append(col_name)
                    
                    # Setup scVI with raw counts
                    scvi.model.SCVI.setup_anndata(
                        adata_scvi,
                        batch_key=batch_key,
                        labels_key=labels_key,
                        continuous_covariate_keys=pca_column_names,
                    )
                    
                    n_latent = params.get('batch_correction_n_latent', None)
                    if n_latent is None:
                        n_latent = 30
                    max_epochs = params.get('batch_correction_epochs', None)
                    if max_epochs is None:
                        max_epochs = 100
                    batch_size = params.get('batch_correction_batch_size', None)
                    if batch_size is None:
                        batch_size = 128
                    
                    model = scvi.model.SCVI(
                        adata_scvi,
                        n_latent=n_latent,
                        n_hidden=128,
                        n_layers=2,
                    )
                    
                    # Select reference batch for correction (first batch found)
                    # We map all cells to this batch to remove batch effects in the output matrix
                    # Convert to string/category value expected by scVI
                    ref_batch = adata.obs[batch_key].unique()[0]
                    self.params.batch_correction_reference_batch = str(ref_batch)
                    logger.info(f"Selected reference batch for scVI correction: '{ref_batch}'")
                    
                    # Determine accelerator
                    import torch
                    accelerator = "cpu"
                    devices = "auto"
                    
                    if torch.cuda.is_available():
                        accelerator = "gpu"
                        devices = 1
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        accelerator = "mps"
                        devices = 1
                    
                    batch_correction_weight = {
                        "batch_celltype_constraint_weight": params.get('batch_correction_dct_weight', 1.0),
                        "corr_mse_weight": params.get('batch_correction_corr_mse_weight', 0.1),
                    }

                    if use_dct_loss:
                        logger.info("Training with DomainClassTripletLoss (DCT-Corr)")
                        try:
                            plan_ok = _train_scvi_model(
                                model,
                                max_epochs=max_epochs,
                                batch_size=batch_size,
                                plan_class=DomainClassTripletLoss_TrainingPlan,
                                plan_kwargs={
                                    'batch_correction_weight': batch_correction_weight,
                                    'pca_matrix': pca_matrix if use_pca_indices else None,
                                },
                                accelerator=accelerator,
                                devices=devices,
                            )
                            if plan_ok:
                                self._scvi_use_dct = True
                            else:
                                logger.warning(
                                    "scvi-tools version does not support custom training plans. "
                                    "Falling back to standard scVI."
                                )
                                self._scvi_use_dct = False
                                _train_scvi_model(
                                    model,
                                    max_epochs=max_epochs,
                                    batch_size=batch_size,
                                    accelerator=accelerator,
                                    devices=devices,
                                )
                        except Exception as e:
                            logger.warning(f"DCT training failed ({e}).")
                            if accelerator != "cpu":
                                logger.warning("Retrying DCT training on CPU for stability.")
                                model = scvi.model.SCVI(
                                    adata_scvi,
                                    n_latent=n_latent,
                                    n_hidden=128,
                                    n_layers=2,
                                )
                                try:
                                    plan_ok = _train_scvi_model(
                                        model,
                                        max_epochs=max_epochs,
                                        batch_size=batch_size,
                                        plan_class=DomainClassTripletLoss_TrainingPlan,
                                        plan_kwargs={
                                            'batch_correction_weight': batch_correction_weight,
                                            'pca_matrix': pca_matrix if use_pca_indices else None,
                                        },
                                        accelerator="cpu",
                                        devices="auto",
                                    )
                                    if plan_ok:
                                        self._scvi_use_dct = True
                                    else:
                                        logger.warning(
                                            "scvi-tools version does not support custom training plans. "
                                            "Falling back to standard scVI."
                                        )
                                        self._scvi_use_dct = False
                                        _train_scvi_model(
                                            model,
                                            max_epochs=max_epochs,
                                            batch_size=batch_size,
                                            accelerator="cpu",
                                            devices="auto",
                                        )
                                except Exception as cpu_e:
                                    logger.warning(f"DCT training failed on CPU ({cpu_e}). Falling back to standard scVI.")
                                    model = scvi.model.SCVI(
                                        adata_scvi,
                                        n_latent=n_latent,
                                        n_hidden=128,
                                        n_layers=2,
                                    )
                                    _train_scvi_model(
                                        model,
                                        max_epochs=max_epochs,
                                        batch_size=batch_size,
                                        accelerator="cpu",
                                        devices="auto",
                                    )
                                    self._scvi_use_dct = False
                            else:
                                logger.warning("Falling back to standard scVI.")
                                _train_scvi_model(
                                    model,
                                    max_epochs=max_epochs,
                                    batch_size=batch_size,
                                    accelerator=accelerator,
                                    devices=devices,
                                )
                                self._scvi_use_dct = False
                    else:
                        logger.info("Labels/PCA missing; training standard scVI without DCT loss")
                        try:
                            _train_scvi_model(
                                model,
                                max_epochs=max_epochs,
                                batch_size=batch_size,
                                accelerator=accelerator,
                                devices=devices,
                            )
                        except Exception as e:
                            if accelerator != "cpu":
                                logger.warning(f"Standard scVI training failed ({e}). Retrying on CPU.")
                                model = scvi.model.SCVI(
                                    adata_scvi,
                                    n_latent=n_latent,
                                    n_hidden=128,
                                    n_layers=2,
                                )
                                _train_scvi_model(
                                    model,
                                    max_epochs=max_epochs,
                                    batch_size=batch_size,
                                    accelerator="cpu",
                                    devices="auto",
                                )
                            else:
                                raise
                        self._scvi_use_dct = False
                    
                    # Store model for transform
                    self._scvi_model = model
                    logger.info(f"scVI batch correction model trained on {adata_scvi.n_obs} cells")

                    # UPDATE adata_temp with corrected data for Scaling calculation
                    logger.info("Generating corrected training data for scaling parameter calculation...")
                    corrected_train = model.get_normalized_expression(
                        library_size=1e4,
                        return_numpy=True,
                        transform_batch=self.params.batch_correction_reference_batch
                    )
                    corrected_train = np.log1p(corrected_train)
                    adata_temp.X = corrected_train
                
                else:
                    logger.info(f"Batch correction method '{method}' selected. Will be applied during transform step.")


        # =====================================================================
        # SCALING: Compute parameters on Train (Corrected or Uncorrected)
        # =====================================================================
        if do_scaling:
            X_temp = adata_temp.X
            if hasattr(X_temp, 'toarray'):
                X_temp = X_temp.toarray()

            self.params.mean = np.mean(X_temp, axis=0)
            self.params.std = np.std(X_temp, axis=0)
            self.params.std[self.params.std == 0] = 1  # Avoid division by zero
            logger.info("Computed scaling parameters on training data (post-correction if applied)")

        self._fitted = True
        logger.info(f"Preprocessor fit on {adata.n_obs} cells, "
                   f"{len(self.params.hvg_genes)} HVGs selected")

        return self

    def transform(self, adata: Any, params: Dict[str, Any] = None) -> Any:
        """
        Transform data using parameters fit on training data.

        Args:
            adata: AnnData object to transform
            params: Optional override params (use fit params if None)

        Returns:
            Transformed AnnData object
        """
        import anndata as ad
        import scanpy as sc
        import scipy.sparse as sp

        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform")

        params = params or {}
        adata = adata.copy()

        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = X.astype(np.float32)

        batch_corr_stage = None
        if hasattr(adata, "uns"):
            batch_corr_stage = adata.uns.get("batch_correction", {}).get("stage")
        batch_corr_at_import = batch_corr_stage == "import"

        # ---------------------------------------------------------------------
        # QUALITY FILTERING (reuse train-fit filtering configuration)
        # ---------------------------------------------------------------------
        do_cell_filtering = bool(params.get('do_cell_filtering', self.params.do_cell_filtering))
        min_genes_per_cell = int(params.get('min_genes_per_cell', self.params.min_genes_per_cell))
        max_genes_raw = params.get('max_genes_per_cell', self.params.max_genes_per_cell)
        if max_genes_raw in (None, "", 0):
            max_genes_per_cell = None
        else:
            max_genes_per_cell = int(max_genes_raw)

        if do_cell_filtering:
            genes_per_cell = np.count_nonzero(X, axis=1)
            cell_mask = genes_per_cell >= min_genes_per_cell
            if max_genes_per_cell is not None:
                cell_mask &= genes_per_cell <= max_genes_per_cell

            kept_cells = int(np.sum(cell_mask))
            if kept_cells == 0:
                raise ValueError(
                    "Cell filtering removed all cells in transform split. "
                    "Relax min/max genes per cell thresholds."
                )
            if kept_cells != adata.n_obs:
                logger.info(
                    "BenchmarkPreprocessor transform: cell filtering kept %d/%d cells.",
                    kept_cells,
                    adata.n_obs,
                )
                adata = adata[cell_mask].copy()
                X = X[cell_mask]

        do_gene_filtering = bool(params.get('do_gene_filtering', self.params.do_gene_filtering))
        if do_gene_filtering and self.params.filtered_gene_names is not None:
            gene_mask = np.isin(adata.var_names, self.params.filtered_gene_names)
            kept_genes = int(np.sum(gene_mask))
            if kept_genes == 0:
                raise ValueError(
                    "Gene filtering removed all genes in transform split. "
                    "Check train/test gene consistency."
                )
            if kept_genes < len(self.params.filtered_gene_names):
                logger.warning(
                    "BenchmarkPreprocessor transform: only %d/%d train-filtered genes found.",
                    kept_genes,
                    len(self.params.filtered_gene_names),
                )
            if kept_genes != adata.n_vars:
                adata = adata[:, gene_mask].copy()
                X = X[:, gene_mask]

        # Store FULL raw counts BEFORE HVG filtering for algorithms with internal preprocessing
        # (e.g., scMAE which needs to do its own HVG selection from complete data)
        # 
        # CRITICAL FIX: At this point, adata.var_names may already be limited to HVG genes
        # (if the input data was split from HVG-filtered data). We need to use the 
        # pre_hvg_counts stored during fit() which contains ALL genes, not just HVG.
        #
        # However, we can only do this if we can map cells between the current adata
        # and the original full-gene data. Since transform() receives a subset of cells,
        # we need to store the cell indices during split and use them here.
        #
        # For now, we propagate the pre_hvg metadata but note that the counts need to be
        # obtained from the original unsplit data file. Algorithms should handle this.
        if hasattr(self.params, 'pre_hvg_var_names') and self.params.pre_hvg_var_names is not None:
            # Check if adata has pre_hvg_counts already (stored during split)
            if 'pre_hvg_counts' in adata.uns:
                # Already have pre-HVG counts from split, just verify
                logger.info(f"Using existing pre_hvg_counts from split: {adata.uns['pre_hvg_counts'].shape}")
            else:
                # Try to extract from current data if gene names match
                # This works when input adata has full genes (before HVG filtering)
                current_genes = list(adata.var_names)
                matching_genes = [g for g in current_genes if g in self.params.pre_hvg_var_names]
                
                if len(matching_genes) == len(self.params.pre_hvg_var_names):
                    # All pre-HVG genes present, extract counts
                    gene_mask = np.isin(current_genes, self.params.pre_hvg_var_names)
                    adata.uns['pre_hvg_counts'] = X[:, gene_mask].astype(np.float32)
                    adata.uns['pre_hvg_var_names'] = self.params.pre_hvg_var_names
                    adata.uns['pre_hvg_n_genes'] = len(self.params.pre_hvg_var_names)
                    logger.info(f"Extracted pre_hvg_counts: {adata.uns['pre_hvg_counts'].shape}")
                else:
                    # Input data already filtered (likely HVG), store what we have
                    # and note that scMAE may need to work with limited genes
                    logger.warning(f"Only {len(matching_genes)}/{len(self.params.pre_hvg_var_names)} "
                                  f"pre-HVG genes found. scMAE may have reduced performance.")
                    adata.uns['pre_hvg_counts'] = X.astype(np.float32)
                    adata.uns['pre_hvg_var_names'] = current_genes
                    adata.uns['pre_hvg_n_genes'] = len(current_genes)

        # Filter to HVG genes (same genes as train)
        if self.params.hvg_genes is not None:
            # Find matching genes
            gene_mask = np.isin(adata.var_names, self.params.hvg_genes)
            if np.sum(gene_mask) < len(self.params.hvg_genes):
                logger.warning(f"Only {np.sum(gene_mask)}/{len(self.params.hvg_genes)} "
                             f"HVG genes found in data")
            adata = adata[:, gene_mask].copy()
            X = X[:, gene_mask]

        # =====================================================================
        # DROPOUT SIMULATION: Applied on RAW counts before normalization
        # Same as DataHandler for consistency
        # =====================================================================
        dropout_method = params.get('dropout_method', self.params.dropout_method)
        dropout_rate = params.get('dropout_rate', self.params.dropout_rate)
        dropout_mid = params.get('dropout_mid', self.params.dropout_mid)
        dropout_shape = params.get('dropout_shape', self.params.dropout_shape)
        noise_level = params.get('noise_level', self.params.noise_level)
        if batch_corr_at_import:
            dropout_method = 'none'
            noise_level = 0.0

        if dropout_method != 'none':
            from .dropout_simulation import apply_dropout
            
            X, dropout_info = apply_dropout(
                X,
                method=dropout_method,
                dropout_rate=dropout_rate,
                dropout_mid=dropout_mid,
                dropout_shape=dropout_shape,
                noise_level=noise_level
            )
            logger.info(f"Applied {dropout_method} dropout in transform: "
                       f"actual rate = {dropout_info['actual_dropout_rate']:.2%}")

        # Store raw counts (dropout-affected) for algorithms that need raw
        X_original = X.copy()
        if batch_corr_at_import and hasattr(adata, 'layers') and 'original_X' in adata.layers:
            X_original = adata.layers['original_X']

        # Compute size factors using train median (after dropout)
        # Ensure 1D array to avoid matrix/broadcasting issues
        counts_per_cell = np.asarray(np.sum(X_original, axis=1)).ravel()
        median_counts = float(self.params.median_counts) if self.params.median_counts is not None else 0.0
        if median_counts <= 0:
            logger.warning("Training median counts is non-positive; using 1.0 for size factors.")
            median_counts = 1.0
        size_factors = counts_per_cell / median_counts

        # Normalization
        do_normalization = params.get('do_normalization', True)
        if batch_corr_at_import:
            do_normalization = False
        if do_normalization:
            row_sums = np.sum(X, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X = X / row_sums * self.params.target_sum

        # Log transform
        do_log_transform = params.get('do_log_transform', True)
        if batch_corr_at_import:
            do_log_transform = False
        if do_log_transform:
            X = np.log1p(X)

        # Scaling using train mean/std
        do_scaling = params.get('do_scaling', True)
        
        # Skip scaling here if batch correction is active (it will be done after correction)
        # Applying corrected-stats to uncorrected data is invalid
        should_scale_here = do_scaling and not (self.params.do_batch_correction and self._scvi_model is not None)

        if should_scale_here and self.params.mean is not None:
            # Ensure dimensions match (in case of gene mismatch)
            if X.shape[1] == len(self.params.mean):
                X = (X - self.params.mean) / self.params.std
                X = np.clip(X, -self.params.scale_max, self.params.scale_max)
            else:
                logger.warning("Gene dimension mismatch, skipping scaling")

        # =====================================================================
        # Batch Correction: Apply scVI model (trained on TRAIN) to this data
        # NOTE: scVI expects RAW COUNTS, not log-normalized data!
        # =====================================================================
        # =====================================================================
        # Batch Correction: Apply selected method
        # =====================================================================
        if self.params.do_batch_correction:
            method = self.params.batch_correction_method
            batch_key = self.params.batch_correction_batch_key
            
            # SCVI Logic (Model-based)
            if method in {'scvi', 'scvi_dct'} and self._scvi_model is not None:
                
                # Check if batch column exists in this data
                if batch_key and batch_key in adata.obs.columns:
                    try:
                        import scvi
                        
                        # IMPORTANT: scVI expects RAW COUNTS in adata.X
                        # It handles normalization internally via its generative model
                        # Using log-normalized data would bias the model output
                        adata_for_scvi = ad.AnnData(
                            X=X_original.copy(),  # Raw counts (HVG-filtered)
                            obs=adata.obs.copy(),
                            var=adata.var.copy()
                        )

                        # Allow new batch categories when transferring scVI setup
                        try:
                            self._scvi_model._register_manager_for_instance(
                                self._scvi_model.adata_manager.transfer_fields(
                                    adata_for_scvi,
                                    extend_categories=True,
                                )
                            )
                        except Exception as transfer_error:
                            logger.info(
                                f"scVI setup transfer with extend_categories failed: {transfer_error}. "
                                "Proceeding with default transfer behavior."
                            )
                        
                        # Get corrected expression using the trained model
                        # This returns normalized expression (not log-transformed)
                        # Map to reference batch to ensure batch correction
                        corrected = self._scvi_model.get_normalized_expression(
                            adata=adata_for_scvi,
                            library_size=1e4,
                            return_numpy=True,
                            transform_batch=self.params.batch_correction_reference_batch
                        )
                        
                        # Apply log1p transformation to match pipeline
                        corrected = np.log1p(corrected)
                        
                        # Apply scaling using train parameters
                        if do_scaling and self.params.mean is not None and corrected.shape[1] == len(self.params.mean):
                            corrected = (corrected - self.params.mean) / self.params.std
                            corrected = np.clip(corrected, -self.params.scale_max, self.params.scale_max)
                        
                        X = corrected
                        logger.info(f"Applied batch correction to {adata.n_obs} cells (using raw counts)")
                        
                    except Exception as e:
                        logger.warning(f"Batch correction transform failed: {e}. Using uncorrected data.")
                        # Other methods (independent application on this split)
            elif method not in {'scvi', 'scvi_dct'}:
                if batch_key and batch_key in adata.obs.columns:
                    try:
                        from .batch_correction import apply_batch_correction
                        
                        logger.info(f"Applying batch correction ({method}) to split...")

                        is_sysvi = method == 'sysvi'
                        default_n_latent = 15 if is_sysvi else 30
                        default_n_hidden = 256 if is_sysvi else 128
                        default_epochs = 200 if is_sysvi else 100

                        def _or_default(key: str, default: Any) -> Any:
                            value = params.get(key, None)
                            return default if value is None else value

                        # Normalize parameter names for dispatcher functions.
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
                            'prior': _or_default('batch_correction_prior', 'vamp'),
                            'n_prior_components': _or_default('batch_correction_n_prior_components', 5),
                            'embed_categorical_covariates': _or_default('batch_correction_embed_covariates', True),
                            'z_distance_cycle_weight': _or_default('batch_correction_cycle_weight', 5.0),
                            'kl_weight': _or_default('batch_correction_kl_weight', 1.0),
                            'categorical_covariate_keys': params.get('batch_correction_categorical_covariate_keys', None),
                        }
                        
                        # Apply correction using the dispatcher
                        # Note: For Benchmark/Split mode, this runs independently on the split
                        # unless the method naturally handles references (which typical scanpy functions don't)
                        adata, _ = apply_batch_correction(
                            adata,
                            method=method,
                            batch_key=batch_key,
                            labels_key=self.params.batch_correction_labels_key,
                            params=bc_params,
                        )
                        
                        # Update local X reference with corrected expression matrix
                        X = adata.X
                        if hasattr(X, 'toarray'):
                            X = X.toarray()
                            
                    except Exception as e:
                        logger.warning(f"Batch correction ({method}) failed: {e}. Using uncorrected data.")
                else:
                    logger.warning(f"Batch column '{batch_key}' not found. Skipping batch correction.")
            else:
                 # scVI selected but model not available
                 if self.params.do_batch_correction and method in {'scvi', 'scvi_dct'}:
                     logger.warning("scVI model not available (training failed?). Skipping correction.")

        # Create new AnnData
        obs_df = adata.obs.copy()
        obs_df['size_factors'] = size_factors

        adata_processed = ad.AnnData(
            X=X,
            obs=obs_df,
            var=adata.var.copy()
        )

        # Store raw and original layers
        adata_processed.layers['original_X'] = X_original
        adata_processed.raw = ad.AnnData(X_original, var=adata.var.copy())
        
        # Store batch correction info
        if self.params.do_batch_correction:
            dct_used = getattr(self, "_scvi_use_dct", False)
            adata_processed.uns['batch_correction'] = {
                'method': self.params.batch_correction_method,
                'method_details': 'scvi_dct' if dct_used and self.params.batch_correction_method == 'scvi' else self.params.batch_correction_method,
                'batch_key': self.params.batch_correction_batch_key,
                'applied': self._scvi_model is not None,
                'dct_loss_used': dct_used,
            }

        return adata_processed

    def _select_hvg_per_batch_union(
        self,
        adata: Any,
        n_top_genes: int,
        hvg_flavor: str,
        batch_col: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Select top N HVGs from each batch and return their union.
        
        This ensures that genes important in any batch are retained,
        avoiding the loss of batch-specific marker genes.
        
        Args:
            adata: Preprocessed AnnData (normalized, log-transformed)
            n_top_genes: Number of top genes to select per batch
            hvg_flavor: HVG selection method ('seurat', 'seurat_v3', 'cell_ranger')
            batch_col: Column name for batch info (auto-detected if None)
            
        Returns:
            Tuple of (gene_names_array, per_batch_hvg_counts_dict)
        """
        import scanpy as sc
        
        per_batch_hvg_counts = {}  # Track HVGs per batch
        
        # Auto-detect batch column
        if batch_col is None:
            splitter = DatasetSplitter()
            batch_col = splitter._detect_batch_column(adata)
        
        if batch_col is None or batch_col not in adata.obs.columns:
            # No batch column found, fall back to standard HVG selection
            logger.warning("No batch column found, using standard HVG selection")
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=min(n_top_genes, adata.n_vars),
                flavor=hvg_flavor
            )
            hvg_genes = adata.var_names[adata.var['highly_variable']].values
            per_batch_hvg_counts['all'] = len(hvg_genes)
            return hvg_genes, per_batch_hvg_counts
        
        # Get unique batches
        batches = adata.obs[batch_col].unique()
        logger.info(f"Selecting HVGs per batch from {len(batches)} batches")

        # Warn if only one batch (union will be equivalent to single selection)
        if len(batches) < 2:
            logger.warning(f"per_batch_union strategy requires multiple batches. "
                          f"Found only {len(batches)} batch(es). "
                          f"Results will be equivalent to 'train_only' strategy.")
        
        # Collect HVGs from each batch
        all_hvg_genes = set()
        
        for batch_name in batches:
            # Subset to this batch
            batch_mask = adata.obs[batch_col] == batch_name
            adata_batch = adata[batch_mask].copy()
            
            if adata_batch.n_obs < 10:
                logger.warning(f"Batch '{batch_name}' has only {adata_batch.n_obs} cells, skipping HVG")
                per_batch_hvg_counts[str(batch_name)] = 0
                continue
            
            # Adjust n_top_genes if batch has fewer genes with variance
            n_genes_batch = min(n_top_genes, adata_batch.n_vars)
            
            try:
                sc.pp.highly_variable_genes(
                    adata_batch,
                    n_top_genes=n_genes_batch,
                    flavor=hvg_flavor
                )
                batch_hvgs = adata_batch.var_names[adata_batch.var['highly_variable']].values
                all_hvg_genes.update(batch_hvgs)
                per_batch_hvg_counts[str(batch_name)] = len(batch_hvgs)
                logger.info(f"  Batch '{batch_name}': {len(batch_hvgs)} HVGs selected")
            except Exception as e:
                logger.warning(f"HVG selection failed for batch '{batch_name}': {e}")
                per_batch_hvg_counts[str(batch_name)] = 0
                continue
        
        # Convert to numpy array
        hvg_array = np.array(list(all_hvg_genes))

        logger.info(f"Per-batch HVG union: {len(hvg_array)} unique genes from {len(batches)} batches")

        return hvg_array, per_batch_hvg_counts

    def _enrich_hvg_with_celltype_markers(
        self,
        adata: Any,
        hvg_genes: np.ndarray,
        label_col: str = 'Group',
        min_markers_per_celltype: int = 5,
        max_additional_genes: int = 500
    ) -> np.ndarray:
        """
        Ensure all cell types have marker genes represented in HVGs.

        For each cell type, checks if there are enough marker genes in the
        current HVG selection. If not, adds top differentially expressed genes.

        Args:
            adata: Preprocessed AnnData (normalized, log-transformed)
            hvg_genes: Current HVG selection
            label_col: Column name for cell type labels
            min_markers_per_celltype: Minimum markers required per cell type
            max_additional_genes: Maximum number of additional genes to add

        Returns:
            Enriched array of gene names
        """
        import scanpy as sc

        # Check if label column exists
        if label_col not in adata.obs.columns:
            for col in ['labels', 'celltype', 'cell_type', 'CellType']:
                if col in adata.obs.columns:
                    label_col = col
                    break
            else:
                logger.warning("No cell type label column found, skipping HVG enrichment")
                return hvg_genes

        celltypes = adata.obs[label_col].unique()
        n_celltypes = len(celltypes)

        logger.info(f"Enriching HVGs to ensure {min_markers_per_celltype} markers per cell type "
                   f"({n_celltypes} cell types found)")

        # Work on a copy
        adata_work = adata.copy()

        # Compute differentially expressed genes per cell type using rank_genes_groups
        try:
            sc.tl.rank_genes_groups(
                adata_work,
                groupby=label_col,
                method='wilcoxon',
                n_genes=min(100, adata_work.n_vars)  # Get top 100 markers per type
            )
        except Exception as e:
            logger.warning(f"Differential expression analysis failed: {e}")
            return hvg_genes

        # Get current HVG set
        hvg_set = set(hvg_genes)
        additional_genes = set()

        # For each cell type, check if we have enough markers
        for celltype in celltypes:
            # Get top markers for this cell type
            try:
                markers = sc.get.rank_genes_groups_df(adata_work, group=str(celltype))
                top_markers = markers.head(50)['names'].values  # Top 50 markers
            except Exception:
                continue

            # Count how many markers are in current HVGs
            markers_in_hvg = [g for g in top_markers if g in hvg_set]
            n_current = len(markers_in_hvg)

            if n_current < min_markers_per_celltype:
                # Add more markers until we reach the minimum
                needed = min_markers_per_celltype - n_current
                for gene in top_markers:
                    if gene not in hvg_set and gene not in additional_genes:
                        additional_genes.add(gene)
                        needed -= 1
                        if needed <= 0:
                            break

                logger.info(f"  Cell type '{celltype}': {n_current} markers in HVGs, "
                           f"added {min_markers_per_celltype - n_current - needed} more")
            else:
                logger.debug(f"  Cell type '{celltype}': {n_current} markers in HVGs (OK)")

        # Limit additional genes
        if len(additional_genes) > max_additional_genes:
            logger.warning(f"Limiting additional genes from {len(additional_genes)} to {max_additional_genes}")
            additional_genes = set(list(additional_genes)[:max_additional_genes])

        # Combine original HVGs with additional markers
        enriched_hvg = np.array(list(hvg_set | additional_genes))

        logger.info(f"HVG enrichment complete: {len(hvg_genes)} → {len(enriched_hvg)} genes "
                   f"(+{len(additional_genes)} celltype markers)")

        return enriched_hvg

    def fit_transform(self, adata: Any, params: Dict[str, Any] = None) -> Any:
        """Fit on data and transform it."""
        self.fit(adata, params)
        return self.transform(adata, params)


def get_batch_column(adata: Any) -> Optional[str]:
    """
    Utility to detect the batch column in an AnnData object.

    Returns:
        Column name if found, None otherwise
    """
    splitter = DatasetSplitter()
    return splitter._detect_batch_column(adata)


def list_batches(adata: Any, batch_col: Optional[str] = None) -> List[str]:
    """
    List all unique batches in the dataset.

    Args:
        adata: AnnData object
        batch_col: Column name (auto-detected if None)

    Returns:
        List of unique batch names
    """
    if batch_col is None:
        batch_col = get_batch_column(adata)

    if batch_col is None or batch_col not in adata.obs.columns:
        return []

    return list(adata.obs[batch_col].unique())
