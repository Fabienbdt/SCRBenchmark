"""
Unit tests for utils/data_handler.py
"""

import pytest
import sys
from pathlib import Path
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

from utils.data_handler import DataHandler


class TestDataHandler:
    """Tests for DataHandler class."""

    def test_supported_formats(self):
        """get_supported_formats should return list of extensions."""
        formats = DataHandler.get_supported_formats()
        
        assert '.h5ad' in formats
        assert '.h5' in formats
        assert '.csv' in formats
        assert '.mtx' in formats

    def test_init(self):
        """DataHandler should initialize correctly."""
        handler = DataHandler()
        
        assert handler.adata is None
        assert handler.labels is None

    def test_load_h5ad(self, synthetic_data_path):
        """DataHandler should load H5AD files."""
        handler = DataHandler()
        handler.load(synthetic_data_path)
        
        assert handler.adata is not None
        assert handler.adata.n_obs > 0
        assert handler.adata.n_vars > 0

    def test_get_info_after_load(self, synthetic_data_path):
        """get_info should return data statistics after loading."""
        handler = DataHandler()
        handler.load(synthetic_data_path)
        
        info = handler.get_info()
        
        assert 'n_cells' in info
        assert 'n_genes' in info
        assert info['n_cells'] > 0
        assert info['n_genes'] > 0

    def test_get_data(self, synthetic_data_path):
        """get_data should return AnnData object."""
        handler = DataHandler()
        handler.load(synthetic_data_path)
        
        data = handler.get_data()
        
        assert data is not None
        assert hasattr(data, 'X')
        assert hasattr(data, 'obs')
        assert hasattr(data, 'var')


class TestDataHandlerPreprocessing:
    """Tests for DataHandler preprocessing."""

    def test_preprocess_basic(self, synthetic_data_path):
        """Basic preprocessing should work."""
        handler = DataHandler()
        handler.load(synthetic_data_path)
        
        params = {
            'do_cell_filtering': True,
            'min_genes_per_cell': 10,
            'max_genes_per_cell': 50000,
            'do_gene_filtering': True,
            'min_cells_per_gene': 1,
            'do_normalization': True,
            'target_sum': 10000,
            'do_log_transform': True,
            'do_hvg': False,  # Skip HVG for speed
            'do_scaling': True,
            'scale_max_value': 10.0,
            'dropout_method': 'none'
        }
        
        handler.preprocess(params)
        
        # Data should still be valid after preprocessing
        data = handler.get_data()
        assert data is not None
        assert data.n_obs > 0
        assert data.n_vars > 0


class TestDataHandlerLabels:
    """Tests for DataHandler label handling."""

    def test_get_labels(self, synthetic_data_path):
        """get_labels should return labels if available."""
        handler = DataHandler()
        handler.load(synthetic_data_path)
        
        labels = handler.get_labels()
        
        # Labels should be available in synthetic data
        if labels is not None:
            assert len(labels) == handler.adata.n_obs
