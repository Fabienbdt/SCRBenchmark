"""
Unit tests for core/algorithm_registry.py
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

from core.algorithm_registry import (
    BaseAlgorithm, AlgorithmRegistry, AlgorithmInfo
)
from core.config import HyperparameterConfig, ParamType


class MockAlgorithm(BaseAlgorithm):
    """Mock algorithm for testing."""

    @classmethod
    def get_info(cls):
        return AlgorithmInfo(
            name='mock_algo',
            display_name='Mock Algorithm',
            description='A mock algorithm for testing',
            category='testing'
        )

    @classmethod
    def get_hyperparameters(cls):
        return [
            HyperparameterConfig(
                name='param1',
                display_name='Parameter 1',
                param_type=ParamType.INTEGER,
                default=10
            )
        ]

    def fit(self, data, labels=None):
        self._labels = [0] * len(data) if hasattr(data, '__len__') else [0]
        self._embeddings = None
        self._fitted = True
        return self

    def predict(self, data=None):
        return self._labels


class TestBaseAlgorithm:
    """Tests for BaseAlgorithm class."""

    def test_init_with_params(self):
        """Algorithm should store params correctly."""
        algo = MockAlgorithm({'param1': 20, 'device': 'cpu'})
        assert algo.params['param1'] == 20
        assert algo.params['device'] == 'cpu'

    def test_init_without_params(self):
        """Algorithm should have empty params dict when none provided."""
        algo = MockAlgorithm()
        assert algo.params == {}

    def test_is_fitted_initially_false(self):
        """is_fitted should be False before fitting."""
        algo = MockAlgorithm()
        assert algo.is_fitted is False

    def test_get_device_cpu(self):
        """get_device should return 'cpu' when explicitly set."""
        algo = MockAlgorithm({'device': 'cpu'})
        assert algo.get_device() == 'cpu'

    def test_get_device_auto_no_cuda_no_mps(self):
        """get_device with 'auto' should return 'cpu' when no GPU available."""
        algo = MockAlgorithm({'device': 'auto'})
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                assert algo.get_device() == 'cpu'

    def test_get_device_auto_with_cuda(self):
        """get_device with 'auto' should return 'cuda' when CUDA available."""
        algo = MockAlgorithm({'device': 'auto'})
        
        with patch('torch.cuda.is_available', return_value=True):
            assert algo.get_device() == 'cuda'

    def test_get_device_auto_with_mps(self):
        """get_device with 'auto' should return 'mps' when MPS available but no CUDA."""
        algo = MockAlgorithm({'device': 'auto'})
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                assert algo.get_device() == 'mps'

    def test_get_device_mps_explicit(self):
        """get_device should return 'mps' when explicitly set and available."""
        algo = MockAlgorithm({'device': 'mps'})
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                assert algo.get_device() == 'mps'

    def test_get_device_mps_fallback_to_cpu(self):
        """get_device should fall back to CPU when MPS not available."""
        algo = MockAlgorithm({'device': 'mps'})
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                assert algo.get_device() == 'cpu'

    def test_get_device_fallback(self):
        """get_device should fall back to auto-detection when no param."""
        algo = MockAlgorithm()
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = algo.get_device()
                assert device == 'cpu'

    def test_fit_predict(self):
        """fit_predict should fit and return predictions."""
        algo = MockAlgorithm()
        data = [[1, 2], [3, 4], [5, 6]]
        
        preds = algo.fit_predict(data)
        
        assert algo.is_fitted is True
        assert len(preds) == 3


class TestAlgorithmRegistry:
    """Tests for AlgorithmRegistry class."""

    def test_register_algorithm(self):
        """Register decorator should add algorithm to registry."""
        # Clear registry first
        original_algos = AlgorithmRegistry._algorithms.copy()
        
        try:
            # Register our mock
            AlgorithmRegistry.register(MockAlgorithm)
            
            assert 'mock_algo' in AlgorithmRegistry._algorithms
            assert AlgorithmRegistry._algorithms['mock_algo'] == MockAlgorithm
        finally:
            # Restore original registry
            AlgorithmRegistry._algorithms = original_algos

    def test_get_algorithm(self):
        """get should return registered algorithm."""
        original_algos = AlgorithmRegistry._algorithms.copy()
        
        try:
            AlgorithmRegistry.register(MockAlgorithm)
            algo_class = AlgorithmRegistry.get('mock_algo')
            
            assert algo_class == MockAlgorithm
        finally:
            AlgorithmRegistry._algorithms = original_algos

    def test_get_nonexistent(self):
        """get should return None for non-existent algorithm."""
        result = AlgorithmRegistry.get('nonexistent_algorithm_xyz')
        assert result is None

    def test_list_algorithms(self):
        """list_algorithms should return AlgorithmInfo objects."""
        algos = AlgorithmRegistry.list_algorithms()
        
        assert isinstance(algos, list)
        # At least one algorithm should be registered
        if len(algos) > 0:
            assert all(isinstance(a, AlgorithmInfo) for a in algos)

    def test_create_algorithm(self):
        """create should instantiate algorithm with params."""
        original_algos = AlgorithmRegistry._algorithms.copy()
        
        try:
            AlgorithmRegistry.register(MockAlgorithm)
            instance = AlgorithmRegistry.create('mock_algo', {'param1': 42})
            
            assert isinstance(instance, MockAlgorithm)
            assert instance.params['param1'] == 42
        finally:
            AlgorithmRegistry._algorithms = original_algos

    def test_create_nonexistent_raises(self):
        """create should raise ValueError for non-existent algorithm."""
        with pytest.raises(ValueError):
            AlgorithmRegistry.create('nonexistent_algorithm_xyz')

    def test_get_categories(self):
        """get_categories should return list of category strings."""
        categories = AlgorithmRegistry.get_categories()
        
        assert isinstance(categories, list)
        if len(categories) > 0:
            assert all(isinstance(c, str) for c in categories)

    def test_report_pca_registered_and_removed_algorithms_hidden(self):
        """The Table 1 PCA entry is available and removed baselines are hidden."""
        import algorithms  # noqa: F401

        assert AlgorithmRegistry.get('pca') is not None
        assert AlgorithmRegistry.get('simple_autoencoder') is None
        assert AlgorithmRegistry.get('scvi_dct') is None


class TestAlgorithmInfo:
    """Tests for AlgorithmInfo dataclass."""

    def test_default_values(self):
        """AlgorithmInfo should have sensible defaults."""
        info = AlgorithmInfo(
            name='test',
            display_name='Test',
            description='Test algorithm',
            category='testing'
        )
        
        assert info.requires_gpu is False
        assert info.supports_labels is True
        assert info.preprocessing_notes == ""
        assert info.has_internal_preprocessing is False
        assert info.recommended_data == "preprocessed"
