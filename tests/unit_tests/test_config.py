"""
Unit tests for core/config.py
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

from core.config import (
    ParamType, HyperparameterConfig, Config,
    PREPROCESSING_PARAMS, get_all_hyperparameters
)


class TestHyperparameterConfig:
    """Tests for HyperparameterConfig class."""

    def test_validate_integer_valid(self):
        """Integer param should validate correctly within range."""
        hp = HyperparameterConfig(
            name='test',
            display_name='Test',
            param_type=ParamType.INTEGER,
            default=10,
            min_value=1,
            max_value=100
        )
        assert hp.validate(10) is True
        assert hp.validate(1) is True
        assert hp.validate(100) is True

    def test_validate_integer_invalid(self):
        """Integer param should reject values outside range."""
        hp = HyperparameterConfig(
            name='test',
            display_name='Test',
            param_type=ParamType.INTEGER,
            default=10,
            min_value=1,
            max_value=100
        )
        assert hp.validate(0) is False
        assert hp.validate(101) is False
        assert hp.validate(10.5) is False  # Not an int

    def test_validate_float_valid(self):
        """Float param should validate correctly."""
        hp = HyperparameterConfig(
            name='test',
            display_name='Test',
            param_type=ParamType.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0
        )
        assert hp.validate(0.5) is True
        assert hp.validate(0.0) is True
        assert hp.validate(1.0) is True
        assert hp.validate(0) is True  # Int should be accepted as float

    def test_validate_float_invalid(self):
        """Float param should reject values outside range."""
        hp = HyperparameterConfig(
            name='test',
            display_name='Test',
            param_type=ParamType.FLOAT,
            default=0.5,
            min_value=0.0,
            max_value=1.0
        )
        assert hp.validate(-0.1) is False
        assert hp.validate(1.1) is False

    def test_validate_choice_valid(self):
        """Choice param should accept valid choices."""
        hp = HyperparameterConfig(
            name='test',
            display_name='Test',
            param_type=ParamType.CHOICE,
            default='a',
            choices=['a', 'b', 'c']
        )
        assert hp.validate('a') is True
        assert hp.validate('b') is True
        assert hp.validate('c') is True

    def test_validate_choice_invalid(self):
        """Choice param should reject invalid choices."""
        hp = HyperparameterConfig(
            name='test',
            display_name='Test',
            param_type=ParamType.CHOICE,
            default='a',
            choices=['a', 'b', 'c']
        )
        assert hp.validate('d') is False
        assert hp.validate(1) is False

    def test_to_dict(self):
        """to_dict should return complete dictionary."""
        hp = HyperparameterConfig(
            name='test_param',
            display_name='Test Parameter',
            param_type=ParamType.INTEGER,
            default=42,
            description='A test parameter',
            min_value=1,
            max_value=100,
            category='Testing'
        )
        d = hp.to_dict()
        
        assert d['name'] == 'test_param'
        assert d['display_name'] == 'Test Parameter'
        assert d['param_type'] == 'integer'
        assert d['default'] == 42
        assert d['description'] == 'A test parameter'
        assert d['min_value'] == 1
        assert d['max_value'] == 100
        assert d['category'] == 'Testing'


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = Config()
        assert config.app_name == "scDeepCluster Analysis Suite"
        assert config.version == "1.0.0"
        assert config.seed == 42
        assert config.device == "auto"

    def test_save_and_load(self):
        """Config should save and load correctly."""
        config = Config(seed=123)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            config.save(temp_path)
            loaded = Config.load(temp_path)
            
            assert loaded.seed == 123
            assert loaded.app_name == config.app_name
        finally:
            temp_path.unlink()

    def test_ensure_dirs(self):
        """ensure_dirs should create directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Config(
                data_dir=Path(tmpdir) / 'data',
                results_dir=Path(tmpdir) / 'results',
                models_dir=Path(tmpdir) / 'models'
            )
            config.ensure_dirs()
            
            assert config.data_dir.exists()
            assert config.results_dir.exists()
            assert config.models_dir.exists()


class TestPreprocessingParams:
    """Tests for preprocessing parameters."""

    def test_preprocessing_params_exist(self):
        """PREPROCESSING_PARAMS should contain expected parameters."""
        param_names = [p.name for p in PREPROCESSING_PARAMS]
        
        assert 'n_top_genes' in param_names
        assert 'min_genes_per_cell' in param_names
        assert 'target_sum' in param_names

    def test_get_all_hyperparameters(self):
        """get_all_hyperparameters should return all parameters."""
        all_params = get_all_hyperparameters()
        
        assert len(all_params) > 10
        assert all(isinstance(p, HyperparameterConfig) for p in all_params)
