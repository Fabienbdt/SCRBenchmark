"""
Unit tests for UI CLI command generation.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

pytest.importorskip("streamlit")

from gui import algorithm_config


class _DummyAlgo:
    @staticmethod
    def get_hyperparameters():
        return []


def _base_state() -> dict:
    return {
        "uploaded_file_path": "data/sample.h5ad",
        "selected_algorithms": ["sc_mae"],
        "algorithm_params": {},
        "preprocessing_params": {},
        "validation_optimization": {
            "enable_lr_optimization": True,
            "lr_optimization_metric": "NMI",
            "lr_optimization_repeats": 1,
            "lr_optimization_scales": [100.0, 10.0, 1.0, 0.1],
        },
        "output_dir": "results",
    }


@pytest.fixture
def patch_registry(monkeypatch):
    monkeypatch.setattr(algorithm_config.AlgorithmRegistry, "get", staticmethod(lambda _name: _DummyAlgo))


def test_standard_mode_does_not_emit_lr_optimization_params(patch_registry):
    """No validation split in standard mode: LR optimization CLI params must be omitted."""
    state = _base_state()
    state.update(
        {
            "benchmark_configured": True,
            "benchmark_setup": {"mode": "standard", "original_settings": {}},
            "benchmark_settings": {},
        }
    )

    cmd = algorithm_config._generate_cli_command(compact=True, state_source=state)

    assert "lr_optimization_scales" not in cmd
    assert "enable_lr_optimization" not in cmd
    assert "lr_optimization_metric" not in cmd
    assert "lr_optimization_repeats" not in cmd


def test_benchmark_mode_with_validation_emits_lr_optimization_params(patch_registry):
    """Benchmark mode with validation split should keep LR optimization params."""
    state = _base_state()
    state.update(
        {
            "benchmark_configured": True,
            "benchmark_setup": {
                "mode": "benchmark",
                "original_settings": {
                    "mode": "stratified",
                    "train_ratio": 0.8,
                    "val_ratio": 0.1,
                    "use_validation": True,
                },
            },
            "benchmark_settings": {},
        }
    )

    cmd = algorithm_config._generate_cli_command(compact=True, state_source=state)

    assert "sc_mae:lr_optimization_scales=[100.0,10.0,1.0,0.1]" in cmd
    assert "sc_mae:enable_lr_optimization=true" in cmd


def test_benchmark_mode_without_validation_omits_lr_optimization_params(patch_registry):
    """If validation is disabled, LR optimization params must be omitted."""
    state = _base_state()
    state.update(
        {
            "benchmark_configured": True,
            "benchmark_setup": {
                "mode": "benchmark",
                "original_settings": {
                    "mode": "stratified",
                    "train_ratio": 0.8,
                    "val_ratio": 0.0,
                    "use_validation": False,
                },
            },
            "benchmark_settings": {},
        }
    )

    cmd = algorithm_config._generate_cli_command(compact=True, state_source=state)

    assert "lr_optimization_scales" not in cmd
    assert "enable_lr_optimization" not in cmd


def test_preprocessing_params_are_exported_when_data_is_preprocessed(patch_registry):
    """When preprocessing has been run in UI, command should emit explicit preprocessing flags."""
    state = _base_state()
    state.update(
        {
            "data_preprocessed": True,
            "preprocessing_params": {
                "n_top_genes": 5000,
                "min_genes_per_cell": 200,
                "min_cells_per_gene": 3,
                "target_sum": 10000,
                "scale_max_value": 10.0,
                "hvg_flavor": "seurat",
                "hvg_strategy": "train_only",
                "dropout_method": "simple",
                "dropout_rate": 0.25,
                "noise_level": 0.1,
            },
        }
    )

    cmd = algorithm_config._generate_cli_command(compact=True, state_source=state)

    assert "--n-top-genes 5000" in cmd
    assert "--min-genes-per-cell 200" in cmd
    assert "--target-sum 10000" in cmd
    assert "--dropout-method simple" in cmd
    assert "--dropout-rate 0.25" in cmd
    assert "--noise-level 0.1" in cmd
