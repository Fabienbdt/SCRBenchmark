import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reproduction"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "reproduction" / "adapters"))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "scraw_inductive" / "src"))

from run_method import build_command  # noqa: E402
from run_scraw_external import (  # noqa: E402
    apply_manual_params,
    apply_preprocessing_args,
    prepare_output_dir,
)
from scrbenchmark.methods import get_method_spec  # noqa: E402
from scraw.config import ScRAWConfig, load_config  # noqa: E402
from scraw.metrics import compute_metrics  # noqa: E402
from scraw.model import resolve_device  # noqa: E402
from scraw.pipeline import _detect_batch_key, _detect_label_key  # noqa: E402
from scraw.preprocessing import PreprocessingState, load_preprocessing_state  # noqa: E402
from scraw.trainer import ScRAWTrainer  # noqa: E402


def _runner_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        method="scRAW",
        data=str(tmp_path / "toy.h5ad"),
        output=str(tmp_path / "out"),
        dataset_key="toy",
        label_key="cell_type",
        batch_key="donor",
        n_labels=3,
        seed=7,
        device="cpu",
        scib_n_jobs=1,
        python_bin="python3",
        n_top_genes=321,
        min_genes_per_cell=12,
        max_genes_per_cell=3456,
        min_cells_per_gene=2,
        target_sum=12345.0,
        scale_max_value=8.0,
        hvg_flavor="seurat_v3",
        n_pcs=50,
        harmony_max_iter=10,
        harmony_nclust=50,
        resolutions="0.5,1.0",
        selection_expected_n_classes=0,
        scraw_preset="baron",
        param=["scraw:training.epochs=3"],
        overwrite=True,
        verbose=True,
        dry_run=True,
    )


def test_registered_scraw_command_forwards_shared_runner_options(tmp_path):
    command = build_command(get_method_spec("scRAW"), _runner_args(tmp_path))

    expected_values = {
        "--n-top-genes": "321",
        "--min-genes-per-cell": "12",
        "--max-genes-per-cell": "3456",
        "--min-cells-per-gene": "2",
        "--target-sum": "12345.0",
        "--scale-max-value": "8.0",
        "--hvg-flavor": "seurat_v3",
    }
    for flag, expected in expected_values.items():
        assert command[command.index(flag) + 1] == expected
    assert "--overwrite" in command
    assert "--verbose" in command
    assert command[-2:] == ["--param", "scraw:training.epochs=3"]


def test_adapter_applies_preprocessing_and_typed_manual_overrides(tmp_path):
    config = ScRAWConfig()
    args = _runner_args(tmp_path)

    apply_preprocessing_args(config, args)
    applied = apply_manual_params(
        config,
        [
            "scraw:training.epochs=4",
            "scraw:hidden_layers=64,32",
            "scraw:rare_triplet_weight=0",
            "another_method:epochs=99",
        ],
    )

    assert config.preprocessing.n_top_genes == 321
    assert config.preprocessing.hvg_flavor == "seurat_v3"
    assert config.training.epochs == 4
    assert config.model.hidden_layers == [64, 32]
    assert config.triplet.weight == 0.0
    assert config.triplet.enabled is False
    assert applied["training.epochs"] == 4


def test_adapter_overwrite_preserves_generic_runner_metadata(tmp_path):
    for relative in (
        "results/stale.csv",
        "figures/stale.png",
        "models/stale.pt",
        "config/config_used.json",
        "config/method_run_manifest.json",
        "logs/run_method.log",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale", encoding="utf-8")

    prepare_output_dir(tmp_path, overwrite=True)

    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "figures").exists()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "config" / "config_used.json").exists()
    assert (tmp_path / "config" / "method_run_manifest.json").exists()
    assert (tmp_path / "logs" / "run_method.log").exists()


def test_yaml_config_is_loaded_as_yaml(tmp_path):
    config_path = tmp_path / "scraw.yaml"
    config_path.write_text(
        "training:\n  epochs: 5\nmodel:\n  hidden_layers: [16, 8]\nruntime:\n  device: cpu\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.training.epochs == 5
    assert config.model.hidden_layers == [16, 8]
    assert config.runtime.device == "cpu"


def test_explicit_label_and_batch_keys_do_not_silently_fall_back():
    adata = SimpleNamespace(
        obs=pd.DataFrame({"cell_type": ["a", "b"], "donor": ["d1", "d2"]})
    )

    assert _detect_label_key(adata, None) == "cell_type"
    assert _detect_batch_key(adata, None) == "donor"
    with pytest.raises(ValueError, match="Configured label key 'missing'"):
        _detect_label_key(adata, "missing")
    with pytest.raises(ValueError, match="Configured batch key 'missing'"):
        _detect_batch_key(adata, "missing")


def test_device_resolution_accepts_indexed_case_insensitive_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert str(resolve_device("CUDA:1")) == "cuda:1"
    assert str(resolve_device("Cpu")) == "cpu"
    with pytest.raises(ValueError, match="index 2 is unavailable"):
        resolve_device("cuda:2")
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("gpu")


def test_trainer_rejects_unimplemented_objectives():
    config = ScRAWConfig()
    config.runtime.device = "cpu"
    config.training.reconstruction_distribution = "nb"
    with pytest.raises(ValueError, match="supports only.*mse"):
        ScRAWTrainer(config)

    config = ScRAWConfig()
    config.runtime.device = "cpu"
    config.batch_correction.mmd_weight = 0.1
    with pytest.raises(ValueError, match="mmd_weight is not implemented"):
        ScRAWTrainer(config)


def test_metrics_include_balanced_rare_accuracy():
    labels_true = np.asarray(["common"] * 184 + ["rare_a"] * 9 + ["rare_b"] * 7)
    labels_pred = np.asarray(["cluster_0"] * 184 + ["cluster_1"] * 16)

    metrics = compute_metrics(labels_true, labels_pred)

    assert metrics["BalancedRareACC"] == pytest.approx(0.5)
    assert metrics["RareACC"] == pytest.approx(9 / 16)

    no_rare = compute_metrics(
        np.asarray(["a"] * 50 + ["b"] * 50),
        np.asarray([0] * 50 + [1] * 50),
    )
    assert math.isnan(no_rare["BalancedRareACC"])


def test_transductive_pipeline_persists_filtered_cell_mapping_and_state(tmp_path, monkeypatch):
    from scraw import pipeline

    obs = pd.DataFrame(
        {"cell_type": ["a", "a", "b", "b"]},
        index=["cell_4", "cell_2", "cell_8", "cell_1"],
    )
    processed = SimpleNamespace(
        obs=obs,
        obs_names=obs.index,
        n_obs=4,
        n_vars=2,
        X=np.asarray([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float32),
    )
    state = PreprocessingState(
        var_names=["g1", "g2"],
        mean=np.asarray([1.0, 2.0], dtype=np.float32),
        std=np.asarray([0.5, 0.75], dtype=np.float32),
        params={"n_top_genes": 2},
        looks_processed=False,
        scale_max_value=10.0,
    )
    result = SimpleNamespace(
        embeddings=np.zeros((4, 2), dtype=np.float32),
        labels=np.asarray([0, 0, 1, 1]),
        pseudo_labels=np.asarray([0, 0, 1, 1]),
        cell_weights=np.asarray([1.0, 1.1, 0.9, 1.2]),
        device="cpu",
        loss_history=[],
        model=None,
    )

    class FakeTrainer:
        def __init__(self, config):
            self.config = config

        def fit(self, X, labels=None, batch_ids=None):
            return result

    fake_scanpy = SimpleNamespace(read_h5ad=lambda path: object())
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    monkeypatch.setattr(pipeline, "fit_preprocess_adata", lambda adata, params: (processed, state))
    monkeypatch.setattr(pipeline, "ScRAWTrainer", FakeTrainer)
    monkeypatch.setattr(pipeline, "compute_metrics", lambda **kwargs: {})

    config = ScRAWConfig()
    config.data.data_path = str(tmp_path / "input.h5ad")
    config.data.output_dir = str(tmp_path / "output")
    config.data.label_key = "cell_type"
    config.batch_correction.enabled = False
    config.outputs.save_model = False
    config.outputs.save_figures = False

    pipeline.run_pipeline(config)

    results_dir = tmp_path / "output" / "results"
    assert np.load(results_dir / "obs_names.npy", allow_pickle=False).tolist() == list(obs.index)
    mapping = pd.read_csv(results_dir / "cell_assignments.csv")
    assert mapping["cell_id"].tolist() == list(obs.index)
    assert mapping["predicted_label"].tolist() == [0, 0, 1, 1]
    restored = load_preprocessing_state(tmp_path / "output" / "models" / "preprocessing_state.npz")
    assert restored.var_names == ["g1", "g2"]


def test_transductive_checkpoint_cli_forwards_label_key(monkeypatch, tmp_path):
    import run_scraw_from_weights
    import scraw

    captured = {}

    def fake_inference(**kwargs):
        captured.update(kwargs)
        return {"output_dir": str(tmp_path / "out"), "mode": "inference_only"}

    monkeypatch.setattr(scraw, "run_inference_from_checkpoint", fake_inference)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_scraw_from_weights.py",
            "--config",
            "config.yaml",
            "--checkpoint",
            "weights.pt",
            "--data",
            "cells.h5ad",
            "--output",
            str(tmp_path / "out"),
            "--label-key",
            "cell_type",
        ],
    )

    assert run_scraw_from_weights.main() == 0
    assert captured["label_key"] == "cell_type"
