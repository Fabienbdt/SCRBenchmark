"""
Unit tests for the Customize Benchmark report presets and report-method command generation.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "scrbenchmark"))

pytest.importorskip("streamlit")

from gui import customize_benchmark


def test_report_method_commands_repeat_without_output_collision():
    config = customize_benchmark._create_default_config("Report Method Test")
    config.update(
        {
            "uploaded_file_path": "data/stable_generalist/baron_human_pancreas.h5ad",
            "dataset_key": "baron_human_pancreas",
            "label_key": "label",
            "batch_key": "batch",
            "n_labels": 14,
            "selected_algorithms": [],
            "selected_report_methods": ["Harmony"],
            "n_repeats": 2,
            "seed": 42,
            "output_dir": "results/report_test",
        }
    )

    commands = customize_benchmark._generate_report_method_commands(config)

    assert len(commands) == 2
    assert "--dataset-key baron_human_pancreas" in commands[0]
    assert "--seed 42" in commands[0]
    assert "seed_42" in commands[0]
    assert "--seed 43" in commands[1]
    assert "seed_43" in commands[1]


def test_loss_transfer_report_preset_is_complete_and_editable():
    configs = customize_benchmark._create_report_preset_configs("loss_transfer_report")

    assert len(configs) == len(customize_benchmark.LOSS_TRANSFER_DATASET_KEYS)
    first = configs[0]
    protocols = first["manual_protocols"]
    loss_cfg = protocols["loss_transfer"]

    assert protocols["enabled"] is True
    assert protocols["selected_protocols"] == ["loss_transfer"]
    assert protocols["seeds"] == "42-46"
    assert loss_cfg["methods"] == ["scMAE", "scDeepCluster", "DESC"]
    assert loss_cfg["variants"] == ["baseline", "weighted", "density_only", "kmeans", "triplet"]
    assert "warmup_epochs=55" in loss_cfg["weight_params"]
    assert first["n_repeats"] == 5
