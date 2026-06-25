"""
Unit tests for the versioned benchmark protocol registry.
"""

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "scrbenchmark"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protocols import registry


def test_loads_report_protocols():
    specs = registry.load_protocol_specs()

    assert "baron_transductive" in specs
    assert "common8_methods_harmony" in specs
    assert "loss_transfer_report" in specs
    assert "inductive_report_splits" in specs


def test_common8_protocol_expands_to_eight_editable_configs():
    spec = registry.get_protocol_spec("common8_methods_harmony")
    configs = registry.protocol_to_customize_configs(spec)

    assert len(configs) == 8
    assert all(config["selected_report_methods"] for config in configs)
    assert "scMAE+Harmony" in configs[0]["selected_report_methods"]
    assert configs[0]["report_method_n_pcs"] == 50


def test_loss_transfer_protocol_matches_report_defaults():
    spec = registry.get_protocol_spec("loss_transfer_report")
    configs = registry.protocol_to_customize_configs(spec)
    first = configs[0]

    assert len(configs) == 5
    assert first["manual_protocols"]["enabled"] is True
    assert first["manual_protocols"]["selected_protocols"] == ["loss_transfer"]
    assert first["manual_protocols"]["seeds"] == "42-46"
    assert first["manual_protocols"]["loss_transfer"]["variants"] == [
        "baseline",
        "weighted",
        "density_only",
        "kmeans",
        "triplet",
    ]
    assert "warmup_epochs=55" in first["manual_protocols"]["loss_transfer"]["weight_params"]
    assert first["n_repeats"] == 5


def test_sweep_expansion_sets_nested_values_and_unique_output_dirs():
    base = {
        "name": "Sweep Test",
        "output_dir": "results/sweep_test",
        "selected_algorithms": ["sc_mae"],
        "algorithm_params": {},
        "preprocessing_params": {"n_top_genes": 2000},
        "manual_protocols": {"loss_transfer": {"weight_params": ""}},
    }

    expanded = registry.expand_sweep_configs(
        base,
        "\n".join(
            [
                "execution.seed=42,43",
                "preprocessing.n_top_genes=1000,2000",
                "algorithm.sc_mae.lr=0.001",
                "manual.loss_transfer.weight_params.warmup_epochs=55",
            ]
        ),
    )

    assert len(expanded) == 4
    assert expanded[0]["seed"] == 42
    assert expanded[-1]["seed"] == 43
    assert expanded[0]["preprocessing_params"]["n_top_genes"] == 1000
    assert expanded[0]["algorithm_params"]["sc_mae"]["lr"] == 0.001
    assert "warmup_epochs=55" in expanded[0]["manual_protocols"]["loss_transfer"]["weight_params"]
    assert len({config["output_dir"] for config in expanded}) == 4


def test_result_collection_and_summary(tmp_path):
    run_dir = tmp_path / "scMAE" / "seed_42"
    results_dir = run_dir / "results"
    config_dir = run_dir / "config"
    results_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    (config_dir / "method_run_manifest.json").write_text(
        '{"method": "scMAE", "args": {"dataset_key": "demo", "seed": 42}}',
        encoding="utf-8",
    )
    pd.DataFrame([{"NMI": 0.8, "ARI": 0.7, "ACC": 0.9}]).to_csv(
        results_dir / "analysis_results.csv",
        index=False,
    )

    rows = registry.collect_result_rows(tmp_path)
    summary = registry.summarize_results(rows)

    assert len(rows) == 1
    assert rows.iloc[0]["method"] == "scMAE"
    assert rows.iloc[0]["dataset_key"] == "demo"
    assert "NMI_mean" in summary.columns
    assert summary.iloc[0]["NMI_mean"] == 0.8
