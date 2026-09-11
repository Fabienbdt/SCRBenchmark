"""Exercise public commands outside the source checkout."""

import json

import numpy as np
import pandas as pd
import pytest


def _run_pca(run_cli, data, output, *extra):
    return run_cli(
        "run", "--data", data, "--algorithms", "pca", "--param", "pca:clustering_method=kmeans",
        "--param", "pca:n_pca_components=5", "--n-clusters", "3", "--label-col", "Group",
        "--skip-preprocessing", "--device", "cpu", "--seed", "42", "--no-scib-metrics",
        "--output", output, "--no-timestamp", "--save-labels", "--save-embeddings", *extra,
    )


def test_algorithm_discovery_uses_installed_package(run_cli):
    result = run_cli("list-algorithms", "--json")
    assert result.returncode == 0, result.stderr
    algorithms = json.loads(result.stdout)
    assert {"pca", "scdeepcluster", "sc_mae", "sccdcg", "scname"} <= {row["name"] for row in algorithms}


def test_cpu_benchmark_exports_reproducible_results(run_cli, toy_h5ad, tmp_path):
    exported_labels, embeddings = [], []
    for output in [tmp_path / "first output", tmp_path / "second output"]:
        result = _run_pca(run_cli, toy_h5ad, output)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = pd.read_csv(output / "results" / "results.csv")
        assert len(summary) == 1
        config = json.loads((output / "config" / "config_used.json").read_text())
        assert config["execution"]["random_seed"] == 42
        assert config["execution"]["device"] == "cpu"
        label_files = list((output / "results" / "labels").glob("labels_*.csv"))
        assert len(label_files) == 1
        labels = pd.read_csv(label_files[0])
        assert len(labels) == 60
        assert labels["predicted_label"].nunique() == 3
        assert labels["cell_id"].tolist() == [f"cell_{i:03d}" for i in range(60)]
        exported_labels.append(labels["predicted_label"].to_numpy())
        embedding_files = list((output / "results" / "embeddings").glob("*.npy"))
        assert len(embedding_files) == 1
        embedding = np.load(embedding_files[0])
        assert embedding.shape == (60, 5)
        assert np.isfinite(embedding).all()
        embeddings.append(embedding)
    np.testing.assert_array_equal(*exported_labels)
    np.testing.assert_allclose(*embeddings, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("extra, message", [
    (("--algorithms", "does_not_exist"), "Unknown algorithm"),
    (("--n-repeats", "0"), "n-repeats"),
    (("--n-clusters", "1000"), "failed"),
    (("--label-col", "missing_labels"), "missing_labels"),
])
def test_invalid_run_exits_with_failure(run_cli, toy_h5ad, tmp_path, extra, message):
    result = _run_pca(run_cli, toy_h5ad, tmp_path / "invalid", *extra)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message.lower() in (result.stdout + result.stderr).lower()


def test_missing_input_exits_with_failure(run_cli, tmp_path):
    result = _run_pca(run_cli, tmp_path / "missing.h5ad", tmp_path / "output")
    assert result.returncode != 0
    assert "missing.h5ad" in result.stderr


def test_demo_can_be_created_and_refuses_to_overwrite(run_cli, tmp_path):
    import anndata as ad
    destination = tmp_path / "demo.h5ad"
    result = run_cli("demo", "--output", destination)
    assert result.returncode == 0, result.stderr
    data = ad.read_h5ad(destination)
    assert data.shape == (120, 60)
    assert data.obs["Group"].nunique() == 3
    original = destination.read_bytes()
    repeat = run_cli("demo", "--output", destination)
    assert repeat.returncode != 0
    assert destination.read_bytes() == original


def test_explicit_label_column_is_used_for_export(run_cli, toy_h5ad, tmp_path):
    import anndata as ad
    data = ad.read_h5ad(toy_h5ad)
    data.obs["requested"] = "requested_" + data.obs["Group"].astype(str)
    data.write_h5ad(toy_h5ad)
    output = tmp_path / "explicit labels"
    result = _run_pca(run_cli, toy_h5ad, output, "--label-col", "requested")
    assert result.returncode == 0, result.stdout + result.stderr
    labels = pd.read_csv(next((output / "results" / "labels").glob("labels_*.csv")))
    assert labels["true_label"].tolist() == data.obs["requested"].tolist()


def test_training_failure_writes_a_failure_manifest(run_cli, toy_h5ad, tmp_path):
    output = tmp_path / "failed"
    result = _run_pca(run_cli, toy_h5ad, output, "--n-clusters", "1000")
    assert result.returncode != 0
    status = json.loads((output / "config" / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_runs"] == 0
    assert "n_clusters" in status["failures"][0]["error"]


@pytest.mark.parametrize("contents", ["[1, 2]", "null", '{"data": "invalid"}'])
def test_malformed_configuration_is_reported_without_traceback(run_cli, tmp_path, contents):
    config = tmp_path / "invalid.json"
    config.write_text(contents)
    result = run_cli("run", "--config", config)
    assert result.returncode != 0
    assert "configuration" in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_gui_generated_command_executes_outside_checkout(toy_h5ad, tmp_path, cli_env):
    import shlex
    import subprocess
    from scrbenchmark import algorithms
    from scrbenchmark.gui.algorithm_config import _generate_cli_command
    output = tmp_path / "GUI generated output"
    state = {
        "uploaded_file_path": str(toy_h5ad), "selected_algorithms": ["pca"],
        "algorithm_params": {"pca": {"clustering_method": "kmeans", "n_clusters": 3, "n_pca_components": 5}},
        "data_preprocessed": True, "output_dir": str(output), "device": "cpu",
        "preprocessing_params": {"min_genes_per_cell": 1, "min_cells_per_gene": 1, "do_hvg": False},
    }
    command = _generate_cli_command(compact=True, state_source=state)
    result = subprocess.run(
        [*shlex.split(command), "--no-timestamp", "--no-scib-metrics", "--save-labels", "--label-col", "Group"], cwd=tmp_path,
        env=cli_env, text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads((output / "config" / "run_status.json").read_text())
    assert status["status"] == "completed"
    assert status["completed_runs"] == 1
    labels = pd.read_csv(next((output / "results" / "labels").glob("labels_*.csv")))
    assert labels["cell_id"].tolist() == [f"cell_{i:03d}" for i in range(60)]


def test_empty_preprocessing_is_reported_without_training(run_cli, toy_h5ad, tmp_path):
    output = tmp_path / "empty preprocessing"
    result = run_cli("run", "--data", toy_h5ad, "--algorithms", "pca", "--device", "cpu",
                     "--output", output, "--no-timestamp", "--no-scib-metrics")
    assert result.returncode != 0
    assert "Preprocessing removed all" in result.stderr
    assert "Traceback" not in result.stderr
    status = json.loads((output / "config" / "run_status.json").read_text())
    assert status["status"] == "failed"
    assert status["completed_runs"] == 0
    assert "Preprocessing removed all" in status["failures"][0]["error"]
