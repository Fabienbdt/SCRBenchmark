"""Ensure failed or missing runs cannot be reported as a successful campaign."""

import json
from types import SimpleNamespace

import numpy as np

from scrbenchmark.execution import write_run_status
from scrbenchmark.utils.analysis_runner import AnalysisResult, AnalysisRunner


def test_partial_failure_preserves_successes_and_records_missing_runs(tmp_path):
    result = SimpleNamespace(algorithm_name="pca", run_id=0)
    code = write_run_status(tmp_path, ["pca", "scdeepcluster"], 2, [result])
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert code == 1
    assert status["status"] == "partial_failure"
    assert status["completed_runs"] == 1
    assert {(row["algorithm"], row["run_id"]) for row in status["failures"]} == {
        ("pca", 1), ("scdeepcluster", 0), ("scdeepcluster", 1),
    }


def test_successful_campaign_has_no_missing_runs(tmp_path):
    results = [SimpleNamespace(algorithm_name="pca", run_id=i) for i in range(2)]
    assert write_run_status(tmp_path, ["pca"], 2, results) == 0
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert status["status"] == "completed"
    assert status["failures"] == []


def test_runner_keeps_failed_algorithms_visible(monkeypatch, tmp_path):
    from scrbenchmark import algorithms  # noqa: F401
    runner = AnalysisRunner(output_dir=tmp_path)

    def run_single(algorithm_name, run_id, **kwargs):
        if algorithm_name == "scdeepcluster":
            raise RuntimeError("deliberate training failure")
        return AnalysisResult(
            algorithm_name=algorithm_name, run_id=run_id, labels=np.arange(6) % 2,
            embeddings=None, metrics={"ARI": 1.0}, runtime=0.01, params={},
        )

    monkeypatch.setattr(runner, "run_algorithm", run_single)
    result = runner.run_comparison(
        ["pca", "scdeepcluster"], np.ones((6, 3)), n_repeats=2, compute_scib_metrics=False,
    )
    assert len(result.results) == 2
    assert result.failures == [
        {"algorithm": "scdeepcluster", "run_id": i, "error": "deliberate training failure"} for i in range(2)
    ]
