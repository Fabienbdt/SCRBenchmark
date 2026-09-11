"""Report launchers and their runtime assets must also work from a wheel."""

import subprocess
import sys

from scrbenchmark.paths import resource_root


def test_registered_report_method_can_be_planned_outside_checkout(tmp_path, cli_env, toy_h5ad):
    script = resource_root() / "scripts" / "reproduction" / "run_method.py"
    result = subprocess.run(
        [sys.executable, str(script), "--method", "scRAW", "--data", str(toy_h5ad),
         "--output", str(tmp_path / "planned"), "--n-labels", "3", "--device", "cpu", "--dry-run"],
        cwd=tmp_path, env=cli_env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "run_scraw_external.py" in result.stdout
    assert str(toy_h5ad) in result.stdout
    assert (resource_root() / "vendor/scraw_inductive/configs/stable_generalist_trial_0017.json").is_file()
