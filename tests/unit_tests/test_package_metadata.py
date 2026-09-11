"""Tests for the installable package metadata and command-line entry point."""

from importlib import metadata
import subprocess
import sys

import scrbenchmark


def test_distribution_metadata_matches_package_version():
    distribution = metadata.distribution("SCRBenchmark")

    assert distribution.metadata["Name"] == "SCRBenchmark"
    assert distribution.metadata["Requires-Python"] == ">=3.10"
    assert distribution.version == scrbenchmark.__version__ == "1.1.0"


def test_console_script_targets_cli_main():
    distribution = metadata.distribution("SCRBenchmark")
    console_scripts = {
        entry_point.name: entry_point.value
        for entry_point in distribution.entry_points
        if entry_point.group == "console_scripts"
    }

    assert console_scripts["scrbenchmark"] == "scrbenchmark.cli:main"


def test_python_module_exposes_cli_help(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-m", "scrbenchmark", "--help"],
        cwd=tmp_path,
        timeout=30,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Single-Cell RNA-seq Clustering Benchmark CLI" in result.stdout
    assert "list-algorithms" in result.stdout
