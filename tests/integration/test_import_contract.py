"""Check registry identity and protocol discovery in an isolated process."""

import subprocess
import sys


def test_package_imports_share_one_registry_and_find_resources(tmp_path, cli_env):
    code = """
import sys
from scrbenchmark import algorithms, cli
from scrbenchmark.core.algorithm_registry import AlgorithmRegistry
from scrbenchmark.algorithms.pca import PCAClusteringAlgorithm
from scrbenchmark.methods import get_method_spec
from scrbenchmark.protocols.registry import get_protocol_spec
assert cli.load_algorithms() is AlgorithmRegistry
assert AlgorithmRegistry.get('pca') is PCAClusteringAlgorithm
assert get_method_spec('scRAW') is not None
assert get_protocol_spec('baron_transductive') is not None
assert 'core.algorithm_registry' not in sys.modules
assert 'algorithms' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], cwd=tmp_path, env=cli_env,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
