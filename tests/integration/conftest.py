"""Deterministic inputs for real CLI and installed-wheel tests."""

import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def cli_env(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("runtime-caches")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        MPLBACKEND="Agg", MPLCONFIGDIR=str(tmp_path / "mpl"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
        NUMBA_CACHE_DIR=str(tmp_path / "numba"), NUMBA_NUM_THREADS="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        SCRB_DISABLE_SCIB_METRICS="1",
    )
    return env


@pytest.fixture
def toy_h5ad(tmp_path):
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 20)
    counts = rng.poisson(2, (60, 30)).astype(np.float32) + 1
    for cluster in range(3):
        counts[labels == cluster, cluster * 10:(cluster + 1) * 10] += 30
    obs = pd.DataFrame(
        {"Group": [f"type_{label}" for label in labels], "batch": np.tile(["a", "b"], 30)},
        index=[f"cell_{index:03d}" for index in range(60)],
    )
    data = ad.AnnData(counts, obs=obs, var=pd.DataFrame(index=[f"gene_{i}" for i in range(30)]))
    path = tmp_path / "input with spaces.h5ad"
    data.write_h5ad(path)
    return path


@pytest.fixture
def run_cli(tmp_path, cli_env):
    def run(*args):
        return subprocess.run(
            [sys.executable, "-I", "-m", "scrbenchmark", *map(str, args)],
            cwd=tmp_path, env=cli_env, capture_output=True, text=True, timeout=120,
        )
    return run
