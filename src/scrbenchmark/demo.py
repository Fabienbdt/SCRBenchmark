"""Generate a small, synthetic dataset for installation checks."""

from pathlib import Path


def create_demo_dataset(path: str | Path, seed: int = 42) -> Path:
    """Write 120 synthetic cells in three groups, without overwriting an input file."""
    import anndata as ad
    import numpy as np
    import pandas as pd

    destination = Path(path).expanduser()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(3), 40)
    counts = rng.poisson(2, (120, 60)).astype(np.float32)
    for cluster in range(3):
        counts[labels == cluster, cluster * 20:(cluster + 1) * 20] += 20
    obs = pd.DataFrame(
        {"Group": [f"type_{label}" for label in labels], "batch": np.tile(["batch_a", "batch_b"], 60)},
        index=[f"cell_{index:03d}" for index in range(120)],
    )
    var = pd.DataFrame(index=[f"gene_{index:03d}" for index in range(60)])
    data = ad.AnnData(counts, obs=obs, var=var)
    data.uns["synthetic"] = True
    data.uns["generation_seed"] = seed
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(destination)
    return destination
