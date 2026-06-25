# Adding an External Algorithm to SCRBenchmark

This document is the single guide for adding a new algorithm from external source code. The principle of SCRBenchmark is to keep the author's code in `external/original_code/` or `vendor/`, then add a small wrapper and a YAML file in `methods/` to make the experiments reproducible.

In this guide, we add a functional example called `my_external_kmeans`. Replace the KMeans parts with the actual code of your external algorithm.

---

## Overview

An external algorithm adds three elements:

| Element | Path | Role |
| --- | --- | --- |
| External source code | `external/original_code/<algo>/` | Author's code kept separate from the SCRBenchmark core. |
| SCRBenchmark wrapper | `external/original_code/<algo>/scrbenchmark_wrapper.py` | Loads the `.h5ad`, calls the external algorithm, writes labels/embeddings. |
| YAML Specification | `methods/<algo>.yaml` | Describes how to run the wrapper and where to find the outputs. |

The new algorithm is then launched with:

```bash
python3 scripts/reproduction/run_method.py --method <algo> ...
```

It also appears in the reproduction interface if `report: true` is kept in the YAML.

Important: for an external algorithm, do not start with `src/scrbenchmark/algorithms/`. This directory is for internal baselines already maintained within the SCRBenchmark engine. The standard path for new external integrations is `external/original_code/` + `methods/*.yaml`.

---

## 1. Choose a Stable Identifier

From the root of the repository:

```bash
cd /data2/fbidet/SCRBenchmark
```

Choose a short, lowercase name without spaces. Example:

```text
my_external_kmeans
```

This name will be used everywhere:

- source folder: `external/original_code/my_external_kmeans/`;
- YAML file: `methods/my_external_kmeans.yaml`;
- command: `--method my_external_kmeans`;
- parameters: `--param my_external_kmeans:n_init=20`.

---

## 2. Add the External Source Code

Create the directory for the algorithm:

```bash
mkdir -p external/original_code/my_external_kmeans
```

For a real algorithm, place the author's code here: Git clone, paper archive, Python file provided by the authors, or minimal adaptation.

Examples:

```text
external/original_code/my_external_kmeans/
  original_author_code.py
  requirements.txt
  README_author.md
```

If the author's code requires additional dependencies, add them to `requirements-reproduction.txt` and note the version in the YAML.

---

## 3. Create the SCRBenchmark Wrapper

Create:

```text
external/original_code/my_external_kmeans/scrbenchmark_wrapper.py
```

Ready-to-copy content:

```python
#!/usr/bin/env python3
"""SCRBenchmark wrapper example for an external clustering algorithm.

Replace the KMeans block with the real external algorithm call. Keep the input
arguments and output files stable so SCRBenchmark can run and evaluate it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


METHOD_NAME = "my_external_kmeans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input .h5ad file.")
    parser.add_argument("--output", required=True, help="Raw output directory.")
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Optional method parameter, e.g. my_external_kmeans:n_init=20.",
    )
    return parser.parse_args()


def coerce_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_method_params(raw_params: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for raw in raw_params:
        text = str(raw)
        if ":" in text:
            prefix, text = text.split(":", 1)
            if prefix and prefix != METHOD_NAME:
                continue
        if "=" not in text:
            raise ValueError(f"Invalid --param value {raw!r}; expected key=value.")
        key, value = text.split("=", 1)
        params[key.strip()] = coerce_value(value.strip())
    return params


def preprocess_for_example(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    """Small example preprocessing used only by this demo wrapper."""
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=float(args.target_sum))
    sc.pp.log1p(adata)

    if 0 < int(args.n_top_genes) < adata.n_vars:
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=int(args.n_top_genes),
            flavor=str(args.hvg_flavor),
        )
        adata = adata[:, adata.var["highly_variable"]].copy()

    return adata


def dense_matrix(adata: ad.AnnData) -> np.ndarray:
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, copy=False)


def main() -> int:
    args = parse_args()
    params = parse_method_params(args.param)

    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2.")

    adata_original = ad.read_h5ad(args.input)
    if args.n_clusters > adata_original.n_obs:
        raise ValueError("--n-clusters cannot exceed the number of cells.")

    adata = preprocess_for_example(adata_original, args)
    X = dense_matrix(adata)

    n_components = min(int(args.n_pcs), max(1, X.shape[0] - 1), X.shape[1])

    # ------------------------------------------------------------------
    # Replace this block with the real external algorithm.
    # The wrapper must finally produce:
    #   labels: one cluster id per cell
    #   embedding: optional latent matrix with one row per cell
    # ------------------------------------------------------------------
    embedding = PCA(n_components=n_components, random_state=int(args.seed)).fit_transform(X)
    model = KMeans(
        n_clusters=int(args.n_clusters),
        n_init=int(params.get("n_init", 20)),
        max_iter=int(params.get("max_iter", 300)),
        random_state=int(args.seed),
    )
    labels = model.fit_predict(embedding)
    # ------------------------------------------------------------------

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_ids = [str(cell_id) for cell_id in adata_original.obs_names]
    pd.DataFrame({"cell_id": cell_ids, "cluster": labels}).to_csv(
        output_dir / "labels.csv",
        index=False,
    )

    latent = pd.DataFrame(
        embedding,
        columns=[f"latent_{idx + 1}" for idx in range(embedding.shape[1])],
    )
    latent.insert(0, "cell_id", cell_ids)
    latent.to_csv(output_dir / "latent.csv", index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The wrapper must respect three rules:

1. read the `.h5ad` file given by `--input`;
2. write `labels.csv` with columns `cell_id,cluster`;
3. write `latent.csv` if the algorithm produces an embedding.

For a real external algorithm, replace only the KMeans block with the call to the author's code. The rest of the wrapper can remain identical.

---

## 4. Create the Method YAML

Create:

```text
methods/my_external_kmeans.yaml
```

Ready-to-copy content:

```yaml
name: my_external_kmeans
display_name: My External KMeans
family: classical
report: true
core_contract: external_source_wrapped
core_status: local_wrapper_smoke_tested
aliases: []
source:
  kind: original_source
  path: external/original_code/my_external_kmeans
  reference: "Replace with paper, DOI, GitHub URL, or author archive."
runner:
  kind: command_template
  command:
    - "{python_bin}"
    - "{source_path}/scrbenchmark_wrapper.py"
    - "--input"
    - "{data}"
    - "--output"
    - "{raw_dir}"
    - "--n-clusters"
    - "{n_labels}"
    - "--seed"
    - "{seed}"
    - "--label-key"
    - "{label_key}"
    - "--batch-key"
    - "{batch_key}"
    - "--n-top-genes"
    - "{n_top_genes}"
    - "--target-sum"
    - "{target_sum}"
    - "--hvg-flavor"
    - "{hvg_flavor}"
    - "--n-pcs"
    - "{n_pcs}"
    - "{param_args}"
output:
  expected_file: results/analysis_results.csv
  labels_file: raw/labels.csv
  labels_column: cluster
  cell_id_column: cell_id
  latent_file: raw/latent.csv
notes: "Example external algorithm wrapper. Replace the KMeans block with the real author code."
```

This YAML tells SCRBenchmark:

- which folder contains the external source code;
- which command to run;
- where to find the raw labels;
- how to convert these labels into the standard format `results/analysis_results.csv`.

---

## 5. Validate the Command Without Running the Calculation

```bash
python3 scripts/reproduction/validate_method.py \
  --method my_external_kmeans \
  --data data/baron_human_pancreas.h5ad \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --n-top-genes 500 \
  --n-pcs 20
```

This command should display:

- the name of the method;
- the runner type `command_template`;
- the source path;
- the exact command that will be launched.

If this step fails, correct the YAML before launching a real run.

---

## 6. Run a Real Smoke Test

```bash
python3 scripts/reproduction/validate_method.py \
  --method my_external_kmeans \
  --data data/baron_human_pancreas.h5ad \
  --output results/smoke_my_external_kmeans \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --n-top-genes 500 \
  --n-pcs 20 \
  --device cpu \
  --run
```

Then check the outputs:

```bash
ls results/smoke_my_external_kmeans/raw
ls results/smoke_my_external_kmeans/results
head results/smoke_my_external_kmeans/results/analysis_results.csv
```

The standard file must contain at least:

```text
cell_id,cluster
```

If there is an embedding, SCRBenchmark also adds `latent_1`, `latent_2`, etc.

---

## 7. Move from the Example to the Real External Algorithm

In `scrbenchmark_wrapper.py`, replace this block:

```python
embedding = PCA(n_components=n_components, random_state=int(args.seed)).fit_transform(X)
model = KMeans(...)
labels = model.fit_predict(embedding)
```

with the call to the author's code, for example:

```python
from original_author_code import ExternalClusteringModel

model = ExternalClusteringModel(
    n_clusters=int(args.n_clusters),
    random_state=int(args.seed),
    **params,
)
model.fit(X)
labels = model.labels_
embedding = model.embedding_
```

Keep the same output contract:

```python
pd.DataFrame({"cell_id": cell_ids, "cluster": labels}).to_csv(
    output_dir / "labels.csv",
    index=False,
)
```

If the algorithm does not produce an embedding, remove `latent_file` from the YAML and do not write `latent.csv`.

---

## 8. Add Hyperparameters

Free hyperparameters pass through `--param`.

Example:

```bash
python3 scripts/reproduction/validate_method.py \
  --method my_external_kmeans \
  --data data/baron_human_pancreas.h5ad \
  --output results/smoke_my_external_kmeans \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --param my_external_kmeans:n_init=10 \
  --param my_external_kmeans:max_iter=100 \
  --run
```

In the wrapper, these values are available in:

```python
params = parse_method_params(args.param)
```

JSON values are automatically converted:

```text
--param my_external_kmeans:n_init=10      -> int
--param my_external_kmeans:use_gpu=true   -> bool
--param my_external_kmeans:mode=fast      -> str
```

---

## 9. Verify That the Algorithm is Visible

List of external algorithms known by the reproduction registry:

```bash
python3 scripts/reproduction/run_method.py --list | grep my_external_kmeans
```

If `report: true` is present in the YAML, the algorithm can also be exposed in the Streamlit reproduction interface.

```bash
./run.sh
```

Then open `Report Reproduction` or `Customize Benchmark`, depending on the workflow used.

---

## 10. Checklist Before Commit

```bash
python3 scripts/reproduction/validate_method.py \
  --method my_external_kmeans \
  --data data/baron_human_pancreas.h5ad \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --n-top-genes 500 \
  --n-pcs 20

python3 scripts/reproduction/validate_method.py \
  --method my_external_kmeans \
  --data data/baron_human_pancreas.h5ad \
  --output results/smoke_my_external_kmeans \
  --label-key Group \
  --batch-key batch \
  --n-labels 14 \
  --n-top-genes 500 \
  --n-pcs 20 \
  --device cpu \
  --run

python3 scripts/reproduction/run_method.py --list | grep my_external_kmeans
git diff --check
```

Add to the commit:

```text
external/original_code/my_external_kmeans/
methods/my_external_kmeans.yaml
requirements-reproduction.txt   # only if a dependency was added
```

---

## 11. Frequent Errors

| Symptom | Probable Cause | Correction |
| --- | --- | --- |
| `Unknown method 'my_external_kmeans'` | The YAML is not in `methods/`, or `name` does not match. | Verify `methods/my_external_kmeans.yaml` and run `run_method.py --list` again. |
| `source path does not exist` | `source.path` points to a missing directory. | Correct `source.path` or create the directory in `external/original_code/`. |
| `labels_file ... does not exist` | The wrapper did not write `raw/labels.csv`. | Verify the `--output {raw_dir}` argument and the write path. |
| `Cannot find labels column` | The labels CSV does not have the declared column. | Keep `cluster` in `labels.csv` or modify `output.labels_column`. |
| Embeddings are not in `analysis_results.csv` | `latent_file` missing, wrong path, or different number of rows than labels. | Write `raw/latent.csv` with the same number of cells as `labels.csv`. |
| The wrapper crashes on `--param` | The wrapper parser does not accept `--param`. | Keep the `parser.add_argument("--param", action="append", default=[])` block. |
| The actual external code cannot find its imports | The source directory is not in `PYTHONPATH` or dependency is missing. | Import from the wrapper with an explicit path, or add the dependency to `requirements-reproduction.txt`. |
