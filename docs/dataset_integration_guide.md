# Add a New Dataset to SCRBenchmark

This guide explains how to integrate a new `.h5ad` dataset so it can be used in
the interface, CLI, and reproduction scripts.

---

## 1. Prepare the AnnData File

The file must be readable by Scanpy:

```python
import scanpy as sc

adata = sc.read_h5ad("data/my_dataset.h5ad")
print(adata)
print(adata.obs.columns)
```

Minimum structure:

- `adata.X`: cells x genes matrix;
- `adata.obs["label"]` or an equivalent cell-type column;
- `adata.obs["batch"]` if the dataset contains several batches, donors,
  patients, or technical conditions;
- `adata.obs_names`: stable cell identifiers;
- `adata.var_names`: stable gene names.

SCRBenchmark accepts other column names, but they must be passed explicitly with
`--label-key`, `--batch-key`, `--label-col`, or the interface fields.

---

## 2. Verify Important Columns

Quick validation example:

```bash
python - <<'PY'
import scanpy as sc

path = "data/stable_generalist/my_dataset.h5ad"
label_key = "label"
batch_key = "batch"

adata = sc.read_h5ad(path)
assert adata.n_obs > 0, "No cells"
assert adata.n_vars > 0, "No genes"
assert label_key in adata.obs, f"Missing {label_key}"
assert batch_key in adata.obs, f"Missing {batch_key}"
print("cells", adata.n_obs)
print("genes", adata.n_vars)
print("labels", adata.obs[label_key].nunique())
print("batches", adata.obs[batch_key].nunique())
PY
```

If the dataset has no batch, use a constant column:

```python
adata.obs["batch"] = "single_batch"
```

---

## 3. Choose a Stable Key

Choose a short key without spaces:

```text
my_dataset
```

This key is used in:

- file names;
- reproduction plans;
- result directories;
- manifests.

Recommended file name:

```text
data/stable_generalist/my_dataset.h5ad
```

---

## 4. Add the Dataset to Reproduction Tables When Needed

For a simple GUI/CLI test, loading the `.h5ad` file is enough.

To integrate it into the `stable_generalist` campaign, add a row to:

```text
reproducibility/stable_generalist/stable_generalist_dataset_table.csv
```

Important fields:

| Column | Role |
| --- | --- |
| `dataset_key` | Stable dataset key. |
| `dataset` | Human-readable name. |
| `data_file` | `.h5ad` path or file name. |
| `label_key` | Biological label column. |
| `dann_batch_column` | Batch/donor column. |
| `n_labels` | Expected number of cell types. |
| `n_batches` | Expected number of batches. |

If the file must be downloaded/verified with the 13 reference datasets, also
add a row to:

```text
data/stable_generalist/download_manifest.csv
```

Compute size and SHA256 without loading the whole file into memory:

```bash
python - <<'PY'
from pathlib import Path
import hashlib

path = Path("data/stable_generalist/my_dataset.h5ad")
digest = hashlib.sha256()
with path.open("rb") as handle:
  for chunk in iter(lambda: handle.read(1024 * 1024), b""):
    digest.update(chunk)

print(f"size={path.stat().st_size}")
print(f"sha256={digest.hexdigest()}")
PY
```

---

## 5. Smoke Test the Dataset With a Simple Method

```bash
./scrbenchmark run \
  --data data/stable_generalist/my_dataset.h5ad \
  --algorithms pca \
  --param pca:clustering_method=kmeans \
  --label-col label \
  --n-clusters 10 \
  --output results/smoke_my_dataset
```

Adapt `--label-col` and `--n-clusters` to the dataset.

Check:

```bash
ls results/smoke_my_dataset/results
```

The directory should contain CSV/JSON results and the main metrics.

---

## 6. Use It in the Interface

```bash
./run.sh
```

Workflow:

1. `Data Upload`: load the `.h5ad`;
2. `Data Split`: select label/batch columns;
3. `Preprocessing`: check filters;
4. `Algorithm Config`: choose algorithms;
5. `Analysis`: run;
6. `Results Explorer`: compare.

---

## 7. Add It to a Reproduction Protocol

For a one-off protocol, use:

```bash
python scripts/reproduction/manual_protocols.py \
  --protocol inductive \
  --data data/stable_generalist/my_dataset.h5ad \
  --dataset-key my_dataset \
  --label-key label \
  --batch-key batch \
  --n-labels 10 \
  --train-batches batch1,batch2 \
  --test-batches batch3 \
  --inductive-algorithms scraw,scname,sc_mae \
  --output-root results/manual_my_dataset \
  --script results/manual_my_dataset/run_jobs.sh
```

Then run:

```bash
bash results/manual_my_dataset/run_jobs.sh
```

For a versioned campaign, create or edit a YAML file in:

```text
protocols/
```

The protocol format is documented in
[`../protocols/README.md`](../protocols/README.md).

---

## 8. Watch Points

- Do not change labels after computing a reference SHA256.
- Keep gene names in `adata.var_names`, not only in an auxiliary column.
- Keep a batch column even if it is constant; this simplifies shared scripts.
- For large datasets, test PCA + KMeans before running deep methods.
- If the dataset is used in the report, document the source and preparation
  command in `data/README.md`.
