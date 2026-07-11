# SCRBenchmark Data Directory

This directory contains the data used to benchmark single-cell RNA-seq
clustering algorithms.

Large `.h5ad` files are not tracked by Git so the repository stays manageable.
The repository therefore provides:

1. small raw files when tracking them is reasonable;
2. scripts to rebuild or materialize prepared datasets;
3. manifests to verify the expected files.

---

## Generate the Baron Human Pancreas Dataset

The Baron human pancreas dataset is the main test dataset. Generate it with:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

The script creates:

```text
data/baron_human_pancreas.h5ad
```

This file contains approximately:

- 8,500 cells from 4 human pancreas donors;
- 14 cell types;
- 20,000 genes;
- batch/donor information.

Expected structure after generation:

```text
data/
├── GSE84133_RAW/
├── baron_human_pancreas.h5ad
└── README.md
```

---

## stable_generalist Datasets

The stable_generalist reproduction expects 13 `.h5ad` files in:

```text
data/stable_generalist/
```

Recommended command when the source data is available locally:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/existing/h5ad/files
```

If the exact files are hosted remotely:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

Verify already prepared files:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

This verification is read-only. Use `--report <path.csv>` to save a separate
verification report without rewriting the source manifest.

The script compares files against `data/stable_generalist/download_manifest.csv`
using SHA256, file size, AnnData dimensions, and expected columns.

The file list is documented in:

```text
data/stable_generalist/README.md
```

If the files are stored elsewhere, pass the directory explicitly to the plan
builder:

```bash
python scripts/reproduction/build_stable_generalist_plan.py --data-root /path/to/h5ad_files
```

To add a new dataset to the project, follow the step-by-step guide:

```text
docs/dataset_integration_guide.md
```
