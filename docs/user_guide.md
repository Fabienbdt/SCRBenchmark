# SCRBenchmark User Guide

This guide explains how to install SCRBenchmark, prepare datasets, run the
Streamlit interface or CLI, reproduce report experiments, and find the right
guide when the project needs to be extended.

---

## Table of Contents

1. [Complete installation](#1-complete-installation)
2. [Dataset preparation](#2-dataset-preparation)
3. [Use the Streamlit interface](#3-use-the-streamlit-interface)
4. [Run CLI benchmarks](#4-run-cli-benchmarks)
5. [Reproduce report experiments](#5-reproduce-report-experiments)
6. [Extend SCRBenchmark](#6-extend-scrbenchmark)
7. [Troubleshooting](#7-troubleshooting)

## Quick Map

| Need | Document |
| --- | --- |
| Quick install, first run, and guide index | [`../README.md`](../README.md) |
| User workflow, datasets, GUI, and CLI | This guide |
| Technical file map | [`developer_file_guide.md`](developer_file_guide.md) |
| Add an external algorithm | [`algorithm_extension_guide.md`](algorithm_extension_guide.md) |
| Understand `methods/` YAML files | [`../methods/README.md`](../methods/README.md) |
| Add a preprocessing step | [`preprocessing_extension_guide.md`](preprocessing_extension_guide.md) |
| Understand reproduction scripts | [`../scripts/reproduction/README.md`](../scripts/reproduction/README.md) |

---

## 1. Complete Installation

SCRBenchmark requires **Python >= 3.9** and a clean virtual environment.

```bash
cd /data2/fbidet/SCRBenchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

To reproduce heavy report experiments or use advanced external methods:

```bash
pip install -r requirements-reproduction.txt
```

Quick check:

```bash
./scrbenchmark list-algorithms
```

This command should list the algorithms available in the environment.

---

## 2. Dataset Preparation

SCRBenchmark mainly uses AnnData/Scanpy-compatible `.h5ad` files.

### Baron Pancreas Dataset

This dataset is used for examples, smoke tests, and quick comparisons:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

The generated file is:

```text
data/baron_human_pancreas.h5ad
```

### stable_generalist Campaign

To reproduce the report experiments, SCRBenchmark expects 13 `.h5ad` files in:

```text
data/stable_generalist/
```

Preparation from the local data root:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /data2/fbidet/scRAW_EXPERIMENTAL/data
```

If the exact files are hosted remotely:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

The script verifies SHA256 hashes, sizes, AnnData dimensions, and expected
label/batch columns. To verify already prepared files:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

---

## 3. Use the Streamlit Interface

The graphical interface is the recommended entry point to explore data,
configure experiments, and compare results.

```bash
./run.sh
```

Then open the URL displayed in the terminal, often:

```text
http://localhost:8501
```

Recommended workflow:

1. `Data Upload`: load a `.h5ad` file.
2. `Data Split`: choose the standard protocol or train/val/test.
3. `Preprocessing`: configure filters, HVG, normalization, dropout, or batch correction.
4. `Algorithm Config`: choose algorithms and hyperparameters.
5. `Analysis`: run the benchmark.
6. `Results Explorer`: compare NMI, ARI, silhouette, labels, and embeddings.

---

## 4. Run CLI Benchmarks

The CLI is useful for automating experiments without the graphical interface.

Simple transductive example:

```bash
./scrbenchmark run \
  --data data/baron_human_pancreas.h5ad \
  --algorithms pca \
  --param pca:clustering_method=kmeans \
  --label-col Group \
  --n-clusters 14 \
  --device cpu \
  --output results/my_experiment \
  --no-timestamp \
  --save-labels
```

Important arguments:

- `--data`: path to the `.h5ad` file;
- `--algorithms`: comma-separated algorithm list;
- `--label-col`: `adata.obs` column containing reference labels;
- `--n-clusters`: expected cluster count;
- `--device`: `cpu`, `cuda`, `mps`, or `auto`;
- `--output`: output directory.

Useful commands:

```bash
./scrbenchmark list-algorithms
./scrbenchmark list-params --algorithm pca
./scrbenchmark generate-config --output config.yaml
./scrbenchmark run --config config.yaml
```

After execution, the result directory notably contains:

- `results.csv`: score and runtime summary;
- `labels/`: predicted per-cell labels;
- `embeddings/`: latent representations when requested;
- `config/`: saved configuration for reproducibility.

---

## 5. Reproduce Report Experiments

The recommended entry point is the Streamlit **Report Reproduction** panel. It
replaces long command lists in the documentation.

```bash
./run.sh
```

Open **Report Reproduction** in the sidebar. The panel contains:

- **Traceability**: map between report figures/tables and campaigns;
- **Stable Generalist**: main stable_generalist plan generation;
- **Report Complements**: inductive, loss-transfer, and DEG experiments;
- **Custom Protocols**: configurable variants without changing scripts.

Each tab writes a planned-job CSV and a shell launcher. The compact figure/table
map is available in [`report_reproduction_map.md`](report_reproduction_map.md).

### scRAW Presets

SCRBenchmark exposes two public scRAW presets:

- `default`: the 0017/stable configuration from `/data2/fbidet/scRAW/configs/default_scraw.json`;
- `baron`: the Baron configuration from `/data2/fbidet/scRAW/configs/baron_jobim.json`.

Use `--scraw-preset default|baron` with
`scripts/reproduction/run_method.py`. Use `--preset default|baron` with the
inductive scRAW scripts.

---

## 6. Extend SCRBenchmark

To add an external algorithm, do not modify `src/scrbenchmark/algorithms/`.
Follow only [`algorithm_extension_guide.md`](algorithm_extension_guide.md). It
covers the external source directory, wrapper, `methods/*.yaml`, validation,
and smoke test.

To add a preprocessing step, use
[`preprocessing_extension_guide.md`](preprocessing_extension_guide.md).

To understand the repository organization, use
[`developer_file_guide.md`](developer_file_guide.md). That document is a file
map, not an algorithm-integration procedure.

---

## 7. Troubleshooting

### `CUDA out of memory`

Reduce `batch_size`, use `--device cpu` for a quick test, or decrease the HVG
gene count in preprocessing.

### An External Algorithm Does Not Appear

Check that the YAML is in `methods/`, that `name` matches the `--method`
argument, then run:

```bash
python3 scripts/reproduction/run_method.py --list
```

For a complete external integration, follow
[`algorithm_extension_guide.md`](algorithm_extension_guide.md).

### Raw Count Error

Methods based on NB/ZINB distributions may require non-normalized raw counts.
Check that the `.h5ad` file keeps counts in `adata.X`, `adata.raw`, or a layer
such as `adata.layers["original_X"]`.
