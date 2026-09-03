# SCRBenchmark

**SCRBenchmark** is a Python suite for comparing clustering algorithms on single-cell RNA-seq data. The repository provides:

- A **Streamlit** graphical user interface to configure, run, and explore experiments;
- A **CLI** interface to automate benchmarks;
- Reproduction scripts for the experiments in the report;
- Guides for adding algorithms, reproducible methods, preprocessings, and new protocols.

Author: **Fabien Bidet**.

Copyright: **(c) 2026 Fabien Bidet. All rights reserved.**

Scientific report: [M2 internship report - Fabien Bidet
(PDF)](docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf).

---

## I Want To... -> Read This

| Goal | Document or entry point |
| --- | --- |
| Take over the whole project | [docs/guide/handover_guide.md](docs/guide/handover_guide.md) |
| Read the M2 internship report | [docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf](docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf) |
| Install and run a first benchmark | README, "Recommended 10-minute path" |
| Regenerate the report figures | [docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md) |
| Add an external algorithm | [docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md) |
| Add a dataset | [docs/guide/dataset_integration_guide.md](docs/guide/dataset_integration_guide.md) |
| Add a preprocessing step | [docs/guide/preprocessing_extension_guide.md](docs/guide/preprocessing_extension_guide.md) |
| Understand the repository files | [docs/guide/developer_file_guide.md](docs/guide/developer_file_guide.md) |
| Understand the reproduction scripts | [scripts/reproduction/README.md](scripts/reproduction/README.md) |

To add an external algorithm, do not modify
`src/scrbenchmark/algorithms/`; follow only
[docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md).

---

## Recommended 10-Minute Path

```bash
cd /path/to/SCRBenchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check the installation:

```bash
./scrbenchmark list-algorithms
```

Prepare the reproduction datasets if the local source is available:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/existing/h5ad/files
```

Run a lightweight first benchmark:

```bash
./scrbenchmark run \
  --data data/stable_generalist/baron_human_pancreas.h5ad \
  --algorithms pca \
  --param pca:clustering_method=kmeans \
  --label-col label \
  --n-clusters 14 \
  --output results/quickstart_baron_pca
```

Or open the graphical interface:

```bash
./run.sh
```

To reproduce the report with the complete script order, read
[docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md).

## Installation

SCRBenchmark requires **Python >= 3.9**. From the root of the repository:

```bash
cd /path/to/SCRBenchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies are split into two files:

| File | Content | When to install |
| --- | --- | --- |
| `requirements.txt` | Core packages: Streamlit, CLI, Scanpy/AnnData, PyTorch, scVI, metrics, integrated algorithms, tests. | Normal installation. |
| `requirements-reproduction.txt` | Additional packages for heavy reproductions: Harmony, Scanorama, JAX/scIB, baselines, and external methods. | Reproduction of the report or advanced external methods. |

Complete installation for reproduction experiments:

```bash
pip install -r requirements-reproduction.txt
```

Quick verification:

```bash
./scrbenchmark list-algorithms
```

---

## Datasets

Benchmarks mainly use AnnData/Scanpy-compatible `.h5ad` files.

### Baron pancreas dataset

To generate the Baron human pancreas dataset used by tests and examples:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

The generated file is:

```text
data/baron_human_pancreas.h5ad
```

### stable_generalist datasets

The stable_generalist report experiments expect 13 `.h5ad` files in:

```text
data/stable_generalist/
```

To materialize these files from your local data root:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/existing/h5ad/files
```

If the exact files are hosted on a release or a web directory:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

See [data/README.md](data/README.md) and
[data/stable_generalist/README.md](data/stable_generalist/README.md) for the
expected format, list of files, and SHA256 verification.

### Minimum expected format

A custom dataset must contain:

- `adata.X`: expression matrix;
- `adata.obs["Group"]` or a label column provided with `--label-col`;
- Optionally `adata.obs["batch"]` or a batch column provided with the split/correction options.

---

## Running SCRBenchmark

### Graphical Interface

```bash
./run.sh
```

Then open the URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

Recommended workflow:

1. `Data Upload`: load a `.h5ad`.
2. `Data Split`: choose standard protocol or train/val/test.
3. `Preprocessing`: configure filters, normalization, HVG, dropout, batch correction.
4. `Algorithm Config`: choose algorithms and hyperparameters.
5. `Analysis`: run the experiment.
6. `Results Explorer`: compare the results.

### Command Line

Simple example with PCA + K-Means:

```bash
./scrbenchmark run \
  --data data/baron_human_pancreas.h5ad \
  --algorithms pca \
  --param pca:clustering_method=kmeans \
  --label-col Group \
  --n-clusters 14 \
  --output results/test_run
```

Other useful commands:

```bash
./scrbenchmark list-algorithms
./scrbenchmark list-params --algorithm pca
./scrbenchmark generate-config --output config.yaml
./scrbenchmark run --config config.yaml
```

The generated configuration targets `data/baron_human_pancreas.h5ad`, the file
created by the Baron preparation command above. Change `data.file` when using a
different dataset.

### Report Reproduction

The recommended entry point is the Streamlit interface:

```bash
./run.sh
```

Then open `Report Reproduction`. This panel generates the `planned_jobs.csv` files and launch shell scripts for:

- the stable_generalist campaign;
- inductive complements;
- loss-transfer experiments;
- Harmony variants;
- biological interpretation / marker-overlap;
- export of already available scRAW artifacts through scripts;
- custom protocols.

The numbered execution order is in
[docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md). The map
of figures/tables from the report is in
[docs/guide/report_reproduction_map.md](docs/guide/report_reproduction_map.md).

### scRAW Presets

SCRBenchmark exposes exactly two public scRAW presets:

- `default`: the vendored 0017/stable configuration in `vendor/scraw_inductive/configs/`;
- `baron`: the vendored Baron-compatible configuration in `vendor/scraw_inductive/configs/`.

For registered report-method runs, select it with `--scraw-preset default` or
`--scraw-preset baron` when calling `scripts/reproduction/run_method.py`. For
inductive scRAW scripts, use `--preset default` or `--preset baron`.

---

## Guides

| Guide | Target Audience | When to use |
| --- | --- | --- |
| [docs/guide/handover_guide.md](docs/guide/handover_guide.md) | New maintainer / intern | Start here to understand scRAW, SCRBenchmark, the tested workflows, and the external assets required for full report reproduction. |
| [docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf](docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf) | Scientific reader | Read the scientific context, methods, experiments, results, limitations, and appendices. |
| [docs/guide/user_guide.md](docs/guide/user_guide.md) | SCRBenchmark User | Understand the workflow: detailed installation, data preparation, GUI, CLI, and report reproduction. |
| [docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md) | Reproduction user | Numbered commands to regenerate report figures and reuse existing artifacts when possible. |
| [docs/guide/dataset_integration_guide.md](docs/guide/dataset_integration_guide.md) | Data user | Add a new `.h5ad` dataset to GUI, CLI, manifests, and reproduction plans. |
| [docs/guide/developer_file_guide.md](docs/guide/developer_file_guide.md) | Developer | Know which file to modify to change the preprocessing, algorithms, interface, metrics, or scripts. |
| [docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md) | External Algorithm Developer | Single step-by-step guide: external source code, wrapper, YAML, validation, and smoke test. |
| [docs/guide/preprocessing_extension_guide.md](docs/guide/preprocessing_extension_guide.md) | Preprocessing Developer | Add a preprocessing step without train/test leakage and without breaking GUI/CLI. |
| [methods/README.md](methods/README.md) | Method Developer | Understand the role of the `methods/` directory and YAML specifications. |
| [protocols/README.md](protocols/README.md) | Experiment Designer | Understand the format of versioned YAML protocols loadable from Customize Benchmark. |
| [scripts/reproduction/README.md](scripts/reproduction/README.md) | Reproduction / automation | Choose the correct script to generate plans, run methods, or replay report experiments. |
| [data/README.md](data/README.md) | Data User | Prepare datasets and verify the expected `.h5ad` format. |

---

## Modifying or extending SCRBenchmark

Quick entry points:

- change a preprocessing parameter: `src/scrbenchmark/core/config.py`;
- modify standard preprocessing: `src/scrbenchmark/utils/data_handler.py`;
- modify train/val/test preprocessing: `src/scrbenchmark/utils/dataset_splitter.py`;
- add an external algorithm: follow only
  [docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md);
- modify the interface: `src/scrbenchmark/gui/`;
- modify metrics and results: `src/scrbenchmark/utils/metrics.py` and
  `src/scrbenchmark/utils/analysis_runner.py`.

The technical map is [docs/guide/developer_file_guide.md](docs/guide/developer_file_guide.md).

---

## Repository Structure

```text
src/scrbenchmark/        main code: CLI, GUI, registries, algorithms, utils
docs/guide/              technical guides and reproduction maps
docs/paper/              report PDF and manuscript source
data/                   local data and documented preparation scripts
methods/                YAML specifications of reproducible methods
protocols/              versioned YAML protocols
scripts/setup/          dataset preparation
scripts/reproduction/   reproduction plans, launchers, and executors
tests/                  unit tests and comparisons
vendor/                 backends and helpers integrated into the repository
external/               external author code kept separately
results/                local outputs of experiments
```

---

## Tests

Main unit tests:

```bash
pytest tests/unit_tests
```

Markdown and spacing check before committing:

```bash
git diff --check
```
