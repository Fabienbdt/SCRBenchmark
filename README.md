# SCRBenchmark

**SCRBenchmark** is a Python suite for comparing clustering algorithms on single-cell RNA-seq data. The repository provides:

- A **Streamlit** graphical user interface to configure, run, and explore experiments;
- A **CLI** interface to automate benchmarks;
- Reproduction plans and launchers for the maintained report campaigns;
- Guides for adding algorithms, reproducible methods, preprocessings, and new protocols.

Author: **Fabien Bidet**.

SCRBenchmark was developed as part of Fabien Bidet's second-year Master's
(M2) internship at **LaBRI** (Laboratoire Bordelais de Recherche en
Informatique). The repository preserves both the maintained software and the
scientific material required to document and reproduce the internship work.

Copyright: **(c) 2026 Fabien Bidet. All rights reserved.**

Scientific report: [M2 internship report - Fabien Bidet
(French PDF)](docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf). The maintained software
documentation and code are in English.

The repository can run new benchmarks from a fresh clone once its dependencies
and input data are installed. Exact numerical reproduction of the complete
report additionally depends on external H5AD files, legacy method environments,
and substantial compute. The current launcher coverage and its explicit limits
are documented in [docs/guide/report_reproduction_map.md](docs/guide/report_reproduction_map.md).

---

## I Want To... -> Read This

| Goal | Document or entry point |
| --- | --- |
| Take over the whole project | [docs/guide/handover_guide.md](docs/guide/handover_guide.md) |
| Read the M2 internship report | [French PDF](docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf) |
| Install and run a first benchmark | README, "Recommended 10-minute path" |
| Contribute and run verification checks | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Maintain or release a version | [Maintenance checks](docs/guide/maintenance.md) |
| Understand the public scRAW pipeline | [docs/guide/scraw_architecture.md](docs/guide/scraw_architecture.md) |
| Regenerate the report figures | [docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md) |
| Add an external algorithm | [docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md) |
| Add a dataset | [docs/guide/dataset_integration_guide.md](docs/guide/dataset_integration_guide.md) |
| Add a preprocessing step | [docs/guide/preprocessing_extension_guide.md](docs/guide/preprocessing_extension_guide.md) |
| Redownload or audit report datasets | [docs/dataset_sources.md](docs/dataset_sources.md) |
| Understand the repository files | [docs/guide/developer_file_guide.md](docs/guide/developer_file_guide.md) |
| Understand the reproduction scripts | [scripts/reproduction/README.md](scripts/reproduction/README.md) |

To add an external algorithm, do not modify
`src/scrbenchmark/algorithms/`; follow only
[docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md).

---

## Recommended 10-Minute Path

```bash
cd /path/to/SCRBenchmark
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
```

Check the installation and create a small synthetic dataset (no data download):

```bash
python -m scrbenchmark list-algorithms
python -m scrbenchmark demo --output data/demo.h5ad
python -m scrbenchmark run \
  --data data/demo.h5ad --algorithms pca \
  --param pca:clustering_method=kmeans --param pca:n_pca_components=5 \
  --label-col Group --n-clusters 3 --skip-preprocessing \
  --device cpu --seed 42 --no-scib-metrics \
  --output results/demo --no-timestamp --save-labels --save-embeddings
```

Inspect `results/demo/results/results.csv` and
`results/demo/config/run_status.json`. The label exports include `cell_id`
to preserve the correspondence with input cells. A failed or incomplete
algorithm run exits with a nonzero status; successful runs remain available.
The synthetic example checks the software workflow, not biological validity.

Open the interface with `scrbenchmark-gui` or `./run.sh`.
For handover, start with the [maintenance guide](docs/guide/maintenance.md) and
the [validation report](docs/guide/validation.md).
To work with real data, prepare Baron with:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

To reproduce the report with the complete script order, read
[docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md).

## Installation

SCRBenchmark requires **Python >= 3.10** and is tested with **Python 3.12**.
From the root of the repository:

```bash
cd /path/to/SCRBenchmark
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[gui]"
```

The editable install exposes `scrbenchmark`, `python -m scrbenchmark`, and
`scrbenchmark-gui`. The core CLI can be installed with `python -m pip install -e .`. Dependency groups are split by purpose:

| File | Content | When to install |
| --- | --- | --- |
| `requirements.txt` | Core runtime: Scanpy/AnnData, PyTorch, metrics and maintained algorithms. | Installed by `pip install -e .`. |
| `.[gui]` | Streamlit and file watching. | Add for the graphical interface. |
| `requirements-dev.txt` | Editable package plus testing and packaging tools. | Development and verification. |
| `requirements-reproduction.txt` | Additional packages for heavy reproductions: Harmony, Scanorama, JAX/scIB, baselines, and external methods. | Reproduction of the report or advanced external methods. |
| `requirements-llm.txt` | Optional Cell2Sentence/transformer stack. | Only when using LLM-based algorithms. |

Development installation:

```bash
python -m pip install -r requirements-dev.txt
```

Complete installation for reproduction experiments:

```bash
python -m pip install -r requirements-reproduction.txt
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

For the audited public GEO/Figshare/GitHub source URLs, the report-versus-Git
mapping, and known historical discrepancies, see
[docs/dataset_sources.md](docs/dataset_sources.md).

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
  --output results/test_run \
  --no-timestamp
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

The maintained launchers cover the stable-generalist, inductive,
loss-transfer, Harmony, and marker-overlap workflows. scRAW ablation and three
historical Optuna-importance campaigns still lack clean archived launch
manifests; they are listed under **Known Limits** in the reproduction map.

### scRAW Presets

SCRBenchmark exposes exactly two public scRAW presets:

- `default`: the vendored 0017/stable configuration in `vendor/scraw_inductive/configs/`;
- `baron`: the vendored Baron-compatible configuration in `vendor/scraw_inductive/configs/`.

For registered report-method runs, select it with `--scraw-preset default` or
`--scraw-preset baron` when calling `scripts/reproduction/run_method.py`. For
inductive scRAW scripts, use `--preset default` or `--preset baron`.
The data flow, artifact contract, supported losses, and focused smoke tests are
described in [docs/guide/scraw_architecture.md](docs/guide/scraw_architecture.md).

---

## Guides

| Guide | Target Audience | When to use |
| --- | --- | --- |
| [docs/guide/handover_guide.md](docs/guide/handover_guide.md) | New maintainer / intern | Start here to understand scRAW, SCRBenchmark, the tested workflows, and the external assets required for full report reproduction. |
| [docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf](docs/paper/Rapport_Stage_M2_Fabien_Bidet.pdf) | Scientific reader | Read the French scientific report: context, methods, results, limitations, and appendices. |
| [docs/guide/user_guide.md](docs/guide/user_guide.md) | SCRBenchmark User | Understand the workflow: detailed installation, data preparation, GUI, CLI, and report reproduction. |
| [docs/guide/scraw_architecture.md](docs/guide/scraw_architecture.md) | scRAW user / developer | Understand the public pipeline, modes, artifacts, checkpoints, and verification commands. |
| [docs/guide/report_reproduction_steps.md](docs/guide/report_reproduction_steps.md) | Reproduction user | Numbered commands to regenerate report figures and reuse existing artifacts when possible. |
| [docs/guide/dataset_integration_guide.md](docs/guide/dataset_integration_guide.md) | Data user | Add a new `.h5ad` dataset to GUI, CLI, manifests, and reproduction plans. |
| [docs/guide/developer_file_guide.md](docs/guide/developer_file_guide.md) | Developer | Know which file to modify to change the preprocessing, algorithms, interface, metrics, or scripts. |
| [docs/guide/algorithm_extension_guide.md](docs/guide/algorithm_extension_guide.md) | External Algorithm Developer | Single step-by-step guide: external source code, wrapper, YAML, validation, and smoke test. |
| [docs/guide/preprocessing_extension_guide.md](docs/guide/preprocessing_extension_guide.md) | Preprocessing Developer | Add a preprocessing step without train/test leakage and without breaking GUI/CLI. |
| [methods/README.md](methods/README.md) | Method Developer | Understand the role of the `methods/` directory and YAML specifications. |
| [protocols/README.md](protocols/README.md) | Experiment Designer | Understand the format of versioned YAML protocols loadable from Customize Benchmark. |
| [scripts/reproduction/README.md](scripts/reproduction/README.md) | Reproduction / automation | Choose the correct script to generate plans, run methods, or replay report experiments. |
| [data/README.md](data/README.md) | Data User | Prepare datasets and verify the expected `.h5ad` format. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contributor | Set up a development environment and follow the verification and scientific-safety rules. |

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
pyproject.toml          installable Python package and console entry point
src/scrbenchmark/        main code: CLI, GUI, registries, algorithms, utils
docs/                   technical guides and reproduction maps
data/                   local data and documented preparation scripts
methods/                YAML specifications of reproducible methods
protocols/              versioned YAML protocols
reproducibility/         report manifests, tables, and traceability metadata
scripts/setup/          dataset preparation
scripts/reproduction/   reproduction plans, launchers, and executors
tests/                  unit tests and comparisons
vendor/                 public and dedicated scRAW backends and other helpers
external/               external author code kept separately
scraw-transductive-stable-generalist/  preserved enriched checkpoint bundles
results/                local outputs of experiments
.github/workflows/      automated package smoke tests
```

---

## Tests

Install the development tools, then run the fast checks:

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q src scripts vendor/scraw_inductive/src vendor/scraw_dedicated/src
pytest -q -m "not slow"
```

Run the full maintained suite and cross-implementation checks before a release:

```bash
pytest -q
python tests/comparisons/run_all_comparisons.py --tests-only
```

Original-author comparison cases may be skipped when their optional legacy
environment is unavailable. Contributor expectations and scRAW-specific tests
are in [CONTRIBUTING.md](CONTRIBUTING.md).

Check whitespace before committing:

```bash
git diff --check
```
