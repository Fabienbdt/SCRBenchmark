# Developer Guide: SCRBenchmark File Map

This guide serves as a map to understand the files that make up SCRBenchmark. It indicates where to look to modify a key value, add a function, integrate an algorithm, or extend the preprocessing without breaking the GUI, CLI, and reproduction flows.

For a complete user manual, see [user_guide.md](user_guide.md).
For a new-maintainer onboarding path, see [handover_guide.md](handover_guide.md).
To add an external algorithm step-by-step, see [algorithm_extension_guide.md](algorithm_extension_guide.md).
To add a preprocessing step-by-step, see [preprocessing_extension_guide.md](preprocessing_extension_guide.md).

---

## 1. General Software Flow

SCRBenchmark has three main entry points:

```text
Graphical User Interface
./run.sh
  -> src/scrbenchmark/app.py
  -> src/scrbenchmark/gui/*
  -> src/scrbenchmark/utils/*
  -> src/scrbenchmark/core/algorithm_registry.py
  -> src/scrbenchmark/algorithms/*

Command Line Interface
./scrbenchmark
  -> src/scrbenchmark/cli.py
  -> DataHandler / DatasetSplitter
  -> AnalysisRunner
  -> AlgorithmRegistry
  -> algorithms/*

Experiment Reproduction
Streamlit Report Reproduction or scripts/reproduction/*.py
  -> protocols/*.yaml or methods/*.yaml
  -> run_method.py / run_external_method.py / ./scrbenchmark
  -> results/*
```

Practical rule: Streamlit pages should mainly collect user choices. Reusable scientific logic must remain in `utils/`, `core/`, `algorithms/`, `methods/`, `protocols/`, or `scripts/reproduction/`.

---

## 2. Root Files

| File | Role | When to modify |
| --- | --- | --- |
| `README.md` | Repository entry point: installation, datasets, launching, links to guides. | When the user path changes. |
| `requirements.txt` | Core dependencies: CLI, Streamlit, preprocessing, classical algorithms, and common modules. | When a core feature requires a new package. |
| `requirements-reproduction.txt` | Additional dependencies to reproduce heavy report experiments. | When an external method or reproduction baseline adds a dependency. |
| `run.sh` | Launches the Streamlit app with the correct Python interpreter. | If the port, app path, or environment selection changes. |
| `scrbenchmark` | Bash wrapper for the Python CLI. | Rarely; only if the CLI launch or Python detection changes. |
| `.streamlit/` | Local Streamlit configuration. | For interface or Streamlit server settings. |

---

## 3. Python Core

| File | Role | Important Points |
| --- | --- | --- |
| `src/scrbenchmark/app.py` | Streamlit entry point. Configures the page, initializes state, displays navigation, calls GUI pages. | Adding a page requires importing it here and adding it to the sidebar. |
| `src/scrbenchmark/cli.py` | CLI entry point. Parses arguments, loads data, builds preprocessing, runs analyses, saves results. | Add a CLI option here if it should be available outside the GUI. |
| `src/scrbenchmark/core/config.py` | Central definitions of hyperparameters, types, defaults, and bounds. | Modify `PREPROCESSING_PARAMS`, `CLUSTERING_PARAMS` or global defaults here. |
| `src/scrbenchmark/core/algorithm_registry.py` | Common contract of internal baselines: `BaseAlgorithm`, `AlgorithmInfo`, `AlgorithmRegistry`. | Consult to maintain algorithms already integrated in `src/scrbenchmark/algorithms/`. |
| `src/scrbenchmark/core/optimization.py` | Internal optimization helpers. | Consult before adding new optimization logic. |
| `src/scrbenchmark/methods/registry.py` | Loads declarative specifications `methods/*.yaml`. | Used by reproduction scripts and the Customize Benchmark interface. |
| `src/scrbenchmark/protocols/registry.py` | Loads, validates, and expands versioned YAML protocols. | Key point for job plans and reproduction presets. |

---

## 4. Data, Preprocessing, and Splits

| File | Role | When to modify |
| --- | --- | --- |
| `src/scrbenchmark/utils/data_handler.py` | Loads `.h5ad`, `.h5`, `.csv`, `.tsv`, `.mtx` files; extracts labels; applies preprocessing in standard mode. | Add an input format or a standard preprocessing step. |
| `src/scrbenchmark/utils/dataset_splitter.py` | Creates train/val/test splits, batch splits, balancing, and then applies a learned preprocessing on train only via `BenchmarkPreprocessor`. | Modify split strategies, data leakage prevention, or preprocessing in benchmark mode. |
| `src/scrbenchmark/utils/dropout_simulation.py` | Simulates dropout and noise on counts. | Add a new data degradation. |
| `src/scrbenchmark/utils/batch_correction.py` | scVI/sysVI/DCT-Corr batch correction. | Modify batch corrections or add a correction method. |
| `src/scrbenchmark/utils/metrics.py` | Calculates standard and benchmark metrics: NMI, ARI, silhouette, scIB, generalization gaps, etc. | Add a metric or change evaluation. |
| `src/scrbenchmark/utils/analysis_runner.py` | Executes algorithms, retrieves labels/embeddings, calculates metrics, saves results. | Modify the execution contract, output artifacts, or train/test logic. |
| `src/scrbenchmark/utils/hyperparam_search.py` | Hyperparameter search, aggregation, and saving search results. | Add a tuning strategy. |
| `src/scrbenchmark/utils/pca_utils.py` | Common PCA functions. | Modify shared PCA heuristics. |
| `src/scrbenchmark/utils/statistics.py` | Statistical tests and syntheses. | Add or adjust statistical analyses. |
| `src/scrbenchmark/utils/visualization.py` | Shared visualizations outside specialized GUI pages. | Add reusable figures. |
| `src/scrbenchmark/utils/reference_datasets.py` | Reference datasets proposed in the interface. | Add a preconfigured dataset. |
| `src/scrbenchmark/utils/ascdt.py` | Functions related to ASCDT/DCT variants. | Modify with caution, as it is linked to correction protocols. |

Important point: in benchmark mode, any step that learns parameters must learn on train only in `BenchmarkPreprocessor.fit()`, then reuse these parameters in `transform()`.

---

## 5. Streamlit Interface

| File | Role |
| --- | --- |
| `src/scrbenchmark/gui/state_manager.py` | Contract of `st.session_state` keys and cascade invalidation. |
| `src/scrbenchmark/gui/widgets.py` | Reusable widgets: parameter inputs, exports, batch/label detection. |
| `src/scrbenchmark/gui/data_upload.py` | Data loading, QC, label harmonization, reference datasets. |
| `src/scrbenchmark/gui/data_split.py` | Configuration of standard, stratified, or batch splits. |
| `src/scrbenchmark/gui/preprocessing.py` | Interactive preprocessing panel. |
| `src/scrbenchmark/gui/algorithm_config.py` | Selection of algorithms, hyperparameters, CLI command generation. |
| `src/scrbenchmark/gui/analysis/` | Pages and execution engine for standard and benchmark analyses. |
| `src/scrbenchmark/gui/customize_benchmark.py` | Construction of experiments, report presets, command generation. |
| `src/scrbenchmark/gui/protocol_designer.py` | Interface to load/validate YAML protocols. |
| `src/scrbenchmark/gui/report_reproduction.py` | Streamlit launchers to reproduce report experiments. |
| `src/scrbenchmark/gui/results_explorer/` | Loading, filtering, tables, and figures to analyze `results/` folders. |
| `src/scrbenchmark/gui/hyperparam_search.py` | Hyperparameter search interface. |
| `src/scrbenchmark/gui/latent_reclustering.py` | Reclustering from existing embeddings. |
| `src/scrbenchmark/gui/documentation.py` | Integrated documentation in the app. |
| `src/scrbenchmark/gui/i18n.py` | Translatable interface texts. |

Before adding a new GUI state, check `state_manager.py`. If an action invalidates results, add the dependency in `DEPENDENCY_GRAPH`.

---

## 6. Maintained Internal Algorithms

Local Python algorithms live in:

```text
src/scrbenchmark/algorithms/
```

Main files:

| File | Role |
| --- | --- |
| `pca.py` | PCA baseline with configurable clustering. |
| `pca_kmeans.py` | PCA + K-Means baseline. |
| `pca_leiden.py` | PCA + Leiden baseline. |
| `scdeepcluster.py` | Local port of scDeepCluster. |
| `scdeepcluster_scraw_weighted.py` | scDeepCluster variant with scRAW weights. |
| `sccdcg.py` | scCDCG implementation. |
| `sc_mae.py` | Local port of scMAE. |
| `sc_mae_scraw_weighted.py` | scMAE variant with scRAW weights. |
| `scname.py` | Local port of scNAME. |
| `template_algorithm.py` | Old internal skeleton; do not use as the main path for new external integrations. |
| `__init__.py` | Auto-discovers modules and triggers registration. |

This directory is for baselines and internal ports already maintained in the SCRBenchmark engine. To add a new algorithm from external source code, use the single path described in [algorithm_extension_guide.md](algorithm_extension_guide.md), based on `external/original_code/` and `methods/*.yaml`.

---

## 7. External Algorithms and Reproducible Methods

Reproducible methods are declared in:

```text
methods/*.yaml
```

These files are the public reproduction contract. They tell SCRBenchmark how to launch a method, where its source code is, what parameters to pass, and where to read the produced labels/embeddings.

| File/Folder | Role |
| --- | --- |
| `methods/README.md` | Detailed contract of YAML specs. |
| `methods/template_method.yaml` | Specification template. |
| `methods/report_methods.yaml` | Methods used in report tables. |
| `methods/PARC.yaml` | Tested example of external integration. |
| `external/original_code/` | Author code kept as close as possible to the original. |
| `vendor/` | Vendored backends or helpers used by integrations. |

The single guide to add an external algorithm is [algorithm_extension_guide.md](algorithm_extension_guide.md). It covers:

- the external source code directory;
- the `scrbenchmark_wrapper.py` wrapper;
- the `methods/<algo>.yaml` file;
- validation with `validate_method.py`;
- the real smoke test.

---

## 8. Protocols, Plans, and Reproduction

| Location | Role |
| --- | --- |
| `protocols/*.yaml` | Versioned protocols loadable from Customize Benchmark. |
| `protocols/report/*.yaml` | Report protocols: Baron, splits, Harmony, inductive, loss transfer. |
| `protocols/README.md` | Format and role of protocols. |
| `scripts/reproduction/README.md` | Detailed map of reproduction scripts. |
| `scripts/reproduction/build_stable_generalist_plan.py` | Generates `planned_jobs.csv` and `run_ready_jobs.sh` for stable_generalist. |
| `scripts/reproduction/download_datasets.py` | Downloads or materializes the 13 stable_generalist datasets and verifies SHA256/dimensions. |
| `scripts/reproduction/prepare_stable_generalist_data.py` | Old local helper: materializes the 13 datasets from an already present source directory. |
| `scripts/reproduction/build_report_plan.py` | Generates plans for report complements. |
| `scripts/reproduction/manual_protocols.py` | Generates configurable loss-transfer, Harmony, and inductive jobs. |
| `scripts/reproduction/run_external_method.py` | Launches multiple external methods via vendored executors. |
| `scripts/reproduction/run_posthoc_harmony.py` | Applies Harmony post-hoc on method embeddings. |
| `scripts/reproduction/run_scrbenchmark_leave_one_batch.py` | Runs leave-one-batch experiences via the SCRBenchmark CLI. |
| `scripts/reproduction/run_scraw_leave_one_batch.py` | Runs inductive scRAW per held-out batch. |
| `scripts/reproduction/run_shared_train_inductive_algorithms.py` | Runs the inductive train-groups/test-groups protocol. |
| `scripts/reproduction/run_marker_overlap.py` | Calculates DEG top-100 overlap from saved labels. |
| `scripts/reproduction/run_batch_baselines.py` | Runs Harmony, ComBat, Scanorama, PCA+Leiden, scVI, and other batch correction baselines. |

For normal report reproduction, prefer the Streamlit tab `Report Reproduction`, which generates these plans without copying long commands.

---

## 9. Datasets and Preparation

| Location | Role |
| --- | --- |
| `data/README.md` | Expected dataset format and Baron pancreas generation. |
| `data/GSE84133_RAW/` | GEO raw data tracked by Git to reconstruct Baron. |
| `data/stable_generalist/README.md` | List of expected files for the stable_generalist campaign. |
| `scripts/setup/prepare_baron_dataset.py` | Builds `data/baron_human_pancreas.h5ad`. |
| `scripts/setup/reconstruct_pancreas_raw_counts.py` | Reconstructs multi-study pancreas sets with raw counts. |

Recommended format:

```text
adata.X              expression matrix
adata.obs["Group"]  biological labels, or explicit selection via --label-col
adata.obs["batch"]  batch/donor/dataset, optional but useful
```

---

## 10. Where to Modify a Key Value

| Goal | Files to look at |
| --- | --- |
| Change a preprocessing default | `src/scrbenchmark/core/config.py`; complete procedure in `docs/guide/preprocessing_extension_guide.md`. |
| Add a CLI option | `src/scrbenchmark/cli.py`, then connect this option to `DataHandler`, `DatasetSplitter` or `AnalysisRunner`. |
| Add a Streamlit control | Relevant `src/scrbenchmark/gui/*.py` page, often with `widgets.py` and `state_manager.py`. |
| Modify a metric | `src/scrbenchmark/utils/metrics.py`, then displays in `results_explorer/`. |
| Modify results saving | `src/scrbenchmark/utils/analysis_runner.py` and loaders in `gui/results_explorer/`. |
| Modify a hyperparameter of an existing internal baseline | `get_hyperparameters()` method in `src/scrbenchmark/algorithms/<algo>.py`. |
| Modify an external method parameter | YAML in `methods/`, or wrapper in `external/original_code/<method>/`. |
| Add a dataset preset | `src/scrbenchmark/utils/reference_datasets.py`, `data/README.md`, and possibly `reproducibility/`. |
| Add a reproducible protocol | New YAML in `protocols/`, then validation via the interface or `protocols/registry.py`. |

---

## 11. Preprocessing

This map indicates the files involved, but the procedure for adding a preprocessing step lives in [preprocessing_extension_guide.md](preprocessing_extension_guide.md). Use this dedicated guide for train/test leakage, GUI, CLI, protocols, and tests.

---

## 12. Useful Tests

Basic commands:

```bash
pytest tests/unit_tests/test_algorithm_registry.py
pytest tests/unit_tests/test_config.py
pytest tests/unit_tests/test_data_handler.py
pytest tests/unit_tests/test_benchmark_preprocessor_filters.py
pytest tests/unit_tests/test_end_to_end.py
```

Method validation commands:

```bash
python3 scripts/reproduction/validate_method.py --method PARC --data data/baron_human_pancreas.h5ad --label-key Group --batch-key batch --n-labels 14
./scrbenchmark list-algorithms
./scrbenchmark generate-config --output config.yaml
```

For documentation changes only, check at least:

```bash
git diff --check
```

---

## 13. Principles to Keep in Mind

- Keep Streamlit pages thin: shared logic should live in `utils/` or `core/`.
- Keep external reproducible methods in `methods/*.yaml` with thin wrappers.
- Preserve raw counts in `adata.layers["original_X"]` when an algorithm needs them.
- In benchmark mode, avoid any data leakage: fit on train, transform on val/test.
- Add tests proportional to risk: preprocessing and splits deserve stricter tests than simple display changes.
