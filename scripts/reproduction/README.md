# Reproduction Scripts

This directory contains the low-level scripts that prepare, plan, or run
SCRBenchmark reproduction experiments.

For normal report reproduction, prefer the Streamlit **Report Reproduction**
panel. It generates job plans and shell launchers without asking users to copy
long command lines by hand.

---

## Scope

This README explains the role of each script. It is not the external algorithm
integration guide and it is not the preprocessing contract.

| Need | Document |
| --- | --- |
| Add an external algorithm | [`../../docs/algorithm_extension_guide.md`](../../docs/algorithm_extension_guide.md) |
| Understand `methods/` YAML files | [`../../methods/README.md`](../../methods/README.md) |
| Add preprocessing | [`../../docs/preprocessing_extension_guide.md`](../../docs/preprocessing_extension_guide.md) |
| Understand repository files | [`../../docs/developer_file_guide.md`](../../docs/developer_file_guide.md) |

---

## Scripts By Task

| Task | Scripts |
| --- | --- |
| Prepare report datasets | `download_datasets.py`, `prepare_stable_generalist_data.py` |
| Build job CSVs and shell launchers | `build_stable_generalist_plan.py`, `build_report_plan.py`, `manual_protocols.py` |
| Reuse existing scRAW artifacts | `export_existing_scraw_artifacts.py`, `run_scraw_from_weights.py`, `regenerate_scraw_weight_figures.py` |
| Extract marker-overlap DEG genes | `extract_marker_overlap_genes.py`, `run_marker_overlap.py` |
| Validate or run a registered external algorithm | `validate_method.py`, `run_method.py` |
| Run external or legacy method families | `run_external_method.py`, `run_scaide_inductive_embeddings.py` |
| Run batch correction or Harmony variants | `run_batch_baselines.py`, `run_posthoc_harmony.py` |
| Run inductive or leave-one-batch protocols | `run_scrbenchmark_leave_one_batch.py`, `run_scraw_leave_one_batch.py`, `run_shared_train_inductive_algorithms.py` |
| Compute report annotations | `run_marker_overlap.py` |

---

## Script Roles

- `download_datasets.py`: downloads or materializes the exact 13 `.h5ad` files
  used by the stable_generalist campaign, then verifies SHA256 hashes, file
  sizes, AnnData dimensions, and label/batch columns.
- `prepare_stable_generalist_data.py`: older local helper that materializes the
  13 `.h5ad` files by hardlink, symlink, or copy.
- `build_stable_generalist_plan.py`: reads stable_generalist reproducibility
  tables and writes `planned_jobs.csv` plus `run_ready_jobs.sh`.
- `build_report_plan.py`: builds plans for report complements: inductive
  protocols, loss-transfer, and DEG overlap. The DEG campaign can reuse exact
  Baron report labels when passed with `--existing-report-baron-labels`.
- Both planners mark missing dataset files as `blocked_missing_data` and omit
  them from runnable launchers. `--allow-missing-data` is available only for
  deliberately generating commands whose data will be mounted later.
- `manual_protocols.py`: creates or runs configurable jobs for loss-transfer,
  Harmony variants, and inductive splits.
- `export_existing_scraw_artifacts.py`: inventories and exports already
  available scRAW cell weights and discovered inductive `autoencoder.pt`
  checkpoints. It does not rerun experiments.
- `run_scraw_from_weights.py`: replays scRAW inference from an existing
  checkpoint and its configuration.
- `regenerate_scraw_weight_figures.py`: regenerates a cell-weight UMAP from an
  existing `.h5ad` file and weight CSV.
- `extract_marker_overlap_genes.py`: converts an existing `degs_top100.json`
  into CSV files split by gene/cluster type; it does not recompute DEGs.
- `add_method.py`: optional low-level helper that creates a starter YAML in
  `methods/`; the user-facing path remains
  [`docs/algorithm_extension_guide.md`](../../docs/algorithm_extension_guide.md).
- `validate_method.py`: loads a `methods/*.yaml` specification, prints the
  final command, and can run a smoke test with `--run`.
- `run_method.py`: uniform launcher driven by `methods/*.yaml`. This is the
  public entry point to execute a method on a dataset.
- `run_external_method.py`: runs historical external integrations such as DESC,
  DeepScena, CellSIUS, GiniClust, scAIDE, scCAD, and Harmony variants.
- `run_scaide_inductive_embeddings.py`: AIDE/scAIDE adapter for the shared
  inductive protocol; it expects a TensorFlow 1.14-compatible interpreter via
  `SCAIDE_PYTHON`.
- `run_batch_baselines.py`: runs the report batch-correction baselines:
  Harmony, ComBat, ComBat-seq, Scanorama, PCA+Leiden, and scVI.
- `run_posthoc_harmony.py`: runs scNAME, scMAE, or scVI, applies Harmony to the
  embedding, and reclusters for `+Harmony` rows.
- `run_scrbenchmark_leave_one_batch.py`: runs one SCRBenchmark CLI job per
  held-out batch.
- `run_scraw_leave_one_batch.py`: runs inductive scRAW per held-out batch via
  `vendor/scraw_inductive`.
- `run_shared_train_inductive_algorithms.py`: runs the representative inductive
  report protocol for scRAW, scNAME, scMAE, scDeepCluster, scAIDE, and
  PCA+Harmony.
- `run_marker_overlap.py`: computes top-100 DEG overlap from saved labels.

Older long-running executors remain in `vendor/stable_generalist_runners/`.
This directory exposes only small entry points.

---

## Recommended Flow

To reproduce the report, use **Report Reproduction** in Streamlit:

```bash
./run.sh
```

Use the scripts in this directory directly only for automation, debugging, or
headless environments. Every script exposes `--help`.

scRAW has exactly two public presets in these scripts:

- `default`: vendored 0017/stable configuration in `vendor/scraw_inductive/configs/`;
- `baron`: vendored Baron-compatible configuration in `vendor/scraw_inductive/configs/`.

Use `--scraw-preset default|baron` with `run_method.py`; use
`--preset default|baron` with the inductive scRAW scripts.

To add a new external algorithm, follow the single guide:
[`../../docs/algorithm_extension_guide.md`](../../docs/algorithm_extension_guide.md).

The numbered command order for regenerating report figures is in:
[`../../docs/report_reproduction_steps.md`](../../docs/report_reproduction_steps.md).
