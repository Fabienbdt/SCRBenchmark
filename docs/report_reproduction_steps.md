# Reproduce Report Figures Step by Step

This guide gives the command order for regenerating SCRBenchmark report outputs.
Commands start from the repository root.

The report being reproduced is
[`Rapport_Stage_M2_Fabien_Bidet.pdf`](Rapport_Stage_M2_Fabien_Bidet.pdf). New
maintainers should first read [`handover_guide.md`](handover_guide.md), which
states the required external datasets and legacy environments.

```bash
cd /path/to/SCRBenchmark
```

The goal is to avoid rerunning heavy experiments unnecessarily. Existing
artifacts are reused whenever possible.

---

## 0. Install the Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-reproduction.txt
```

Quick check:

```bash
./scrbenchmark list-algorithms
```

---

## 1. Download or Materialize Datasets

The 13 `stable_generalist` datasets must be in:

```text
data/stable_generalist/
```

From the local reference copy:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/existing/h5ad/files
```

If the exact files are hosted remotely:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

Verify without downloading anything:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

Verification is read-only. Add `--report results/dataset_verification.csv` only
when a persistent operation report is desired.

An independent metadata table can optionally be checked with
`--reference-table /path/to/stable_generalist_dataset_table.csv`.

---

## 2. Check Existing scRAW Artifacts

Before rerunning experiments, inventory existing weights and models:

```bash
python scripts/reproduction/export_existing_scraw_artifacts.py \
  --weights-root /path/to/scraw_transductive_runs \
  --model-root /path/to/scraw_inductive_runs \
  --dry-run
```

Current local findings:

- seed-60 scRAW cell weights already exist for the 13 datasets in the
  `scRAW_default_from_scRAW_seed60_stage_umaps_20260421` campaign;
- the exact Baron scRAW labels from the report exist in
  `Rapport_Stage_M2_git/Images/analyse_biologique_scraw_baron_stable_generalist/tsne_coordinates.csv`
  and can be reused for marker-overlap analysis without rerunning the model;
- `autoencoder.pt` checkpoints exist for several inductive scRAW experiments,
  but not for all 13 transductive stable-generalist datasets. The datasets
  without a model checkpoint found in the current local state are
  `bbag094_zeisel`, `pancreas_raw_counts`, `paul15_bone_marrow_raw_counts`,
  `Human_Pancreas_1_raw_counts`, `Human_Pancreas_2_raw_counts`,
  `Mouse_Pancreas_1_raw_counts`, and `Tabula_Muris_liver_filtered_raw_counts`.

Export existing artifacts without rerunning:

```bash
python scripts/reproduction/export_existing_scraw_artifacts.py \
  --weights-root /path/to/scraw_transductive_runs \
  --model-root /path/to/scraw_inductive_runs \
  --output-root results/report_artifacts/scraw_existing_artifacts
```

The main produced file is:

```text
results/report_artifacts/scraw_existing_artifacts/scraw_existing_artifacts_manifest.csv
```

---

## 3. Regenerate Main stable_generalist Figures

Generate the plan:

```bash
python scripts/reproduction/build_stable_generalist_plan.py \
  --output-root results/stable_generalist_repro \
  --python-bin "$(which python)" \
  --device cuda \
  --seed 42
```

Missing datasets are marked `blocked_missing_data` and omitted from the shell
launcher by default. `--allow-missing-data` is an explicit unsafe override.

Run the jobs:

```bash
bash results/stable_generalist_repro/run_ready_jobs.sh
```

These outputs feed, among others:

- `fig:scraw_common8_family_top3_plus_scraw`;
- `fig:rank_matrix_common8`;
- `fig:common8_family_all_methods`;
- `tab:runtime_moyen_algorithmes`;
- `tab:scraw_holdout_pancreas_results`.

The complete mapping is in
[`docs/report_reproduction_map.md`](report_reproduction_map.md).

---

## 4. Regenerate Loss-Transfer and Generalization Complements

Fixed report plan:

```bash
python scripts/reproduction/build_report_plan.py \
  --output-root results/report_repro \
  --python-bin "$(which python)" \
  --device cuda \
  --campaigns inductive,loss_transfer,deg
```

As with the stable-generalist planner, missing `.h5ad` inputs are blocked by
default and never inserted into the runnable launcher.

Run:

```bash
bash results/report_repro/run_ready_report_jobs.sh
```

From the interface:

```bash
./run.sh
```

Then open `Report Reproduction`:

- `scRAW Weighting` to select loss-transfer algorithms;
- `Generalization` to select inductive algorithms;
- `Biological Interpretation` for Baron marker-overlap.

---

## 5. Regenerate Only Baron Marker-Overlap

The planner can reuse the exact Baron scRAW report labels when their path is
provided. They come from `tsne_coordinates.csv` with `true_label` and
`predicted_label` columns, so no new source scRAW run is planned.

```bash
python scripts/reproduction/build_report_plan.py \
  --output-root results/report_repro \
  --python-bin "$(which python)" \
  --device cuda \
  --campaigns deg \
  --existing-report-baron-labels /path/to/tsne_coordinates.csv
```

Run only the generated launcher:

```bash
bash results/report_repro/run_ready_report_jobs.sh
```

Expected outputs:

```text
results/report_repro/deg_marker_overlap/baron_human_pancreas/marker_overlap/results/marker_overlap_matrix.csv
results/report_repro/deg_marker_overlap/baron_human_pancreas/marker_overlap/results/degs_top100.json
results/report_repro/deg_marker_overlap/baron_human_pancreas/marker_overlap/results/marker_overlap_genes_long.csv
results/report_repro/deg_marker_overlap/baron_human_pancreas/marker_overlap/figures/marker_overlap_heatmap.png
```

Omit the existing-label option, or explicitly add the following flag, to force
a new source scRAW run:

```bash
--no-reuse-existing-artifacts
```

---

## 6. Extract Already Produced Marker-Overlap DEG Genes

If a `degs_top100.json` already exists:

```bash
python scripts/reproduction/extract_marker_overlap_genes.py \
  --input results/report_repro/deg_marker_overlap/baron_human_pancreas/marker_overlap/results/degs_top100.json \
  --output-dir results/report_artifacts/marker_overlap_genes
```

The script produces:

- `*_ground_truth_degs.csv`;
- `*_predicted_cluster_degs.csv`;
- `*_marker_overlap_genes_long.csv`;
- `*_cluster_to_type.csv`.

Current local finding after verification: several marker-overlap heatmaps and
matrices already exist, but no stable `degs_top100.json` was found in the
inspected directories. Heatmaps alone do not contain the gene list, so either
recover that JSON or rerun only the marker-overlap analysis from existing scRAW
labels.

---

## 7. Replay scRAW From Available Weights

For a complete inductive checkpoint:

```bash
python scripts/reproduction/run_scraw_from_weights.py \
  --mode inductive \
  --config results/report_artifacts/scraw_existing_artifacts/configs/config_DATASET.json \
  --checkpoint results/report_artifacts/scraw_existing_artifacts/models/model_DATASET.pt \
  --preprocessing-state results/report_artifacts/scraw_existing_artifacts/models/preprocessing_state_DATASET.npz \
  --centroid-reference results/report_artifacts/scraw_existing_artifacts/models/centroid_reference_DATASET.npz \
  --data data/stable_generalist/DATASET.h5ad \
  --output results/replay_scraw/DATASET \
  --device cuda
```

For a transductive checkpoint compatible with `scraw_inductive`:

```bash
python scripts/reproduction/run_scraw_from_weights.py \
  --mode transductive \
  --config path/to/config_used.json \
  --checkpoint path/to/autoencoder.pt \
  --data data/stable_generalist/DATASET.h5ad \
  --output results/replay_scraw/DATASET \
  --device cuda
```

---

## 8. Regenerate a scRAW Cell-Weight Figure

Example for Baron:

```bash
python scripts/reproduction/regenerate_scraw_weight_figures.py \
  --data data/stable_generalist/baron_human_pancreas.h5ad \
  --weights-csv results/report_artifacts/scraw_existing_artifacts/weights/cell_weights_baron_human_pancreas.csv \
  --output results/report_artifacts/scraw_weight_figures/baron_human_pancreas_weights.png \
  --label-key label \
  --batch-key batch
```

This command recomputes only a UMAP visualization from an existing weight file;
it does not retrain scRAW.
