# Other Internship Artifacts - scRAW

This directory groups scRAW artifacts that are useful for internship handoff.

## Contents

- `metadata/stable_generalist_all_results_table.csv`: global stable-generalist
  results table.
- `marker_overlap_analysis/`: Baron scRAW biological interpretation artifacts:
  marker-overlap matrix, top-100 DEG gene lists, annotation metrics, and
  report figures.
- `scraw_model_weights/checkpoints/`: reserved location for transductive scRAW
  model checkpoints corresponding to the scRAW rows in
  `metadata/stable_generalist_all_results_table.csv`.
- `scraw_model_weights/preprocessing_state/`: preprocessing states associated
  with already exported scRAW artifacts.
- `scraw_model_weights/centroid_reference/`: centroid references associated
  with already exported scRAW artifacts.
- `scraw_model_weights/configs/`: JSON configurations associated with already
  exported scRAW artifacts.
- `scraw_cell_weights/`: exported scRAW cell weights.
  - 13 `cell_weights_*.csv` files, one per stable-generalist dataset.
  - 9 `train_cell_weights_*.npy` files for available inductive runs.
- `scraw_cell_weight_umaps/`: 13 UMAP figures colored by scRAW reconstruction
  weight.
- `scripts/`: scripts used to export artifacts, replay scRAW from available
  weights, and regenerate UMAP figures.

## Current Coverage

scRAW cell weights are available for the 13 stable-generalist datasets.

The transductive model checkpoints corresponding to the 13 `method=scRAW` rows
in `metadata/stable_generalist_all_results_table.csv` were not present in the
local `scRAW_EXPERIMENTAL` results. The `scraw_model_weights/checkpoints/`
directory is therefore left empty here to avoid mixing transductive results with
inductive checkpoints available elsewhere.

## Notes

The figures in `scraw_cell_weight_umaps/` correspond to scRAW cell weights,
dataset by dataset. They were regenerated from exported weight files with
`scripts/regenerate_scraw_weight_figures.py`.

The files in this directory are copies of validated artifacts from
`results/report_artifacts/`.

`metadata/stable_generalist_all_results_table.csv` contains benchmark metrics
and source result paths. It does not directly contain model weight files.

## Handoff FAQ

### Where is the global results table?

Use:

```text
autres-stage/metadata/stable_generalist_all_results_table.csv
```

This is the table to upload when a single CSV with all stable-generalist
benchmark rows is needed.

Load it with:

```python
import pandas as pd

results = pd.read_csv("autres-stage/metadata/stable_generalist_all_results_table.csv")
```

### Where is the DEG / marker-overlap analysis?

Use:

```text
autres-stage/marker_overlap_analysis/
```

Important files:

- `degs_top100.json`: original top-100 DEG payload.
- `ground_truth_degs.csv`: top-100 genes per ground-truth cell type.
- `predicted_cluster_degs.csv`: top-100 genes per predicted cluster.
- `marker_overlap_genes_long.csv`: concatenated long-format DEG table.
- `marker_overlap_matrix.csv`: overlap matrix used for annotation.
- `annotation_metrics_summary.csv`: marker/Hungarian annotation summary.
- `marker_overlap_heatmap.png`: heatmap figure.
- `tsne_coordinates.csv`, `tsne_annotations_comparison.png`,
  `tsne_qualitative_panel.png`: report visual artifacts.

The source copy was:

```text
/data2/fbidet/Rapport_Stage_M2_git/Images/analyse_biologique_scraw_baron_stable_generalist/
```

Load the tables with:

```python
import pandas as pd

ground_truth_degs = pd.read_csv("autres-stage/marker_overlap_analysis/ground_truth_degs.csv")
predicted_cluster_degs = pd.read_csv("autres-stage/marker_overlap_analysis/predicted_cluster_degs.csv")
marker_overlap = pd.read_csv("autres-stage/marker_overlap_analysis/marker_overlap_matrix.csv", index_col=0)
```

To regenerate extracted CSV files from the JSON payload:

```bash
python scripts/reproduction/extract_marker_overlap_genes.py \
  --input autres-stage/marker_overlap_analysis/degs_top100.json \
  --output-dir /tmp/marker_overlap_genes
```

To recompute the marker-overlap analysis from labels instead of reusing the
saved payload:

```bash
python scripts/reproduction/run_marker_overlap.py \
  --data data/baron_human_pancreas.h5ad \
  --labels-csv scraw-transductive-stable-generalist/runs/baron_human_pancreas/seed_42/results/labels/labels_scraw_run0.csv \
  --output /tmp/baron_marker_overlap \
  --label-key cell_type \
  --true-label-col true_label \
  --pred-label-col predicted_label \
  --n-top-genes 100 \
  --method wilcoxon
```

### Does `scraw_reconstruction_weight` correspond to omega?

Yes. In the exported `cell_weights_*.csv` files,
`scraw_reconstruction_weight` is the final per-cell scRAW weight, i.e. the
omega value used to weight cells during training. The name is historical: the
value is the fused dynamic cell weight, not a raw reconstruction error.

Load one dataset with:

```python
import pandas as pd

weights = pd.read_csv("autres-stage/scraw_cell_weights/cell_weights_Human_Pancreas_1_raw_counts.csv")
omega = weights["scraw_reconstruction_weight"].to_numpy()
```

### Where are the UMAP figures colored by weights?

Use:

```text
autres-stage/scraw_cell_weight_umaps/
```

These PNG files were regenerated from `autres-stage/scraw_cell_weights/` with:

```bash
python autres-stage/scripts/regenerate_scraw_weight_figures.py \
  --data data/stable_generalist/Human_Pancreas_1_raw_counts.h5ad \
  --weights-csv autres-stage/scraw_cell_weights/cell_weights_Human_Pancreas_1_raw_counts.csv \
  --output /tmp/Human_Pancreas_1_raw_counts_scraw_cell_weights_umap.png \
  --label-key label \
  --batch-key batch
```

### Which folders should be uploaded to a shared storage area?

For a compact handoff, upload:

```text
autres-stage/metadata/
autres-stage/marker_overlap_analysis/
autres-stage/scraw_cell_weights/
autres-stage/scraw_cell_weight_umaps/
```

Add `autres-stage/scraw_model_weights/` only if the downstream user needs
inductive checkpoint-side artifacts, preprocessing states, or centroid
references.

### Where are the report and presentation sources?

The report repository is:

```text
git@github.com:Fabienbdt/Rapport_Stage_M2.git
https://github.com/Fabienbdt/Rapport_Stage_M2
```

The local checkout is:

```text
/data2/fbidet/Rapport_Stage_M2_git
```

Oral-presentation artifacts are also present locally in:

```text
/data2/fbidet/Rapport_Stage_M2_git/Images/PresentationOrale/
/data2/fbidet/slides/
```

`/data2/fbidet/slides/` is a local slide workspace and is not itself a git
repository.
