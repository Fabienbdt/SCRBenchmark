# Other Internship Artifacts - scRAW

This directory groups scRAW artifacts that are useful for internship handoff.

## Contents

- `metadata/stable_generalist_all_results_table.csv`: global stable-generalist
  results table.
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
