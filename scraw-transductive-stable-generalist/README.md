# scRAW Transductive stable_generalist

This directory contains the transductive scRAW rerun on the 13
stable-generalist datasets, with explicit final model checkpoint saving.

## Organization

- `metadata/stable_generalist_all_results_table.csv`: reference table of
  results to reproduce.
- `metadata/stable_generalist_dataset_table.csv`: table of the 13 datasets.
- `metadata/stable_generalist_trial_config.json`: source scRAW configuration.
- `runs/<dataset_key>/seed_42/`: complete scRAW rerun outputs.
- `model_weights/checkpoints/model_<dataset_key>.pt`: enriched final
  checkpoint.
- `figures/umaps/<dataset_key>_batch_label_groundtruth_weights_umap.png`:
  combined UMAP for batch, predicted label, ground truth, and scRAW cell
  weight.
- `validation/metrics_comparison.csv`: comparison between source metrics and
  regenerated metrics.
- `validation/accepted_differences.md`: review of the only accepted difference.
- `run_status.csv`: per-dataset status.

## Validation

`validation/metrics_comparison.csv` keeps the raw comparison against the
presentation table: 12 datasets are `ok` and `pancreas_raw_counts` is marked
`different`.

This difference is accepted. The presentation row for `pancreas_raw_counts`
points to source/archive paths that are no longer available exactly as written,
whereas the preserved `stable_generalist_trial_0017` directory contains a
`results/clustering_final/final_clustering_comparison.csv` that is byte-for-byte
identical to the rerun. The regenerated metrics are also better than the
presentation row metrics. See `validation/accepted_differences.md`.

## Rerun the 13 Datasets

The following command retrains the 13 models. It is intended for the original
research workspace and requires the source trial directory passed with
`--trial-root`, plus the exact datasets. Replaying the versioned checkpoints
does not require retraining; use the next section instead.

From the SCRBenchmark root:

```bash
python \
  scripts/reproduction/rerun_scraw_transductive_with_checkpoints.py \
  --source-results-table /path/to/stable_generalist_all_results_table.csv \
  --dataset-table /path/to/stable_generalist_dataset_table.csv \
  --trial-root /path/to/stable_generalist_trial_0017 \
  --gpus 1,2
```

The script uses the public scRAW `default` preset, which is the 0017/stable
configuration, seed 42, the source campaign hyperparameters, and GPU execution.
It reruns training because the original transductive checkpoints were not
retained in the source results.

The launcher also sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0`,
`OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` in each worker to limit GPU
nondeterminism. The final validation remains `validation/metrics_comparison.csv`.

## Replay a Result From a Checkpoint

Example for Kang PBMC:

```bash
python \
  scripts/reproduction/replay_scraw_transductive_checkpoint.py \
  --checkpoint scraw-transductive-stable-generalist/model_weights/checkpoints/model_kang_pbmc_gse96583_singlets_raw_counts.pt \
  --config scraw-transductive-stable-generalist/runs/kang_pbmc_gse96583_singlets_raw_counts/seed_42/config/config_used.json \
  --data data/stable_generalist/kang_pbmc_gse96583_singlets_raw_counts.h5ad \
  --output scraw-transductive-stable-generalist/replayed/kang_pbmc_gse96583_singlets_raw_counts \
  --device auto
```

Replay reloads the model `state_dict`, recomputes embeddings and final
clustering, then regenerates per-cell CSV files and the combined UMAP. Final
cell weights are read from the enriched checkpoint: they are a dynamic training
artifact, not a direct function of the network alone.

All 13 enriched transductive checkpoints and their matching configurations are
tracked in this repository. The matching H5AD input is still required because
the checkpoint stores the model and dynamic weights, not the expression matrix.

Associated scripts:

- `scripts/reproduction/rerun_scraw_transductive_with_checkpoints.py`
- `scripts/reproduction/replay_scraw_transductive_checkpoint.py`
