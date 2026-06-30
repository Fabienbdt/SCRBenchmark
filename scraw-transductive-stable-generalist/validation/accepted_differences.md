# Accepted Differences After Review

Review date: 2026-06-30

## Summary

The raw validation file `metrics_comparison.csv` compares regenerated results
against the presentation table
`metadata/stable_generalist_all_results_table.csv`. It contains 12 datasets
marked `ok` and one dataset marked `different`: `pancreas_raw_counts`.

This difference is accepted and should be ignored for the scRAW rerun
validation.

## pancreas_raw_counts

The presentation scRAW row for `pancreas_raw_counts` reports:

- `selected_final_method=leiden_target14_final`
- `n_clusters_found=26`
- NMI `0.7258628448549318`
- ARI `0.5038730609858753`
- ACC `0.6638414634146341`
- F1_Macro `0.3619167088092313`
- BalancedACC `0.6575788271063779`
- RareACC `0.8626077586206896`
- UltraRareACC `0.7351351351351352`

The rerun reports for `hdbscan_final`:

- `n_clusters_found=24`
- NMI `0.7853832306090607`
- ARI `0.7062141556149881`
- ACC `0.7535551381808425`
- F1_Macro `0.4465807470679713`
- BalancedACC `0.7679456038984988`
- RareACC `0.9322416713721061`
- UltraRareACC `0.8054054054054054`

The rerun is therefore better on every compared metric in this row, except for
the cluster count, which is not a score to maximize.

## Diagnostic

The presentation row points to two source paths that are no longer available on
disk exactly as written:

- `.../stage1/runs/stable_generalist_stable_generalist/pancreas_raw_counts/seed_42`
- `.../presentation_stable_generalist_nonbaron_20260324__archive_not_needed_for_data_exploration_20260421/...`

The preserved trial directory on disk is:

`/data2/fbidet/scRAW_EXPERIMENTAL/results/optuna_stable_generalist_search_20260415_161134/phase1/stable_generalist/stage1/runs/stable_generalist_trial_0017/pancreas_raw_counts/seed_42`

Its file:

`results/clustering_final/final_clustering_comparison.csv`

is byte-for-byte identical to the regenerated file:

`/data2/fbidet/SCRBenchmark/scraw-transductive-stable-generalist/runs/pancreas_raw_counts/seed_42/results/clustering_final/final_clustering_comparison.csv`

Verification command:

```bash
cmp -s \
  /data2/fbidet/scRAW_EXPERIMENTAL/results/optuna_stable_generalist_search_20260415_161134/phase1/stable_generalist/stage1/runs/stable_generalist_trial_0017/pancreas_raw_counts/seed_42/results/clustering_final/final_clustering_comparison.csv \
  /data2/fbidet/SCRBenchmark/scraw-transductive-stable-generalist/runs/pancreas_raw_counts/seed_42/results/clustering_final/final_clustering_comparison.csv
```

The command returns `0`.

The dataset used is:

`/data2/fbidet/scRAW_EXPERIMENTAL/data/pancreas_raw_counts_no_smarter.h5ad`

SHA256:

`389410cdb86dc93c085bd231b2d5c2587e18ea36b4288cc3279ac98e982443e6`

## Decision

The `different` status in `metrics_comparison.csv` is kept as a trace of the
raw comparison against the presentation table. For final interpretation,
`pancreas_raw_counts` is considered valid because the rerun exactly reproduces
the preserved trial and gives better metrics than the presentation row.
