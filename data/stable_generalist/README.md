# Datasets stable_generalist

Place here the 13 `.h5ad` files used by the stable_generalist reproduction.

Recommended preparation:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /path/to/existing/h5ad/files
```

Remote download mode, when the exact files are hosted:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

Verify already prepared files:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

The command is read-only. Pass `--report <path.csv>` if a separate verification
report should be written.

Expected sizes, SHA256 hashes, and AnnData dimensions are in
`download_manifest.csv`.

Expected files:

- `bbag094_zeisel.h5ad`
- `bbag094_spleen.h5ad`
- `baron_human_pancreas.h5ad`
- `gse112013_human_testis_raw_counts.h5ad`
- `kang_pbmc_gse96583_singlets_raw_counts.h5ad`
- `macaque_retina_gse118480_bipolar_raw_counts.h5ad`
- `pancreas_raw_counts_no_smarter.h5ad`
- `pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2.h5ad`
- `paul15_bone_marrow_raw_counts.h5ad`
- `Human_Pancreas_1_raw_counts.h5ad`
- `Human_Pancreas_2_raw_counts.h5ad`
- `Mouse_Pancreas_1_raw_counts.h5ad`
- `Tabula_Muris_liver_filtered_raw_counts.h5ad`

The planner can also use another directory:

```bash
python scripts/reproduction/build_stable_generalist_plan.py --data-root /path/to/h5ad_files
```
