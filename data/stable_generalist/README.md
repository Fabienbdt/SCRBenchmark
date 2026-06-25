# Datasets stable_generalist

Placer ici les 13 fichiers `.h5ad` utilises par la reproduction
stable_generalist.

Preparation recommandee:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /data2/fbidet/scRAW_EXPERIMENTAL/data
```

Mode telechargement distant, si les fichiers exacts sont heberges:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

Verifier des fichiers deja prepares:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

Les tailles, SHA256 et dimensions AnnData attendues sont dans
`download_manifest.csv`.

Fichiers attendus:

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

Le planificateur accepte aussi un autre dossier:

```bash
python scripts/reproduction/build_stable_generalist_plan.py --data-root /path/to/h5ad_files
```
