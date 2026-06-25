# Dossier de donnees SCRBenchmark

Ce dossier contient les donnees utilisees pour benchmarker les algorithmes de
clustering single-cell RNA-seq.

Les gros fichiers `.h5ad` ne sont pas suivis par Git afin de garder le depot
leger. Le depot fournit donc:

1. des donnees brutes suivies par Git quand c'est raisonnable;
2. des scripts pour reconstruire ou materialiser les datasets prepares;
3. des manifests pour verifier les fichiers attendus.

---

## Generer le dataset Baron human pancreas

Le dataset Baron human pancreas est le dataset de test principal. Pour le
generer:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

Le script cree:

```text
data/baron_human_pancreas.h5ad
```

Ce fichier contient environ:

- 8 500 cellules issues de 4 donneurs pancreas humains;
- 14 types cellulaires;
- 20 000 genes;
- une information de batch/donneur.

Structure attendue apres generation:

```text
data/
├── GSE84133_RAW/
├── baron_human_pancreas.h5ad
└── README.md
```

---

## Datasets stable_generalist

La reproduction stable_generalist attend 13 fichiers `.h5ad` dans:

```text
data/stable_generalist/
```

Commande recommandee si les donnees source sont disponibles localement:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /data2/fbidet/scRAW_EXPERIMENTAL/data
```

Si les fichiers exacts sont heberges a distance:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

Verifier des fichiers deja prepares:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

Le script compare les fichiers a `data/stable_generalist/download_manifest.csv`
avec SHA256, taille, dimensions AnnData et colonnes attendues.

La liste des fichiers est documentee dans:

```text
data/stable_generalist/README.md
```

Si les fichiers sont stockes ailleurs, passer explicitement le dossier au
generateur de plan:

```bash
python scripts/reproduction/build_stable_generalist_plan.py --data-root /path/to/h5ad_files
```
