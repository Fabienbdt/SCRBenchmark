# Guide utilisateur SCRBenchmark

Ce guide explique comment installer SCRBenchmark, preparer les datasets,
lancer l'interface graphique ou la CLI, reproduire les experiences du rapport
et trouver le bon guide quand il faut modifier le projet.

---

## Sommaire

1. [Installation complete](#1-installation-complete)
2. [Preparation des datasets](#2-preparation-des-datasets)
3. [Utiliser l'interface Streamlit](#3-utiliser-linterface-streamlit)
4. [Lancer des benchmarks en CLI](#4-lancer-des-benchmarks-en-cli)
5. [Reproduire les experiences du rapport](#5-reproduire-les-experiences-du-rapport)
6. [Etendre SCRBenchmark](#6-etendre-scrbenchmark)
7. [Depannage](#7-depannage)

## Carte rapide

| Besoin | Document |
| --- | --- |
| Installation rapide, lancement et index des guides | [`../README.md`](../README.md) |
| Workflow utilisateur, datasets, GUI et CLI | Ce guide |
| Carte technique fichier par fichier | [`developer_file_guide.md`](developer_file_guide.md) |
| Ajouter un algorithme externe | [`algorithm_extension_guide.md`](algorithm_extension_guide.md) |
| Comprendre les YAML `methods/` | [`../methods/README.md`](../methods/README.md) |
| Ajouter une etape de preprocessing | [`preprocessing_extension_guide.md`](preprocessing_extension_guide.md) |
| Comprendre les scripts de reproduction | [`../scripts/reproduction/README.md`](../scripts/reproduction/README.md) |

---

## 1. Installation complete

SCRBenchmark demande **Python >= 3.9** et un environnement virtuel propre.

```bash
cd /data2/fbidet/SCRBenchmark
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Pour reproduire les experiences lourdes du rapport ou utiliser certaines
methodes externes avancees:

```bash
pip install -r requirements-reproduction.txt
```

Verification rapide:

```bash
./scrbenchmark list-algorithms
```

Cette commande doit afficher les algorithmes disponibles dans l'environnement.

---

## 2. Preparation des datasets

SCRBenchmark utilise principalement des fichiers `.h5ad` compatibles
AnnData/Scanpy.

### Dataset Baron pancreas

Ce dataset sert aux exemples, smoke tests et comparaisons rapides:

```bash
python scripts/setup/prepare_baron_dataset.py --download
```

Le fichier produit est:

```text
data/baron_human_pancreas.h5ad
```

### Campagne stable_generalist

Pour reproduire les experiences du rapport, SCRBenchmark attend 13 fichiers
`.h5ad` dans:

```text
data/stable_generalist/
```

Preparation depuis la racine de donnees locale:

```bash
python scripts/reproduction/download_datasets.py \
  --source-root /data2/fbidet/scRAW_EXPERIMENTAL/data
```

Si les fichiers exacts sont heberges a distance:

```bash
python scripts/reproduction/download_datasets.py \
  --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/
```

Le script verifie les SHA256, tailles, dimensions AnnData et colonnes
label/batch attendues. Pour verifier des fichiers deja prepares:

```bash
python scripts/reproduction/download_datasets.py --verify-only
```

---

## 3. Utiliser l'interface Streamlit

L'interface graphique est le point d'entree recommande pour explorer les
donnees, configurer les experiences et comparer les resultats.

```bash
./run.sh
```

Ouvrir ensuite l'URL affichee dans le terminal, souvent:

```text
http://localhost:8501
```

Parcours conseille:

1. `Data Upload`: charger un fichier `.h5ad`.
2. `Data Split`: choisir le protocole standard ou train/val/test.
3. `Preprocessing`: configurer filtres, HVG, normalisation, dropout ou batch correction.
4. `Algorithm Config`: choisir les algorithmes et leurs hyperparametres.
5. `Analysis`: lancer le benchmark.
6. `Results Explorer`: comparer NMI, ARI, silhouette, labels et embeddings.

---

## 4. Lancer des benchmarks en CLI

La CLI sert a automatiser des experiences sans passer par l'interface.

Exemple simple en mode transductif:

```bash
./scrbenchmark run \
  --data data/baron_human_pancreas.h5ad \
  --algorithms pca \
  --param pca:clustering_method=kmeans \
  --label-col Group \
  --n-clusters 14 \
  --device cpu \
  --output results/my_experiment \
  --no-timestamp \
  --save-labels
```

Arguments importants:

- `--data`: chemin du fichier `.h5ad`;
- `--algorithms`: liste d'algorithmes separes par des virgules;
- `--label-col`: colonne `adata.obs` contenant les labels de reference;
- `--n-clusters`: nombre de clusters attendu;
- `--device`: `cpu`, `cuda`, `mps` ou `auto`;
- `--output`: dossier de sortie.

Commandes utiles:

```bash
./scrbenchmark list-algorithms
./scrbenchmark list-params --algorithm pca
./scrbenchmark generate-config --output config.yaml
./scrbenchmark run --config config.yaml
```

Apres execution, le dossier de resultats contient notamment:

- `results.csv`: resume des scores et temps d'execution;
- `labels/`: labels predits par cellule;
- `embeddings/`: representations latentes si demandees;
- `config/`: configuration sauvegardee pour la reproductibilite.

---

## 5. Reproduire les experiences du rapport

Le point d'entree recommande est le panneau Streamlit **Report Reproduction**.
Il remplace les longues listes de commandes dans la documentation.

```bash
./run.sh
```

Dans la sidebar, ouvrir **Report Reproduction**. Le panneau contient:

- **Traceability**: carte entre figures/tables du rapport et campagnes;
- **Stable Generalist**: generation du plan principal stable_generalist;
- **Report Complements**: experiences inductives, loss-transfer et DEG;
- **Custom Protocols**: variantes configurables sans modifier les scripts.

Chaque onglet ecrit un CSV de jobs planifies et un script shell de lancement.
La carte compacte figure/table est disponible dans
[`report_reproduction_map.md`](report_reproduction_map.md).

---

## 6. Etendre SCRBenchmark

Pour ajouter un algorithme externe, ne modifiez pas
`src/scrbenchmark/algorithms/`; suivez uniquement
[`algorithm_extension_guide.md`](algorithm_extension_guide.md). Ce guide couvre
le dossier de code source externe, le wrapper, le YAML `methods/*.yaml`, la
validation et le smoke test.

Pour ajouter une etape de preprocessing, utilisez
[`preprocessing_extension_guide.md`](preprocessing_extension_guide.md).

Pour comprendre l'organisation technique du depot, utilisez
[`developer_file_guide.md`](developer_file_guide.md). Ce document est une carte
des fichiers, pas une procedure d'ajout d'algorithme.

---

## 7. Depannage

### Erreur `CUDA out of memory`

Reduire `batch_size`, utiliser `--device cpu` pour un test rapide, ou diminuer
le nombre de genes HVG dans le preprocessing.

### Un algorithme externe n'apparait pas

Verifier que le YAML est dans `methods/`, que `name` correspond a la commande
`--method`, puis lancer:

```bash
python3 scripts/reproduction/run_method.py --list
```

Pour une integration externe complete, suivre
[`algorithm_extension_guide.md`](algorithm_extension_guide.md).

### Erreur sur les comptes bruts

Les methodes fondees sur des distributions NB/ZINB peuvent exiger des comptes
bruts non normalises. Verifier que le `.h5ad` conserve les comptes dans
`adata.X`, `adata.raw` ou une couche comme `adata.layers["original_X"]`.
