# Ajouter une etape de preprocessing dans SCRBenchmark

Ce guide explique comment ajouter une etape de preprocessing sans casser les
flux existants: interface Streamlit, CLI, protocoles versionnes et benchmarks
avec splits train/val/test.

Le point le plus important: si une etape apprend des parametres a partir des
donnees, ces parametres doivent etre appris sur le train uniquement, puis
reutilises tels quels sur val/test.

---

## 1. Comprendre les deux chemins de preprocessing

SCRBenchmark a deux chemins principaux:

| Contexte | Fichier principal | Role |
| --- | --- | --- |
| Dataset complet / mode standard | `src/scrbenchmark/utils/data_handler.py` | Charge le fichier, applique le preprocessing sur tout le dataset, puis lance les algorithmes. |
| Benchmark train/val/test | `src/scrbenchmark/utils/dataset_splitter.py` (`BenchmarkPreprocessor`) | Apprend les parametres de preprocessing sur train, puis transforme train/val/test avec les memes parametres. |

Une nouvelle etape generale doit donc etre branchee dans les deux chemins. Si
elle est ajoutee seulement a `DataHandler`, elle peut fonctionner en mode
standard mais produire une fuite de donnees ou une erreur en mode split.

---

## 2. Choisir le type d'etape

Avant de coder, definir ou placer l'etape dans le pipeline:

| Type d'etape | Placement typique | Exemple |
| --- | --- | --- |
| Filtrage cellules | Avant normalisation | Supprimer les cellules avec trop peu de genes ou trop de mitochondrial. |
| Filtrage genes | Avant normalisation et HVG | Supprimer les genes exprimes dans trop peu de cellules. |
| Transformation des comptes bruts | Avant normalisation/log1p | Simulation de dropout, correction de comptes. |
| Normalisation ou scaling | Autour de `normalize_total`, `log1p`, `scale` | Nouvelle normalisation, transformation stabilisatrice de variance. |
| Selection de features | Apres normalisation/log1p, avant scaling | Alternative a HVG. |
| Batch correction sur matrice d'entree | Apres HVG, avant scaling | scVI/sysVI comme preprocessing commun. |
| Correction post-hoc sur latents | Apres entrainement methode | Variante `+Harmony`, pas une etape de preprocessing commune. |

Regle pratique: si l'etape modifie `adata.X` avant l'entrainement de tous les
algorithmes, elle appartient au preprocessing commun. Si elle corrige un latent
produit par une methode precise, l'implementer plutot comme variante de methode
ou post-processing.

---

## 3. Declarer les parametres

Ajouter les parametres par defaut dans:

```text
src/scrbenchmark/core/config.py
```

La liste principale est `PREPROCESSING_PARAMS`. Exemple:

```python
HyperparameterConfig(
    name="do_mito_filter",
    display_name="Filter High Mito Cells",
    param_type=ParamType.BOOLEAN,
    default=False,
    description="Remove cells with a mitochondrial fraction above the threshold.",
    category="Preprocessing",
)
```

Si l'etape a un seuil:

```python
HyperparameterConfig(
    name="max_mito_fraction",
    display_name="Max Mito Fraction",
    param_type=ParamType.FLOAT,
    default=0.2,
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    description="Maximum allowed mitochondrial fraction per cell.",
    category="Preprocessing",
)
```

Nommer les parametres explicitement:

- `do_<nom_etape>` pour activer/desactiver;
- `<nom_etape>_<parametre>` pour les options;
- eviter `enabled`, `threshold`, `mode` sans prefixe clair.

---

## 4. Mettre la logique dans `utils/`

Creer une fonction dediee dans `src/scrbenchmark/utils/` plutot que de mettre
la logique scientifique directement dans l'interface.

Exemple:

```text
src/scrbenchmark/utils/mitochondrial_filter.py
```

```python
from typing import Any

import numpy as np


def filter_high_mito_cells(
    adata: Any,
    max_mito_fraction: float = 0.2,
    gene_prefix: str = "MT-",
) -> Any:
    """Return a copy of AnnData after removing high-mitochondrial cells."""
    adata = adata.copy()

    mito_mask = adata.var_names.str.upper().str.startswith(gene_prefix.upper())
    if not mito_mask.any():
        adata.uns["mitochondrial_filter"] = {
            "applied": False,
            "reason": "no_mito_genes_found",
            "gene_prefix": gene_prefix,
        }
        return adata

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    total_counts = np.asarray(X.sum(axis=1)).ravel()
    mito_counts = np.asarray(X[:, mito_mask].sum(axis=1)).ravel()
    mito_fraction = np.divide(
        mito_counts,
        total_counts,
        out=np.zeros_like(mito_counts, dtype=float),
        where=total_counts > 0,
    )

    keep_mask = mito_fraction <= max_mito_fraction
    adata = adata[keep_mask].copy()
    adata.obs["mito_fraction"] = mito_fraction[keep_mask]
    adata.uns["mitochondrial_filter"] = {
        "applied": True,
        "max_mito_fraction": float(max_mito_fraction),
        "n_removed_cells": int((~keep_mask).sum()),
        "gene_prefix": gene_prefix,
    }
    return adata
```

Bonnes pratiques:

- retourner un objet `AnnData` coherent;
- preserver autant que possible `obs`, `var`, `layers` et `uns`;
- enregistrer l'etape dans `adata.uns` pour la reproductibilite;
- eviter de modifier l'objet original si ce n'est pas necessaire;
- ne jamais utiliser les labels test pour choisir un seuil.

---

## 5. Brancher l'etape dans `DataHandler`

Le mode standard utilise:

```text
src/scrbenchmark/utils/data_handler.py
```

La fonction cle est:

```python
DataHandler._preprocess_builtin(self, params)
```

Lire les parametres pres des autres toggles:

```python
do_mito_filter = params.get("do_mito_filter", False)
max_mito_fraction = params.get("max_mito_fraction", 0.2)
```

Puis appeler l'etape au bon endroit. Pour un filtrage cellule fonde sur les
comptes bruts, le placement naturel est avant normalisation:

```python
if do_mito_filter:
    from .mitochondrial_filter import filter_high_mito_cells

    self.adata = filter_high_mito_cells(
        self.adata,
        max_mito_fraction=max_mito_fraction,
    )
```

Apres une etape qui supprime des cellules ou des genes, verifier les objets qui
dependent des dimensions:

- `self.labels`;
- `adata.obs`;
- `adata.var`;
- `layers["original_X"]`;
- `uns["pre_hvg_counts"]`;
- matrices utilisees par les algorithmes avec preprocessing interne.

---

## 6. Brancher l'etape dans `BenchmarkPreprocessor`

Le mode train/val/test utilise:

```text
src/scrbenchmark/utils/dataset_splitter.py
```

Classe importante:

```python
BenchmarkPreprocessor
```

Methodes a traiter:

```python
fit(self, adata_train, params)
transform(self, adata, params)
```

### Cas A: etape sans apprentissage

Si l'etape applique une regle fixe donnee par l'utilisateur, par exemple
`max_mito_fraction=0.2`, appliquer la meme regle dans `fit()` et `transform()`.

Dans `PreprocessingParams`, ajouter:

```python
do_mito_filter: bool = False
max_mito_fraction: float = 0.2
```

Dans `fit()`:

```python
self.params.do_mito_filter = bool(params.get("do_mito_filter", False))
self.params.max_mito_fraction = float(params.get("max_mito_fraction", 0.2))
```

Dans `transform()`:

```python
if self.params.do_mito_filter:
    # Apply the same fixed threshold to val/test.
```

### Cas B: etape avec parametres appris

Si l'etape apprend un parametre depuis les donnees, ce parametre doit etre
appris uniquement dans `fit()` sur le train. `transform()` reutilise ensuite la
valeur apprise, sans regarder la distribution globale ni le test set.

Exemples:

- selection de genes;
- moyenne/ecart-type;
- seuil choisi par percentile;
- modele de correction entraine.

Stocker les valeurs apprises dans `self.params`:

```python
self.params.my_selected_genes = selected_genes_from_train
self.params.my_train_threshold = threshold_from_train
```

Puis les reutiliser dans `transform()`.

---

## 7. Exposer l'option dans Streamlit

### Page Preprocessing

Modifier:

```text
src/scrbenchmark/gui/preprocessing.py
```

Ajouter une checkbox et les parametres associes:

```python
do_mito_filter = st.checkbox(
    "Filter high mitochondrial cells",
    value=params.get("do_mito_filter", False),
    help="Remove cells whose mitochondrial count fraction is above the threshold.",
)
params["do_mito_filter"] = do_mito_filter

if do_mito_filter:
    params["max_mito_fraction"] = st.number_input(
        "Max mitochondrial fraction",
        min_value=0.0,
        max_value=1.0,
        value=float(params.get("max_mito_fraction", 0.2)),
        step=0.01,
    )
```

Ajouter aussi ces parametres aux fonctions qui synchronisent ou comparent
l'etat de preprocessing:

```text
_snapshot_preprocessing_params(...)
_sync_preprocessing_params_from_widgets(...)
```

Cela evite que Streamlit considere un preprocessing deja calcule comme encore
valide alors que l'utilisateur a change la nouvelle option.

### Page Customize Benchmark

Modifier aussi:

```text
src/scrbenchmark/gui/customize_benchmark.py
```

La configuration de preprocessing y est reproduite pour generer des commandes
et scripts. Ajouter la meme option dans l'onglet `Preprocessing`, puis les
valeurs par defaut dans `_create_default_config()`.

---

## 8. Exposer l'option dans la CLI

Modifier:

```text
src/scrbenchmark/cli.py
```

Ajouter les arguments au groupe preprocessing:

```python
preproc_group.add_argument(
    "--mito-filter",
    action="store_true",
    help="Filter cells with high mitochondrial fraction.",
)
preproc_group.add_argument(
    "--max-mito-fraction",
    type=float,
    default=None,
    help="Maximum mitochondrial fraction allowed per cell.",
)
```

Puis ajouter les valeurs dans `preprocessing_params`:

```python
"do_mito_filter": (
    args.mito_filter
    if getattr(args, "mito_filter", False)
    else preprocess_config.get("do_mito_filter", False)
),
"max_mito_fraction": _arg_or_config(
    args.max_mito_fraction,
    preprocess_config,
    "max_mito_fraction",
    0.2,
),
```

Mettre aussi a jour `generate_default_config()` pour que les fichiers de config
YAML/JSON generes documentent la nouvelle option.

---

## 9. Mettre a jour protocoles et methodes externes si besoin

Si l'etape doit etre disponible dans les protocoles versionnes, verifier:

```text
protocols/report/*.yaml
src/scrbenchmark/protocols/registry.py
src/scrbenchmark/gui/protocol_designer.py
```

Pour une option simple, l'ajouter a la section `preprocessing` des YAML est
generalement suffisant.

Si l'etape concerne les algorithmes externes lances via `run_method.py`,
verifier aussi les placeholders disponibles dans:

```text
scripts/reproduction/run_method.py
methods/*.yaml
```

Ne pas hardcoder l'etape dans un wrapper externe si elle doit rester un choix
global du benchmark.

---

## 10. Mettre a jour la documentation utilisateur

Ajouter une explication breve dans:

```text
src/scrbenchmark/gui/documentation.py
docs/user_guide.md
```

La documentation doit repondre a quatre questions:

1. Que fait l'etape ?
2. Quand faut-il l'utiliser ?
3. Ou se situe-t-elle dans le pipeline ?
4. Quels parametres changent les resultats ?

---

## 11. Ajouter des tests

Creer un test dedie dans:

```text
tests/unit_tests/
```

Exemple:

```text
tests/unit_tests/test_mitochondrial_filter.py
```

Tests minimum recommandes:

- l'etape retourne un objet `AnnData` valide;
- desactiver l'option conserve le comportement par defaut;
- les dimensions `obs`, `var` et `X` restent coherentes;
- `adata.uns` contient une trace de l'etape;
- le mode split utilise les parametres appris sur train, pas sur test;
- la CLI accepte les nouveaux arguments;
- `Customize Benchmark` genere les commandes avec les nouveaux parametres.

Commandes utiles:

```bash
python -m compileall -q src/scrbenchmark
pytest tests/unit_tests/test_mitochondrial_filter.py
pytest tests/unit_tests/test_gui_cli_command.py
```

Pour une modification coeur importante:

```bash
pytest tests/unit_tests
```

---

## 12. Checklist avant commit

- [ ] Les parametres sont declares dans `core/config.py`.
- [ ] La logique scientifique est dans `utils/`, pas dans l'interface.
- [ ] `DataHandler` applique l'etape en mode standard.
- [ ] `BenchmarkPreprocessor` applique l'etape sans fuite train/test.
- [ ] Streamlit expose l'option dans `Preprocessing`.
- [ ] `Customize Benchmark` expose l'option et ses valeurs par defaut.
- [ ] La CLI peut configurer l'etape.
- [ ] Les protocoles YAML restent compatibles.
- [ ] Les resultats gardent une trace dans `adata.uns`, les manifests ou la config sauvegardee.
- [ ] Les tests unitaires passent.
- [ ] La documentation explique l'ordre de l'etape dans le pipeline.

---

## Resume oral

Pour presenter le travail:

> J'ajoute d'abord les parametres dans la configuration centrale. Ensuite,
> j'implemente la transformation dans `utils/` sur un objet `AnnData`. Je la
> branche dans `DataHandler` pour le mode dataset complet et dans
> `BenchmarkPreprocessor` pour le mode train/val/test, en verifiant que les
> parametres appris ne voient jamais le test set. Enfin, j'expose l'option dans
> Streamlit, la CLI et `Customize Benchmark`, puis j'ajoute les tests et une
> trace de reproductibilite.
