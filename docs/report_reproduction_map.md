# Carte de reproduction du rapport

Ce document relie les figures et tables du rapport de stage aux lanceurs
SCRBenchmark qui permettent de regenerer leurs donnees.

La carte machine-readable canonique est:

```text
reproducibility/report_reproduction_map.csv
```

Utiliser les labels LaTeX (`fig:...`, `tab:...`) comme identifiants stables.
Les numeros finaux des figures peuvent changer quand le rapport est edite.

---

## Workflow recommande

Utiliser le panneau Streamlit **Report Reproduction** comme interface principale
pour relancer les experiences du rapport. Il est destine a etre livre avec le
depot et remplace les longues listes de commandes dans la documentation.

Dans la sidebar de l'application, ouvrir **Report Reproduction**:

- **Traceability** affiche les correspondances figures/tables relancables;
- **Stable Generalist** genere le lanceur principal du benchmark du rapport;
- **Report Complements** genere les lanceurs inductifs, loss-transfer et DEG;
- **Custom Protocols** permet de creer des variantes sans modifier les lignes
  de commande a la main.

Chaque onglet ecrit un CSV de jobs planifies et un lanceur shell, puis affiche
la commande unique a executer. Les scripts bas niveau restent disponibles pour
l'automatisation avancee, mais le panneau GUI est l'entree publique documentee.

---

## Couverture relancable

| Cible du rapport | Famille de lanceur | Datasets | Seeds |
| --- | --- | --- | --- |
| `fig:scraw_common8_family_top3_plus_scraw` | Onglet Stable Generalist | common-8 | seed de relance 42; table source `n_seeds=1` |
| `fig:rank_matrix_common8` | Onglet Stable Generalist | common-8 | seed de relance 42 |
| `tab:scraw_holdout_pancreas_results` | Onglet Stable Generalist | validation externe | seed de relance 42 |
| `fig:loss_transfer_baseline_vs_best` | Onglet Report Complements | 5 datasets loss-transfer | 42-46 |
| `fig:inductive_metrics_boxplots_default` | Onglet Report Complements | 6 datasets inductifs representatifs | 42 |

---

## Campagne principale

Cibles couvertes:

- `fig:scraw_common8_family_top3_plus_scraw`
- `fig:rank_matrix_common8`
- `fig:common8_family_all_methods`
- `tab:runtime_moyen_algorithmes`
- `tab:scraw_holdout_pancreas_results`

Utiliser l'onglet **Stable Generalist**. Il expose le dossier de sortie,
l'interpreteur Python, le device, la seed, les filtres de datasets et les
filtres de methodes. Le CSV genere contient un job par ligne et le script shell
genere contient les jobs executables.

---

## Complements du rapport

Cibles couvertes:

- `fig:loss_transfer_baseline_vs_best`
- `fig:inductive_metrics_boxplots_default`
- `fig:inductive_metrics_dataset_boxplots`
- `fig:baron_scraw_bio_interpretation`
- `tab:baron_annotation_comparison`

Utiliser l'onglet **Report Complements** et selectionner les campagnes
pertinentes: `inductive`, `loss_transfer` et/ou `deg`.

---

## Protocoles personnalises

Utiliser l'onglet **Custom Protocols** pour des variantes loss-transfer, des
complements Harmony ou des splits inductifs train/test. Le panneau expose le
dataset, la colonne de labels, la colonne batch, les seeds, les algorithmes,
les groupes de split et le chemin du lanceur de sortie.

---

## Limites connues

Les figures d'ablation et d'importance Optuna sont documentees comme artefacts
du rapport, mais elles ne sont pas encore exposees via un lanceur SCRBenchmark
propre. Avant d'annoncer une reproductibilite entierement cle en main, ajouter:

- une campagne explicite `ablation_scraw` pour `tab:scraw_ablation` et
  `fig:scraw_ablation_barplot`;
- un manifest archive ou relancable pour la campagne Optuna de
  `fig:scraw_hparam_importance_baron`,
  `fig:scraw_hparam_importance_generalist8` et
  `fig:scraw_hparam_importance_stable_generalist`.
