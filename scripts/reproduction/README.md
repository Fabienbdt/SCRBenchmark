# Scripts de reproduction

Ce dossier contient les scripts bas niveau qui preparent, planifient ou
lancent les experiences de reproduction SCRBenchmark.

Pour relancer normalement les experiences du rapport, preferer le panneau
Streamlit **Report Reproduction**. Il genere les plans de jobs et les scripts
shell sans obliger l'utilisateur a recopier de longues commandes.

---

## Perimetre

Ce README decrit le role de chaque script. Il n'est pas le guide d'integration
des algorithmes externes ni le contrat de preprocessing.

| Besoin | Document |
| --- | --- |
| Ajouter un algorithme externe | [`../../docs/algorithm_extension_guide.md`](../../docs/algorithm_extension_guide.md) |
| Comprendre les YAML `methods/` | [`../../methods/README.md`](../../methods/README.md) |
| Ajouter un preprocessing | [`../../docs/preprocessing_extension_guide.md`](../../docs/preprocessing_extension_guide.md) |
| Comprendre les fichiers du depot | [`../../docs/developer_file_guide.md`](../../docs/developer_file_guide.md) |

---

## Scripts par tache

| Tache | Scripts |
| --- | --- |
| Preparer les datasets du rapport | `download_datasets.py`, `prepare_stable_generalist_data.py` |
| Construire des CSV de jobs et lanceurs shell | `build_stable_generalist_plan.py`, `build_report_plan.py`, `manual_protocols.py` |
| Valider ou lancer un algorithme externe enregistre | `validate_method.py`, `run_method.py` |
| Lancer des familles externes ou legacy | `run_external_method.py`, `run_scaide_inductive_embeddings.py` |
| Lancer batch correction ou variantes Harmony | `run_batch_baselines.py`, `run_posthoc_harmony.py` |
| Lancer des protocoles inductifs ou leave-one-batch | `run_scrbenchmark_leave_one_batch.py`, `run_scraw_leave_one_batch.py`, `run_shared_train_inductive_algorithms.py` |
| Calculer des annotations du rapport | `run_marker_overlap.py` |

---

## Role des scripts

- `download_datasets.py`: telecharge ou materialise localement les 13 fichiers
  `.h5ad` exacts de la campagne stable_generalist, puis verifie SHA256, tailles,
  dimensions AnnData et colonnes label/batch.
- `prepare_stable_generalist_data.py`: ancien helper local qui materialise les
  13 fichiers `.h5ad` par hardlink, symlink ou copie.
- `build_stable_generalist_plan.py`: lit les tables de reproductibilite
  stable_generalist et genere `planned_jobs.csv` plus `run_ready_jobs.sh`.
- `build_report_plan.py`: genere les plans des complements du rapport:
  protocoles inductifs, loss-transfer et recouvrement DEG.
- `manual_protocols.py`: cree ou execute des jobs configurables pour
  loss-transfer, variantes Harmony et splits inductifs.
- `add_method.py`: helper bas niveau optionnel qui cree un YAML de depart dans
  `methods/`; le parcours utilisateur reste
  [`docs/algorithm_extension_guide.md`](../../docs/algorithm_extension_guide.md).
- `validate_method.py`: charge une specification `methods/*.yaml`, affiche la
  commande finale et peut lancer un smoke test avec `--run`.
- `run_method.py`: lanceur uniforme pilote par `methods/*.yaml`. C'est le point
  d'entree public pour executer une methode sur un dataset.
- `run_external_method.py`: lance des integrations externes historiques comme
  DESC, DeepScena, CellSIUS, GiniClust, scAIDE, scCAD et variantes Harmony.
- `run_scaide_inductive_embeddings.py`: adaptateur AIDE/scAIDE pour le protocole
  inductif partage; il attend un interpreteur compatible TensorFlow 1.14 via
  `SCAIDE_PYTHON`.
- `run_batch_baselines.py`: lance les baselines de batch correction du rapport:
  Harmony, ComBat, ComBat-seq, Scanorama, PCA+Leiden et scVI.
- `run_posthoc_harmony.py`: lance scNAME, scMAE ou scVI, applique Harmony sur
  l'embedding, puis reclusterise pour les lignes `+Harmony`.
- `run_scrbenchmark_leave_one_batch.py`: lance une execution SCRBenchmark CLI
  par batch tenu de cote.
- `run_scraw_leave_one_batch.py`: lance scRAW inductif par batch tenu de cote
  via `vendor/scraw_inductive`.
- `run_shared_train_inductive_algorithms.py`: lance le protocole inductif
  representatif du rapport pour scRAW, scNAME, scMAE, scDeepCluster, scAIDE et
  PCA+Harmony.
- `run_marker_overlap.py`: calcule le recouvrement DEG top-100 depuis des
  labels sauvegardes.

Les anciens executants longs restent dans `vendor/stable_generalist_runners/`.
Ce dossier expose seulement de petits points d'entree.

---

## Flux recommande

Pour reproduire le rapport, utiliser **Report Reproduction** dans Streamlit:

```bash
./run.sh
```

Utiliser directement les scripts de ce dossier seulement pour l'automatisation,
le debug ou un environnement sans interface graphique. Chaque script expose
`--help`.

Pour ajouter un nouvel algorithme externe, suivre le guide unique:
[`../../docs/algorithm_extension_guide.md`](../../docs/algorithm_extension_guide.md).
