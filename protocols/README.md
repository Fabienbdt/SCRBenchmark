# Registre des protocoles de benchmark

Ce dossier contient des protocoles de benchmark versionnes que l'interface
Streamlit peut charger depuis **Customize Benchmark**.

Chaque fichier YAML decrit les datasets, la strategie de split, le
preprocessing, les methodes, les protocoles manuels, les options d'execution,
les metriques et les sweeps optionnels. Un utilisateur peut copier un protocole
du rapport, modifier uniquement les champs necessaires, puis le charger dans
l'interface sans modifier le code Python.

Les presets du rapport sont dans:

```text
protocols/report/
```

Exemples:

- `baron_transductive.yaml`
- `baron_split_701020.yaml`
- `common8_methods_harmony.yaml`
- `loss_transfer_report.yaml`
- `inductive_report_splits.yaml`

Avant execution, l'interface valide les chemins de donnees, les colonnes
AnnData quand c'est possible, les noms de methodes, les ratios de split, les
seeds et les exigences des protocoles manuels.
