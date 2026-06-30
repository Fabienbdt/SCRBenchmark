# Report Reproduction Map

This document connects internship report figures and tables to the SCRBenchmark
launchers that regenerate their data.

The canonical machine-readable map is:

```text
reproducibility/report_reproduction_map.csv
```

The exact command order is documented in:

```text
docs/report_reproduction_steps.md
```

Use LaTeX labels (`fig:...`, `tab:...`) as stable identifiers. Final figure
numbers may change when the report is edited.

---

## Recommended Workflow

Use the Streamlit **Report Reproduction** panel as the main interface for
rerunning report experiments. It is intended to ship with the repository and
replaces long command lists in the documentation.

In the application sidebar, open **Report Reproduction**:

- **Traceability** shows runnable figure/table mappings;
- **Stable Generalist** builds the main report benchmark launcher;
- **Report Complements** builds inductive, loss-transfer, and DEG launchers;
- **scRAW Weighting** exposes loss-transfer variants directly;
- **Generalization** exposes inductive protocols directly;
- **Biological Interpretation** builds the Baron marker-overlap workflow and
  reuses the exact Baron report labels (`tsne_coordinates.csv`) when present;
- **Custom Protocols** creates variants without hand-editing command lines.

Each tab writes a planned-job CSV and shell launcher, then displays the single
command to execute. Low-level scripts remain available for advanced automation,
but the GUI panel is the documented public entry point.

---

## Runnable Coverage

| Report target | Launcher family | Datasets | Seeds |
| --- | --- | --- | --- |
| `fig:scraw_common8_family_top3_plus_scraw` | Stable Generalist tab | common-8 | rerun seed 42; source table `n_seeds=1` |
| `fig:rank_matrix_common8` | Stable Generalist tab | common-8 | rerun seed 42 |
| `tab:scraw_holdout_pancreas_results` | Stable Generalist tab | external validation | rerun seed 42 |
| `fig:loss_transfer_baseline_vs_best` | Report Complements tab | 5 loss-transfer datasets | 42-46 |
| `fig:inductive_metrics_boxplots_default` | Report Complements tab | 6 representative inductive datasets | 42 |

---

## Main Campaign

Covered targets:

- `fig:scraw_common8_family_top3_plus_scraw`
- `fig:rank_matrix_common8`
- `fig:common8_family_all_methods`
- `tab:runtime_moyen_algorithmes`
- `tab:scraw_holdout_pancreas_results`

Use the **Stable Generalist** tab. It exposes the output directory, Python
interpreter, device, seed, dataset filters, and method filters. The generated
CSV contains one job per row and the generated shell script contains executable
jobs.

---

## Report Complements

Covered targets:

- `fig:loss_transfer_baseline_vs_best`
- `fig:inductive_metrics_boxplots_default`
- `fig:inductive_metrics_dataset_boxplots`
- `fig:baron_scraw_bio_interpretation`
- `tab:baron_annotation_comparison`

Use the **Report Complements** tab and select the relevant campaigns:
`inductive`, `loss_transfer`, and/or `deg`.

---

## Custom Protocols

Use the **Custom Protocols** tab for loss-transfer variants, Harmony complements,
or inductive train/test splits. The panel exposes the dataset, label column,
batch column, seeds, algorithms, split groups, and output launcher path.

---

## Known Limits

Ablation and Optuna-importance figures are documented as report artifacts, but
they are not yet exposed through a clean SCRBenchmark launcher. Before claiming
fully turnkey reproducibility, add:

- an explicit `ablation_scraw` campaign for `tab:scraw_ablation` and
  `fig:scraw_ablation_barplot`;
- an archived or runnable manifest for the Optuna campaigns behind
  `fig:scraw_hparam_importance_baron`,
  `fig:scraw_hparam_importance_generalist8`, and
  `fig:scraw_hparam_importance_stable_generalist`.
