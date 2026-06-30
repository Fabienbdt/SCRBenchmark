# Benchmark Protocol Registry

This directory contains versioned benchmark protocols that the Streamlit
interface can load from **Customize Benchmark**.

Each YAML file describes datasets, the split strategy, preprocessing, methods,
manual protocols, execution options, metrics, and optional sweeps. A user can
copy a report protocol, edit only the required fields, and load it in the
interface without changing Python code.

Report presets are in:

```text
protocols/report/
```

Exemples:

- `baron_transductive.yaml`
- `baron_split_701020.yaml`
- `common8_methods_harmony.yaml`
- `loss_transfer_report.yaml`
- `inductive_report_splits.yaml`

Before execution, the interface validates data paths, AnnData columns when
possible, method names, split ratios, seeds, and manual protocol requirements.
