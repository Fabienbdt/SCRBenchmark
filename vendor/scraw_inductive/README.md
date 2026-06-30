# Vendored scRAW Inductive Backend

This directory contains the lightweight `scraw` package used for leave-one-batch
inductive tests.

The main entry point is:

```bash
python scripts/reproduction/run_scraw_leave_one_batch.py
```

Available public presets:

- `default`: stable trial 0017 configuration.
- `baron`: legacy Baron configuration.

The `default` preset resolves to:

```text
vendor/scraw_inductive/configs/stable_generalist_trial_0017.json
```
