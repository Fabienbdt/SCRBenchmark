# stable_generalist Vendored Runners

This directory contains legacy research runners used to reproduce the
`stable_generalist` campaign from inside SCRBenchmark.

They are intentionally kept out of `scripts/reproduction/` because some of them
are long, method-specific wrappers around original code. The public entry points
are the shorter scripts in `scripts/reproduction/`:

- `run_batch_baselines.py`
- `run_external_method.py`
- `run_posthoc_harmony.py`

Exact legacy method stacks are used when available. When a legacy dependency is
missing, wrappers write explicit fallback metadata in the output files so that a
full reproduction plan remains executable and auditable.
