# Contributing to SCRBenchmark

This repository is both a benchmarking application and a scientific
reproduction workspace. Changes must therefore preserve software behavior,
the experimental protocol, and the provenance of generated artifacts.

## Development Setup

Python 3.12 is the tested development version. Python 3.10 or newer is
required by the source syntax.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/python312-tested.txt -e ".[dev,gui]"
```

Install the additional reproduction stack only when working on the report
campaigns:

```bash
python -m pip install -r requirements-reproduction.txt
```

## Verification Tiers

Run the smallest tier that covers the change, then run the full unit suite
before handing work over.

```bash
# Syntax and fast unit checks
python -m compileall -q src scripts vendor/scraw_inductive/src vendor/scraw_dedicated/src
pytest -q -m "not slow"

# Full maintained unit and integration suite, including slow synthetic workflows
pytest -q

# Cross-implementation checks; original-author comparisons may be skipped
python tests/comparisons/run_all_comparisons.py --tests-only

# Whitespace errors in the pending patch
git diff --check
```

For scRAW changes, also run the focused public-backend tests documented in
[`docs/guide/scraw_architecture.md`](docs/guide/scraw_architecture.md).

## Code Guidelines

- Write maintained code, comments, logs, UI copy, and documentation in English.
- Use descriptive names and type hints for new public functions. Add a short
  docstring that states the contract rather than repeating the function name.
- Keep Streamlit pages focused on state and presentation. Put reusable logic in
  `core/`, `utils/`, or a small dedicated module.
- Do not add new behavior to `gui/analysis/legacy.py` or
  `gui/results_explorer/legacy.py` when it can live in the modular packages next
  to them.
- Avoid new `sys.path` mutations. Standalone legacy adapters still need explicit backend paths; maintained
  application modules must use `scrbenchmark.*` imports.
- Catch only exceptions that can be handled locally. If a broad exception is
  required at an application boundary, log the operation and the exception.
- Keep generated data, model checkpoints, caches, and result directories out of
  normal source changes unless they are deliberate, documented handover assets.

Follow the existing indentation in a file when making a small change. New
standalone Python modules should use four spaces and standard library imports
before third-party and local imports.

## Scientific Safety

- Learn preprocessing parameters on training data only, then reuse the frozen
  state for validation and test data.
- Preserve raw counts when NB/ZINB methods need them; never silently substitute
  normalized values.
- Do not use ground-truth labels to choose clusters or hyperparameters unless
  the option is explicitly marked as oracle evaluation.
- Save the effective configuration, random seed, data identifiers, labels, and
  cell order with every reproducible run.
- Reject unsupported scientific options with a clear error instead of silently
  falling back to a different loss, metric, or data column.

## Where Changes Belong

Use [`docs/guide/developer_file_guide.md`](docs/guide/developer_file_guide.md) for the main
file map. In particular:

- internal maintained baselines: `src/scrbenchmark/algorithms/`;
- registered external/report methods: `methods/*.yaml` plus a thin adapter;
- public scRAW pipeline: `vendor/scraw_inductive/src/scraw/`;
- dedicated scRAW research/search backend:
  `vendor/scraw_dedicated/src/scraw_dedicated/`;
- reusable report designs: `protocols/*.yaml`;
- generated report plans and runners: `scripts/reproduction/`.

## Documentation and Review

Update the nearest guide whenever a command, output path, configuration field,
or external requirement changes. Keep examples runnable from the repository
root and use `--no-timestamp` when the following step refers to an exact output
path.

Before requesting review, summarize:

1. the behavior changed;
2. the tests and smoke commands executed;
3. any data, hardware, or legacy environment that was unavailable;
4. whether numerical results are expected to change.

External source code must retain its license and provenance. Record the exact
upstream revision in the method specification or adjacent README.
