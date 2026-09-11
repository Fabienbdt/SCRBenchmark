# scRAW architecture and verification

SCRBenchmark exposes scRAW through the method registry in
[`methods/report_methods.yaml`](../../methods/report_methods.yaml).
It is separate from the internal baseline algorithm registry.

## Execution path

1. `scripts/reproduction/run_method.py` resolves a method and records its command.
2. `scripts/reproduction/adapters/run_scraw_external.py` applies typed overrides
   and calls the public pipeline.
3. `vendor/scraw_inductive/src/scraw/` contains configuration, preprocessing,
   model, training, inference and evaluation.
4. `vendor/scraw_dedicated/src/scraw_dedicated/` contains the dedicated research
   backend used by historical checkpoint replay and weighted baseline losses.

The public presets are `default` and `baron`. Preset selection and manual
overrides must remain visible in the effective configuration. Unsupported
objectives and explicitly missing label/batch columns must raise an error.

## Scientific contracts

- Keep raw counts available for objectives that require count data.
- In inductive workflows, fit preprocessing on training cells and apply the
  saved state to held-out cells.
- Preserve cell identifiers and filtering indices with predictions and
  embeddings. Never match outputs to inputs by length alone after filtering.
- Distinguish an inductive evaluation from a transductive run that has seen
  the complete dataset.
- Save preprocessing state together with model and clustering state when
  inference depends on that state.
- Record the effective seed, parameters, dataset and mode. Exact numerical
  reproduction also depends on software versions, hardware and input data.

## Verification

From a development installation:

```bash
python -m pytest -q tests/unit_tests/test_scraw_public_backend_contract.py
python -m pytest -q tests/unit_tests/test_scraw_weighted_loss_components.py
python -m pytest -q tests/unit_tests/test_reproduction_safety.py
```

These tests cover runner options, configuration parsing, unsupported losses,
cell filtering, saved preprocessing state and checkpoint replay contracts.
They complement the synthetic CLI and Streamlit tests under `tests/integration`.
They do not rerun the complete scientific report.

See the [handover guide](handover_guide.md) for commands and the
[reproduction map](report_reproduction_map.md) for the limits of report coverage.
