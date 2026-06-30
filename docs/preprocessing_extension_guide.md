# Add a Preprocessing Step to SCRBenchmark

This guide explains how to add a preprocessing step without breaking existing
flows: Streamlit, CLI, versioned protocols, and train/val/test benchmarks.

The key rule: if a step learns parameters from data, those parameters must be
learned on train only, then reused unchanged on validation/test.

---

## 1. Understand the Two Preprocessing Paths

SCRBenchmark has two main preprocessing paths:

| Context | Main file | Role |
| --- | --- | --- |
| Full dataset / standard mode | `src/scrbenchmark/utils/data_handler.py` | Loads the file, applies preprocessing to the full dataset, then runs algorithms. |
| Train/val/test benchmark | `src/scrbenchmark/utils/dataset_splitter.py` (`BenchmarkPreprocessor`) | Learns preprocessing parameters on train, then transforms train/val/test with the same parameters. |

A general preprocessing step must therefore be wired into both paths. If it is
added only to `DataHandler`, it may work in standard mode while causing leakage
or errors in split mode.

---

## 2. Choose the Step Type

Before coding, decide where the step belongs in the pipeline:

| Step type | Typical placement | Example |
| --- | --- | --- |
| Cell filtering | Before normalization | Remove cells with too few genes or too much mitochondrial signal. |
| Gene filtering | Before normalization and HVG | Remove genes expressed in too few cells. |
| Raw-count transformation | Before normalization/log1p | Dropout simulation, count correction. |
| Normalization or scaling | Around `normalize_total`, `log1p`, `scale` | New normalization, variance-stabilizing transform. |
| Feature selection | After normalization/log1p, before scaling | Alternative to HVG. |
| Batch correction on input matrix | After HVG, before scaling | scVI/sysVI as shared preprocessing. |
| Post-hoc correction on latents | After method training | `+Harmony` variant, not shared preprocessing. |

Practical rule: if the step modifies `adata.X` before all algorithms train, it
belongs to shared preprocessing. If it corrects a latent representation produced
by one method, implement it as a method variant or post-processing step instead.

---

## 3. Declare Parameters

Add default parameters in:

```text
src/scrbenchmark/core/config.py
```

The main list is `PREPROCESSING_PARAMS`. Example:

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

If the step has a threshold:

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

Name parameters explicitly:

- `do_<step_name>` to enable/disable;
- `<step_name>_<parameter>` for options;
- avoid `enabled`, `threshold`, or `mode` without a clear prefix.

---

## 4. Put the Logic in `utils/`

Create a dedicated function in `src/scrbenchmark/utils/` instead of putting
scientific logic directly in the interface.

Example:

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

Good practices:

- return a coherent `AnnData` object;
- preserve `obs`, `var`, `layers`, and `uns` as much as possible;
- record the step in `adata.uns` for reproducibility;
- avoid mutating the original object unless needed;
- never use test labels to choose a threshold.

---

## 5. Wire the Step Into `DataHandler`

Standard mode uses:

```text
src/scrbenchmark/utils/data_handler.py
```

The key function is:

```python
DataHandler._preprocess_builtin(self, params)
```

Read parameters near the other toggles:

```python
do_mito_filter = params.get("do_mito_filter", False)
max_mito_fraction = params.get("max_mito_fraction", 0.2)
```

Then call the step at the right place. For cell filtering based on raw counts,
the natural placement is before normalization:

```python
if do_mito_filter:
    from .mitochondrial_filter import filter_high_mito_cells

    self.adata = filter_high_mito_cells(
        self.adata,
        max_mito_fraction=max_mito_fraction,
    )
```

After a step that removes cells or genes, verify objects that depend on
dimensions:

- `self.labels`;
- `adata.obs`;
- `adata.var`;
- `layers["original_X"]`;
- `uns["pre_hvg_counts"]`;
- matrices used by algorithms with internal preprocessing.

---

## 6. Wire the Step Into `BenchmarkPreprocessor`

Train/val/test mode uses:

```text
src/scrbenchmark/utils/dataset_splitter.py
```

Important class:

```python
BenchmarkPreprocessor
```

Methods to handle:

```python
fit(self, adata_train, params)
transform(self, adata, params)
```

### Case A: Step Without Learning

If the step applies a fixed user rule, for example `max_mito_fraction=0.2`,
apply the same rule in both `fit()` and `transform()`.

In `PreprocessingParams`, add:

```python
do_mito_filter: bool = False
max_mito_fraction: float = 0.2
```

In `fit()`:

```python
self.params.do_mito_filter = bool(params.get("do_mito_filter", False))
self.params.max_mito_fraction = float(params.get("max_mito_fraction", 0.2))
```

In `transform()`:

```python
if self.params.do_mito_filter:
    # Apply the same fixed threshold to val/test.
```

### Case B: Step With Learned Parameters

If the step learns a parameter from data, learn it only in `fit()` on train.
`transform()` then reuses the learned value without looking at the global
distribution or test set.

Examples:

- gene selection;
- mean/standard deviation;
- percentile-derived threshold;
- trained correction model.

Store learned values in `self.params`:

```python
self.params.my_selected_genes = selected_genes_from_train
self.params.my_train_threshold = threshold_from_train
```

Then reuse them in `transform()`.

---

## 7. Expose the Option in Streamlit

### Preprocessing Page

Modify:

```text
src/scrbenchmark/gui/preprocessing.py
```

Add a checkbox and associated parameters:

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

Also add these parameters to functions that sync or compare preprocessing
state:

```text
_snapshot_preprocessing_params(...)
_sync_preprocessing_params_from_widgets(...)
```

This prevents Streamlit from treating an already computed preprocessing state as
still valid after the user changed the new option.

### Customize Benchmark Page

Also modify:

```text
src/scrbenchmark/gui/customize_benchmark.py
```

Preprocessing configuration is reproduced there to generate commands and
scripts. Add the same option in the `Preprocessing` tab, then add defaults in
`_create_default_config()`.

---

## 8. Expose the Option in the CLI

Modify:

```text
src/scrbenchmark/cli.py
```

Add arguments to the preprocessing group:

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

Then add values to `preprocessing_params`:

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

Also update `generate_default_config()` so generated YAML/JSON configs document
the new option.

---

## 9. Update Protocols and External Methods if Needed

If the step must be available in versioned protocols, check:

```text
protocols/report/*.yaml
src/scrbenchmark/protocols/registry.py
src/scrbenchmark/gui/protocol_designer.py
```

For a simple option, adding it to the YAML `preprocessing` section is usually
enough.

If the step concerns external algorithms launched through `run_method.py`, also
check placeholders in:

```text
scripts/reproduction/run_method.py
methods/*.yaml
```

Do not hardcode the step in an external wrapper if it should remain a global
benchmark choice.

---

## 10. Update User Documentation

Add a short explanation in:

```text
src/scrbenchmark/gui/documentation.py
docs/user_guide.md
```

The documentation must answer four questions:

1. What does the step do?
2. When should it be used?
3. Where does it sit in the pipeline?
4. Which parameters change results?

---

## 11. Add Tests

Create a dedicated test in:

```text
tests/unit_tests/
```

Example:

```text
tests/unit_tests/test_mitochondrial_filter.py
```

Recommended minimum tests:

- the step returns a valid `AnnData` object;
- disabling the option preserves default behavior;
- `obs`, `var`, and `X` dimensions stay coherent;
- `adata.uns` contains a trace of the step;
- split mode uses parameters learned on train, not test;
- the CLI accepts the new arguments;
- `Customize Benchmark` generates commands with the new parameters.

Useful commands:

```bash
python -m compileall -q src/scrbenchmark
pytest tests/unit_tests/test_mitochondrial_filter.py
pytest tests/unit_tests/test_gui_cli_command.py
```

For a core change:

```bash
pytest tests/unit_tests
```

---

## 12. Pre-Commit Checklist

- [ ] Parameters are declared in `core/config.py`.
- [ ] Scientific logic is in `utils/`, not in the interface.
- [ ] `DataHandler` applies the step in standard mode.
- [ ] `BenchmarkPreprocessor` applies the step without train/test leakage.
- [ ] Streamlit exposes the option in `Preprocessing`.
- [ ] `Customize Benchmark` exposes the option and its defaults.
- [ ] The CLI can configure the step.
- [ ] YAML protocols remain compatible.
- [ ] Results keep a trace in `adata.uns`, manifests, or saved config.
- [ ] Unit tests pass.
- [ ] Documentation explains the step order in the pipeline.

---

## Short Explanation

To present the work:

> I first add parameters to the central configuration. Then I implement the
> transformation in `utils/` on an `AnnData` object. I wire it into
> `DataHandler` for full-dataset mode and into `BenchmarkPreprocessor` for
> train/val/test mode, checking that learned parameters never see the test set.
> Finally, I expose the option in Streamlit, the CLI, and `Customize Benchmark`,
> then add tests and a reproducibility trace.
