# Maintenance and release checks

Use a checkout outside cloud-synchronized folders, for example
`~/Developer/SCRBenchmark`. Keep each virtual environment local to its checkout.
Cloud file eviction can stall Python imports and Git operations even when a
filename remains visible.

## Supported verification path

Python 3.12 is the maintained test environment. The dependency snapshot in
`constraints/python312-tested.txt` records the core, GUI and development stack;
it is not an environment for reproducing every legacy external method.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/python312-tested.txt -e ".[dev,gui]"
python -m pip check
python -m compileall -q src scripts vendor/scraw_inductive/src vendor/scraw_dedicated/src
python -m pytest -q
python -m build
```

The GitHub Actions workflow is configured to run the source suite on Linux and
macOS, then install the wheel and repeat integration tests outside the checkout.
Hosted execution starts once the branch is pushed; local results are recorded in
[the validation report](validation.md). This checks
CLI entry points, algorithm registry identity, runtime assets, synthetic
benchmarks, cell IDs, deterministic PCA output, failure statuses, report
command generation and Streamlit pages. Optional scientific backends and
complete report campaigns require their own environments, datasets and compute.

## Runtime layout

Application imports use `scrbenchmark.*`. `paths.resource_root()` locates the
small report assets either in the source checkout or in the wheel's `_assets`
directory. Build rules live in `src/scrbenchmark/_build.py` and `MANIFEST.in`.
User inputs and experiment outputs must not be written into installed assets.
Historical standalone adapters still establish paths for their separate
backends; avoid introducing that pattern into application modules.

A run has succeeded only after the CLI exits with zero and
`config/run_status.json` says `completed`. Partial failures retain successful
runs, list missing repetitions and return a nonzero exit code. CSV label
exports include cell identifiers; consumers should join by those identifiers.

## Release prerequisites

- All required checks pass on the proposed revision.
- A new maintainer can execute the README's synthetic example.
- Documentation commands, dependency groups and output contracts agree.
- Changes affecting scientific behavior have focused regression tests.
- Review [third-party provenance](../../THIRD_PARTY.md) and choose the project
  license with the rights holders before publishing a licensed distribution.
- Preserve checkpoint replay while preparing any external artifact archive;
  do not remove the only versioned copy or rewrite shared history implicitly.

The two original large legacy GUI modules remain. Extract behavior in small
changes covered by tests; a wholesale rewrite is not a release prerequisite.
