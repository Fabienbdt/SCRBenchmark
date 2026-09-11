# Vendored scRAW Dedicated Backend

This directory contains the lightweight `scraw_dedicated` Python package used by
the transductive checkpoint and stable-generalist reproduction scripts. The
registered SCRBenchmark `scRAW` adapter uses the public backend in the sibling
`vendor/scraw_inductive` directory.

The import path is:

```text
vendor/scraw_dedicated/src/scraw_dedicated/
```

SCRBenchmark loads this path automatically. To override it with another checkout,
set:

```bash
export SCRAW_EXPERIMENTAL_ROOT=/path/to/scRAW_EXPERIMENTAL
```
