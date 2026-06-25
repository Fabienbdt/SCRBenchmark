# Vendored scRAW Dedicated Backend

This directory contains the lightweight `scraw_dedicated` Python package used by
the SCRBenchmark `scraw` adapter and the stable_generalist reproduction scripts.

The import path is:

```text
vendor/scraw_dedicated/src/scraw_dedicated/
```

SCRBenchmark loads this path automatically. To override it with another checkout,
set:

```bash
export SCRAW_DEDICATED_ROOT=/path/to/scRAW_EXPERIMENTAL
```
