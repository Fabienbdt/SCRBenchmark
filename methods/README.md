# Method Specification Registry

The `methods/` directory contains YAML specifications that instruct SCRBenchmark on how to run external algorithms within the reproduction workflows.

To add a new external algorithm, do not follow multiple documents; instead, use the single step-by-step guide:
[`../docs/algorithm_extension_guide.md`](../docs/algorithm_extension_guide.md).

This README serves only as a reference to understand the role of this directory.

## Role of YAML Files

A `methods/<algo>.yaml` file declares:

- `name`: stable identifier used by `--method`;
- `display_name`: name displayed in interfaces;
- `source.path`: directory of the external source code, often `external/original_code/<algo>`;
- `runner.kind`: launch type, generally `command_template` for new external integrations;
- `runner.command`: command executed by `scripts/reproduction/run_method.py`;
- `output.*`: paths of the labels and embeddings produced by the wrapper.

The complete format, the ready-to-copy wrapper, validation commands, and the smoke test are documented in
[`../docs/algorithm_extension_guide.md`](../docs/algorithm_extension_guide.md).

## Reading Command

To list the already available specifications:

```bash
python3 scripts/reproduction/run_method.py --list
```
