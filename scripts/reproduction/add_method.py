#!/usr/bin/env python3
"""Create a starter SCRBenchmark method spec.

The generated YAML is intentionally runnable after small edits: fill the command
that launches the author code, then validate it with validate_method.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


class _IndentedDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "my_method"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Canonical method name used by run_method.py.")
    parser.add_argument("--display-name", default="", help="Human readable name shown in the UI.")
    parser.add_argument("--family", default="external", help="Method family, e.g. deep_learning or rare_specific.")
    parser.add_argument(
        "--source",
        default="",
        help="Author code directory. Default: external/original_code/<method>.",
    )
    parser.add_argument(
        "--runner",
        default="command_template",
        choices=["command_template", "scrbenchmark_cli"],
        help="Use command_template for external code; scrbenchmark_cli for an existing SCRBenchmark algorithm.",
    )
    parser.add_argument(
        "--script",
        default="",
        help="Main script for command_template. Default: <source>/main.py.",
    )
    parser.add_argument(
        "--algorithm",
        default="",
        help="Algorithm name for runner=scrbenchmark_cli. Default: normalized method name.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="YAML output path. Default: methods/<method>.yaml.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing YAML file.")
    return parser.parse_args()


def _command_template_spec(args: argparse.Namespace, source: str) -> dict[str, Any]:
    script = args.script or f"{{source_path}}/main.py"
    return {
        "name": args.name,
        "display_name": args.display_name or args.name,
        "family": args.family,
        "report": True,
        "core_contract": "author_core_unchanged",
        "core_status": "original_source_pending_validation",
        "aliases": [_slug(args.name).lower()],
        "source": {
            "kind": "original_source",
            "path": source,
        },
        "runner": {
            "kind": "command_template",
            "command": [
                "{python_bin}",
                script,
                "--input",
                "{data}",
                "--output",
                "{raw_dir}",
                "--clusters",
                "{n_labels}",
                "--seed",
                "{seed}",
                "--n-top-genes",
                "{n_top_genes}",
                "--target-sum",
                "{target_sum}",
                "--hvg-flavor",
                "{hvg_flavor}",
                "{param_args}",
            ],
        },
        "output": {
            "expected_file": "results/analysis_results.csv",
            "labels_file": "raw/labels.csv",
            "labels_column": "cluster",
            "cell_id_column": "cell_id",
        },
        "notes": "Edit runner.command and output.* to match the author code, then run validate_method.py.",
    }


def _scrbenchmark_cli_spec(args: argparse.Namespace, source: str) -> dict[str, Any]:
    algorithm = args.algorithm or _slug(args.name).lower()
    return {
        "name": args.name,
        "display_name": args.display_name or args.name,
        "family": args.family,
        "report": True,
        "core_contract": "local_port",
        "core_status": "local_port_pending_validation",
        "aliases": [_slug(args.name).lower()],
        "source": {
            "kind": "local_port",
            "path": source,
        },
        "runner": {
            "kind": "scrbenchmark_cli",
            "algorithm": algorithm,
            "default_params": {},
        },
        "output": {
            "expected_file": "results/results.csv",
        },
        "notes": "This method is still registered through methods/*.yaml for reproduction and UI benchmark plans.",
    }


def main() -> int:
    args = parse_args()
    slug = _slug(args.name)
    source = args.source or f"external/original_code/{slug}"
    output_path = Path(args.output or REPO_ROOT / "methods" / f"{slug}.yaml")
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path

    if output_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path}. Use --force to replace it.")

    if args.runner == "scrbenchmark_cli":
        payload = _scrbenchmark_cli_spec(args, source)
    else:
        payload = _command_template_spec(args, source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(payload, Dumper=_IndentedDumper, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    rel_output = output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path
    print(f"Created {rel_output}")
    print()
    print("Next steps:")
    print(f"1. Edit {rel_output} and adapt runner.command / output.* to the author code.")
    print(f"2. Check the generated command:")
    print(f"   python3 scripts/reproduction/validate_method.py --method {args.name} --data data/small_test.h5ad --n-labels 8")
    print(f"3. Run a smoke test:")
    print(f"   python3 scripts/reproduction/validate_method.py --method {args.name} --data data/small_test.h5ad --n-labels 8 --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
