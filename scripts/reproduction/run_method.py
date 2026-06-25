#!/usr/bin/env python3
"""Uniform launcher for SCRBenchmark method specs.

This script is the public reproduction entry point for one method on one
dataset. It delegates to the correct backend while preserving a single CLI
contract and a single method registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from _runner_utils import REPO_ROOT, reproduction_env, run_logged, write_failure


SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scrbenchmark.methods import MethodSpec, get_method_spec, load_method_specs  # noqa: E402


DEFAULT_RESOLUTIONS = ",".join(f"{x / 20:.2f}".rstrip("0").rstrip(".") for x in range(1, 61))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List registered methods and exit.")
    parser.add_argument("--method", help="Canonical method name or alias from methods/*.yaml.")
    parser.add_argument("--data", help="Input .h5ad file.")
    parser.add_argument("--output", help="Output directory.")
    parser.add_argument("--dataset-key", default="")
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-pcs", type=int, default=50, help="PCA dimensions for PCA/Harmony-style runners.")
    parser.add_argument("--harmony-max-iter", type=int, default=10, help="Harmony max_iter_harmony.")
    parser.add_argument("--harmony-nclust", type=int, default=50, help="Harmony nclust.")
    parser.add_argument(
        "--resolutions",
        default=DEFAULT_RESOLUTIONS,
        help="Leiden resolution grid passed to baseline runners.",
    )
    parser.add_argument(
        "--selection-expected-n-classes",
        type=int,
        default=0,
        help="Override class count used by baseline Leiden resolution selection.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra SCRBenchmark CLI param, e.g. algo:key=value. Can be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _join(cmd: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def list_methods() -> None:
    specs = {
        name: spec
        for name, spec in load_method_specs().items()
        if name == spec.name
    }
    print(f"{'method':<28} {'runner':<20} {'core_status'}")
    print("-" * 88)
    for name in sorted(specs):
        spec = specs[name]
        print(f"{spec.name:<28} {spec.runner_kind:<20} {spec.core_status}")


def _data_path(args: argparse.Namespace) -> str:
    return str(Path(str(args.data)).expanduser().resolve())


def _output_path(args: argparse.Namespace) -> str:
    return str(Path(str(args.output)).expanduser().resolve())


def _format_param_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))


def _dataset_key(args: argparse.Namespace) -> str:
    return str(args.dataset_key or Path(str(args.data)).stem)


def _source_path(spec: MethodSpec) -> str:
    raw_path = (spec.source or {}).get("path")
    if not raw_path:
        return ""
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _param_args(args: argparse.Namespace) -> list[str]:
    parts: list[str] = []
    for param in args.param or []:
        parts.extend(["--param", str(param)])
    return parts


def _template_context(spec: MethodSpec, args: argparse.Namespace) -> dict[str, Any]:
    output = _output_path(args)
    return {
        "repo_root": str(REPO_ROOT),
        "source_path": _source_path(spec),
        "method": spec.name,
        "method_name": spec.name,
        "data": _data_path(args),
        "output": output,
        "results_dir": str(Path(output) / "results"),
        "raw_dir": str(Path(output) / "raw"),
        "dataset_key": _dataset_key(args),
        "label_key": str(args.label_key),
        "batch_key": str(args.batch_key),
        "n_labels": str(int(args.n_labels)),
        "seed": str(int(args.seed)),
        "device": str(args.device),
        "python_bin": str(args.python_bin),
        "scib_n_jobs": str(int(args.scib_n_jobs)),
        "n_top_genes": str(int(args.n_top_genes)),
        "min_genes_per_cell": str(int(args.min_genes_per_cell)),
        "max_genes_per_cell": str(int(args.max_genes_per_cell)),
        "min_cells_per_gene": str(int(args.min_cells_per_gene)),
        "target_sum": str(float(args.target_sum)),
        "scale_max_value": str(float(args.scale_max_value)),
        "hvg_flavor": str(args.hvg_flavor),
        "n_pcs": str(int(args.n_pcs)),
        "harmony_max_iter": str(int(args.harmony_max_iter)),
        "harmony_nclust": str(int(args.harmony_nclust)),
        "resolutions": str(args.resolutions),
        "selection_expected_n_classes": str(int(args.selection_expected_n_classes)),
        "param_args": _param_args(args),
        "overwrite_flag": ["--overwrite"] if args.overwrite else [],
        "verbose_flag": ["--verbose"] if args.verbose else [],
    }


def _format_template_part(part: Any, context: Mapping[str, Any]) -> list[str]:
    text = str(part)
    if text.startswith("{") and text.endswith("}") and text.count("{") == 1 and text.count("}") == 1:
        key = text[1:-1]
        value = context.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]
    try:
        formatted = text.format(**{key: value for key, value in context.items() if not isinstance(value, list)})
    except KeyError as exc:
        known = ", ".join(sorted(context))
        raise ValueError(f"Unknown command_template placeholder {exc.args[0]!r}. Known placeholders: {known}") from exc
    return [formatted] if formatted else []


def _build_command_template(spec: MethodSpec, args: argparse.Namespace) -> list[str]:
    template = spec.runner.get("command")
    if not template:
        raise ValueError(f"Method {spec.name!r} uses command_template without runner.command.")

    context = _template_context(spec, args)
    if isinstance(template, str):
        rendered = template.format(**{key: value for key, value in context.items() if not isinstance(value, list)})
        return shlex.split(rendered)

    if not isinstance(template, Sequence):
        raise ValueError(f"Method {spec.name!r} runner.command must be a list or a string.")

    command: list[str] = []
    for part in template:
        command.extend(_format_template_part(part, context))
    return command


def _common_scrbenchmark_args(
    args: argparse.Namespace,
    algo: str,
    default_params: Mapping[str, Any] | None = None,
) -> list[str]:
    if int(args.n_labels) <= 0:
        raise ValueError("--n-labels must be provided for SCRBenchmark CLI methods.")
    cmd = [
        str(args.python_bin),
        str(REPO_ROOT / "src" / "scrbenchmark" / "cli.py"),
        "run",
        "--data",
        _data_path(args),
        "--algorithms",
        str(algo),
        "--output",
        _output_path(args),
        "--no-timestamp",
        "--label-col",
        str(args.label_key),
        "--n-clusters",
        str(int(args.n_labels)),
        "--n-top-genes",
        str(int(args.n_top_genes)),
        "--min-genes-per-cell",
        str(int(args.min_genes_per_cell)),
        "--max-genes-per-cell",
        str(int(args.max_genes_per_cell)),
        "--min-cells-per-gene",
        str(int(args.min_cells_per_gene)),
        "--target-sum",
        str(float(args.target_sum)),
        "--scale-max-value",
        str(float(args.scale_max_value)),
        "--hvg-flavor",
        str(args.hvg_flavor),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--n-repeats",
        "1",
        "--csv",
        "--save-labels",
        "--save-embeddings",
        "--param",
        f"{algo}:n_clusters={int(args.n_labels)}",
    ]
    for key, value in (default_params or {}).items():
        cmd.extend(["--param", f"{algo}:{key}={_format_param_value(value)}"])
    for param in args.param or []:
        cmd.extend(["--param", str(param)])
    return cmd


def build_command(spec: MethodSpec, args: argparse.Namespace) -> list[str]:
    kind = spec.runner_kind
    runner: Mapping[str, Any] = spec.runner



    if kind == "scrbenchmark_cli":
        algo = str(runner.get("algorithm") or spec.name)
        default_params = runner.get("default_params") or {}
        if not isinstance(default_params, Mapping):
            default_params = {}
        return _common_scrbenchmark_args(args, algo, default_params=default_params)

    if kind == "command_template":
        return _build_command_template(spec, args)

    if kind == "batch_baseline":
        return [
            str(args.python_bin),
            str(REPO_ROOT / "scripts" / "reproduction" / "run_batch_baselines.py"),
            "--data",
            _data_path(args),
            "--output",
            _output_path(args),
            "--method",
            str(runner.get("method") or spec.name),
            "--seed",
            str(int(args.seed)),
            "--label-key",
            str(args.label_key),
            "--batch-key",
            str(args.batch_key),
            "--n-top-genes",
            str(int(args.n_top_genes)),
            "--n-pcs",
            "50",
            "--resolutions",
            str(args.resolutions),
            "--selection-expected-n-classes",
            str(int(args.selection_expected_n_classes)),
            "--compute-scib",
            "--scib-n-jobs",
            str(int(args.scib_n_jobs)),
            "--skip-umap-plots",
            "--verbose",
        ]

    if kind == "external_method":
        if int(args.n_labels) <= 0:
            raise ValueError("--n-labels must be provided for external methods.")
        cmd = [
            str(args.python_bin),
            str(REPO_ROOT / "scripts" / "reproduction" / "run_external_method.py"),
            "--method",
            spec.name,
            "--data",
            _data_path(args),
            "--output",
            _output_path(args),
            "--dataset-key",
            _dataset_key(args),
            "--label-key",
            str(args.label_key),
            "--batch-key",
            str(args.batch_key),
            "--n-labels",
            str(int(args.n_labels)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
            "--scib-n-jobs",
            str(int(args.scib_n_jobs)),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.verbose:
            cmd.append("--verbose")
        for param in args.param or []:
            cmd.extend(["--param", str(param)])
        return cmd

    if kind == "posthoc_harmony":
        if int(args.n_labels) <= 0:
            raise ValueError("--n-labels must be provided for posthoc Harmony protocols.")
        cmd = [
            str(args.python_bin),
            str(REPO_ROOT / "scripts" / "reproduction" / "run_posthoc_harmony.py"),
            "--method",
            spec.name,
            "--data",
            _data_path(args),
            "--output",
            _output_path(args),
            "--label-key",
            str(args.label_key),
            "--batch-key",
            str(args.batch_key),
            "--n-labels",
            str(int(args.n_labels)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
            "--scib-n-jobs",
            str(int(args.scib_n_jobs)),
        ]
        if args.verbose:
            cmd.append("--verbose")
        return cmd

    if kind == "passthrough_script":
        script = runner.get("script")
        if not script:
            raise ValueError(f"Method {spec.name!r} uses passthrough_script without runner.script.")
        return [
            str(args.python_bin),
            str((REPO_ROOT / str(script)).resolve()),
            "--method",
            spec.name,
            "--data",
            _data_path(args),
            "--output",
            _output_path(args),
            "--dataset-key",
            _dataset_key(args),
            "--label-key",
            str(args.label_key),
            "--batch-key",
            str(args.batch_key),
            "--n-labels",
            str(int(args.n_labels)),
            "--seed",
            str(int(args.seed)),
            "--device",
            str(args.device),
        ]

    raise ValueError(f"Unsupported runner kind {kind!r} for method {spec.name!r}.")


def _resolve_output_path(args: argparse.Namespace, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return Path(_output_path(args)) / path


def expected_output_path(spec: MethodSpec, args: argparse.Namespace) -> Path:
    return _resolve_output_path(args, spec.expected_file)


def _read_table(path: Path, *, header: bool = True):
    import pandas as pd

    if path.suffix.lower() == ".tsv":
        sep = "\t"
    else:
        sep = None
    if header:
        return pd.read_csv(path, sep=sep, engine="python")
    return pd.read_csv(path, sep=sep, engine="python", header=None)


def _fallback_cell_ids(data_path: str, n_rows: int) -> list[str]:
    try:
        import anndata as ad

        adata = ad.read_h5ad(data_path, backed="r")
        try:
            if adata.n_obs == n_rows:
                return [str(value) for value in adata.obs_names]
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()
    except Exception:
        pass
    return [str(index) for index in range(n_rows)]


def _normalize_command_template_output(spec: MethodSpec, args: argparse.Namespace) -> None:
    output = spec.output or {}
    labels_file = output.get("labels_file")
    if not labels_file:
        return

    import numpy as np
    import pandas as pd

    labels_path = _resolve_output_path(args, labels_file)
    if not labels_path.exists():
        raise FileNotFoundError(f"labels_file declared for {spec.name!r} does not exist: {labels_path}")

    labels_header = bool(output.get("labels_header", True))
    labels_df = _read_table(labels_path, header=labels_header)
    cell_id_column = str(output.get("cell_id_column") or "cell_id")
    labels_column = str(output.get("labels_column") or "cluster")

    if cell_id_column in labels_df.columns:
        cell_ids = labels_df[cell_id_column].astype(str).tolist()
    else:
        cell_ids = _fallback_cell_ids(_data_path(args), len(labels_df))

    if labels_column in labels_df.columns:
        clusters = labels_df[labels_column].tolist()
    elif len(labels_df.columns) == 1:
        clusters = labels_df.iloc[:, 0].tolist()
    else:
        raise ValueError(
            f"Cannot find labels column {labels_column!r} in {labels_path}. "
            "Declare output.labels_column or provide a one-column labels file."
        )

    normalized = pd.DataFrame({"cell_id": cell_ids, "cluster": clusters})

    latent_file = output.get("latent_file")
    if latent_file:
        latent_path = _resolve_output_path(args, latent_file)
        if not latent_path.exists():
            raise FileNotFoundError(f"latent_file declared for {spec.name!r} does not exist: {latent_path}")
        if latent_path.suffix.lower() == ".npy":
            latent = np.load(latent_path)
            if latent.ndim == 1:
                latent = latent.reshape(-1, 1)
            latent_df = pd.DataFrame(latent)
        else:
            latent_df = _read_table(latent_path, header=bool(output.get("latent_header", True)))
            if cell_id_column in latent_df.columns:
                latent_df = latent_df.drop(columns=[cell_id_column])
        if len(latent_df) != len(normalized):
            raise ValueError(
                f"latent_file row count ({len(latent_df)}) does not match labels row count ({len(normalized)})."
            )
        for index, column in enumerate(latent_df.columns, start=1):
            normalized[f"latent_{index}"] = latent_df[column].to_numpy()

    destination = expected_output_path(spec, args)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(destination, index=False)


def finalize_method_output(spec: MethodSpec, args: argparse.Namespace) -> None:
    if spec.runner_kind == "command_template":
        _normalize_command_template_output(spec, args)
        expected = expected_output_path(spec, args)
        if not expected.exists():
            raise FileNotFoundError(
                f"Method {spec.name!r} finished but expected output was not found: {expected}. "
                "Either make the command write output.expected_file or declare output.labels_file."
            )


def write_uniform_manifest(spec: MethodSpec, args: argparse.Namespace, command: list[str]) -> None:
    output_dir = Path(_output_path(args))
    config_dir = output_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": spec.name,
        "display_name": spec.display_name,
        "family": spec.family,
        "runner": dict(spec.runner),
        "source": dict(spec.source),
        "core_contract": spec.core_contract,
        "core_status": spec.core_status,
        "args": vars(args),
        "command": command,
        "command_string": _join(command),
    }
    (config_dir / "method_run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (config_dir / "method_spec_used.yaml").write_text(
        yaml.safe_dump(
            {
                "name": spec.name,
                "display_name": spec.display_name,
                "family": spec.family,
                "runner": dict(spec.runner),
                "source": dict(spec.source),
                "core_contract": spec.core_contract,
                "core_status": spec.core_status,
                "aliases": list(spec.aliases),
                "output": dict(spec.output or {}),
                "notes": spec.notes,
            },
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.list:
        list_methods()
        return 0

    missing = [name for name in ["method", "data", "output"] if not getattr(args, name)]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join('--' + name for name in missing)}")

    spec = get_method_spec(str(args.method))
    if spec is None:
        known = sorted({spec.name for spec in load_method_specs().values()})
        raise SystemExit(f"Unknown method {args.method!r}. Known methods: {', '.join(known)}")

    output_dir = Path(_output_path(args))
    command = build_command(spec, args)
    write_uniform_manifest(spec, args, command)

    if args.dry_run:
        print(_join(command))
        return 0

    try:
        run_logged(command, output_dir / "logs" / "run_method.log", env=reproduction_env())
        finalize_method_output(spec, args)
        failure_path = output_dir / "results" / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
    except Exception as exc:
        write_failure(output_dir / "results" / "failure.json", method=spec.name, error=exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
