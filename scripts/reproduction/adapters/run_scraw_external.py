#!/usr/bin/env python3
"""Run the self-contained vendored scRAW backend through the method registry."""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRAW_SOURCE_DIR = REPO_ROOT / "vendor" / "scraw_inductive" / "src"
if not SCRAW_SOURCE_DIR.exists():
    raise RuntimeError(f"Vendored scRAW source directory is missing: {SCRAW_SOURCE_DIR}")
sys.path.insert(0, str(SCRAW_SOURCE_DIR))

PUBLIC_PRESETS = ("baron", "default")
CONFIG_SECTIONS = (
    "data",
    "runtime",
    "preprocessing",
    "model",
    "training",
    "weighting",
    "triplet",
    "clustering",
    "batch_correction",
    "outputs",
)
PARAM_ALIASES = {
    "lr": "training.learning_rate",
    "z_dim": "model.latent_dim",
    "rare_triplet_weight": "triplet.weight",
    "rare_triplet_start_epoch": "triplet.start_epoch",
    "rare_triplet_margin": "triplet.margin",
    "rare_triplet_min_weight": "triplet.min_anchor_weight",
    "max_triplet_anchors_per_batch": "triplet.max_anchors_per_batch",
    "use_batch_conditioning": "batch_correction.enabled",
    "adversarial_batch_weight": "batch_correction.adversarial_weight",
    "mmd_batch_weight": "batch_correction.mmd_weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the vendored scRAW backend")
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preset",
        choices=PUBLIC_PRESETS,
        default="default",
        help=(
            "Vendored public preset. `default` is the 0017/stable configuration; "
            "`baron` is the Baron configuration."
        ),
    )
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "scRAW config override as key=value or scraw:key=value. "
            "Dotted paths such as training.epochs=20 are supported."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _parse_param_value(raw: str) -> Any:
    """Parse one CLI value with YAML scalar/list semantics."""
    return yaml.safe_load(str(raw))


def _manual_params(values: list[str], method: str = "scraw") -> dict[str, Any]:
    """Return config overrides addressed to scRAW from repeated ``--param`` values."""
    overrides: dict[str, Any] = {}
    accepted_targets = {str(method).strip().lower(), "scraw", "*", "all"}
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(
                f"Invalid --param value {raw!r}; expected key=value or scraw:key=value."
            )
        left, raw_value = text.split("=", 1)
        if ":" in left:
            target, key = left.split(":", 1)
            if target.strip().lower() not in accepted_targets:
                continue
        else:
            key = left
        key = key.strip().replace("-", "_")
        if not key:
            raise ValueError(f"Invalid --param value {raw!r}; the parameter name is empty.")
        overrides[key] = _parse_param_value(raw_value)
    return overrides


def _coerce_config_value(current: Any, value: Any, dotted_key: str) -> Any:
    """Preserve dataclass field types while accepting convenient CLI representations."""
    if current is None or value is None:
        return value
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{dotted_key} expects a boolean, got {value!r}.")
        return value
    if isinstance(current, int):
        if isinstance(value, bool):
            raise ValueError(f"{dotted_key} expects an integer, got {value!r}.")
        return int(value)
    if isinstance(current, float):
        if isinstance(value, bool):
            raise ValueError(f"{dotted_key} expects a number, got {value!r}.")
        return float(value)
    if isinstance(current, list):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{dotted_key} expects a list, got {value!r}.")
        return [int(item) for item in value]
    if isinstance(current, str):
        return str(value)
    return value


def _resolve_param_path(config: Any, key: str) -> tuple[Any, str, str]:
    """Resolve a flat or dotted config key to its owning object and canonical path."""
    canonical_key = PARAM_ALIASES.get(key, key)
    if "." in canonical_key:
        section_name, field_name, *extra = canonical_key.split(".")
        if extra or section_name not in CONFIG_SECTIONS:
            raise ValueError(f"Unknown scRAW config override {key!r}.")
        section = getattr(config, section_name, None)
        if section is None or not hasattr(section, field_name):
            raise ValueError(f"Unknown scRAW config override {key!r}.")
        return section, field_name, canonical_key

    matches: list[tuple[Any, str, str]] = []
    for section_name in CONFIG_SECTIONS:
        section = getattr(config, section_name, None)
        if section is not None and hasattr(section, canonical_key):
            matches.append((section, canonical_key, f"{section_name}.{canonical_key}"))
    if not matches:
        raise ValueError(
            f"Unknown scRAW config override {key!r}. Use a dotted path such as "
            "training.epochs=20."
        )
    if len(matches) > 1:
        choices = ", ".join(match[2] for match in matches)
        raise ValueError(
            f"Ambiguous scRAW config override {key!r}; use one of: {choices}."
        )
    return matches[0]


def apply_manual_params(config: Any, values: list[str], method: str = "scraw") -> dict[str, Any]:
    """Apply validated command-line overrides to a ``ScRAWConfig`` instance."""
    if not is_dataclass(config):
        raise TypeError("config must be a ScRAWConfig dataclass instance.")

    applied: dict[str, Any] = {}
    for key, value in _manual_params(values, method=method).items():
        target, attr, canonical_key = _resolve_param_path(config, key)
        valid_fields = {field.name for field in fields(target)}
        if attr not in valid_fields:
            raise ValueError(f"Unknown scRAW config override {key!r}.")
        coerced = _coerce_config_value(getattr(target, attr), value, canonical_key)
        setattr(target, attr, coerced)
        applied[canonical_key] = coerced

        if canonical_key == "triplet.weight":
            config.triplet.enabled = float(coerced) > 0.0
    return applied


def apply_preprocessing_args(config: Any, args: argparse.Namespace) -> None:
    """Copy the shared runner's preprocessing flags into the scRAW config."""
    config.preprocessing.n_top_genes = int(args.n_top_genes)
    config.preprocessing.min_genes_per_cell = int(args.min_genes_per_cell)
    config.preprocessing.max_genes_per_cell = int(args.max_genes_per_cell)
    config.preprocessing.min_cells_per_gene = int(args.min_cells_per_gene)
    config.preprocessing.target_sum = float(args.target_sum)
    config.preprocessing.scale_max_value = float(args.scale_max_value)
    config.preprocessing.hvg_flavor = str(args.hvg_flavor)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """Remove stale backend artifacts while preserving the generic runner manifest/logs."""
    if not overwrite:
        return
    for name in ("results", "figures", "models"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
    config_used = output_dir / "config" / "config_used.json"
    if config_used.exists():
        config_used.unlink()


def main() -> int:
    args = parse_args()

    from scraw.pipeline import run_pipeline
    from scraw.presets import resolve_preset_config

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    config = resolve_preset_config(str(args.preset))

    apply_preprocessing_args(config, args)
    apply_manual_params(config, list(args.param), method=str(args.method))

    # Configure path values
    config.data.data_path = str(Path(args.data).expanduser().resolve())
    config.data.output_dir = str(output_dir)
    config.data.label_key = str(args.label_key).strip() or None

    # Configure runtime properties
    config.runtime.seed = int(args.seed)
    config.runtime.device = str(args.device)

    # Target number of clusters
    if int(args.n_labels) > 0:
        config.clustering.pseudo_k = int(args.n_labels)

    # Configure batch correction
    if args.batch_key:
        config.batch_correction.key = str(args.batch_key)
        config.batch_correction.enabled = True
    else:
        config.batch_correction.enabled = False

    # Execute the vendored pipeline.
    run_pipeline(config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
