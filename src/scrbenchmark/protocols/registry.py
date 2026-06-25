"""Versioned benchmark protocol registry.

The GUI and CLI helpers use this module to load report-derived benchmark
designs from YAML, validate editable configurations, expand sweeps, write
execution manifests, and aggregate finished result folders.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOLS_DIR = REPO_ROOT / "protocols"
REPORT_DATASET_TABLE = REPO_ROOT / "reproducibility" / "stable_generalist" / "stable_generalist_dataset_table.csv"
REPORT_DATA_ROOT = Path("data") / "stable_generalist"

PLAN_FIELDS = [
    "job_id",
    "config_index",
    "config_name",
    "protocol_id",
    "command_index",
    "output_dir",
    "status",
    "command",
]


@dataclass(frozen=True)
class ProtocolSpec:
    """One reusable benchmark design loaded from YAML."""

    id: str
    name: str
    experiment_type: str
    description: str
    raw: Mapping[str, Any]
    path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], path: Path | None = None) -> "ProtocolSpec":
        protocol_id = str(raw.get("id", "")).strip()
        if not protocol_id:
            raise ValueError(f"Protocol in {path or '<memory>'} is missing required field 'id'.")
        return cls(
            id=protocol_id,
            name=str(raw.get("name") or protocol_id),
            experiment_type=str(raw.get("experiment_type") or raw.get("type") or "custom"),
            description=str(raw.get("description") or ""),
            raw=dict(raw),
            path=path,
        )

    @property
    def tags(self) -> list[str]:
        return [str(item) for item in self.raw.get("tags", [])]


@dataclass
class ValidationResult:
    """Structured validation outcome for one editable benchmark config."""

    errors: list[str]
    warnings: list[str]
    infos: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def rows(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for severity, items in (
            ("error", self.errors),
            ("warning", self.warnings),
            ("info", self.infos),
        ):
            out.extend({"severity": severity, "message": item} for item in items)
        return out


def _read_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load benchmark protocols.") from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _safe_dump_yaml(value: Any) -> str:
    try:
        import yaml

        return yaml.safe_dump(value, sort_keys=False, allow_unicode=False)
    except Exception:
        return json.dumps(value, indent=2, ensure_ascii=True)


def _iter_protocol_files(protocols_dir: Path) -> Iterable[Path]:
    if not protocols_dir.exists():
        return []
    return sorted(
        path
        for path in protocols_dir.rglob("*.yaml")
        if not path.name.startswith("_") and path.name != "template_protocol.yaml"
    )


def _iter_raw_protocols(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, Mapping) and isinstance(payload.get("protocols"), list):
        for item in payload["protocols"]:
            if isinstance(item, Mapping):
                yield item
        return
    if isinstance(payload, Mapping) and payload.get("id"):
        yield payload


def load_protocol_specs(protocols_dir: str | Path | None = None) -> dict[str, ProtocolSpec]:
    """Load all protocol YAML files keyed by protocol id."""

    root = Path(protocols_dir).expanduser().resolve() if protocols_dir else DEFAULT_PROTOCOLS_DIR
    specs: dict[str, ProtocolSpec] = {}
    for path in _iter_protocol_files(root):
        payload = _read_yaml(path)
        for raw in _iter_raw_protocols(payload):
            spec = ProtocolSpec.from_mapping(raw, path=path)
            if spec.id in specs:
                raise ValueError(f"Duplicate protocol id: {spec.id}")
            specs[spec.id] = spec
    return specs


def get_protocol_spec(protocol_id: str, protocols_dir: str | Path | None = None) -> ProtocolSpec | None:
    """Return a protocol by id."""

    return load_protocol_specs(protocols_dir).get(str(protocol_id))


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "item"


def _tokens(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        values = []
        for item in raw:
            values.extend(str(item).split(","))
    else:
        values = [str(raw)]
    return [value.strip() for value in values if value and value.strip()]


def _stable_generalist_target_name(raw_path: Any) -> str:
    name = Path(str(raw_path)).name
    aliases = {"pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad"}
    return aliases.get(name, name)


def _load_report_dataset_specs() -> dict[str, dict[str, Any]]:
    if not REPORT_DATASET_TABLE.exists():
        return {}
    try:
        import pandas as pd

        frame = pd.read_csv(REPORT_DATASET_TABLE)
    except Exception:
        return {}

    specs: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        dataset_key = str(row.get("dataset_key", "")).strip()
        if not dataset_key:
            continue
        label_key = row.get("label_key", "Group")
        batch_key = row.get("dann_batch_column", "batch")
        n_labels = row.get("n_labels", 0)
        data_file = row.get("data_file", f"{dataset_key}.h5ad")
        specs[dataset_key] = {
            "key": dataset_key,
            "display_name": str(row.get("dataset", dataset_key)).strip() or dataset_key,
            "path": str(REPORT_DATA_ROOT / _stable_generalist_target_name(data_file)),
            "label_key": str(label_key if not pd.isna(label_key) else "Group").strip() or "Group",
            "batch_key": str(batch_key if not pd.isna(batch_key) else "batch").strip() or "batch",
            "n_labels": int(n_labels if not pd.isna(n_labels) else 0),
        }
    return specs


def _dataset_from_key(dataset_key: str) -> dict[str, Any]:
    specs = _load_report_dataset_specs()
    if dataset_key in specs:
        return copy.deepcopy(specs[dataset_key])
    return {
        "key": dataset_key,
        "display_name": dataset_key,
        "path": str(REPORT_DATA_ROOT / f"{dataset_key}.h5ad"),
        "label_key": "Group",
        "batch_key": "batch",
        "n_labels": 0,
    }


def _normalise_dataset(raw: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw, str):
        dataset = _dataset_from_key(raw)
    else:
        dataset_key = str(raw.get("key") or raw.get("dataset_key") or "").strip()
        dataset = _dataset_from_key(dataset_key) if dataset_key else {}
        dataset.update({str(key): value for key, value in raw.items() if value is not None})
        if "dataset_key" in dataset and "key" not in dataset:
            dataset["key"] = dataset["dataset_key"]
    dataset.setdefault("key", Path(str(dataset.get("path", "dataset.h5ad"))).stem)
    dataset.setdefault("display_name", dataset["key"])
    dataset.setdefault("path", str(REPORT_DATA_ROOT / f"{dataset['key']}.h5ad"))
    dataset.setdefault("label_key", "Group")
    dataset.setdefault("batch_key", "batch")
    dataset.setdefault("n_labels", 0)
    return dataset


def _report_preprocessing_defaults() -> dict[str, Any]:
    return {
        "do_cell_filtering": True,
        "do_gene_filtering": True,
        "min_genes_per_cell": 200,
        "max_genes_per_cell": 10000,
        "min_cells_per_gene": 3,
        "do_normalization": True,
        "do_log_transform": True,
        "target_sum": 20000,
        "do_hvg": True,
        "n_top_genes": 2000,
        "hvg_flavor": "seurat",
        "hvg_strategy": "train_only",
        "do_scaling": True,
        "scale_max_value": 10.0,
        "do_batch_correction": False,
        "batch_correction_method": "scvi",
        "batch_correction_batch_key": "batch",
    }


def _default_manual_protocols() -> dict[str, Any]:
    return {
        "enabled": False,
        "selected_protocols": [],
        "seeds": "42",
        "scib_n_jobs": 4,
        "overwrite": False,
        "verbose": True,
        "extra_params": "",
        "loss_transfer": {
            "methods": ["scMAE", "scDeepCluster", "DESC"],
            "variants": ["baseline", "weighted"],
            "weight_params": "",
        },
        "harmony": {
            "methods": [
                "Harmony",
                "scMAE+Harmony",
                "scNAME+Harmony",
                "scvi+Harmony",
                "DeepScena+Harmony",
                "CellSIUS+Harmony",
                "GiniClust+Harmony",
                "scAIDE+Harmony",
                "scCAD+Harmony",
            ],
            "n_pcs": 50,
            "harmony_max_iter": 10,
            "harmony_nclust": 50,
        },
        "inductive": {
            "algorithms": ["scraw", "scname", "sc_mae", "scdeepcluster"],
            "split_key": "batch",
            "train_split_key": "",
            "test_split_key": "",
            "train_batches": "",
            "test_batches": "",
            "preset": "stable_generalist",
            "trial_config_path": "",
            "baseline_runtime_profile": "scrbenchmark-default",
            "skip_existing": False,
        },
    }


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _config_preprocessing(raw: Mapping[str, Any]) -> dict[str, Any]:
    pre = _report_preprocessing_defaults()
    update = dict(raw or {})
    preset = str(update.pop("preset", "report_default"))
    if preset in {"none", "minimal"}:
        pre = {
            "do_cell_filtering": True,
            "do_gene_filtering": True,
            "min_genes_per_cell": 100,
            "max_genes_per_cell": 10000,
            "min_cells_per_gene": 3,
            "do_normalization": True,
            "do_log_transform": True,
            "target_sum": 20000,
            "do_hvg": True,
            "n_top_genes": 2000,
            "hvg_flavor": "seurat",
            "hvg_strategy": "train_only",
            "do_scaling": True,
            "scale_max_value": 10.0,
            "do_batch_correction": False,
            "batch_correction_method": "scvi",
            "batch_correction_batch_key": "batch",
        }
    pre.update(update)
    return pre


def _split_config(split: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(split.get("mode", "all")).lower().replace("-", "_")
    label_key = str(dataset.get("label_key", "Group"))
    batch_key = str(dataset.get("batch_key", "batch"))

    if mode in {"all", "standard", "transductive"}:
        return {"mode": "standard", "original_settings": {}}

    if mode in {"stratified", "classical", "train_val_test"}:
        train_ratio = float(split.get("train_ratio", 0.8))
        val_ratio = float(split.get("val_ratio", 0.1))
        return {
            "mode": "benchmark",
            "original_settings": {
                "mode": "stratified",
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
                "test_ratio": max(0.0, 1.0 - train_ratio - val_ratio),
                "use_validation": val_ratio > 0.0,
                "stratify_by_batch": bool(split.get("stratify_by_batch", True)),
                "stratify_by_labels": bool(split.get("stratify_by_labels", True)),
                "use_stratified_base": bool(split.get("use_stratified_base", True)),
                "batch_col": str(split.get("batch_col", batch_key)),
                "label_col": str(split.get("label_col", label_key)),
            },
        }

    if mode in {"batch", "leave_one_batch"}:
        train_batches = _tokens(split.get("train_batches", []))
        test_batches = _tokens(split.get("test_batches", []))
        return {
            "mode": "benchmark",
            "original_settings": {
                "mode": "batch",
                "batch_col": str(split.get("batch_col", batch_key)),
                "train_batches": train_batches,
                "test_batches": test_batches or None,
                "use_stratified_base": not bool(test_batches),
                "use_validation": bool(split.get("use_validation", True)),
                "train_ratio": float(split.get("train_ratio", 0.7)),
                "val_ratio": float(split.get("val_ratio", 0.1)),
                "test_ratio": float(split.get("test_ratio", 0.2)),
                "stratify_by_labels": bool(split.get("stratify_by_labels", True)),
                "balance": bool(split.get("balance", False)),
                "balance_target": split.get("balance_target", "auto"),
                "balance_mode": str(split.get("balance_mode", "eliminate")),
            },
        }

    return {"mode": "standard", "original_settings": {}}


def _base_customize_config(spec: ProtocolSpec, dataset: Mapping[str, Any], name: str) -> dict[str, Any]:
    raw = spec.raw
    execution = raw.get("execution", {}) or {}
    methods = raw.get("methods", {}) or {}
    split = raw.get("split", {}) or {}
    manual_raw = raw.get("manual_protocols", {}) or {}
    report_method = raw.get("report_method", {}) or {}

    manual_protocols = _default_manual_protocols()
    if manual_raw:
        _deep_merge(manual_protocols, manual_raw)
    if execution.get("seeds") and not manual_raw.get("seeds"):
        manual_protocols["seeds"] = str(execution["seeds"])

    report_params = methods.get("report_params", report_method.get("params", ""))
    if isinstance(report_params, Sequence) and not isinstance(report_params, (str, bytes, bytearray)):
        report_params = "\n".join(str(item) for item in report_params)

    config = {
        "name": name,
        "uploaded_file_path": str(dataset.get("path")),
        "dataset_key": str(dataset.get("key")),
        "label_key": str(dataset.get("label_key", "Group")),
        "batch_key": str(dataset.get("batch_key", "batch")),
        "n_labels": int(dataset.get("n_labels", 0) or 0),
        "selected_algorithms": list(methods.get("local_algorithms", [])),
        "selected_report_methods": list(methods.get("report_methods", [])),
        "report_method_params": str(report_params or ""),
        "report_method_n_pcs": int(report_method.get("n_pcs", 50) or 50),
        "report_method_harmony_max_iter": int(report_method.get("harmony_max_iter", 10) or 10),
        "report_method_harmony_nclust": int(report_method.get("harmony_nclust", 50) or 50),
        "preprocessing_params": _config_preprocessing(raw.get("preprocessing", {}) or {}),
        "benchmark_configured": True,
        "benchmark_setup": _split_config(split, dataset),
        "balance_settings_standard": dict(raw.get("balance_settings_standard", {}) or {}),
        "global_network_config": {
            "enabled": False,
            "encoder_layers": "256,64",
            "z_dim": 32,
        },
        "algorithm_params": copy.deepcopy(methods.get("algorithm_params", {}) or {}),
        "device_preference": str(execution.get("device", "auto")),
        "n_repeats": int(execution.get("n_repeats", 1) or 1),
        "seed": int(execution.get("seed", 42) or 42),
        "validation_optimization": {
            "enable_lr_optimization": bool(raw.get("validation_optimization", {}).get("enable_lr_optimization", True)),
            "lr_optimization_metric": str(raw.get("validation_optimization", {}).get("lr_optimization_metric", "NMI")),
            "lr_optimization_repeats": int(raw.get("validation_optimization", {}).get("lr_optimization_repeats", 1) or 1),
            "lr_optimization_scales": list(raw.get("validation_optimization", {}).get("lr_optimization_scales", [100.0, 10.0, 1.0, 0.1])),
        },
        "manual_protocols": manual_protocols,
        "output_dir": str(execution.get("output_dir", "results")),
        "save_labels": bool(execution.get("save_labels", True)),
        "save_embeddings": bool(execution.get("save_embeddings", False)),
        "protocol_id": spec.id,
        "protocol_source": str(spec.path or ""),
        "protocol_description": spec.description,
        "protocol_experiment_type": spec.experiment_type,
        "protocol_tags": list(spec.tags),
        "protocol_metrics": list(raw.get("metrics", [])),
        "protocol_limitations": list(raw.get("limitations", [])),
        "sweep_params": copy.deepcopy(raw.get("sweeps", {}) or {}),
    }
    return config


def protocol_to_customize_configs(spec: ProtocolSpec | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a protocol spec into one or more editable UI configurations."""

    if not isinstance(spec, ProtocolSpec):
        spec = ProtocolSpec.from_mapping(spec)
    raw = spec.raw
    configs: list[dict[str, Any]] = []

    inductive_splits = raw.get("inductive_splits") or []
    if inductive_splits:
        for split in inductive_splits:
            if not isinstance(split, Mapping):
                continue
            dataset = _normalise_dataset(split.get("dataset") or split.get("dataset_key") or raw.get("dataset", {}))
            split_name = str(split.get("name") or "split")
            name = f"{spec.name} - {dataset.get('display_name', dataset.get('key'))} {split_name}"
            config = _base_customize_config(spec, dataset, name)
            manual = config.setdefault("manual_protocols", _default_manual_protocols())
            manual["enabled"] = True
            selected = list(manual.get("selected_protocols", []))
            if "inductive" not in selected:
                selected.append("inductive")
            manual["selected_protocols"] = selected
            inductive = manual.setdefault("inductive", {})
            inductive.update({
                "split_key": str(split.get("split_key", dataset.get("batch_key", "batch"))),
                "train_split_key": str(split.get("train_split_key", "")),
                "test_split_key": str(split.get("test_split_key", "")),
                "train_batches": ",".join(_tokens(split.get("train_batches", []))),
                "test_batches": ",".join(_tokens(split.get("test_batches", []))),
            })
            config["name"] = name
            configs.append(config)
        return configs

    datasets_raw = raw.get("datasets")
    if datasets_raw is None:
        datasets_raw = [raw.get("dataset", {})]
    for item in datasets_raw:
        dataset = _normalise_dataset(item)
        name = spec.name
        if len(datasets_raw) > 1:
            name = f"{spec.name} - {dataset.get('display_name', dataset.get('key'))}"
        configs.append(_base_customize_config(spec, dataset, name))
    return configs


def _resolve_repo_path(raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_report_method_names() -> set[str]:
    try:
        from scrbenchmark.methods import load_method_specs
    except Exception:
        try:
            from methods import load_method_specs
        except Exception:
            return set()
    try:
        specs = load_method_specs()
    except Exception:
        return set()
    return {spec.name for spec in specs.values()}


def _load_algorithm_names() -> set[str]:
    try:
        import algorithms  # noqa: F401
        from core.algorithm_registry import AlgorithmRegistry

        return set(AlgorithmRegistry.get_all().keys())
    except Exception:
        return set()


def _check_h5ad_columns(path: Path, label_key: str, batch_key: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if path.suffix.lower() != ".h5ad" or not path.exists():
        return errors, warnings
    try:
        import anndata as ad
    except Exception as exc:
        warnings.append(f"Cannot inspect AnnData columns because anndata is unavailable: {type(exc).__name__}.")
        return errors, warnings

    adata = None
    try:
        adata = ad.read_h5ad(path, backed="r")
        columns = {str(column) for column in adata.obs.columns}
        if label_key and label_key not in columns:
            errors.append(f"Label column {label_key!r} is missing from {path}.")
        if batch_key and batch_key not in columns:
            warnings.append(f"Batch column {batch_key!r} is missing from {path}; batch metrics/Harmony may fail.")
    except Exception as exc:
        warnings.append(f"Cannot inspect {path}: {type(exc).__name__}: {exc}")
    finally:
        try:
            if adata is not None and getattr(adata, "file", None) is not None:
                adata.file.close()
        except Exception:
            pass
    return errors, warnings


def validate_customize_config(config: Mapping[str, Any], *, check_data: bool = True) -> ValidationResult:
    """Validate one editable benchmark configuration before execution."""

    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []

    name = str(config.get("name") or "configuration")
    data_path = _resolve_repo_path(config.get("uploaded_file_path", ""))
    if check_data:
        if not str(config.get("uploaded_file_path", "")).strip():
            errors.append(f"{name}: missing dataset path.")
        elif not data_path.exists():
            errors.append(f"{name}: dataset file does not exist: {data_path}.")
        else:
            col_errors, col_warnings = _check_h5ad_columns(
                data_path,
                str(config.get("label_key", "Group")),
                str(config.get("batch_key", "batch")),
            )
            errors.extend(f"{name}: {item}" for item in col_errors)
            warnings.extend(f"{name}: {item}" for item in col_warnings)

    try:
        n_labels = int(config.get("n_labels", 0) or 0)
    except Exception:
        n_labels = 0
        errors.append(f"{name}: Expected Classes must be an integer.")

    selected_report = [method for method in config.get("selected_report_methods", []) if method]
    report_names = _load_report_method_names()
    if selected_report:
        if n_labels <= 0:
            errors.append(f"{name}: Expected Classes must be > 0 for report methods.")
        if report_names:
            missing = [method for method in selected_report if method not in report_names]
            if missing:
                errors.append(f"{name}: unknown report methods: {', '.join(missing)}.")
        else:
            warnings.append(f"{name}: report method registry could not be loaded.")

    selected_algos = [algo for algo in config.get("selected_algorithms", []) if algo]
    if selected_algos:
        algo_names = _load_algorithm_names()
        if algo_names:
            missing_algos = [algo for algo in selected_algos if algo not in algo_names]
            if missing_algos:
                errors.append(f"{name}: unknown local algorithms: {', '.join(missing_algos)}.")
        else:
            warnings.append(f"{name}: local algorithm registry could not be loaded for validation.")

    setup = config.get("benchmark_setup", {}) or {}
    settings = setup.get("original_settings", {}) if isinstance(setup, Mapping) else {}
    if setup.get("mode") == "benchmark" and settings.get("mode") == "stratified":
        train_ratio = float(settings.get("train_ratio", 0.0) or 0.0)
        val_ratio = float(settings.get("val_ratio", 0.0) or 0.0)
        if train_ratio <= 0.0:
            errors.append(f"{name}: train ratio must be > 0.")
        if train_ratio + val_ratio >= 1.0:
            errors.append(f"{name}: train ratio + validation ratio must be < 1.")

    manual = config.get("manual_protocols", {}) or {}
    if manual.get("enabled"):
        protocols = [str(item) for item in manual.get("selected_protocols", []) if item]
        if not protocols:
            warnings.append(f"{name}: manual protocols are enabled but no protocol is selected.")
        if any(proto in {"loss_transfer", "harmony"} for proto in protocols) and n_labels <= 0:
            errors.append(f"{name}: Expected Classes must be > 0 for loss-transfer/Harmony protocols.")
        if "loss_transfer" in protocols:
            loss = manual.get("loss_transfer", {}) or {}
            if not loss.get("methods"):
                errors.append(f"{name}: loss-transfer protocol needs at least one method.")
            if not loss.get("variants"):
                errors.append(f"{name}: loss-transfer protocol needs at least one variant.")
        if "inductive" in protocols:
            inductive = manual.get("inductive", {}) or {}
            if not _tokens(inductive.get("train_batches")):
                errors.append(f"{name}: inductive protocol needs train groups.")
            if not _tokens(inductive.get("test_batches")):
                errors.append(f"{name}: inductive protocol needs test groups.")
            algorithms = {algo.lower() for algo in _tokens(inductive.get("algorithms", []))}
            if "scvi" in algorithms:
                warnings.append(
                    f"{name}: the current inductive runner does not expose a native scvi backend; "
                    "use pca_harmony or add a runner backend first."
                )

    try:
        n_repeats = int(config.get("n_repeats", 1) or 1)
        if n_repeats <= 0:
            errors.append(f"{name}: Number of Repetitions must be >= 1.")
    except Exception:
        errors.append(f"{name}: Number of Repetitions must be an integer.")

    try:
        parse_sweep_mapping(config.get("sweep_params", {}) or {})
    except Exception as exc:
        errors.append(f"{name}: invalid sweep definition: {exc}")

    if selected_algos or selected_report or manual.get("enabled"):
        infos.append(f"{name}: validation completed for configured jobs.")
    else:
        warnings.append(f"{name}: no local algorithm, report method, or manual protocol is selected.")

    return ValidationResult(errors=errors, warnings=warnings, infos=infos)


def _parse_scalar(raw: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(raw)
    except Exception:
        text = raw.strip()
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        for caster in (int, float):
            try:
                return caster(text)
            except Exception:
                pass
        return text


def parse_sweep_mapping(raw: Mapping[str, Any] | str | None) -> dict[str, list[Any]]:
    """Parse editable sweep definitions into path -> candidate values."""

    if raw is None or raw == "":
        return {}

    parsed: dict[str, list[Any]] = {}
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        lines = []
        for line in str(raw).splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
        items = []
        for line in lines:
            if "=" not in line:
                raise ValueError(f"Invalid sweep line {line!r}; use path=value1,value2.")
            key, value = line.split("=", 1)
            items.append((key.strip(), value.strip()))

    for key, value in items:
        path = str(key).strip()
        if not path:
            continue
        if isinstance(value, list):
            values = value
        elif isinstance(value, tuple):
            values = list(value)
        elif isinstance(value, str):
            text = value.strip()
            loaded = None
            if text.startswith("[") or text.startswith("{"):
                loaded = _parse_scalar(text)
            if isinstance(loaded, list):
                values = loaded
            elif "," in text:
                values = [_parse_scalar(part.strip()) for part in text.split(",") if part.strip()]
            else:
                values = [_parse_scalar(text)]
        else:
            values = [value]
        if not values:
            raise ValueError(f"Sweep path {path!r} has no candidate values.")
        parsed[path] = values
    return parsed


def _update_key_value_text(raw: Any, key: str, value: Any) -> str:
    values: dict[str, str] = {}
    order: list[str] = []
    for line in str(raw or "").replace(",", "\n").splitlines():
        text = line.strip()
        if not text or "=" not in text:
            continue
        current_key, current_value = text.split("=", 1)
        current_key = current_key.strip()
        if current_key not in values:
            order.append(current_key)
        values[current_key] = current_value.strip()
    if key not in values:
        order.append(key)
    values[key] = str(value)
    return "\n".join(f"{item}={values[item]}" for item in order)


def _set_config_path(config: dict[str, Any], path: str, value: Any) -> None:
    path = path.strip()
    aliases = {
        "execution.seed": "seed",
        "execution.n_repeats": "n_repeats",
        "execution.output_dir": "output_dir",
        "execution.device": "device_preference",
        "report_method.n_pcs": "report_method_n_pcs",
        "report_method.harmony_max_iter": "report_method_harmony_max_iter",
        "report_method.harmony_nclust": "report_method_harmony_nclust",
    }
    path = aliases.get(path, path)
    if path.startswith("preprocessing."):
        path = "preprocessing_params." + path.split(".", 1)[1]
    if path.startswith("manual."):
        path = "manual_protocols." + path.split(".", 1)[1]

    parts = path.split(".")
    if len(parts) >= 3 and parts[0] == "algorithm":
        _, algo_name, *param = parts
        config.setdefault("algorithm_params", {}).setdefault(algo_name, {})[".".join(param)] = value
        if algo_name not in config.setdefault("selected_algorithms", []):
            config["selected_algorithms"].append(algo_name)
        return

    if (
        len(parts) >= 4
        and parts[:3] == ["manual_protocols", "loss_transfer", "weight_params"]
    ):
        key = ".".join(parts[3:])
        loss = config.setdefault("manual_protocols", {}).setdefault("loss_transfer", {})
        loss["weight_params"] = _update_key_value_text(loss.get("weight_params", ""), key, value)
        return

    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = value


def expand_sweep_configs(
    config: Mapping[str, Any],
    sweep_text: Mapping[str, Any] | str | None = None,
    *,
    max_configs: int = 500,
) -> list[dict[str, Any]]:
    """Expand one config into a cartesian product of sweep values."""

    base = copy.deepcopy(dict(config))
    sweep = parse_sweep_mapping(sweep_text if sweep_text not in (None, "") else base.get("sweep_params", {}))
    if not sweep:
        return [base]

    keys = list(sweep)
    combinations = list(itertools.product(*(sweep[key] for key in keys)))
    if len(combinations) > max_configs:
        raise ValueError(f"Sweep expands to {len(combinations)} configs; limit is {max_configs}.")

    expanded: list[dict[str, Any]] = []
    for index, values in enumerate(combinations, start=1):
        cfg = copy.deepcopy(base)
        suffix_bits = []
        for key, value in zip(keys, values):
            _set_config_path(cfg, key, value)
            suffix_bits.append(f"{key}={value}")
        cfg["name"] = f"{base.get('name', 'Config')} | sweep {index}: " + ", ".join(suffix_bits)
        output_dir = str(base.get("output_dir", "results") or "results")
        cfg["output_dir"] = f"{output_dir}/sweep_{index:03d}"
        cfg["sweep_assignment"] = dict(zip(keys, values))
        expanded.append(cfg)
    return expanded


def extract_cli_option(command: str, option: str) -> str:
    try:
        parts = shlex.split(command)
    except Exception:
        return ""
    for idx, part in enumerate(parts):
        if part == option and idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def build_job_plan(
    configs: Sequence[Mapping[str, Any]],
    command_builder: Callable[[Mapping[str, Any]], Sequence[str]],
) -> list[dict[str, str]]:
    """Build a flat job plan from editable configs and a command builder callback."""

    rows: list[dict[str, str]] = []
    for config_index, config in enumerate(configs, start=1):
        commands = list(command_builder(config))
        for command_index, command in enumerate(commands, start=1):
            output_dir = extract_cli_option(str(command), "--output") or extract_cli_option(str(command), "--output-root")
            if not output_dir:
                output_dir = str(config.get("output_dir", "results"))
            job_id = _slug(f"{config_index}_{command_index}_{config.get('name', 'config')}")
            rows.append({
                "job_id": job_id,
                "config_index": str(config_index),
                "config_name": str(config.get("name", "")),
                "protocol_id": str(config.get("protocol_id", "")),
                "command_index": str(command_index),
                "output_dir": output_dir,
                "status": "pending",
                "command": str(command),
            })
    return rows


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def write_protocol_artifacts(
    rows: Sequence[Mapping[str, str]],
    output_root: str | Path,
    configs: Sequence[Mapping[str, Any]],
    *,
    name: str = "custom_protocol",
) -> dict[str, str]:
    """Write planned jobs, a runnable shell script, and a protocol manifest."""

    root = _resolve_repo_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    plan_csv = root / "planned_jobs.csv"
    shell_path = root / "run_protocol_jobs.sh"
    manifest_path = root / "protocol_manifest.json"

    with plan_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in PLAN_FIELDS})

    with shell_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n\n")
        for row in rows:
            handle.write(f"# {row.get('job_id', '')} | {row.get('config_name', '')}\n")
            output_dir = row.get("output_dir", "")
            if output_dir:
                handle.write(f"mkdir -p {shlex.quote(str(Path(output_dir).parent))}\n")
            handle.write(f"{row.get('command', '')}\n\n")
    shell_path.chmod(0o755)

    payload = {
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "git_commit": _git_commit(),
        "n_configs": len(configs),
        "n_jobs": len(rows),
        "configs": list(configs),
        "jobs": list(rows),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "plan_csv": str(plan_csv),
        "script": str(shell_path),
        "manifest": str(manifest_path),
    }


def run_plan_job(row: Mapping[str, str], *, log_dir: str | Path | None = None) -> dict[str, Any]:
    """Run one planned command and capture its log."""

    job_id = row.get("job_id") or "job"
    log_root = _resolve_repo_path(log_dir or (Path(row.get("output_dir", "results")) / "logs"))
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{_slug(job_id)}.log"
    command = str(row.get("command", ""))

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        parts = shlex.split(command)
        result = subprocess.run(
            parts,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        output = result.stdout or ""
        status = "done" if result.returncode == 0 else "failed"
        returncode = int(result.returncode)
    except Exception as exc:
        output = f"{type(exc).__name__}: {exc}\n"
        status = "failed"
        returncode = 1

    finished_at = datetime.now(timezone.utc).isoformat()
    log_path.write_text(
        f"$ {command}\nstarted_at={started_at}\nfinished_at={finished_at}\nreturncode={returncode}\n\n{output}",
        encoding="utf-8",
    )
    return {
        "job_id": job_id,
        "status": status,
        "returncode": returncode,
        "log_path": str(log_path),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _read_run_manifest(run_dir: Path) -> dict[str, Any]:
    for candidate in [
        run_dir / "config" / "method_run_manifest.json",
        run_dir.parent / "config" / "method_run_manifest.json",
    ]:
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _run_dir_from_result_file(path: Path) -> Path:
    if path.parent.name == "results":
        return path.parent.parent
    return path.parent


def collect_result_rows(results_root: str | Path) -> Any:
    """Collect per-run result CSVs and failures under a result root."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to aggregate protocol results.") from exc

    root = _resolve_repo_path(results_root)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame()

    for path in sorted(root.rglob("*")):
        if path.name not in {"analysis_results.csv", "results.csv", "failure.json"}:
            continue
        run_dir = _run_dir_from_result_file(path)
        manifest = _read_run_manifest(run_dir)
        args = manifest.get("args", {}) if isinstance(manifest, Mapping) else {}
        base = {
            "run_dir": str(run_dir),
            "source_file": str(path),
            "method": manifest.get("method") or args.get("method") or run_dir.name,
            "dataset_key": args.get("dataset_key", ""),
            "seed": args.get("seed", ""),
        }
        if path.name == "failure.json":
            try:
                failure = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                failure = {}
            row = dict(base)
            row.update({
                "status": "failed",
                "error": failure.get("error", ""),
            })
            rows.append(row)
            continue

        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            row = dict(base)
            row.update({"status": "unreadable", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
            continue

        if frame.empty:
            row = dict(base)
            row["status"] = "empty"
            rows.append(row)
            continue

        for _, result_row in frame.iterrows():
            row = dict(base)
            row.update({str(key): value for key, value in result_row.to_dict().items()})
            row["status"] = "done"
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(results: Any) -> Any:
    """Return mean/std/count summary for common numeric metrics."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to summarize protocol results.") from exc

    if results is None or len(results) == 0:
        return pd.DataFrame()

    frame = results.copy()
    preferred = ["NMI", "ARI", "ACC", "Silhouette", "silhouette", "runtime", "runtime_seconds"]
    metric_cols = [col for col in preferred if col in frame.columns]
    if not metric_cols:
        metric_cols = [
            col
            for col in frame.select_dtypes(include="number").columns
            if col not in {"seed", "run_id", "n_clusters_found", "n_samples"}
        ]
    if not metric_cols:
        return pd.DataFrame()

    group_cols = [col for col in ["dataset_key", "method"] if col in frame.columns]
    if not group_cols:
        group_cols = ["method"] if "method" in frame.columns else []
    if not group_cols:
        return frame[metric_cols].agg(["mean", "std", "count"]).reset_index()

    summary = frame.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def protocol_to_yaml(spec: ProtocolSpec | Mapping[str, Any]) -> str:
    """Serialize a protocol spec for downloads or debugging."""

    raw = spec.raw if isinstance(spec, ProtocolSpec) else dict(spec)
    return _safe_dump_yaml(raw)
