#!/usr/bin/env python3
"""Generalist multi-dataset Optuna search for scRAW."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .presets import PRESETS

logger = logging.getLogger("scraw_generalist_search")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_TOPK_CSV = (
    PROJECT_ROOT
    / "results"
    / "ultra_searches"
    / "baron_pancreas_ultra_search"
    / "stage2_refine_topk"
    / "summaries"
    / "aggregated_topk.csv"
)
DEFAULT_MANIFEST_CSV = PROJECT_ROOT / "config" / "generalist_search_manifest.csv"
DEFAULT_SEARCH_SPACE_JSON = PROJECT_ROOT / "config" / "generalist_narrow_search_space.json"
DEFAULT_STABLE_GENERALIST_RESULTS_CSV = (
    PROJECT_ROOT
    / "results"
    / "presentation_stable_generalist_nonbaron_20260324"
    / "00_source_tables"
    / "stable_generalist_all_results_table.csv"
)

DATASET_SCORE_WEIGHTS: Dict[str, float] = {
    "ARI": 0.20,
    "NMI": 0.17,
    "ACC": 0.15,
    "F1_Macro": 0.13,
    "BalancedACC": 0.10,
    "RareACC": 0.10,
    "UltraRareACC": 0.15,
}

RARE_FOCUS_SCORE_WEIGHTS: Dict[str, float] = {
    "ARI": 0.25,
    "BalancedACC": 0.20,
    "RareACC": 0.275,
    "UltraRareACC": 0.275,
}

SCORE_PROFILES: Dict[str, Dict[str, float]] = {
    "default": dict(DATASET_SCORE_WEIGHTS),
    "dominance_mix": dict(DATASET_SCORE_WEIGHTS),
    "rare_focus": dict(RARE_FOCUS_SCORE_WEIGHTS),
}

STABLE_GENERALIST_BASELINE_METHODS: Tuple[str, ...] = (
    "scNAME",
    "scMAE",
    "Harmony",
    "ComBat",
    "Scanorama",
    "DESC",
)

DEFAULT_SEARCH_OUTPUT_PROFILE = "search_minimal"
DEFAULT_RETENTION_OUTPUT_PROFILE = "standard"
ALLOWED_MANIFEST_ROLES = {"train", "validation", "holdout", "benchmark"}
DOMINANCE_MIX_EPS = 1e-12

FROZEN_PARAMS: Dict[str, Any] = {
    "use_batch_conditioning": True,
    "masking_apply_weighted": True,
    "pseudo_label_method": "leiden",
    "hdbscan_cluster_selection_method": "eom",
}

CONTINUOUS_PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "lr": {"type": "float", "low": 4.7e-4, "high": 2.4e-3, "log": True, "fixed_bounds": True},
    "nb_theta": {"type": "float", "low": 2.6, "high": 104.8, "log": True, "fixed_bounds": True},
    "rare_triplet_weight": {
        "type": "float",
        "low": 0.0226,
        "high": 0.2346,
        "log": True,
        "fixed_bounds": True,
    },
    "adversarial_batch_weight": {
        "type": "float",
        "low": 0.028,
        "high": 0.314,
        "log": True,
        "fixed_bounds": True,
    },
    "cluster_density_alpha": {
        "type": "float",
        "low": 0.2,
        "high": 0.6,
        "log": False,
        "fixed_bounds": True,
    },
    "dynamic_weight_momentum": {
        "type": "float",
        "low": 0.5,
        "high": 0.85,
        "log": False,
        "fixed_bounds": True,
    },
    "min_cell_weight": {
        "type": "float",
        "low": 0.29,
        "high": 0.505,
        "log": False,
        "fixed_bounds": True,
    },
}

DISCRETE_PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    "hidden_layers": {
        "choices": ["512,256,128", "512,256"],
        "force_include": ["512,256,128", "512,256"],
    },
    "z_dim": {"choices": [128, 192, 256], "force_include": [128, 192, 256]},
    "dropout": {"choices": [0.20, 0.25, 0.30], "force_include": [0.20, 0.25, 0.30]},
    "epochs": {"choices": list(range(80, 221, 10))},
    "warmup_epochs": {"choices": list(range(5, 121))},
    "batch_size": {"choices": [192, 384], "force_include": [192, 384]},
    "reconstruction_distribution": {"choices": ["nb", "mse"]},
    "nb_input_transform": {"choices": ["log1p", "pearson_residuals"]},
    "masking_rate": {"choices": [0.10, 0.15, 0.20], "force_include": [0.10, 0.15, 0.20]},
    "masked_recon_weight": {"choices": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]},
    "weight_exponent": {"choices": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]},
    "density_knn_k": {"choices": [15], "force_include": [15]},
    "density_weight_clip": {"choices": [3.0, 8.0], "force_include": [3.0, 8.0]},
    "dynamic_weight_update_interval": {"choices": [10, 20], "force_include": [10, 20]},
    "max_cell_weight": {"choices": [5.0, 8.0, 10.0, 15.0, 20.0]},
    "weight_fusion_mode": {"choices": ["additive", "multiplicative"]},
    "rare_triplet_start_epoch": {"choices": list(range(0, 221))},
    "rare_triplet_margin": {"choices": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
    "rare_triplet_min_weight": {"choices": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]},
    "max_triplet_anchors_per_batch": {"choices": [64, 128], "force_include": [64, 128]},
    "hdbscan_min_cluster_size": {"choices": list(range(2, 21))},
    "hdbscan_min_samples": {"choices": list(range(1, 11))},
    "hdbscan_reassign_noise": {"choices": [False, True]},
    "adversarial_lambda": {"choices": [0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50]},
    "adversarial_start_epoch": {"choices": list(range(0, 61, 5))},
    "adversarial_ramp_epochs": {"choices": list(range(0, 61, 5))},
    "mmd_batch_weight": {"choices": [0.0, 0.02, 0.05, 0.10]},
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    path: Path
    label_key: str
    batch_key: str
    family: str
    role: str
    enabled: bool
    n_labels_expected: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            key_s = str(key)
            if key_s not in seen:
                seen.add(key_s)
                fieldnames.append(key_s)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonable(v) for k, v in row.items()})


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(out):
        return float("nan")
    return float(out)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _as_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _resolve_score_weights(profile_name: str) -> Dict[str, float]:
    profile_key = str(profile_name).strip().lower()
    if profile_key not in SCORE_PROFILES:
        raise ValueError(f"Unsupported score profile: {profile_name}")
    weights = {
        str(key): float(value)
        for key, value in SCORE_PROFILES[profile_key].items()
        if float(value) > 0.0
    }
    if not weights:
        raise ValueError(f"Score profile '{profile_name}' produced no positive weights.")
    return weights


def _seed_sequence(base_seed: int, n_seeds: int, seed_step: int) -> List[int]:
    count = max(1, int(n_seeds))
    step = max(1, int(seed_step))
    start = int(base_seed)
    return [start + i * step for i in range(count)]


def _build_env(output_root: Path) -> Dict[str, str]:
    env = os.environ.copy()
    src_dir = Path(__file__).resolve().parents[1]
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src_dir) if not prev else f"{src_dir}:{prev}"

    cache_root = output_root / ".cache"
    numba_dir = cache_root / "numba"
    mpl_dir = cache_root / "mpl"
    xdg_dir = cache_root / "xdg_cache"
    for path in (numba_dir, mpl_dir, xdg_dir):
        path.mkdir(parents=True, exist_ok=True)
    env.setdefault("NUMBA_CACHE_DIR", str(numba_dir))
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    env.setdefault("XDG_CACHE_HOME", str(xdg_dir))
    env.setdefault("NUMBA_DISABLE_JIT", "1")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return env


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    seen: set[str] = set()
    out: List[Any] = []
    for value in values:
        key = json.dumps(_jsonable(value), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _canonicalize_choice(value: Any, choices: Sequence[Any]) -> Any:
    for choice in choices:
        if isinstance(choice, bool):
            if bool(value) is bool(choice):
                return choice
            continue
        if isinstance(choice, (int, float)) and not isinstance(choice, bool):
            try:
                if math.isclose(float(value), float(choice), rel_tol=1e-9, abs_tol=1e-9):
                    return choice
            except Exception:
                pass
            continue
        if value == choice:
            return choice
    return value


def _choice_in_choices(value: Any, choices: Sequence[Any]) -> bool:
    canonical = _canonicalize_choice(value, choices)
    for choice in choices:
        if isinstance(choice, (int, float)) and not isinstance(choice, bool):
            try:
                if math.isclose(float(canonical), float(choice), rel_tol=1e-9, abs_tol=1e-9):
                    return True
            except Exception:
                continue
        elif canonical == choice:
            return True
    return False


def _clip_to_float_bounds(value: Any, spec: Mapping[str, Any]) -> Any:
    try:
        clipped = float(value)
    except Exception:
        return value
    low = _safe_float(spec.get("low"))
    high = _safe_float(spec.get("high"))
    if np.isfinite(low):
        clipped = max(clipped, float(low))
    if np.isfinite(high):
        clipped = min(clipped, float(high))
    return clipped


def _normalize_bool(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_manifest(path: Path) -> List[DatasetSpec]:
    rows = _read_csv(path)
    if not rows:
        raise FileNotFoundError(f"Dataset manifest is missing or empty: {path}")

    out: List[DatasetSpec] = []
    seen_ids: set[str] = set()
    for row in rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id:
            raise ValueError(f"Missing dataset_id in manifest row: {row}")
        if dataset_id in seen_ids:
            raise ValueError(f"Duplicate dataset_id in manifest: {dataset_id}")
        seen_ids.add(dataset_id)
        spec = DatasetSpec(
            dataset_id=dataset_id,
            path=Path(str(row.get("path", "")).strip()).expanduser().resolve(),
            label_key=str(row.get("label_key", "")).strip(),
            batch_key=str(row.get("batch_key", "")).strip(),
            family=str(row.get("family", "")).strip(),
            role=str(row.get("role", "")).strip().lower(),
            enabled=_normalize_bool(row.get("enabled", "true")),
            n_labels_expected=max(1, int(row.get("n_labels_expected", "1"))),
        )
        if not spec.path.exists():
            raise FileNotFoundError(f"Manifest dataset not found: {spec.path}")
        if spec.role not in ALLOWED_MANIFEST_ROLES:
            raise ValueError(f"Unsupported role '{spec.role}' for dataset {dataset_id}")
        if not spec.label_key or not spec.batch_key:
            raise ValueError(f"Dataset {dataset_id} is missing label_key or batch_key")
        out.append(spec)

    enabled = [spec for spec in out if spec.enabled]
    if not enabled:
        raise ValueError(f"No enabled datasets found in manifest: {path}")
    return enabled


def _filter_by_role(specs: Sequence[DatasetSpec], role: str) -> List[DatasetSpec]:
    return [spec for spec in specs if spec.role == role]


def _ranked_overrides_from_csv(source_csv: Path, top_k: int) -> List[Dict[str, Any]]:
    rows = _read_csv(source_csv)
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        score = _safe_float(row.get("score_mean"))
        overrides_raw = row.get("overrides_json")
        if np.isnan(score) or not overrides_raw:
            continue
        try:
            overrides = json.loads(str(overrides_raw))
        except Exception:
            continue
        if not isinstance(overrides, dict):
            continue
        cleaned.append(
            {
                "candidate_rank": _safe_int(row.get("candidate_rank")) or 0,
                "candidate_id": str(row.get("candidate_id", "")).strip(),
                "trial": _safe_int(row.get("trial")) or -1,
                "score_mean": float(score),
                "overrides": overrides,
            }
        )
    cleaned.sort(
        key=lambda item: (
            -float(item["score_mean"]),
            int(item["candidate_rank"]),
            int(item["trial"]),
        )
    )
    return cleaned[: max(1, int(top_k))]


def _snap_low(value: float, low: float, high: float, step: float) -> float:
    snapped = low + math.ceil((float(value) - low) / step) * step
    return float(min(max(snapped, low), high))


def _snap_high(value: float, low: float, high: float, step: float) -> float:
    snapped = low + math.floor((float(value) - low) / step) * step
    return float(min(max(snapped, low), high))


def _build_discrete_choices(
    param_name: str,
    overrides_list: Sequence[Mapping[str, Any]],
    top1: Mapping[str, Any],
) -> List[Any]:
    spec = DISCRETE_PARAM_SPECS[param_name]
    base_choices = list(spec["choices"])
    force_include = list(spec.get("force_include", []))
    counts: Dict[str, int] = {}
    value_by_key: Dict[str, Any] = {}
    for overrides in overrides_list:
        if param_name not in overrides:
            continue
        value = overrides[param_name]
        key = json.dumps(_jsonable(value), sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
        value_by_key[key] = value

    selected: List[Any] = []
    for choice in base_choices:
        key = json.dumps(_jsonable(choice), sort_keys=True)
        if counts.get(key, 0) >= 2:
            selected.append(choice)

    if param_name in top1:
        selected.append(top1[param_name])
    selected.extend(force_include)
    selected = _ordered_unique(selected)

    allowed: List[Any] = []
    for choice in base_choices:
        key = json.dumps(_jsonable(choice), sort_keys=True)
        if any(key == json.dumps(_jsonable(candidate), sort_keys=True) for candidate in selected):
            allowed.append(choice)
    if not allowed:
        allowed = [top1[param_name]] if param_name in top1 else []
    return _ordered_unique(allowed)


def _build_continuous_spec(
    param_name: str,
    overrides_list: Sequence[Mapping[str, Any]],
    top1: Mapping[str, Any],
) -> Dict[str, Any]:
    spec = dict(CONTINUOUS_PARAM_SPECS[param_name])
    if bool(spec.pop("fixed_bounds", False)):
        return spec
    values = [float(overrides[param_name]) for overrides in overrides_list if param_name in overrides]
    if not values:
        return spec
    arr = np.asarray(values, dtype=float)
    top1_value = float(top1[param_name]) if param_name in top1 else float(arr[0])
    q10 = float(np.quantile(arr, 0.10))
    q90 = float(np.quantile(arr, 0.90))
    low = max(float(spec["low"]), min(top1_value, q10))
    high = min(float(spec["high"]), max(top1_value, q90))
    step = spec.get("step")
    if step is not None:
        low = _snap_low(low, float(spec["low"]), float(spec["high"]), float(step))
        high = _snap_high(high, float(spec["low"]), float(spec["high"]), float(step))
        if high < low:
            high = low
    spec["low"] = low
    spec["high"] = high
    return spec


def build_narrow_search_space(
    *,
    source_csv: Path,
    output_json: Optional[Path] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    ranked = _ranked_overrides_from_csv(source_csv, top_k=top_k)
    if not ranked:
        raise FileNotFoundError(f"No ranked overrides found in {source_csv}")

    overrides_list = [dict(item["overrides"]) for item in ranked]
    top1 = overrides_list[0]

    sampled_params: Dict[str, Dict[str, Any]] = {
        "final_clustering_requested": {
            "type": "categorical",
            "choices": ["hdbscan", "leiden"],
        }
    }
    for param_name in DISCRETE_PARAM_SPECS:
        if param_name in FROZEN_PARAMS:
            continue
        allowed = _build_discrete_choices(param_name, overrides_list, top1)
        if not allowed:
            continue
        sampled_params[param_name] = {"type": "categorical", "choices": allowed}

    for param_name in CONTINUOUS_PARAM_SPECS:
        if param_name in FROZEN_PARAMS:
            continue
        sampled_params[param_name] = _build_continuous_spec(param_name, overrides_list, top1)

    warm_starts: List[Dict[str, Any]] = []
    sampled_names = set(sampled_params.keys())
    for item in ranked[:5]:
        overrides = dict(item["overrides"])
        sampled: Dict[str, Any] = {}
        for key in sampled_names:
            if key not in overrides:
                continue
            raw_value = overrides[key]
            spec = sampled_params.get(key, {})
            if str(spec.get("type", "categorical")) == "categorical":
                choices = list(spec.get("choices", []))
                raw_value = _canonicalize_choice(raw_value, choices)
                if not _choice_in_choices(raw_value, choices):
                    continue
            else:
                raw_value = _clip_to_float_bounds(raw_value, spec)
            sampled[key] = raw_value
        sampled["final_clustering_requested"] = _canonicalize_choice(
            "hdbscan",
            list(sampled_params["final_clustering_requested"]["choices"]),
        )
        warm_starts.append(
            {
                "candidate_rank": int(item["candidate_rank"]),
                "candidate_id": str(item["candidate_id"]),
                "trial": int(item["trial"]),
                "score_mean": float(item["score_mean"]),
                "params": sampled,
            }
        )

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "source_csv": str(source_csv),
        "top_k_considered": int(top_k),
        "score_weights": DATASET_SCORE_WEIGHTS,
        "frozen_params": dict(FROZEN_PARAMS),
        "sampled_params": sampled_params,
        "warm_start_candidates": warm_starts,
        "selection_rules": {
            "discrete": "keep values seen at least twice in top-k plus top1 and force_include",
            "continuous": "use fixed bounds derived from historical top runs, otherwise clip to [min(top1, q10), max(top1, q90)]",
        },
        "final_clustering_mode": "mixed",
    }
    if output_json is not None:
        _write_json(output_json, payload)
    return payload


def _sample_config(trial: Any, search_space: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    sampled_params = dict(search_space.get("sampled_params", {}))
    frozen_params = dict(search_space.get("frozen_params", {}))

    overrides: Dict[str, Any] = {}
    final_clustering_requested = "hdbscan"
    for param_name, spec in sampled_params.items():
        spec_type = str(spec.get("type", "categorical"))
        if spec_type == "categorical":
            value = trial.suggest_categorical(param_name, list(spec.get("choices", [])))
        else:
            low = float(spec["low"])
            high = float(spec["high"])
            log = bool(spec.get("log", False))
            step = spec.get("step")
            if step is None:
                value = trial.suggest_float(param_name, low, high, log=log)
            else:
                value = trial.suggest_float(param_name, low, high, step=float(step), log=log)
        if param_name == "final_clustering_requested":
            final_clustering_requested = str(value)
        else:
            overrides[param_name] = value

    overrides.update(frozen_params)

    hdbscan_min_cluster_size = _safe_int(overrides.get("hdbscan_min_cluster_size"))
    hdbscan_min_samples = _safe_int(overrides.get("hdbscan_min_samples"))
    if hdbscan_min_cluster_size is not None and hdbscan_min_samples is not None:
        overrides["hdbscan_min_samples"] = int(min(hdbscan_min_samples, hdbscan_min_cluster_size))

    return final_clustering_requested, overrides


def _build_cli_cmd(
    *,
    python_bin: str,
    preset: str,
    dataset: DatasetSpec,
    output_dir: Path,
    device: str,
    seed: int,
    overrides: Mapping[str, Any],
    compute_scib_metrics: bool,
    scib_n_jobs: int,
    output_profile: str = DEFAULT_SEARCH_OUTPUT_PROFILE,
    metrics_only: Optional[bool] = None,
    capture_snapshots: str = "off",
    save_processed_data: str = "off",
    auto_hparams: str = "off",
) -> List[str]:
    output_profile_norm = str(output_profile or DEFAULT_SEARCH_OUTPUT_PROFILE).strip().lower()
    metrics_only_enabled = bool(metrics_only) if metrics_only is not None else output_profile_norm == "search_minimal"
    cmd: List[str] = [
        python_bin,
        "-m",
        "scraw_dedicated.cli",
        "--preset",
        preset,
        "--data",
        str(dataset.path),
        "--output",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
        "--output-profile",
        output_profile_norm,
        "--compute-scib-metrics",
        "on" if compute_scib_metrics else "off",
        "--scib-n-jobs",
        str(max(1, int(scib_n_jobs))),
        "--capture-snapshots",
        str(capture_snapshots),
        "--save-processed-data",
        str(save_processed_data),
        "--auto-hparams",
        str(auto_hparams),
        "--dann",
        "auto",
        "--batch-key",
        str(dataset.batch_key),
        "--leiden-target-clusters",
        str(int(dataset.n_labels_expected)),
    ]
    if metrics_only_enabled:
        cmd.append("--metrics-only")
    for key, value in sorted(overrides.items()):
        if key == "batch_correction_key":
            continue
        cmd.extend(["--param", f"{key}={_as_cli_value(value)}"])
    cmd.extend(["--param", f"batch_correction_key={dataset.batch_key}"])
    return cmd


def _read_analysis_metrics(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "results" / "analysis_results.csv")
    if not rows:
        return {}
    row = rows[0]
    return {
        "runtime": _safe_float(row.get("runtime")),
        "ACC": _safe_float(row.get("ACC")),
        "Batch correction": _safe_float(row.get("Batch correction")),
        "Inter cell-type conservation": _safe_float(row.get("Inter cell-type conservation")),
        "Intra cell-type conservation": _safe_float(row.get("Intra cell-type conservation")),
        "scIB-E Total score": _safe_float(row.get("scIB-E Total score")),
    }


def _read_final_clustering_table(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows = _read_csv(run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv")
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip()
        if not method:
            continue
        out[method] = {
            "ARI": _safe_float(row.get("ARI")),
            "NMI": _safe_float(row.get("NMI")),
            "ACC": _safe_float(row.get("ACC")),
            "F1_Macro": _safe_float(row.get("F1_Macro")),
            "BalancedACC": _safe_float(row.get("BalancedACC")),
            "RareACC": _safe_float(row.get("RareACC")),
            "UltraRareACC": _safe_float(row.get("UltraRareACC")),
            "n_clusters_found": _safe_float(row.get("n_clusters_found")),
            "resolution": _safe_float(row.get("resolution")),
        }
    return out


def _selected_method(requested: str, target_clusters: int) -> str:
    requested_norm = str(requested).strip().lower()
    if requested_norm in {"best_of_both", "both_max", "oracle"}:
        return "best_of_both"
    if requested_norm == "leiden":
        return f"leiden_target{int(target_clusters)}_final"
    return "hdbscan_final"


def _dataset_score(metrics: Mapping[str, Any], *, score_weights: Optional[Mapping[str, float]] = None) -> float:
    total = 0.0
    for key, weight in (score_weights or DATASET_SCORE_WEIGHTS).items():
        value = _safe_float(metrics.get(key))
        if np.isnan(value):
            value = 0.0
        total += float(weight) * float(value)
    return float(total)


def _row_dataset_id(row: Mapping[str, Any]) -> str:
    return str(row.get("dataset_id") or row.get("dataset_key") or "").strip()


def _row_dataset_name(row: Mapping[str, Any]) -> str:
    return str(row.get("dataset") or row.get("dataset_name") or _row_dataset_id(row)).strip()


def _row_method_name(row: Mapping[str, Any]) -> str:
    return str(row.get("method") or row.get("candidate_id") or "").strip()


def _is_dominance_profile(profile_name: str) -> bool:
    return str(profile_name).strip().lower() == "dominance_mix"


def _score_from_row(
    row: Mapping[str, Any],
    *,
    score_weights: Optional[Mapping[str, float]] = None,
) -> float:
    score = _safe_float(row.get("score"))
    if np.isfinite(score):
        return float(score)
    return _dataset_score(row, score_weights=score_weights)


def _best_non_scraw_reference(
    results_csv: Path,
    *,
    baseline_methods: Sequence[str] = STABLE_GENERALIST_BASELINE_METHODS,
    score_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    rows = _read_csv(results_csv)
    if not rows:
        raise FileNotFoundError(f"Trial206 results table is missing or empty: {results_csv}")

    baseline_method_set = {str(name).strip() for name in baseline_methods if str(name).strip()}
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        method = _row_method_name(row)
        dataset_id = _row_dataset_id(row)
        if not dataset_id or method not in baseline_method_set:
            continue
        score = _dataset_score(row, score_weights=score_weights)
        current = out.get(dataset_id)
        if current is None or float(score) > float(current["best_non_scraw_score"]) + DOMINANCE_MIX_EPS:
            out[dataset_id] = {
                "dataset_id": dataset_id,
                "dataset": _row_dataset_name(row),
                "best_non_scraw_method": method,
                "best_non_scraw_score": float(score),
                "result_row_id": str(row.get("result_row_id", "")).strip(),
            }
    if not out:
        raise ValueError(f"No non-scRAW baseline rows found in {results_csv}")
    return out


def _mean_by_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        dataset_id = _row_dataset_id(row)
        if not dataset_id:
            continue
        bucket = grouped.setdefault(
            dataset_id,
            {
                "dataset_id": dataset_id,
                "dataset": _row_dataset_name(row),
                "scores": [],
                "rows": [],
                "used_methods": set(),
                "seeds": set(),
            },
        )
        bucket["scores"].append(_score_from_row(row, score_weights=score_weights))
        bucket["rows"].append(dict(row))
        used_method = str(row.get("used_method", "")).strip()
        if used_method:
            bucket["used_methods"].add(used_method)
        seed = _safe_int(row.get("seed"))
        if seed is not None:
            bucket["seeds"].add(int(seed))

    out: List[Dict[str, Any]] = []
    for dataset_id, payload in grouped.items():
        scores = [float(value) for value in payload["scores"] if np.isfinite(float(value))]
        dataset_score = float(np.mean(scores)) if scores else 0.0
        out.append(
            {
                "dataset_id": dataset_id,
                "dataset": str(payload["dataset"]),
                "score": dataset_score,
                "n_rows": int(len(payload["rows"])),
                "n_seeds": int(len(payload["seeds"])) if payload["seeds"] else 0,
                "used_methods": sorted(str(name) for name in payload["used_methods"]),
                "rows": list(payload["rows"]),
            }
        )
    out.sort(key=lambda item: str(item["dataset_id"]))
    return out


def _dominance_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    best_non_scraw_by_dataset: Optional[Mapping[str, Mapping[str, Any]]],
    score_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    per_dataset = _mean_by_dataset(rows, score_weights=score_weights)
    if not per_dataset:
        return {
            "aggregate_score": 0.0,
            "dominance_mix": 0.0,
            "mean_core": 0.0,
            "mean_gap": 0.0,
            "worst_gap": 0.0,
            "win_rate": 0.0,
            "n_wins": 0,
            "n_datasets": 0,
            "missing_reference_count": 0,
            "per_dataset": [],
        }

    ref = dict(best_non_scraw_by_dataset or {})
    scores: List[float] = []
    gaps: List[float] = []
    n_wins = 0
    missing_reference_count = 0
    resolved_rows: List[Dict[str, Any]] = []
    for item in per_dataset:
        dataset_id = str(item["dataset_id"])
        score = float(item["score"])
        scores.append(score)
        baseline = ref.get(dataset_id)
        baseline_score = _safe_float(baseline.get("best_non_scraw_score")) if baseline else float("nan")
        if not np.isfinite(baseline_score):
            baseline_score = 0.0
            missing_reference_count += 1
        gap = float(score - float(baseline_score))
        win = gap >= -DOMINANCE_MIX_EPS
        if win:
            n_wins += 1
        resolved = dict(item)
        resolved.update(
            {
                "best_non_scraw_method": (
                    str(baseline.get("best_non_scraw_method", "")).strip() if baseline else ""
                ),
                "best_non_scraw_score": float(baseline_score),
                "gap_to_best_non_scraw": gap,
                "win_vs_best_non_scraw": int(bool(win)),
            }
        )
        resolved_rows.append(resolved)
        gaps.append(gap)

    mean_core = float(np.mean(scores)) if scores else 0.0
    mean_gap = float(np.mean(gaps)) if gaps else 0.0
    worst_gap = float(min(gaps)) if gaps else 0.0
    win_rate = float(n_wins) / float(len(resolved_rows)) if resolved_rows else 0.0
    dominance_mix = float(mean_core + 0.30 * mean_gap + 0.15 * worst_gap + 0.05 * win_rate)
    return {
        "aggregate_score": dominance_mix,
        "dominance_mix": dominance_mix,
        "mean_core": mean_core,
        "mean_gap": mean_gap,
        "worst_gap": worst_gap,
        "win_rate": win_rate,
        "n_wins": int(n_wins),
        "n_datasets": int(len(resolved_rows)),
        "missing_reference_count": int(missing_reference_count),
        "per_dataset": resolved_rows,
    }


def _dominance_sort_key(summary: Mapping[str, Any]) -> Tuple[float, float, float, str]:
    return (
        _safe_float(summary.get("dominance_mix")),
        _safe_float(summary.get("mean_core")),
        _safe_float(summary.get("win_rate")),
        str(summary.get("method") or summary.get("candidate_id") or ""),
    )


def _summaries_from_stable_generalist_table(
    results_csv: Path,
    *,
    score_weights: Optional[Mapping[str, float]] = None,
    best_non_scraw_by_dataset: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows = _read_csv(results_csv)
    if not rows:
        raise FileNotFoundError(f"Trial206 results table is missing or empty: {results_csv}")

    ref = dict(best_non_scraw_by_dataset or _best_non_scraw_reference(results_csv, score_weights=score_weights))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        method = _row_method_name(row)
        if not method:
            continue
        grouped.setdefault(method, []).append(dict(row))

    summaries: List[Dict[str, Any]] = []
    for method, method_rows in grouped.items():
        summary = _dominance_summary(
            method_rows,
            best_non_scraw_by_dataset=ref,
            score_weights=score_weights,
        )
        summary.update(
            {
                "method": method,
                "source": "stable_generalist_reference",
                "is_scraw_method": int(
                    any(str(row.get("is_scraw_method", "")).strip().lower() == "true" for row in method_rows)
                ),
            }
        )
        summaries.append(summary)
    summaries.sort(key=_dominance_sort_key, reverse=True)
    return summaries


def _promotion_decision(
    *,
    champion_summary: Mapping[str, Any],
    leaderboard_rows: Sequence[Mapping[str, Any]],
    retention_ready: bool,
    min_win_count: int = 8,
) -> Dict[str, Any]:
    champion_name = str(champion_summary.get("method") or champion_summary.get("candidate_id") or "").strip()
    ordered = sorted((dict(row) for row in leaderboard_rows), key=_dominance_sort_key, reverse=True)
    dominance_rank = 1
    for idx, row in enumerate(ordered, start=1):
        row_name = str(row.get("method") or row.get("candidate_id") or "").strip()
        if row_name == champion_name:
            dominance_rank = idx
            break

    max_mean_core = max((_safe_float(row.get("mean_core")) for row in ordered), default=0.0)
    champion_mean_core = _safe_float(champion_summary.get("mean_core"))
    champion_wins = _safe_int(champion_summary.get("n_wins")) or 0

    reasons: List[str] = []
    if dominance_rank != 1:
        reasons.append("candidate is not rank 1 in dominance_mix")
    if champion_mean_core + DOMINANCE_MIX_EPS < max_mean_core:
        reasons.append("candidate is not rank 1 in mean_core")
    if champion_wins < int(min_win_count):
        reasons.append(f"candidate win count {champion_wins} is below {int(min_win_count)}")
    if not retention_ready:
        reasons.append("retention rerun did not produce the expected embedding artifacts")

    return {
        "promote": len(reasons) == 0,
        "reasons": reasons,
        "dominance_rank": int(dominance_rank),
        "mean_core_rank_eligible": champion_mean_core + DOMINANCE_MIX_EPS >= max_mean_core,
        "required_min_wins": int(min_win_count),
        "candidate_wins": int(champion_wins),
    }


def _metric_sort_value(value: Any) -> float:
    metric = _safe_float(value)
    if np.isnan(metric):
        return float("-inf")
    return float(metric)


def _cluster_distance_to_ground_truth(metrics: Mapping[str, Any], target_clusters: int) -> float:
    n_clusters_found = _safe_float(metrics.get("n_clusters_found"))
    if np.isnan(n_clusters_found):
        return float("inf")
    return abs(float(n_clusters_found) - float(target_clusters))


def _best_clustering_sort_key(
    method_name: str,
    metrics: Mapping[str, Any],
    *,
    target_clusters: int,
    score_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[float, float, float, float, float, int]:
    return (
        _dataset_score(metrics, score_weights=score_weights),
        _metric_sort_value(metrics.get("ARI")),
        _metric_sort_value(metrics.get("RareACC")),
        _metric_sort_value(metrics.get("UltraRareACC")),
        -_cluster_distance_to_ground_truth(metrics, target_clusters),
        1 if str(method_name).startswith("leiden_target") else 0,
    )


def _choose_best_clustering_metrics(
    *,
    cluster_rows: Mapping[str, Mapping[str, Any]],
    target_clusters: int,
    score_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, Any], str]:
    leiden_method = f"leiden_target{int(target_clusters)}_final"
    candidates: List[Tuple[str, Mapping[str, Any]]] = []
    if "hdbscan_final" in cluster_rows:
        candidates.append(("hdbscan_final", cluster_rows["hdbscan_final"]))
    if leiden_method in cluster_rows:
        candidates.append((leiden_method, cluster_rows[leiden_method]))
    if not candidates:
        return {}, "best_of_both"

    used_method, chosen_metrics = max(
        candidates,
        key=lambda item: _best_clustering_sort_key(
            item[0],
            item[1],
            target_clusters=target_clusters,
            score_weights=score_weights,
        ),
    )
    metrics = dict(chosen_metrics)
    metrics["selection_score"] = _dataset_score(metrics, score_weights=score_weights)
    metrics["cluster_distance_to_ground_truth"] = _cluster_distance_to_ground_truth(
        metrics,
        target_clusters=int(target_clusters),
    )
    return metrics, str(used_method)


def _select_final_metrics(
    *,
    cluster_rows: Mapping[str, Mapping[str, Any]],
    requested: str,
    target_clusters: int,
    score_weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, Any], str, bool]:
    requested_norm = str(requested).strip().lower()
    selected_method = _selected_method(requested_norm, target_clusters)
    if requested_norm in {"best_of_both", "both_max", "oracle"}:
        metrics, used_method = _choose_best_clustering_metrics(
            cluster_rows=cluster_rows,
            target_clusters=int(target_clusters),
            score_weights=score_weights,
        )
        metrics["used_method"] = used_method
        return metrics, used_method, False

    chosen = cluster_rows.get(selected_method)
    used_method = selected_method
    fallback = False
    if chosen is None and selected_method != "hdbscan_final":
        chosen = cluster_rows.get("hdbscan_final")
        used_method = "hdbscan_final"
        fallback = chosen is not None
    if chosen is None:
        return {}, used_method, fallback
    metrics = dict(chosen)
    metrics["selection_score"] = _dataset_score(metrics, score_weights=score_weights)
    metrics["cluster_distance_to_ground_truth"] = _cluster_distance_to_ground_truth(
        metrics,
        target_clusters=int(target_clusters),
    )
    return metrics, used_method, fallback


def _aggregate_scores(dataset_scores: Sequence[float]) -> Dict[str, float]:
    finite = [float(score) for score in dataset_scores if np.isfinite(float(score))]
    if not finite:
        return {
            "aggregate_score": 0.0,
            "mean_score": 0.0,
            "bottom_quartile_mean": 0.0,
            "n_scores": 0,
        }
    ordered = sorted(finite)
    worst_count = max(1, int(math.ceil(len(ordered) / 4.0)))
    mean_score = float(np.mean(ordered))
    bottom_mean = float(np.mean(ordered[:worst_count]))
    aggregate = float(0.8 * mean_score + 0.2 * bottom_mean)
    return {
        "aggregate_score": aggregate,
        "mean_score": mean_score,
        "bottom_quartile_mean": bottom_mean,
        "n_scores": int(len(ordered)),
    }


def _run_dataset_eval(
    *,
    dataset: DatasetSpec,
    preset: str,
    python_bin: str,
    device: str,
    seed: int,
    overrides: Mapping[str, Any],
    requested_method: str,
    run_dir: Path,
    log_file: Path,
    env: Mapping[str, str],
    skip_existing: bool,
    dry_run: bool,
    score_weights: Optional[Mapping[str, float]],
    compute_scib_metrics: bool,
    scib_n_jobs: int,
    output_profile: str = DEFAULT_SEARCH_OUTPUT_PROFILE,
    metrics_only: Optional[bool] = None,
    capture_snapshots: str = "off",
    save_processed_data: str = "off",
    auto_hparams: str = "off",
) -> Dict[str, Any]:
    cmd = _build_cli_cmd(
        python_bin=python_bin,
        preset=preset,
        dataset=dataset,
        output_dir=run_dir,
        device=device,
        seed=seed,
        overrides=overrides,
        compute_scib_metrics=compute_scib_metrics,
        scib_n_jobs=scib_n_jobs,
        output_profile=output_profile,
        metrics_only=metrics_only,
        capture_snapshots=capture_snapshots,
        save_processed_data=save_processed_data,
        auto_hparams=auto_hparams,
    )
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    analysis_csv = run_dir / "results" / "analysis_results.csv"
    clustering_csv = run_dir / "results" / "clustering_final" / "final_clustering_comparison.csv"

    status = "ok"
    if dry_run:
        status = "dry_run"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(cmd_str + "\n", encoding="utf-8")
    elif skip_existing and analysis_csv.exists() and clustering_csv.exists():
        status = "existing"
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(env))
        if int(proc.returncode) != 0:
            status = f"failed_{int(proc.returncode)}"

    analysis_metrics = _read_analysis_metrics(run_dir)
    cluster_rows = _read_final_clustering_table(run_dir)
    metrics, used_method, fallback = _select_final_metrics(
        cluster_rows=cluster_rows,
        requested=requested_method,
        target_clusters=int(dataset.n_labels_expected),
        score_weights=score_weights,
    )
    if not np.isfinite(_safe_float(metrics.get("ACC"))):
        metrics["ACC"] = analysis_metrics.get("ACC")
    metrics["runtime"] = analysis_metrics.get("runtime", float("nan"))
    metrics["Batch correction"] = analysis_metrics.get("Batch correction", float("nan"))
    metrics["Inter cell-type conservation"] = analysis_metrics.get(
        "Inter cell-type conservation",
        float("nan"),
    )
    metrics["Intra cell-type conservation"] = analysis_metrics.get(
        "Intra cell-type conservation",
        float("nan"),
    )
    metrics["scIB-E Total score"] = analysis_metrics.get("scIB-E Total score", float("nan"))

    score = _dataset_score(metrics, score_weights=score_weights)
    if status.startswith("failed") or status == "dry_run" or not metrics:
        score = 0.0

    return {
        "dataset_id": dataset.dataset_id,
        "family": dataset.family,
        "role": dataset.role,
        "seed": int(seed),
        "status": status,
        "requested_method": str(requested_method),
        "used_method": str(used_method),
        "fallback_to_hdbscan": int(bool(fallback)),
        "score": float(score),
        "selection_score": _safe_float(metrics.get("selection_score")),
        "cluster_distance_to_ground_truth": _safe_float(metrics.get("cluster_distance_to_ground_truth")),
        "ARI": _safe_float(metrics.get("ARI")),
        "NMI": _safe_float(metrics.get("NMI")),
        "ACC": _safe_float(metrics.get("ACC")),
        "F1_Macro": _safe_float(metrics.get("F1_Macro")),
        "BalancedACC": _safe_float(metrics.get("BalancedACC")),
        "RareACC": _safe_float(metrics.get("RareACC")),
        "UltraRareACC": _safe_float(metrics.get("UltraRareACC")),
        "Batch correction": _safe_float(metrics.get("Batch correction")),
        "Inter cell-type conservation": _safe_float(metrics.get("Inter cell-type conservation")),
        "Intra cell-type conservation": _safe_float(metrics.get("Intra cell-type conservation")),
        "scIB-E Total score": _safe_float(metrics.get("scIB-E Total score")),
        "n_clusters_found": _safe_float(metrics.get("n_clusters_found")),
        "resolution": _safe_float(metrics.get("resolution")),
        "runtime": _safe_float(metrics.get("runtime")),
        "run_dir": str(run_dir),
        "log_file": str(log_file),
        "command": cmd_str,
    }


def _summarize_split_rows(
    rows: Sequence[Mapping[str, Any]],
    split_name: str,
    *,
    score_profile: str = "default",
    best_non_scraw_by_dataset: Optional[Mapping[str, Mapping[str, Any]]] = None,
    score_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    if _is_dominance_profile(score_profile):
        aggregate = _dominance_summary(
            rows,
            best_non_scraw_by_dataset=best_non_scraw_by_dataset,
            score_weights=score_weights,
        )
        summary = {
            "split": split_name,
            "aggregate_score": aggregate["dominance_mix"],
            "mean_score": aggregate["mean_core"],
            "bottom_quartile_mean": aggregate["worst_gap"],
            "n_scores": aggregate["n_datasets"],
            "mean_gap": aggregate["mean_gap"],
            "worst_gap": aggregate["worst_gap"],
            "win_rate": aggregate["win_rate"],
            "n_wins": aggregate["n_wins"],
            "n_rows": int(len(rows)),
            "missing_reference_count": aggregate["missing_reference_count"],
            "n_failed": int(sum(1 for row in rows if str(row.get("status", "")).startswith("failed"))),
        }
        return summary

    scores = [_safe_float(row.get("score")) for row in rows]
    aggregate = _aggregate_scores(scores)
    n_failed = sum(1 for row in rows if str(row.get("status", "")).startswith("failed"))
    return {
        "split": split_name,
        "aggregate_score": aggregate["aggregate_score"],
        "mean_score": aggregate["mean_score"],
        "bottom_quartile_mean": aggregate["bottom_quartile_mean"],
        "n_scores": aggregate["n_scores"],
        "n_failed": int(n_failed),
        "n_rows": int(len(rows)),
    }


def _stage2_candidate_summary(
    *,
    candidate_id: str,
    trial_number: int,
    config_params: Mapping[str, Any],
    requested_method: str,
    stage1_score: float,
    rows: Sequence[Mapping[str, Any]],
    score_profile: str = "default",
    best_non_scraw_by_dataset: Optional[Mapping[str, Mapping[str, Any]]] = None,
    score_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    train_rows = [row for row in rows if str(row.get("split")) == "train"]
    val_rows = [row for row in rows if str(row.get("split")) == "validation"]
    all_active_rows = [row for row in rows if str(row.get("split")) == "all_active"]

    if not train_rows and all_active_rows:
        train_rows = list(all_active_rows)
    if not val_rows and all_active_rows:
        val_rows = list(all_active_rows)

    train_by_seed: List[float] = []
    for seed in sorted({int(row["seed"]) for row in train_rows}):
        seed_rows = [row for row in train_rows if int(row["seed"]) == seed]
        train_by_seed.append(
            _summarize_split_rows(
                seed_rows,
                "train",
                score_profile=score_profile,
                best_non_scraw_by_dataset=best_non_scraw_by_dataset,
                score_weights=score_weights,
            )["aggregate_score"]
        )

    val_by_seed: List[float] = []
    for seed in sorted({int(row["seed"]) for row in val_rows}):
        seed_rows = [row for row in val_rows if int(row["seed"]) == seed]
        val_by_seed.append(
            _summarize_split_rows(
                seed_rows,
                "validation",
                score_profile=score_profile,
                best_non_scraw_by_dataset=best_non_scraw_by_dataset,
                score_weights=score_weights,
            )["aggregate_score"]
        )

    return {
        "candidate_id": candidate_id,
        "trial": int(trial_number),
        "requested_method": requested_method,
        "stage1_train_aggregate": float(stage1_score),
        "train_aggregate_mean": float(np.mean(train_by_seed)) if train_by_seed else float("nan"),
        "train_aggregate_std": float(np.std(train_by_seed)) if train_by_seed else float("nan"),
        "validation_aggregate_mean": float(np.mean(val_by_seed)) if val_by_seed else float("nan"),
        "validation_aggregate_std": float(np.std(val_by_seed)) if val_by_seed else float("nan"),
        "n_seeds": int(len(set(int(row["seed"]) for row in rows))) if rows else 0,
        "params_json": json.dumps(_jsonable(dict(config_params)), sort_keys=True),
    }


def _evaluate_candidate(
    *,
    candidate_id: str,
    trial_number: int,
    config_params: Mapping[str, Any],
    requested_method: str,
    datasets_by_split: Mapping[str, Sequence[DatasetSpec]],
    seeds: Sequence[int],
    output_root: Path,
    preset: str,
    python_bin: str,
    device: str,
    env: Mapping[str, str],
    skip_existing: bool,
    dry_run: bool,
    score_weights: Optional[Mapping[str, float]],
    compute_scib_metrics: bool,
    scib_n_jobs: int,
    output_profile: str = DEFAULT_SEARCH_OUTPUT_PROFILE,
    metrics_only: Optional[bool] = None,
    capture_snapshots: str = "off",
    save_processed_data: str = "off",
    auto_hparams: str = "off",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split_name, datasets in datasets_by_split.items():
        for dataset in datasets:
            for seed in seeds:
                run_dir = output_root / "runs" / candidate_id / split_name / dataset.dataset_id / f"seed_{int(seed)}"
                log_file = output_root / "logs" / candidate_id / split_name / f"{dataset.dataset_id}_seed_{int(seed)}.log"
                row = _run_dataset_eval(
                    dataset=dataset,
                    preset=preset,
                    python_bin=python_bin,
                    device=device,
                    seed=int(seed),
                    overrides=config_params,
                    requested_method=requested_method,
                    run_dir=run_dir,
                    log_file=log_file,
                    env=env,
                    skip_existing=skip_existing,
                    dry_run=dry_run,
                    score_weights=score_weights,
                    compute_scib_metrics=compute_scib_metrics,
                    scib_n_jobs=scib_n_jobs,
                    output_profile=output_profile,
                    metrics_only=metrics_only,
                    capture_snapshots=capture_snapshots,
                    save_processed_data=save_processed_data,
                    auto_hparams=auto_hparams,
                )
                row.update(
                    {
                        "candidate_id": candidate_id,
                        "trial": int(trial_number),
                        "split": split_name,
                    }
                )
                rows.append(row)
    return rows


def _validate_retained_artifacts(run_dir: Path) -> Dict[str, Any]:
    embedding_path = run_dir / "results" / "embeddings" / "embeddings_scraw_run0.npy"
    config_used_path = run_dir / "config" / "config_used.json"
    algo_hparams_path = run_dir / "config" / "algorithm_hyperparams_used.json"
    results_json_path = run_dir / "results" / "results.json"
    required_paths = {
        "embedding_path": embedding_path,
        "config_used_path": config_used_path,
        "algorithm_hyperparams_path": algo_hparams_path,
        "results_json_path": results_json_path,
    }
    missing = [label for label, path in required_paths.items() if not path.exists()]
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "embedding_path": str(embedding_path),
        "config_used_path": str(config_used_path),
        "algorithm_hyperparams_path": str(algo_hparams_path),
        "results_json_path": str(results_json_path),
    }


def _selected_final_method_from_results(run_dir: Path) -> str:
    results_json = run_dir / "results" / "results.json"
    if not results_json.exists():
        return ""
    try:
        payload = _read_json(results_json)
    except Exception:
        return ""
    results = payload.get("results", [])
    if not results:
        return ""
    result0 = results[0] if isinstance(results, list) else {}
    if not isinstance(result0, dict):
        return ""
    final_info = result0.get("final_clustering_info", {})
    if isinstance(final_info, dict):
        used_method = str(final_info.get("used_method") or final_info.get("selected_method") or "").strip()
        if used_method:
            return used_method
    return ""


def _write_embedding_bundle(
    *,
    dataset: DatasetSpec,
    run_dir: Path,
    seed: int,
    preset_name: str,
    trial_id_or_candidate_id: str,
    selected_final_method: str,
) -> Dict[str, Any]:
    validation = _validate_retained_artifacts(run_dir)
    if not validation["ok"]:
        missing = ", ".join(str(name) for name in validation["missing"])
        raise FileNotFoundError(f"Retention artifacts missing for {run_dir}: {missing}")

    effective_method = str(selected_final_method).strip() or _selected_final_method_from_results(run_dir)
    payload = {
        "dataset_id": dataset.dataset_id,
        "dataset_path": str(dataset.path),
        "label_key": dataset.label_key,
        "batch_key": dataset.batch_key,
        "seed": int(seed),
        "preset_name": str(preset_name),
        "trial_id_or_candidate_id": str(trial_id_or_candidate_id),
        "selected_final_method": str(effective_method),
        "embedding_path": validation["embedding_path"],
        "config_used_path": validation["config_used_path"],
        "algorithm_hyperparams_used_path": validation["algorithm_hyperparams_path"],
        "results_json_path": validation["results_json_path"],
        "timestamp": datetime.now().isoformat(),
    }
    bundle_path = run_dir / "embedding_bundle.json"
    _write_json(bundle_path, payload)
    payload["embedding_bundle_path"] = str(bundle_path)
    return payload


def _best_stage2_candidates(summaries: Sequence[Mapping[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    ordered = sorted(
        (dict(item) for item in summaries),
        key=lambda row: (
            -_safe_float(row.get("validation_aggregate_mean")),
            -_safe_float(row.get("train_aggregate_mean")),
            -_safe_float(row.get("stage1_train_aggregate")),
            str(row.get("candidate_id")),
        ),
    )
    return ordered[: max(1, int(top_k))]


def _study_sampler(args: argparse.Namespace, optuna_mod: Any) -> Any:
    if args.sampler == "random":
        return optuna_mod.samplers.RandomSampler(seed=int(args.seed))
    return optuna_mod.samplers.TPESampler(
        seed=int(args.seed),
        multivariate=True,
        group=True,
        constant_liar=False,
        n_startup_trials=max(5, int(args.n_startup_trials)),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generalist multi-dataset Optuna search for scRAW.")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST_CSV), help="Dataset manifest CSV")
    p.add_argument("--search-space-json", default=str(DEFAULT_SEARCH_SPACE_JSON), help="Narrow search space JSON")
    p.add_argument("--source-topk-csv", default=str(DEFAULT_SOURCE_TOPK_CSV), help="Stage-2 CSV used to derive the narrow search space")
    p.add_argument("--output-root", required=True, help="Output root for the full generalist search")
    p.add_argument("--preset", default="default", choices=sorted(PRESETS.keys()))
    p.add_argument("--device", default="auto")
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument("--seed", type=int, default=60)
    p.add_argument("--seed-step", type=int, default=97)
    p.add_argument("--n-trials-stage1", type=int, default=80)
    p.add_argument("--n-startup-trials", type=int, default=16)
    p.add_argument(
        "--final-clustering-mode",
        choices=["mixed", "best_of_both", "hdbscan", "leiden"],
        default="mixed",
        help="How to select the final clustering result for scoring on each dataset.",
    )
    p.add_argument(
        "--score-profile",
        choices=sorted(SCORE_PROFILES.keys()),
        default="default",
        help="Dataset-level scoring profile used for trial ranking and best-of-both selection.",
    )
    p.add_argument(
        "--stable_generalist-results-csv",
        default=str(DEFAULT_STABLE_GENERALIST_RESULTS_CSV),
        help="Reference Stable Generalist results table used when score-profile=dominance_mix.",
    )
    p.add_argument("--stage2-top-k", type=int, default=6)
    p.add_argument("--n-seeds-stage2", type=int, default=3)
    p.add_argument("--stage3-top-k", type=int, default=3)
    p.add_argument("--n-seeds-stage3", type=int, default=1)
    p.add_argument(
        "--compute-scib-metrics",
        action="store_true",
        help="Compute scIB-E metrics during the primary run.",
    )
    p.add_argument("--scib-n-jobs", type=int, default=1, help="Worker count for scIB metric computation.")
    p.add_argument("--stage1-only", action="store_true", help="Stop after writing stage1 outputs.")
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--verbose", action="store_true")
    return p


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    manifest_path = Path(args.manifest).expanduser().resolve()
    search_space_path = Path(args.search_space_json).expanduser().resolve()
    source_topk_csv = Path(args.source_topk_csv).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    stage1_root = output_root / "stage1"
    stage2_root = output_root / "stage2"
    stage3_root = output_root / "stage3"
    config_root = output_root / "config"
    summaries_root = output_root / "summaries"
    for path in (stage1_root, stage2_root, stage3_root, config_root, summaries_root):
        path.mkdir(parents=True, exist_ok=True)

    specs = _load_manifest(manifest_path)
    train_specs = _filter_by_role(specs, "train")
    validation_specs = _filter_by_role(specs, "validation")
    holdout_specs = _filter_by_role(specs, "holdout")
    if not train_specs or not validation_specs or not holdout_specs:
        raise ValueError("Manifest must contain enabled train, validation, and holdout datasets.")

    resolved_manifest_rows = [
        {
            "dataset_id": spec.dataset_id,
            "path": str(spec.path),
            "label_key": spec.label_key,
            "batch_key": spec.batch_key,
            "family": spec.family,
            "role": spec.role,
            "enabled": spec.enabled,
            "n_labels_expected": spec.n_labels_expected,
        }
        for spec in specs
    ]
    _write_rows_csv(config_root / "resolved_manifest.csv", resolved_manifest_rows)

    if not search_space_path.exists():
        build_narrow_search_space(
            source_csv=source_topk_csv,
            output_json=search_space_path,
            top_k=10,
        )
    search_space = json.loads(search_space_path.read_text(encoding="utf-8"))
    score_weights = _resolve_score_weights(args.score_profile)
    stable_generalist_results_csv = Path(args.stable_generalist_results_csv).expanduser().resolve()
    best_non_scraw_by_dataset = (
        _best_non_scraw_reference(stable_generalist_results_csv, score_weights=score_weights)
        if _is_dominance_profile(args.score_profile)
        else None
    )
    sampled_params = dict(search_space.get("sampled_params", {}))
    if args.final_clustering_mode != "mixed":
        sampled_params["final_clustering_requested"] = {
            "type": "categorical",
            "choices": [str(args.final_clustering_mode)],
        }
    search_space["sampled_params"] = sampled_params
    search_space["final_clustering_mode"] = str(args.final_clustering_mode)
    search_space["score_profile"] = str(args.score_profile)
    search_space["score_weights"] = dict(score_weights)
    warm_start_candidates = list(search_space.get("warm_start_candidates", []))
    if args.final_clustering_mode != "mixed":
        for candidate in warm_start_candidates:
            params = dict(candidate.get("params", {}))
            params["final_clustering_requested"] = str(args.final_clustering_mode)
            candidate["params"] = params
        search_space["warm_start_candidates"] = warm_start_candidates
    _write_json(config_root / "generalist_narrow_search_space.json", search_space)

    search_manifest = {
        "timestamp": datetime.now().isoformat(),
        "manifest_csv": str(manifest_path),
        "search_space_json": str(search_space_path),
        "source_topk_csv": str(source_topk_csv),
        "preset": args.preset,
        "device": args.device,
        "python_bin": args.python_bin,
        "seed": int(args.seed),
        "seed_step": int(args.seed_step),
        "n_trials_stage1": int(args.n_trials_stage1),
        "final_clustering_mode": str(args.final_clustering_mode),
        "score_profile": str(args.score_profile),
        "score_weights": dict(score_weights),
        "stable_generalist_results_csv": str(stable_generalist_results_csv),
        "stage2_top_k": int(args.stage2_top_k),
        "n_seeds_stage2": int(args.n_seeds_stage2),
        "stage3_top_k": int(args.stage3_top_k),
        "n_seeds_stage3": int(args.n_seeds_stage3),
        "roles": {
            "train": [spec.dataset_id for spec in train_specs],
            "validation": [spec.dataset_id for spec in validation_specs],
            "holdout": [spec.dataset_id for spec in holdout_specs],
        },
        "compute_scib_metrics": bool(args.compute_scib_metrics),
        "scib_n_jobs": int(args.scib_n_jobs),
        "stage1_only": bool(args.stage1_only),
        "dry_run": bool(args.dry_run),
        "skip_existing": bool(args.skip_existing),
    }
    if best_non_scraw_by_dataset is not None:
        search_manifest["best_non_scraw_reference"] = list(best_non_scraw_by_dataset.values())
    _write_json(config_root / "search_manifest.json", search_manifest)

    env = _build_env(output_root)

    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("optuna is required for generalist search.") from exc

    study_name = f"scraw_generalist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_db = stage1_root / "optuna_study.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{study_db}",
        sampler=_study_sampler(args, optuna),
        direction="maximize",
        load_if_exists=True,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    stage1_dataset_rows: List[Dict[str, Any]] = []
    stage1_aggregate_rows: List[Dict[str, Any]] = []

    for candidate in warm_start_candidates:
        params = dict(candidate.get("params", {}))
        if params:
            study.enqueue_trial(params)

    def objective(trial: Any) -> float:
        requested_method, overrides = _sample_config(trial, search_space)
        trial_id = f"trial_{int(trial.number):04d}"
        trial_root = stage1_root / "trials" / trial_id
        trial_root.mkdir(parents=True, exist_ok=True)
        _write_json(
            trial_root / "trial_config.json",
            {
                "trial_number": int(trial.number),
                "requested_method": requested_method,
                "param_overrides": overrides,
            },
        )

        rows: List[Dict[str, Any]] = []
        for dataset in train_specs:
            seed = int(args.seed)
            run_dir = trial_root / dataset.dataset_id / f"seed_{seed}"
            log_file = stage1_root / "logs" / trial_id / f"{dataset.dataset_id}_seed_{seed}.log"
            row = _run_dataset_eval(
                dataset=dataset,
                preset=args.preset,
                python_bin=args.python_bin,
                device=args.device,
                seed=seed,
                overrides=overrides,
                requested_method=requested_method,
                run_dir=run_dir,
                log_file=log_file,
                env=env,
                skip_existing=bool(args.skip_existing),
                dry_run=bool(args.dry_run),
                score_weights=score_weights,
                compute_scib_metrics=bool(args.compute_scib_metrics),
                scib_n_jobs=int(args.scib_n_jobs),
            )
            row.update(
                {
                    "stage": "stage1",
                    "trial": int(trial.number),
                    "candidate_id": trial_id,
                    "split": "train",
                }
            )
            rows.append(row)

        failure_count = sum(1 for row in rows if str(row.get("status", "")).startswith("failed"))
        summary = _summarize_split_rows(
            rows,
            "train",
            score_profile=args.score_profile,
            best_non_scraw_by_dataset=best_non_scraw_by_dataset,
            score_weights=score_weights,
        )
        aggregate_row = {
            "stage": "stage1",
            "trial": int(trial.number),
            "candidate_id": trial_id,
            "requested_method": requested_method,
            "aggregate_score": summary["aggregate_score"],
            "mean_score": summary["mean_score"],
            "bottom_quartile_mean": summary["bottom_quartile_mean"],
            "mean_gap": _safe_float(summary.get("mean_gap")),
            "worst_gap": _safe_float(summary.get("worst_gap")),
            "win_rate": _safe_float(summary.get("win_rate")),
            "n_wins": _safe_int(summary.get("n_wins")) or 0,
            "n_failed": summary["n_failed"],
            "n_scores": summary["n_scores"],
            "params_json": json.dumps(_jsonable(overrides), sort_keys=True),
        }

        stage1_dataset_rows.extend(rows)
        stage1_aggregate_rows.append(aggregate_row)
        trial.set_user_attr("requested_method", requested_method)
        trial.set_user_attr("param_overrides", overrides)
        trial.set_user_attr("train_aggregate", summary["aggregate_score"])

        if failure_count > max(0, int(math.floor(len(train_specs) * 0.25))):
            raise optuna.exceptions.TrialPruned(f"Too many dataset failures ({failure_count})")
        return float(summary["aggregate_score"])

    study.optimize(
        objective,
        n_trials=max(1, int(args.n_trials_stage1)),
        timeout=args.timeout,
        gc_after_trial=True,
    )

    _write_rows_csv(stage1_root / "summaries" / "trial_dataset_metrics.csv", stage1_dataset_rows)
    _write_rows_csv(stage1_root / "summaries" / "trial_aggregate_scores.csv", stage1_aggregate_rows)

    study_rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        study_rows.append(
            {
                "trial": int(trial.number),
                "state": str(trial.state.name),
                "value": _safe_float(trial.value),
                "requested_method": trial.user_attrs.get("requested_method"),
                "train_aggregate": _safe_float(trial.user_attrs.get("train_aggregate")),
                "params_json": json.dumps(_jsonable(dict(trial.params)), sort_keys=True),
                "overrides_json": json.dumps(_jsonable(trial.user_attrs.get("param_overrides", {})), sort_keys=True),
            }
        )
    _write_rows_csv(stage1_root / "summaries" / "study_trials.csv", study_rows)

    complete_trials = [trial for trial in study.trials if str(trial.state.name) == "COMPLETE" and trial.value is not None]
    complete_trials.sort(key=lambda trial: (-float(trial.value), int(trial.number)))
    if not complete_trials:
        raise RuntimeError("No completed stage-1 trials were produced.")

    if args.stage1_only:
        best_trial = complete_trials[0]
        best_stage1_payload = {
            "timestamp": datetime.now().isoformat(),
            "trial": int(best_trial.number),
            "requested_method": best_trial.user_attrs.get("requested_method"),
            "stage1_train_aggregate": float(best_trial.value),
            "params": dict(best_trial.user_attrs.get("param_overrides", {})),
            "manifest_csv": str(manifest_path),
            "search_space_json": str(search_space_path),
            "final_clustering_mode": str(args.final_clustering_mode),
            "score_profile": str(args.score_profile),
            "score_weights": dict(score_weights),
            "compute_scib_metrics": bool(args.compute_scib_metrics),
            "scib_n_jobs": int(args.scib_n_jobs),
        }
        _write_json(output_root / "best_stage1_config.json", best_stage1_payload)
        _write_json(
            stage1_root / "summaries" / "stage1_summary.json",
            {
                "timestamp": datetime.now().isoformat(),
                "n_trials_total": len(study.trials),
                "n_trials_complete": len(complete_trials),
                "best_trial": best_stage1_payload,
                "final_clustering_mode": str(args.final_clustering_mode),
                "score_profile": str(args.score_profile),
                "score_weights": dict(score_weights),
                "compute_scib_metrics": bool(args.compute_scib_metrics),
                "scib_n_jobs": int(args.scib_n_jobs),
            },
        )
        logger.info("Stage1-only generalist search completed: %s", output_root)
        return 0

    stage2_candidates = complete_trials[: max(1, int(args.stage2_top_k))]
    stage2_rows: List[Dict[str, Any]] = []
    stage2_summaries: List[Dict[str, Any]] = []
    stage2_seeds = _seed_sequence(args.seed, args.n_seeds_stage2, args.seed_step)
    for rank, trial in enumerate(stage2_candidates, start=1):
        candidate_id = f"rank_{rank:03d}_trial_{int(trial.number):04d}"
        requested_method = str(trial.user_attrs.get("requested_method", "hdbscan"))
        overrides = dict(trial.user_attrs.get("param_overrides", {}))
        rows = _evaluate_candidate(
            candidate_id=candidate_id,
            trial_number=int(trial.number),
            config_params=overrides,
            requested_method=requested_method,
            datasets_by_split={"train": train_specs, "validation": validation_specs},
            seeds=stage2_seeds,
            output_root=stage2_root,
            preset=args.preset,
            python_bin=args.python_bin,
            device=args.device,
            env=env,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            score_weights=score_weights,
            compute_scib_metrics=bool(args.compute_scib_metrics),
            scib_n_jobs=int(args.scib_n_jobs),
        )
        stage2_rows.extend(rows)
        stage2_summaries.append(
            _stage2_candidate_summary(
                candidate_id=candidate_id,
                trial_number=int(trial.number),
                config_params=overrides,
                requested_method=requested_method,
                stage1_score=float(trial.value),
                rows=rows,
                score_profile=args.score_profile,
                best_non_scraw_by_dataset=best_non_scraw_by_dataset,
                score_weights=score_weights,
            )
        )

    _write_rows_csv(stage2_root / "summaries" / "stage2_seed_dataset_metrics.csv", stage2_rows)
    _write_rows_csv(stage2_root / "summaries" / "stage2_refine_summary.csv", stage2_summaries)

    best_stage2 = _best_stage2_candidates(stage2_summaries, top_k=max(1, int(args.stage3_top_k)))
    selected_best = best_stage2[0]
    selected_params = json.loads(str(selected_best["params_json"]))

    stage3_rows: List[Dict[str, Any]] = []
    stage3_summaries: List[Dict[str, Any]] = []
    stage3_seeds = _seed_sequence(args.seed, args.n_seeds_stage3, args.seed_step)
    for candidate in best_stage2:
        params = json.loads(str(candidate["params_json"]))
        rows = _evaluate_candidate(
            candidate_id=str(candidate["candidate_id"]),
            trial_number=int(candidate["trial"]),
            config_params=params,
            requested_method=str(candidate["requested_method"]),
            datasets_by_split={"holdout": holdout_specs},
            seeds=stage3_seeds,
            output_root=stage3_root,
            preset=args.preset,
            python_bin=args.python_bin,
            device=args.device,
            env=env,
            skip_existing=bool(args.skip_existing),
            dry_run=bool(args.dry_run),
            score_weights=score_weights,
            compute_scib_metrics=bool(args.compute_scib_metrics),
            scib_n_jobs=int(args.scib_n_jobs),
        )
        stage3_rows.extend(rows)
        summary = _summarize_split_rows(
            rows,
            "holdout",
            score_profile=args.score_profile,
            best_non_scraw_by_dataset=best_non_scraw_by_dataset,
            score_weights=score_weights,
        )
        stage3_summaries.append(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "trial": int(candidate["trial"]),
                "requested_method": str(candidate["requested_method"]),
                "holdout_aggregate": summary["aggregate_score"],
                "holdout_mean_score": summary["mean_score"],
                "holdout_bottom_quartile_mean": summary["bottom_quartile_mean"],
                "holdout_mean_gap": _safe_float(summary.get("mean_gap")),
                "holdout_worst_gap": _safe_float(summary.get("worst_gap")),
                "holdout_win_rate": _safe_float(summary.get("win_rate")),
                "holdout_n_wins": _safe_int(summary.get("n_wins")) or 0,
                "params_json": json.dumps(_jsonable(params), sort_keys=True),
            }
        )

    _write_rows_csv(stage3_root / "summaries" / "holdout_dataset_metrics.csv", stage3_rows)
    _write_rows_csv(stage3_root / "summaries" / "holdout_report.csv", stage3_summaries)

    best_payload = {
        "timestamp": datetime.now().isoformat(),
        "candidate_id": selected_best["candidate_id"],
        "trial": int(selected_best["trial"]),
        "requested_method": selected_best["requested_method"],
        "stage1_train_aggregate": _safe_float(selected_best["stage1_train_aggregate"]),
        "stage2_train_aggregate_mean": _safe_float(selected_best["train_aggregate_mean"]),
        "stage2_validation_aggregate_mean": _safe_float(selected_best["validation_aggregate_mean"]),
        "params": selected_params,
        "manifest_csv": str(manifest_path),
        "search_space_json": str(search_space_path),
        "final_clustering_mode": str(args.final_clustering_mode),
        "score_profile": str(args.score_profile),
        "score_weights": dict(score_weights),
        "compute_scib_metrics": bool(args.compute_scib_metrics),
        "scib_n_jobs": int(args.scib_n_jobs),
    }
    _write_json(output_root / "best_generalist_config.json", best_payload)

    logger.info("Generalist search completed: %s", output_root)
    return 0


def build_search_space_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the narrow generalist scRAW search space JSON.")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_TOPK_CSV))
    parser.add_argument("--output-json", default=str(DEFAULT_SEARCH_SPACE_JSON))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    build_narrow_search_space(
        source_csv=Path(args.source_csv).expanduser().resolve(),
        output_json=Path(args.output_json).expanduser().resolve(),
        top_k=int(args.top_k),
    )
    return 0
