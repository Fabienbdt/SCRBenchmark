#!/usr/bin/env python3
"""Small helpers shared by reproduction launchers."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def reproduction_env(extra_paths: Sequence[Path] = ()) -> dict[str, str]:
    env = os.environ.copy()
    paths = [
        REPO_ROOT / "vendor" / "scraw_dedicated" / "src",
        REPO_ROOT / "vendor" / "scraw_inductive" / "src",
        REPO_ROOT / "src",
        REPO_ROOT / "src" / "scrbenchmark",
        REPO_ROOT / "external" / "original_code" / "desc",
        REPO_ROOT / "external" / "original_code" / "aide",
        *extra_paths,
    ]
    if env.get("PYTHONPATH"):
        paths.append(Path(env["PYTHONPATH"]))
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in paths)
    env["SCRBENCHMARK_ROOT"] = str(REPO_ROOT)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-scrbenchmark-repro")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-scrbenchmark-repro")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_logged(cmd: Sequence[str], log_path: Path, *, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(list(cmd), stdout=log, stderr=subprocess.STDOUT, env=dict(env))
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}; log={log_path}")


def write_manifest(
    path: Path,
    *,
    dataset_key: str,
    data_path: Path,
    label_key: str,
    batch_key: str,
    n_labels: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["enabled", "dataset_id", "path", "label_key", "batch_key", "family", "n_labels_expected"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "enabled": "true",
                "dataset_id": dataset_key,
                "path": str(data_path),
                "label_key": label_key,
                "batch_key": batch_key,
                "family": "stable_generalist",
                "n_labels_expected": int(n_labels),
            }
        )


def expose_output(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    pointer = {
        "source_output_dir": str(source_dir.resolve()),
        "note": "Heavy artifacts remain in the vendored runner output; top-level entries are links when possible.",
    }
    (output_dir / "config" / "external_runner.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    for name in ["results", "figures", "logs", "models"]:
        src = source_dir / name
        dst = output_dir / name
        if not src.exists():
            continue
        if dst.exists() and src.is_dir() and dst.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            continue
        if dst.exists():
            continue
        try:
            dst.symlink_to(src, target_is_directory=src.is_dir())
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def write_failure(path: Path, *, method: str, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"method": method, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
