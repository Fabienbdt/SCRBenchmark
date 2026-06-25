#!/usr/bin/env python3
"""Materialize stable_generalist datasets under SCRBenchmark/data.

Large .h5ad files are intentionally not tracked by git. This script makes the
project operational on a handoff machine by creating hardlinks, symlinks, or
copies from a local source directory to ``data/stable_generalist``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPO_ROOT / "reproducibility" / "stable_generalist"
DEFAULT_DATASET_TABLE = REFERENCE_ROOT / "stable_generalist_dataset_table.csv"
DEFAULT_SOURCE_ROOT = Path("/data2/fbidet/scRAW_EXPERIMENTAL/data")
DEFAULT_TARGET_ROOT = REPO_ROOT / "data" / "stable_generalist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-table", default=str(DEFAULT_DATASET_TABLE))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument("--mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _target_name(raw_path: Any) -> str:
    name = Path(str(raw_path)).name
    aliases = {
        "pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad",
    }
    return aliases.get(name, name)


def _find_source(raw_path: Any, source_root: Path) -> Path:
    original = Path(str(raw_path)).expanduser()
    candidates = [
        original,
        source_root / original.name,
        source_root / _target_name(original),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing source dataset for {original.name}: tried {candidates}")


def _same_file_or_size(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    try:
        if src.samefile(dst):
            return True
    except OSError:
        pass
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def _materialize(src: Path, dst: Path, *, mode: str, overwrite: bool, dry_run: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if _same_file_or_size(src, dst) and not overwrite:
            return "exists"
        if not overwrite:
            return "blocked_existing_different_file"
        if dry_run:
            return "would_overwrite"
        dst.unlink()

    if dry_run:
        return f"would_{mode}"

    if mode == "symlink":
        dst.symlink_to(src)
        return "symlinked"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"

    try:
        os.link(src, dst)
        return "hardlinked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied_after_hardlink_failed"


def main() -> int:
    args = parse_args()
    dataset_table = Path(args.dataset_table).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    frame = pd.read_csv(dataset_table)

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        dataset_key = str(row["dataset_key"]).strip()
        target = target_root / _target_name(row["data_file"])
        try:
            source = _find_source(row["data_file"], source_root)
            status = _materialize(
                source,
                target,
                mode=str(args.mode),
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
            )
            size_bytes = source.stat().st_size
        except Exception as exc:
            source = Path(str(row["data_file"]))
            status = f"failed:{type(exc).__name__}:{exc}"
            size_bytes = ""
        rows.append(
            {
                "dataset_key": dataset_key,
                "dataset": str(row["dataset"]).strip(),
                "source": str(source),
                "target": str(target),
                "mode": args.mode,
                "status": status,
                "size_bytes": size_bytes,
            }
        )
        print(f"{dataset_key}: {status} -> {target}", flush=True)

    manifest = target_root / "MANIFEST.csv"
    if not args.dry_run:
        target_root.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dataset_key", "dataset", "source", "target", "mode", "status", "size_bytes"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"manifest = {manifest}", flush=True)

    failed = [row for row in rows if str(row["status"]).startswith("failed")]
    blocked = [row for row in rows if str(row["status"]).startswith("blocked")]
    return 1 if failed or blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
