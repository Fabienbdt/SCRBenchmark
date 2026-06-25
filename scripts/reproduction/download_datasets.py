#!/usr/bin/env python3
"""Download and verify the 13 stable_generalist H5AD datasets.

The benchmark uses exact preprocessed ``.h5ad`` files. Rebuilding them from raw
public archives can change filtering, labels, or AnnData metadata, so this
script verifies the prepared files with the SHA256 hashes recorded in
``data/stable_generalist/download_manifest.csv``.

Usage examples:

  # Download from a hosted folder/release that contains the 13 H5AD files.
  python scripts/reproduction/download_datasets.py \
      --base-url https://YOUR_HOST/scrbenchmark/stable_generalist/

  # Handoff/local machine fallback: materialize from an existing local source.
  python scripts/reproduction/download_datasets.py \
      --source-root /data2/fbidet/scRAW_EXPERIMENTAL/data

  # Verify already prepared files only.
  python scripts/reproduction/download_datasets.py --verify-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import sys
import urllib.parse
import urllib.request
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "stable_generalist" / "download_manifest.csv"
DEFAULT_DATASET_TABLE = (
    REPO_ROOT / "reproducibility" / "stable_generalist" / "stable_generalist_dataset_table.csv"
)
DEFAULT_REFERENCE_TABLE = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "presentation_stable_generalist_nonbaron_20260324/"
    "00_source_tables/stable_generalist_dataset_table.csv"
)
DEFAULT_TARGET_ROOT = REPO_ROOT / "data" / "stable_generalist"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--dataset-table", default=str(DEFAULT_DATASET_TABLE))
    parser.add_argument("--reference-table", default=str(DEFAULT_REFERENCE_TABLE))
    parser.add_argument("--target-root", default=str(DEFAULT_TARGET_ROOT))
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL containing the 13 H5AD files. Used when source_url is "
            "empty in the manifest. Example: https://YOUR_HOST/path/stable_generalist/"
        ),
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Optional local directory containing the exact H5AD files.",
    )
    parser.add_argument(
        "--local-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to materialize files when --source-root is used.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--skip-h5ad-check",
        action="store_true",
        help="Skip AnnData shape/obs-column checks; SHA256 is still verified.",
    )
    parser.add_argument(
        "--skip-reference-table-check",
        action="store_true",
        help="Do not compare --dataset-table with --reference-table when the reference exists.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference_table(dataset_table: Path, reference_table: Path) -> None:
    if not reference_table.exists():
        print(f"reference_table: skipped_missing -> {reference_table}", flush=True)
        return
    dataset_rows = read_csv_rows(dataset_table)
    reference_rows = read_csv_rows(reference_table)
    if dataset_rows != reference_rows:
        raise RuntimeError(
            f"Dataset table differs from reference table:\n"
            f"  dataset_table={dataset_table}\n"
            f"  reference_table={reference_table}"
        )
    print(f"reference_table: identical -> {reference_table}", flush=True)


def verify_manifest_against_table(manifest_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    table_by_key = {row["dataset_key"]: row for row in table_rows}
    errors: list[str] = []
    for row in manifest_rows:
        key = row["dataset_key"]
        table = table_by_key.get(key)
        if table is None:
            errors.append(f"{key}: missing from dataset table")
            continue
        if row["filename"] != target_name(table.get("data_file", "")):
            errors.append(
                f"{key}: manifest filename={row['filename']!r} "
                f"!= table data_file={table.get('data_file')!r}"
            )
        checks = {
            "n_labels": "n_labels",
            "label_key": "label_key",
            "batch_key": "dann_batch_column",
        }
        for manifest_col, table_col in checks.items():
            if str(row.get(manifest_col, "")).strip() != str(table.get(table_col, "")).strip():
                errors.append(
                    f"{key}: manifest {manifest_col}={row.get(manifest_col)!r} "
                    f"!= table {table_col}={table.get(table_col)!r}"
                )
    if errors:
        raise RuntimeError("Manifest/table mismatch:\n  " + "\n  ".join(errors))
    print("manifest: matches dataset table metadata", flush=True)


def target_name(raw_path: Any) -> str:
    name = Path(str(raw_path)).name
    aliases = {
        "pancreas_raw_counts.h5ad": "pancreas_raw_counts_no_smarter.h5ad",
    }
    return aliases.get(name, name)


def _url_for(row: dict[str, str], base_url: str | None) -> str | None:
    source_url = (row.get("source_url") or "").strip()
    if source_url:
        return source_url
    if not base_url:
        return None
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", row["filename"])


def _download(url: str, target: Path, dry_run: bool) -> str:
    if dry_run:
        return f"would_download:{url}"

    tmp = target.with_suffix(target.suffix + ".part")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    tmp.replace(target)
    return "downloaded"


def _materialize_local(source: Path, target: Path, mode: str, dry_run: bool) -> str:
    if dry_run:
        return f"would_{mode}:{source}"
    if mode == "symlink":
        target.symlink_to(source)
        return "symlinked"
    if mode == "copy":
        shutil.copy2(source, target)
        return "copied"
    try:
        os.link(source, target)
        return "hardlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied_after_hardlink_failed"


def _prepare_one(
    row: dict[str, str],
    *,
    target_root: Path,
    base_url: str | None,
    source_root: Path | None,
    local_mode: str,
    overwrite: bool,
    dry_run: bool,
    verify_only: bool,
    skip_h5ad_check: bool,
) -> dict[str, Any]:
    target = target_root / row["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)

    action = "exists"
    if not target.exists() and verify_only:
        raise FileNotFoundError(f"{row['dataset_key']}: missing target {target}")

    if not target.exists() or overwrite:
        if target.exists() and overwrite and not dry_run:
            target.unlink()
        if verify_only:
            action = "verify_only"
        else:
            url = _url_for(row, base_url)
            if url:
                action = _download(url, target, dry_run)
            elif source_root is not None:
                source = source_root / row["filename"]
                if not source.exists():
                    raise FileNotFoundError(f"{row['dataset_key']}: missing local source {source}")
                action = _materialize_local(source.resolve(), target, local_mode, dry_run)
            else:
                raise RuntimeError(
                    f"{row['dataset_key']}: no source_url/base_url and no --source-root. "
                    "Host the exact H5AD files and pass --base-url, or pass --source-root."
                )

    if dry_run:
        return {
            "dataset_key": row["dataset_key"],
            "filename": row["filename"],
            "target": str(target),
            "status": action,
            "sha256": "",
            "size_bytes": "",
        }

    verify_file(row, target, skip_h5ad_check=skip_h5ad_check)
    return {
        "dataset_key": row["dataset_key"],
        "filename": row["filename"],
        "target": str(target),
        "status": action,
        "sha256": row["sha256"],
        "size_bytes": row["size_bytes"],
    }


def verify_file(row: dict[str, str], target: Path, *, skip_h5ad_check: bool) -> None:
    expected_size = int(row["size_bytes"])
    actual_size = target.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"{row['dataset_key']}: size mismatch for {target}: "
            f"expected {expected_size}, got {actual_size}"
        )

    actual_hash = sha256_file(target)
    if actual_hash != row["sha256"]:
        raise RuntimeError(
            f"{row['dataset_key']}: sha256 mismatch for {target}: "
            f"expected {row['sha256']}, got {actual_hash}"
        )

    if not skip_h5ad_check:
        verify_h5ad_metadata(row, target)


def verify_h5ad_metadata(row: dict[str, str], target: Path) -> None:
    try:
        import anndata as ad
    except ImportError:
        print("h5ad_check: skipped_missing_anndata", flush=True)
        return

    adata = ad.read_h5ad(target, backed="r")
    try:
        expected_shape = (int(row["h5ad_n_obs"]), int(row["h5ad_n_vars"]))
        if tuple(adata.shape) != expected_shape:
            raise RuntimeError(
                f"{row['dataset_key']}: AnnData shape mismatch for {target}: "
                f"expected {expected_shape}, got {tuple(adata.shape)}"
            )
        for key_name, col in (("label_key", row["label_key"]), ("batch_key", row["batch_key"])):
            if col and col not in adata.obs.columns:
                raise RuntimeError(
                    f"{row['dataset_key']}: missing {key_name} column {col!r} in {target}"
                )
    finally:
        adata.file.close()


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_key", "filename", "target", "status", "sha256", "size_bytes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest = Path(args.manifest).expanduser().resolve()
    dataset_table = Path(args.dataset_table).expanduser().resolve()
    reference_table = Path(args.reference_table).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None

    if not args.skip_reference_table_check:
        verify_reference_table(dataset_table, reference_table)

    manifest_rows = read_csv_rows(manifest)
    table_rows = read_csv_rows(dataset_table)
    verify_manifest_against_table(manifest_rows, table_rows)

    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in manifest_rows:
        try:
            result = _prepare_one(
                row,
                target_root=target_root,
                base_url=args.base_url,
                source_root=source_root,
                local_mode=args.local_mode,
                overwrite=bool(args.overwrite),
                dry_run=bool(args.dry_run),
                verify_only=bool(args.verify_only),
                skip_h5ad_check=bool(args.skip_h5ad_check),
            )
            print(f"{row['dataset_key']}: {result['status']} -> {result['target']}", flush=True)
            results.append(result)
        except Exception as exc:  # pylint: disable=broad-except
            message = f"{row['dataset_key']}: failed:{type(exc).__name__}:{exc}"
            print(message, flush=True)
            failures.append(message)

    if not args.dry_run:
        write_manifest(results, target_root / "DOWNLOAD_MANIFEST.csv")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
