#!/usr/bin/env python3
"""List, download, and verify the upstream assets used by the M2 report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "report_sources" / "source_manifest.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "report_sources" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Select every upstream asset.")
    selection.add_argument(
        "--datasets",
        help="Comma-separated prepared dataset keys; selects every matching source asset.",
    )
    selection.add_argument(
        "--assets",
        help="Comma-separated asset IDs from the source manifest.",
    )
    parser.add_argument("--list", action="store_true", help="Print selected assets without downloading.")
    parser.add_argument("--verify-only", action="store_true", help="Do not download missing files.")
    parser.add_argument("--force", action="store_true", help="Replace existing files.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"asset_id", "dataset_keys", "filename", "url"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Invalid source manifest: {path}")
    asset_ids = [row["asset_id"].strip() for row in rows]
    if len(asset_ids) != len(set(asset_ids)) or any(not asset_id for asset_id in asset_ids):
        raise ValueError(f"Asset IDs must be non-empty and unique: {path}")
    for row in rows:
        filename = Path(row["filename"])
        if filename.name != row["filename"] or filename.is_absolute():
            raise ValueError(f"Unsafe filename in source manifest: {row['filename']!r}")
        if not row["url"].startswith("https://"):
            raise ValueError(f"Only HTTPS source URLs are accepted: {row['url']!r}")
    return rows


def split_values(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.all:
        return rows
    if args.datasets:
        wanted = {item.strip() for item in args.datasets.split(",") if item.strip()}
        selected = [row for row in rows if wanted & split_values(row["dataset_keys"])]
        found = set().union(*(split_values(row["dataset_keys"]) for row in selected)) if selected else set()
        missing = wanted - found
    elif args.assets:
        wanted = {item.strip() for item in args.assets.split(",") if item.strip()}
        selected = [row for row in rows if row["asset_id"] in wanted]
        missing = wanted - {row["asset_id"] for row in selected}
    else:
        selected = rows
        missing = set()
    if missing:
        raise ValueError(f"Unknown dataset/asset selection: {', '.join(sorted(missing))}")
    return selected


def digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify(path: Path, row: dict[str, str]) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    expected_size = row.get("size_bytes", "").strip()
    if expected_size and path.stat().st_size != int(expected_size):
        return False, f"size={path.stat().st_size}, expected={expected_size}"
    algorithm = row.get("checksum_algorithm", "").strip().lower()
    expected = row.get("checksum", "").strip().lower()
    if algorithm and expected:
        actual = digest(path, algorithm)
        if actual != expected:
            return False, f"{algorithm}={actual}, expected={expected}"
    return True, "verified" if expected_size or expected else "present (no pinned checksum)"


def download(row: dict[str, str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(row["url"], headers={"User-Agent": "SCRBenchmark-source-downloader/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    rows = select_rows(read_manifest(args.manifest.resolve()), args)
    if args.list or (not args.all and not args.datasets and not args.assets and not args.verify_only):
        for row in rows:
            print(f"{row['asset_id']}\t{row['dataset_keys']}\t{row['filename']}\t{row['url']}")
        if not args.list:
            print("\nNo download started. Pass --all, --datasets, or --assets explicitly.", file=sys.stderr)
        return 0

    failures = 0
    for row in rows:
        destination = args.output_dir.resolve() / row["filename"]
        ok, message = verify(destination, row)
        if ok and not args.force:
            print(f"OK      {row['asset_id']}: {message}")
            continue
        if args.verify_only:
            print(f"MISSING {row['asset_id']}: {message}")
            failures += 1
            continue
        print(f"GET     {row['asset_id']}: {row['url']}")
        download(row, destination)
        ok, message = verify(destination, row)
        print(f"{'OK' if ok else 'ERROR':7} {row['asset_id']}: {message}")
        failures += int(not ok)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
