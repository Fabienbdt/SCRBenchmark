#!/usr/bin/env python3
"""Extract marker-overlap DEG gene lists already present on disk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "report_artifacts" / "marker_overlap_genes"
DEFAULT_SCAN_ROOTS = [
    REPO_ROOT / "results",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Path to an existing degs_top100.json file. Can be repeated.",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[str(path) for path in DEFAULT_SCAN_ROOTS],
        help="Directory searched for degs_top100.json when --input is omitted.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("_")


def find_inputs(paths: list[str], scan_roots: list[str]) -> list[Path]:
    if paths:
        return [Path(path).expanduser().resolve() for path in paths]

    found: list[Path] = []
    for raw_root in scan_roots:
        root = Path(raw_root).expanduser().resolve()
        if root.exists():
            found.extend(sorted(root.rglob("degs_top100.json")))
    return sorted(dict.fromkeys(found))


def _gene_rows(payload: dict[str, Any], key: str, source_kind: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    groups = payload.get(key) or {}
    if not isinstance(groups, dict):
        return rows
    for group_name, genes in sorted(groups.items(), key=lambda item: str(item[0])):
        if not isinstance(genes, list):
            continue
        for rank, gene in enumerate(genes, start=1):
            rows.append(
                {
                    "source_kind": source_kind,
                    "group": str(group_name),
                    "rank": str(rank),
                    "gene": str(gene),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_one(input_path: Path, output_dir: Path) -> list[dict[str, str]]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    stem = safe_name(input_path.parent.parent.parent.name or input_path.parent.name)
    if stem == "results":
        stem = safe_name(input_path.parent.parent.name)

    gold_rows = _gene_rows(payload, "ground_truth_degs", "ground_truth")
    pred_rows = _gene_rows(payload, "predicted_cluster_degs", "predicted_cluster")
    all_rows = gold_rows + pred_rows
    mapping = payload.get("cluster_to_type") or {}
    mapping_rows = [
        {"predicted_cluster": str(cluster), "assigned_cell_type": str(cell_type)}
        for cluster, cell_type in sorted(mapping.items(), key=lambda item: str(item[0]))
    ]

    write_csv(
        output_dir / f"{stem}_ground_truth_degs.csv",
        gold_rows,
        ["source_kind", "group", "rank", "gene"],
    )
    write_csv(
        output_dir / f"{stem}_predicted_cluster_degs.csv",
        pred_rows,
        ["source_kind", "group", "rank", "gene"],
    )
    write_csv(
        output_dir / f"{stem}_marker_overlap_genes_long.csv",
        all_rows,
        ["source_kind", "group", "rank", "gene"],
    )
    write_csv(
        output_dir / f"{stem}_cluster_to_type.csv",
        mapping_rows,
        ["predicted_cluster", "assigned_cell_type"],
    )
    return [
        {
            "input": str(input_path),
            "output_prefix": str(output_dir / stem),
            "ground_truth_gene_rows": str(len(gold_rows)),
            "predicted_cluster_gene_rows": str(len(pred_rows)),
            "cluster_mapping_rows": str(len(mapping_rows)),
            "status": "extracted",
        }
    ]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    inputs = find_inputs(args.input, args.scan_root)
    manifest_rows: list[dict[str, str]] = []

    if not inputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        missing = output_dir / "missing_marker_overlap_degs.txt"
        missing.write_text(
            "No degs_top100.json file was found. Existing marker-overlap heatmaps "
            "or overlap matrices do not contain the DEG gene lists themselves.\n",
            encoding="utf-8",
        )
        print(f"missing = {missing}", flush=True)
        return 2

    for path in inputs:
        if not path.exists():
            manifest_rows.append(
                {
                    "input": str(path),
                    "output_prefix": "",
                    "ground_truth_gene_rows": "0",
                    "predicted_cluster_gene_rows": "0",
                    "cluster_mapping_rows": "0",
                    "status": "missing",
                }
            )
            continue
        manifest_rows.extend(extract_one(path, output_dir))

    write_csv(
        output_dir / "marker_overlap_gene_extraction_manifest.csv",
        manifest_rows,
        [
            "input",
            "output_prefix",
            "ground_truth_gene_rows",
            "predicted_cluster_gene_rows",
            "cluster_mapping_rows",
            "status",
        ],
    )
    print(f"output_dir = {output_dir}", flush=True)
    print(f"inputs = {len(inputs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
