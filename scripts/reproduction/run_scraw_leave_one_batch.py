#!/usr/bin/env python3
"""Run vendored scRAW inductively by holding out each batch of one AnnData file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAW_INDUCTIVE_ROOT = REPO_ROOT / "vendor" / "scraw_inductive"
SCRAW_INDUCTIVE_SRC = SCRAW_INDUCTIVE_ROOT / "src"
if str(SCRAW_INDUCTIVE_SRC) not in sys.path:
    sys.path.insert(0, str(SCRAW_INDUCTIVE_SRC))

from scraw import run_inductive_baron_split  # noqa: E402
from scraw.metrics import NOISE_LABELS, align_labels  # noqa: E402
from scraw.presets import resolve_preset_config  # noqa: E402


SUMMARY_FIELDS = [
    "dataset_name",
    "preset",
    "split",
    "train_batches",
    "test_batch",
    "device",
    "n_train",
    "n_test",
    "n_genes",
    "ACC",
    "ARI",
    "NMI",
    "RareACC",
    "BalancedRareACC",
    "UltraRareACC",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(REPO_ROOT / "data" / "baron_human_pancreas.h5ad"),
        help="Input .h5ad file.",
    )
    parser.add_argument("--dataset-name", default="", help="Name written to summary.csv.")
    parser.add_argument(
        "--preset",
        default="default",
        choices=["default", "baron"],
    )
    parser.add_argument("--split-key", default="batch")
    parser.add_argument("--label-key", default=None)
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "results" / "leave_one_batch" / "scraw_inductive"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Optional held-out batches to run. Defaults to all batches in the file.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-filter-inference-cells",
        action="store_true",
        help="Do not apply deterministic QC filtering to held-out batches.",
    )
    return parser.parse_args()


def read_batches(data_path: str, split_key: str) -> list[str]:
    import scanpy as sc

    adata = sc.read_h5ad(data_path, backed="r")
    try:
        if split_key not in adata.obs.columns:
            raise ValueError(f"Split key {split_key!r} was not found in adata.obs.")
        return sorted(adata.obs[split_key].astype(str).unique().tolist())
    finally:
        adata.file.close()


def ultra_rare_acc(result_dir: Path) -> float:
    true_path = result_dir / "results" / "test_true_labels.npy"
    pred_path = result_dir / "results" / "test_pred_labels.npy"
    if not true_path.exists() or not pred_path.exists():
        return float("nan")

    labels_true = np.load(true_path, allow_pickle=True)
    labels_pred = np.load(pred_path, allow_pickle=True)
    if len(labels_true) != len(labels_pred) or len(labels_true) == 0:
        return float("nan")

    mask = np.ones(len(labels_pred), dtype=bool)
    pred_str = labels_pred.astype(str)
    true_str = labels_true.astype(str)
    for value in NOISE_LABELS:
        mask &= pred_str != str(value)
        mask &= true_str != str(value)
    labels_true = labels_true[mask]
    labels_pred = labels_pred[mask]
    if len(labels_true) == 0:
        return float("nan")

    aligned = align_labels(labels_true, labels_pred)
    classes, counts = np.unique(labels_true, return_counts=True)
    ultra_classes = classes[(counts / len(labels_true)) < 0.01]
    if len(ultra_classes) == 0:
        return float("nan")
    ultra_mask = np.isin(labels_true, ultra_classes)
    return float(np.mean(aligned[ultra_mask] == labels_true[ultra_mask]))


def write_summary_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _fmt(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path).expanduser().resolve()
    dataset_name = args.dataset_name or data_path.stem
    available_batches = read_batches(str(data_path), args.split_key)
    if len(available_batches) < 2:
        raise ValueError("Need at least two batches for a leave-one-batch run.")
    if args.batches is None:
        test_batches = available_batches
    else:
        test_batches = list(dict.fromkeys(str(batch) for batch in args.batches))
        missing = sorted(set(test_batches) - set(available_batches))
        if missing:
            raise ValueError(f"Unknown held-out batch values for {args.split_key!r}: {missing}")

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"

    print(f"dataset = {dataset_name}", flush=True)
    print(f"data_path = {data_path}", flush=True)
    print(f"preset = {args.preset}", flush=True)
    print(f"split_key = {args.split_key}", flush=True)
    print(f"label_key = {args.label_key}", flush=True)
    print(f"available_batches = {available_batches}", flush=True)
    print(f"heldout_batches = {test_batches}", flush=True)
    print(f"output_root = {output_root}", flush=True)

    for test_batch in test_batches:
        train_batches = [batch for batch in available_batches if batch != test_batch]
        split_name = "train_" + "-".join(train_batches) + "_to_" + test_batch
        split_name = split_name.replace("/", "_").replace(" ", "_")
        out_dir = output_root / split_name
        results_json = out_dir / "results" / "results.json"

        if args.skip_existing and results_json.exists():
            print(f"[{split_name}] skip existing", flush=True)
            continue

        if args.dry_run:
            print(f"[dry-run] train={train_batches} test={test_batch} -> {out_dir}", flush=True)
            continue

        config = resolve_preset_config(args.preset, repo_root=SCRAW_INDUCTIVE_ROOT)
        config.data.data_path = str(data_path)
        config.data.output_dir = str(out_dir)
        config.data.label_key = args.label_key
        config.runtime.device = args.device

        print(f"[{split_name}] train={train_batches} test={test_batch}", flush=True)
        result = run_inductive_baron_split(
            config=config,
            train_batches=train_batches,
            test_batches=[test_batch],
            split_key=args.split_key,
            output_dir=out_dir,
            device=args.device,
            filter_inference_cells=not args.no_filter_inference_cells,
        )

        metrics = result["test_metrics"]
        result_payload = json.loads(results_json.read_text(encoding="utf-8"))
        row = {
            "dataset_name": dataset_name,
            "preset": args.preset,
            "split": split_name,
            "train_batches": ",".join(train_batches),
            "test_batch": test_batch,
            "device": result_payload.get("device", ""),
            "n_train": result_payload.get("n_train_cells", ""),
            "n_test": result_payload.get("n_test_cells", ""),
            "n_genes": result_payload.get("n_genes", ""),
            "ACC": metrics.get("ACC", ""),
            "ARI": metrics.get("ARI", ""),
            "NMI": metrics.get("NMI", ""),
            "RareACC": metrics.get("RareACC", ""),
            "BalancedRareACC": metrics.get("BalancedRareACC", ""),
            "UltraRareACC": ultra_rare_acc(out_dir),
            "output_dir": str(out_dir),
        }
        write_summary_row(summary_path, row)
        print(
            f"[{split_name}] ACC={_fmt(row['ACC'])} ARI={_fmt(row['ARI'])} NMI={_fmt(row['NMI'])}",
            flush=True,
        )

    print(f"summary = {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
