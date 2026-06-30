#!/usr/bin/env python3
"""Export existing scRAW weights and checkpoint artifacts without rerunning experiments."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "stable_generalist" / "download_manifest.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "report_artifacts" / "scraw_existing_artifacts"
DEFAULT_WEIGHTS_ROOT = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "presentation_stable_generalist_nonbaron_20260324/"
    "Exp\u00e9riences/scRAW_default_from_scRAW_seed60_stage_umaps_20260421"
)
DEFAULT_MODEL_ROOTS = [
    Path(
        "/data2/fbidet/scRAW_Inductif/results/"
        "inductive_scraw_stable_generalist_exact_all_datasets_20260507_145430/"
        "01_scraw_runs"
    ),
    Path(
        "/data2/fbidet/scRAW_Inductif/results/"
        "inductive_multidataset_top4_representative_20260428/01_new_runs"
    ),
]

MANIFEST_FIELDS = [
    "dataset_key",
    "artifact_type",
    "split",
    "source",
    "target",
    "status",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--weights-root", default=str(DEFAULT_WEIGHTS_ROOT))
    parser.add_argument(
        "--model-root",
        action="append",
        default=[str(path) for path in DEFAULT_MODEL_ROOTS],
        help="Root searched for inductive scRAW model artifacts. Can be repeated.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def dataset_keys(manifest: Path) -> list[str]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return [row["dataset_key"] for row in csv.DictReader(handle)]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("_")


def materialize(source: Path, target: Path, *, mode: str, overwrite: bool, dry_run: bool) -> str:
    if not source.exists():
        return "missing_source"
    if target.exists() or target.is_symlink():
        if not overwrite:
            return "exists"
        if not dry_run:
            target.unlink()
    if dry_run:
        return f"would_{mode}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source)
        return "symlinked"
    if mode == "hardlink":
        try:
            os.link(source, target)
            return "hardlinked"
        except OSError:
            shutil.copy2(source, target)
            return "copied_after_hardlink_failed"
    shutil.copy2(source, target)
    return "copied"


def row(
    *,
    dataset_key: str,
    artifact_type: str,
    split: str,
    source: Path | None,
    target: Path | None,
    status: str,
    note: str = "",
) -> dict[str, str]:
    return {
        "dataset_key": dataset_key,
        "artifact_type": artifact_type,
        "split": split,
        "source": "" if source is None else str(source),
        "target": "" if target is None else str(target),
        "status": status,
        "note": note,
    }


def weight_source(weights_root: Path, dataset_key: str) -> Path:
    return weights_root / dataset_key / "seed_60" / "results" / "weights" / "cell_weights_scraw_run0.csv"


def iter_model_bundles(model_roots: Iterable[Path], dataset_key: str) -> Iterable[tuple[str, Path]]:
    seen_splits: set[str] = set()
    for root in model_roots:
        direct = root / dataset_key / "scraw" / "artifacts"
        if (direct / "autoencoder.pt").exists() and "default" not in seen_splits:
            seen_splits.add("default")
            yield "default", direct

        dataset_root = root / dataset_key
        if dataset_root.exists():
            for child in sorted(dataset_root.iterdir()):
                if (
                    child.is_dir()
                    and child.name not in seen_splits
                    and (child / "models" / "autoencoder.pt").exists()
                ):
                    seen_splits.add(child.name)
                    yield child.name, child


def export_weight(
    *,
    dataset_key: str,
    source: Path,
    output_root: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, str]:
    target = output_root / "weights" / f"cell_weights_{safe_name(dataset_key)}.csv"
    status = materialize(source, target, mode=mode, overwrite=overwrite, dry_run=dry_run)
    note = "Existing stable-generalist scRAW seed 60 cell weights."
    return row(
        dataset_key=dataset_key,
        artifact_type="cell_weights",
        split="seed_60",
        source=source,
        target=target,
        status=status,
        note=note,
    )


def export_model_bundle(
    *,
    dataset_key: str,
    split: str,
    bundle_dir: Path,
    output_root: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> list[dict[str, str]]:
    suffix = safe_name(dataset_key if split == "default" else f"{dataset_key}_{split}")
    items = [
        ("model", bundle_dir / "autoencoder.pt", output_root / "models" / f"model_{suffix}.pt"),
        ("config", bundle_dir / "config_used.json", output_root / "configs" / f"config_{suffix}.json"),
        (
            "preprocessing_state",
            bundle_dir / "preprocessing_state.npz",
            output_root / "models" / f"preprocessing_state_{suffix}.npz",
        ),
        (
            "centroid_reference",
            bundle_dir / "centroid_reference.npz",
            output_root / "models" / f"centroid_reference_{suffix}.npz",
        ),
        (
            "train_cell_weights",
            bundle_dir / "train_cell_weights.npy",
            output_root / "weights" / f"train_cell_weights_{suffix}.npy",
        ),
    ]
    if (bundle_dir / "models").exists():
        items = [
            ("model", bundle_dir / "models" / "autoencoder.pt", output_root / "models" / f"model_{suffix}.pt"),
            (
                "config",
                bundle_dir / "config" / "config_used.json",
                output_root / "configs" / f"config_{suffix}.json",
            ),
            (
                "preprocessing_state",
                bundle_dir / "models" / "preprocessing_state.npz",
                output_root / "models" / f"preprocessing_state_{suffix}.npz",
            ),
            (
                "centroid_reference",
                bundle_dir / "models" / "centroid_reference.npz",
                output_root / "models" / f"centroid_reference_{suffix}.npz",
            ),
            (
                "train_cell_weights",
                bundle_dir / "results" / "train_cell_weights.npy",
                output_root / "weights" / f"train_cell_weights_{suffix}.npy",
            ),
        ]

    rows = []
    for artifact_type, source, target in items:
        status = materialize(source, target, mode=mode, overwrite=overwrite, dry_run=dry_run)
        rows.append(
            row(
                dataset_key=dataset_key,
                artifact_type=artifact_type,
                split=split,
                source=source,
                target=target,
                status=status,
                note="Existing inductive scRAW artifact.",
            )
        )
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    keys = dataset_keys(Path(args.manifest).expanduser().resolve())
    weights_root = Path(args.weights_root).expanduser().resolve()
    model_roots = [Path(path).expanduser().resolve() for path in args.model_root]
    output_root = Path(args.output_root).expanduser().resolve()

    rows: list[dict[str, str]] = []
    for key in keys:
        rows.append(
            export_weight(
                dataset_key=key,
                source=weight_source(weights_root, key),
                output_root=output_root,
                mode=args.mode,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

        bundles = list(iter_model_bundles(model_roots, key))
        if not bundles:
            rows.append(
                row(
                    dataset_key=key,
                    artifact_type="model",
                    split="",
                    source=None,
                    target=None,
                    status="missing",
                    note="No existing autoencoder.pt found in the configured model roots.",
                )
            )
        for split, bundle_dir in bundles:
            rows.extend(
                export_model_bundle(
                    dataset_key=key,
                    split=split,
                    bundle_dir=bundle_dir,
                    output_root=output_root,
                    mode=args.mode,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                )
            )

    manifest_path = output_root / "scraw_existing_artifacts_manifest.csv"
    write_manifest(manifest_path, rows)
    print(f"manifest = {manifest_path}", flush=True)
    print(f"rows = {len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
