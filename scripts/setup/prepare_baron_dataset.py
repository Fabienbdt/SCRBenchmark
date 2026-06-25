#!/usr/bin/env python3
"""Download and assemble the Baron human pancreas dataset (GSE84133)."""

from __future__ import annotations

import argparse
import gzip
import logging
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


LOGGER = logging.getLogger("prepare_baron")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "GSE84133_RAW"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "baron_human_pancreas.h5ad"

DONORS = {
    "human1": "GSM2230757_human1_umifm_counts.csv.gz",
    "human2": "GSM2230758_human2_umifm_counts.csv.gz",
    "human3": "GSM2230759_human3_umifm_counts.csv.gz",
    "human4": "GSM2230760_human4_umifm_counts.csv.gz",
}

URLS = {
    filename: (
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2230nnn/"
        f"{filename.split('_', 1)[0]}/suppl/{filename}"
    )
    for filename in DONORS.values()
}


def download_missing(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in URLS.items():
        destination = raw_dir / filename
        if destination.exists():
            continue
        LOGGER.info("Downloading %s", filename)
        urllib.request.urlretrieve(url, destination)


def load_donor(path: Path, donor: str) -> ad.AnnData:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run this script with --download or place the GEO file there."
        )

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        frame = pd.read_csv(handle, index_col=0)

    required = {"barcode", "assigned_cluster"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    gene_columns = [
        column
        for column in frame.columns
        if column not in {"barcode", "assigned_cluster"}
    ]
    labels = frame["assigned_cluster"].astype(str)
    obs = pd.DataFrame(
        {
            "barcode": frame["barcode"].astype(str),
            "assigned_cluster": labels,
            "Group": labels,
            "label": labels,
            "cell_type": labels,
            "labels": labels,
            "batch": donor,
        },
        index=frame.index.astype(str),
    )
    matrix = frame[gene_columns].to_numpy(dtype=np.float32, copy=False)
    return ad.AnnData(
        X=csr_matrix(matrix),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(gene_columns, dtype=str)),
    )


def build_dataset(raw_dir: Path, output: Path) -> ad.AnnData:
    datasets = [load_donor(raw_dir / filename, donor) for donor, filename in DONORS.items()]
    merged = ad.concat(
        datasets,
        join="inner",
        merge="same",
        keys=list(DONORS),
        index_unique="-",
    )
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    merged.uns.update(
        {
            "dataset_name": "Baron Human Pancreas",
            "source": "GEO GSE84133",
            "reference": "Baron et al., Cell Systems 2016",
            "donors": list(DONORS),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output, compression="gzip")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.download:
        download_missing(args.raw_dir)
    dataset = build_dataset(args.raw_dir, args.output)
    LOGGER.info("Wrote %s: %d cells x %d genes", args.output, dataset.n_obs, dataset.n_vars)


if __name__ == "__main__":
    main()
