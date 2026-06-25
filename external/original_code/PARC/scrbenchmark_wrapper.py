#!/usr/bin/env python3
"""SCRBenchmark wrapper for the external PARC source tree.

The PARC core is left untouched. This wrapper only translates SCRBenchmark's
AnnData/CLI contract into PARC's ``PARC(...).run_PARC()`` API and writes the
standard labels/embedding files consumed by ``run_method.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import parc


def _parse_scalar(raw: str) -> Any:
    text = str(raw).strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _parse_params(items: list[str], method_name: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for item in items:
        key, sep, value = str(item).partition("=")
        if not sep:
            continue
        key = key.strip()
        if ":" in key:
            prefix, key = key.split(":", 1)
            if prefix.casefold() not in {method_name.casefold(), "all", "*"}:
                continue
        key = key.strip().replace("-", "_")
        if key:
            params[key] = _parse_scalar(value)
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input .h5ad file.")
    parser.add_argument("--output", required=True, help="Raw output directory.")
    parser.add_argument("--clusters", type=int, default=0, help="Ignored by PARC; kept for SCRBenchmark compatibility.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--param", action="append", default=[])
    return parser.parse_args()


def _preprocess(input_path: str, args: argparse.Namespace) -> tuple[ad.AnnData, np.ndarray]:
    data = ad.read_h5ad(input_path)
    data.var_names_make_unique()

    if args.min_genes_per_cell > 0:
        sc.pp.filter_cells(data, min_genes=args.min_genes_per_cell)
    if args.max_genes_per_cell > 0:
        counts_per_cell = np.asarray(data.X.sum(axis=1)).reshape(-1)
        data = data[counts_per_cell <= args.max_genes_per_cell].copy()
    if args.min_cells_per_gene > 0:
        sc.pp.filter_genes(data, min_cells=args.min_cells_per_gene)

    if data.n_obs == 0 or data.n_vars == 0:
        raise ValueError("No cells or genes remain after preprocessing filters.")

    if args.target_sum > 0:
        sc.pp.normalize_total(data, target_sum=args.target_sum)
    sc.pp.log1p(data)

    if 0 < args.n_top_genes < data.n_vars:
        try:
            sc.pp.highly_variable_genes(data, n_top_genes=args.n_top_genes, flavor=args.hvg_flavor)
            if "highly_variable" in data.var and bool(data.var["highly_variable"].any()):
                data = data[:, data.var["highly_variable"]].copy()
        except Exception as exc:
            print(f"Warning: HVG selection failed ({exc}); using all remaining genes.")

    sc.pp.scale(data, max_value=args.scale_max_value if args.scale_max_value > 0 else None)

    n_comps = min(int(args.n_pcs), data.n_obs - 1, data.n_vars - 1)
    if n_comps >= 1:
        sc.tl.pca(data, n_comps=n_comps, random_state=int(args.seed))
        matrix = np.asarray(data.obsm["X_pca"], dtype=np.float64)
    else:
        matrix = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
        matrix = np.asarray(matrix, dtype=np.float64)
    return data, matrix


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    params = _parse_params(args.param, "PARC")
    parc_kwargs = {
        "knn": int(params.pop("knn", min(30, max(2, args.clusters or 30)))),
        "dist_std_local": params.pop("dist_std_local", 3),
        "jac_std_global": params.pop("jac_std_global", "median"),
        "small_pop": int(params.pop("small_pop", 10)),
        "n_iter_leiden": int(params.pop("n_iter_leiden", 5)),
        "random_seed": int(params.pop("random_seed", args.seed)),
        "resolution_parameter": float(params.pop("resolution_parameter", 1.0)),
        "jac_weighted_edges": bool(params.pop("jac_weighted_edges", True)),
        "num_threads": int(params.pop("num_threads", -1)),
    }
    for key in ("too_big_factor", "keep_all_local_dist", "distance", "partition_type", "time_smallpop"):
        if key in params:
            parc_kwargs[key] = params.pop(key)
    if params:
        print(f"Warning: unused PARC params: {json.dumps(params, sort_keys=True)}")

    data, matrix = _preprocess(args.input, args)
    model = parc.PARC(matrix, true_label=None, **parc_kwargs)
    model.run_PARC()

    labels = pd.DataFrame({"cell_id": data.obs_names.astype(str), "cluster": list(model.labels)})
    labels.to_csv(output / "labels.csv", index=False)

    latent = pd.DataFrame(matrix, columns=[f"latent_{idx + 1}" for idx in range(matrix.shape[1])])
    latent.insert(0, "cell_id", data.obs_names.astype(str))
    latent.to_csv(output / "latent.csv", index=False)

    metadata = {
        "method": "PARC",
        "n_cells": int(data.n_obs),
        "n_genes": int(data.n_vars),
        "n_latent": int(matrix.shape[1]),
        "parc_kwargs": parc_kwargs,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
