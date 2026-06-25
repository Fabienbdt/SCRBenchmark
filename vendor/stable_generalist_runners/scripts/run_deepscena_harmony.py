#!/usr/bin/env python3
"""Add a DeepScena+Harmony variant for the stable_generalist benchmark datasets.

This script reuses already trained DeepScena latent embeddings, applies Harmony
batch correction to that latent space, reclusters the corrected representation,
and writes outputs in the same external-method layout as ``run_deepscena_stable_generalist``.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from run_deepscena import (
    DEFAULT_MANIFEST,
    DEFAULT_PRESENTATION_ROOT,
    PREPROCESSING,
    DatasetSpec,
    _read_manifest,
    _safe_json,
    _select_datasets,
    _setup_logging,
    _write_json,
    preprocess_deepscena,
    save_failed_run,
    save_successful_run,
)


LOGGER = logging.getLogger("run_deepscena_harmony_stable_generalist")
DEFAULT_BASE_ROOT = (
    DEFAULT_PRESENTATION_ROOT / "external_rare_methods_deepscena_20260513_1135"
)


def _ensure_obs_first(embedding: np.ndarray, n_obs: int, context: str) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{context} returned ndim={arr.ndim}, expected 2.")
    if arr.shape[0] == int(n_obs):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[1] == int(n_obs):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"{context} returned shape {arr.shape}; expected one dimension to match n_obs={n_obs}.")


def apply_harmony(
    embedding: np.ndarray,
    adata_proc: ad.AnnData,
    batch_key: str,
    *,
    max_iter: int,
    nclust: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    embedding = _ensure_obs_first(embedding, adata_proc.n_obs, "DeepScena embedding")
    if batch_key not in adata_proc.obs.columns:
        raise KeyError(f"Batch key '{batch_key}' not found in adata.obs")
    n_batches = int(adata_proc.obs[batch_key].astype(str).nunique())
    if n_batches < 2:
        return embedding, {
            "harmony_status": "skipped_single_batch",
            "n_batches": n_batches,
            "harmony_on": "DeepScena latent u",
        }

    import harmonypy as hm

    ho = hm.run_harmony(
        np.ascontiguousarray(embedding, dtype=np.float32),
        adata_proc.obs.copy(),
        vars_use=[batch_key],
        max_iter_harmony=int(max_iter),
        nclust=int(nclust),
    )
    corrected = _ensure_obs_first(ho.Z_corr, adata_proc.n_obs, "Harmony")
    return corrected, {
        "harmony_status": "ok",
        "n_batches": n_batches,
        "harmony_on": "DeepScena latent u",
        "harmony_max_iter": int(max_iter),
        "harmony_nclust": int(nclust),
    }


def _base_embedding_path(base_root: Path, dataset_key: str) -> Path:
    return base_root / "standard" / dataset_key / "deepscena" / "results" / "embeddings" / "embedding_deepscena_standard.npy"


def run_one_dataset(
    spec: DatasetSpec,
    *,
    base_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    output_dir = output_root / "harmony" / spec.dataset_key / "deepscena"
    analysis_csv = output_dir / "results" / "analysis_results.csv"
    if analysis_csv.exists() and not bool(args.overwrite):
        LOGGER.info("[skip] %s DeepScena+Harmony already exists", spec.dataset_key)
        return {
            "dataset_key": spec.dataset_key,
            "method": "DeepScena+Harmony",
            "variant": "harmony",
            "status": "ok",
            "error": "",
            "output_dir": str(output_dir),
            "skipped_existing": True,
        }

    embedding_path = _base_embedding_path(base_root, spec.dataset_key)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing standard DeepScena embedding: {embedding_path}")

    LOGGER.info("[run] dataset=%s method=DeepScena+Harmony", spec.dataset_key)
    start = time.time()
    adata = ad.read_h5ad(spec.path)
    adata_proc, _matrix_784, preprocess_stats = preprocess_deepscena(
        adata,
        label_key=spec.label_key,
        batch_key=spec.batch_key,
        n_top_genes=int(args.n_top_genes),
    )
    embedding = _ensure_obs_first(np.load(embedding_path), adata_proc.n_obs, "DeepScena embedding")
    corrected, harmony_info = apply_harmony(
        embedding,
        adata_proc,
        spec.batch_key,
        max_iter=int(args.harmony_max_iter),
        nclust=int(args.harmony_nclust),
    )

    labels_pred = KMeans(
        n_clusters=int(spec.n_labels_expected),
        n_init=int(args.kmeans_n_init),
        random_state=int(args.seed),
    ).fit_predict(corrected)
    runtime = float(time.time() - start)
    method_info = {
        "implementation": "DeepScena_latent_embedding_plus_Harmony_then_KMeans",
        "base_deepscena_root": str(base_root),
        "base_embedding": str(embedding_path),
        "clusterer": "KMeans_on_Harmony_corrected_DeepScena_latent",
        "kmeans_n_clusters": int(spec.n_labels_expected),
        "kmeans_n_init": int(args.kmeans_n_init),
        "kmeans_random_state": int(args.seed),
        **harmony_info,
    }
    save_successful_run(
        output_dir,
        spec=spec,
        adata_proc=adata_proc,
        preprocess_stats=preprocess_stats,
        labels_pred=np.asarray(labels_pred, dtype=object).astype(str),
        embedding=np.asarray(corrected, dtype=np.float32),
        runtime=runtime,
        method_info=method_info,
        scib_n_jobs=int(args.scib_n_jobs),
        variant="harmony",
        display_name="DeepScena+Harmony",
    )
    return {
        "dataset_key": spec.dataset_key,
        "method": "DeepScena+Harmony",
        "variant": "harmony",
        "status": "ok",
        "error": "",
        "output_dir": str(output_dir),
        "runtime": runtime,
        "skipped_existing": False,
    }


def _output_root(raw: str) -> Path:
    if str(raw).strip():
        return Path(raw).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_PRESENTATION_ROOT / f"external_rare_methods_deepscena_harmony_{stamp}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--base-root", default=str(DEFAULT_BASE_ROOT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--n-top-genes", type=int, default=784)
    parser.add_argument("--harmony-max-iter", type=int, default=10)
    parser.add_argument("--harmony-nclust", type=int, default=50)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(bool(args.verbose))
    base_root = Path(args.base_root).resolve()
    output_root = _output_root(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    specs = _select_datasets(_read_manifest(Path(args.manifest)), str(args.datasets))

    _write_json(
        output_root / "run_metadata.json",
        {
            "timestamp": datetime.now().isoformat(),
            "manifest": str(Path(args.manifest).resolve()),
            "base_root": str(base_root),
            "output_root": str(output_root),
            "method": "DeepScena+Harmony",
            "variant": "harmony",
            "seed": int(args.seed),
            "preprocessing": dict(PREPROCESSING),
            "n_datasets": len(specs),
            "harmony_max_iter": int(args.harmony_max_iter),
            "harmony_nclust": int(args.harmony_nclust),
            "kmeans_n_init": int(args.kmeans_n_init),
            "scib_n_jobs": int(args.scib_n_jobs),
        },
    )

    run_rows: List[Dict[str, Any]] = []
    overall_status = "ok"
    for spec in specs:
        try:
            row = run_one_dataset(spec, base_root=base_root, output_root=output_root, args=args)
            run_rows.append(row)
            LOGGER.info("Finished dataset=%s status=%s", spec.dataset_key, row.get("status"))
        except Exception as exc:
            overall_status = "failed"
            output_dir = output_root / "harmony" / spec.dataset_key / "deepscena"
            save_failed_run(output_dir, spec, exc, variant="harmony")
            LOGGER.exception("Failed dataset=%s method=DeepScena+Harmony", spec.dataset_key)
            run_rows.append(
                {
                    "dataset_key": spec.dataset_key,
                    "method": "DeepScena+Harmony",
                    "variant": "harmony",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_dir": str(output_dir),
                }
            )
            if not bool(args.continue_on_error):
                break
        finally:
            pd.DataFrame(run_rows).to_csv(output_root / "run_status.csv", index=False)

    _write_json(
        output_root / "completion_summary.json",
        {
            "status": overall_status,
            "n_rows": len(run_rows),
            "n_ok": int(sum(1 for row in run_rows if row.get("status") == "ok")),
            "n_failed": int(sum(1 for row in run_rows if row.get("status") != "ok")),
            "output_root": str(output_root),
        },
    )
    print(str(output_root))
    return 0 if overall_status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
