#!/usr/bin/env python3
"""Run a base method, apply Harmony to its embedding, then recluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from _runner_utils import REPO_ROOT, reproduction_env, run_logged, write_failure


BASE_ALGOS = {
    "scNAME+Harmony": ("scrbenchmark", "scname"),
    "scMAE+Harmony": ("scrbenchmark", "sc_mae"),
    "scvi+Harmony": ("batch_baseline", "scvi"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(BASE_ALGOS))
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-key", default="Group")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--n-labels", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument("--min-genes-per-cell", type=int, default=200)
    parser.add_argument("--max-genes-per-cell", type=int, default=10000)
    parser.add_argument("--min-cells-per-gene", type=int, default=3)
    parser.add_argument("--target-sum", type=float, default=20000.0)
    parser.add_argument("--scale-max-value", type=float, default=10.0)
    parser.add_argument("--hvg-flavor", default="seurat")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--harmony-max-iter", type=int, default=10)
    parser.add_argument("--harmony-nclust", type=int, default=50)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Base model parameter as algo:key=value or key=value. Harmony-specific CLI options have priority.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _split_param(raw: str) -> tuple[str, str, str]:
    text = str(raw).strip()
    if "=" not in text:
        raise ValueError(f"Invalid --param {raw!r}; expected algo:key=value or key=value.")
    left, value = text.split("=", 1)
    if ":" in left:
        target, key = left.split(":", 1)
    else:
        target, key = "", left
    return target.strip().lower(), key.strip(), value.strip()


def base_param_args(args: argparse.Namespace, algo: str) -> list[str]:
    out: list[str] = []
    ignored_keys = {"n_pcs", "harmony_max_iter", "harmony_nclust"}
    allowed_targets = {"", "*", "all", "base", algo.lower(), str(args.method).lower()}
    for raw in args.param or []:
        target, key, value = _split_param(raw)
        if key in ignored_keys:
            continue
        if target not in allowed_targets:
            continue
        out.extend(["--param", f"{algo}:{key}={value}"])
    return out


def run_scrbenchmark_base(args: argparse.Namespace, algo: str, base_dir: Path) -> tuple[Path, Path]:
    cmd = [
        str(args.python_bin),
        str(REPO_ROOT / "src" / "scrbenchmark" / "cli.py"),
        "run",
        "--data",
        str(Path(args.data).expanduser().resolve()),
        "--algorithms",
        algo,
        "--output",
        str(base_dir),
        "--no-timestamp",
        "--label-col",
        str(args.label_key),
        "--n-clusters",
        str(int(args.n_labels)),
        "--n-top-genes",
        str(int(args.n_top_genes)),
        "--min-genes-per-cell",
        str(int(args.min_genes_per_cell)),
        "--max-genes-per-cell",
        str(int(args.max_genes_per_cell)),
        "--min-cells-per-gene",
        str(int(args.min_cells_per_gene)),
        "--target-sum",
        str(float(args.target_sum)),
        "--scale-max-value",
        str(float(args.scale_max_value)),
        "--hvg-flavor",
        str(args.hvg_flavor),
        "--device",
        str(args.device),
        "--seed",
        str(int(args.seed)),
        "--n-repeats",
        "1",
        "--csv",
        "--save-labels",
        "--save-embeddings",
    ]
    cmd.extend(base_param_args(args, algo))
    run_logged(cmd, base_dir / "logs" / f"run_{algo}.log", env=reproduction_env())
    return (
        base_dir / "results" / "embeddings" / f"embeddings_{algo}_run0.npy",
        base_dir / "results" / "labels" / f"labels_{algo}_run0.csv",
    )


def run_scvi_base(args: argparse.Namespace, base_dir: Path) -> tuple[Path, Path]:
    cmd = [
        str(args.python_bin),
        str(REPO_ROOT / "scripts" / "reproduction" / "run_batch_baselines.py"),
        "--data",
        str(Path(args.data).expanduser().resolve()),
        "--output",
        str(base_dir),
        "--method",
        "scvi",
        "--seed",
        str(int(args.seed)),
        "--label-key",
        str(args.label_key),
        "--batch-key",
        str(args.batch_key),
        "--n-top-genes",
        str(int(args.n_top_genes)),
        "--min-genes-per-cell",
        str(int(args.min_genes_per_cell)),
        "--max-genes-per-cell",
        str(int(args.max_genes_per_cell)),
        "--min-cells-per-gene",
        str(int(args.min_cells_per_gene)),
        "--target-sum",
        str(float(args.target_sum)),
        "--scale-max-value",
        str(float(args.scale_max_value)),
        "--hvg-flavor",
        str(args.hvg_flavor),
        "--n-pcs",
        str(int(args.n_pcs)),
        "--compute-scib",
        "--scib-n-jobs",
        str(int(args.scib_n_jobs)),
        "--skip-umap-plots",
    ]
    if args.verbose:
        cmd.append("--verbose")
    run_logged(cmd, base_dir / "logs" / "run_scvi.log", env=reproduction_env())
    by_resolution = pd.read_csv(base_dir / "results" / "analysis_results_by_resolution.csv")
    best_resolution = str(float(by_resolution.iloc[0]["resolution"]))
    labels_path = base_dir / "results" / "labels" / f"per_cell_scvi_res_{best_resolution}.csv"
    return base_dir / "results" / "embeddings" / "embedding_scvi.npy", labels_path


def _ensure_obs_first(values: np.ndarray, n_obs: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape[0] == int(n_obs):
        return np.ascontiguousarray(arr)
    if arr.ndim == 2 and arr.shape[1] == int(n_obs):
        return np.ascontiguousarray(arr.T)
    raise ValueError(f"Embedding shape {arr.shape} does not match n_obs={n_obs}")


def apply_harmony(
    embedding: np.ndarray,
    batches: pd.Series,
    *,
    harmony_max_iter: int,
    harmony_nclust: int,
) -> np.ndarray:
    if batches.astype(str).nunique() < 2:
        return np.asarray(embedding, dtype=np.float32)
    import harmonypy as hm

    meta = pd.DataFrame({"batch": batches.astype(str).to_numpy()})
    result = hm.run_harmony(
        np.ascontiguousarray(embedding, dtype=np.float32),
        meta,
        vars_use=["batch"],
        max_iter_harmony=int(harmony_max_iter),
        nclust=int(harmony_nclust),
    )
    return _ensure_obs_first(np.asarray(result.Z_corr), embedding.shape[0])


def write_outputs(args: argparse.Namespace, output_dir: Path, embedding_path: Path, labels_path: Path) -> None:
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing base embedding: {embedding_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing base labels: {labels_path}")

    labels = pd.read_csv(labels_path)
    true_col = "true_label" if "true_label" in labels.columns else str(args.label_key)
    if true_col not in labels.columns or "batch" not in labels.columns:
        raise ValueError(f"Labels file must contain true labels and batch column: {labels_path}")

    embedding = _ensure_obs_first(np.load(embedding_path), len(labels))
    corrected = apply_harmony(
        embedding,
        labels["batch"],
        harmony_max_iter=int(args.harmony_max_iter),
        harmony_nclust=int(args.harmony_nclust),
    )
    predicted = KMeans(n_clusters=int(args.n_labels), n_init=10, random_state=int(args.seed)).fit_predict(corrected)

    from scraw_dedicated.metrics import align_labels, compute_metrics, compute_scib_metrics

    labels_true = labels[true_col].astype(str).to_numpy()
    metrics = compute_metrics(labels_true=labels_true, labels_pred=predicted.astype(str), embeddings=corrected)
    try:
        adata = ad.read_h5ad(Path(args.data).expanduser().resolve())
        if adata.n_obs == len(labels) and str(args.batch_key) in adata.obs and str(args.label_key) in adata.obs:
            metrics.update(
                compute_scib_metrics(
                    adata=adata,
                    embeddings=corrected,
                    batch_key=str(args.batch_key),
                    label_key=str(args.label_key),
                    n_jobs=int(args.scib_n_jobs),
                )
            )
    except Exception as exc:
        metrics["scib_status"] = f"skipped: {type(exc).__name__}: {exc}"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    labels_dir = results_dir / "labels"
    emb_dir = results_dir / "embeddings"
    config_dir = output_dir / "config"
    for path in [results_dir, labels_dir, emb_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)

    aligned = align_labels(labels_true, predicted.astype(str))
    per_cell = pd.DataFrame(
        {
            "true_label": labels_true,
            "batch": labels["batch"].astype(str).to_numpy(),
            "predicted_label": predicted.astype(str),
            "aligned_predicted_label": np.asarray(aligned, dtype=object).astype(str),
        }
    )
    per_cell.to_csv(labels_dir / "labels_posthoc_harmony_run0.csv", index=False)
    np.save(emb_dir / "embedding_posthoc_harmony.npy", corrected)
    pd.DataFrame([{k: v for k, v in metrics.items() if not isinstance(v, dict)}]).to_csv(
        results_dir / "analysis_results.csv",
        index=False,
    )
    (results_dir / "results.json").write_text(
        json.dumps(
            {
                "method": str(args.method),
                "base_embedding": str(embedding_path),
                "metrics": {k: v for k, v in metrics.items() if not isinstance(v, dict)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (config_dir / "config_used.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).expanduser().resolve()
    base_dir = output_dir / "_base"
    try:
        family, algo = BASE_ALGOS[str(args.method)]
        if family == "scrbenchmark":
            embedding_path, labels_path = run_scrbenchmark_base(args, algo, base_dir)
        else:
            embedding_path, labels_path = run_scvi_base(args, base_dir)
        write_outputs(args, output_dir, embedding_path, labels_path)
        failure_path = output_dir / "results" / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
    except Exception as exc:
        write_failure(output_dir / "results" / "failure.json", method=str(args.method), error=exc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
