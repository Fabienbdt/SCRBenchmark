#!/usr/bin/env python3
"""Run DeepScena on the stable_generalist benchmark datasets.

The output layout intentionally matches ``eval_trans_02_run_rare_cell_methods.py``
so that ``merge_external_rare_methods_stable_generalist.py`` can import the rows into the
presentation CSVs used by ``data_exploration.ipynb``.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
import torch
_TORCH_CUDA_WARMUP_AVAILABLE = torch.cuda.is_available()
import anndata as ad
import scanpy as sc
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, Dataset


RUNNER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get("SCRBENCHMARK_ROOT", Path(__file__).resolve().parents[3])).resolve()
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", REPO_ROOT.parent)).resolve()
SCRBENCH_ROOT = REPO_ROOT
SCRAW_SRC = SCRBENCH_ROOT / "vendor" / "scraw_dedicated" / "src"
DEEPSCENA_ROOT = SCRBENCH_ROOT / "external" / "original_code" / "DeepScena"
DEFAULT_MANIFEST = SCRBENCH_ROOT / "reproducibility" / "stable_generalist" / "stable_generalist_benchmark_13_manifest.csv"
DEFAULT_PRESENTATION_ROOT = SCRBENCH_ROOT / "results" / "stable_generalist_external"

if str(SCRAW_SRC) not in sys.path:
    sys.path.insert(0, str(SCRAW_SRC))
if str(DEEPSCENA_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPSCENA_ROOT))

from scraw_dedicated.metrics import align_labels, compute_metrics

from DeepScena import DeepScena as DeepScenaTrainer
from Network import AutoEncoder, Mutual_net, myBottleneck


LOGGER = logging.getLogger("run_deepscena_stable_generalist")

SCIB_COLUMNS = [
    "Batch correction",
    "Inter cell-type conservation",
    "Intra cell-type conservation",
    "scIB-E Total score",
]

PREPROCESSING = {
    "min_cells_per_gene": 3,
    "target_sum": 10000.0,
    "n_top_genes": 784,
    "hvg_flavor": "seurat",
    "pad_to_features": 784,
}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_key: str
    path: Path
    label_key: str
    batch_key: str
    family: str
    n_labels_expected: int


class DeepScenaDataset(Dataset):
    """Dataset returning DeepScena's expected 1 x 28 x 28 tensors."""

    def __init__(self, matrix_784: np.ndarray, label_codes: np.ndarray) -> None:
        matrix_784 = np.asarray(matrix_784, dtype=np.float32)
        if matrix_784.ndim != 2 or matrix_784.shape[1] != 784:
            raise ValueError(f"DeepScena expects an n_obs x 784 matrix, got {matrix_784.shape}")
        self.matrix = matrix_784.reshape(matrix_784.shape[0], 1, 28, 28)
        self.label_codes = np.asarray(label_codes, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.matrix[index]),
            torch.tensor(int(self.label_codes[index]), dtype=torch.long),
            torch.tensor(int(index), dtype=torch.long),
        )


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _to_dense(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if math.isnan(out) or math.isinf(out) else out
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(dict(payload)), indent=2), encoding="utf-8")


def _write_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(path, index=False)


def _read_manifest(path: Path) -> List[DatasetSpec]:
    specs: List[DatasetSpec] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("enabled", "true")).lower() not in {"true", "1", "yes"}:
                continue
            specs.append(
                DatasetSpec(
                    dataset_key=str(row["dataset_id"]).strip(),
                    path=Path(str(row["path"]).strip()),
                    label_key=str(row["label_key"]).strip(),
                    batch_key=str(row["batch_key"]).strip(),
                    family=str(row.get("family", "")).strip(),
                    n_labels_expected=int(row["n_labels_expected"]),
                )
            )
    return specs


def _select_datasets(specs: Sequence[DatasetSpec], raw: str) -> List[DatasetSpec]:
    tokens = {token.strip() for token in str(raw).split(",") if token.strip()}
    if not tokens:
        return list(specs)
    return [spec for spec in specs if spec.dataset_key in tokens]


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "dataset"


@contextmanager
def _pushd(path: Path) -> Iterable[None]:
    old = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _patch_torch_load_for_full_modules() -> None:
    """PyTorch 2.6+ defaults to weights_only=True; DeepScena saves modules."""

    original_load = torch.load

    def load_compat(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_compat  # type: ignore[assignment]


def weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias.data)
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight.data)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias.data)


def preprocess_deepscena(
    adata: ad.AnnData,
    *,
    label_key: str,
    batch_key: str,
    n_top_genes: int,
) -> Tuple[ad.AnnData, np.ndarray, Dict[str, Any]]:
    if label_key not in adata.obs.columns:
        raise KeyError(f"Missing label key '{label_key}'")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"Missing batch key '{batch_key}'")

    work = adata.copy()
    work.obs_names_make_unique()
    work.var_names_make_unique()

    stats: Dict[str, Any] = {
        "n_obs_input": int(work.n_obs),
        "n_vars_input": int(work.n_vars),
        "n_top_genes_requested": int(n_top_genes),
    }

    if "counts" not in work.layers:
        work.layers["counts"] = work.X.copy()

    sc.pp.filter_genes(work, min_cells=int(PREPROCESSING["min_cells_per_gene"]))
    stats["n_obs_after_filter_genes"] = int(work.n_obs)
    stats["n_vars_after_filter_genes"] = int(work.n_vars)
    if work.n_vars < 2:
        raise ValueError("Too few genes remain after DeepScena min_cells filtering.")

    work.raw = work.copy()
    sc.pp.normalize_total(work, target_sum=float(PREPROCESSING["target_sum"]))
    sc.pp.log1p(work)

    n_hvg = min(int(n_top_genes), int(work.n_vars))
    try:
        sc.pp.highly_variable_genes(
            work,
            n_top_genes=n_hvg,
            flavor=str(PREPROCESSING["hvg_flavor"]),
            subset=True,
        )
        stats["hvg_status"] = "scanpy_seurat"
    except Exception as exc:
        LOGGER.warning("Seurat HVG failed (%s); retrying with cell_ranger.", exc)
        try:
            sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor="cell_ranger", subset=True)
            stats["hvg_status"] = "scanpy_cell_ranger_fallback"
        except Exception as exc2:
            LOGGER.warning("Cell Ranger HVG failed (%s); using variance fallback.", exc2)
            dense = np.asarray(_to_dense(work.X), dtype=np.float32)
            variances = np.nanvar(dense, axis=0)
            keep_idx = np.argsort(-np.nan_to_num(variances, nan=-np.inf))[:n_hvg]
            work = work[:, keep_idx].copy()
            stats["hvg_status"] = "numpy_variance_fallback"

    X = np.asarray(_to_dense(work.X), dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    stats["n_vars_after_hvg"] = int(X.shape[1])

    if X.shape[1] > 784:
        X = X[:, :784]
        work = work[:, :784].copy()
        stats["feature_adjustment"] = "truncated_to_784"
    elif X.shape[1] < 784:
        pad_width = 784 - int(X.shape[1])
        X = np.pad(X, ((0, 0), (0, pad_width)), mode="constant")
        stats["feature_adjustment"] = f"zero_padded_{pad_width}_features"
    else:
        stats["feature_adjustment"] = "none"

    return work, np.ascontiguousarray(X, dtype=np.float32), stats


def _write_per_cell_labels(
    path: Path,
    adata_proc: ad.AnnData,
    label_key: str,
    batch_key: str,
    labels_pred: np.ndarray,
) -> None:
    true_labels = adata_proc.obs[label_key].astype(str).to_numpy()
    aligned = np.asarray(align_labels(true_labels, labels_pred), dtype=object).astype(str)
    batch_values = (
        adata_proc.obs[batch_key].astype(str).to_numpy()
        if batch_key in adata_proc.obs.columns
        else np.repeat("", adata_proc.n_obs)
    )
    pd.DataFrame(
        {
            "cell_index": np.arange(adata_proc.n_obs, dtype=int),
            "cell_id": adata_proc.obs_names.astype(str),
            "batch": batch_values,
            "true_label": true_labels,
            "predicted_label": np.asarray(labels_pred, dtype=object).astype(str),
            "aligned_predicted_label": aligned,
        }
    ).to_csv(path, index=False)


def save_successful_run(
    output_dir: Path,
    *,
    spec: DatasetSpec,
    adata_proc: ad.AnnData,
    preprocess_stats: Mapping[str, Any],
    labels_pred: np.ndarray,
    embedding: np.ndarray,
    runtime: float,
    method_info: Mapping[str, Any],
    scib_n_jobs: int,
    variant: str = "standard",
    display_name: str = "DeepScena",
) -> None:
    method_slug = "deepscena"
    results_dir = output_dir / "results"
    labels_dir = results_dir / "labels"
    embeddings_dir = results_dir / "embeddings"
    config_dir = output_dir / "config"
    labels_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    labels_true = adata_proc.obs[spec.label_key].astype(str).to_numpy()
    metrics = compute_metrics(
        labels_true=labels_true,
        labels_pred=np.asarray(labels_pred, dtype=object).astype(str),
        embeddings=np.asarray(embedding, dtype=np.float32),
        adata=adata_proc,
        batch_key=spec.batch_key,
        label_key=spec.label_key,
        compute_scib=True,
        scib_n_jobs=int(scib_n_jobs),
    )
    for column in SCIB_COLUMNS:
        metrics.setdefault(column, float("nan"))

    label_path = labels_dir / f"labels_{method_slug}_{variant}_run0.csv"
    embedding_path = embeddings_dir / f"embedding_{method_slug}_{variant}.npy"
    _write_per_cell_labels(label_path, adata_proc, spec.label_key, spec.batch_key, labels_pred)
    np.save(embedding_path, np.asarray(embedding, dtype=np.float32))

    result_row: Dict[str, Any] = {
        "algorithm": method_slug,
        "method_display_name": display_name,
        "method": display_name,
        "variant": variant,
        "run_id": 0,
        "runtime": float(runtime),
        "runtime_total": float(runtime),
        "implementation_status": "ok",
    }
    for key, value in metrics.items():
        if isinstance(value, dict):
            result_row[key] = json.dumps(_safe_json(value), sort_keys=True)
        else:
            result_row[key] = value
    _write_csv_row(results_dir / "analysis_results.csv", result_row)
    _write_csv_row(results_dir / "results.csv", result_row)

    config_payload = {
        "data": {"file": str(spec.path), "dataset_key": spec.dataset_key},
        "preprocessing": dict(PREPROCESSING),
        "context": {
            "label_key": spec.label_key,
            "batch_key": spec.batch_key,
            "n_labels_expected": int(spec.n_labels_expected),
            "preprocess_stats": dict(preprocess_stats),
        },
        "method_params": {
            "method": "DeepScena",
            "variant": variant,
            "display_name": display_name,
            "method_info": dict(method_info),
            "source_repository": "https://github.com/shaoqiangzhang/DeepScena",
            "source_root": str(DEEPSCENA_ROOT),
        },
        "output": {"directory": str(output_dir)},
    }
    _write_json(config_dir / "config_used.json", config_payload)

    results_payload = {
        "results": [
            {
                "algorithm_name": method_slug,
                "method_display_name": display_name,
                "run_id": 0,
                "runtime": float(runtime),
                "metrics": metrics,
                "params": config_payload["method_params"],
                "embeddings_shape": list(np.asarray(embedding).shape),
                "labels_csv": str(label_path),
                "embedding_npy": str(embedding_path),
            }
        ]
    }
    _write_json(results_dir / "results.json", results_payload)
    _write_json(
        results_dir / "summary.json",
        {
            "data_file": str(spec.path),
            "dataset_key": spec.dataset_key,
            "method": "DeepScena",
            "variant": variant,
            "method_display_name": display_name,
            "runtime_seconds": float(runtime),
            "best_metrics": metrics,
            "labels_csv": str(label_path),
            "embedding_npy": str(embedding_path),
        },
    )
    _write_json(output_dir / "run_status.json", {"status": "ok", "method": display_name, "variant": variant})


def save_failed_run(output_dir: Path, spec: DatasetSpec, exc: BaseException, variant: str = "standard") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "run_status.json",
        {
            "status": "failed",
            "method": "DeepScena",
            "variant": variant,
            "dataset_key": spec.dataset_key,
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )


def run_one_dataset(spec: DatasetSpec, output_root: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = output_root / "standard" / spec.dataset_key / "deepscena"
    analysis_csv = output_dir / "results" / "analysis_results.csv"
    if analysis_csv.exists() and not bool(args.overwrite):
        LOGGER.info("[skip] %s DeepScena already exists", spec.dataset_key)
        return {
            "dataset_key": spec.dataset_key,
            "method": "DeepScena",
            "variant": "standard",
            "status": "ok",
            "error": "",
            "output_dir": str(output_dir),
            "skipped_existing": True,
        }

    LOGGER.info("[run] dataset=%s method=DeepScena", spec.dataset_key)
    LOGGER.info(
        "CUDA before loading data: available=%s count=%s visible=%r",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    start = time.time()
    model_dir = output_dir / "models" / "deepscena"
    log_dir = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(spec.path)
    LOGGER.info("CUDA after read_h5ad: available=%s count=%s", torch.cuda.is_available(), torch.cuda.device_count())
    adata_proc, matrix_784, preprocess_stats = preprocess_deepscena(
        adata,
        label_key=spec.label_key,
        batch_key=spec.batch_key,
        n_top_genes=int(args.n_top_genes),
    )
    LOGGER.info("CUDA after preprocessing: available=%s count=%s", torch.cuda.is_available(), torch.cuda.device_count())
    labels_true = adata_proc.obs[spec.label_key].astype(str).to_numpy()
    label_codes = pd.Categorical(labels_true).codes.astype(np.int64)

    train_dataset = DeepScenaDataset(matrix_784, label_codes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    LOGGER.info("CUDA after DataLoader init: available=%s count=%s", torch.cuda.is_available(), torch.cuda.device_count())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DeepScena original code requires CUDA, but torch.cuda.is_available() is False. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}, "
            f"torch_cuda={torch.version.cuda!r}, device_count={torch.cuda.device_count()}"
        )
    torch.backends.cudnn.benchmark = True
    _patch_torch_load_for_full_modules()

    dataset_name = _slug(spec.dataset_key)
    with _pushd(model_dir):
        ae = AutoEncoder(myBottleneck, [1, 1, 1]).cuda()
        ae.apply(weights_init)
        mnet = Mutual_net(int(spec.n_labels_expected)).cuda()
        mnet.apply(weights_init)

        trainer = DeepScenaTrainer(
            ae,
            mnet,
            train_loader,
            int(adata_proc.n_obs),
            batch_size=int(args.batch_size),
            pretraining_epoch=int(args.pretraining_epoch),
            MaxIter1=int(args.maxiter1),
            MaxIter2=int(args.maxiter2),
            num_cluster=int(spec.n_labels_expected),
            m=float(args.fuzzy_m),
            T1=int(args.t1),
            T2=int(args.t2),
            latent_size=10,
            zeta=float(args.zeta),
            gamma=float(1.0 - float(args.zeta)),
            dataset_name=dataset_name,
            a=float(args.a),
        )

        if int(args.pretraining_epoch) != 0:
            trainer.pretrain()
        if int(args.maxiter1) != 0:
            trainer.first_module()
        if int(args.maxiter2) != 0:
            trainer.second_module()

        eval_loader = DataLoader(
            train_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        trainer.AE.eval()
        trainer.MNet.eval()
        embedding = np.zeros((adata_proc.n_obs, 10), dtype=np.float32)
        labels_pred = np.empty(adata_proc.n_obs, dtype=object)
        with torch.no_grad():
            for x, _target, index in eval_loader:
                x = x.cuda(non_blocking=True)
                _mean, _disp, u, _recon = trainer.AE(x)
                q = trainer.MNet(u)
                pred = torch.argmax(q, dim=1)
                idx = index.cpu().numpy().astype(int)
                embedding[idx] = u.detach().cpu().numpy().astype(np.float32)
                labels_pred[idx] = pred.detach().cpu().numpy().astype(str)

    runtime = float(time.time() - start)
    method_info = {
        "implementation": "original_DeepScena_code_wrapped_for_stable_generalist",
        "deep_scena_root": str(DEEPSCENA_ROOT),
        "batch_size": int(args.batch_size),
        "pretraining_epoch": int(args.pretraining_epoch),
        "MaxIter1": int(args.maxiter1),
        "MaxIter2": int(args.maxiter2),
        "T1": int(args.t1),
        "T2": int(args.t2),
        "latent_size": 10,
        "num_cluster": int(spec.n_labels_expected),
        "fuzzy_m": float(args.fuzzy_m),
        "zeta": float(args.zeta),
        "gamma": float(1.0 - float(args.zeta)),
        "a": float(args.a),
        "cuda_device_count_visible": int(torch.cuda.device_count()),
        "cuda_device_name": torch.cuda.get_device_name(0),
    }
    save_successful_run(
        output_dir,
        spec=spec,
        adata_proc=adata_proc,
        preprocess_stats=preprocess_stats,
        labels_pred=np.asarray(labels_pred, dtype=object).astype(str),
        embedding=embedding,
        runtime=runtime,
        method_info=method_info,
        scib_n_jobs=int(args.scib_n_jobs),
    )
    return {
        "dataset_key": spec.dataset_key,
        "method": "DeepScena",
        "variant": "standard",
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
    return DEFAULT_PRESENTATION_ROOT / f"external_rare_methods_deepscena_{stamp}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-top-genes", type=int, default=784)
    parser.add_argument("--pretraining-epoch", type=int, default=0)
    parser.add_argument("--maxiter1", type=int, default=20)
    parser.add_argument("--maxiter2", type=int, default=20)
    parser.add_argument("--t1", type=int, default=2)
    parser.add_argument("--t2", type=int, default=1)
    parser.add_argument("--fuzzy-m", type=float, default=1.5)
    parser.add_argument("--zeta", type=float, default=0.8)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--scib-n-jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    _setup_logging(bool(args.verbose))
    _seed_everything(int(args.seed))
    LOGGER.info(
        "CUDA after seeding: available=%s count=%s visible=%r torch_cuda=%r",
        torch.cuda.is_available(),
        torch.cuda.device_count(),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        torch.version.cuda,
    )

    output_root = _output_root(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    specs = _select_datasets(_read_manifest(Path(args.manifest)), str(args.datasets))

    _write_json(
        output_root / "run_metadata.json",
        {
            "timestamp": datetime.now().isoformat(),
            "manifest": str(Path(args.manifest).resolve()),
            "output_root": str(output_root),
            "method": "DeepScena",
            "variant": "standard",
            "seed": int(args.seed),
            "preprocessing": dict(PREPROCESSING),
            "n_datasets": len(specs),
            "scib_n_jobs": int(args.scib_n_jobs),
            "deep_scena_root": str(DEEPSCENA_ROOT),
        },
    )

    run_rows: List[Dict[str, Any]] = []
    overall_status = "ok"
    for spec in specs:
        try:
            row = run_one_dataset(spec, output_root, args)
            run_rows.append(row)
            LOGGER.info("Finished dataset=%s status=%s", spec.dataset_key, row.get("status"))
        except Exception as exc:
            overall_status = "failed"
            output_dir = output_root / "standard" / spec.dataset_key / "deepscena"
            save_failed_run(output_dir, spec, exc)
            LOGGER.exception("Failed dataset=%s method=DeepScena", spec.dataset_key)
            run_rows.append(
                {
                    "dataset_key": spec.dataset_key,
                    "method": "DeepScena",
                    "variant": "standard",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "output_dir": str(output_dir),
                }
            )
            if not bool(args.continue_on_error):
                break
        finally:
            pd.DataFrame(run_rows).to_csv(output_root / "run_status.csv", index=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
