#!/usr/bin/env python3
"""Run the report representative inductive protocol for one dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys
import subprocess
import time
import traceback
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAW_INDUCTIVE_ROOT = REPO_ROOT / "vendor" / "scraw_inductive"
SCRAW_INDUCTIVE_SRC = SCRAW_INDUCTIVE_ROOT / "src"
SCRBENCHMARK_SRC = REPO_ROOT / "src"
SCRBENCHMARK_PACKAGE_SRC = SCRBENCHMARK_SRC / "scrbenchmark"

for path in (SCRAW_INDUCTIVE_SRC, SCRBENCHMARK_SRC, SCRBENCHMARK_PACKAGE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scraw.inductive import fit_centroid_reference, predict_nearest_centroid  # noqa: E402
from scraw.metrics import NOISE_LABELS, align_labels, compute_metrics  # noqa: E402
from scraw.model import encode_in_batches  # noqa: E402
from scraw.plots import plot_loss_history, save_figure  # noqa: E402
from scraw.preprocessing import fit_preprocess_adata, save_preprocessing_state, transform_adata_with_state  # noqa: E402
from scraw.presets import resolve_preset_config  # noqa: E402
from scraw.trainer import ScRAWTrainer  # noqa: E402

from algorithms.sc_mae import ScMaeAlgorithm  # noqa: E402
from algorithms.scdeepcluster import ScDeepClusterAlgorithm  # noqa: E402
from algorithms.scname import ScNAMEAlgorithm  # noqa: E402


ALGORITHM_CLASSES = {
    "scname": ScNAMEAlgorithm,
    "sc_mae": ScMaeAlgorithm,
    "scdeepcluster": ScDeepClusterAlgorithm,
}

SCAIDE_PYTHON = Path(os.environ.get("SCAIDE_PYTHON", sys.executable))
SCAIDE_INDUCTIVE_SCRIPT = REPO_ROOT / "scripts" / "reproduction" / "run_scaide_inductive_embeddings.py"

SUMMARY_FIELDS = [
    "dataset_name",
    "algorithm",
    "preset",
    "split_key",
    "train_batches",
    "test_batch",
    "status",
    "device_requested",
    "device_used",
    "n_train",
    "n_test",
    "n_genes",
    "ACC",
    "ARI",
    "NMI",
    "RareACC",
    "UltraRareACC",
    "elapsed_sec",
    "error",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-key", default="donor")
    parser.add_argument(
        "--train-split-key",
        default=None,
        help="Column used to select/train batch ids. Defaults to --split-key.",
    )
    parser.add_argument(
        "--test-split-key",
        default=None,
        help="Column used to select held-out groups. Defaults to --split-key.",
    )
    parser.add_argument("--label-key", required=True)
    parser.add_argument("--train-batches", nargs="+", required=True)
    parser.add_argument("--test-batches", nargs="+", required=True)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["scraw", "scname", "sc_mae", "scdeepcluster"],
        choices=["scraw", "scname", "sc_mae", "scdeepcluster", "scaide", "pca_harmony"],
    )
    parser.add_argument("--preset", default="stable_generalist", choices=["default", "0017", "stable_generalist"])
    parser.add_argument(
        "--trial-config-path",
        default=None,
        help="Exact trial_config.json path used when resolving preset stable_generalist/0017.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-top-genes", type=int, default=2000)
    parser.add_argument(
        "--baseline-runtime-profile",
        choices=["scrbenchmark-default", "debug-fast"],
        default="scrbenchmark-default",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help=(
            "Manual algorithm override. Use algo:key=value, e.g. "
            "sc_mae:epochs=40 or pca_harmony:n_pcs=30. "
            "For scRAW, dotted config paths are accepted, e.g. scraw:training.epochs=80."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def require_cuda_if_requested(device: str) -> None:
    if str(device).lower() != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")


def read_adata(path: str):
    import scanpy as sc

    return sc.read_h5ad(path)


def _parse_value(raw: str) -> Any:
    text = str(raw).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        if any(char in text for char in [".", "e", "E"]):
            return float(text)
        return int(text)
    except Exception:
        return text


def _manual_params_for(args: argparse.Namespace, algorithm: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for raw in args.param or []:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text:
            raise ValueError(f"Invalid --param {raw!r}; expected algo:key=value or key=value.")
        left, value = text.split("=", 1)
        if ":" in left:
            target, key = left.split(":", 1)
            target = target.strip().lower()
            if target not in {algorithm.lower(), "*", "all"}:
                continue
        else:
            key = left
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --param {raw!r}; empty parameter name.")
        params[key] = _parse_value(value)
    return params


def _set_dotted_attr(root: Any, dotted_key: str, value: Any) -> bool:
    parts = [part for part in str(dotted_key).split(".") if part]
    if not parts:
        return False
    target = root
    for part in parts[:-1]:
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        return False
    setattr(target, parts[-1], value)
    return True


def apply_scraw_manual_params(config: Any, args: argparse.Namespace) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    search_sections = [
        "training",
        "model",
        "preprocessing",
        "runtime",
        "weighting",
        "triplet",
        "batch",
        "clustering",
        "outputs",
    ]
    for key, value in _manual_params_for(args, "scraw").items():
        done = False
        if "." in key:
            done = _set_dotted_attr(config, key, value)
        else:
            for section in search_sections:
                target = getattr(config, section, None)
                if target is not None and hasattr(target, key):
                    setattr(target, key, value)
                    done = True
                    break
        if not done:
            raise ValueError(
                f"Unknown scRAW config override {key!r}. Use a valid flat parameter "
                "or a dotted path such as training.epochs=80."
            )
        applied[key] = value
    return applied


def labels_for(adata, label_key: str) -> np.ndarray:
    return adata.obs[label_key].astype(str).to_numpy()


def dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _obs_first_embedding(embedding: Any, n_obs: int, context: str) -> np.ndarray:
    arr = dense_float32(embedding)
    if arr.ndim != 2:
        raise ValueError(f"{context} returned an embedding with ndim={arr.ndim}, expected 2.")
    if arr.shape[0] == int(n_obs):
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.shape[1] == int(n_obs):
        return np.ascontiguousarray(arr.T, dtype=np.float32)
    raise ValueError(f"{context} returned shape {arr.shape}; expected one dimension to match n_obs={n_obs}.")


def _raw_matrix_for_hvg(data: Any) -> np.ndarray:
    if hasattr(data, "layers") and "original_X" in data.layers:
        return dense_float32(data.layers["original_X"])
    return dense_float32(data.X if hasattr(data, "X") else data)


def _log_normalized_hvg_matrix(data: Any, target_sum: float = 1e4) -> np.ndarray:
    x = _raw_matrix_for_hvg(data).astype(np.float32, copy=True)
    totals = x.sum(axis=1, keepdims=True)
    totals[~np.isfinite(totals)] = 0.0
    totals[totals <= 0.0] = 1.0
    x *= float(target_sum)
    x /= totals
    np.log1p(x, out=x)
    x[~np.isfinite(x)] = 0.0
    return x


class PCAHarmonyInductiveAlgorithm:
    """Train-only PCA+Harmony reference with a linear out-of-sample projection."""

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}
        self.scaler = None
        self.pca = None
        self.projector = None
        self.clusterer = None
        self._embeddings = None
        self._labels = None
        self._fitted = False

    def _encode_pca(self, data: Any) -> np.ndarray:
        x = _log_normalized_hvg_matrix(data)
        x_scaled = self.scaler.transform(x)
        x_pca = self.pca.transform(x_scaled)
        if self.projector is not None:
            x_pca = self.projector.predict(x_pca)
        return np.asarray(x_pca, dtype=np.float32)

    def fit(self, data: Any, labels: Any | None = None) -> "PCAHarmonyInductiveAlgorithm":
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        seed = int(self.params.get("random_state", 42))
        n_clusters = int(self.params.get("n_clusters", 0))
        if n_clusters <= 0:
            n_clusters = len(np.unique(labels)) if labels is not None else 8

        x = _log_normalized_hvg_matrix(data)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        x_scaled = self.scaler.fit_transform(x)

        n_pcs = int(self.params.get("n_pcs", 50))
        n_pcs = max(2, min(n_pcs, int(x_scaled.shape[0]) - 1, int(x_scaled.shape[1]) - 1))
        self.pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=seed)
        train_pca = self.pca.fit_transform(x_scaled).astype(np.float32, copy=False)

        train_embedding = train_pca
        batch_key = str(self.params.get("batch_key", "batch"))
        harmony_applied = False
        if hasattr(data, "obs") and batch_key in data.obs.columns:
            batch_values = data.obs[batch_key].astype(str)
            if int(batch_values.nunique()) >= 2:
                import harmonypy as hm

                ho = hm.run_harmony(
                    np.ascontiguousarray(train_pca, dtype=np.float32),
                    data.obs[[batch_key]].copy(),
                    vars_use=[batch_key],
                    max_iter_harmony=int(self.params.get("harmony_max_iter", 10)),
                    nclust=int(self.params.get("harmony_nclust", 50)),
                )
                train_embedding = _obs_first_embedding(ho.Z_corr, data.n_obs, "Harmony")
                harmony_applied = True

        if harmony_applied:
            self.projector = Ridge(alpha=float(self.params.get("projection_alpha", 1e-3)))
            self.projector.fit(train_pca, train_embedding)

        self.clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
        self._labels = self.clusterer.fit_predict(train_embedding)
        self._embeddings = np.asarray(train_embedding, dtype=np.float32)
        self._fitted = True
        return self

    def predict(self, data: Any = None) -> Any:
        if not self._fitted:
            raise RuntimeError("Algorithm must be fitted before prediction")
        if data is None:
            return self._labels
        return self.clusterer.predict(self._encode_pca(data))

    def encode(self, data: Any) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Algorithm must be fitted before encoding")
        return self._encode_pca(data)

    def get_embeddings(self) -> Any:
        return self._embeddings


ALGORITHM_CLASSES["pca_harmony"] = PCAHarmonyInductiveAlgorithm


def make_raw_hvg_view(raw_adata, obs_names, var_names):
    view = raw_adata[obs_names, list(var_names)].copy()
    view.layers["original_X"] = view.X.astype(np.float32).copy()
    return view


def prepare_common_space(args: argparse.Namespace, adata):
    if args.trial_config_path:
        config = resolve_preset_config(
            args.preset,
            repo_root=SCRAW_INDUCTIVE_ROOT,
            stable_generalist_config_path=Path(args.trial_config_path).expanduser().resolve(),
        )
    else:
        config = resolve_preset_config(args.preset, repo_root=SCRAW_INDUCTIVE_ROOT)
    config.data.data_path = str(Path(args.data_path).expanduser().resolve())
    config.data.label_key = args.label_key
    config.runtime.device = args.device
    config.runtime.seed = int(args.seed)
    config.outputs.save_figures = False
    config.outputs.save_model = True
    config.preprocessing.n_top_genes = int(args.n_top_genes)
    manual_scraw_params = apply_scraw_manual_params(config, args)

    train_split_key = args.train_split_key or args.split_key
    test_split_key = args.test_split_key or args.split_key
    train_values = adata.obs[train_split_key].astype(str)
    test_values = adata.obs[test_split_key].astype(str)
    train_mask = train_values.isin(set(args.train_batches)).to_numpy()
    if not bool(np.any(train_mask)):
        raise ValueError("No training cells matched --train-batches.")

    train_raw = adata[train_mask].copy()
    train_proc, preprocessing_state = fit_preprocess_adata(train_raw, config.preprocessing)
    train_raw_hvg = make_raw_hvg_view(train_raw, train_proc.obs_names, preprocessing_state.var_names)

    tests = {}
    for test_batch in args.test_batches:
        test_mask = test_values.to_numpy() == str(test_batch)
        if not bool(np.any(test_mask)):
            raise ValueError(f"No test cells matched {test_batch!r}.")
        test_raw = adata[test_mask].copy()
        test_proc = transform_adata_with_state(test_raw, preprocessing_state, filter_cells=True)
        test_raw_hvg = make_raw_hvg_view(test_raw, test_proc.obs_names, preprocessing_state.var_names)
        tests[str(test_batch)] = {
            "proc": test_proc,
            "raw_hvg": test_raw_hvg,
            "labels": labels_for(test_proc, args.label_key),
        }

    return {
        "config": config,
        "preprocessing_state": preprocessing_state,
        "train_proc": train_proc,
        "train_raw_hvg": train_raw_hvg,
        "train_labels": labels_for(train_proc, args.label_key),
        "train_batch_ids": train_proc.obs[train_split_key].astype(str).to_numpy(),
        "tests": tests,
        "train_split_key": train_split_key,
        "test_split_key": test_split_key,
        "manual_scraw_params": manual_scraw_params,
    }


def algorithm_params(name: str, n_clusters: int, args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {
        "device": args.device,
        "random_state": args.seed,
        "n_clusters": int(n_clusters),
        "input_type": "raw_filtered",
        "use_raw_data": True,
    }
    if name == "sc_mae":
        params.update({"use_own_preprocessing": True, "n_hvg": 1000})
    elif name == "scdeepcluster":
        params.update({"select_genes": 0, "use_ground_truth_k": False})
    elif name == "scname":
        params.update({"map_class": False})
    elif name == "pca_harmony":
        params.update(
            {
                "batch_key": args.train_split_key or args.split_key,
                "n_pcs": 50,
                "harmony_max_iter": 10,
                "harmony_nclust": 50,
                "projection_alpha": 1e-3,
            }
        )
    if args.baseline_runtime_profile == "debug-fast":
        if name == "sc_mae":
            params.update({"epochs": 5, "eval_epoch": 4})
        elif name == "scdeepcluster":
            params.update({"pretrain_epochs": 5, "maxiter": 5, "update_interval": 1})
        elif name == "scname":
            params.update({"pretrain_epochs": 5, "finetune_epochs": 5, "update_interval": 5})
        elif name == "pca_harmony":
            params.update({"n_pcs": 10, "harmony_max_iter": 2, "harmony_nclust": 10})
    params.update(_manual_params_for(args, name))
    return params


def valid_subset(algo: Any, labels: np.ndarray, pred: np.ndarray, embeddings: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    labels = np.asarray(labels)
    pred = np.asarray(pred)
    if len(labels) == len(pred):
        return labels, pred, embeddings

    idx = None
    if hasattr(algo, "get_valid_indices"):
        try:
            idx = algo.get_valid_indices()
        except Exception:
            idx = None
    if idx is None and hasattr(algo, "_valid_indices"):
        idx = getattr(algo, "_valid_indices")
    if idx is None and hasattr(algo, "_filtered_index"):
        idx = getattr(algo, "_filtered_index")

    if idx is not None:
        idx = np.asarray(idx, dtype=int)
        if len(idx) == len(pred) and idx.max(initial=-1) < len(labels):
            labels = labels[idx]
            if embeddings is not None and len(embeddings) != len(pred):
                embeddings = None
            return labels, pred, embeddings

    n = min(len(labels), len(pred))
    embeddings = embeddings[:n] if embeddings is not None and len(embeddings) >= n else None
    return labels[:n], pred[:n], embeddings


def ultra_rare_acc(labels_true: np.ndarray, labels_pred: np.ndarray, threshold: float = 0.01) -> float:
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
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
    rare_classes = classes[(counts / len(labels_true)) < threshold]
    if len(rare_classes) == 0:
        return float("nan")
    rare_mask = np.isin(labels_true, rare_classes)
    return float(np.mean(aligned[rare_mask] == labels_true[rare_mask]))


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def save_loss_history_csv(loss_history: list[dict[str, Any]], path: Path) -> None:
    if not loss_history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "phase",
        "total_loss",
        "reconstruction_loss",
        "triplet_loss",
        "batch_adv_loss",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in loss_history:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def device_used(algo: Any, fallback: str = "unknown") -> str:
    try:
        if getattr(algo, "model", None) is not None:
            model = algo.model
            if hasattr(model, "parameters"):
                return str(next(model.parameters()).device)
            if hasattr(model, "device"):
                return str(model.device)
    except Exception:
        pass
    return fallback


def save_metric_bundle(out_dir: Path, payload: dict[str, Any], pred: np.ndarray, emb: Any) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(jsonify(payload), indent=2), encoding="utf-8")
    np.save(out_dir / "pred_labels.npy", np.asarray(pred))
    if emb is not None:
        np.save(out_dir / "embeddings.npy", np.asarray(emb, dtype=np.float32))


def add_test_rows(
    *,
    summary_path: Path,
    dataset_name: str,
    algorithm: str,
    preset: str,
    split_key: str,
    train_batches: list[str],
    test_batch: str,
    status: str,
    device_requested: str,
    device: str,
    n_train: int,
    n_genes: int,
    metrics: dict[str, Any] | None,
    n_test: int,
    elapsed: float,
    output_dir: Path,
    error: str = "",
) -> None:
    metrics = metrics or {}
    row = {
        "dataset_name": dataset_name,
        "algorithm": algorithm,
        "preset": preset,
        "split_key": split_key,
        "train_batches": ",".join(train_batches),
        "test_batch": test_batch,
        "status": status,
        "device_requested": device_requested,
        "device_used": device,
        "n_train": n_train,
        "n_test": n_test,
        "n_genes": n_genes,
        "ACC": metrics.get("ACC", ""),
        "ARI": metrics.get("ARI", ""),
        "NMI": metrics.get("NMI", ""),
        "RareACC": metrics.get("RareACC", ""),
        "UltraRareACC": metrics.get("UltraRareACC", ""),
        "elapsed_sec": elapsed,
        "error": error,
        "output_dir": str(output_dir),
    }
    write_summary(summary_path, row)


def run_scraw(common: dict[str, Any], args: argparse.Namespace, output_root: Path, summary_path: Path) -> None:
    start = time.time()
    alg_dir = output_root / "scraw"
    if args.skip_existing and all((alg_dir / batch / "metrics.json").exists() for batch in args.test_batches):
        print("[scraw] skip existing", flush=True)
        return

    config = common["config"]
    trainer = ScRAWTrainer(config)
    result = trainer.fit(
        dense_float32(common["train_proc"].X),
        labels=common["train_labels"],
        batch_ids=common["train_batch_ids"],
    )
    reference = fit_centroid_reference(result.embeddings, result.labels)

    import torch

    artifacts_dir = alg_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "config_used.json").write_text(
        json.dumps(jsonify(config.to_dict()), indent=2),
        encoding="utf-8",
    )
    if args.trial_config_path:
        trial_config_path = Path(args.trial_config_path).expanduser().resolve()
        (artifacts_dir / "trial_config_source.json").write_text(
            trial_config_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    save_loss_history_csv(result.loss_history, artifacts_dir / "loss_history.csv")
    save_figure(plot_loss_history(result.loss_history), artifacts_dir / "loss_history.png")
    train_metrics = compute_metrics(common["train_labels"], result.labels, result.embeddings)
    (artifacts_dir / "train_metrics.json").write_text(
        json.dumps(jsonify(train_metrics), indent=2),
        encoding="utf-8",
    )
    np.save(artifacts_dir / "train_embeddings.npy", np.asarray(result.embeddings, dtype=np.float32))
    np.save(artifacts_dir / "train_labels.npy", np.asarray(result.labels, dtype=np.int64))
    np.save(artifacts_dir / "train_cell_weights.npy", np.asarray(result.cell_weights, dtype=np.float32))
    torch.save(result.model.state_dict(), artifacts_dir / "autoencoder.pt")
    save_preprocessing_state(common["preprocessing_state"], artifacts_dir / "preprocessing_state.npz")
    np.savez_compressed(
        artifacts_dir / "centroid_reference.npz",
        labels=np.asarray(reference.labels, dtype=np.int64),
        centroids=np.asarray(reference.centroids, dtype=np.float32),
    )

    for test_batch, test in common["tests"].items():
        test_start = time.time()
        test_embeddings = encode_in_batches(
            result.model,
            dense_float32(test["proc"].X),
            device=trainer.device,
            batch_size=int(config.training.batch_size),
        )
        pred = predict_nearest_centroid(test_embeddings, reference)
        metrics = compute_metrics(test["labels"], pred, test_embeddings)
        metrics["UltraRareACC"] = ultra_rare_acc(test["labels"], pred)
        out_dir = alg_dir / test_batch
        payload = {
            "dataset_name": args.dataset_name,
            "algorithm": "scraw",
            "preset": args.preset,
            "trial_config_path": args.trial_config_path,
            "manual_params": common.get("manual_scraw_params", {}),
            "train_batches": args.train_batches,
            "test_batch": test_batch,
            "metrics": metrics,
            "n_train": int(common["train_proc"].n_obs),
            "n_test": int(test["proc"].n_obs),
            "n_genes": int(common["train_proc"].n_vars),
            "device": str(trainer.device),
            "elapsed_total_sec": float(time.time() - start),
            "elapsed_test_sec": float(time.time() - test_start),
        }
        save_metric_bundle(out_dir, payload, pred, test_embeddings)
        add_test_rows(
            summary_path=summary_path,
            dataset_name=args.dataset_name,
            algorithm="scraw",
            preset=args.preset,
            split_key=args.split_key,
            train_batches=args.train_batches,
            test_batch=test_batch,
            status="ok",
            device_requested=args.device,
            device=str(trainer.device),
            n_train=int(common["train_proc"].n_obs),
            n_genes=int(common["train_proc"].n_vars),
            metrics=metrics,
            n_test=int(test["proc"].n_obs),
            elapsed=float(time.time() - start),
            output_dir=out_dir,
        )
        print(f"[scraw {test_batch}] ACC={metrics['ACC']:.4f} ARI={metrics['ARI']:.4f}", flush=True)


def safe_slug(value: str) -> str:
    out = []
    for char in str(value):
        out.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(out) or "value"


def run_scaide_inductive(
    common: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    summary_path: Path,
) -> None:
    from sklearn.cluster import KMeans

    start = time.time()
    name = "scaide"
    alg_dir = output_root / name
    if args.skip_existing and all((alg_dir / batch / "metrics.json").exists() for batch in args.test_batches):
        print("[scaide] skip existing", flush=True)
        return

    artifacts_dir = alg_dir / "artifacts"
    inputs_dir = artifacts_dir / "inputs"
    embeddings_dir = artifacts_dir / "embeddings"
    logs_dir = alg_dir / "logs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_input = inputs_dir / "train_scaled_hvg.npy"
    train_embedding_path = embeddings_dir / "train_embedding.npy"
    np.save(train_input, dense_float32(common["train_proc"].X))

    test_input_paths: dict[str, Path] = {}
    test_embedding_paths: dict[str, Path] = {}
    for test_batch, test in common["tests"].items():
        slug = safe_slug(test_batch)
        test_input_paths[test_batch] = inputs_dir / f"test_{slug}_scaled_hvg.npy"
        test_embedding_paths[test_batch] = embeddings_dir / f"test_{slug}_embedding.npy"
        np.save(test_input_paths[test_batch], dense_float32(test["proc"].X))

    cmd = [
        str(SCAIDE_PYTHON),
        str(SCAIDE_INDUCTIVE_SCRIPT),
        "--train-input-npy",
        str(train_input),
        "--train-output-npy",
        str(train_embedding_path),
        "--save-folder",
        str(artifacts_dir / "aide_model"),
        "--name",
        f"scaide_inductive_{safe_slug(args.dataset_name)}",
        "--seed",
        str(int(args.seed)),
    ]
    if args.baseline_runtime_profile == "debug-fast":
        cmd.append("--fast-smoke")
    for test_batch in common["tests"]:
        cmd.extend(["--test-input", f"{safe_slug(test_batch)}={test_input_paths[test_batch]}"])
        cmd.extend(["--test-output", f"{safe_slug(test_batch)}={test_embedding_paths[test_batch]}"])

    log_file = logs_dir / "scaide_inductive_embedding.log"
    env = os.environ.copy()
    cache_root = alg_dir / ".cache"
    env["MPLCONFIGDIR"] = str(cache_root / "mpl")
    env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
    env["PYTHONUNBUFFERED"] = "1"
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    try:
        with log_file.open("w", encoding="utf-8") as handle:
            proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"scAIDE/AIDE embedding failed with code {proc.returncode}; log={log_file}")

        train_embedding = _obs_first_embedding(
            np.load(train_embedding_path),
            common["train_proc"].n_obs,
            "AIDE train",
        )
        n_clusters = len(np.unique(common["train_labels"]))
        clusterer = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(args.seed))
        train_pred = clusterer.fit_predict(train_embedding)
        train_metrics = compute_metrics(common["train_labels"], train_pred, train_embedding)
        train_metrics["UltraRareACC"] = ultra_rare_acc(common["train_labels"], train_pred)
        (artifacts_dir / "train_metrics.json").write_text(
            json.dumps(jsonify(train_metrics), indent=2),
            encoding="utf-8",
        )
        np.save(artifacts_dir / "train_pred_labels.npy", np.asarray(train_pred))

        for test_batch, test in common["tests"].items():
            test_start = time.time()
            slug = safe_slug(test_batch)
            test_embedding = _obs_first_embedding(
                np.load(test_embedding_paths[test_batch]),
                test["proc"].n_obs,
                f"AIDE test {test_batch}",
            )
            pred = clusterer.predict(test_embedding)
            metrics = compute_metrics(test["labels"], pred, test_embedding)
            metrics["UltraRareACC"] = ultra_rare_acc(test["labels"], pred)
            out_dir = alg_dir / test_batch
            payload = {
                "dataset_name": args.dataset_name,
                "algorithm": name,
                "params": {
                    "embedding_backend": "AIDE_local_source_train_session_project_test",
                    "clusterer": "KMeans_on_train_AIDE_embedding",
                    "n_clusters": int(n_clusters),
                    "input": "scRAW_train_fitted_scaled_HVG_matrix",
                    "test_embedding_file": str(test_embedding_paths[test_batch]),
                    "test_slug": slug,
                },
                "train_batches": args.train_batches,
                "test_batch": test_batch,
                "metrics": metrics,
                "n_train": int(common["train_proc"].n_obs),
                "n_test": int(test["proc"].n_obs),
                "n_genes": int(common["train_proc"].n_vars),
                "device": "scAIDE_external_tf1",
                "elapsed_total_sec": float(time.time() - start),
                "elapsed_test_sec": float(time.time() - test_start),
            }
            save_metric_bundle(out_dir, payload, pred, test_embedding)
            add_test_rows(
                summary_path=summary_path,
                dataset_name=args.dataset_name,
                algorithm=name,
                preset=args.preset,
                split_key=args.split_key,
                train_batches=args.train_batches,
                test_batch=test_batch,
                status="ok",
                device_requested=args.device,
                device="scAIDE_external_tf1",
                n_train=int(common["train_proc"].n_obs),
                n_genes=int(common["train_proc"].n_vars),
                metrics=metrics,
                n_test=int(test["proc"].n_obs),
                elapsed=float(time.time() - start),
                output_dir=out_dir,
            )
            print(f"[scaide {test_batch}] ACC={metrics['ACC']:.4f} ARI={metrics['ARI']:.4f}", flush=True)
    except Exception as exc:
        for test_batch, test in common["tests"].items():
            out_dir = alg_dir / test_batch
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.json").write_text(
                json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8",
            )
            add_test_rows(
                summary_path=summary_path,
                dataset_name=args.dataset_name,
                algorithm=name,
                preset=args.preset,
                split_key=args.split_key,
                train_batches=args.train_batches,
                test_batch=test_batch,
                status="error",
                device_requested=args.device,
                device="scAIDE_external_tf1",
                n_train=int(common["train_proc"].n_obs),
                n_genes=int(common["train_proc"].n_vars),
                metrics=None,
                n_test=int(test["proc"].n_obs),
                elapsed=float(time.time() - start),
                output_dir=out_dir,
                error=repr(exc),
            )
            print(f"[scaide {test_batch}] ERROR {exc!r}", flush=True)


def run_baseline(
    name: str,
    common: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
    summary_path: Path,
) -> None:
    start = time.time()
    alg_dir = output_root / name
    if args.skip_existing and all((alg_dir / batch / "metrics.json").exists() for batch in args.test_batches):
        print(f"[{name}] skip existing", flush=True)
        return

    params = algorithm_params(name, len(np.unique(common["train_labels"])), args)
    algo = ALGORITHM_CLASSES[name](params=params)
    print(f"[{name}] fit params={params}", flush=True)
    algo.fit(common["train_raw_hvg"], common["train_labels"])

    for test_batch, test in common["tests"].items():
        test_start = time.time()
        try:
            pred = np.asarray(algo.predict(test["raw_hvg"]))
            try:
                emb = algo.encode(test["raw_hvg"])
            except Exception as exc:
                print(f"[{name} {test_batch}] encode failed: {exc}", flush=True)
                emb = None
            eval_labels, eval_pred, emb = valid_subset(algo, test["labels"], pred, emb)
            metrics = compute_metrics(eval_labels, eval_pred, emb)
            metrics["UltraRareACC"] = ultra_rare_acc(eval_labels, eval_pred)
            out_dir = alg_dir / test_batch
            payload = {
                "dataset_name": args.dataset_name,
                "algorithm": name,
                "params": params,
                "train_batches": args.train_batches,
                "test_batch": test_batch,
                "metrics": metrics,
                "n_train": int(common["train_raw_hvg"].n_obs),
                "n_test": int(test["raw_hvg"].n_obs),
                "n_test_eval": int(len(eval_labels)),
                "n_genes": int(common["train_raw_hvg"].n_vars),
                "device": device_used(algo, fallback=args.device),
                "elapsed_total_sec": float(time.time() - start),
                "elapsed_test_sec": float(time.time() - test_start),
            }
            save_metric_bundle(out_dir, payload, pred, emb)
            add_test_rows(
                summary_path=summary_path,
                dataset_name=args.dataset_name,
                algorithm=name,
                preset=args.preset,
            split_key=args.split_key,
                train_batches=args.train_batches,
                test_batch=test_batch,
                status="ok",
                device_requested=args.device,
                device=device_used(algo, fallback=args.device),
                n_train=int(common["train_raw_hvg"].n_obs),
                n_genes=int(common["train_raw_hvg"].n_vars),
                metrics=metrics,
                n_test=int(test["raw_hvg"].n_obs),
                elapsed=float(time.time() - start),
                output_dir=out_dir,
            )
            print(f"[{name} {test_batch}] ACC={metrics['ACC']:.4f} ARI={metrics['ARI']:.4f}", flush=True)
        except Exception as exc:
            out_dir = alg_dir / test_batch
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "error.json").write_text(
                json.dumps({"error": repr(exc), "traceback": traceback.format_exc()}, indent=2),
                encoding="utf-8",
            )
            add_test_rows(
                summary_path=summary_path,
                dataset_name=args.dataset_name,
                algorithm=name,
                preset=args.preset,
                split_key=args.split_key,
                train_batches=args.train_batches,
                test_batch=test_batch,
                status="error",
                device_requested=args.device,
                device=device_used(algo, fallback=args.device),
                n_train=int(common["train_raw_hvg"].n_obs),
                n_genes=int(common["train_raw_hvg"].n_vars),
                metrics=None,
                n_test=int(test["raw_hvg"].n_obs),
                elapsed=float(time.time() - start),
                output_dir=out_dir,
                error=repr(exc),
            )
            print(f"[{name} {test_batch}] ERROR {exc!r}", flush=True)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    require_cuda_if_requested(args.device)

    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"

    print(f"dataset={args.dataset_name}", flush=True)
    print(f"data_path={args.data_path}", flush=True)
    print(
        f"split_key={args.split_key} train_split_key={args.train_split_key or args.split_key} "
        f"test_split_key={args.test_split_key or args.split_key} label_key={args.label_key}",
        flush=True,
    )
    print(f"train_batches={args.train_batches}", flush=True)
    print(f"test_batches={args.test_batches}", flush=True)
    print(f"algorithms={args.algorithms}", flush=True)
    print(f"output_root={output_root}", flush=True)

    adata = read_adata(args.data_path)
    common = prepare_common_space(args, adata)
    print(
        f"common_space train={common['train_proc'].n_obs} genes={common['train_proc'].n_vars} "
        f"tests={[(k, v['proc'].n_obs) for k, v in common['tests'].items()]}",
        flush=True,
    )

    for algorithm in args.algorithms:
        if algorithm == "scraw":
            run_scraw(common, args, output_root, summary_path)
        elif algorithm == "scaide":
            run_scaide_inductive(common, args, output_root, summary_path)
        else:
            run_baseline(algorithm, common, args, output_root, summary_path)

    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
