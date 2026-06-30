"""Inductive train/test utilities for scRAW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import csv
import json
import numpy as np
import torch

from .clustering import sanitize_embeddings
from .config import ScRAWConfig, load_config
from .metrics import compute_metrics
from .model import encode_in_batches
from .pipeline import (
    _as_jsonable,
    _detect_batch_key,
    _detect_label_key,
    _load_checkpoint_model,
    _prepare_output_dirs,
    _save_metrics_csv,
)
from .plots import (
    plot_embedding_categories,
    plot_loss_history,
    plot_train_test_metrics,
    save_figure,
)
from .preprocessing import (
    PreprocessingState,
    fit_preprocess_adata,
    load_preprocessing_state,
    save_preprocessing_state,
    transform_adata_with_state,
)
from .presets import resolve_preset_config
from .trainer import ScRAWTrainer


@dataclass
class CentroidReference:
    """Frozen nearest-centroid classifier in scRAW latent space."""

    labels: np.ndarray
    centroids: np.ndarray


def _copy_config(config: ScRAWConfig) -> ScRAWConfig:
    return ScRAWConfig.from_dict(config.to_dict())


def _coerce_config(config: ScRAWConfig | str | Path) -> ScRAWConfig:
    if isinstance(config, ScRAWConfig):
        return _copy_config(config)

    raw = str(config)
    if raw.strip().lower() in {"default", "baron"}:
        return resolve_preset_config(raw)

    return load_config(Path(raw))


def _preset_output_name(config: ScRAWConfig | str | Path) -> Optional[str]:
    if isinstance(config, ScRAWConfig):
        return None
    preset_name = str(config).strip().lower()
    if preset_name == "default":
        return "default"
    if preset_name == "baron":
        return "baron"
    return None


def _repo_root_from_data_path(config: ScRAWConfig) -> Path:
    data_path = Path(config.data.data_path).expanduser().resolve()
    if data_path.parent.name == "data":
        return data_path.parent.parent
    return Path.cwd().resolve()


def _as_str_array(values: Iterable[Any]) -> np.ndarray:
    return np.asarray([str(value) for value in values], dtype=object)


def _labels_from_obs(adata: Any, label_key: Optional[str]) -> Optional[np.ndarray]:
    if label_key is None:
        return None
    return np.asarray(adata.obs[label_key].astype(str).to_numpy(), dtype=object)


def fit_centroid_reference(embeddings: np.ndarray, labels: np.ndarray) -> CentroidReference:
    """Fit one centroid per frozen training cluster."""
    emb = sanitize_embeddings(embeddings)
    labels = np.asarray(labels, dtype=np.int64)
    if emb.shape[0] != labels.shape[0]:
        raise ValueError("Embeddings and labels must contain the same number of rows.")
    if emb.shape[0] == 0:
        raise ValueError("Cannot fit a centroid reference on an empty embedding matrix.")

    unique_labels = np.asarray(sorted(np.unique(labels).tolist()), dtype=np.int64)
    centroids = np.asarray(
        [np.mean(emb[labels == label], axis=0) for label in unique_labels],
        dtype=np.float32,
    )
    return CentroidReference(labels=unique_labels, centroids=centroids)


def predict_nearest_centroid(
    embeddings: np.ndarray,
    reference: CentroidReference,
) -> np.ndarray:
    """Assign embeddings to the nearest frozen training centroid."""
    emb = sanitize_embeddings(embeddings)
    centroids = sanitize_embeddings(reference.centroids)
    if centroids.ndim != 2 or centroids.shape[0] == 0:
        raise ValueError("Centroid reference is empty.")
    if emb.ndim != 2 or emb.shape[1] != centroids.shape[1]:
        raise ValueError("Embedding dimensionality does not match the centroid reference.")

    distances = np.sum((emb[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.asarray(reference.labels[np.argmin(distances, axis=1)], dtype=np.int64)


def save_centroid_reference(reference: CentroidReference, path: str | Path) -> None:
    """Persist a centroid reference to an NPZ file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        labels=np.asarray(reference.labels, dtype=np.int64),
        centroids=np.asarray(reference.centroids, dtype=np.float32),
    )


def load_centroid_reference(path: str | Path) -> CentroidReference:
    """Load a centroid reference from an NPZ file."""
    with np.load(Path(path), allow_pickle=False) as payload:
        return CentroidReference(
            labels=np.asarray(payload["labels"], dtype=np.int64),
            centroids=np.asarray(payload["centroids"], dtype=np.float32),
        )


def _save_inductive_arrays(
    output_dir: Path,
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    train_pseudo_labels: np.ndarray,
    train_cell_weights: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    train_obs_names: np.ndarray,
    test_obs_names: np.ndarray,
    train_true_labels: Optional[np.ndarray],
    test_true_labels: Optional[np.ndarray],
) -> None:
    np.save(output_dir / "train_embeddings.npy", np.asarray(train_embeddings, dtype=np.float32))
    np.save(output_dir / "train_labels.npy", np.asarray(train_labels, dtype=np.int64))
    np.save(output_dir / "train_pseudo_labels.npy", np.asarray(train_pseudo_labels, dtype=np.int64))
    np.save(output_dir / "train_cell_weights.npy", np.asarray(train_cell_weights, dtype=np.float32))
    np.save(output_dir / "test_embeddings.npy", np.asarray(test_embeddings, dtype=np.float32))
    np.save(output_dir / "test_pred_labels.npy", np.asarray(test_labels, dtype=np.int64))
    np.save(output_dir / "train_obs_names.npy", np.asarray(train_obs_names, dtype=object))
    np.save(output_dir / "test_obs_names.npy", np.asarray(test_obs_names, dtype=object))
    if train_true_labels is not None:
        np.save(output_dir / "train_true_labels.npy", np.asarray(train_true_labels, dtype=object))
    if test_true_labels is not None:
        np.save(output_dir / "test_true_labels.npy", np.asarray(test_true_labels, dtype=object))


def _save_loss_history_csv(loss_history: list[dict[str, Any]], path: str | Path) -> None:
    if not loss_history:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "phase",
        "total_loss",
        "reconstruction_loss",
        "triplet_loss",
        "batch_adv_loss",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in loss_history:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _save_inductive_figures(
    output_dir: Path,
    *,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    train_true_labels: Optional[np.ndarray],
    test_true_labels: Optional[np.ndarray],
    loss_history: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    seed: int,
) -> None:
    save_figure(plot_loss_history(loss_history), output_dir / "loss_history.png")
    save_figure(
        plot_train_test_metrics(train_metrics, test_metrics),
        output_dir / "train_vs_test_metrics.png",
    )
    save_figure(
        plot_embedding_categories(
            train_embeddings,
            train_labels,
            title="scRAW train latent space colored by frozen clusters",
            random_state=seed,
        ),
        output_dir / "train_latent_clusters.png",
    )
    save_figure(
        plot_embedding_categories(
            test_embeddings,
            test_labels,
            title="scRAW held-out latent space colored by centroid predictions",
            random_state=seed,
        ),
        output_dir / "test_latent_predictions.png",
    )
    if train_true_labels is not None:
        save_figure(
            plot_embedding_categories(
                train_embeddings,
                train_true_labels,
                title="scRAW train latent space colored by ground-truth labels",
                random_state=seed,
            ),
            output_dir / "train_latent_ground_truth.png",
        )
    if test_true_labels is not None:
        save_figure(
            plot_embedding_categories(
                test_embeddings,
                test_true_labels,
                title="scRAW held-out latent space colored by ground-truth labels",
                random_state=seed,
            ),
            output_dir / "test_latent_ground_truth.png",
        )


def run_inductive_baron_split(
    config: ScRAWConfig | str | Path = "default",
    *,
    train_batches: Iterable[str] = ("human1", "human2", "human3"),
    test_batches: Iterable[str] = ("human4",),
    split_key: str = "batch",
    output_dir: str | Path | None = None,
    device: Optional[str] = None,
    filter_inference_cells: bool = True,
) -> dict[str, Any]:
    """Train on selected Baron humans and predict held-out humans without retraining."""
    config_obj = _coerce_config(config)
    if device is not None:
        config_obj.runtime.device = str(device)
    if output_dir is not None:
        config_obj.data.output_dir = str(output_dir)
    else:
        preset_name = _preset_output_name(config)
        if preset_name is not None:
            config_obj.data.output_dir = str(
                _repo_root_from_data_path(config_obj)
                / "results"
                / f"inductive_baron_{preset_name}"
            )

    resolved_output_dir = Path(config_obj.data.output_dir).expanduser().resolve()
    output_paths = _prepare_output_dirs(resolved_output_dir)

    import scanpy as sc

    adata = sc.read_h5ad(Path(config_obj.data.data_path).expanduser().resolve())
    if split_key not in adata.obs.columns:
        raise ValueError(f"Split key {split_key!r} was not found in adata.obs.")

    train_values = {str(value) for value in train_batches}
    test_values = {str(value) for value in test_batches}
    split_values = adata.obs[split_key].astype(str)
    train_mask = split_values.isin(train_values).to_numpy()
    test_mask = split_values.isin(test_values).to_numpy()
    if not bool(np.any(train_mask)):
        raise ValueError("No training cells matched the requested train_batches.")
    if not bool(np.any(test_mask)):
        raise ValueError("No held-out cells matched the requested test_batches.")

    adata_train_raw = adata[train_mask].copy()
    adata_test_raw = adata[test_mask].copy()
    adata_train, preprocessing_state = fit_preprocess_adata(
        adata_train_raw,
        config_obj.preprocessing,
    )
    adata_test = transform_adata_with_state(
        adata_test_raw,
        preprocessing_state,
        filter_cells=filter_inference_cells,
    )

    label_key = _detect_label_key(adata_train, config_obj.data.label_key)
    batch_key = _detect_batch_key(adata_train, preferred=split_key)
    train_true_labels = _labels_from_obs(adata_train, label_key)
    test_label_key = label_key if label_key is not None and label_key in adata_test.obs else None
    test_true_labels = _labels_from_obs(adata_test, test_label_key)
    train_batch_ids = (
        None
        if batch_key is None
        else np.asarray(adata_train.obs[batch_key].astype(str).to_numpy(), dtype=object)
    )

    X_train = np.asarray(adata_train.X, dtype=np.float32)
    X_test = np.asarray(adata_test.X, dtype=np.float32)
    trainer = ScRAWTrainer(config_obj)
    result = trainer.fit(X_train, labels=train_true_labels, batch_ids=train_batch_ids)
    reference = fit_centroid_reference(result.embeddings, result.labels)
    test_embeddings = encode_in_batches(
        result.model,
        X_test,
        device=trainer.device,
        batch_size=int(config_obj.training.batch_size),
    )
    test_pred_labels = predict_nearest_centroid(test_embeddings, reference)

    train_metrics = compute_metrics(
        labels_true=train_true_labels,
        labels_pred=result.labels,
        embeddings=result.embeddings,
    )
    test_metrics = compute_metrics(
        labels_true=test_true_labels,
        labels_pred=test_pred_labels,
        embeddings=test_embeddings,
    )
    config_used = config_obj.to_dict()
    summary = {
        "mode": "inductive_baron_split",
        "split_key": split_key,
        "train_batches": sorted(train_values),
        "test_batches": sorted(test_values),
        "filter_inference_cells": bool(filter_inference_cells),
        "label_key": label_key,
        "batch_key": batch_key,
        "n_train_cells": int(adata_train.n_obs),
        "n_test_cells": int(adata_test.n_obs),
        "n_genes": int(adata_train.n_vars),
        "device": str(trainer.device),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "loss_history": result.loss_history,
        "assignment": "nearest_train_cluster_centroid",
    }

    (output_paths["config"] / "config_used.json").write_text(
        json.dumps(_as_jsonable(config_used), indent=2),
        encoding="utf-8",
    )
    (output_paths["results"] / "results.json").write_text(
        json.dumps(_as_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    _save_metrics_csv(train_metrics, output_paths["results"] / "train_analysis_results.csv")
    _save_metrics_csv(test_metrics, output_paths["results"] / "test_analysis_results.csv")
    _save_loss_history_csv(result.loss_history, output_paths["results"] / "loss_history.csv")
    _save_inductive_arrays(
        output_paths["results"],
        train_embeddings=result.embeddings,
        train_labels=result.labels,
        train_pseudo_labels=result.pseudo_labels,
        train_cell_weights=result.cell_weights,
        test_embeddings=test_embeddings,
        test_labels=test_pred_labels,
        train_obs_names=_as_str_array(adata_train.obs_names),
        test_obs_names=_as_str_array(adata_test.obs_names),
        train_true_labels=train_true_labels,
        test_true_labels=test_true_labels,
    )
    save_preprocessing_state(
        preprocessing_state,
        output_paths["models"] / "preprocessing_state.npz",
    )
    save_centroid_reference(reference, output_paths["models"] / "centroid_reference.npz")

    if bool(config_obj.outputs.save_model):
        torch.save(result.model.state_dict(), output_paths["models"] / "autoencoder.pt")

    if bool(config_obj.outputs.save_figures):
        _save_inductive_figures(
            output_paths["figures"],
            train_embeddings=result.embeddings,
            train_labels=result.labels,
            test_embeddings=test_embeddings,
            test_labels=test_pred_labels,
            train_true_labels=train_true_labels,
            test_true_labels=test_true_labels,
            loss_history=result.loss_history,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            seed=int(config_obj.runtime.seed),
        )

    return {
        "config": config_used,
        "mode": "inductive_baron_split",
        "label_key": label_key,
        "batch_key": batch_key,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_embeddings": result.embeddings,
        "train_labels": result.labels,
        "test_embeddings": test_embeddings,
        "test_labels": test_pred_labels,
        "output_dir": str(resolved_output_dir),
    }


def run_inductive_prediction(
    config: ScRAWConfig | str | Path,
    *,
    checkpoint_path: str | Path,
    preprocessing_state_path: str | Path,
    centroid_reference_path: str | Path,
    data_path: str | Path,
    output_dir: str | Path,
    device: Optional[str] = None,
    label_key: Optional[str] = None,
    filter_cells: bool = True,
) -> dict[str, Any]:
    """Predict a new AnnData object using only frozen inductive artifacts."""
    config_obj = _coerce_config(config)
    config_obj.data.data_path = str(data_path)
    config_obj.data.output_dir = str(output_dir)
    if device is not None:
        config_obj.runtime.device = str(device)

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    output_paths = _prepare_output_dirs(resolved_output_dir)

    import scanpy as sc

    preprocessing_state = load_preprocessing_state(preprocessing_state_path)
    reference = load_centroid_reference(centroid_reference_path)
    adata = sc.read_h5ad(Path(data_path).expanduser().resolve())
    adata_proc = transform_adata_with_state(
        adata,
        preprocessing_state,
        filter_cells=filter_cells,
    )
    resolved_label_key = _detect_label_key(adata_proc, label_key or config_obj.data.label_key)
    true_labels = _labels_from_obs(adata_proc, resolved_label_key)
    X_proc = np.asarray(adata_proc.X, dtype=np.float32)

    trainer = ScRAWTrainer(config_obj)
    trainer._set_random_seeds()
    model = _load_checkpoint_model(
        checkpoint_path=checkpoint_path,
        input_dim=int(X_proc.shape[1]),
        config=config_obj,
        device=trainer.device,
    )
    embeddings = encode_in_batches(
        model,
        X_proc,
        device=trainer.device,
        batch_size=int(config_obj.training.batch_size),
    )
    labels = predict_nearest_centroid(embeddings, reference)
    metrics = compute_metrics(labels_true=true_labels, labels_pred=labels, embeddings=embeddings)

    summary = {
        "mode": "inductive_prediction",
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "preprocessing_state_path": str(Path(preprocessing_state_path).expanduser().resolve()),
        "centroid_reference_path": str(Path(centroid_reference_path).expanduser().resolve()),
        "label_key": resolved_label_key,
        "n_cells": int(adata_proc.n_obs),
        "n_genes": int(adata_proc.n_vars),
        "device": str(trainer.device),
        "metrics": metrics,
        "assignment": "nearest_train_cluster_centroid",
    }
    (output_paths["config"] / "config_used.json").write_text(
        json.dumps(_as_jsonable(config_obj.to_dict()), indent=2),
        encoding="utf-8",
    )
    (output_paths["results"] / "results.json").write_text(
        json.dumps(_as_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    _save_metrics_csv(metrics, output_paths["results"] / "analysis_results.csv")
    np.save(output_paths["results"] / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    np.save(output_paths["results"] / "pred_labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(output_paths["results"] / "obs_names.npy", _as_str_array(adata_proc.obs_names))
    if true_labels is not None:
        np.save(output_paths["results"] / "true_labels.npy", np.asarray(true_labels, dtype=object))

    if bool(config_obj.outputs.save_figures):
        save_figure(
            plot_embedding_categories(
                embeddings,
                labels,
                title="scRAW new observations colored by centroid predictions",
                random_state=int(config_obj.runtime.seed),
            ),
            output_paths["figures"] / "latent_predictions.png",
        )
        if true_labels is not None:
            save_figure(
                plot_embedding_categories(
                    embeddings,
                    true_labels,
                    title="scRAW new observations colored by ground-truth labels",
                    random_state=int(config_obj.runtime.seed),
                ),
                output_paths["figures"] / "latent_ground_truth.png",
            )

    return {
        "config": config_obj.to_dict(),
        "mode": "inductive_prediction",
        "metrics": metrics,
        "embeddings": embeddings,
        "labels": labels,
        "output_dir": str(resolved_output_dir),
    }
