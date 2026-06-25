"""Preset helpers for reproducible scRAW experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import ScRAWConfig, load_config
from .model import parse_hidden_layers


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


STABLE_GENERALIST_CONFIG = Path(
    os.environ.get(
        "SCRAW_STABLE_GENERALIST_CONFIG",
        _repo_root() / "configs" / "stable_generalist_trial_0017.json",
    )
)


def _set_if_present(target: Any, attr: str, overrides: dict[str, Any], key: str) -> None:
    if key in overrides:
        setattr(target, attr, overrides[key])


def _apply_stable_generalist_overrides(config: ScRAWConfig, trial_config_path: Path) -> ScRAWConfig:
    payload = json.loads(trial_config_path.read_text(encoding="utf-8"))
    overrides = dict(payload.get("param_overrides", {}))

    if "hidden_layers" in overrides:
        config.model.hidden_layers = parse_hidden_layers(overrides["hidden_layers"])
    if "z_dim" in overrides:
        config.model.latent_dim = int(overrides["z_dim"])
    _set_if_present(config.model, "dropout", overrides, "dropout")

    _set_if_present(config.training, "epochs", overrides, "epochs")
    _set_if_present(config.training, "warmup_epochs", overrides, "warmup_epochs")
    _set_if_present(config.training, "batch_size", overrides, "batch_size")
    if "lr" in overrides:
        config.training.learning_rate = float(overrides["lr"])
    _set_if_present(
        config.training,
        "reconstruction_distribution",
        overrides,
        "reconstruction_distribution",
    )
    _set_if_present(config.training, "nb_input_transform", overrides, "nb_input_transform")
    _set_if_present(config.training, "nb_theta", overrides, "nb_theta")
    _set_if_present(config.training, "masking_rate", overrides, "masking_rate")
    _set_if_present(config.training, "masked_recon_weight", overrides, "masked_recon_weight")
    if "masking_apply_weighted" in overrides:
        config.training.masking_in_weighted_phase = bool(overrides["masking_apply_weighted"])

    _set_if_present(config.weighting, "weight_fusion_mode", overrides, "weight_fusion_mode")
    _set_if_present(config.weighting, "cluster_weight_power", overrides, "cluster_weight_power")
    _set_if_present(config.weighting, "density_weight_power", overrides, "density_weight_power")
    _set_if_present(config.weighting, "weight_exponent", overrides, "weight_exponent")
    _set_if_present(config.weighting, "cluster_density_alpha", overrides, "cluster_density_alpha")
    _set_if_present(config.weighting, "density_knn_k", overrides, "density_knn_k")
    _set_if_present(config.weighting, "density_weight_clip", overrides, "density_weight_clip")
    _set_if_present(
        config.weighting,
        "dynamic_weight_update_interval",
        overrides,
        "dynamic_weight_update_interval",
    )
    _set_if_present(config.weighting, "dynamic_weight_momentum", overrides, "dynamic_weight_momentum")
    _set_if_present(config.weighting, "min_cell_weight", overrides, "min_cell_weight")
    _set_if_present(config.weighting, "max_cell_weight", overrides, "max_cell_weight")

    if "rare_triplet_weight" in overrides:
        config.triplet.weight = float(overrides["rare_triplet_weight"])
        config.triplet.enabled = config.triplet.weight > 0.0
    if "rare_triplet_start_epoch" in overrides:
        config.triplet.start_epoch = int(overrides["rare_triplet_start_epoch"])
    if "rare_triplet_margin" in overrides:
        config.triplet.margin = float(overrides["rare_triplet_margin"])
    if "rare_triplet_min_weight" in overrides:
        config.triplet.min_anchor_weight = float(overrides["rare_triplet_min_weight"])
    if "max_triplet_anchors_per_batch" in overrides:
        config.triplet.max_anchors_per_batch = int(overrides["max_triplet_anchors_per_batch"])

    _set_if_present(config.clustering, "pseudo_label_method", overrides, "pseudo_label_method")
    _set_if_present(config.clustering, "hdbscan_min_cluster_size", overrides, "hdbscan_min_cluster_size")
    _set_if_present(config.clustering, "hdbscan_min_samples", overrides, "hdbscan_min_samples")
    _set_if_present(
        config.clustering,
        "hdbscan_cluster_selection_method",
        overrides,
        "hdbscan_cluster_selection_method",
    )
    _set_if_present(config.clustering, "hdbscan_reassign_noise", overrides, "hdbscan_reassign_noise")

    if "use_batch_conditioning" in overrides:
        config.batch_correction.enabled = bool(overrides["use_batch_conditioning"])
    if "adversarial_batch_weight" in overrides:
        config.batch_correction.adversarial_weight = float(overrides["adversarial_batch_weight"])
    if "adversarial_lambda" in overrides:
        config.batch_correction.adversarial_lambda = float(overrides["adversarial_lambda"])
    if "adversarial_start_epoch" in overrides:
        config.batch_correction.start_epoch = int(overrides["adversarial_start_epoch"])
    if "adversarial_ramp_epochs" in overrides:
        config.batch_correction.ramp_epochs = int(overrides["adversarial_ramp_epochs"])
    if "mmd_batch_weight" in overrides:
        config.batch_correction.mmd_weight = float(overrides["mmd_batch_weight"])

    return config


def resolve_preset_config(
    preset: str,
    *,
    repo_root: str | Path | None = None,
    stable_generalist_config_path: str | Path = STABLE_GENERALIST_CONFIG,
) -> ScRAWConfig:
    """Return a ScRAWConfig for `default` or trial `0017`."""
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else _repo_root()
    preset_name = str(preset or "default").strip().lower()
    config = load_config(root / "configs" / "default_scraw.json")
    config.data.data_path = str(root / "data" / "baron_human_pancreas.h5ad")

    if preset_name in {"default", "default_scraw"}:
        return config

    if preset_name in {"0017", "17", "stable_generalist", "stable_generalist_stable_generalist"}:
        config = _apply_stable_generalist_overrides(
            config,
            trial_config_path=Path(stable_generalist_config_path).expanduser().resolve(),
        )
        return config

    raise ValueError("Unknown preset. Expected one of: default, stable_generalist.")
