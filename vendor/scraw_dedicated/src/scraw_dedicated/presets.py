#!/usr/bin/env python3
"""Strict scRAW presets.

Includes the default tuned standalone configuration plus legacy Baron/Pancreas
reference presets.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict

from .defaults import (
    DEFAULT_PRESET_NAME,
    copy_baron_algorithm_params,
    copy_baron_preprocessing,
    copy_default_algorithm_params,
    copy_default_preprocessing,
)


@dataclass(frozen=True)
class ScrawPreset:
    name: str
    description: str
    preprocessing: Dict[str, Any]
    algorithm_params: Dict[str, Any]
    supports_dann: bool


_COMMON_PREPROCESSING: Dict[str, Any] = {
    "n_top_genes": 2000,
    "min_cells_per_gene": 3,
    "target_sum": 20000,
    "scale_max_value": 10.0,
    "hvg_flavor": "seurat",
    "hvg_strategy": "train_only",
    "dropout_method": "none",
    "noise_level": 0.0,
}

_COMMON_ALGO: Dict[str, Any] = {
    "input_type": "processed",
    "hidden_layers": "512,256,128",
    "z_dim": 128,
    "masking_rate": 0.2,
    "masked_recon_weight": 0.75,
    "masking_value": 0.0,
    "masking_apply_weighted": False,
    "reconstruction_distribution": "nb",
    "nb_theta": 10,
    "nb_mu_clip_max": 1e6,
    "weight_exponent": 0.4,
    "weight_fusion_mode": "additive",
    "cluster_weight_power": 1.0,
    "density_weight_power": 1.0,
    "cluster_density_alpha": 0.6,
    "density_weight_clip": 5.0,
    "clustering_method": "hdbscan",
    "hdbscan_min_cluster_size": 4,
    "hdbscan_min_samples": 2,
    "hdbscan_cluster_selection_method": "eom",
    "hdbscan_reassign_noise": True,
    "pseudo_label_method": "leiden",
    "density_knn_k": 15,
    "density_weight_exponent": 1.0,
    "n_clusters": 0,
    "unsupervised_k_fallback": 0,
    "unsupervised_k_selection": "stability_consensus",
    "unsupervised_k_min": 8,
    "unsupervised_k_max": 30,
    "unsupervised_k_num_candidates": 12,
    "unsupervised_k_pca_dim": 32,
    "unsupervised_k_eval_sample_size": 3000,
    "unsupervised_k_stability_runs": 5,
    "unsupervised_k_stability_sample_size": 4000,
    "unsupervised_k_weight_stability": 0.45,
    "unsupervised_k_weight_silhouette": 0.25,
    "unsupervised_k_weight_ch": 0.20,
    "unsupervised_k_weight_db": 0.10,
    "unsupervised_k_weight_tiny_clusters": 0.20,
    "unsupervised_k_min_cluster_fraction": 0.005,
    "unsupervised_k_overseg_penalty": 0.25,
    "unsupervised_k_underseg_penalty": 0.05,
    "random_state": 42,
    "seed": 42,
    "rare_loss_type": "triplet",
    "rare_triplet_margin": 0.4,
    "rare_triplet_min_weight": 1.2,
    "max_triplet_anchors_per_batch": 64,
    "min_cell_weight": 0.25,
}


def _merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Return `base` updated with `update`, without mutating the input dict."""
    out = deepcopy(base)
    out.update(update)
    return out


def _default_variant_preset(
    *,
    name: str,
    description: str,
    param_overrides: Dict[str, Any],
) -> ScrawPreset:
    """Build a default-derived preset with a compact declaration."""
    return ScrawPreset(
        name=name,
        description=description,
        preprocessing=copy_default_preprocessing(),
        algorithm_params=_merge(copy_default_algorithm_params(), param_overrides),
        supports_dann=True,
    )


_DEFAULT_V_2 = _default_variant_preset(
    name="default_v_2",
    description=(
        "Consensus fixed configuration distilled from the most stable stage-1 "
        "generalist trials (notably trial_0060, trial_0029, trial_0026, "
        "trial_0074 and trial_0057)."
    ),
    param_overrides={
        "hidden_layers": "512,256",
        "z_dim": 192,
        "dropout": 0.2,
        "epochs": 80,
        "warmup_epochs": 74,
        "lr": 0.0010689964738835216,
        "batch_size": 192,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "nb_theta": 11.623582708841571,
        "masking_rate": 0.1,
        "masked_recon_weight": 0.8,
        "masking_apply_weighted": True,
        "weight_exponent": 0.1,
        "cluster_density_alpha": 0.5135545572653624,
        "density_knn_k": 15,
        "density_weight_clip": 8.0,
        "dynamic_weight_momentum": 0.7966299232499916,
        "dynamic_weight_update_interval": 10,
        "weight_fusion_mode": "additive",
        "min_cell_weight": 0.39770069412392606,
        "max_cell_weight": 8.0,
        "rare_triplet_weight": 0.1928616915439295,
        "rare_triplet_start_epoch": 60,
        "rare_triplet_margin": 0.4,
        "rare_triplet_min_weight": 1.0,
        "max_triplet_anchors_per_batch": 128,
        "pseudo_label_method": "leiden",
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 8,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_reassign_noise": False,
        "use_batch_conditioning": True,
        "adversarial_batch_weight": 0.06848282044120413,
        "adversarial_lambda": 0.5,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 55,
        "mmd_batch_weight": 0.05,
        "capture_embedding_snapshots": False,
        "batch_correction_key": "batch",
    },
)

_DEFAULT_V_3 = _default_variant_preset(
    name="default_v_3",
    description=(
        "Batch-heavy stable challenger anchored on trial_0060. This family was "
        "the closest fixed alternative to `default` on non-Baron batch-heavy "
        "datasets and plate/FACS/microfluidic cohorts."
    ),
    param_overrides={
        "hidden_layers": "512,256",
        "z_dim": 256,
        "dropout": 0.3,
        "epochs": 80,
        "warmup_epochs": 74,
        "lr": 0.0010882018879642033,
        "batch_size": 384,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "nb_theta": 20.03038299445062,
        "masking_rate": 0.1,
        "masked_recon_weight": 0.8,
        "masking_apply_weighted": True,
        "weight_exponent": 0.1,
        "cluster_density_alpha": 0.5898109200409181,
        "density_knn_k": 15,
        "density_weight_clip": 3.0,
        "dynamic_weight_momentum": 0.7966299232499916,
        "dynamic_weight_update_interval": 20,
        "weight_fusion_mode": "additive",
        "min_cell_weight": 0.4321784952556028,
        "max_cell_weight": 10.0,
        "rare_triplet_weight": 0.2005030550668264,
        "rare_triplet_start_epoch": 60,
        "rare_triplet_margin": 0.4,
        "rare_triplet_min_weight": 1.0,
        "max_triplet_anchors_per_batch": 128,
        "pseudo_label_method": "leiden",
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 8,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_reassign_noise": False,
        "use_batch_conditioning": True,
        "adversarial_batch_weight": 0.059106192930047874,
        "adversarial_lambda": 0.5,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 55,
        "mmd_batch_weight": 0.05,
        "capture_embedding_snapshots": False,
        "batch_correction_key": "batch",
    },
)

_DEFAULT_V_4 = _default_variant_preset(
    name="default_v_4",
    description=(
        "Large-droplet / no-ultra-rare variant anchored on trial_0027. It keeps "
        "the longer 210-epoch family that helped on Kang PBMC-like 10x runs."
    ),
    param_overrides={
        "hidden_layers": "512,256,128",
        "z_dim": 192,
        "dropout": 0.25,
        "epochs": 210,
        "warmup_epochs": 36,
        "lr": 0.0020623018431402975,
        "batch_size": 384,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "nb_theta": 15.394639396275059,
        "masking_rate": 0.1,
        "masked_recon_weight": 0.8,
        "masking_apply_weighted": True,
        "weight_exponent": 0.2,
        "cluster_density_alpha": 0.400225331671659,
        "density_knn_k": 15,
        "density_weight_clip": 8.0,
        "dynamic_weight_momentum": 0.7969584881178149,
        "dynamic_weight_update_interval": 20,
        "weight_fusion_mode": "additive",
        "min_cell_weight": 0.4013887962875491,
        "max_cell_weight": 8.0,
        "rare_triplet_weight": 0.15455130673181883,
        "rare_triplet_start_epoch": 60,
        "rare_triplet_margin": 0.4,
        "rare_triplet_min_weight": 1.8,
        "max_triplet_anchors_per_batch": 128,
        "pseudo_label_method": "leiden",
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 8,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_reassign_noise": True,
        "use_batch_conditioning": True,
        "adversarial_batch_weight": 0.0673220897350643,
        "adversarial_lambda": 1.75,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 60,
        "mmd_batch_weight": 0.0,
        "capture_embedding_snapshots": False,
        "batch_correction_key": "batch",
    },
)

_DEFAULT_V_5 = _default_variant_preset(
    name="default_v_5",
    description=(
        "Rare-focused variant anchored on trial_0023. It preserves the compact "
        "80-epoch family but adds more rare-class pressure than `default_v_2`."
    ),
    param_overrides={
        "hidden_layers": "512,256,128",
        "z_dim": 192,
        "dropout": 0.2,
        "epochs": 80,
        "warmup_epochs": 74,
        "lr": 0.0017421921548481738,
        "batch_size": 192,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "nb_theta": 6.441398035240585,
        "masking_rate": 0.1,
        "masked_recon_weight": 0.8,
        "masking_apply_weighted": True,
        "weight_exponent": 0.1,
        "cluster_density_alpha": 0.4267657491217307,
        "density_knn_k": 15,
        "density_weight_clip": 8.0,
        "dynamic_weight_momentum": 0.7771617976907875,
        "dynamic_weight_update_interval": 10,
        "weight_fusion_mode": "additive",
        "min_cell_weight": 0.4495550197183071,
        "max_cell_weight": 8.0,
        "rare_triplet_weight": 0.1973239259371167,
        "rare_triplet_start_epoch": 60,
        "rare_triplet_margin": 0.4,
        "rare_triplet_min_weight": 1.0,
        "max_triplet_anchors_per_batch": 128,
        "pseudo_label_method": "leiden",
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 8,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_reassign_noise": True,
        "use_batch_conditioning": True,
        "adversarial_batch_weight": 0.0541465948175273,
        "adversarial_lambda": 1.5,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 55,
        "mmd_batch_weight": 0.05,
        "capture_embedding_snapshots": False,
        "batch_correction_key": "batch",
    },
)


_TRIAL_0009 = _default_variant_preset(
    name="trial_0009",
    description=(
        "Exact stage-1 trial_0009 configuration from the stable_default_v2 "
        "generalist search, reused by the stage-UMAP reruns and trial009 "
        "scIB-E completion pass."
    ),
    param_overrides={
        "hidden_layers": "512,256",
        "z_dim": 256,
        "dropout": 0.2,
        "epochs": 100,
        "warmup_epochs": 74,
        "batch_size": 192,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "masking_rate": 0.2,
        "masked_recon_weight": 0.8,
        "weight_exponent": 0.2,
        "density_knn_k": 15,
        "density_weight_clip": 8.0,
        "dynamic_weight_update_interval": 20,
        "max_cell_weight": 10.0,
        "weight_fusion_mode": "additive",
        "rare_triplet_start_epoch": 84,
        "rare_triplet_margin": 0.5,
        "rare_triplet_min_weight": 1.0,
        "max_triplet_anchors_per_batch": 128,
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 5,
        "hdbscan_reassign_noise": True,
        "adversarial_lambda": 0.5,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 55,
        "mmd_batch_weight": 0.05,
        "lr": 0.0008947361247775492,
        "nb_theta": 8.220634993032638,
        "rare_triplet_weight": 0.09569875560065497,
        "adversarial_batch_weight": 0.0646853688954063,
        "cluster_density_alpha": 0.5334836306684376,
        "dynamic_weight_momentum": 0.8158914576114217,
        "min_cell_weight": 0.4368097033542426,
        "use_batch_conditioning": True,
        "masking_apply_weighted": True,
        "weight_fusion_mode": "additive",
        "pseudo_label_method": "leiden",
        "hdbscan_cluster_selection_method": "eom",
    },
)

_STABLE_GENERALIST_TRIAL_0017 = ScrawPreset(
    name="stable_generalist_stable_generalist",
    description=(
        "Exact stage-1 stable_generalist_stable_generalist configuration from the "
        "stable_generalist_default_v6_search stable_generalist branch. This preset keeps "
        "the hdbscan final clustering configuration used in the source rerun."
    ),
    preprocessing=_merge(
        copy_default_preprocessing(),
        {
            "n_top_genes": 2000,
            "min_genes_per_cell": 200,
            "max_genes_per_cell": None,
            "min_cells_per_gene": 3,
            "target_sum": 20000,
            "scale_max_value": 10.0,
            "hvg_flavor": "seurat",
            "hvg_strategy": "train_only",
            "dropout_method": "none",
            "noise_level": 0.0,
        },
    ),
    algorithm_params=_merge(
        copy_default_algorithm_params(),
        {
            "input_type": "processed",
            "masking_value": 0.0,
            "nb_mu_clip_max": 1e6,
            "clustering_method": "hdbscan",
            "final_target_k": 0,
            "hdbscan_scan_enabled": False,
            "hdbscan_scan_eval_sample_size": 3000,
            "hdbscan_scan_include_alternative_method": True,
            "density_weight_exponent": 1.0,
            "cluster_weight_power": 1.0,
            "density_weight_power": 1.0,
            "n_clusters": 0,
            "pseudo_leiden_resolution_strategy": "scraw_default",
            "pseudo_leiden_target_clusters": 0,
            "unsupervised_k_fallback": 0,
            "unsupervised_k_selection": "heuristic",
            "unsupervised_k_min": 8,
            "unsupervised_k_max": 30,
            "unsupervised_k_num_candidates": 12,
            "unsupervised_k_pca_dim": 32,
            "unsupervised_k_eval_sample_size": 3000,
            "unsupervised_k_stability_runs": 5,
            "unsupervised_k_stability_sample_size": 4000,
            "unsupervised_k_weight_stability": 0.45,
            "unsupervised_k_weight_silhouette": 0.25,
            "unsupervised_k_weight_ch": 0.2,
            "unsupervised_k_weight_db": 0.1,
            "unsupervised_k_weight_tiny_clusters": 0.2,
            "unsupervised_k_min_cluster_fraction": 0.005,
            "unsupervised_k_overseg_penalty": 0.25,
            "unsupervised_k_underseg_penalty": 0.05,
            "rare_loss_type": "triplet",
            "snapshot_interval_epochs": 10,
            "hidden_layers": "512,256,128",
            "z_dim": 256,
            "dropout": 0.3,
            "epochs": 120,
            "warmup_epochs": 55,
            "lr": 0.00164076083297036,
            "batch_size": 192,
            "reconstruction_distribution": "mse",
            "nb_input_transform": "log1p",
            "nb_theta": 19.237616208909103,
            "masking_rate": 0.1,
            "masked_recon_weight": 0.8,
            "masking_apply_weighted": True,
            "weight_exponent": 0.2,
            "cluster_density_alpha": 0.3483603718613933,
            "density_knn_k": 15,
            "density_weight_clip": 3.0,
            "dynamic_weight_momentum": 0.6884621079434989,
            "dynamic_weight_update_interval": 20,
            "weight_fusion_mode": "multiplicative",
            "min_cell_weight": 0.3845423008053828,
            "max_cell_weight": 10.0,
            "rare_triplet_weight": 0.05007581780188212,
            "rare_triplet_start_epoch": 60,
            "rare_triplet_margin": 0.4,
            "rare_triplet_min_weight": 1.2,
            "max_triplet_anchors_per_batch": 64,
            "pseudo_label_method": "leiden",
            "hdbscan_min_cluster_size": 8,
            "hdbscan_min_samples": 6,
            "hdbscan_cluster_selection_method": "eom",
            "hdbscan_reassign_noise": False,
            "use_batch_conditioning": True,
            "adversarial_batch_weight": 0.11763398875166495,
            "adversarial_lambda": 1.0,
            "adversarial_start_epoch": 30,
            "adversarial_ramp_epochs": 30,
            "mmd_batch_weight": 0.0,
            "capture_embedding_snapshots": False,
            "batch_correction_key": "batch",
        },
    ),
    supports_dann=True,
)


PRESETS: Dict[str, ScrawPreset] = {
    DEFAULT_PRESET_NAME: ScrawPreset(
        name=DEFAULT_PRESET_NAME,
        description=(
            "Stable default configuration anchored on "
            "stable_generalist_stable_generalist from stable_generalist_default_v6_search."
        ),
        preprocessing=copy_default_preprocessing(),
        algorithm_params=copy_default_algorithm_params(),
        supports_dann=True,
    ),
    "stable_default": ScrawPreset(
        name="stable_default",
        description=(
            "Alias of the stable default configuration anchored on "
            "stable_generalist_stable_generalist."
        ),
        preprocessing=copy_default_preprocessing(),
        algorithm_params=copy_default_algorithm_params(),
        supports_dann=True,
    ),
    "default_v_2": _DEFAULT_V_2,
    "stable_default_v2": _DEFAULT_V_2,
    "trial_0009": _TRIAL_0009,
    "trial009": _TRIAL_0009,
    "stable_generalist_stable_generalist": _STABLE_GENERALIST_TRIAL_0017,
    "default_v_3": _DEFAULT_V_3,
    "default_v_4": _DEFAULT_V_4,
    "default_v_5": _DEFAULT_V_5,
    "baron": ScrawPreset(
        name="baron",
        description=(
            "Legacy preset aligned with the baseline defaults from "
            "/data2/fbidet/scRAW."
        ),
        preprocessing=copy_baron_preprocessing(),
        algorithm_params=copy_baron_algorithm_params(),
        supports_dann=True,
    ),
    "baron_best": ScrawPreset(
        name="baron_best",
        description=(
            "Best Baron run (trip10_d10_s35) with HDBSCAN + Leiden pseudo-labels + "
            "NB(log1p) and no DANN."
        ),
        preprocessing=_merge(
            _COMMON_PREPROCESSING,
            {
                "min_genes_per_cell": 200,
            },
        ),
        algorithm_params=_merge(
            _COMMON_ALGO,
            {
                "epochs": 120,
                "warmup_epochs": 30,
                "dropout": 0.1,
                "dynamic_weight_momentum": 0.7,
                "nb_input_transform": "log1p",
                "rare_triplet_weight": 0.1,
                "rare_triplet_start_epoch": 35,
                "capture_embedding_snapshots": False,
                "use_batch_conditioning": False,
                "batch_correction_key": "auto",
                "adversarial_batch_weight": 0.0,
                "adversarial_start_epoch": 0,
                "adversarial_ramp_epochs": 0,
                "mmd_batch_weight": 0.0,
            },
        ),
        supports_dann=False,
    ),
    "pancreas_best": ScrawPreset(
        name="pancreas_best",
        description=(
            "Best Pancreas baseline with DANN on study batches, HDBSCAN + Leiden "
            "pseudo-labels and NB(Pearson residuals)."
        ),
        preprocessing=_merge(
            _COMMON_PREPROCESSING,
            {
                "min_genes_per_cell": 100,
                "max_genes_per_cell": 10000,
            },
        ),
        algorithm_params=_merge(
            _COMMON_ALGO,
            {
                "epochs": 80,
                "warmup_epochs": 20,
                "nb_input_transform": "pearson_residuals",
                "pearson_residual_clip": 10.0,
                "rare_triplet_weight": 0.05,
                "rare_triplet_start_epoch": 30,
                "weight_exponent": 0.2,
                "capture_embedding_snapshots": True,
                "use_batch_conditioning": True,
                "batch_correction_key": "study",
                "adversarial_batch_weight": 0.1,
                "adversarial_lambda": 1.0,
                "adversarial_start_epoch": 10,
                "adversarial_ramp_epochs": 0,
                "mmd_batch_weight": 0.0,
            },
        ),
        supports_dann=True,
    ),
}


def get_preset(name: str) -> ScrawPreset:
    """Resolve a preset name (case-insensitive) or raise a clear error."""
    key = name.strip().lower()
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[key]
