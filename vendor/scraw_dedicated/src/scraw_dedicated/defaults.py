"""Central preset configurations for standalone scRAW.

`default` now tracks the stable stage-1 `stable_generalist_stable_generalist`
configuration from stable_generalist_default_v6_search. The previous `/data2/fbidet/scRAW`
baseline is preserved as an explicit `baron` preset.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


DEFAULT_PRESET_NAME = "default"
BARON_PRESET_NAME = "baron"

DEFAULT_TRIAL_CONFIGURATION: Dict[str, Any] = {
    "trial_number": 17,
    "candidate_id": "stable_generalist_stable_generalist",
    "branch": "stable_generalist",
    "preset_name": "default_v_3",
    "requested_method": "best_of_both",
    "source": (
        "optuna_stable_generalist_search_20260415_161134/phase1/stable_generalist/"
        "stage1/trials/stable_generalist_stable_generalist"
    ),
    "final_clustering_requested": "hdbscan",
    "dann_enabled": True,
    "param_overrides": {
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
}

BARON_TRIAL_CONFIGURATION: Dict[str, Any] = {
    "trial_number": None,
    "source": "scraw_default",
    "final_clustering_requested": "hdbscan",
    "dann_enabled": True,
    "param_overrides": {
        "hidden_layers": "512,256,128",
        "z_dim": 192,
        "dropout": 0.3,
        "epochs": 210,
        "warmup_epochs": 74,
        "lr": 0.00233670337683859,
        "batch_size": 192,
        "reconstruction_distribution": "mse",
        "nb_input_transform": "log1p",
        "nb_theta": 2.5815883941220323,
        "masking_rate": 0.15000000000000002,
        "masked_recon_weight": 0.8,
        "masking_apply_weighted": True,
        "weight_exponent": 0.7000000000000001,
        "cluster_density_alpha": 0.30000000000000004,
        "density_knn_k": 15,
        "density_weight_clip": 8.0,
        "dynamic_weight_momentum": 0.8500000000000001,
        "dynamic_weight_update_interval": 10,
        "weight_fusion_mode": "additive",
        "min_cell_weight": 0.45000000000000007,
        "max_cell_weight": 8.0,
        "rare_triplet_weight": 0.2346243650039478,
        "rare_triplet_start_epoch": 84,
        "rare_triplet_margin": 0.4,
        "rare_triplet_min_weight": 1.8,
        "max_triplet_anchors_per_batch": 64,
        "pseudo_label_method": "leiden",
        "hdbscan_min_cluster_size": 8,
        "hdbscan_min_samples": 8,
        "hdbscan_cluster_selection_method": "eom",
        "hdbscan_reassign_noise": True,
        "use_batch_conditioning": True,
        "adversarial_batch_weight": 0.056150696336115635,
        "adversarial_lambda": 1.75,
        "adversarial_start_epoch": 55,
        "adversarial_ramp_epochs": 60,
        "mmd_batch_weight": 0.0,
        "capture_embedding_snapshots": False,
        "batch_correction_key": "batch",
    },
}

DEFAULT_PARAM_OVERRIDES: Dict[str, Any] = deepcopy(DEFAULT_TRIAL_CONFIGURATION["param_overrides"])
BARON_PARAM_OVERRIDES: Dict[str, Any] = deepcopy(BARON_TRIAL_CONFIGURATION["param_overrides"])

DEFAULT_PREPROCESSING: Dict[str, Any] = {
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
}

DEFAULT_ALGORITHM_STATIC_PARAMS: Dict[str, Any] = {
    "input_type": "processed",
    "masking_value": 0.0,
    "nb_mu_clip_max": 1e6,
    "clustering_method": str(DEFAULT_TRIAL_CONFIGURATION["final_clustering_requested"]),
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
    "unsupervised_k_weight_ch": 0.20,
    "unsupervised_k_weight_db": 0.10,
    "unsupervised_k_weight_tiny_clusters": 0.20,
    "unsupervised_k_min_cluster_fraction": 0.005,
    "unsupervised_k_overseg_penalty": 0.25,
    "unsupervised_k_underseg_penalty": 0.05,
    "rare_loss_type": "triplet",
    "snapshot_interval_epochs": 10,
    "random_state": 64,
    "seed": 64,
}

DEFAULT_ALGORITHM_PARAMS: Dict[str, Any] = {
    **DEFAULT_ALGORITHM_STATIC_PARAMS,
    **DEFAULT_PARAM_OVERRIDES,
}

BARON_ALGORITHM_PARAMS: Dict[str, Any] = {
    **DEFAULT_ALGORITHM_STATIC_PARAMS,
    **BARON_PARAM_OVERRIDES,
}


def copy_default_algorithm_params() -> Dict[str, Any]:
    """Return a deep copy of the default scRAW algorithm parameters."""
    return deepcopy(DEFAULT_ALGORITHM_PARAMS)


def copy_default_preprocessing() -> Dict[str, Any]:
    """Return a deep copy of the default scRAW preprocessing configuration."""
    return deepcopy(DEFAULT_PREPROCESSING)


def copy_default_trial_configuration() -> Dict[str, Any]:
    """Return a deep copy of the default standalone scRAW configuration manifest."""
    return deepcopy(DEFAULT_TRIAL_CONFIGURATION)


def copy_baron_algorithm_params() -> Dict[str, Any]:
    """Return a deep copy of the legacy `/data2/fbidet/scRAW` baseline parameters."""
    return deepcopy(BARON_ALGORITHM_PARAMS)


def copy_baron_preprocessing() -> Dict[str, Any]:
    """Return the preprocessing used by the legacy `/data2/fbidet/scRAW` baseline."""
    return deepcopy(DEFAULT_PREPROCESSING)


def copy_baron_trial_configuration() -> Dict[str, Any]:
    """Return a deep copy of the legacy baseline standalone scRAW manifest."""
    return deepcopy(BARON_TRIAL_CONFIGURATION)
