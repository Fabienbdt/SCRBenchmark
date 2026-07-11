#!/usr/bin/env python3
"""
SCRBenchmark CLI - Command Line Interface for Single-Cell RNA-seq Clustering Benchmark

Usage:
    ./scrbenchmark run --data <file> --algorithms <algo1,algo2> [options]
    ./scrbenchmark list-algorithms
    ./scrbenchmark list-params --algorithm <name>
    ./scrbenchmark generate-config --output <file>
    ./scrbenchmark run --config <config.yaml>

Examples:
    # Run analysis with default settings
    ./scrbenchmark run --data data/pbmc3k.h5ad --algorithms pca,scdeepcluster \\
        --param pca:clustering_method=kmeans

    # Run with custom preprocessing
    ./scrbenchmark run --data data/pbmc3k.h5ad --algorithms scdeepcluster \\
        --n-top-genes 2000 --min-genes-per-cell 200

    # Run with custom hyperparameters
    ./scrbenchmark run --data data/pbmc3k.h5ad --algorithms scdeepcluster \\
        --param scdeepcluster:pretrain_epochs=100 --param scdeepcluster:z_dim=64

    # Run from config file
    ./scrbenchmark run --config my_config.yaml

    # Generate a template config file
    ./scrbenchmark generate-config --output my_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_path():
    """Add project root to path."""
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


setup_path()


def load_algorithms():
    """Import and register available algorithms without hard-failing on one bad dependency."""
    from core.algorithm_registry import AlgorithmRegistry
    try:
        import algorithms  # noqa: F401 - triggers module-level auto-registration
    except Exception as exc:
        logger.warning(
            "Partial algorithm loading due to import error: %s. "
            "Some algorithms may be unavailable in this environment.",
            exc,
        )
    return AlgorithmRegistry


def get_available_algorithms() -> List[Dict[str, Any]]:
    """Get list of available algorithms with their info."""
    registry = load_algorithms()
    return [
        {
            'name': info.name,
            'display_name': info.display_name,
            'category': info.category,
            'description': info.description,
            'requires_gpu': info.requires_gpu,
            'mps_compatible': info.mps_compatible,
        }
        for info in registry.list_algorithms()
    ]


def get_algorithm_params(algorithm_name: str) -> List[Dict[str, Any]]:
    """Get hyperparameters for a specific algorithm."""
    registry = load_algorithms()
    algo_class = registry.get(algorithm_name)
    if algo_class is None:
        raise ValueError(f"Algorithm '{algorithm_name}' not found")

    params = algo_class.get_hyperparameters()
    return [
        {
            'name': p.name,
            'display_name': p.display_name,
            'type': p.param_type.value,
            'default': p.default,
            'description': p.description,
            'min_value': p.min_value,
            'max_value': p.max_value,
            'choices': p.choices,
            'category': p.category,
            'advanced': p.advanced,
        }
        for p in params
    ]


def parse_param_string(param_str: str) -> tuple:
    """
    Parse a parameter string in format 'algorithm:param=value' or 'param=value'.

    Returns:
        (algorithm_name or None, param_name, value)
    """
    if ':' in param_str:
        algo_part, param_part = param_str.split(':', 1)
    else:
        algo_part = None
        param_part = param_str

    if '=' not in param_part:
        raise ValueError(f"Invalid parameter format: {param_str}. Expected 'param=value' or 'algo:param=value'")

    param_name, value_str = param_part.split('=', 1)

    # Try to parse value as JSON (handles numbers, bools, lists)
    try:
        value = json.loads(value_str)
    except json.JSONDecodeError:
        # Keep as string
        value = value_str

    return algo_part, param_name, value


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        else:
            return json.load(f)


def generate_default_config() -> Dict[str, Any]:
    """Generate a default configuration template."""
    algos = get_available_algorithms()

    # Get preprocessing params
    from core.config import PREPROCESSING_PARAMS
    preprocessing = {p.name: p.default for p in PREPROCESSING_PARAMS}

    # Get default params for each algorithm
    algorithm_params = {}
    for algo in algos:
        params = get_algorithm_params(algo['name'])
        algorithm_params[algo['name']] = {p['name']: p['default'] for p in params}

    return {
        'data': {
            'file': 'data/baron_human_pancreas.h5ad',
            'format': 'auto',  # auto, h5ad, h5, csv, tsv, mtx
        },
        'preprocessing': {
            'enabled': True,
            'n_top_genes': preprocessing.get('n_top_genes', 5000),
            'min_genes_per_cell': preprocessing.get('min_genes_per_cell', 200),
            'max_genes_per_cell': preprocessing.get('max_genes_per_cell', 10000),
            'min_cells_per_gene': preprocessing.get('min_cells_per_gene', 3),
            'target_sum': preprocessing.get('target_sum', 10000),
            'scale_max_value': preprocessing.get('scale_max_value', 10.0),
            'hvg_flavor': preprocessing.get('hvg_flavor', 'seurat'),
            'do_normalization': True,
            'do_log_transform': True,
            'do_hvg': True,
            'do_scaling': True,
            'do_cell_filtering': True,
            'do_gene_filtering': True,
            'do_batch_correction': False,
            'batch_correction_labels_key': None,
            'batch_correction_epochs': 100,
            'batch_correction_n_latent': 30,
            'batch_correction_batch_size': 128,
            'batch_correction_dct_weight': 1.0,
            'batch_correction_corr_mse_weight': 0.1,
            # sysVI defaults
            'batch_correction_prior': 'vamp',
            'batch_correction_n_prior_components': 5,
            'batch_correction_n_hidden': 256,
            'batch_correction_embed_covariates': True,
            'batch_correction_cycle_weight': 5.0,
            'batch_correction_kl_weight': 1.0,
        },
        'algorithms': ['pca'],  # List of algorithms to run
        'algorithm_params': {
            'pca': algorithm_params.get('pca', {
                'clustering_method': 'leiden',
                'n_pca_components': 0,  # 0 = auto (Cattell scree test)
                'graph_resolution': 0.0,  # 0 = auto-search
            }),

            # Add more algorithm-specific params as needed
        },
        'execution': {
            'device': 'auto',  # auto, cpu, cuda, mps
            'n_repeats': 1,
            'random_seed': 42,
            'compute_scib_metrics': True,
        },
        'output': {
            'directory': 'results_CLI',
            'save_embeddings': False,
            'save_labels': True,
            'format': 'json',  # json, csv
            'verbose': True,
        }
    }


def print_algorithms_table(algorithms: List[Dict[str, Any]]):
    """Print algorithms in a formatted table."""
    # Group by category
    by_category = {}
    for algo in algorithms:
        cat = algo['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(algo)

    print("\n" + "=" * 80)
    print("AVAILABLE ALGORITHMS")
    print("=" * 80)

    for category, algos in sorted(by_category.items()):
        print(f"\n[{category.upper()}]")
        print("-" * 80)
        print(f"{'Name':<20} {'Display Name':<25} {'GPU':<6} {'MPS':<6}")
        print("-" * 80)
        for algo in algos:
            gpu = "Yes" if algo['requires_gpu'] else "No"
            mps = "Yes" if algo['mps_compatible'] else "No"
            print(f"{algo['name']:<20} {algo['display_name']:<25} {gpu:<6} {mps:<6}")

    print("\n" + "=" * 80)


def print_params_table(algorithm_name: str, params: List[Dict[str, Any]]):
    """Print algorithm parameters in a formatted table."""
    print(f"\n{'=' * 90}")
    print(f"HYPERPARAMETERS FOR: {algorithm_name}")
    print(f"{'=' * 90}")

    # Group by category
    by_category = {}
    for p in params:
        cat = p['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(p)

    for category, cat_params in sorted(by_category.items()):
        print(f"\n[{category}]")
        print("-" * 90)
        print(f"{'Parameter':<25} {'Type':<10} {'Default':<15} {'Description'}")
        print("-" * 90)
        for p in cat_params:
            default_str = str(p['default'])[:12]
            desc = p['description'][:40] + "..." if len(p['description']) > 40 else p['description']
            advanced = " (adv)" if p['advanced'] else ""
            print(f"{p['name']:<25} {p['type']:<10} {default_str:<15} {desc}{advanced}")

            if p['choices']:
                print(f"{'':25} Choices: {p['choices']}")
            if p['min_value'] is not None or p['max_value'] is not None:
                range_str = f"Range: [{p['min_value']}, {p['max_value']}]"
                print(f"{'':25} {range_str}")

    print(f"\n{'=' * 90}")


def _sanitize_for_json(value: Any) -> Any:
    """Convert values to JSON-serializable representations."""
    if callable(value):
        return f"<callable:{getattr(value, '__name__', 'anonymous')}>"

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # numpy scalars
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]

    return str(value)


def _save_algorithm_hyperparams_snapshot(
    *,
    config_dir: Path,
    algorithms: List[str],
    algorithm_params: Dict[str, Dict[str, Any]],
    run_results: Optional[List[Any]] = None,
    search_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
    execution: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Save a complete snapshot of algorithm params/hyperparams used by CLI runs.

    The snapshot includes:
    - declared hyperparameters for each algorithm
    - default params declared by each algorithm
    - explicit overrides provided to CLI/config
    - effective params (defaults merged with overrides)
    - per-run params actually used (e.g. seed offset per repeat)
    """
    if not config_dir:
        return None

    config_dir.mkdir(parents=True, exist_ok=True)

    declared_hyperparams: Dict[str, List[Dict[str, Any]]] = {}
    defaults_by_algorithm: Dict[str, Dict[str, Any]] = {}
    overrides_by_algorithm: Dict[str, Dict[str, Any]] = {}
    effective_by_algorithm: Dict[str, Dict[str, Any]] = {}
    effective_source_by_algorithm: Dict[str, str] = {}

    for algo in algorithms:
        try:
            declared = get_algorithm_params(algo)
        except Exception as e:
            logger.warning(f"Could not load declared hyperparameters for {algo}: {e}")
            declared = []

        declared_hyperparams[algo] = _sanitize_for_json(declared)
        defaults = {p["name"]: p.get("default") for p in declared}
        defaults_by_algorithm[algo] = _sanitize_for_json(defaults)

        overrides = (algorithm_params or {}).get(algo, {}) or {}
        overrides_by_algorithm[algo] = _sanitize_for_json(overrides)

        effective = defaults.copy()
        effective.update(overrides)
        effective_by_algorithm[algo] = _sanitize_for_json(effective)
        effective_source_by_algorithm[algo] = "declared_defaults+overrides"

    declared_effective_by_algorithm = {
        algo: _sanitize_for_json(dict(effective_by_algorithm.get(algo, {}) or {}))
        for algo in algorithms
    }

    runs_payload: List[Dict[str, Any]] = []
    leiden_resolution_by_run: List[Dict[str, Any]] = []
    best_run_params_by_algorithm: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    if run_results:
        for result in run_results:
            algo_name = getattr(result, "algorithm_name", "unknown")
            run_id = getattr(result, "run_id", None)
            params_used = getattr(result, "params", {}) or {}
            runs_payload.append(
                {
                    "algorithm": algo_name,
                    "run_id": run_id,
                    "params_used": _sanitize_for_json(params_used),
                }
            )

            selected_resolution = params_used.get("leiden_resolution_selected")
            if selected_resolution is not None:
                leiden_resolution_by_run.append(
                    {
                        "algorithm": algo_name,
                        "run_id": run_id,
                        "requested_resolution": _sanitize_for_json(params_used.get("leiden_resolution")),
                        "selected_resolution": _sanitize_for_json(selected_resolution),
                        "selection_mode": _sanitize_for_json(
                            params_used.get("leiden_resolution_selection_mode", "unknown")
                        ),
                    }
                )

            if isinstance(run_id, int):
                prev = best_run_params_by_algorithm.get(algo_name)
                if prev is None or run_id < prev[0]:
                    best_run_params_by_algorithm[algo_name] = (run_id, dict(params_used))

    for algo in algorithms:
        run_payload = best_run_params_by_algorithm.get(algo)
        if not run_payload:
            continue

        run_id, run_params = run_payload
        merged = dict(effective_by_algorithm.get(algo, {}) or {})
        merged.update(run_params)
        effective_by_algorithm[algo] = _sanitize_for_json(merged)
        effective_source_by_algorithm[algo] = f"run_{run_id}_params_overrides_declared_effective"

    payload = {
        "timestamp": datetime.now().isoformat(),
        "algorithms": algorithms,
        "declared_hyperparameters": declared_hyperparams,
        "defaults_by_algorithm": defaults_by_algorithm,
        "overrides_by_algorithm": overrides_by_algorithm,
        "declared_effective_params_by_algorithm": declared_effective_by_algorithm,
        "effective_params_by_algorithm": effective_by_algorithm,
        "effective_params_source_by_algorithm": effective_source_by_algorithm,
        "search_spaces": _sanitize_for_json(search_spaces or {}),
        "execution": _sanitize_for_json(execution or {}),
        "context": _sanitize_for_json(context or {}),
        "per_run_params": runs_payload,
        "leiden_resolution_selected_by_run": leiden_resolution_by_run,
    }

    snapshot_path = config_dir / "algorithm_hyperparams_used.json"
    with open(snapshot_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"Algorithm hyperparameter snapshot saved to {snapshot_path}")
    return snapshot_path


def _save_weight_history_artifacts(
    *,
    run_results: List[Any],
    results_dir: Path,
    figures_dir: Path,
) -> None:
    """Save per-epoch cell-weight stats as JSON, CSV, and quick-look PNG curves."""
    histories = [
        (result, getattr(result, "weight_history", None) or [])
        for result in run_results
        if getattr(result, "weight_history", None)
    ]
    if not histories:
        return

    import pandas as pd
    import matplotlib.pyplot as plt

    weight_dir = results_dir / "weight_history"
    weight_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for result, history in histories:
        safe_algo = str(result.algorithm_name).replace(" ", "_").replace("/", "_")
        payload = {
            "algorithm": result.algorithm_name,
            "run_id": result.run_id,
            "records": history,
        }
        with open(weight_dir / f"weights_{safe_algo}_run{result.run_id}.json", "w") as f:
            json.dump(payload, f, indent=2)

        df = pd.DataFrame(history)
        df.to_csv(weight_dir / f"weights_{safe_algo}_run{result.run_id}.csv", index=False)

        if "epoch" not in df.columns or df.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if "std" in df.columns:
            ax.plot(df["epoch"], df["std"], label="std", color="#1f77b4", linewidth=2)
        if "min" in df.columns:
            ax.plot(df["epoch"], df["min"], label="min", color="#2ca02c", linewidth=1.5)
        if "max" in df.columns:
            ax.plot(df["epoch"], df["max"], label="max", color="#d62728", linewidth=1.5)
        if "mean" in df.columns:
            ax.plot(df["epoch"], df["mean"], label="mean", color="#444444", linewidth=1, linestyle="--")
        ax.set_title(f"{result.algorithm_name} run {result.run_id} cell-weight dynamics")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cell weight")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"weight_dynamics_{safe_algo}_run{result.run_id}.png",
            bbox_inches="tight",
            dpi=150,
        )
        plt.close(fig)

    logger.info(f"Weight history saved to {weight_dir}")


def _extract_effective_params_from_run_results(
    run_results: Optional[List[Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract effective params from completed run results.

    Returns:
        - effective_by_algorithm: first run (lowest run_id) params per algorithm
        - effective_per_run: list of per-run params payload
    """
    if not run_results:
        return {}, []

    per_run: List[Dict[str, Any]] = []
    best_by_algo: Dict[str, Tuple[int, Dict[str, Any]]] = {}

    for result in run_results:
        algo_name = str(getattr(result, "algorithm_name", "unknown"))
        run_id = getattr(result, "run_id", None)
        params_used = dict(getattr(result, "params", {}) or {})

        per_run.append(
            {
                "algorithm": algo_name,
                "run_id": run_id,
                "params_used": _sanitize_for_json(params_used),
            }
        )

        if isinstance(run_id, int):
            prev = best_by_algo.get(algo_name)
            if prev is None or run_id < prev[0]:
                best_by_algo[algo_name] = (run_id, params_used)

    effective_by_algorithm: Dict[str, Dict[str, Any]] = {
        algo: _sanitize_for_json(params)
        for algo, (_run_id, params) in best_by_algo.items()
    }

    return effective_by_algorithm, per_run


def run_analysis(args: argparse.Namespace) -> int:
    """Run the clustering analysis."""
    import numpy as np

    # Load config from file if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config_file(args.config)
    else:
        config = {}

    # Override config with command line arguments
    data_file = args.data or config.get('data', {}).get('file')
    if not data_file:
        logger.error("No data file specified. Use --data or provide in config file.")
        return 1

    # Parse algorithms
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(',')]
    else:
        algorithms = config.get('algorithms', [])

    if not algorithms:
        logger.error("No algorithms specified. Use --algorithms or provide in config file.")
        return 1

    # Validate algorithms
    available = {a['name'] for a in get_available_algorithms()}
    for algo in algorithms:
        if algo not in available:
            logger.error(f"Unknown algorithm: {algo}. Use 'list-algorithms' to see available options.")
            return 1

    # Build preprocessing params
    preprocess_config = config.get('preprocessing', {})

    def _arg_or_config(arg_val, cfg, key, default):
        return arg_val if arg_val is not None else cfg.get(key, default)

    # Helper to determine boolean toggle: CLI flag overrides config file
    def _bool_toggle(cli_negation_flag, cfg, key, default=True):
        """Returns False if CLI --no-X flag is set, else uses config or default."""
        if cli_negation_flag:
            return False
        return cfg.get(key, default)

    def _parse_balance_target(value, arg_name: str) -> Optional[int]:
        """
        Parse balancing target from CLI value.

        Accepted values:
        - None / '' / 'auto' => None (auto-detect smallest batch)
        - positive integer string/int => int
        """
        if value is None:
            return None

        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"", "auto", "none"}:
                return None
            try:
                parsed = int(v)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {arg_name}: '{value}'. "
                    "Use 'auto' or a positive integer."
                )
        else:
            try:
                parsed = int(value)
            except Exception:
                raise ValueError(
                    f"Invalid value for {arg_name}: '{value}'. "
                    "Use 'auto' or a positive integer."
                )

        if parsed <= 0:
            raise ValueError(
                f"Invalid value for {arg_name}: '{value}'. "
                "Expected a positive integer or 'auto'."
            )
        return parsed

    def _select_periodic_snapshots_every_10(snapshots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep periodic embedding snapshots every 10 epochs (+first/last periodic)."""
        periodic = [
            s for s in snapshots
            if (
                s.get("embeddings") is not None
                and len(s.get("embeddings")) > 0
                and s.get("snapshot_type", "periodic") == "periodic"
            )
        ]
        if not periodic:
            periodic = [
                s for s in snapshots
                if s.get("embeddings") is not None and len(s.get("embeddings")) > 0
            ]
        if not periodic:
            return []

        selected: List[Dict[str, Any]] = []
        for s in periodic:
            epoch = s.get("epoch")
            if isinstance(epoch, (int, np.integer)):
                if int(epoch) % 10 == 0:
                    selected.append(s)

        if periodic[0] not in selected:
            selected.insert(0, periodic[0])
        if periodic[-1] not in selected:
            selected.append(periodic[-1])

        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for s in selected:
            sid = id(s)
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            deduped.append(s)
        return deduped if deduped else periodic

    def _get_umap_evolution_render_config() -> Dict[str, Any]:
        """
        Read UMAP-evolution rendering options from environment.

        Supported env vars:
        - SCRBENCHMARK_UMAP_EVOLUTION_COLOR_MODE:
          ground_truth | ground_truth_weighted | batch | pseudo_cluster
        - SCRBENCHMARK_UMAP_EVOLUTION_MAX_POINTS:
          max plotted cells per snapshot (0 = no subsampling)
        - SCRBENCHMARK_UMAP_EVOLUTION_POINT_SIZE:
          base point size for scatter
        """
        import os

        allowed_modes = {"ground_truth", "ground_truth_weighted", "batch", "pseudo_cluster"}
        color_mode = str(
            os.getenv("SCRBENCHMARK_UMAP_EVOLUTION_COLOR_MODE", "ground_truth")
        ).strip().lower()
        if color_mode not in allowed_modes:
            logger.warning(
                "Invalid SCRBENCHMARK_UMAP_EVOLUTION_COLOR_MODE='%s'. "
                "Falling back to 'ground_truth'.",
                color_mode,
            )
            color_mode = "ground_truth"

        try:
            max_points = int(os.getenv("SCRBENCHMARK_UMAP_EVOLUTION_MAX_POINTS", "5000"))
        except ValueError:
            logger.warning(
                "Invalid SCRBENCHMARK_UMAP_EVOLUTION_MAX_POINTS; using 5000."
            )
            max_points = 5000
        if max_points < 0:
            max_points = 5000

        try:
            point_size = int(os.getenv("SCRBENCHMARK_UMAP_EVOLUTION_POINT_SIZE", "3"))
        except ValueError:
            logger.warning(
                "Invalid SCRBENCHMARK_UMAP_EVOLUTION_POINT_SIZE; using 3."
            )
            point_size = 3
        if point_size <= 0:
            point_size = 3

        return {
            "color_mode": color_mode,
            "max_points": max_points,
            "point_size": point_size,
        }

    preprocessing_params = {
        'skip': not preprocess_config.get('enabled', True) if preprocess_config else False,
        'n_top_genes': _arg_or_config(args.n_top_genes, preprocess_config, 'n_top_genes', 5000),
        'min_genes_per_cell': _arg_or_config(args.min_genes_per_cell, preprocess_config, 'min_genes_per_cell', 200),
        'max_genes_per_cell': _arg_or_config(args.max_genes_per_cell, preprocess_config, 'max_genes_per_cell', None),  # Default None (disabled)
        'min_cells_per_gene': _arg_or_config(args.min_cells_per_gene, preprocess_config, 'min_cells_per_gene', 3),
        'target_sum': _arg_or_config(args.target_sum, preprocess_config, 'target_sum', 10000),
        'scale_max_value': _arg_or_config(args.scale_max_value, preprocess_config, 'scale_max_value', 10.0),
        'hvg_flavor': _arg_or_config(args.hvg_flavor, preprocess_config, 'hvg_flavor', 'seurat'),
        'hvg_strategy': _arg_or_config(args.hvg_strategy, preprocess_config, 'hvg_strategy', 'train_only'),
        'dropout_method': _arg_or_config(args.dropout_method, preprocess_config, 'dropout_method', 'none'),
        'dropout_rate': _arg_or_config(args.dropout_rate, preprocess_config, 'dropout_rate', 0.2),
        'dropout_mid': _arg_or_config(args.dropout_mid, preprocess_config, 'dropout_mid', 0.0),
        'dropout_shape': _arg_or_config(args.dropout_shape, preprocess_config, 'dropout_shape', -1.0),
        'noise_level': _arg_or_config(args.noise_level, preprocess_config, 'noise_level', 0.0),
        'ensure_celltype_markers': _arg_or_config(
            args.ensure_celltype_markers,
            preprocess_config,
            'ensure_celltype_markers',
            False,
        ),
        'min_markers_per_celltype': _arg_or_config(
            args.min_markers_per_celltype,
            preprocess_config,
            'min_markers_per_celltype',
            5,
        ),
        'max_additional_hvg_genes': _arg_or_config(
            args.max_additional_hvg_genes,
            preprocess_config,
            'max_additional_hvg_genes',
            500,
        ),
        # Preprocessing toggles: CLI --no-X flags override config
        'do_normalization': _bool_toggle(getattr(args, 'no_normalization', False), preprocess_config, 'do_normalization', True),
        'do_log_transform': _bool_toggle(getattr(args, 'no_log_transform', False), preprocess_config, 'do_log_transform', True),
        'do_hvg': _bool_toggle(getattr(args, 'no_hvg', False), preprocess_config, 'do_hvg', True),
        'do_scaling': _bool_toggle(getattr(args, 'no_scaling', False), preprocess_config, 'do_scaling', True),
        'do_cell_filtering': _bool_toggle(getattr(args, 'no_cell_filtering', False), preprocess_config, 'do_cell_filtering', True),
        'do_gene_filtering': _bool_toggle(getattr(args, 'no_gene_filtering', False), preprocess_config, 'do_gene_filtering', True),
        # Batch correction params
        'do_batch_correction': args.batch_correction if hasattr(args, 'batch_correction') else preprocess_config.get('do_batch_correction', False),
        'batch_correction_method': args.batch_correction_method if hasattr(args, 'batch_correction_method') else preprocess_config.get('batch_correction_method', 'scvi'),
        'batch_correction_batch_key': (args.stratify_by if (hasattr(args, 'batch_key') and args.batch_key == 'auto' and args.stratify_by) 
                                      else (args.batch_key if hasattr(args, 'batch_key') else preprocess_config.get('batch_correction_batch_key', 'auto'))),
        'batch_correction_labels_key': (
            args.batch_correction_labels_key
            if hasattr(args, 'batch_correction_labels_key') and args.batch_correction_labels_key is not None
            else (
                args.label_col
                if hasattr(args, 'label_col') and args.label_col is not None
                else preprocess_config.get('batch_correction_labels_key', None)
            )
        ),
        'batch_correction_dct_weight': _arg_or_config(
            args.batch_correction_dct_weight,
            preprocess_config,
            'batch_correction_dct_weight',
            1.0,
        ),
        'batch_correction_corr_mse_weight': _arg_or_config(
            args.batch_correction_corr_mse_weight,
            preprocess_config,
            'batch_correction_corr_mse_weight',
            0.1,
        ),
        'batch_correction_epochs': _arg_or_config(
            args.batch_correction_epochs,
            preprocess_config,
            'batch_correction_epochs',
            None,
        ),
        'batch_correction_n_latent': (
            args.batch_correction_latent
            if args.batch_correction_latent is not None
            else preprocess_config.get(
                'batch_correction_n_latent',
                preprocess_config.get('batch_correction_latent', None),
            )
        ),
        'batch_correction_batch_size': _arg_or_config(
            args.batch_correction_batch_size,
            preprocess_config,
            'batch_correction_batch_size',
            None,
        ),
        # sysVI params
        'batch_correction_cycle_weight': _arg_or_config(
            args.batch_correction_cycle_weight,
            preprocess_config,
            'batch_correction_cycle_weight',
            5.0,
        ),
        'batch_correction_kl_weight': _arg_or_config(
            args.batch_correction_kl_weight,
            preprocess_config,
            'batch_correction_kl_weight',
            1.0,
        ),
        'batch_correction_prior': _arg_or_config(
            args.batch_correction_prior,
            preprocess_config,
            'batch_correction_prior',
            'vamp',
        ),
        'batch_correction_n_prior_components': _arg_or_config(
            args.batch_correction_n_prior_components,
            preprocess_config,
            'batch_correction_n_prior_components',
            5,
        ),
        'batch_correction_embed_covariates': _arg_or_config(
            args.batch_correction_embed_covariates,
            preprocess_config,
            'batch_correction_embed_covariates',
            True,
        ),
        'batch_correction_n_hidden': preprocess_config.get('batch_correction_n_hidden', None),
        'batch_correction_strict': _arg_or_config(
            args.batch_correction_strict,
            preprocess_config,
            'batch_correction_strict',
            False,
        ),
        'batch_correction_seed': preprocess_config.get('batch_correction_seed', None),
        'batch_correction_categorical_covariate_keys': preprocess_config.get(
            'batch_correction_categorical_covariate_keys',
            None,
        ),
    }

    if args.skip_preprocessing:
        preprocessing_params['skip'] = True

    try:
        balance_target_split = _parse_balance_target(getattr(args, 'balance_target', None), '--balance-target')
        balance_target_standard = _parse_balance_target(getattr(args, 'balance_target_standard', None), '--balance-target-standard')
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Build algorithm params
    algo_params_config = config.get('algorithm_params', {})
    algorithm_params = {algo: algo_params_config.get(algo, {}).copy() for algo in algorithms}

    # Parse --param arguments
    if args.param:
        for param_str in args.param:
            algo_name, param_name, value = parse_param_string(param_str)
            if algo_name:
                if algo_name not in algorithm_params:
                    algorithm_params[algo_name] = {}
                algorithm_params[algo_name][param_name] = value
            else:
                for algo in algorithms:
                    if algo not in algorithm_params:
                        algorithm_params[algo] = {}
                    algorithm_params[algo][param_name] = value

    # Parse --search arguments for AutoML
    search_spaces = {}
    if args.search:
        for search_str in args.search:
            # Expected format: algo:param=[v1,v2]
            try:
                algo_name, param_name, value = parse_param_string(search_str)
                if not algo_name:
                    logger.warning(f"Search space '{search_str}' must specify algorithm name (e.g., algo:param=[...]). Ignoring.")
                    continue
                
                if not isinstance(value, list):
                    logger.warning(f"Search space value for '{algo_name}:{param_name}' must be a list. Got {type(value)}. Ignoring.")
                    continue
                
                if algo_name not in search_spaces:
                    search_spaces[algo_name] = {}
                search_spaces[algo_name][param_name] = value
                logger.info(f"Added search space for {algo_name}: {param_name}={value}")
            except Exception as e:
                logger.warning(f"Failed to parse search space '{search_str}': {e}")

    # Set device and clusters (same as before)
    exec_config = config.get('execution', {})
    device = args.device or exec_config.get('device', 'auto')
    for algo in algorithms:
        if algo not in algorithm_params:
            algorithm_params[algo] = {}
        algorithm_params[algo]['device'] = device

    
    if args.n_clusters is not None:
        for algo in algorithms:
            algorithm_params[algo]['n_clusters'] = args.n_clusters

    for algo in algorithms:
        if algo == 'sccdcg' and 'input_type' not in algorithm_params[algo]:
            algorithm_params[algo]['input_type'] = 'raw_original'

    # Set network architecture parameters (aligned with UI _apply_architecture_inputs)
    if args.encoder_layers or args.decoder_layers or args.z_dim:
        encoder_layers = args.encoder_layers
        decoder_layers = args.decoder_layers
        z_dim = args.z_dim
        
        if encoder_layers and not decoder_layers and args.symmetric_layers:
            enc_list = [x.strip() for x in encoder_layers.split(',')]
            decoder_layers = ','.join(reversed(enc_list))
        
        # Parse lists
        enc_list = [int(x.strip()) for x in encoder_layers.split(',')] if encoder_layers else []
        dec_list = [int(x.strip()) for x in decoder_layers.split(',')] if decoder_layers else []
        
        for algo in algorithms:
            if algo not in algorithm_params:
                algorithm_params[algo] = {}
            
            # Apply per-algorithm mapping (matches _apply_architecture_inputs in UI)
            if algo == 'scdeepcluster':
                if enc_list:
                    algorithm_params[algo]['encoder_layers'] = ','.join(str(v) for v in enc_list)
                if dec_list:
                    algorithm_params[algo]['decoder_layers'] = ','.join(str(v) for v in dec_list)
                if z_dim is not None:
                    algorithm_params[algo]['z_dim'] = z_dim
                    
            elif algo == 'sccdcg':
                if enc_list:
                    algorithm_params[algo]['encoder_layers'] = ','.join(str(v) for v in enc_list)
                if dec_list:
                    algorithm_params[algo]['decoder_layers'] = ','.join(str(v) for v in dec_list)
                if z_dim is not None:
                    algorithm_params[algo]['embedding_dim'] = z_dim
                    
            elif algo == 'scname':
                # scNAME: hidden_dims = encoder + latent (last value)
                dims = []
                if enc_list:
                    dims.extend(enc_list)
                if z_dim is not None:
                    dims.append(z_dim)
                if dims:
                    algorithm_params[algo]['hidden_dims'] = ','.join(str(v) for v in dims)
                    
            elif algo == 'sc_mae':
                # scMAE only uses latent size
                if z_dim is not None:
                    algorithm_params[algo]['hidden_size'] = z_dim

    # Execution settings
    n_repeats = args.n_repeats if args.n_repeats is not None else exec_config.get('n_repeats', 1)
    random_seed = args.seed if args.seed is not None else exec_config.get('random_seed', 42)
    compute_scib_metrics = (
        False if getattr(args, 'no_scib_metrics', False)
        else exec_config.get('compute_scib_metrics', True)
    )

    # Reuse the run seed for batch correction unless explicitly overridden in config.
    if preprocessing_params.get('batch_correction_seed') is None:
        preprocessing_params['batch_correction_seed'] = random_seed

    import os

    # Output settings
    output_config = config.get('output', {})
    base_output_dir = args.output or output_config.get('directory', 'results')
    
    # Add timestamp to avoid overwriting (unless disabled)
    if getattr(args, 'no_timestamp', False):
        output_dir = str(base_output_dir)
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_dir = Path(base_output_dir) / timestamp
        output_dir = str(output_dir)
    
    verbose = args.verbose if args.verbose is not None else output_config.get('verbose', True)
    metrics_only_mode = str(os.getenv("SCRBENCHMARK_METRICS_ONLY", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }

    # Set random seed
    np.random.seed(random_seed)

    # Import required modules
    from utils.data_handler import DataHandler
    from utils.analysis_runner import AnalysisRunner
    from utils.dataset_splitter import DatasetSplitter, BenchmarkPreprocessor
    import utils.visualization as viz

    # Create output directory structure
    output_path = Path(output_dir)
    results_dir = output_path / 'results'
    figures_dir = output_path / 'figures'
    data_dir = output_path / 'data'
    config_dir = output_path / 'config'
    labels_dir = results_dir / 'labels'

    for path in [output_path, results_dir, figures_dir, data_dir, config_dir, labels_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Save planned algorithm params/hyperparams at run start.
    # This guarantees traceability even if execution stops before completion.
    try:
        _save_algorithm_hyperparams_snapshot(
            config_dir=config_dir,
            algorithms=algorithms,
            algorithm_params=algorithm_params,
            run_results=None,
            search_spaces=search_spaces,
            execution={
                'device': device,
                'n_repeats': n_repeats,
                'random_seed': random_seed,
                'compute_scib_metrics': compute_scib_metrics,
            },
            context={
                'mode': 'benchmark' if (args.benchmark_mode or config.get('benchmark_mode', False)) else 'standard',
                'data_file': str(data_file),
                'output_dir': str(output_dir),
                'status': 'planned',
            },
        )
    except Exception as e:
        logger.warning(f"Could not save planned hyperparameter snapshot: {e}")

    # Load data
    logger.info(f"Loading data from {data_file}...")
    data_handler = DataHandler()
    try:
        data_handler.load(data_file)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    info = data_handler.get_info()
    logger.info(f"Loaded: {info['n_cells']} cells, {info['n_genes']} genes")

    # Exclude specified batches (e.g., 'smarter' which has RPKM instead of raw counts)
    if hasattr(args, 'exclude_batches') and args.exclude_batches:
        from utils.dataset_splitter import get_batch_column
        adata = data_handler.get_data()
        batch_col = get_batch_column(adata)
        
        if batch_col:
            exclude_list = [b.strip() for b in args.exclude_batches.split(',')]
            n_before = adata.n_obs
            mask = ~adata.obs[batch_col].astype(str).isin(exclude_list)
            mask_array = mask.values if hasattr(mask, 'values') else mask
            adata_filtered = adata[mask].copy()
            n_after = adata_filtered.n_obs
            n_excluded = n_before - n_after
            
            if n_excluded > 0:
                logger.info(f"Excluded batches {exclude_list}: {n_excluded} cells removed ({n_before} -> {n_after})")
                data_handler.adata = adata_filtered
                # Update labels if they exist
                if data_handler.labels is not None:
                    data_handler.labels = data_handler.labels[mask_array]
                # Update original copy too if exists
                if hasattr(data_handler, 'adata_original') and data_handler.adata_original is not None:
                    mask_orig = ~data_handler.adata_original.obs[batch_col].astype(str).isin(exclude_list)
                    data_handler.adata_original = data_handler.adata_original[mask_orig].copy()
            else:
                logger.warning(f"No cells matched excluded batches: {exclude_list}")
        else:
            logger.warning("Cannot exclude batches: no batch column detected")

    # Build reverse label map (encoded int -> original string) if available
    reverse_label_map = None
    if hasattr(data_handler, 'adata') and data_handler.adata is not None:
        lmap = data_handler.adata.uns.get('label_map', None)
        if lmap:
            try:
                    reverse_label_map = {int(v): k for k, v in lmap.items()}
            except ValueError:
                    reverse_label_map = {v: k for k, v in lmap.items()}
            logger.info(f"Loaded label map with {len(reverse_label_map)} entries")

    # Check for Benchmark Mode vs Standard Mode
    is_benchmark = args.benchmark_mode or config.get('benchmark_mode', False)

    # Batch correction behavior:
    # BIOINFORMATICS WORKFLOW: Batch correction must be applied AFTER HVG selection,
    # not before preprocessing. The correct order is:
    #   1. Cell/gene filtering
    #   2. Normalization + log-transform
    #   3. HVG selection (captures biological variation)
    #   4. Batch correction (removes technical variation on HVG subset)
    #   5. Scaling
    #
    # Therefore, we do NOT apply batch correction here at import.
    # Instead, we let _preprocess_builtin() handle it at the correct stage.
    # We only auto-detect the batch column if needed.
    if preprocessing_params.get('do_batch_correction', False):
        batch_key = preprocessing_params.get('batch_correction_batch_key', 'auto')
        if batch_key == 'auto':
            from utils.dataset_splitter import get_batch_column
            adata = data_handler.get_data()
            detected_batch_key = get_batch_column(adata)
            if detected_batch_key:
                preprocessing_params['batch_correction_batch_key'] = detected_batch_key
                logger.info(f"Auto-detected batch column: '{detected_batch_key}'")
            else:
                logger.warning("Could not auto-detect batch column. Batch correction may fail.")
                preprocessing_params['batch_correction_batch_key'] = 'batch'  # Fallback

        # NOTE: Batch correction will be applied during preprocessing (after HVG selection)
        logger.info("Batch correction enabled - will be applied after HVG selection during preprocessing.")
    
    if is_benchmark:
        logger.info(">>> BENCHMARK MODE ENABLED <<<")
        logger.info("Initializing train/test split and leakage-free preprocessing...")

        if args.train_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio < 0:
            logger.error("Invalid split ratios: train_ratio and test_ratio must be > 0; val_ratio must be >= 0.")
            return 1
        
        splitter = DatasetSplitter(random_state=random_seed)
        adata_raw = data_handler.get_data() # This is raw because no preprocessing happened yet

        
        # Auto-detect label column if not provided
        label_col = args.label_col
        if label_col is None:
            for candidate in ['cell_type', 'celltype', 'Group', 'labels', 'cluster', 'CellType']:
                if candidate in adata_raw.obs.columns:
                    label_col = candidate
                    logger.info(f"Auto-detected label column: '{label_col}'")
                    break
            
            if label_col is None:
                label_col = 'Group' # Default fallback
                logger.warning(f"Could not auto-detect label column. Using default '{label_col}'")
        else:
            logger.info(f"Using specified label column: '{label_col}'")
        
        # Balance batches logic (pre-split)
        balance_mode = getattr(args, 'balance_mode', 'eliminate')
        adata_surplus = None
        
        if args.balance_batches:
             batch_col = args.stratify_by  # Using stratify-by as batch col reuse
             if not batch_col:
                 # try to auto-detect batch
                 from utils.dataset_splitter import get_batch_column
                 batch_col = get_batch_column(adata_raw)

             if batch_col:
                 if balance_mode == 'eliminate':
                     logger.info(f"Balancing batches using {batch_col} (eliminate mode)...")
                     adata_raw = splitter.balance_by_batch(adata_raw, batch_col, balance_target_split)
                 
                 elif balance_mode in ['reinject_train', 'reinject_both']:
                     logger.info(f"Balancing batches using {batch_col} (reinject mode - pre-split extraction)...")
                     adata_raw, adata_surplus = splitter.balance_with_surplus(adata_raw, batch_col, balance_target_split)
                     logger.info(f"Retained {len(adata_raw)} core cells. Extracted {len(adata_surplus) if adata_surplus else 0} surplus cells for reinjection.")

        # Split Data
        batch_split = args.train_batches is not None
        
        if batch_split:
            train_batches = [b.strip() for b in args.train_batches.split(',') if b.strip()]
            test_batches = [b.strip() for b in args.test_batches.split(',') if b.strip()] if args.test_batches else None
            batch_col = args.stratify_by
            label_col = args.label_col

            if not train_batches:
                logger.error("--train-batches is set but no valid batch names were parsed.")
                return 1
            if args.test_batches is not None and not test_batches:
                logger.error("--test-batches is set but no valid batch names were parsed.")
                return 1

            # Strategy selection:
            # - No explicit test batches -> stratified base split, then filter train/val (test keeps all batches).
            # - Explicit test batches -> strict batch split (train on selected batches, test on explicit batches).
            use_stratified_base = test_batches is None
            if use_stratified_base:
                logger.info("Batch split strategy: stratified base (train/val filtered by batch, test from all batches).")
            else:
                logger.info("Batch split strategy: explicit batch sets (strict cross-batch/generalization mode).")
                logger.info(f"  train_batches={train_batches}")
                logger.info(f"  test_batches={test_batches}")
            
            # Parse stratify mode
            stratify_mode = getattr(args, 'stratify_mode', 'batch+labels')
            stratify_by_labels = stratify_mode in ['batch+labels', 'labels-only']

            split_result = splitter.split_by_batch(
                adata_raw,
                train_batches=train_batches,
                test_batches=test_batches,
                val_ratio=args.val_ratio,
                batch_col=batch_col,
                label_col=label_col,
                train_ratio=args.train_ratio,
                test_ratio=args.test_ratio,
                use_stratified_base=use_stratified_base,
                stratify_by_labels=stratify_by_labels
            )
        else: # Stratified
            # Parse stratify mode
            stratify_mode = getattr(args, 'stratify_mode', 'batch+labels')
            stratify_by_batch = stratify_mode in ['batch+labels', 'batch-only'] and args.stratify_by is not None
            stratify_by_labels = stratify_mode in ['batch+labels', 'labels-only']

            split_result = splitter.split_stratified(
                adata_raw,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                label_col=label_col,
                batch_col=args.stratify_by,
                stratify_by_batch=stratify_by_batch,
                stratify_by_labels=stratify_by_labels
            )
            
        stratify_mode_used = getattr(args, 'stratify_mode', 'batch+labels')
        logger.info(f"Stratification Mode: {stratify_mode_used}")
        logger.info(f"Split Summary: Train={split_result.split_info['n_train']}, Val={split_result.split_info['n_val']}, Test={split_result.split_info['n_test']}")

        # Post-split balancing/reinjection
        # Logic Change: If in reinject mode, we already extracted surplus. Now we merge it.
        # Original logic (balance_test_and_reinject) is replaced by this pre-split workflow.
        
        if adata_surplus is not None and balance_mode in ('reinject_train', 'reinject_both'):
            import anndata as ad
            reinject_to = 'train' if balance_mode == 'reinject_train' else 'both'
            n_surplus = len(adata_surplus)
            
            # Filter surplus if training on specific batches (Mono-batch purity)
            if batch_split: # If train_batches was used (legacy batch split or batch-filtered)
                # Even in batch-filtered mode, args.train_batches is a string
                if args.train_batches: 
                    train_batches_list = args.train_batches.split(',')
                    # Need to check batch column in surplus
                    # batch_col should be defined from args.stratify_by or auto-detected earlier
                    if batch_col in adata_surplus.obs.columns:
                        surplus_batches = adata_surplus.obs[batch_col]
                        mask = surplus_batches.isin(train_batches_list)
                        n_before = len(adata_surplus)
                        adata_surplus = adata_surplus[mask].copy()
                        n_after = len(adata_surplus)
                        n_surplus = n_after # Update count
                        if n_before != n_after:
                            logger.info(f"Filtered surplus for mono-batch purity: kept {n_after}/{n_before} cells from batches {train_batches_list}")
                    else:
                        logger.warning(f"Batch column '{batch_col}' not found in surplus. Cannot filter by train batches.")

            logger.info(f"Reinjecting {n_surplus} surplus cells into {reinject_to.upper()}...")
            
            if reinject_to == 'train':
                new_train = ad.concat([split_result.adata_train, adata_surplus], join='outer')
                # Update split result
                split_result.adata_train = new_train
                # Update indices (recalc needed but skipping for simplicity as simple concat)
                split_result.split_info['n_train'] += n_surplus
                
            elif reinject_to == 'both':
                # Split surplus proportionally to train/val ratio
                n_train = split_result.split_info['n_train']
                n_val = split_result.split_info['n_val']
                if n_val == 0:
                    # All to train
                     new_train = ad.concat([split_result.adata_train, adata_surplus], join='outer')
                     split_result.adata_train = new_train
                     split_result.split_info['n_train'] += n_surplus
                else:
                    ratio = n_train / (n_train + n_val)
                    n_to_train = int(n_surplus * ratio)
                    
                    # Random split of surplus
                    np.random.seed(random_seed)
                    indices = np.arange(n_surplus)
                    np.random.shuffle(indices)
                    
                    surplus_train = adata_surplus[indices[:n_to_train]]
                    surplus_val = adata_surplus[indices[n_to_train:]]
                    
                    split_result.adata_train = ad.concat([split_result.adata_train, surplus_train], join='outer')
                    split_result.adata_val = ad.concat([split_result.adata_val, surplus_val], join='outer')
                    
                    split_result.split_info['n_train'] += len(surplus_train)
                    split_result.split_info['n_val'] += len(surplus_val)

            split_result.split_info['n_reinjected'] = n_surplus
            logger.info(f"Reinjection Complete. Final Train: {split_result.split_info['n_train']}, "
                        f"Final Test: {split_result.split_info['n_test']} (Balanced)")

        # Legacy logic fallback (if someone uses old reinject without new flow? Unlikely given arg parsing)
        # We keep the old block only if balance_test_and_reinject is explicitly desired, which it isn't anymore.
        # So we leave this block empty or removed.


        # Preprocess (Fit on Train, Transform on all)
        if not preprocessing_params.get('skip', False):
            logger.info("Applying Benchmark Preprocessing (Fit on Train)...")
            
            # Add batch_col to preprocessing params for per_batch_union HVG strategy
            if args.stratify_by:
                preprocessing_params['batch_col'] = args.stratify_by
            
            preprocessor = BenchmarkPreprocessor()
            
            # For 'all_data' strategy, we need to fit on ALL data first (before split)
            hvg_strategy = preprocessing_params.get('hvg_strategy', 'train_only')
            if hvg_strategy == 'all_data':
                logger.info("HVG Strategy: Selecting HVGs from ALL data (before split)...")
                # First pass: fit on full data to get HVGs
                preprocessor.fit(adata_raw, preprocessing_params)
            else:
                # Default: fit on training data only
                preprocessor.fit(split_result.adata_train, preprocessing_params)
            
            # Transform
            adata_train = preprocessor.transform(split_result.adata_train)
            adata_test = preprocessor.transform(split_result.adata_test)
            adata_val = preprocessor.transform(split_result.adata_val) if split_result.adata_val else None
        else:
             logger.info("Skipping preprocessing (using raw split data)...")
             adata_train = split_result.adata_train
             adata_test = split_result.adata_test
             adata_val = split_result.adata_val

        # Get Labels from split result objects
        labels_train = split_result.get_labels('train')
        labels_test = split_result.get_labels('test')
        labels_val = split_result.get_labels('val') if adata_val else None

        # Save preprocessed split data (UI parity)
        try:
            bench_data_dir = data_dir / 'benchmark'
            bench_data_dir.mkdir(parents=True, exist_ok=True)
            adata_train.write_h5ad(bench_data_dir / 'train.h5ad', compression='gzip')
            adata_test.write_h5ad(bench_data_dir / 'test.h5ad', compression='gzip')
            if adata_val is not None:
                adata_val.write_h5ad(bench_data_dir / 'val.h5ad', compression='gzip')
            logger.info(f"Saved benchmark preprocessed splits to {bench_data_dir}")
        except Exception as e:
            logger.warning(f"Could not save benchmark split data: {e}")
        
        # Run Analysis
        logger.info(f"\nRunning benchmark analysis with {len(algorithms)} algorithm(s)...")
        print("-" * 60)
        
        runner = AnalysisRunner(output_dir=results_dir)
        from utils.analysis_runner import BenchmarkComparisonResult
        import numpy as np
        
        def progress_callback(message: str, progress: float = None):
             if verbose: print(f"  {message}")
        
        # Split algorithms: transductive (graph-based) get special treatment
        inductive_algos = []
        transductive_algos = []
        
        from core.algorithm_registry import AlgorithmRegistry
        for algo_name in algorithms:
            algo_class = AlgorithmRegistry.get(algo_name)
            if algo_class:
                info = algo_class.get_info()
                if not getattr(info, "supports_out_of_sample", True):
                    transductive_algos.append(algo_name)
                else:
                    inductive_algos.append(algo_name)
            else:
                inductive_algos.append(algo_name)

        all_benchmark_results = []
        
        # 1. Run inductive algorithms (with Train/Test split)
        if inductive_algos:
            try:
                results_ind = runner.run_benchmark_comparison(
                    algorithm_names=inductive_algos,
                    data_train=adata_train,
                    data_test=adata_test,
                    labels_train=labels_train,
                    labels_test=labels_test,
                    data_val=adata_val,
                    labels_val=labels_val,
                    params=algorithm_params,
                    batch_col=args.stratify_by,
                    n_repeats=n_repeats,
                    compute_scib_metrics=compute_scib_metrics,
                    progress_callback=progress_callback,
                    search_spaces=search_spaces
                )
                all_benchmark_results.extend(results_ind.results)
            except Exception as e:
                logger.error(f"Inductive benchmark failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        # 2. Run transductive algorithms (on full dataset, mirroring UI behavior)
        if transductive_algos:
            logger.warning(f"⚠️ TRANSDUCTIVE ALGORITHMS: {', '.join(transductive_algos)}")
            logger.warning("   These graph-based algorithms train on the FULL dataset (including test cells).")
            logger.warning("   Their 'test' metrics are NOT generalizable. Use only for clustering quality assessment.")
            print(f"\n{'='*60}")
            print("⚠️  TRANSDUCTIVE MODE (No Out-of-Sample Prediction)")
            print(f"{'='*60}")
            try:
                # We need the full data (preprocessed if possible, or raw if fallback)
                # For transductive parity with UI, we concatenate or use pre-split raw
                # Actually, the simplest for parity is to run on the full preprocessed data
                # but we only have preprocessed splits. Let's reconstruct full preprocessed if possible.
                import anndata as ad
                parts = [adata_train, adata_val, adata_test]
                parts = [p for p in parts if p is not None]
                adata_full = ad.concat(parts, axis=0) if len(parts) > 1 else parts[0].copy()
                labels_full = np.concatenate([l for l in [labels_train, labels_val, labels_test] if l is not None])
                
                results_trans = runner.run_benchmark_comparison(
                    algorithm_names=transductive_algos,
                    data_train=adata_full,
                    data_test=adata_full,
                    labels_train=labels_full,
                    labels_test=labels_full,
                    data_val=None,
                    labels_val=None,
                    params=algorithm_params,
                    batch_col=args.stratify_by,
                    n_repeats=n_repeats,
                    compute_scib_metrics=compute_scib_metrics,
                    progress_callback=progress_callback,
                    search_spaces=None # HPO usually disabled for transductive in this fallback
                )
                all_benchmark_results.extend(results_trans.results)
            except Exception as e:
                logger.error(f"Transductive fallback failed: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        if not all_benchmark_results:
            logger.error("No benchmark results generated.")
            return 1

        # Create combined result object
        results = BenchmarkComparisonResult(
            results=all_benchmark_results,
            summary=runner._compute_benchmark_summary(all_benchmark_results)
        )

        try:

            # Print Summary specialized for benchmark
            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS SUMMARY")
            print("=" * 60)
            print(f"{'Algorithm':<20} {'Train NMI':<10} {'Test NMI':<10} {'Gap':<10} {'Time':<10}")
            print("-" * 60)
            for algo_name, stats in results.summary.items():
                train_nmi = stats.get('train_NMI_mean', 0)
                test_nmi = stats.get('test_NMI_mean', 0)
                gap = stats.get('NMI_gap_mean', train_nmi - test_nmi)
                time_s = stats.get('fit_time_mean', 0) + stats.get('predict_time_mean', 0)
                print(f"{algo_name:<20} {train_nmi:.4f}     {test_nmi:.4f}     {gap:.4f}     {time_s:.2f}s")
    
                if n_repeats > 1:
                    train_std = stats.get('train_NMI_std', 0)
                    test_std = stats.get('test_NMI_std', 0)
                    gap_std = stats.get('NMI_gap_std', 0)
                    print(f"{'':20} ±{train_std:.4f}     ±{test_std:.4f}     ±{gap_std:.4f}")
                    
                    
            # Generate Benchmark Plots
            logger.info("Generating benchmark figures...")
            figures_dir.mkdir(parents=True, exist_ok=True)
            import matplotlib.pyplot as plt
            
            # 1. Train vs Test Comparison
            fig = viz.plot_benchmark_comparison(results.summary)
            if fig:
                fig.savefig(figures_dir / 'benchmark_comparison.png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                
            # 2. Generalization Gap
            fig = viz.plot_generalization_gap(results.summary)
            if fig:
                fig.savefig(figures_dir / 'generalization_gap.png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                
            # 3. Batch Metrics Heatmaps (if applicable)
            # Aggregate batch metrics by algorithm across all runs, then compute MEAN
            import pandas as pd
            from collections import defaultdict

            algo_batch_metrics = defaultdict(lambda: defaultdict(lambda: {'NMI': [], 'ARI': [], 'ACC': []}))

            for result in results.results:
                if result.benchmark_metrics.test_by_group:
                    algo = result.algorithm_name
                    for group_name, metrics in result.benchmark_metrics.test_by_group.items():
                        for metric_name in ['NMI', 'ARI', 'ACC']:
                            if metric_name in metrics:
                                algo_batch_metrics[algo][group_name][metric_name].append(metrics[metric_name])

            # Generate one heatmap per algorithm with MEAN values
            for algo, batches in algo_batch_metrics.items():
                group_data = []
                n_runs = 0
                for group_name in sorted(batches.keys()):
                    row = {'Batch': group_name}
                    for metric_name in ['NMI', 'ARI', 'ACC']:
                        values = batches[group_name][metric_name]
                        if values:
                            row[metric_name] = sum(values) / len(values)
                            n_runs = max(n_runs, len(values))
                    group_data.append(row)

                if group_data:
                    df_groups = pd.DataFrame(group_data)
                    fig = viz.plot_batch_metrics_heatmap(
                        df_groups,
                        title=f'Metrics by Batch - {algo}\n(Mean over {n_runs} runs)'
                    )
                    if fig:
                        safe_algo_name = algo.replace(' ', '_').replace('/', '_')
                        fig.savefig(figures_dir / f'batch_metrics_{safe_algo_name}.png', bbox_inches='tight', dpi=150)
                        plt.close(fig)

            # 3b. Cell type error by batch heatmaps (if available)
            algo_ct_batch_errors = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for result in results.results:
                error_info = result.benchmark_metrics.error_analysis
                if not error_info:
                    continue
                ct_by_group = error_info.get('error_by_celltype_by_group', {})
                if not ct_by_group:
                    continue
                algo = result.algorithm_name
                for batch, ct_map in ct_by_group.items():
                    for ct, ct_data in ct_map.items():
                        if isinstance(ct_data, dict) and 'error_rate' in ct_data:
                            algo_ct_batch_errors[algo][batch][ct].append(ct_data['error_rate'])

            for algo, batch_map in algo_ct_batch_errors.items():
                mean_dict = {}
                n_runs = 0
                for batch, ct_map in batch_map.items():
                    mean_dict[batch] = {}
                    for ct, values in ct_map.items():
                        if values:
                            mean_dict[batch][ct] = {'error_rate': sum(values) / len(values)}
                            n_runs = max(n_runs, len(values))
                if mean_dict:
                    fig = viz.plot_celltype_errors_by_batch(
                        mean_dict,
                        title=f'Error by Cell Type and Batch - {algo}\n(Mean over {n_runs} runs)'
                    )
                    if fig:
                        safe_algo_name = algo.replace(' ', '_').replace('/', '_')
                        fig.savefig(figures_dir / f'celltype_error_by_batch_{safe_algo_name}.png', bbox_inches='tight', dpi=150)
                        plt.close(fig)

            # 4. Radar Chart (GUI parity)
            try:
                fig = viz.plot_radar_chart(results.summary)
                if fig:
                    fig.savefig(figures_dir / 'radar_chart.png', bbox_inches='tight', dpi=150)
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not generate radar chart: {e}")

            # 5. UMAP Embeddings for Train and Test (separately)
            logger.info("Generating UMAP visualizations for train and test sets...")

            # Build dataset info and params info for UMAP annotations
            from pathlib import Path as _Path
            _dataset_info_str = f"{_Path(data_file).stem} | Train/Test split"

            def _build_params_info(result):
                p = result.params or {}
                return {
                    'normalization': p.get('nb_input_transform', 'log1p'),
                    'DANN_weight': p.get('adversarial_batch_weight', 0),
                    'MMD_weight': p.get('mmd_batch_weight', 0),
                    'clustering': p.get('clustering_method', 'hdbscan'),
                    'HVG_flavor': p.get('internal_hvg_flavor', 'seurat'),
                    'epochs': p.get('epochs', '?'),
                    'z_dim': p.get('z_dim', '?'),
                }

            for result in results.results:
                safe_algo_name = result.algorithm_name.replace(' ', '_').replace('/', '_')
                _pinfo = _build_params_info(result)

                # UMAP comparison for Train embeddings (ground truth vs predicted)
                if result.train_embeddings is not None and len(result.train_embeddings) > 0:
                    try:
                        train_true = (
                            result.train_true_labels
                            if result.train_true_labels is not None
                            else labels_train
                        )
                        train_pred = result.train_labels
                        if (
                            train_true is not None
                            and train_pred is not None
                            and len(train_true) == len(result.train_embeddings)
                            and len(train_pred) == len(result.train_embeddings)
                        ):
                            fig = viz.plot_umap_comparison(
                                embeddings=result.train_embeddings,
                                true_labels=train_true,
                                predicted_labels=train_pred,
                                algorithm_name=f"{result.algorithm_name} - Train",
                                label_names=reverse_label_map,
                                params_info=_pinfo,
                                dataset_info=_dataset_info_str,
                            )
                        else:
                            fig = viz.plot_umap_embeddings(
                                result.train_embeddings,
                                train_true if train_true is not None else train_pred,
                                title=f"{result.algorithm_name} - Train Set",
                                predicted_labels=None,
                                show_cluster_hulls=False,
                                show_cluster_centroids=False,
                                show_label_centroids=True,
                                label_names=reverse_label_map,
                                params_info=_pinfo,
                                dataset_info=_dataset_info_str,
                            )
                        if fig:
                            fig.savefig(figures_dir / f'umap_{safe_algo_name}_train.png', bbox_inches='tight', dpi=150)
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Could not generate train UMAP for {result.algorithm_name}: {e}")

                # UMAP comparison for Test embeddings (ground truth vs predicted)
                if result.test_embeddings is not None and len(result.test_embeddings) > 0:
                    try:
                        test_true = (
                            result.test_true_labels
                            if result.test_true_labels is not None
                            else labels_test
                        )
                        test_pred = result.test_labels
                        if (
                            test_true is not None
                            and test_pred is not None
                            and len(test_true) == len(result.test_embeddings)
                            and len(test_pred) == len(result.test_embeddings)
                        ):
                            fig = viz.plot_umap_comparison(
                                embeddings=result.test_embeddings,
                                true_labels=test_true,
                                predicted_labels=test_pred,
                                algorithm_name=f"{result.algorithm_name} - Test",
                                label_names=reverse_label_map,
                                params_info=_pinfo,
                                dataset_info=_dataset_info_str,
                            )
                        else:
                            fig = viz.plot_umap_embeddings(
                                result.test_embeddings,
                                test_true if test_true is not None else test_pred,
                                title=f"{result.algorithm_name} - Test Set",
                                predicted_labels=None,
                                show_cluster_hulls=False,
                                show_cluster_centroids=False,
                                show_label_centroids=True,
                                label_names=reverse_label_map,
                                params_info=_pinfo,
                                dataset_info=_dataset_info_str,
                            )

                        if fig:
                            fig.savefig(figures_dir / f'umap_{safe_algo_name}_test.png', bbox_inches='tight', dpi=150)
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Could not generate test UMAP for {result.algorithm_name}: {e}")

                # Final weighted UMAP (train): ground truth + per-cell alpha from reconstruction weights
                if result.train_embeddings is not None and result.embedding_snapshots:
                    train_labels_for_plot = (
                        result.train_true_labels
                        if result.train_true_labels is not None
                        else result.train_labels
                    )
                    if train_labels_for_plot is not None and len(train_labels_for_plot) == len(result.train_embeddings):
                        final_weights = None
                        for _snap in reversed(result.embedding_snapshots):
                            _w = _snap.get("cell_weights")
                            if _w is not None and len(_w) == len(result.train_embeddings):
                                final_weights = _w
                                break
                        if final_weights is not None:
                            try:
                                fig = viz.plot_umap_weighted(
                                    result.train_embeddings,
                                    train_labels_for_plot,
                                    final_weights,
                                    title=f"{result.algorithm_name} - Train (Cell Weights)",
                                    label_names=reverse_label_map,
                                    params_info=_pinfo,
                                    dataset_info=_dataset_info_str,
                                )
                                if fig:
                                    fig.savefig(
                                        figures_dir / f'umap_{safe_algo_name}_train_weighted.png',
                                        bbox_inches='tight',
                                        dpi=150,
                                    )
                                    plt.close(fig)

                                fig_grad = viz.plot_umap_weighted_gradient(
                                    result.train_embeddings,
                                    final_weights,
                                    title=f"{result.algorithm_name} - Train (Cell Weights Gradient)",
                                    params_info=_pinfo,
                                    dataset_info=_dataset_info_str,
                                )
                                if fig_grad:
                                    fig_grad.savefig(
                                        figures_dir / f'umap_{safe_algo_name}_train_weighted_gradient.png',
                                        bbox_inches='tight',
                                        dpi=150,
                                    )
                                    plt.close(fig_grad)
                            except Exception as e:
                                logger.warning(f"Could not generate weighted UMAP for {result.algorithm_name}: {e}")

            # Save loss history (JSON + PNG)
            try:
                has_any_loss = any(r.loss_history for r in results.results)
                if has_any_loss:
                    loss_dir = results_dir / 'loss_history'
                    loss_dir.mkdir(parents=True, exist_ok=True)
                    for result in results.results:
                        if result.loss_history:
                            safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                            # Save JSON
                            loss_json = {
                                'algorithm': result.algorithm_name,
                                'run_id': result.run_id,
                                'phases': result.loss_history,
                            }
                            with open(loss_dir / f'loss_{safe_algo}_run{result.run_id}.json', 'w') as f:
                                json.dump(loss_json, f, indent=2)
                            # Save PNG
                            fig = viz.plot_loss_curves(result.loss_history, result.algorithm_name)
                            if fig:
                                fig.savefig(figures_dir / f'loss_curves_{safe_algo}_run{result.run_id}.png',
                                            bbox_inches='tight', dpi=150)
                                plt.close(fig)
                    logger.info(f"Loss history saved to {loss_dir}")
            except Exception as e:
                logger.warning(f"Could not save loss history: {e}")

            try:
                _save_weight_history_artifacts(
                    run_results=results.results,
                    results_dir=results_dir,
                    figures_dir=figures_dir,
                )
            except Exception as e:
                logger.warning(f"Could not save weight history: {e}")

            # Save UMAP evolution snapshots (PNG) — ground truth every 10 epochs
            try:
                _umap_evo_cfg = _get_umap_evolution_render_config()
                has_any_umap_evolution = any(r.embedding_snapshots for r in results.results)
                if has_any_umap_evolution:
                    exported = 0
                    for result in results.results:
                        snapshots = result.embedding_snapshots or []
                        if not snapshots:
                            continue
                        snapshots_for_plot = _select_periodic_snapshots_every_10(snapshots)
                        if not snapshots_for_plot:
                            continue

                        first_valid_snapshot = next(
                            (
                                s for s in snapshots_for_plot
                                if s.get('embeddings') is not None and len(s.get('embeddings')) > 0
                            ),
                            None
                        )
                        if first_valid_snapshot is None:
                            continue

                        labels_for_plot = (
                            result.train_true_labels
                            if result.train_true_labels is not None
                            else result.train_labels
                        )
                        if labels_for_plot is None or len(labels_for_plot) != len(first_valid_snapshot.get('embeddings')):
                            logger.warning(
                                "Skipping UMAP evolution export for %s run %s: labels/snapshot size mismatch.",
                                result.algorithm_name,
                                result.run_id,
                            )
                            continue

                        safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                        _pinfo_evo = _build_params_info(result)
                        _color_mode = _umap_evo_cfg["color_mode"]
                        _cell_weights_for_plot = None
                        _batch_labels_for_plot = (
                            np.asarray(result.train_batch_ids)
                            if result.train_batch_ids is not None
                            else None
                        )
                        if _color_mode == "ground_truth_weighted":
                            _weights_ok = True
                            _cell_weights_for_plot = []
                            for _snap in snapshots_for_plot:
                                _w = _snap.get("cell_weights")
                                _e = _snap.get("embeddings")
                                if _w is None or _e is None or len(_w) != len(_e):
                                    _weights_ok = False
                                    break
                                _cell_weights_for_plot.append(_w)
                            if not _weights_ok:
                                logger.warning(
                                    "Falling back to ground_truth UMAP evolution for %s run %s: "
                                    "missing/mismatched cell_weights in snapshots.",
                                    result.algorithm_name,
                                    result.run_id,
                                )
                                _color_mode = "ground_truth"
                                _cell_weights_for_plot = None
                        elif _color_mode == "batch":
                            if _batch_labels_for_plot is None or len(_batch_labels_for_plot) != len(first_valid_snapshot.get('embeddings')):
                                logger.warning(
                                    "Falling back to ground_truth UMAP evolution for %s run %s: "
                                    "batch labels unavailable/mismatched.",
                                    result.algorithm_name,
                                    result.run_id,
                                )
                                _color_mode = "ground_truth"
                                _batch_labels_for_plot = None

                        fig = viz.plot_umap_evolution(
                            embedding_snapshots=snapshots_for_plot,
                            labels=labels_for_plot,
                            algorithm_name=f"{result.algorithm_name} (run {result.run_id + 1})",
                            color_mode=_color_mode,
                            cell_weights_per_snapshot=_cell_weights_for_plot,
                            batch_labels=_batch_labels_for_plot,
                            point_size=_umap_evo_cfg["point_size"],
                            max_points=_umap_evo_cfg["max_points"],
                            max_snapshots=0,
                            params_info=_pinfo_evo,
                            dataset_info=_dataset_info_str,
                        )
                        if fig:
                            fig.savefig(
                                figures_dir / f'umap_evolution_{safe_algo}_run{result.run_id}.png',
                                bbox_inches='tight',
                                dpi=150,
                            )
                            plt.close(fig)
                            exported += 1
                    if exported > 0:
                        logger.info("UMAP evolution figures saved to %s (%d figure(s)).", figures_dir, exported)
            except Exception as e:
                logger.warning(f"Could not save UMAP evolution figures: {e}")

            # Save benchmark results JSON (with actual benchmark data)
            try:
                benchmark_json_data = {
                    'results': [res.to_dict() for res in results.results],
                    'summary': results.summary,
                    'timestamp': datetime.now().isoformat()
                }
                with open(results_dir / 'benchmark_results.json', 'w') as f:
                    json.dump(benchmark_json_data, f, indent=2, default=str)
                logger.info(f"Saved benchmark results to {results_dir / 'benchmark_results.json'}")
            except Exception as e:
                logger.warning(f"Could not save benchmark_results.json: {e}")

            # Save benchmark summary and detailed CSV (UI parity)
            try:
                import pandas as pd
                from core.algorithm_registry import AlgorithmRegistry

                summary_rows = []
                for algo_name, stats in results.summary.items():
                    algo_info = AlgorithmRegistry.get(algo_name).get_info()
                    row = {'Algorithm': algo_info.display_name}
                    row.update(stats)
                    summary_rows.append(row)
                df_summary = pd.DataFrame(summary_rows)
                df_summary.to_csv(results_dir / 'benchmark_summary.csv', index=False)

                detail_data = [res.to_dict() for res in results.results]
                df_detail = pd.DataFrame(detail_data)
                for col in ['train_metrics', 'test_metrics', 'generalization_gap']:
                    if col in df_detail.columns:
                        expanded = df_detail[col].apply(pd.Series)
                        expanded.columns = [f'{col}_{c}' for c in expanded.columns]
                        df_detail = pd.concat([df_detail.drop(col, axis=1), expanded], axis=1)
                df_detail.to_csv(results_dir / 'benchmark_detailed.csv', index=False)
            except Exception as e:
                logger.warning(f"Could not save benchmark CSV outputs: {e}")

            # Save benchmark labels by split (both true and predicted)
            try:
                for result in results.results:
                    safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                    # Train labels
                    train_df = {'predicted_label': result.train_labels}
                    if result.train_true_labels is not None:
                        train_df['true_label'] = result.train_true_labels
                    if result.train_batch_ids is not None:
                        train_df['batch'] = result.train_batch_ids
                    pd.DataFrame(train_df).to_csv(
                        labels_dir / f"benchmark_{safe_algo}_run{result.run_id}_train.csv",
                        index=False
                    )
                    # Test labels
                    test_df = {'predicted_label': result.test_labels}
                    if result.test_true_labels is not None:
                        test_df['true_label'] = result.test_true_labels
                    if result.test_batch_ids is not None:
                        test_df['batch'] = result.test_batch_ids
                    pd.DataFrame(test_df).to_csv(
                        labels_dir / f"benchmark_{safe_algo}_run{result.run_id}_test.csv",
                        index=False
                    )
                    # Validation labels
                    if result.val_labels is not None:
                        val_df = {'predicted_label': result.val_labels}
                        if result.val_true_labels is not None:
                            val_df['true_label'] = result.val_true_labels
                        if result.val_batch_ids is not None:
                            val_df['batch'] = result.val_batch_ids
                        pd.DataFrame(val_df).to_csv(
                            labels_dir / f"benchmark_{safe_algo}_run{result.run_id}_val.csv",
                            index=False
                        )
            except Exception as e:
                logger.warning(f"Could not save benchmark labels: {e}")

            effective_algo_params, effective_per_run_params = _extract_effective_params_from_run_results(
                results.results
            )

            # Save benchmark config used
            config_used = {
                'data': {'file': str(data_file)},
                'preprocessing': preprocessing_params,
                'algorithms': algorithms,
                'algorithm_params': algorithm_params,
                'algorithm_effective_params_by_algorithm': effective_algo_params,
                'algorithm_effective_params_per_run': effective_per_run_params,
                'execution': {
                    'device': device,
                    'n_repeats': n_repeats,
                    'random_seed': random_seed,
                    'compute_scib_metrics': compute_scib_metrics,
                },
                'benchmark': {
                    'mode': 'benchmark',
                    'train_ratio': args.train_ratio,
                    'val_ratio': args.val_ratio,
                    'test_ratio': args.test_ratio,
                    'train_batches': args.train_batches,
                    'test_batches': args.test_batches,
                    'batch_col': args.stratify_by,
                    'stratify_mode': getattr(args, 'stratify_mode', 'batch+labels'),
                    'balance_batches': args.balance_batches,
                    'balance_target': args.balance_target,
                    'balance_mode': getattr(args, 'balance_mode', 'eliminate'),
                    'split_info': split_result.split_info,
                },
                'output': {'directory': str(output_dir)},
            }
            config_file = config_dir / 'config_used.json'
            with open(config_file, 'w') as f:
                json.dump(config_used, f, indent=2, default=str)
            logger.info(f"Configuration saved to {config_file}")

            try:
                _save_algorithm_hyperparams_snapshot(
                    config_dir=config_dir,
                    algorithms=algorithms,
                    algorithm_params=algorithm_params,
                    run_results=results.results,
                    search_spaces=search_spaces,
                    execution={
                        'device': device,
                        'n_repeats': n_repeats,
                        'random_seed': random_seed,
                        'compute_scib_metrics': compute_scib_metrics,
                    },
                    context={
                        'mode': 'benchmark',
                        'data_file': str(data_file),
                        'output_dir': str(output_dir),
                        'status': 'completed',
                        'split_info': split_result.split_info,
                    },
                )
            except Exception as e:
                logger.warning(f"Could not save completed hyperparameter snapshot: {e}")
            return 0
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    else:
        # --- STANDARD MODE (Original Logic) ---

        # Preprocess data
        if not preprocessing_params.get('skip', False):
            logger.info("Preprocessing data (Global)...")
            data_handler.preprocess(preprocessing_params)
            info = data_handler.get_info()
            logger.info(f"After preprocessing: {info['n_cells']} cells, {info['n_genes']} genes")
        else:
            logger.info("Skipping preprocessing as requested")

        # Apply batch balancing if requested (standard mode)
        balance_standard = getattr(args, 'balance_batches_standard', False)
        if balance_standard:
            from utils.dataset_splitter import DatasetSplitter, get_batch_column

            adata = data_handler.get_data()
            original_n_cells = adata.n_obs

            # Determine batch column
            batch_col = getattr(args, 'batch_col_standard', None)
            if batch_col is None or batch_col == 'auto':
                batch_col = get_batch_column(adata)
                if batch_col is None:
                    logger.error("Could not auto-detect batch column for balancing! Balancing will be SKIPPED.")
            
            if batch_col:
                splitter = DatasetSplitter(random_state=random_seed)
                target = balance_target_standard

                # Determine label column for stratified subsampling
                label_col = None
                for candidate in ['Group', 'labels', 'cell_type', 'cluster']:
                    if candidate in adata.obs.columns:
                        label_col = candidate
                        break

                logger.info(f"Balancing batches using column '{batch_col}'...")
                balanced_adata = splitter.balance_by_batch(
                    adata,
                    batch_col=batch_col,
                    target_per_batch=target,
                    preserve_labels=True,
                    label_col=label_col or 'Group'
                )

                # Update handler with balanced data
                data_handler.adata = balanced_adata
                # data_handler._preprocessed_adata = balanced_adata # Removed as likely unused/wrong

                logger.info(f"Balanced data: {balanced_adata.n_obs} cells (from {original_n_cells})")
            else:
                logger.error("No batch column found (and none specified). Skipping batch balancing!")

        # Get data and labels
        adata = data_handler.get_data()
        labels = data_handler.get_labels()

        # Save preprocessed data (UI parity)
        try:
            adata.write_h5ad(data_dir / 'processed.h5ad', compression='gzip')
            logger.info(f"Saved preprocessed data to {data_dir / 'processed.h5ad'}")
        except Exception as e:
            logger.warning(f"Could not save preprocessed data: {e}")
    
        # Run analysis (Standard)
        logger.info(f"\nRunning analysis with {len(algorithms)} algorithm(s), {n_repeats} repeat(s)...")
        print("-" * 60)
    
        runner = AnalysisRunner(output_dir=results_dir)
    
        def progress_callback(message: str, progress: float = None):
            if verbose:
                if progress is not None:
                    print(f"  [{progress*100:.0f}%] {message}")
                else:
                    print(f"  {message}")
    
        try:
            comparison_result = runner.run_comparison(
                algorithm_names=algorithms,
                data=data_handler,
                labels=None,
                params=algorithm_params,
                n_repeats=n_repeats,
                compute_scib_metrics=compute_scib_metrics,
                progress_callback=progress_callback
            )
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
        # Print results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
    
        if labels is not None:
            print(f"\n{'Algorithm':<20} {'NMI':<10} {'ARI':<10} {'ACC':<10} {'Time (s)':<10}")
            print("-" * 60)
    
            for algo_name, stats in comparison_result.summary.items():
                nmi = f"{stats.get('NMI_mean', 0):.4f}"
                ari = f"{stats.get('ARI_mean', 0):.4f}"
                acc = f"{stats.get('ACC_mean', 0):.4f}"
                time_s = f"{stats.get('runtime_mean', 0):.2f}"
                print(f"{algo_name:<20} {nmi:<10} {ari:<10} {acc:<10} {time_s:<10}")
    
                if n_repeats > 1:
                    nmi_std = f"±{stats.get('NMI_std', 0):.4f}"
                    ari_std = f"±{stats.get('ARI_std', 0):.4f}"
                    acc_std = f"±{stats.get('ACC_std', 0):.4f}"
                    print(f"{'':20} {nmi_std:<10} {ari_std:<10} {acc_std:<10}")
        else:
            print(f"\n{'Algorithm':<20} {'Silhouette':<12} {'Clusters':<10} {'Time (s)':<10}")
            print("-" * 60)
    
            for algo_name, stats in comparison_result.summary.items():
                sil = f"{stats.get('Silhouette_mean', 0):.4f}"
                n_clust = f"{stats.get('n_clusters_found_mean', 0):.0f}"
                time_s = f"{stats.get('runtime_mean', 0):.2f}"
                print(f"{algo_name:<20} {sil:<12} {n_clust:<10} {time_s:<10}")
                if n_repeats > 1:
                    sil_std = f"±{stats.get('Silhouette_std', 0):.4f}"
                    n_clust_std = f"±{stats.get('n_clusters_found_std', 0):.2f}"
                    time_std = f"±{stats.get('runtime_std', 0):.2f}"
                    print(f"{'':20} {sil_std:<12} {n_clust_std:<10} {time_std:<10}")
    
        print("=" * 60)
    
        # Find best algorithm
        if labels is not None:
            best = comparison_result.get_best_algorithm('NMI')
            if best:
                print(f"\nBest algorithm (by NMI): {best}")
                
        if metrics_only_mode:
            logger.info("SCRBENCHMARK_METRICS_ONLY=1 -> skipping figures, UMAPs and marker-overlap exports.")
        else:
            logger.info("Generating figure plots...")
        figures_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt
        
        # 1. Metrics Comparison
        if (not metrics_only_mode) and len(comparison_result.results) > 1 and labels is not None:
            fig = viz.plot_metrics_comparison(comparison_result.summary, comparison_result.results)
            if fig:
                fig.savefig(figures_dir / 'metrics_comparison.png', bbox_inches='tight', dpi=150)
                import matplotlib.pyplot as plt
                plt.close(fig)
                
        # 2. UMAP Embeddings (if available)

        # Note: reverse_label_map is already built earlier (lines 600+)

        # Build dataset info and params info for UMAP annotations (standard mode)
        from pathlib import Path as _Path
        _dataset_info_str_std = f"{_Path(data_file).stem} | Full data"

        def _build_params_info_std(result):
            p = result.params or {}
            return {
                'normalization': p.get('nb_input_transform', 'log1p'),
                'DANN_weight': p.get('adversarial_batch_weight', 0),
                'MMD_weight': p.get('mmd_batch_weight', 0),
                'clustering': p.get('clustering_method', 'hdbscan'),
                'HVG_flavor': p.get('internal_hvg_flavor', 'seurat'),
                'epochs': p.get('epochs', '?'),
                'z_dim': p.get('z_dim', '?'),
            }

        def _resolve_batch_labels_std(result) -> Optional[np.ndarray]:
            if isinstance(result.extra_info, dict):
                batch_values = result.extra_info.get('batch_labels')
                if batch_values is not None:
                    arr = np.asarray(batch_values)
                    if len(arr) == len(result.embeddings):
                        return arr

            if hasattr(adata, 'obs'):
                batch_col_local = None
                for candidate in ['batch', 'Batch', 'tech', 'study', 'dataset', 'donor', 'sample']:
                    if candidate in adata.obs.columns:
                        batch_col_local = candidate
                        break
                if batch_col_local is not None:
                    try:
                        arr = np.asarray(adata.obs[batch_col_local].astype(str).to_numpy())
                        if len(arr) == len(result.embeddings):
                            return arr
                    except Exception:
                        pass
            return None

        for result in comparison_result.results:
            if result.embeddings is not None and len(result.embeddings) > 0:
                if metrics_only_mode:
                    continue
                # Use aligned ground truth labels if available, else predicted labels
                # result.true_labels is populated by AnalysisRunner and aligned with embeddings
                if result.true_labels is not None:
                     plot_labels = result.true_labels
                elif labels is not None and len(labels) == len(result.embeddings):
                     # Fallback to global labels if lengths match (legacy behavior)
                     plot_labels = labels
                else:
                     plot_labels = result.labels

                safe_algo_name = result.algorithm_name.replace(' ', '_').replace('/', '_')
                _pinfo_std = _build_params_info_std(result)
                gt_labels = result.true_labels if result.true_labels is not None else plot_labels
                pred_labels = result.labels
                batch_labels_std = _resolve_batch_labels_std(result)
                batch_col_std = (
                    result.extra_info.get('batch_col', 'batch')
                    if isinstance(result.extra_info, dict)
                    else 'batch'
                )

                # Generate side-by-side UMAP comparison: Ground Truth vs Predicted
                try:
                    fig = viz.plot_umap_comparison(
                        embeddings=result.embeddings,
                        true_labels=gt_labels,
                        predicted_labels=pred_labels,
                        algorithm_name=result.algorithm_name,
                        label_names=reverse_label_map,
                        params_info=_pinfo_std,
                        dataset_info=_dataset_info_str_std,
                    )
                    if fig:
                        fig.savefig(figures_dir / f'umap_comparison_{safe_algo_name}.png', bbox_inches='tight', dpi=150)
                        plt.close(fig)
                except Exception as e:
                    logger.warning(f"Could not generate UMAP comparison for {result.algorithm_name}: {e}")

                # Final batch-colored UMAP
                if batch_labels_std is not None:
                    try:
                        fig = viz.plot_umap_batch(
                            embeddings=result.embeddings,
                            batch_labels=batch_labels_std,
                            title=f"{result.algorithm_name} (Batch: {batch_col_std})",
                            params_info=_pinfo_std,
                            dataset_info=_dataset_info_str_std,
                        )
                        if fig:
                            fig.savefig(
                                figures_dir / f'umap_batch_{safe_algo_name}.png',
                                bbox_inches='tight',
                                dpi=150,
                            )
                            plt.close(fig)
                    except Exception as e:
                        logger.warning(f"Could not generate batch UMAP for {result.algorithm_name}: {e}")

                # Final weighted UMAP: ground truth + per-cell alpha from reconstruction weights
                if result.embedding_snapshots:
                    final_weights = None
                    for _snap in reversed(result.embedding_snapshots):
                        _w = _snap.get("cell_weights")
                        if _w is not None and len(_w) == len(result.embeddings):
                            final_weights = _w
                            break
                    if final_weights is not None:
                        try:
                            fig = viz.plot_umap_weighted(
                                result.embeddings,
                                gt_labels,
                                final_weights,
                                title=f"{result.algorithm_name} (Cell Weights)",
                                label_names=reverse_label_map,
                                params_info=_pinfo_std,
                                dataset_info=_dataset_info_str_std,
                            )
                            if fig:
                                fig.savefig(figures_dir / f'umap_{safe_algo_name}_weighted.png', bbox_inches='tight', dpi=150)
                                plt.close(fig)

                            fig_grad = viz.plot_umap_weighted_gradient(
                                result.embeddings,
                                final_weights,
                                title=f"{result.algorithm_name} (Cell Weights Gradient)",
                                params_info=_pinfo_std,
                                dataset_info=_dataset_info_str_std,
                            )
                            if fig_grad:
                                fig_grad.savefig(
                                    figures_dir / f'umap_{safe_algo_name}_weighted_gradient.png',
                                    bbox_inches='tight',
                                    dpi=150,
                                )
                                plt.close(fig_grad)
                        except Exception as e:
                            logger.warning(f"Could not generate weighted UMAP for {result.algorithm_name}: {e}")

                # --- Marker-Overlap Annotation + Sankey ---
                try:
                    if gt_labels is not None and len(gt_labels) == len(pred_labels):
                        from utils.metrics import marker_overlap_annotation
                        logger.info(f"Computing marker-overlap annotation for {result.algorithm_name}...")

                        overlap_result = marker_overlap_annotation(
                            adata=adata,
                            labels_true=gt_labels,
                            labels_pred=pred_labels,
                            n_top_genes=100,
                            method='wilcoxon',
                        )

                        # Heatmap of overlap scores
                        fig_hm = viz.plot_marker_overlap_heatmap(
                            overlap_matrix=overlap_result['overlap_matrix'],
                            algorithm_name=result.algorithm_name,
                        )
                        if fig_hm:
                            fig_hm.savefig(
                                figures_dir / f'marker_overlap_heatmap_{safe_algo_name}.png',
                                bbox_inches='tight', dpi=150,
                            )
                            plt.close(fig_hm)

                        # Sankey diagram
                        sankey_path = str(figures_dir / f'annotation_sankey_{safe_algo_name}.html')
                        viz.plot_annotation_sankey(
                            labels_true=gt_labels,
                            hungarian_labels=overlap_result['hungarian_labels'],
                            marker_labels=overlap_result['marker_labels'],
                            algorithm_name=result.algorithm_name,
                            save_path=sankey_path,
                        )

                        # Save annotation comparison CSV
                        import pandas as pd
                        annot_df = pd.DataFrame({
                            'true_label': np.asarray(gt_labels, dtype=str),
                            'predicted_cluster': np.asarray(pred_labels, dtype=str),
                            'hungarian_annotation': np.asarray(overlap_result['hungarian_labels'], dtype=str),
                            'marker_overlap_annotation': np.asarray(overlap_result['marker_labels'], dtype=str),
                        })
                        annot_csv = results_dir / f'annotation_comparison_{safe_algo_name}.csv'
                        annot_df.to_csv(annot_csv, index=False)

                        # Save overlap matrix
                        overlap_csv = results_dir / f'marker_overlap_matrix_{safe_algo_name}.csv'
                        overlap_result['overlap_matrix'].to_csv(overlap_csv)

                        logger.info(f"Marker-overlap annotation saved: {annot_csv}")
                        logger.info(f"Overlap matrix saved: {overlap_csv}")
                        logger.info(f"Sankey diagram saved: {sankey_path}")
                except Exception as e:
                    logger.warning(f"Marker-overlap annotation failed for {result.algorithm_name}: {e}")

        # Save results
        results_file = runner.save_results('results.json')
        logger.info(f"\nResults saved to {results_file}")
    
        # Save detailed results as CSV (UI parity)
        import pandas as pd
        rows = []
        for result in comparison_result.results:
            row = {
                'algorithm': result.algorithm_name,
                'run_id': result.run_id,
                'runtime': result.runtime,
                'num_parameters': result.extra_info.get('num_parameters'),
                **result.metrics,
                **{f'param_{k}': v for k, v in result.params.items() if not callable(v)}
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = results_dir / 'results.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV results saved to {csv_path}")

        # UI parity name
        csv_ui_path = results_dir / 'analysis_results.csv'
        df.to_csv(csv_ui_path, index=False)
        logger.info(f"Analysis CSV saved to {csv_ui_path}")
    
        # Save labels if requested (both predicted and true labels)
        if output_config.get('save_labels', True) or args.save_labels:
            import pandas as pd
            
            # Save label_map.json for decoding labels later
            if reverse_label_map is not None:
                lmap_file = labels_dir / 'label_map.json'
                # Save as {int_str: original_name}
                json_map = {str(k): v for k, v in reverse_label_map.items()}
                with open(lmap_file, 'w') as f:
                    json.dump(json_map, f, indent=2)
            # Detect batch column from adata for standard mode
            batch_col_std = None
            for candidate in ['batch', 'Batch', 'tech', 'study', 'dataset', 'donor', 'sample']:
                if candidate in adata.obs.columns:
                    batch_col_std = candidate
                    break

            for result in comparison_result.results:
                safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                labels_file = labels_dir / f"labels_{safe_algo}_run{result.run_id}.csv"
                labels_df = {'predicted_label': result.labels}
                if result.true_labels is not None:
                    if reverse_label_map is not None:
                        # Decode integer labels back to original cell type names
                        labels_df['true_label'] = [reverse_label_map.get(int(l), str(l)) for l in result.true_labels]
                    else:
                        labels_df['true_label'] = result.true_labels
                # Add batch column if available
                if batch_col_std is not None:
                    batch_values = adata.obs[batch_col_std].values
                    if len(batch_values) == len(result.labels):
                        labels_df['batch'] = batch_values
                pd.DataFrame(labels_df).to_csv(labels_file, index=False)
            logger.info(f"Labels saved to {labels_dir}")
            
        # Save embeddings if requested
        if output_config.get('save_embeddings', False) or args.save_embeddings:
            embeddings_dir = results_dir / 'embeddings'
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            for result in comparison_result.results:
                if result.embeddings is not None:
                    safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                    emb_file = embeddings_dir / f"embeddings_{safe_algo}_run{result.run_id}.npy"
                    np.save(emb_file, result.embeddings)
            logger.info(f"Embeddings saved to {embeddings_dir}")
    
        # Save loss history (JSON + PNG)
        try:
            has_any_loss = any(r.loss_history for r in comparison_result.results)
            if has_any_loss:
                loss_dir = results_dir / 'loss_history'
                loss_dir.mkdir(parents=True, exist_ok=True)
                for result in comparison_result.results:
                    if result.loss_history:
                        safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                        # Save JSON
                        loss_json = {
                            'algorithm': result.algorithm_name,
                            'run_id': result.run_id,
                            'phases': result.loss_history,
                        }
                        with open(loss_dir / f'loss_{safe_algo}_run{result.run_id}.json', 'w') as f:
                            json.dump(loss_json, f, indent=2)
                        # Save PNG
                        if not metrics_only_mode:
                            fig = viz.plot_loss_curves(result.loss_history, result.algorithm_name)
                            if fig:
                                fig.savefig(figures_dir / f'loss_curves_{safe_algo}_run{result.run_id}.png',
                                            bbox_inches='tight', dpi=150)
                                plt.close(fig)
                logger.info(f"Loss history saved to {loss_dir}")
        except Exception as e:
            logger.warning(f"Could not save loss history: {e}")

        try:
            _save_weight_history_artifacts(
                run_results=comparison_result.results,
                results_dir=results_dir,
                figures_dir=figures_dir,
            )
        except Exception as e:
            logger.warning(f"Could not save weight history: {e}")

        # Save UMAP evolution snapshots (PNG) — ground truth every 10 epochs
        if not metrics_only_mode:
            try:
                _umap_evo_cfg = _get_umap_evolution_render_config()
                has_any_umap_evolution = any(r.embedding_snapshots for r in comparison_result.results)
                if has_any_umap_evolution:
                    exported = 0
                    for result in comparison_result.results:
                        snapshots = result.embedding_snapshots or []
                        if not snapshots:
                            continue
                        snapshots_for_plot = _select_periodic_snapshots_every_10(snapshots)
                        if not snapshots_for_plot:
                            continue

                        first_valid_snapshot = next(
                            (
                                s for s in snapshots_for_plot
                                if s.get('embeddings') is not None and len(s.get('embeddings')) > 0
                            ),
                            None
                        )
                        if first_valid_snapshot is None:
                            continue

                        labels_for_plot = result.true_labels if result.true_labels is not None else labels
                        if labels_for_plot is None or len(labels_for_plot) != len(first_valid_snapshot.get('embeddings')):
                            logger.warning(
                                "Skipping UMAP evolution export for %s run %s: labels/snapshot size mismatch.",
                                result.algorithm_name,
                                result.run_id,
                            )
                            continue

                        safe_algo = result.algorithm_name.replace(' ', '_').replace('/', '_')
                        _pinfo_evo_std = _build_params_info_std(result)
                        _color_mode = _umap_evo_cfg["color_mode"]
                        _cell_weights_for_plot = None
                        _batch_labels_for_plot = _resolve_batch_labels_std(result)
                        if _color_mode == "ground_truth_weighted":
                            _weights_ok = True
                            _cell_weights_for_plot = []
                            for _snap in snapshots_for_plot:
                                _w = _snap.get("cell_weights")
                                _e = _snap.get("embeddings")
                                if _w is None or _e is None or len(_w) != len(_e):
                                    _weights_ok = False
                                    break
                                _cell_weights_for_plot.append(_w)
                            if not _weights_ok:
                                logger.warning(
                                    "Falling back to ground_truth UMAP evolution for %s run %s: "
                                    "missing/mismatched cell_weights in snapshots.",
                                    result.algorithm_name,
                                    result.run_id,
                                )
                                _color_mode = "ground_truth"
                                _cell_weights_for_plot = None
                        elif _color_mode == "batch":
                            if _batch_labels_for_plot is None or len(_batch_labels_for_plot) != len(first_valid_snapshot.get('embeddings')):
                                logger.warning(
                                    "Falling back to ground_truth UMAP evolution for %s run %s: "
                                    "batch labels unavailable/mismatched.",
                                    result.algorithm_name,
                                    result.run_id,
                                )
                                _color_mode = "ground_truth"
                                _batch_labels_for_plot = None

                        fig = viz.plot_umap_evolution(
                            embedding_snapshots=snapshots_for_plot,
                            labels=labels_for_plot,
                            algorithm_name=f"{result.algorithm_name} (run {result.run_id + 1})",
                            color_mode=_color_mode,
                            cell_weights_per_snapshot=_cell_weights_for_plot,
                            batch_labels=_batch_labels_for_plot,
                            point_size=_umap_evo_cfg["point_size"],
                            max_points=_umap_evo_cfg["max_points"],
                            max_snapshots=0,
                            params_info=_pinfo_evo_std,
                            dataset_info=_dataset_info_str_std,
                        )
                        if fig:
                            fig.savefig(
                                figures_dir / f'umap_evolution_{safe_algo}_run{result.run_id}.png',
                                bbox_inches='tight',
                                dpi=150,
                            )
                            plt.close(fig)
                            exported += 1
                    if exported > 0:
                        logger.info("UMAP evolution figures saved to %s (%d figure(s)).", figures_dir, exported)
            except Exception as e:
                logger.warning(f"Could not save UMAP evolution figures: {e}")

        effective_algo_params, effective_per_run_params = _extract_effective_params_from_run_results(
            comparison_result.results
        )

        # Save config used
        config_used = {
            'data': {'file': str(data_file)},
            'preprocessing': preprocessing_params,
            'algorithms': algorithms,
            'algorithm_params': algorithm_params,
            'algorithm_effective_params_by_algorithm': effective_algo_params,
            'algorithm_effective_params_per_run': effective_per_run_params,
            'execution': {
                'device': device,
                'n_repeats': n_repeats,
                'random_seed': random_seed,
                'compute_scib_metrics': compute_scib_metrics,
            },
            'batch_balancing': {
                'enabled': getattr(args, 'balance_batches_standard', False),
                'batch_col': getattr(args, 'batch_col_standard', 'auto'),
                'target': getattr(args, 'balance_target_standard', None),
            },
            'output': {'directory': str(output_dir)},
        }
        config_file = config_dir / 'config_used.json'
        with open(config_file, 'w') as f:
            json.dump(config_used, f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_file}")

        try:
            _save_algorithm_hyperparams_snapshot(
                config_dir=config_dir,
                algorithms=algorithms,
                algorithm_params=algorithm_params,
                run_results=comparison_result.results,
                search_spaces=search_spaces,
                execution={
                    'device': device,
                    'n_repeats': n_repeats,
                    'random_seed': random_seed,
                    'compute_scib_metrics': compute_scib_metrics,
                },
                context={
                    'mode': 'standard',
                    'data_file': str(data_file),
                    'output_dir': str(output_dir),
                    'status': 'completed',
                },
            )
        except Exception as e:
            logger.warning(f"Could not save completed hyperparameter snapshot: {e}")

        return 0


def cmd_list_algorithms(args: argparse.Namespace) -> int:
    """List available algorithms."""
    algorithms = get_available_algorithms()

    if args.json:
        print(json.dumps(algorithms, indent=2))
    else:
        print_algorithms_table(algorithms)

    return 0


def cmd_list_params(args: argparse.Namespace) -> int:
    """List parameters for an algorithm."""
    if not args.algorithm:
        logger.error("Please specify an algorithm with --algorithm")
        return 1

    try:
        params = get_algorithm_params(args.algorithm)
    except ValueError as e:
        logger.error(str(e))
        return 1

    if args.json:
        print(json.dumps(params, indent=2))
    else:
        print_params_table(args.algorithm, params)

    return 0


def cmd_generate_config(args: argparse.Namespace) -> int:
    """Generate a template configuration file."""
    config = generate_default_config()

    output_path = Path(args.output)

    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                logger.error("PyYAML is required for YAML output. Install with: pip install pyyaml")
                return 1
        else:
            json.dump(config, f, indent=2)

    logger.info(f"Configuration template saved to {output_path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a data file."""
    from utils.data_handler import DataHandler

    if not args.data:
        logger.error("Please specify a data file with --data")
        return 1

    handler = DataHandler()
    try:
        handler.load(args.data)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    info = handler.get_info()

    print("\n" + "=" * 50)
    print("DATA FILE INFORMATION")
    print("=" * 50)
    print(f"File: {args.data}")
    print(f"Cells: {info['n_cells']}")
    print(f"Genes: {info['n_genes']}")
    print(f"Has labels: {info['has_labels']}")
    if info['has_labels']:
        print(f"Number of clusters: {info['n_clusters']}")
        if 'label_names' in info:
            print(f"Label names: {info['label_names']}")
    print(f"Observation columns: {info.get('obs_columns', [])}")
    print(f"Variable columns: {info.get('var_columns', [])}")
    print(f"Has raw data: {info.get('has_raw', False)}")
    print(f"Layers: {info.get('layers', [])}")
    print("=" * 50)

    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Preprocess data and save to a new file."""
    from utils.data_handler import DataHandler
    import anndata as ad

    if not args.data:
        logger.error("Please specify a data file with --data")
        return 1

    if not args.output:
        logger.error("Please specify an output file with --output")
        return 1

    # Build preprocessing params
    preprocessing_params = {
        'n_top_genes': args.n_top_genes,
        'min_genes_per_cell': args.min_genes_per_cell,
        'max_genes_per_cell': args.max_genes_per_cell,
        'min_cells_per_gene': args.min_cells_per_gene,
        'target_sum': args.target_sum,
        'scale_max_value': args.scale_max_value,
        'hvg_flavor': args.hvg_flavor,
        'dropout_method': args.dropout_method,
        'dropout_rate': args.dropout_rate,
        'dropout_mid': args.dropout_mid,
        'dropout_shape': args.dropout_shape,
        'noise_level': args.noise_level,
        'ensure_celltype_markers': args.ensure_celltype_markers,
        'min_markers_per_celltype': args.min_markers_per_celltype,
        'max_additional_hvg_genes': args.max_additional_hvg_genes,
        'do_batch_correction': args.batch_correction,
        'batch_correction_method': args.batch_correction_method,
        'batch_correction_batch_key': args.batch_key,
        'batch_correction_labels_key': args.batch_correction_labels_key,
        'batch_correction_dct_weight': args.batch_correction_dct_weight,
        'batch_correction_corr_mse_weight': args.batch_correction_corr_mse_weight,
        'batch_correction_epochs': args.batch_correction_epochs,
        'batch_correction_n_latent': args.batch_correction_latent,
        'batch_correction_batch_size': args.batch_correction_batch_size,
        'batch_correction_cycle_weight': args.batch_correction_cycle_weight,
        'batch_correction_kl_weight': args.batch_correction_kl_weight,
        'batch_correction_prior': args.batch_correction_prior,
        'batch_correction_n_prior_components': args.batch_correction_n_prior_components,
        'batch_correction_embed_covariates': args.batch_correction_embed_covariates,
        'batch_correction_strict': args.batch_correction_strict,
        'batch_correction_seed': args.seed,
    }

    handler = DataHandler()
    try:
        logger.info(f"Loading data from {args.data}...")
        handler.load(args.data)

        info = handler.get_info()
        logger.info(f"Loaded: {info['n_cells']} cells, {info['n_genes']} genes")

        logger.info("Preprocessing...")
        handler.preprocess(preprocessing_params)

        info = handler.get_info()
        logger.info(f"After preprocessing: {info['n_cells']} cells, {info['n_genes']} genes")

        adata = handler.get_data()
        adata.write_h5ad(args.output, compression='gzip')
        logger.info(f"Preprocessed data saved to {args.output}")

    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1

    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='SCRBenchmark - Single-Cell RNA-seq Clustering Benchmark CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available algorithms
  ./scrbenchmark list-algorithms

  # Show parameters for an algorithm
  ./scrbenchmark list-params --algorithm scdeepcluster

  # Run analysis with default settings
  ./scrbenchmark run --data data/pbmc3k.h5ad --algorithms pca,scdeepcluster \\
      --param pca:clustering_method=kmeans

  # Run with custom hyperparameters
  ./scrbenchmark run --data data/pbmc3k.h5ad --algorithms scdeepcluster \\
      --param scdeepcluster:pretrain_epochs=100 --param scdeepcluster:z_dim=64


  # Generate a configuration template
  ./scrbenchmark generate-config --output my_config.yaml

  # Run from configuration file
  ./scrbenchmark run --config my_config.yaml

  # Get information about a data file
  ./scrbenchmark info --data data/pbmc3k.h5ad
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # -------------------------------------------------------------------------
    # run command
    # -------------------------------------------------------------------------
    run_parser = subparsers.add_parser('run', help='Run clustering analysis')

    # Data options
    run_parser.add_argument('--data', '-d', type=str, help='Path to data file (h5ad, h5, csv, tsv, mtx)')
    run_parser.add_argument('--config', '-c', type=str, help='Path to configuration file (yaml or json)')

    # Algorithm options
    run_parser.add_argument('--algorithms', '-a', type=str,
                           help='Comma-separated list of algorithms to run')
    run_parser.add_argument('--param', '-p', action='append',
                           help='Algorithm parameter in format "algo:param=value" or "param=value"')
    run_parser.add_argument('--search', action='append',
                           help='Hyperparameter search space in format "algo:param=[v1,v2,v3]"')
    run_parser.add_argument('--n-clusters', '-k', type=int,
                           help='Number of clusters (0 = auto-detect)')

    # Network architecture options (for autoencoder-based algorithms)
    network_group = run_parser.add_argument_group('Network Architecture Options')
    network_group.add_argument('--encoder-layers', type=str,
                               help='Encoder layer sizes (comma-separated), e.g., "512,256,64"')
    network_group.add_argument('--decoder-layers', type=str,
                               help='Decoder layer sizes (comma-separated), e.g., "64,256,512". '
                                    'If not specified with --symmetric-layers, mirrors encoder.')
    network_group.add_argument('--symmetric-layers', action='store_true', default=True,
                               help='Auto-generate decoder as mirror of encoder (default: True)')
    network_group.add_argument('--z-dim', type=int,
                               help='Latent space dimension (default varies by algorithm)')

    # Preprocessing options
    preproc_group = run_parser.add_argument_group('Preprocessing Options')
    preproc_group.add_argument('--skip-preprocessing', action='store_true',
                               help='Skip all preprocessing (use raw data)')
    # Individual preprocessing step toggles
    preproc_group.add_argument('--no-cell-filtering', action='store_true',
                               help='Disable cell filtering (keep all cells)')
    preproc_group.add_argument('--no-gene-filtering', action='store_true',
                               help='Disable gene filtering (keep all genes)')
    preproc_group.add_argument('--no-normalization', action='store_true',
                               help='Disable library size normalization')
    preproc_group.add_argument('--no-log-transform', action='store_true',
                               help='Disable log1p transformation')
    preproc_group.add_argument('--no-hvg', action='store_true',
                               help='Disable HVG selection (use all genes)')
    preproc_group.add_argument('--no-scaling', action='store_true',
                               help='Disable z-score scaling')
    # Preprocessing parameters
    preproc_group.add_argument('--n-top-genes', type=int,
                               help='Number of highly variable genes (default: 5000)')
    preproc_group.add_argument('--min-genes-per-cell', type=int,
                               help='Minimum genes per cell (default: 200)')
    preproc_group.add_argument('--max-genes-per-cell', type=int,
                               help='Maximum genes per cell (default: None)')
    preproc_group.add_argument('--min-cells-per-gene', type=int,
                               help='Minimum cells per gene (default: 3)')
    preproc_group.add_argument('--target-sum', type=float,
                               help='Normalization target sum (default: 10000)')
    preproc_group.add_argument('--scale-max-value', type=float,
                               help='Scale max value for clipping (default: 10.0)')
    preproc_group.add_argument('--hvg-flavor', type=str, choices=['seurat', 'seurat_v3', 'cell_ranger'],
                               help='HVG selection method (default: seurat)')
    preproc_group.add_argument('--hvg-strategy', type=str, 
                               choices=['train_only', 'all_data', 'per_batch_union'],
                               default='train_only',
                               help='HVG selection strategy in benchmark mode: '
                                    'train_only (default), all_data (before split), '
                                    'per_batch_union (X genes per batch, then union)')
    preproc_group.add_argument('--dropout-method', type=str, default=None,
                               choices=['none', 'simple', 'logistic'],
                               help='Artificial dropout simulation method (default: none)')
    preproc_group.add_argument('--dropout-rate', type=float, default=None,
                               help='[dropout-method=simple] Dropout rate (default: 0.2)')
    preproc_group.add_argument('--dropout-mid', type=float, default=None,
                               help='[dropout-method=logistic] Logistic midpoint (default: 0.0)')
    preproc_group.add_argument('--dropout-shape', type=float, default=None,
                               help='[dropout-method=logistic] Logistic shape (default: -1.0)')
    preproc_group.add_argument('--noise-level', type=float, default=None,
                               help='Gaussian noise level after dropout (default: 0.0)')
    preproc_group.add_argument('--ensure-celltype-markers', action='store_true', default=None,
                               help='Enrich HVGs with supervised cell-type markers (benchmark mode)')
    preproc_group.add_argument('--min-markers-per-celltype', type=int, default=None,
                               help='[ensure-celltype-markers] Minimum markers per cell type (default: 5)')
    preproc_group.add_argument('--max-additional-hvg-genes', type=int, default=None,
                               help='[ensure-celltype-markers] Max marker genes added to HVGs (default: 500)')
    preproc_group.add_argument('--batch-correction', action='store_true',
                               help='Enable batch correction')
    preproc_group.add_argument('--batch-correction-method', type=str, default='scvi',
                               choices=['scvi', 'sysvi'],
                               help='Batch correction method (default: scvi)')
    preproc_group.add_argument('--batch-key', type=str, default='auto',
                               help='Column name for batch information. Use "auto" to detect automatically (default: auto)')
    preproc_group.add_argument('--batch-correction-labels-key', type=str, default=None,
                               help='Optional cell-type label column for supervised scVI-DCT. '
                                    'If omitted, SCRBenchmark auto-detects a label column when available.')
    
    # Method-specific params
    preproc_group.add_argument('--batch-correction-dct-weight', type=float, default=None,
                               help='[scvi] DCT weight (default: 1.0)')
    preproc_group.add_argument('--batch-correction-corr-mse-weight', type=float, default=None,
                               help='[scvi] Correlation MSE weight (default: 0.1)')
    preproc_group.add_argument('--batch-correction-epochs', type=int, default=None,
                               help='[scvi/sysvi] Training epochs (method default if omitted)')
    preproc_group.add_argument('--batch-correction-latent', type=int, default=None,
                               help='[scvi/sysvi] Latent dimension (method default if omitted)')
    preproc_group.add_argument('--batch-correction-batch-size', type=int, default=None,
                               help='[scvi/sysvi] Batch size (method default if omitted)')
    
    # sysVI params
    preproc_group.add_argument('--batch-correction-cycle-weight', type=float, default=None,
                               help='[sysvi] Cycle-consistency loss weight (default: 5.0)')
    preproc_group.add_argument('--batch-correction-kl-weight', type=float, default=None,
                               help='[sysvi] KL divergence weight (default: 1.0)')
    preproc_group.add_argument('--batch-correction-prior', type=str, default=None,
                               choices=['vamp', 'standard_normal'],
                               help='[sysvi] Prior type (default: vamp)')
    preproc_group.add_argument('--batch-correction-n-prior-components', type=int, default=None,
                               help='[sysvi] Number of VampPrior pseudoinputs (default: 5)')
    preproc_group.add_argument('--batch-correction-embed-covariates', action=argparse.BooleanOptionalAction, default=None,
                               help='[sysvi] Embed categorical covariates (default: True)')
    preproc_group.add_argument('--batch-correction-strict', action=argparse.BooleanOptionalAction, default=None,
                               help='Fail run if batch correction fails instead of silently continuing (default: False)')

    # Execution options
    exec_group = run_parser.add_argument_group('Execution Options')
    exec_group.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'],
                            help='Compute device (default: auto)')
    exec_group.add_argument('--n-repeats', '-n', type=int, default=None,
                            help='Number of repetitions per algorithm (default: 1)')
    exec_group.add_argument('--seed', '-s', type=int, default=None,
                            help='Random seed (default: 42)')
    exec_group.add_argument('--no-scib-metrics', action='store_true',
                            help='Disable scIB/scIB-E metric computation')



    # Batch balancing options (standard mode - no split)
    balance_group = run_parser.add_argument_group('Batch Balancing Options (Standard Mode)',
                                                   'Options for balancing batches without train/test split')
    balance_group.add_argument('--balance-batches-standard', action='store_true',
                               help='Balance batches by subsampling larger batches (standard mode only). '
                                    'This equalizes contribution from each batch without splitting.')
    balance_group.add_argument('--batch-col-standard', type=str, default='auto',
                               help='Column name for batch information (default: auto-detect)')
    balance_group.add_argument('--balance-target-standard', type=str, default='auto',
                               help='Target cells per batch ("auto" = smallest batch size, default: auto)')
    balance_group.add_argument('--exclude-batches', type=str,
                               help='Comma-separated list of batch names to exclude from analysis. '
                                    'E.g., "smarter" to exclude RPKM-only batches for ZINB models.')

    # Output options
    output_group = run_parser.add_argument_group('Output Options')
    output_group.add_argument('--output', '-o', type=str, default='results_CLI',
                              help='Output directory (default: results_CLI)')
    output_group.add_argument('--no-timestamp', action='store_true',
                              help='Do not create a timestamped sub-folder in the output directory')
    output_group.add_argument('--verbose', '-v', action='store_true', default=True,
                              help='Verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                              help='Quiet output (only errors)')
    output_group.add_argument('--csv', action='store_true',
                              help='Also save results as CSV')
    output_group.add_argument('--save-labels', action='store_true',
                              help='Save predicted labels to files')
    output_group.add_argument('--save-embeddings', action='store_true',
                              help='Save embeddings to files (npy format)')

    # Benchmark mode options
    benchmark_group = run_parser.add_argument_group('Benchmark Mode Options',
                                                     'Options for train/test split evaluation')
    benchmark_group.add_argument('--benchmark-mode', action='store_true',
                                 help='Enable benchmark mode with train/test split')
    benchmark_group.add_argument('--train-ratio', type=float, default=0.7,
                                 help='Training set ratio (default: 0.7)')
    benchmark_group.add_argument('--val-ratio', type=float, default=0.1,
                                 help='Validation set ratio (default: 0.1)')
    benchmark_group.add_argument('--test-ratio', type=float, default=0.2,
                                 help='Test set ratio (default: 0.2)')
    benchmark_group.add_argument('--train-batches', type=str,
                                 help='Comma-separated list of batches for training (split by batch mode)')
    benchmark_group.add_argument('--test-batches', type=str,
                                 help='Comma-separated list of batches for testing. '
                                      'If provided with --train-batches, enables strict batch split '
                                      '(train on selected train batches, test on these test batches). '
                                      'If omitted, test remains a stratified split over all batches.')
    benchmark_group.add_argument('--stratify-by', type=str,
                                 help='Column name for batch stratification (e.g., "tech", "batch")')
    benchmark_group.add_argument('--label-col', type=str, default=None,
                                 help='Column name for cell type labels (default: auto-detect Group/labels)')
    benchmark_group.add_argument('--stratify-mode', type=str, default='batch+labels',
                                 choices=['batch+labels', 'batch-only', 'labels-only', 'none'],
                                 help='Stratification mode for train/test split: '
                                      'batch+labels (default, recommended), '
                                      'batch-only (ignore cell types), '
                                      'labels-only (ignore batches), '
                                      'none (random split)')
    benchmark_group.add_argument('--balance-batches', action='store_true',
                                 help='Balance batches by subsampling larger batches')
    benchmark_group.add_argument('--balance-target', type=str, default='auto',
                                 help='Target cells per batch for balancing ("auto" = smallest batch size, default: auto)')
    benchmark_group.add_argument('--balance-mode', type=str, default='eliminate',
                                 choices=['eliminate', 'reinject_train', 'reinject_both'],
                                 help='Balance mode: eliminate (discard excess cells before split), '
                                      'reinject_train (balance TEST only, reinject excess to TRAIN), '
                                      'reinject_both (balance TEST only, reinject excess to TRAIN+VAL). '
                                      'Default: eliminate')

    # -------------------------------------------------------------------------
    # list-algorithms command
    # -------------------------------------------------------------------------
    list_algo_parser = subparsers.add_parser('list-algorithms',
                                              help='List available algorithms')
    list_algo_parser.add_argument('--json', action='store_true',
                                  help='Output as JSON')

    # -------------------------------------------------------------------------
    # list-params command
    # -------------------------------------------------------------------------
    list_params_parser = subparsers.add_parser('list-params',
                                                help='List hyperparameters for an algorithm')
    list_params_parser.add_argument('--algorithm', '-a', type=str, required=True,
                                    help='Algorithm name')
    list_params_parser.add_argument('--json', action='store_true',
                                    help='Output as JSON')

    # -------------------------------------------------------------------------
    # generate-config command
    # -------------------------------------------------------------------------
    gen_config_parser = subparsers.add_parser('generate-config',
                                               help='Generate a template configuration file')
    gen_config_parser.add_argument('--output', '-o', type=str, default='config.yaml',
                                   help='Output file path (default: config.yaml)')

    # -------------------------------------------------------------------------
    # info command
    # -------------------------------------------------------------------------
    info_parser = subparsers.add_parser('info', help='Show information about a data file')
    info_parser.add_argument('--data', '-d', type=str, required=True,
                             help='Path to data file')

    # -------------------------------------------------------------------------
    # preprocess command
    # -------------------------------------------------------------------------
    preproc_parser = subparsers.add_parser('preprocess',
                                            help='Preprocess data and save to new file')
    preproc_parser.add_argument('--data', '-d', type=str, required=True,
                                help='Input data file')
    preproc_parser.add_argument('--output', '-o', type=str, required=True,
                                help='Output file path (h5ad)')
    preproc_parser.add_argument('--n-top-genes', type=int, default=5000)
    preproc_parser.add_argument('--min-genes-per-cell', type=int, default=200)
    preproc_parser.add_argument('--max-genes-per-cell', type=int, default=None)
    preproc_parser.add_argument('--min-cells-per-gene', type=int, default=3)
    preproc_parser.add_argument('--target-sum', type=float, default=10000)
    preproc_parser.add_argument('--scale-max-value', type=float, default=10.0)
    preproc_parser.add_argument('--hvg-flavor', type=str, default='seurat',
                                choices=['seurat', 'seurat_v3', 'cell_ranger'])
    preproc_parser.add_argument('--dropout-method', type=str, default='none',
                                choices=['none', 'simple', 'logistic'])
    preproc_parser.add_argument('--dropout-rate', type=float, default=0.2)
    preproc_parser.add_argument('--dropout-mid', type=float, default=0.0)
    preproc_parser.add_argument('--dropout-shape', type=float, default=-1.0)
    preproc_parser.add_argument('--noise-level', type=float, default=0.0)
    preproc_parser.add_argument('--ensure-celltype-markers', action='store_true',
                                help='Enrich HVGs with supervised cell-type markers')
    preproc_parser.add_argument('--min-markers-per-celltype', type=int, default=5)
    preproc_parser.add_argument('--max-additional-hvg-genes', type=int, default=500)
    preproc_parser.add_argument('--batch-correction', action='store_true',
                               help='Enable batch correction')
    preproc_parser.add_argument('--batch-correction-method', type=str, default='scvi',
                               choices=['scvi', 'sysvi'],
                               help='Batch correction method (default: scvi)')
    preproc_parser.add_argument('--batch-key', type=str, default='auto',
                               help='Column name for batch information')
    preproc_parser.add_argument('--batch-correction-labels-key', type=str, default=None,
                               help='Optional cell-type label column for supervised scVI-DCT.')
    
    # Method-specific params
    preproc_parser.add_argument('--batch-correction-dct-weight', type=float, default=1.0)
    preproc_parser.add_argument('--batch-correction-corr-mse-weight', type=float, default=0.1)
    preproc_parser.add_argument('--batch-correction-epochs', type=int, default=None)
    preproc_parser.add_argument('--batch-correction-latent', type=int, default=None)
    preproc_parser.add_argument('--batch-correction-batch-size', type=int, default=None)
    
    # sysVI params
    preproc_parser.add_argument('--batch-correction-cycle-weight', type=float, default=5.0)
    preproc_parser.add_argument('--batch-correction-kl-weight', type=float, default=1.0)
    preproc_parser.add_argument('--batch-correction-prior', type=str, default='vamp',
                               choices=['vamp', 'standard_normal'])
    preproc_parser.add_argument('--batch-correction-n-prior-components', type=int, default=5)
    preproc_parser.add_argument('--batch-correction-embed-covariates', action=argparse.BooleanOptionalAction, default=None)
    preproc_parser.add_argument('--batch-correction-strict', action=argparse.BooleanOptionalAction, default=None)
    preproc_parser.add_argument('--seed', '-s', type=int, default=None,
                                help='Random seed used for batch correction reproducibility')

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Handle quiet mode
    if hasattr(args, 'quiet') and args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        args.verbose = False

    # Dispatch to command handlers
    if args.command == 'run':
        return run_analysis(args)
    elif args.command == 'list-algorithms':
        return cmd_list_algorithms(args)
    elif args.command == 'list-params':
        return cmd_list_params(args)
    elif args.command == 'generate-config':
        return cmd_generate_config(args)
    elif args.command == 'info':
        return cmd_info(args)
    elif args.command == 'preprocess':
        return cmd_preprocess(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
