"""Public API for the scRAW package."""

from .config import ScRAWConfig, load_config, save_config


def run_pipeline(*args, **kwargs):
    """Import the execution pipeline only when needed."""
    from .pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def run_inference_from_checkpoint(*args, **kwargs):
    """Import the checkpoint replay path only when needed."""
    from .pipeline import run_inference_from_checkpoint as _run_inference_from_checkpoint

    return _run_inference_from_checkpoint(*args, **kwargs)


def run_inductive_baron_split(*args, **kwargs):
    """Import the inductive Baron train/test path only when needed."""
    from .inductive import run_inductive_baron_split as _run_inductive_baron_split

    return _run_inductive_baron_split(*args, **kwargs)


def run_inductive_prediction(*args, **kwargs):
    """Import the frozen-artifact inductive prediction path only when needed."""
    from .inductive import run_inductive_prediction as _run_inductive_prediction

    return _run_inductive_prediction(*args, **kwargs)


def resolve_preset_config(*args, **kwargs):
    """Import preset resolution only when needed."""
    from .presets import resolve_preset_config as _resolve_preset_config

    return _resolve_preset_config(*args, **kwargs)

__all__ = [
    "ScRAWConfig",
    "load_config",
    "save_config",
    "run_pipeline",
    "run_inference_from_checkpoint",
    "run_inductive_baron_split",
    "run_inductive_prediction",
    "resolve_preset_config",
]
