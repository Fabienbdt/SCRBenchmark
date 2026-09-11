"""Validate configuration structure before a benchmark creates output files."""

import json
from pathlib import Path
from typing import Any


def load_run_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path).expanduser()
    try:
        with path.open(encoding="utf-8") as handle:
            if path.suffix.lower() in {".yaml", ".yml"}:
                import yaml
                try:
                    config = yaml.safe_load(handle)
                except yaml.YAMLError as exc:
                    raise ValueError(f"Invalid YAML configuration: {exc}") from exc
            else:
                config = json.load(handle)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON configuration: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object or YAML mapping.")
    for section in ("data", "preprocessing", "algorithm_params", "execution", "output", "benchmark"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"Configuration section '{section}' must be a mapping.")
    algorithms = config.get("algorithms", [])
    if not isinstance(algorithms, list) or any(not isinstance(name, str) for name in algorithms):
        raise ValueError("Configuration 'algorithms' must be a list of names.")
    if "file" in config.get("data", {}) and not isinstance(config["data"]["file"], str):
        raise ValueError("Configuration 'data.file' must be a path string.")
    for name, params in config.get("algorithm_params", {}).items():
        if not isinstance(params, dict):
            raise ValueError(f"Configuration parameters for '{name}' must be a mapping.")
    return config
