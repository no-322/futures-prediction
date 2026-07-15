"""Load and access the centralized pipeline configuration.

The config file (config.yaml by default) is the single source of truth for
all tuneable parameters. A GUI can overwrite it; the pipeline re-reads it on
every run.
"""
from pathlib import Path

import yaml

_DEFAULT_PATH = Path("config.yaml")


def load_config(path: Path = _DEFAULT_PATH) -> dict:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to the YAML config file (default: config.yaml).

    Returns:
        Parsed config dict with keys 'data' and 'models'.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f)


def model_params(config: dict, algo: str) -> dict:
    """Return a copy of the hyperparameter dict for the given algorithm.

    Args:
        config: Config dict returned by load_config().
        algo: One of 'baseline', 'rf', 'gbm'.

    Returns:
        Dict of hyperparameters for the algo; empty dict if algo is not found.
    """
    return dict(config.get("models", {}).get(algo, {}))
