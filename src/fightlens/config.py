from pathlib import Path
from typing import Any

import yaml


# Absolute path to the root directory of the FightLens project.
#
# __file__:
#   fightlens/src/fightlens/config.py
#
# parents[0]:
#   fightlens/src/fightlens
#
# parents[1]:
#   fightlens/src
#
# parents[2]:
#   fightlens
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default configuration file used by the application.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load the FightLens configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration represented as a Python dictionary.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If the YAML file is empty or invalid.
    """

    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {config_path}"
        )

    return config


def resolve_project_path(path: str | Path) -> Path:
    """
    Convert a configuration path into an absolute path.

    Absolute paths are returned unchanged.
    Relative paths are resolved from PROJECT_ROOT.

    Example:
        "data/raw/fight.mp4"

    becomes:
        "/home/bohdan/PycharmProjects/fightlens/data/raw/fight.mp4"
    """

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path