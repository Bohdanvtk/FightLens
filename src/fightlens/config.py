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


def _is_positive_number(value: Any) -> bool:
    """True for a strictly positive int or float (but not bool)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def validate_video_config(video_config: Any) -> dict[str, Any]:
    """
    Validate the 'video' section of the configuration.

    Every error message names the offending parameter, the value that
    was received, and what was expected. Nothing is silently defaulted.

    Args:
        video_config: The 'video' section from the loaded configuration.

    Returns:
        A mapping of validated parameters:
            "input_path", "output_dir",
            "n_sec_per_window", "n_img_per_window", "fps_override".

    Raises:
        ValueError: If any parameter is missing or invalid.
    """

    if not isinstance(video_config, dict):
        raise ValueError(
            "The configuration must contain a 'video' section "
            "(a YAML mapping)."
        )

    input_path = video_config.get("input_path")
    if not isinstance(input_path, str) or not input_path.strip():
        raise ValueError(
            "video.input_path must be a non-empty string path to the "
            f"source video, got: {input_path!r}."
        )

    output_dir = video_config.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError(
            "video.output_dir must be a non-empty string path, "
            f"got: {output_dir!r}."
        )

    n_sec_per_window = video_config.get("n_sec_per_window")
    if not _is_positive_number(n_sec_per_window):
        raise ValueError(
            "video.n_sec_per_window must be a positive number "
            f"(window duration in seconds), got: {n_sec_per_window!r}."
        )

    n_img_per_window = video_config.get("n_img_per_window")
    if (
        isinstance(n_img_per_window, bool)
        or not isinstance(n_img_per_window, int)
        or n_img_per_window <= 0
    ):
        raise ValueError(
            "video.n_img_per_window must be a positive integer "
            f"(images per window), got: {n_img_per_window!r}."
        )

    overwrite = video_config.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise ValueError(
            f"video.overwrite must be true or false, got: {overwrite!r}."
        )

    fps_override = video_config.get("fps_override")
    if fps_override is not None and not _is_positive_number(fps_override):
        raise ValueError(
            "video.fps_override must be a positive number or null "
            f"(manual FPS fallback), got: {fps_override!r}."
        )

    return {
        "input_path": input_path,
        "output_dir": output_dir,
        "n_sec_per_window": float(n_sec_per_window),
        "n_img_per_window": int(n_img_per_window),
        "overwrite": overwrite,
        "fps_override": (
            float(fps_override) if fps_override is not None else None
        ),
    }