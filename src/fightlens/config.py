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


def _is_non_negative_number(value: Any) -> bool:
    """True for a zero-or-positive int or float (but not bool)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
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
            "n_sec_per_window", "n_img_per_window", "fps_override",
            "start_seconds", "end_seconds", "max_windows".

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

    start_seconds = video_config.get("start_seconds", 0)
    if not _is_non_negative_number(start_seconds):
        raise ValueError(
            "video.start_seconds must be zero or a positive number "
            f"(where extraction starts), got: {start_seconds!r}."
        )

    end_seconds = video_config.get("end_seconds")
    if end_seconds is not None:
        if not _is_positive_number(end_seconds):
            raise ValueError(
                "video.end_seconds must be a positive number or null "
                f"(null = end of video), got: {end_seconds!r}."
            )
        if float(end_seconds) <= float(start_seconds):
            raise ValueError(
                "video.end_seconds must be greater than video.start_seconds, "
                f"got: start={start_seconds!r}, end={end_seconds!r}."
            )

    max_windows = video_config.get("max_windows")
    if max_windows is not None and (
        isinstance(max_windows, bool)
        or not isinstance(max_windows, int)
        or max_windows <= 0
    ):
        raise ValueError(
            "video.max_windows must be a positive integer or null "
            f"(null = no limit), got: {max_windows!r}."
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
        "start_seconds": float(start_seconds),
        "end_seconds": (
            float(end_seconds) if end_seconds is not None else None
        ),
        "max_windows": max_windows,
    }


def validate_error_log_dir(value: Any) -> str:
    """
    Validate the top-level 'error_log_dir' configuration value.

    The parameter is top-level (not inside 'descriptions') because the
    error log covers the whole launch, extraction included.

    Args:
        value: The 'error_log_dir' value from the loaded configuration,
            or None when the key is absent (defaults to "logs").

    Returns:
        The directory as a string (relative paths are resolved from the
        project root by the caller).

    Raises:
        ValueError: If the value is present but not a non-empty string.
    """

    if value is None:
        return "logs"

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "error_log_dir must be a non-empty string path to the "
            f"directory for per-run error JSON files, got: {value!r}."
        )

    return value


def validate_descriptions_config(descriptions_config: Any) -> dict[str, Any]:
    """
    Validate the 'descriptions' section of the configuration.

    Args:
        descriptions_config: The 'descriptions' section from the loaded
            configuration.

    Returns:
        A mapping of validated parameters:
            "manifest_path", "output_path", "request_delay_seconds",
            "retry_attempts", "response_timeout_seconds", "prompt".

    Raises:
        ValueError: If any parameter is missing or invalid.
    """

    if not isinstance(descriptions_config, dict):
        raise ValueError(
            "The configuration must contain a 'descriptions' section "
            "(a YAML mapping)."
        )

    manifest_path = descriptions_config.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise ValueError(
            "descriptions.manifest_path must be a non-empty string path to "
            f"the extraction manifest.json, got: {manifest_path!r}."
        )

    output_path = descriptions_config.get("output_path")
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(
            "descriptions.output_path must be a non-empty string path to "
            f"the output JSON file, got: {output_path!r}."
        )

    request_delay_seconds = descriptions_config.get("request_delay_seconds", 4.0)
    if not _is_non_negative_number(request_delay_seconds):
        raise ValueError(
            "descriptions.request_delay_seconds must be zero or a positive "
            f"number (pause between Gemini calls), got: "
            f"{request_delay_seconds!r}."
        )

    retry_attempts = descriptions_config.get("retry_attempts", 1)
    if (
        isinstance(retry_attempts, bool)
        or not isinstance(retry_attempts, int)
        or retry_attempts < 0
    ):
        raise ValueError(
            "descriptions.retry_attempts must be zero or a positive integer "
            f"(extra attempts after a failed call), got: {retry_attempts!r}."
        )

    response_timeout_seconds = descriptions_config.get(
        "response_timeout_seconds", 30
    )
    if response_timeout_seconds is not None and not _is_positive_number(
        response_timeout_seconds
    ):
        raise ValueError(
            "descriptions.response_timeout_seconds must be a positive "
            "number or null (max seconds to wait for one Gemini response, "
            f"null = wait forever), got: {response_timeout_seconds!r}."
        )

    prompt = descriptions_config.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            "descriptions.prompt must be a non-empty string (the text sent "
            f"to Gemini with each window's frames), got: {prompt!r}."
        )

    return {
        "manifest_path": manifest_path,
        "output_path": output_path,
        "request_delay_seconds": float(request_delay_seconds),
        "retry_attempts": retry_attempts,
        "response_timeout_seconds": (
            float(response_timeout_seconds)
            if response_timeout_seconds is not None
            else None
        ),
        "prompt": prompt,
    }