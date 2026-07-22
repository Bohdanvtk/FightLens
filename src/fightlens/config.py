from pathlib import Path
from typing import Any

import yaml


# Absolute path to the project root (3 levels up from src/fightlens/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default configuration file used by the application.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config into a dict. Raises FileNotFoundError / ValueError if missing or invalid."""

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
    """Resolve a config path to an absolute one (relative paths resolve from PROJECT_ROOT)."""

    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def is_positive_number(value: Any) -> bool:
    """True for a strictly positive int or float (but not bool)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def is_non_negative_number(value: Any) -> bool:
    """True for a zero-or-positive int or float (but not bool)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def validate_video_config(video_config: Any) -> dict[str, Any]:
    """Validate the 'video' config section; raises ValueError naming the bad key and value."""

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
    if not is_positive_number(n_sec_per_window):
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
    if fps_override is not None and not is_positive_number(fps_override):
        raise ValueError(
            "video.fps_override must be a positive number or null "
            f"(manual FPS fallback), got: {fps_override!r}."
        )

    start_seconds = video_config.get("start_seconds", 0)
    if not is_non_negative_number(start_seconds):
        raise ValueError(
            "video.start_seconds must be zero or a positive number "
            f"(where extraction starts), got: {start_seconds!r}."
        )

    end_seconds = video_config.get("end_seconds")
    if end_seconds is not None:
        if not is_positive_number(end_seconds):
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
    """Validate the top-level 'error_log_dir' (covers the whole run, not just 'descriptions'). Defaults to "logs"."""

    if value is None:
        return "logs"

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "error_log_dir must be a non-empty string path to the "
            f"directory for per-run error JSON files, got: {value!r}."
        )

    return value


def validate_descriptions_config(descriptions_config: Any) -> dict[str, Any]:
    """Validate the 'descriptions' config section (output path is derived, not configured here)."""

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

    request_delay_seconds = descriptions_config.get("request_delay_seconds", 4.0)
    if not is_non_negative_number(request_delay_seconds):
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
    if response_timeout_seconds is not None and not is_positive_number(
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
        "request_delay_seconds": float(request_delay_seconds),
        "retry_attempts": retry_attempts,
        "response_timeout_seconds": (
            float(response_timeout_seconds)
            if response_timeout_seconds is not None
            else None
        ),
        "prompt": prompt,
    }


def video_processed_dir(config: dict[str, Any]) -> Path:
    """Absolute path to <video.output_dir>/<video stem> — where all per-video artifacts live."""

    params = validate_video_config(config.get("video"))
    output_dir = resolve_project_path(params["output_dir"])
    return output_dir / Path(params["input_path"]).stem


def validate_embedding_config(embedding_config: Any) -> dict[str, Any]:
    """Validate the 'embedding' config section (dimension is fixed by the model, not a key)."""

    if not isinstance(embedding_config, dict):
        raise ValueError(
            "The configuration must contain an 'embedding' section "
            "(a YAML mapping)."
        )

    model_name = embedding_config.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(
            "embedding.model_name must be a non-empty string (the local "
            f"sentence-transformers model), got: {model_name!r}."
        )

    batch_size = embedding_config.get("batch_size", 32)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError(
            "embedding.batch_size must be a positive integer "
            f"(descriptions encoded per forward pass), got: {batch_size!r}."
        )

    device = embedding_config.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise ValueError(
            "embedding.device must be a non-empty string such as 'auto', "
            f"'cpu' or 'cuda', got: {device!r}."
        )

    normalize = embedding_config.get("normalize", True)
    if not isinstance(normalize, bool):
        raise ValueError(
            f"embedding.normalize must be true or false, got: {normalize!r}."
        )

    return {
        "model_name": model_name,
        "batch_size": int(batch_size),
        "device": device,
        "normalize": normalize,
    }


def validate_search_config(search_config: Any) -> dict[str, Any]:
    """Validate the 'search' config section (top_k is how many results to print)."""

    if not isinstance(search_config, dict):
        raise ValueError(
            "The configuration must contain a 'search' section "
            "(a YAML mapping)."
        )

    top_k = search_config.get("top_k", 10)
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise ValueError(
            "search.top_k must be a positive integer "
            f"(how many results to print), got: {top_k!r}."
        )

    return {"top_k": top_k}


def validate_rerank_config(rerank_config: Any) -> dict[str, Any]:
    """Validate the 'rerank' config section (the model is reused from Gemini's description model, not a key)."""

    if not isinstance(rerank_config, dict):
        raise ValueError(
            "The configuration must contain a 'rerank' section "
            "(a YAML mapping)."
        )

    enabled = rerank_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"rerank.enabled must be true or false, got: {enabled!r}."
        )

    top_n = rerank_config.get("top_n", 10)
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n <= 0
    ):
        raise ValueError(
            "rerank.top_n must be a positive integer "
            f"(embedding candidates sent to the reranker), got: {top_n!r}."
        )

    return {"enabled": enabled, "top_n": top_n}


_PREVIEW_PLAYERS = ("gif", "mp4")


def validate_preview_config(preview_config: Any) -> dict[str, Any]:
    """Validate the 'preview' config section (used only by scripts/app.py, not the CLI)."""

    if not isinstance(preview_config, dict):
        raise ValueError(
            "The configuration must contain a 'preview' section "
            "(a YAML mapping)."
        )

    fps = preview_config.get("fps", 6.0)
    if fps != "auto" and not is_positive_number(fps):
        raise ValueError(
            "preview.fps must be a positive number or 'auto' (frames per "
            f"second for the window preview clip), got: {fps!r}."
        )

    player = preview_config.get("player", "gif")
    if player not in _PREVIEW_PLAYERS:
        raise ValueError(
            f"preview.player must be one of {_PREVIEW_PLAYERS} "
            f"('gif' needs no system tools, 'mp4' needs ffmpeg on PATH), "
            f"got: {player!r}."
        )

    return {"fps": "auto" if fps == "auto" else float(fps), "player": player}