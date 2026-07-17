import argparse
from typing import Any

from fightlens.config import (
    load_config,
    resolve_project_path,
    validate_descriptions_config,
    validate_error_log_dir,
    validate_video_config,
)
from fightlens.describe import describe_windows
from fightlens.errorlog import ErrorLog
from fightlens.video import extract_windows


def main() -> None:
    """
    Run one step of the FightLens pipeline.

    Commands:
        extract  — split the video into time windows and save sampled
                   frames (no API calls, spends no tokens). Default.
        describe — send each extracted window to Gemini and save the
                   descriptions JSON (spends API tokens).
        full     — run extract, then describe.

    All parameters are read from configs/default.yaml.
    """

    parser = argparse.ArgumentParser(
        prog="fightlens",
        description="FightLens video pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "extract",
        help="Extract window frames from the video (no Gemini calls).",
    )
    subparsers.add_parser(
        "describe",
        help="Generate Gemini descriptions for extracted windows.",
    )
    subparsers.add_parser(
        "full",
        help="Run extract, then describe.",
    )

    args = parser.parse_args()
    config = load_config()

    # One error log per launch; its JSON file is named after the exact
    # start date and time and is only created if an error is recorded.
    # The directory comes from the config (error_log_dir, default "logs").
    error_log = ErrorLog(
        log_dir=resolve_project_path(
            validate_error_log_dir(config.get("error_log_dir"))
        )
    )

    try:
        # Running without a command keeps the old behaviour: extraction only.
        if args.command in (None, "extract", "full"):
            _run_extract(config)
        if args.command in ("describe", "full"):
            _run_describe(config, error_log)
    except Exception as error:
        # A fatal error still ends up in the per-run error file.
        error_log.record(where="run", error=error)
        raise
    finally:
        if error_log.count:
            print(
                f"Errors this run: {error_log.count} "
                f"(saved to {error_log.path})"
            )


def _run_extract(config: dict[str, Any]) -> None:
    """Split the video into time windows and save sampled frames."""

    # Read and validate the video section (clear errors on bad values).
    params = validate_video_config(config.get("video"))

    # Resolve relative paths from the project root.
    video_path = resolve_project_path(params["input_path"])
    output_dir = resolve_project_path(params["output_dir"])

    manifest, manifest_path = extract_windows(
        video_path=video_path,
        output_dir=output_dir,
        n_sec_per_window=params["n_sec_per_window"],
        n_img_per_window=params["n_img_per_window"],
        fps_override=params["fps_override"],
        overwrite=params["overwrite"],
        start_seconds=params["start_seconds"],
        end_seconds=params["end_seconds"],
        max_windows=params["max_windows"],
    )

    _print_extract_summary(manifest, manifest_path)


def _run_describe(config: dict[str, Any], error_log: ErrorLog) -> None:
    """Generate a Gemini description for every extracted window."""

    params = validate_descriptions_config(config.get("descriptions"))

    summary = describe_windows(
        manifest_path=resolve_project_path(params["manifest_path"]),
        output_path=resolve_project_path(params["output_path"]),
        request_delay_seconds=params["request_delay_seconds"],
        prompt=params["prompt"],
        retry_attempts=params["retry_attempts"],
        timeout_seconds=params["response_timeout_seconds"],
        error_log=error_log,
    )

    _print_describe_summary(summary)


def _print_extract_summary(manifest: dict[str, Any], manifest_path: Any) -> None:
    """Print the final processing statistics."""

    used_override = manifest["fps_source"] == "config"

    print("Video preprocessing completed")
    print(f"Video: {manifest['video_path']}")
    print(f"FPS (video metadata): {manifest['video_fps']:.2f}")
    print(f"Effective FPS: {manifest['effective_fps']:.2f}")
    print(f"FPS source: {manifest['fps_source']}")
    print(f"Manual FPS override used: {used_override}")
    print(f"Total source frames: {manifest['total_frames']}")
    print(f"Video duration: {manifest['video_duration_seconds']:.2f} s")
    print(f"Scope: start={manifest['start_seconds']} s, "
          f"end={manifest['end_seconds'] or 'video end'}, "
          f"max_windows={manifest['max_windows'] or 'no limit'}")
    print(f"Window duration: {manifest['n_sec_per_window']} s")
    print(f"Frames per full window: {manifest['frames_per_window']}")
    print(f"Images per window: {manifest['n_img_per_window']}")
    print(f"Total windows: {manifest['total_windows']}")
    print(f"Full windows: {manifest['full_windows']}")
    print(f"Has last partial window: {manifest['has_partial_window']}")
    print(f"Total saved images: {manifest['total_saved_images']}")
    print(f"Manifest: {manifest_path}")


def _print_describe_summary(summary: dict[str, Any]) -> None:
    """Print the final description statistics."""

    print("Window description completed")
    print(f"Total windows in manifest: {summary['total_windows']}")
    print(f"Newly described: {summary['described']}")
    print(f"Skipped (already described or empty): {summary['skipped']}")
    print(f"Failed: {len(summary['failed'])}")
    if summary["failed"]:
        print(f"Failed windows: {', '.join(summary['failed'])}")
        print("Re-run 'python -m fightlens describe' to retry them.")
    print(f"Descriptions: {summary['output_path']}")


if __name__ == "__main__":
    main()
