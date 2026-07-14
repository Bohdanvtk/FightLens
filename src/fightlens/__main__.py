from typing import Any

from fightlens.config import (
    load_config,
    resolve_project_path,
    validate_video_config,
)
from fightlens.video import extract_windows


def main() -> None:
    """
    Run the FightLens video preprocessing pipeline.

    Steps:
        1. Load the YAML configuration.
        2. Read and validate the 'video' section.
        3. Resolve relative paths from the project root.
        4. Split the video into time windows and save sampled frames.
        5. Print a human-readable summary.

    All parameters are read from configs/default.yaml.
    """

    # Load all application parameters from YAML.
    config = load_config()

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
    )

    _print_summary(manifest, manifest_path)


def _print_summary(manifest: dict[str, Any], manifest_path: Any) -> None:
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
    print(f"Window duration: {manifest['n_sec_per_window']} s")
    print(f"Frames per full window: {manifest['frames_per_window']}")
    print(f"Images per window: {manifest['n_img_per_window']}")
    print(f"Total windows: {manifest['total_windows']}")
    print(f"Full windows: {manifest['full_windows']}")
    print(f"Has last partial window: {manifest['has_partial_window']}")
    print(f"Total saved images: {manifest['total_saved_images']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
