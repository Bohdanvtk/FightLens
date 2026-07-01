from fightlens.config import load_config, resolve_project_path
from fightlens.video import extract_frames


def main() -> None:
    """
    Run the FightLens preprocessing pipeline.

    All parameters are read from configs/default.yaml.
    """

    # Load all application parameters from YAML.
    config = load_config()

    # Read the video section.
    video_config = config.get("video")

    if not isinstance(video_config, dict):
        raise ValueError(
            "The configuration must contain a 'video' section."
        )

    # Read and validate the required parameters.
    try:
        input_path = video_config["input_path"]
        output_dir = video_config["output_dir"]
        frame_step = int(video_config["frame_step"])
    except KeyError as error:
        raise ValueError(
            f"Missing video configuration parameter: {error.args[0]}"
        ) from error

    # Resolve relative paths from the project root.
    video_path = resolve_project_path(input_path)
    frames_output_dir = resolve_project_path(output_dir)

    extract_frames(
        video_path=video_path,
        output_dir=frames_output_dir,
        frame_step=frame_step,
    )


if __name__ == "__main__":
    main()