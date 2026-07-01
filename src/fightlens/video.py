from pathlib import Path

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    frame_step: int,
) -> list[Path]:
    """
    Extract every N-th frame from a video.

    For example, frame_step=30 saves frames:

        0, 30, 60, 90, 120, ...

    Args:
        video_path:
            Path to the source video.

        output_dir:
            Base directory for extracted frames.
            A subdirectory named after the video is created automatically.

        frame_step:
            Save one frame for every N video frames.

    Returns:
        Paths to all saved frame images.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if frame_step <= 0:
        raise ValueError("frame_step must be greater than zero.")

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    # Each video gets its own output directory.
    #
    # Example:
    #   input:  data/raw/fight.mp4
    #   output: data/processed/frames/fight/
    frames_dir = output_dir / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Open the source video.
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV could not open the video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_index = 0
    saved_paths: list[Path] = []

    try:
        while True:
            # Read the next frame.
            success, frame = capture.read()

            # The video has ended or the next frame cannot be read.
            if not success:
                break

            # Save frames 0, N, 2N, 3N, ...
            if frame_index % frame_step == 0:
                timestamp_seconds = (
                    frame_index / fps if fps > 0 else 0.0
                )

                frame_name = (
                    f"frame_{frame_index:08d}_"
                    f"{timestamp_seconds:010.2f}s.jpg"
                )

                frame_path = frames_dir / frame_name

                was_saved = cv2.imwrite(
                    str(frame_path),
                    frame,
                )

                if not was_saved:
                    raise RuntimeError(
                        f"Could not save frame: {frame_path}"
                    )

                saved_paths.append(frame_path)

            frame_index += 1

    finally:
        # Release the video even if an exception occurs.
        capture.release()

    print("Frame extraction completed")
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Total video frames: {total_frames}")
    print(f"Frame step: {frame_step}")
    print(f"Saved frames: {len(saved_paths)}")
    print(f"Output directory: {frames_dir}")

    return saved_paths