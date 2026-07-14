from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# =============================================================================
# HARDCODED PARAMETERS
# Change only these values.
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory containing extracted frames.
FRAMES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_video"

)

# Sampling step used when frames were extracted.
FRAME_STEP = 1

# Number of frames displayed in one window.
FRAMES_PER_PAGE = 2

# None means display all available frames.
# Example: MAX_FRAMES = 20
MAX_FRAMES = None


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_frame_paths(frames_dir: Path) -> list[Path]:
    """Return all frame images sorted by filename."""

    if not frames_dir.is_dir():
        raise FileNotFoundError(
            f"Frames directory not found: {frames_dir}"
        )

    frame_paths = sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"No frame images found in: {frames_dir}"
        )

    return frame_paths


def get_frame_label(image_position: int) -> int:
    """
    Return the frame label according to FRAME_STEP.

    For FRAME_STEP = 30:

        position 0 -> frame 1
        position 1 -> frame 30
        position 2 -> frame 60
        position 3 -> frame 90
    """

    if image_position == 0:
        return 1

    return image_position * FRAME_STEP


def show_page(
    frame_paths: list[Path],
    start_position: int,
    page_number: int,
    total_pages: int,
) -> None:
    """Display one page of frames in a single vertical column."""

    row_count = len(frame_paths)

    figure, axes = plt.subplots(
        nrows=row_count,
        ncols=1,
        figsize=(10, 4 * row_count),
        squeeze=False,
    )

    figure.suptitle(
        f"Extracted Frames — Step {FRAME_STEP} "
        f"— Page {page_number}/{total_pages}",
        fontsize=16,
        fontweight="bold",
    )

    for local_position, frame_path in enumerate(frame_paths):
        global_position = start_position + local_position

        image = cv2.imread(str(frame_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {frame_path}"
            )

        # Convert OpenCV image for correct Matplotlib display.
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        frame_label = get_frame_label(global_position)
        axis = axes[local_position, 0]

        axis.imshow(image)

        axis.set_title(
            f"Sample {global_position + 1}  |  "
            f"Source frame: {frame_label}",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )

        # Remove coordinate axes for a cleaner preview.
        axis.axis("off")

    figure.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    plt.show()
    plt.close(figure)


def main() -> None:
    if FRAME_STEP <= 0:
        raise ValueError(
            "FRAME_STEP must be greater than zero."
        )

    if FRAMES_PER_PAGE <= 0:
        raise ValueError(
            "FRAMES_PER_PAGE must be greater than zero."
        )

    frame_paths = get_frame_paths(FRAMES_DIR)

    if MAX_FRAMES is not None:
        frame_paths = frame_paths[:MAX_FRAMES]

    total_pages = (
        len(frame_paths) + FRAMES_PER_PAGE - 1
    ) // FRAMES_PER_PAGE

    for start_position in range(
        0,
        len(frame_paths),
        FRAMES_PER_PAGE,
    ):
        page_paths = frame_paths[
            start_position:
            start_position + FRAMES_PER_PAGE
        ]

        page_number = (
            start_position // FRAMES_PER_PAGE
        ) + 1

        show_page(
            frame_paths=page_paths,
            start_position=start_position,
            page_number=page_number,
            total_pages=total_pages,
        )


if __name__ == "__main__":
    main()