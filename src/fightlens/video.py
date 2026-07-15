import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from fightlens.config import PROJECT_ROOT


# Digits in window folder names, e.g. "window_000042" (keeps them sorted).
WINDOW_ID_WIDTH = 6


def get_video_metadata(video_path: str | Path) -> dict[str, Any]:
    """Open a video and read its FPS and total frame count."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    return {"fps": fps, "total_frames": total_frames}


def resolve_effective_fps(
    video_fps: float,
    fps_override: float | None,
) -> tuple[float, str]:
    """
    Choose the FPS for the pipeline.

    A manual override wins; otherwise the video metadata FPS is used.
    Errors clearly if neither is valid instead of continuing with FPS 0.
    Returns (fps, source) where source is "config" or "video".
    """

    if fps_override is not None:
        if not _is_positive_number(fps_override):
            raise ValueError(
                f"fps_override must be a positive number, got: {fps_override!r}."
            )
        return float(fps_override), "config"

    if video_fps is not None and video_fps > 0:
        return float(video_fps), "video"

    raise ValueError(
        "Could not determine a valid FPS. OpenCV reported "
        f"fps={video_fps!r}, and no positive 'fps_override' was provided. "
        "Set 'fps_override' for videos with missing or broken metadata."
    )


def frames_per_window(effective_fps: float, n_sec_per_window: float) -> int:
    """Source frames spanned by a full window (30fps * 2s -> 60). At least 1."""

    return max(1, int(round(effective_fps * n_sec_per_window)))


def build_windows(
    total_frames: int,
    window_size: int,
    start_frame: int = 0,
) -> list[dict[str, Any]]:
    """
    Split a frame range into contiguous frame-range windows (no physical cut).

    window 0 -> [start_frame, start_frame + window_size), and so on up to
    total_frames (exclusive). The final window may be shorter; it is kept
    as the last partial window.
    """

    windows: list[dict[str, Any]] = []
    window_id = 0
    start = start_frame

    while start < total_frames:
        end = min(start + window_size, total_frames)  # exclusive
        available = end - start
        windows.append(
            {
                "window_id": window_id,
                "first_source_frame": start,
                "last_source_frame": end - 1,
                "available_frames": available,
                "is_full": available == window_size,
            }
        )
        window_id += 1
        start = end

    return windows


def select_frame_indices(
    first_frame: int,
    available_frames: int,
    count: int,
) -> list[int]:
    """
    Pick evenly spaced frame indices inside a window.

    The first and last picks sit near the window boundaries. If fewer
    frames exist than requested, all are used; frames are never duplicated.
    """

    if available_frames <= 0 or count <= 0:
        return []

    count = min(count, available_frames)
    if count == 1:
        return [first_frame + (available_frames - 1) // 2]

    span = available_frames - 1
    indices = [first_frame + int(i * span / (count - 1) + 0.5) for i in range(count)]
    return sorted(dict.fromkeys(indices))  # stay unique, keep order


def extract_windows(
    video_path: str | Path,
    output_dir: str | Path,
    n_sec_per_window: float,
    n_img_per_window: int,
    fps_override: float | None = None,
    overwrite: bool = False,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    max_windows: int | None = None,
) -> tuple[dict[str, Any], Path]:
    """
    Split a video into time windows and save sampled frames per window.

    Layout: <output_dir>/<video_stem>/{manifest.json, windows/window_000000/...}
    If the video output folder already exists: overwrite=True deletes it
    first, overwrite=False stops with an error.

    The processed scope can be limited so only part of a long video is
    extracted: start_seconds / end_seconds bound the time range (end_seconds
    None = end of video) and max_windows caps how many windows are kept.
    Timestamps always stay relative to the original video, not the scope.
    Returns (manifest, manifest_path).
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_output_dir = output_dir / video_path.stem
    if video_output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {video_output_dir}. "
                "Set 'overwrite: true' in the config to rewrite it."
            )
        shutil.rmtree(video_output_dir)

    metadata = get_video_metadata(video_path)
    video_fps = metadata["fps"]
    total_frames = metadata["total_frames"]

    if total_frames <= 0:
        raise ValueError(
            f"The video reports a non-positive frame count ({total_frames}). "
            f"The file may be empty or corrupted: {video_path}"
        )

    effective_fps, fps_source = resolve_effective_fps(video_fps, fps_override)
    window_size = frames_per_window(effective_fps, n_sec_per_window)

    first_frame, end_frame = _resolve_scope(
        total_frames, effective_fps, start_seconds, end_seconds
    )
    windows = build_windows(end_frame, window_size, start_frame=first_frame)
    if max_windows is not None:
        windows = windows[:max_windows]

    windows_root = video_output_dir / "windows"

    # Plan the frames to save, mapping each absolute frame index back to
    # its (window_id, local position) so one pass over the video saves them all.
    targets: dict[int, tuple[int, int]] = {}
    for window in windows:
        selected = select_frame_indices(
            window["first_source_frame"],
            window["available_frames"],
            n_img_per_window,
        )
        window["saved_images"] = 0
        window["image_paths"] = []
        for local_position, frame_index in enumerate(selected):
            targets[frame_index] = (window["window_id"], local_position)

    window_by_id = {window["window_id"]: window for window in windows}
    _read_and_save_frames(
        video_path, windows_root, targets, window_by_id, effective_fps
    )

    manifest = _build_manifest(
        video_path=video_path,
        windows=windows,
        video_fps=video_fps,
        effective_fps=effective_fps,
        fps_source=fps_source,
        total_frames=total_frames,
        n_sec_per_window=n_sec_per_window,
        n_img_per_window=n_img_per_window,
        window_size=window_size,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        max_windows=max_windows,
    )

    manifest_path = video_output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)
    return manifest, manifest_path


def _resolve_scope(
    total_frames: int,
    effective_fps: float,
    start_seconds: float,
    end_seconds: float | None,
) -> tuple[int, int]:
    """
    Convert the requested time range into a [first_frame, end_frame) range.

    end_seconds None means "until the end of the video". Errors clearly if
    the range falls outside the video instead of producing zero windows.
    """

    first_frame = int(round(start_seconds * effective_fps))
    if first_frame >= total_frames:
        raise ValueError(
            f"start_seconds ({start_seconds}) is at or past the end of the "
            f"video ({total_frames / effective_fps:.2f} s)."
        )

    if end_seconds is None:
        end_frame = total_frames
    else:
        end_frame = min(total_frames, int(round(end_seconds * effective_fps)))

    if end_frame <= first_frame:
        raise ValueError(
            f"The requested range [{start_seconds}, {end_seconds}] s spans "
            "no frames. Increase end_seconds or lower start_seconds."
        )

    return first_frame, end_frame


def _read_and_save_frames(
    video_path: Path,
    windows_root: Path,
    targets: dict[int, tuple[int, int]],
    window_by_id: dict[int, dict[str, Any]],
    effective_fps: float,
) -> None:
    """
    Read the video once, in order, and save every planned frame.

    Reading sequentially (no seeking) keeps timestamps aligned even for
    codecs with imprecise seeking. Unwanted frames are only grabbed, never
    decoded, so skipping ahead to a late start_seconds stays cheap.
    Window folders are created lazily, so a window that yields no frame
    leaves no empty folder behind.
    """

    if not targets:
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")

    last_target = max(targets)  # stop once the last wanted frame is saved
    frame_index = 0

    try:
        while frame_index <= last_target:
            if not capture.grab():
                break

            if frame_index in targets:
                success, frame = capture.retrieve()
                if not success:
                    break

                window_id, local_position = targets[frame_index]
                window = window_by_id[window_id]

                folder = windows_root / f"window_{window_id:0{WINDOW_ID_WIDTH}d}"
                folder.mkdir(parents=True, exist_ok=True)

                timestamp = frame_index / effective_fps
                frame_path = folder / (
                    f"img_{local_position:02d}_"
                    f"frame_{frame_index:08d}_"
                    f"{timestamp:010.2f}s.jpg"
                )
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not save frame: {frame_path}")

                window["saved_images"] += 1
                window["image_paths"].append(_relative_path(frame_path))

            frame_index += 1
    finally:
        capture.release()


def _build_manifest(
    video_path: Path,
    windows: list[dict[str, Any]],
    video_fps: float,
    effective_fps: float,
    fps_source: str,
    total_frames: int,
    n_sec_per_window: float,
    n_img_per_window: int,
    window_size: int,
    start_seconds: float,
    end_seconds: float | None,
    max_windows: int | None,
) -> dict[str, Any]:
    """Assemble the processing manifest for the whole video."""

    manifest_windows: list[dict[str, Any]] = []
    full_windows = 0
    total_saved_images = 0

    for window in windows:
        first_frame = window["first_source_frame"]
        last_frame = window["last_source_frame"]

        full_windows += window["is_full"]
        total_saved_images += window["saved_images"]

        manifest_windows.append(
            {
                "window_id": window["window_id"],
                "first_source_frame": first_frame,
                "last_source_frame": last_frame,
                "start_timestamp": round(first_frame / effective_fps, 3),
                "end_timestamp": round((last_frame + 1) / effective_fps, 3),
                "available_frames": window["available_frames"],
                "saved_images": window["saved_images"],
                "image_paths": window["image_paths"],
                "is_full": window["is_full"],
            }
        )

    return {
        "video_path": _relative_path(video_path),
        "video_fps": round(video_fps, 3),
        "effective_fps": round(effective_fps, 3),
        "fps_source": fps_source,
        "total_frames": total_frames,
        "video_duration_seconds": round(total_frames / effective_fps, 3),
        "n_sec_per_window": n_sec_per_window,
        "n_img_per_window": n_img_per_window,
        "frames_per_window": window_size,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
        "max_windows": max_windows,
        "total_windows": len(windows),
        "full_windows": full_windows,
        "has_partial_window": bool(windows) and not windows[-1]["is_full"],
        "total_saved_images": total_saved_images,
        "windows": manifest_windows,
    }


def write_manifest(manifest: dict[str, Any], manifest_path: str | Path) -> None:
    """Write the manifest as pretty-printed JSON."""

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)


def _relative_path(path: str | Path) -> str:
    """Path relative to the project root when possible (keeps manifest portable)."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _is_positive_number(value: Any) -> bool:
    """True for a strictly positive int or float (but not bool)."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )
