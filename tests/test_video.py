"""Tests for the time-window video preprocessing pipeline."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from fightlens import video
from fightlens.config import (
    PROJECT_ROOT,
    resolve_project_path,
    validate_video_config,
)
from fightlens.video import (
    build_windows,
    extract_windows,
    frames_per_window,
    resolve_effective_fps,
    select_frame_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_video(path: Path, frame_count: int, fps: float, size=(64, 48)) -> Path:
    """Write a short synthetic video with a known number of frames."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened(), "Could not open VideoWriter in the test env."

    for i in range(frame_count):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return path


def frame_index_from_name(name: str) -> int:
    """Extract the source frame index encoded in an image file name."""

    # img_00_frame_00000059_0000001.97s.jpg -> 59
    return int(name.split("_frame_")[1].split("_")[0])


def window_dirs(manifest_path: Path) -> list[Path]:
    return sorted((manifest_path.parent / "windows").glob("window_*"))


# ---------------------------------------------------------------------------
# Pure logic: build_windows
# ---------------------------------------------------------------------------


def test_build_windows_exact_division_has_no_extra_window():
    windows = build_windows(total_frames=180, window_size=60)

    assert len(windows) == 3
    assert all(w["is_full"] for w in windows)
    assert windows[0]["first_source_frame"] == 0
    assert windows[0]["last_source_frame"] == 59
    assert windows[-1]["last_source_frame"] == 179


def test_build_windows_with_remainder_keeps_partial_last_window():
    windows = build_windows(total_frames=200, window_size=60)

    # 3 full windows (180 frames) + 1 partial window (20 frames).
    assert len(windows) == 4
    assert [w["is_full"] for w in windows] == [True, True, True, False]

    last = windows[-1]
    assert last["first_source_frame"] == 180
    assert last["last_source_frame"] == 199
    assert last["available_frames"] == 20


def test_build_windows_shorter_than_one_window():
    windows = build_windows(total_frames=10, window_size=60)

    assert len(windows) == 1
    assert windows[0]["is_full"] is False
    assert windows[0]["available_frames"] == 10


# ---------------------------------------------------------------------------
# Pure logic: select_frame_indices
# ---------------------------------------------------------------------------


def test_select_frame_indices_is_evenly_spread_across_the_window():
    indices = select_frame_indices(first_frame=0, available_frames=60, count=6)

    assert len(indices) == 6
    assert indices == sorted(indices)
    assert len(set(indices)) == 6
    # First and last picks hug the window boundaries.
    assert indices[0] == 0
    assert indices[-1] == 59
    # Spread out, not the first six frames in a row.
    assert indices == [0, 12, 24, 35, 47, 59]


def test_select_frame_indices_respects_window_offset():
    indices = select_frame_indices(first_frame=180, available_frames=20, count=6)

    assert indices[0] == 180
    assert indices[-1] == 199
    assert len(set(indices)) == 6


def test_select_frame_indices_fewer_frames_than_requested_no_duplicates():
    indices = select_frame_indices(first_frame=0, available_frames=4, count=6)

    # Only 4 frames exist; keep all 4, never duplicate to reach 6.
    assert indices == [0, 1, 2, 3]


def test_select_frame_indices_single_frame_requested():
    assert select_frame_indices(0, 60, 1) == [29]


def test_select_frame_indices_empty_window():
    assert select_frame_indices(0, 0, 6) == []


# ---------------------------------------------------------------------------
# Pure logic: FPS resolution and window sizing
# ---------------------------------------------------------------------------


def test_resolve_effective_fps_uses_video_metadata_by_default():
    fps, source = resolve_effective_fps(video_fps=30.0, fps_override=None)
    assert fps == 30.0
    assert source == "video"


def test_resolve_effective_fps_manual_override_wins():
    fps, source = resolve_effective_fps(video_fps=30.0, fps_override=24.0)
    assert fps == 24.0
    assert source == "config"


def test_resolve_effective_fps_invalid_metadata_without_override_errors():
    with pytest.raises(ValueError, match="valid FPS"):
        resolve_effective_fps(video_fps=0.0, fps_override=None)


def test_resolve_effective_fps_invalid_metadata_with_override_ok():
    fps, source = resolve_effective_fps(video_fps=0.0, fps_override=25.0)
    assert fps == 25.0
    assert source == "config"


def test_resolve_effective_fps_rejects_non_positive_override():
    with pytest.raises(ValueError, match="positive"):
        resolve_effective_fps(video_fps=30.0, fps_override=0.0)


def test_frames_per_window():
    assert frames_per_window(30.0, 2) == 60
    assert frames_per_window(25.0, 2) == 50
    # Never below one frame.
    assert frames_per_window(30.0, 0.001) == 1


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def _valid_config() -> dict:
    return {
        "input_path": "data/raw/x.mp4",
        "output_dir": "data/processed/",
        "n_sec_per_window": 2,
        "n_img_per_window": 6,
        "fps_override": None,
    }


def test_validate_video_config_accepts_valid_config():
    params = validate_video_config(_valid_config())
    assert params["n_sec_per_window"] == 2.0
    assert params["n_img_per_window"] == 6
    assert params["fps_override"] is None


def test_validate_video_config_accepts_manual_fps():
    cfg = _valid_config()
    cfg["fps_override"] = 24
    params = validate_video_config(cfg)
    assert params["fps_override"] == 24.0


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c.pop("input_path"), "input_path"),
        (lambda c: c.pop("output_dir"), "output_dir"),
        (lambda c: c.update(n_sec_per_window=0), "n_sec_per_window"),
        (lambda c: c.update(n_sec_per_window=-2), "n_sec_per_window"),
        (lambda c: c.update(n_img_per_window=0), "n_img_per_window"),
        (lambda c: c.update(n_img_per_window=2.5), "n_img_per_window"),
        (lambda c: c.update(fps_override=0), "fps_override"),
        (lambda c: c.update(fps_override=-5), "fps_override"),
    ],
)
def test_validate_video_config_rejects_bad_values(mutate, match):
    cfg = _valid_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        validate_video_config(cfg)


def test_validate_video_config_requires_mapping():
    with pytest.raises(ValueError, match="video"):
        validate_video_config(None)


# ---------------------------------------------------------------------------
# Integration: extract_windows
# ---------------------------------------------------------------------------


def test_extract_windows_full_division_no_extra_folder(tmp_path):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=180, fps=30)

    manifest, manifest_path = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=6,
        fps_override=30,
    )

    assert manifest["total_windows"] == 3
    assert manifest["full_windows"] == 3
    assert manifest["has_partial_window"] is False
    assert manifest["total_saved_images"] == 18
    # No stray extra window folder.
    assert len(window_dirs(manifest_path)) == 3
    for window in manifest["windows"]:
        assert window["saved_images"] == 6


def test_extract_windows_creates_partial_last_window(tmp_path):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=200, fps=30)

    manifest, _ = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=6,
        fps_override=30,
    )

    assert manifest["total_windows"] == 4
    assert manifest["has_partial_window"] is True

    last = manifest["windows"][-1]
    assert last["is_full"] is False
    assert last["end_timestamp"] - last["start_timestamp"] < 2.0
    assert last["saved_images"] == 6


def test_extract_windows_very_short_video(tmp_path):
    # 4 frames total, window would be 60 frames: one short window, all frames.
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=4, fps=30)

    manifest, manifest_path = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=6,
        fps_override=30,
    )

    assert manifest["total_windows"] == 1
    window = manifest["windows"][0]
    assert window["is_full"] is False
    # Fewer frames than requested -> keep all unique frames, no duplication.
    assert window["saved_images"] == 4
    saved = sorted(p.name for p in window_dirs(manifest_path)[0].iterdir())
    frame_ids = [frame_index_from_name(name) for name in saved]
    assert frame_ids == [0, 1, 2, 3]
    assert len(set(frame_ids)) == len(frame_ids)


def test_extract_windows_manual_fps_override_changes_window_size(tmp_path):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=120, fps=30)

    manifest, _ = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=6,
        fps_override=60,  # 60 fps * 2s = 120 frames per window.
    )

    assert manifest["fps_source"] == "config"
    assert manifest["effective_fps"] == 60.0
    assert manifest["frames_per_window"] == 120
    assert manifest["total_windows"] == 1


def test_extract_windows_invalid_auto_fps(tmp_path, monkeypatch):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=120, fps=30)

    # Simulate a codec that reports a broken FPS of 0.
    monkeypatch.setattr(
        video,
        "get_video_metadata",
        lambda _path: {"fps": 0.0, "total_frames": 120},
    )

    # No override -> clear error instead of dividing by zero.
    with pytest.raises(ValueError, match="valid FPS"):
        extract_windows(
            video_path=video_path,
            output_dir=tmp_path / "out",
            n_sec_per_window=2,
            n_img_per_window=6,
            fps_override=None,
        )

    # With override -> processing continues.
    manifest, _ = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=6,
        fps_override=30,
    )
    assert manifest["fps_source"] == "config"


def test_extract_windows_existing_output_stops_without_overwrite(tmp_path):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=60, fps=30)
    kwargs = dict(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=3,
        fps_override=30,
    )

    extract_windows(**kwargs)  # first run creates the folder

    # Second run without overwrite -> clear error.
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_windows(**kwargs)


def test_extract_windows_overwrite_removes_stale_files(tmp_path):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=60, fps=30)
    kwargs = dict(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=3,
        fps_override=30,
    )

    _, manifest_path = extract_windows(**kwargs)
    stale = manifest_path.parent / "windows" / "window_000000" / "stale.jpg"
    stale.write_bytes(b"old")

    extract_windows(**kwargs, overwrite=True)

    # The old folder was deleted, so leftovers from previous runs are gone.
    assert not stale.exists()


def test_extract_windows_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_windows(
            video_path=tmp_path / "nope.mp4",
            output_dir=tmp_path / "out",
            n_sec_per_window=2,
            n_img_per_window=6,
        )


def test_extract_windows_no_empty_folder_when_frames_unreadable(
    tmp_path, monkeypatch
):
    # Real video has 60 frames, but pretend metadata claims 130.
    # The trailing planned window points at frames that can never be read,
    # so its folder must not be created (no empty folders).
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=60, fps=30)

    monkeypatch.setattr(
        video,
        "get_video_metadata",
        lambda _path: {"fps": 30.0, "total_frames": 130},
    )

    manifest, manifest_path = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=6,
        fps_override=30,
    )

    # Planned: 3 windows (0-59, 60-119, 120-129). Only the first is readable.
    assert manifest["total_windows"] == 3
    existing = {p.name for p in window_dirs(manifest_path)}
    assert "window_000000" in existing
    # Unreadable windows leave no empty folder behind.
    assert "window_000002" not in existing
    for window in manifest["windows"][1:]:
        assert window["saved_images"] == 0
        assert window["image_paths"] == []


def test_extract_windows_manifest_written_with_relative_paths(tmp_path):
    video_path = make_video(tmp_path / "raw" / "clip.mp4", frame_count=90, fps=30)

    _, manifest_path = extract_windows(
        video_path=video_path,
        output_dir=tmp_path / "out",
        n_sec_per_window=2,
        n_img_per_window=3,
        fps_override=30,
    )

    assert manifest_path.is_file()
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk["windows"][0]["image_paths"]

    # tmp_path is outside the project, so paths stay absolute here; but they
    # must never contain machine-specific project-root prefixes doubled up,
    # and inside the project they are stored relative to the project root.
    inside = video._relative_path(PROJECT_ROOT / "data" / "x.jpg")
    assert inside == "data/x.jpg"


# ---------------------------------------------------------------------------
# Relative paths / working-directory independence
# ---------------------------------------------------------------------------


def test_resolve_project_path_is_cwd_independent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    resolved = resolve_project_path("data/raw/test_video.mp4")
    assert resolved == PROJECT_ROOT / "data" / "raw" / "test_video.mp4"
    assert resolved.is_absolute()


def test_resolve_project_path_keeps_absolute_paths(tmp_path):
    absolute = tmp_path / "video.mp4"
    assert resolve_project_path(absolute) == absolute
