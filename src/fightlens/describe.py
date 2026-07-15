"""Gemini descriptions for extracted time windows.

Runs as a separate step AFTER frame extraction, so extracting frames never
spends API tokens. Each window folder becomes one multimodal Gemini request:
all of its frames in chronological order plus the description prompt.
"""

import json
import time
from pathlib import Path
from typing import Any

from fightlens import gemini
from fightlens.config import resolve_project_path
from fightlens.video import WINDOW_ID_WIDTH


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load the extraction manifest produced by the preprocessing step."""

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. "
            "Run frame extraction first (python -m fightlens extract)."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_descriptions(output_path: str | Path) -> list[dict[str, Any]]:
    """Load previously saved descriptions, or an empty list if none exist."""

    output_path = Path(output_path)
    if not output_path.is_file():
        return []

    with output_path.open("r", encoding="utf-8") as file:
        entries = json.load(file)

    if not isinstance(entries, list):
        raise ValueError(
            f"The descriptions file must contain a JSON list: {output_path}"
        )

    return entries


def write_descriptions(
    entries: list[dict[str, Any]], output_path: str | Path
) -> None:
    """Write the descriptions as pretty-printed JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, indent=2, ensure_ascii=False)


def window_name(window_id: int) -> str:
    """Folder-style window name, e.g. 0 -> "window_000000"."""

    return f"window_{window_id:0{WINDOW_ID_WIDTH}d}"


def describe_windows(
    manifest_path: str | Path,
    output_path: str | Path,
    request_delay_seconds: float,
    prompt: str,
    retry_attempts: int = 1,
) -> dict[str, Any]:
    """
    Generate a Gemini description for every extracted window.

    Windows are processed sequentially with a pause between API calls
    (free-tier rate limits). The run is idempotent: windows that already
    have a description in the output JSON are skipped, and the JSON is
    saved after every window so an interrupted run loses nothing.
    A failed call is retried retry_attempts times; a window that still
    fails is reported and skipped, so one bad window never aborts the
    whole run.
    Returns a summary with "described", "skipped" and "failed" counts.
    """

    manifest = load_manifest(manifest_path)
    entries = load_descriptions(output_path)
    described_ids = {
        entry["window_id"]
        for entry in entries
        if entry.get("description")
    }

    described = 0
    skipped = 0
    failed: list[str] = []
    calls_made = 0

    for window in manifest["windows"]:
        name = window_name(window["window_id"])

        if name in described_ids:
            skipped += 1
            continue

        frame_paths = window["image_paths"]
        if not frame_paths:
            print(f"{name}: no saved frames, skipping.")
            skipped += 1
            continue

        # Manifest paths are relative to the project root.
        resolved_paths = [resolve_project_path(path) for path in frame_paths]

        if calls_made > 0 and request_delay_seconds > 0:
            time.sleep(request_delay_seconds)

        description = _describe_with_retry(
            name, resolved_paths, prompt, request_delay_seconds, retry_attempts
        )
        calls_made += 1

        if description is None:
            failed.append(name)
            continue

        entries.append(
            {
                "window_id": name,
                "start_sec": window["start_timestamp"],
                "end_sec": window["end_timestamp"],
                "frames": frame_paths,
                "description": description,
            }
        )
        write_descriptions(entries, output_path)
        described += 1
        print(f"{name}: described ({len(frame_paths)} frames).")

    return {
        "total_windows": len(manifest["windows"]),
        "described": described,
        "skipped": skipped,
        "failed": failed,
        "output_path": str(output_path),
    }


def _describe_with_retry(
    name: str,
    frame_paths: list[Path],
    prompt: str,
    request_delay_seconds: float,
    retry_attempts: int,
) -> str | None:
    """Call Gemini for one window; return None once all attempts fail."""

    for attempt in range(1 + retry_attempts):
        try:
            return gemini.describe_images(frame_paths, prompt)
        except Exception as error:  # noqa: BLE001 - keep the batch running
            if attempt < retry_attempts:
                print(f"{name}: Gemini call failed ({error}), retrying...")
                if request_delay_seconds > 0:
                    time.sleep(request_delay_seconds)
            else:
                print(f"{name}: Gemini call failed, skipping ({error}).")

    return None
