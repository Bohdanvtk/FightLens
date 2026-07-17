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
from fightlens.errorlog import ErrorLog
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
    timeout_seconds: float | None = 30.0,
    error_log: ErrorLog | None = None,
) -> dict[str, Any]:
    """
    Generate a Gemini description for every extracted window.

    Windows are processed sequentially with a pause between API calls
    (free-tier rate limits). The run is idempotent: windows that already
    have a description in the output JSON are skipped, and the JSON is
    saved after every window so an interrupted run loses nothing.

    A call that errors OR takes longer than timeout_seconds counts as one
    failed attempt: it is reported to the console (and to error_log when
    given) and a fresh request is sent, up to retry_attempts extra times.
    A window whose attempts are all spent is put on the "failed" list and
    the loop moves on to the next window, so one bad or stuck window
    never blocks the run.
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
            name,
            resolved_paths,
            prompt,
            request_delay_seconds,
            retry_attempts,
            timeout_seconds,
            error_log,
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
    timeout_seconds: float | None = None,
    error_log: ErrorLog | None = None,
) -> str | None:
    """Call Gemini for one window; return None once all attempts fail.

    Every attempt is timed. An answer that does not arrive within
    timeout_seconds is abandoned and counted as a failed attempt, exactly
    like an API error, so a window is never waited on forever: after
    1 + retry_attempts attempts it is given up and the caller moves on.
    """

    total_attempts = 1 + retry_attempts

    for attempt in range(1, total_attempts + 1):
        print(
            f"{name}: sending Gemini request "
            f"(attempt {attempt}/{total_attempts})...",
            flush=True,
        )
        started = time.monotonic()
        try:
            description = gemini.describe_images(
                frame_paths, prompt, timeout_seconds=timeout_seconds
            )
        except Exception as error:  # noqa: BLE001 - keep the batch running
            elapsed = time.monotonic() - started
            timed_out = isinstance(error, gemini.GeminiTimeoutError)
            verdict = "TIMED OUT" if timed_out else "FAILED"
            print(
                f"{name}: attempt {attempt}/{total_attempts} {verdict} "
                f"after {elapsed:.1f} s — "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            if error_log is not None:
                error_log.record(
                    where=name,
                    error=error,
                    attempt=attempt,
                    total_attempts=total_attempts,
                    elapsed_seconds=round(elapsed, 2),
                    timed_out=timed_out,
                )
            if attempt < total_attempts and request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
        else:
            elapsed = time.monotonic() - started
            print(
                f"{name}: described "
                f"({len(frame_paths)} frames, {elapsed:.1f} s).",
                flush=True,
            )
            return description

    print(
        f"{name}: all {total_attempts} attempts failed, window skipped "
        "(will be retried on the next run).",
        flush=True,
    )
    return None
