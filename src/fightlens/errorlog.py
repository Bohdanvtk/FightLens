"""Per-run JSON error log.

Every error noticed during one launch of the pipeline is collected here
and mirrored into logs/errors_<launch date and time>.json, e.g.
logs/errors_2026-07-16_14-05-33.json. The file is created lazily, on the
first recorded error, so clean runs leave no empty files behind.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from fightlens.config import PROJECT_ROOT


# Fallback location for the per-run error files, used when no log_dir is
# given. The CLI passes the directory from the config (error_log_dir).
ERROR_LOG_DIR = PROJECT_ROOT / "logs"


class ErrorLog:
    """Collects the errors of one run and keeps them in one JSON file.

    The file is rewritten after every recorded error, so even a crashed
    or interrupted run keeps everything noticed up to that point.
    """

    def __init__(self, log_dir: str | Path = ERROR_LOG_DIR) -> None:
        self.started_at = datetime.now()
        self.path = (
            Path(log_dir) / f"errors_{self.started_at:%Y-%m-%d_%H-%M-%S}.json"
        )
        self.entries: list[dict[str, Any]] = []

    @property
    def count(self) -> int:
        """How many errors have been recorded so far."""

        return len(self.entries)

    def record(
        self, where: str, error: BaseException, **context: Any
    ) -> dict[str, Any]:
        """Store one error and immediately flush the JSON file.

        Args:
            where: What was being done when the error happened,
                e.g. "window_000004" or "run".
            error: The caught exception.
            **context: Extra fields saved with the entry, e.g.
                attempt=2, elapsed_seconds=30.1, timed_out=True.

        Returns:
            The entry that was stored.
        """

        entry: dict[str, Any] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "where": where,
            "error_type": type(error).__name__,
            "message": str(error),
            **context,
        }
        self.entries.append(entry)
        self._flush()
        return entry

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(self.entries, file, indent=2, ensure_ascii=False)
