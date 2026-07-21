"""Shared atomic file writes (temp file + rename) used by every step's output."""

import os
import tempfile
from pathlib import Path
from typing import Callable, IO


def atomic_write(path: str | Path, write: Callable[[IO], None], mode: str = "w") -> None:
    """Write `path` via write(file), atomically. A crash mid-write leaves no partial file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as file:
            write(file)
        os.replace(temp_name, path)
    except BaseException:
        if os.path.exists(temp_name):
            os.remove(temp_name)
        raise
