"""Standard logging configuration for DBP scripts.

Absolute imports mean the stdlib ``logging`` module is still reachable from inside
this file, so the shadowing name is safe.

A log file is a convenience; the run is the point. On Kaggle the repository can end up
mounted read-only under ``/kaggle/input``, and in P0 that turned a logging detail into
a crash before any work had started (issue I-015). Since then an unwritable log
directory degrades to console-only and says so, rather than taking the script down.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def log_file_name(name: str, pid: int | None = None) -> str:
    """``<name>_<UTC second>_pid<pid>.log``.

    A timestamp alone is not unique: both S3 recovery jobs started in the same second
    (I-053), and two processes of one script would then have appended to one file. The
    process id makes the name unique per process.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}_{stamp}_pid{os.getpid() if pid is None else pid}.log"


def setup_logging(
    name: str,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger writing to the console and, optionally, to a file.

    Args:
        name: Logger name, normally the script's stem.
        log_dir: Directory for the log file. When given, a timestamped file
            ``<name>_<UTC timestamp>.log`` is created there.
        level: Threshold for both handlers.

    Returns:
        The configured logger. Calling this twice with the same name does not
        duplicate handlers.

    Note:
        If ``log_dir`` cannot be created or opened -- a read-only mount, for instance --
        the logger falls back to console-only and logs a warning naming the directory
        and the error. It never raises for that reason alone.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        directory = Path(log_dir)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(directory / log_file_name(name), encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "log file disabled: cannot write to %s (%s: %s). Continuing on console only.",
                directory,
                type(exc).__name__,
                exc,
            )
        else:
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger
