"""Disk-space reporting and guards for Kaggle sessions (user rule, 2026-10-12).

Every stage logs ``df -h`` for the directories it writes to, and a run stops before its
first download when free space is below what it needs. ``df`` itself is printed when it
exists (Kaggle); the numbers recorded in reports come from ``shutil.disk_usage`` so they
are the same on every platform.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

GB = 1024**3


def usage(path: Path) -> dict[str, Any]:
    """Total, used and free bytes for the filesystem holding ``path``."""
    path = Path(path)
    probe = path if path.exists() else next((p for p in path.parents if p.exists()), path)
    total, used, free = shutil.disk_usage(probe)
    return {"path": str(path), "total_gb": round(total / GB, 2), "used_gb": round(used / GB, 2),
            "free_gb": round(free / GB, 2)}


def df_report(stage: str, paths: Iterable[Path], log: Any) -> dict[str, Any]:
    """Log ``df -h`` (when available) and return the recorded usage for ``stage``."""
    paths = [Path(p) for p in paths]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    record = {"stage": stage, "filesystems": [usage(p) for p in paths]}
    df = shutil.which("df")
    if df:
        result = subprocess.run([df, "-h", *map(str, paths)], capture_output=True, text=True,
                                check=False)
        for line in (result.stdout or result.stderr).splitlines():
            log.info("df [%s] %s", stage, line)
    for fs in record["filesystems"]:
        log.info("disk [%s] %s: free %.2f GB of %.2f GB", stage, fs["path"], fs["free_gb"],
                 fs["total_gb"])
    return record


def free_gb(path: Path) -> float:
    """Free space, in GiB, on the filesystem holding ``path``."""
    return usage(path)["free_gb"]


def require_free(needs: dict[Path, float]) -> list[str]:
    """Problems for every path whose free space is below its need (GiB); empty if fine."""
    return [f"{p}: {free_gb(p):.2f} GB free, the run needs {gb:.2f} GB"
            for p, gb in needs.items() if free_gb(p) < gb]
