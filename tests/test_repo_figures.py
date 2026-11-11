"""CLAUDE.md section 2b: the only images git may track are our own charts.

A tracked image must be an SVG, PDF or PNG under ``thesis_dbp/reports/**/figures/`` and
at most 1 MB. Anything else -- a dataset image, a crop, a reconstruction -- fails here
before it can be pushed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

THESIS_DBP = Path(__file__).resolve().parents[1]
REPO = THESIS_DBP.parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
                  ".svg", ".pdf", ".heic", ".avif"}
CHART_SUFFIXES = {".svg", ".pdf", ".png"}
MAX_BYTES = 1_000_000


def tracked_files() -> list[str]:
    """Every tracked file under thesis_dbp/, spelled ``thesis_dbp/...``.

    Run from thesis_dbp/ itself, so it works whether thesis_dbp/ is the repository root
    (since 2026-11-05) or a folder in it: ``git ls-files`` lists paths relative to the
    working directory either way.
    """
    result = subprocess.run(["git", "ls-files"], cwd=THESIS_DBP, capture_output=True,
                            text=True, check=False)
    if result.returncode != 0:
        pytest.skip("not a git checkout (NOT-RUN: git ls-files failed)")
    return [f"thesis_dbp/{line}" for line in result.stdout.splitlines() if line]


def is_allowed_chart(path: str, size: int) -> bool:
    parts = Path(path).parts
    return (Path(path).suffix.lower() in CHART_SUFFIXES
            and parts[:2] == ("thesis_dbp", "reports") and "figures" in parts[2:-1]
            and size <= MAX_BYTES)


def test_every_tracked_image_is_an_allowed_chart() -> None:
    offenders = []
    files = tracked_files()
    assert len(files) > 100, f"only {len(files)} tracked files found: the listing is wrong"
    for path in files:
        if Path(path).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        size = (REPO / path).stat().st_size
        if not is_allowed_chart(path, size):
            offenders.append(f"{path} ({size} bytes)")
    assert not offenders, f"images outside the section 2b rule: {offenders}"


@pytest.mark.parametrize(("path", "size", "allowed"), [
    ("thesis_dbp/reports/p2_x1/figures/x1.svg", 90_000, True),
    ("thesis_dbp/reports/a/b/figures/c.png", 1_000_000, True),
    ("thesis_dbp/reports/p2_x1/figures/x1.png", 1_000_001, False),     # too large
    ("thesis_dbp/reports/p2_x1/x1.png", 1_000, False),                 # not in figures/
    ("thesis_dbp/reports/figures/panel_0001.jpg", 1_000, False),       # not a chart type
    ("thesis_dbp/data/figures/x.png", 1_000, False),                   # not under reports
])
def test_the_rule(path: str, size: int, allowed: bool) -> None:
    assert is_allowed_chart(path, size) is allowed
