"""The split script must not silently ignore a path it was given.

A `--pairs-manifest` that does not exist used to fall through to "pair rows inherit
their real's split at load time", which writes a 6,588-row split -- reals and
gen_matched only -- while the log says something reassuring. It happened twice.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
THESIS_DBP = Path(__file__).resolve().parents[1]


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path]:
    reports = tmp_path / "reports"
    _write(tmp_path / "instances.json", ["x"], [])          # replaced below
    (tmp_path / "instances.json").write_text(json.dumps({
        "categories": [{"id": 1, "supercategory": "person"},
                       {"id": 2, "supercategory": "appliance"}],
        "annotations": [{"image_id": i, "category_id": (i % 2) + 1} for i in range(1, 41)],
    }))

    real_cols = ["image_id", "source_image_id", "source_relative_path", "excluded_reason"]
    _write(reports / "p1_reals_manifest_t_real.csv", real_cols,
           [{"image_id": f"r{i}", "source_image_id": f"g{i}",
             "source_relative_path": f"{i:012d}.jpg", "excluded_reason": ""}
            for i in range(1, 21)])
    _write(reports / "p1_reals_manifest_t_ctrl.csv", real_cols,
           [{"image_id": f"c{i}_c0", "source_image_id": f"d{i}",
             "source_relative_path": f"{i:04d}.png", "excluded_reason": ""}
            for i in range(1, 6)])
    _write(reports / "p1_gen_matched_manifest.csv", ["gen_id", "source_real_id"],
           [{"gen_id": f"coco_{i:012d}", "source_real_id": "x"} for i in range(1, 11)])
    _write(reports / "p1_pairs_manifest.csv",
           ["pair_id", "image_id", "member", "relative_path"],
           [{"pair_id": f"r{i}_sd15", "image_id": f"r{i}", "member": "sd15",
             "relative_path": f"pairs/sd15/r{i}.png"} for i in range(1, 21)])
    return {"root": tmp_path, "reports": reports}


def _run(fixture: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable, str(SCRIPTS / "p2_build_splits.py"),
        "--instances", str(fixture["root"] / "instances.json"),
        "--manifests", str(fixture["reports"]),
        "--out", str(fixture["reports"] / "p2_splits.csv"),
        *extra,
    ]
    return subprocess.run(argv, capture_output=True, text=True, cwd=THESIS_DBP)


def test_a_missing_pairs_manifest_that_was_asked_for_is_fatal(
    fixture: dict[str, Path],
) -> None:
    result = _run(fixture, "--pairs-manifest", str(fixture["reports"] / "nope.csv"))
    assert result.returncode == 2
    assert "does not exist" in result.stderr + result.stdout
    assert not (fixture["reports"] / "p2_splits.csv").is_file()


def test_pair_rows_are_tagged_when_the_manifest_is_there(fixture: dict[str, Path]) -> None:
    result = _run(fixture, "--pairs-manifest", str(fixture["reports"] / "p1_pairs_manifest.csv"))
    assert result.returncode == 0, result.stderr
    rows = list(csv.DictReader((fixture["reports"] / "p2_splits.csv").open(encoding="utf-8")))
    kinds = {r["set_id"] for r in rows}
    assert "pair_sd15" in kinds
    assert len(rows) == 20 + 5 + 10 + 20      # reals + ctrl + gen + pairs


def test_omitting_the_flag_entirely_is_still_allowed(fixture: dict[str, Path]) -> None:
    """Not passing it is a choice; passing a path that is not there is a mistake."""
    result = _run(fixture)
    assert result.returncode == 0, result.stderr
    assert "no pair manifest given" in result.stderr + result.stdout
    rows = list(csv.DictReader((fixture["reports"] / "p2_splits.csv").open(encoding="utf-8")))
    assert len(rows) == 20 + 5 + 10
