"""Tests for the greyscale re-scan (D-039), including the PNG-corpus defect (I-037).

The first version of this script asked "is the subsampling label mappable?" without
first asking "is this row even a JPEG?". On T-real, where every row is a JPEG, that
was harmless. On T-ctrl, where every row is a lossless PNG and the label is the
expected `not-jpeg`, it marked **1,594 colour DIV2K rows as greyscale**.

Two properties are asserted here, and both are about restraint rather than detection:
a row is excluded only when its pixels say greyscale, and an unreadable chroma setting
is reported without being treated as a verdict about the image.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from dbp.data.manifest import read_manifest, write_manifest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

COLUMNS = (
    "image_id", "set_id", "label", "relative_path", "source_image_id",
    "source_relative_path", "is_jpeg", "jpeg_quality", "subsampling",
)


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "p1_rescan_greyscale", SCRIPTS / "p1_rescan_greyscale.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rescan = _load()


class _Log:
    def info(self, message, *args):
        pass

    warning = error = info


def _colour(size: int = 48) -> np.ndarray:
    rng = np.random.default_rng(3)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    base = 120.0 + 40.0 * np.sin(xs / 7.0) + 30.0 * np.cos(ys / 9.0)
    stack = np.stack([base, base * 0.7 + 40.0, base * 0.5 + 70.0], axis=-1)
    return np.clip(stack + rng.normal(0, 2, (size, size, 3)), 0, 255).astype(np.uint8)


def _grey(size: int = 48) -> np.ndarray:
    plane = _colour(size)[..., 0]
    return np.stack([plane] * 3, axis=2)


def _corpus(tmp_path: Path, rows: list[dict], arrays: list[np.ndarray]) -> tuple[Path, Path]:
    """Write the crops and a manifest describing them."""
    reals_root = tmp_path / "outputs" / "p1" / "reals"
    for row, array in zip(rows, arrays, strict=True):
        path = reals_root.parent / row["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
    manifest = tmp_path / "reports" / "p1_reals_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, manifest, COLUMNS)
    return reals_root, manifest


def _row(index: int, set_id: str, is_jpeg: bool, subsampling: str, quality="") -> dict:
    return {
        "image_id": f"img_{index:03d}",
        "set_id": set_id,
        "label": "real",
        "relative_path": f"reals/{set_id}/img_{index:03d}.png",
        "source_image_id": f"src_{index:03d}",
        "source_relative_path": f"{index:04d}.png",
        "is_jpeg": is_jpeg,
        "jpeg_quality": quality,
        "subsampling": subsampling,
    }


def test_a_png_corpus_excludes_only_the_greyscale_rows(tmp_path):
    """I-037: `not-jpeg` is the expected label for a PNG, not an unreadable one."""
    rows = [_row(0, "t_ctrl", False, "not-jpeg"),
            _row(1, "t_ctrl", False, "not-jpeg"),
            _row(2, "t_ctrl", False, "not-jpeg")]
    reals_root, _ = _corpus(tmp_path, rows, [_colour(), _colour(), _grey()])

    findings, counts = rescan.scan(rows, reals_root, rescan.GREYSCALE_TOLERANCE, _Log())
    excluded = [f for f in findings if f["excluded"]]

    assert counts["greyscale"] == 1
    assert counts.get("unmappable_but_colour", 0) == 0, "a PNG has no chroma setting to lose"
    assert [f["image_id"] for f in excluded] == ["img_002"]
    assert excluded[0]["kind"] == "rgb_equal", "a PNG row cannot claim a single-channel source"


def test_a_jpeg_with_an_unreadable_chroma_setting_is_reported_but_kept(tmp_path):
    """Colour pixels are colour pixels; only the chroma reference is affected."""
    rows = [_row(0, "t_real", True, "unknown(-1)", 96), _row(1, "t_real", True, "4:4:4", 96)]
    reals_root, _ = _corpus(tmp_path, rows, [_colour(), _colour()])

    findings, counts = rescan.scan(rows, reals_root, rescan.GREYSCALE_TOLERANCE, _Log())

    assert counts["greyscale"] == 0
    assert counts["unmappable_but_colour"] == 1
    assert [f["excluded"] for f in findings] == [False]


def test_a_greyscale_jpeg_is_named_by_its_source(tmp_path):
    rows = [_row(0, "t_real", True, "unknown(-1)", 96), _row(1, "t_real", True, "4:4:4", 96)]
    reals_root, _ = _corpus(tmp_path, rows, [_grey(), _grey()])

    findings, counts = rescan.scan(rows, reals_root, rescan.GREYSCALE_TOLERANCE, _Log())

    assert counts["greyscale"] == 2
    assert counts["single_channel_source"] == 1
    assert counts["rgb_equal"] == 1
    assert all(f["excluded"] for f in findings)


def test_the_manifest_is_rewritten_with_only_the_greyscale_rows_flagged(tmp_path, monkeypatch):
    """End to end, because the damage was done by main(), not by scan()."""
    rows = [_row(i, "t_ctrl", False, "not-jpeg") for i in range(4)]
    reals_root, manifest = _corpus(
        tmp_path, rows, [_colour(), _colour(), _grey(), _colour()]
    )
    monkeypatch.setattr(rescan, "THESIS_DBP", tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    argv = ["p1_rescan_greyscale.py", "--set-id", "t_ctrl", "--manifest", str(manifest),
            "--reals-root", str(reals_root)]
    old = sys.argv
    try:
        sys.argv = argv
        assert rescan.main() == 0
    finally:
        sys.argv = old

    written = read_manifest(manifest)
    flagged = [r["image_id"] for r in written if r["excluded_reason"]]
    assert flagged == ["img_002"], "colour rows must keep an empty excluded_reason"

    summary = json.loads((tmp_path / "reports" / "p1_greyscale_rescan_t_ctrl.json").read_text())
    assert summary["rows_excluded"] == 1
    assert summary["rows_after"] == 3
