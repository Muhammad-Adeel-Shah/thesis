"""Tests for the B4 corpus filter, specifically the greyscale rejection (D-039).

`build_rows` reports *why* it produced nothing, because "too small" and "greyscale" are
different facts about the corpus and both belong in the summary. Eleven greyscale images
reached T-real while the filter said only "not produced", and the count went into
`skipped_too_small` where nobody would look for it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_builder():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "p1_build_reals", SCRIPTS / "p1_build_reals.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _colour(size: int) -> np.ndarray:
    rng = np.random.default_rng(11)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    base = 120.0 + 40.0 * np.sin(xs / 9.0) + 30.0 * np.cos(ys / 11.0)
    stack = np.stack([base, base * 0.8 + 30.0, base * 0.6 + 60.0], axis=-1)
    return np.clip(stack + rng.normal(0, 3, (size, size, 3)), 0, 255).astype(np.uint8)


def _write(array: np.ndarray, path: Path, mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).convert(mode).save(path, quality=95)
    return path


def test_a_colour_source_is_kept_and_flagged_empty(tmp_path):
    root, out = tmp_path / "src", tmp_path / "out"
    _write(_colour(96), root / "a.jpg")
    rows, reason = builder.build_rows(root / "a.jpg", root, "t_real", "real", 64, out)

    assert reason is None
    assert len(rows) == 1
    assert rows[0]["excluded_reason"] == ""
    assert (out / f"{rows[0]['image_id']}.png").is_file()


def test_a_greyscale_source_is_rejected_with_its_kind(tmp_path):
    root, out = tmp_path / "src", tmp_path / "out"
    _write(_colour(96), root / "grey.jpg", mode="L")
    rows, reason = builder.build_rows(root / "grey.jpg", root, "t_real", "real", 64, out)

    assert rows == []
    assert reason == "greyscale:rgb_equal"
    assert not list(out.rglob("*.png")), "a rejected source writes nothing"


def test_a_too_small_source_is_distinguished_from_a_greyscale_one(tmp_path):
    root, out = tmp_path / "src", tmp_path / "out"
    _write(_colour(32), root / "small.jpg")
    rows, reason = builder.build_rows(root / "small.jpg", root, "t_real", "real", 64, out)

    assert (rows, reason) == ([], "too_small")


def test_the_filter_can_be_switched_off_deliberately(tmp_path):
    """Off only by explicit argument, so no code path drops the filter by accident."""
    root, out = tmp_path / "src", tmp_path / "out"
    _write(_colour(96), root / "grey.jpg", mode="L")
    rows, reason = builder.build_rows(
        root / "grey.jpg", root, "t_real", "real", 64, out, skip_greyscale=False
    )

    assert reason is None
    assert len(rows) == 1


def test_a_locally_grey_crop_of_a_colour_photograph_is_kept(tmp_path):
    """The verdict is taken on the whole source, not on one crop.

    A colour photograph can have a colourless corner -- a grey wall, an overcast sky --
    and rejecting it would quietly shrink the corpus for no reason.
    """
    root, out = tmp_path / "src", tmp_path / "out"
    image = _colour(128)
    image[:64, :64] = 128  # a flat grey quadrant
    _write(image, root / "mixed.jpg")
    rows, reason = builder.build_rows(root / "mixed.jpg", root, "t_real", "real", 64, out)

    assert reason is None
    assert len(rows) == 1
