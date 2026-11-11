"""Tests for the I/O invariants the protocol depends on.

The two properties tested here are the ones whose violation would be invisible in the
results: a silent resize, or a lossy round-trip that changes the residual.
"""

from __future__ import annotations

import numpy as np
import pytest

from dbp.utils.io import (
    center_crop,
    center_crop_to_multiple,
    load_rgb_uint8,
    residual_int16,
    save_png_uint8,
)

RNG = np.random.default_rng(2026)


def _random_image(height: int, width: int) -> np.ndarray:
    return RNG.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    ("height", "width", "multiple"),
    [
        (512, 512, 8),
        (513, 519, 8),
        (500, 700, 8),
        (1000, 1000, 32),
        (1023, 1025, 32),
        (401, 403, 4),
        (64, 64, 8),
    ],
)
def test_crop_to_multiple_produces_valid_dimensions(height, width, multiple):
    out = center_crop_to_multiple(_random_image(height, width), multiple)
    assert out.shape[0] % multiple == 0
    assert out.shape[1] % multiple == 0


@pytest.mark.parametrize(("height", "width", "multiple"), [(513, 519, 8), (1023, 1025, 32)])
def test_crop_to_multiple_never_resizes(height, width, multiple):
    """Cropping must only remove border pixels; it must never resample."""
    img = _random_image(height, width)
    out = center_crop_to_multiple(img, multiple)

    # It shrinks by less than one full multiple on each axis.
    assert 0 <= height - out.shape[0] < multiple
    assert 0 <= width - out.shape[1] < multiple

    # Every surviving pixel is bit-identical to the source pixel it came from.
    top = (height - out.shape[0]) // 2
    left = (width - out.shape[1]) // 2
    expected = img[top : top + out.shape[0], left : left + out.shape[1]]
    assert np.array_equal(out, expected)


def test_crop_to_multiple_is_a_noop_when_already_aligned():
    img = _random_image(512, 512)
    assert np.array_equal(center_crop_to_multiple(img, 8), img)


def test_crop_to_multiple_rejects_images_smaller_than_the_multiple():
    with pytest.raises(ValueError, match="smaller than the crop multiple"):
        center_crop_to_multiple(_random_image(4, 4), 8)


def test_center_crop_refuses_to_upscale():
    """A crop that would need padding or resizing must raise, not silently resize."""
    with pytest.raises(ValueError, match="without resizing"):
        center_crop(_random_image(256, 256), 512)


def test_center_crop_exact_size():
    out = center_crop(_random_image(600, 800), 512)
    assert out.shape == (512, 512, 3)


def test_uint8_png_round_trip_is_exact(tmp_path):
    """PNG is lossless: the residual must not be polluted by the storage format."""
    img = _random_image(64, 96)
    path = save_png_uint8(img, tmp_path / "round_trip.png")
    reloaded = load_rgb_uint8(path)
    assert reloaded.dtype == np.uint8
    assert np.array_equal(reloaded, img)


def test_save_png_rejects_non_png_suffix(tmp_path):
    with pytest.raises(ValueError, match="PNG only"):
        save_png_uint8(_random_image(8, 8), tmp_path / "recon.jpg")


def test_save_png_rejects_non_uint8(tmp_path):
    with pytest.raises(TypeError, match="uint8"):
        save_png_uint8(_random_image(8, 8).astype(np.float32), tmp_path / "x.png")


def test_save_png_is_atomic_and_leaves_no_temp_files(tmp_path):
    save_png_uint8(_random_image(16, 16), tmp_path / "a.png")
    assert not list(tmp_path.glob("*.tmp"))


def test_residual_is_signed_and_does_not_wrap():
    """int16 widening keeps the sign; uint8 subtraction would wrap to 255."""
    dark = np.zeros((4, 4, 3), dtype=np.uint8)
    bright = np.full((4, 4, 3), 200, dtype=np.uint8)
    residual = residual_int16(dark, bright)
    assert residual.dtype == np.int16
    assert residual.min() == -200
    assert np.array_equal(residual_int16(bright, dark), -residual)


def test_residual_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        residual_int16(_random_image(8, 8), _random_image(8, 16))


# ---------------------------------------------------------------------------
# COMMIT.txt provenance fallback (env_report)
# ---------------------------------------------------------------------------


def test_commit_file_is_read_at_repo_root(tmp_path):
    from dbp.utils.env_report import read_commit_file

    sha = "a" * 40
    (tmp_path / "COMMIT.txt").write_text(sha + "\n", encoding="utf-8")
    assert read_commit_file(tmp_path) == sha


def test_commit_file_is_read_under_the_archive_prefix(tmp_path):
    """git archive --prefix=thesis_dbp/ puts COMMIT.txt one level down."""
    from dbp.utils.env_report import read_commit_file

    sha = "b" * 40
    nested = tmp_path / "thesis_dbp"
    nested.mkdir()
    (nested / "COMMIT.txt").write_text(sha, encoding="utf-8")
    assert read_commit_file(tmp_path) == sha


def test_missing_commit_file_returns_none(tmp_path):
    from dbp.utils.env_report import read_commit_file

    assert read_commit_file(tmp_path) is None


@pytest.mark.parametrize("bad", ["not-a-sha", "c" * 39, "g" * 40, ""])
def test_malformed_commit_file_is_rejected(tmp_path, bad):
    """A truncated or non-hex value must not be reported as a real commit."""
    from dbp.utils.env_report import read_commit_file

    (tmp_path / "COMMIT.txt").write_text(bad, encoding="utf-8")
    assert read_commit_file(tmp_path) is None


def test_collect_git_falls_back_to_commit_file_without_a_git_dir(tmp_path):
    from dbp.utils.env_report import collect_git

    sha = "d" * 40
    (tmp_path / "COMMIT.txt").write_text(sha, encoding="utf-8")
    info = collect_git(tmp_path)

    assert info["available"] is True
    assert info["commit"] == sha
    assert info["source"] == "COMMIT.txt"
    assert info["branch"] is None


def test_collect_git_reports_unavailable_with_neither_source(tmp_path):
    from dbp.utils.env_report import collect_git

    info = collect_git(tmp_path)
    assert info["available"] is False
    assert info.get("commit") is None
    assert "COMMIT.txt" in info["reason"]


# ---------------------------------------------------------------------------
# Logging must never be the reason a run dies (issue I-015)
# ---------------------------------------------------------------------------


def test_logging_falls_back_to_console_when_the_log_dir_is_unwritable(tmp_path, capsys):
    """On Kaggle the repo can be mounted read-only; in P0 that crashed two scripts.

    ``capsys`` rather than ``caplog``: ``setup_logging`` sets ``propagate = False``, so
    the record never reaches the root logger that ``caplog`` attaches to. Reading the
    console is also the more faithful test -- it is what the operator actually sees.
    """
    from dbp.utils.logging import setup_logging

    blocker = tmp_path / "blocker"
    blocker.write_text("this is a file, so it cannot also be a directory", encoding="utf-8")

    logger = setup_logging("unwritable_probe", log_dir=blocker / "logs")
    logger.info("the run continues")

    handler_types = {type(h).__name__ for h in logger.handlers}
    assert handler_types == {"StreamHandler"}, "a file handler was attached anyway"
    console = capsys.readouterr().out
    assert "log file disabled" in console
    assert "the run continues" in console


def test_logging_attaches_a_file_handler_when_the_directory_is_writable(tmp_path):
    from dbp.utils.logging import setup_logging

    logger = setup_logging("writable_probe", log_dir=tmp_path / "logs")
    assert "FileHandler" in {type(h).__name__ for h in logger.handlers}
    logger.info("hello")
    assert list((tmp_path / "logs").glob("writable_probe_*.log"))
