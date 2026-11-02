"""X1 format control: the re-encode, the image list and the runbook's expectations."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.utils.io import reencode_jpeg  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("x1_format_control_test",
                                                  ROOT / "scripts" / "x1_format_control.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["x1_format_control_test"] = module
    spec.loader.exec_module(module)
    return module


fc = _load()


def _chroma_edges(size: int = 64) -> np.ndarray:
    """Red/blue columns one pixel wide: chroma detail that 4:2:0 must blur."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, 0::2] = (220, 30, 30)
    image[:, 1::2] = (30, 30, 220)
    return image


def test_reencode_keeps_shape_and_dtype_and_is_deterministic() -> None:
    image = np.random.default_rng(0).integers(0, 256, (37, 53, 3), dtype=np.uint8)
    once, twice = reencode_jpeg(image, 95), reencode_jpeg(image, 95)
    assert once.shape == image.shape and once.dtype == np.uint8
    np.testing.assert_array_equal(once, twice)
    assert np.any(once != image)                        # it really went through JPEG


def test_reencode_is_444_not_420() -> None:
    image = _chroma_edges()
    ours = reencode_jpeg(image, 95).astype(int)
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95, subsampling=2)
    buffer.seek(0)
    subsampled = np.asarray(Image.open(buffer).convert("RGB")).astype(int)
    assert np.abs(ours - image).mean() < 0.5 * np.abs(subsampled - image).mean()


@pytest.mark.parametrize("quality", [0, 101])
def test_reencode_rejects_a_bad_quality(quality: int) -> None:
    with pytest.raises(ValueError):
        reencode_jpeg(_chroma_edges(), quality)


def test_build_list_refuses_an_e7_row_without_the_panel_root() -> None:
    with pytest.raises(ValueError, match="root=panel"):
        fc.build_list([], {}, [{"image_id": "a", "relative_path": "e7/a.png", "label": "real",
                                "view": "e7_dda_coco", "root": "corpus"}])


def test_the_committed_list_matches_the_runbook_expectation() -> None:
    """The runbook checks the list's sha256 on Kaggle; a regenerated list must update it.

    Hashed on LF bytes, which is what git stores and the code zip carries.
    """
    expect = json.loads((ROOT / "env" / "x1_format_expect.json").read_text(encoding="utf-8"))
    data = (ROOT / expect["images"]).read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == expect["images_sha256"]
    assert data.count(b"\n") - 1 == expect["images_rows"] == 14205
    summary = json.loads((ROOT / "reports" / "p2_x1" / "x1_format_control_images_summary.json")
                         .read_text(encoding="utf-8"))
    assert summary["rows"] == expect["images_rows"]
    assert summary["counts"]["total"] == expect["images_rows"]
    assert expect["quality"] == fc.QUALITY == 95
