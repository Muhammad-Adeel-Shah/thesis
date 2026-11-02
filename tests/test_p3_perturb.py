"""H5 conditions: exactly the plan's 12 + clean, deterministic, size-preserving."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dbp.eval import perturb


def _image(seed: int = 0, size: tuple[int, int] = (333, 401)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size[0], 0:size[1]]
    base = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(float)
    return np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)


def test_the_conditions_are_the_plans_twelve_plus_clean() -> None:
    assert list(perturb.CONDITIONS) == [
        "clean", "jpeg95", "jpeg75", "jpeg50", "jpeg30", "webp90", "webp70", "webp50",
        "resize075", "resize050", "blur05", "blur1", "blur2"]
    assert set(perturb.H5_DECISION) <= set(perturb.CONDITIONS)


@pytest.mark.parametrize("condition", list(perturb.CONDITIONS))
def test_each_condition_is_deterministic_and_keeps_the_size(condition: str) -> None:
    img = _image()
    a = perturb.apply_condition(img, condition)
    b = perturb.apply_condition(img.copy(), condition)
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b)
    if condition == "clean":
        assert a is img
    else:
        assert not np.array_equal(a, img)


def test_stronger_levels_change_the_image_more() -> None:
    img = _image(1).astype(float)

    def error(condition: str) -> float:
        return float(np.mean(np.abs(perturb.apply_condition(img.astype(np.uint8), condition)
                                    - img)))
    assert error("jpeg95") < error("jpeg50") < error("jpeg30")
    assert error("blur05") < error("blur1") < error("blur2")
    assert error("resize075") < error("resize050")


def test_jpeg_uses_420_chroma() -> None:
    from PIL import JpegImagePlugin

    buffer = io.BytesIO()
    Image.fromarray(_image(), "RGB").save(buffer, "JPEG", quality=50, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        assert JpegImagePlugin.get_sampling(decoded) == 2


def test_unknown_conditions_and_bad_inputs_are_refused() -> None:
    with pytest.raises(perturb.PerturbError):
        perturb.apply_condition(_image(), "jpeg10")
    with pytest.raises(perturb.PerturbError):
        perturb.apply_condition(_image()[..., 0], "clean")
