"""H5 robustness conditions on Panel B (P3_PLAN.md section 5; definitions fixed 2026-10-22).

Twelve conditions plus clean, applied **to the whole image as distributed**, before the
deployed protocol's centre crop, identically to reals and fakes:

* **JPEG**, quality 95, 75, 50, 30: PIL ``quality=q``, chroma 4:2:0 (``subsampling=2``,
  PIL's default);
* **WebP**, quality 90, 70, 50: PIL lossy ``quality=q``;
* **resize-and-restore**, 0.75 and 0.5: bicubic down to ``round(side * f)``, then bicubic
  back to the original size;
* **Gaussian blur**, sigma 0.5, 1, 2: PIL ``GaussianBlur(radius=sigma)``.

User decisions of 2026-10-22: these definitions; the test rows of Panel B only; S2-D seed
0, P2 ``full_bank`` and P2 P-A. Each condition is deterministic for a given Pillow build.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

#: Condition id -> (operation, level); ``clean`` first, then the plan's order.
CONDITIONS: dict[str, tuple[str, float] | None] = {
    "clean": None,
    "jpeg95": ("jpeg", 95), "jpeg75": ("jpeg", 75), "jpeg50": ("jpeg", 50),
    "jpeg30": ("jpeg", 30),
    "webp90": ("webp", 90), "webp70": ("webp", 70), "webp50": ("webp", 50),
    "resize075": ("resize", 0.75), "resize050": ("resize", 0.5),
    "blur05": ("blur", 0.5), "blur1": ("blur", 1.0), "blur2": ("blur", 2.0),
}
#: The two conditions H5 judges (pre-registration section 6.6).
H5_DECISION = ("jpeg50", "resize050")


class PerturbError(ValueError):
    """An unknown condition or an unusable image."""


def _encode(image: Image.Image, fmt: str, **options: object) -> np.ndarray:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()


def apply_condition(img: np.ndarray, condition: str) -> np.ndarray:
    """The image under ``condition``: uint8 RGB in, uint8 RGB of the same size out.

    Raises:
        PerturbError: On an unknown condition or an input that is not HxWx3 uint8.
    """
    if condition not in CONDITIONS:
        raise PerturbError(f"unknown condition {condition!r}")
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise PerturbError(f"expected HxWx3 uint8, got {img.dtype} {img.shape}")
    spec = CONDITIONS[condition]
    if spec is None:
        return img
    op, level = spec
    image = Image.fromarray(img, "RGB")
    if op == "jpeg":
        return _encode(image, "JPEG", quality=int(level), subsampling=2)
    if op == "webp":
        return _encode(image, "WEBP", quality=int(level), lossless=False)
    if op == "resize":
        width, height = image.size
        small = (max(1, round(width * level)), max(1, round(height * level)))
        restored = image.resize(small, Image.Resampling.BICUBIC).resize(
            (width, height), Image.Resampling.BICUBIC)
        return np.asarray(restored, dtype=np.uint8).copy()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(level)))
    return np.asarray(blurred, dtype=np.uint8).copy()
