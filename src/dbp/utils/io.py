"""Image I/O for DBP.

Two rules are enforced here rather than left to call sites:

1. **Never resize.** Images are centre-cropped to a multiple of the decoder's
   downsampling factor. Resampling destroys the high-frequency generator traces the
   method relies on and introduces a train/test bias (decision D-001).
2. **PNG only, uint8.** Reconstructions are quantised the same way a real generator
   output is, so the residual is computed against a comparable representation
   (decision D-002).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_uint8(path: str | Path) -> np.ndarray:
    """Load an image as an ``uint8`` HWC RGB array.

    Uses Pillow, never OpenCV, so the channel order is RGB and not BGR.

    Args:
        path: Path to any Pillow-readable image.

    Returns:
        Array of shape ``(H, W, 3)`` and dtype ``uint8``.
    """
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HWC RGB after convert, got shape {arr.shape} for {path}")
    return arr


def reencode_jpeg(arr: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip an RGB image through JPEG at ``quality``, chroma 4:4:4, in memory.

    A format control (X1): every image -- real or fake, PNG or JPEG -- is given the same
    compression history, so a detector can no longer separate them by format. 4:4:4
    keeps chroma at full resolution, so the only change is the luminance and chroma
    quantisation at ``quality``; no file is written.

    Args:
        arr: ``(H, W, 3)`` ``uint8`` RGB.
        quality: libjpeg quality, 1-100.

    Returns:
        The decoded ``(H, W, 3)`` ``uint8`` RGB array.
    """
    import io

    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HWC uint8 RGB, got {arr.dtype} {arr.shape}")
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"JPEG quality must be in 1..100, got {quality}")
    buffer = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buffer, format="JPEG", quality=int(quality),
                                     subsampling=0, optimize=False, progressive=False)
    buffer.seek(0)
    with Image.open(buffer) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_png_uint8(arr: np.ndarray, path: str | Path) -> Path:
    """Write an ``uint8`` HWC RGB array to PNG atomically.

    The file is written to a temporary name in the destination directory and then
    moved into place with :func:`os.replace`, so a crash can never leave a
    half-written report artefact behind.

    Args:
        arr: Array of shape ``(H, W, 3)``, dtype ``uint8``.
        path: Destination path. Must end in ``.png``.

    Returns:
        The destination path.
    """
    if arr.dtype != np.uint8:
        raise TypeError(f"save_png_uint8 requires uint8, got {arr.dtype}")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"save_png_uint8 requires HWC RGB, got shape {arr.shape}")
    dest = Path(path)
    if dest.suffix.lower() != ".png":
        raise ValueError(f"reconstructions are stored as PNG only, got {dest.suffix!r}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".png.tmp")
    os.close(fd)
    try:
        Image.fromarray(arr, mode="RGB").save(tmp, format="PNG", compress_level=6)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


def center_crop_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    """Centre-crop ``img`` so both spatial dimensions are multiples of ``m``.

    This only ever removes at most ``m - 1`` pixels per side. It never resamples.

    Args:
        img: ``uint8`` HWC array.
        m: Required divisor (8 for ``vae_kl``, 32 for ``vae_dc``, 4 for ``sr_gan``).

    Returns:
        A cropped view with ``H % m == 0`` and ``W % m == 0``.
    """
    if m < 1:
        raise ValueError(f"crop multiple must be >= 1, got {m}")
    h, w = img.shape[:2]
    new_h, new_w = h - (h % m), w - (w % m)
    if new_h == 0 or new_w == 0:
        raise ValueError(f"image {h}x{w} is smaller than the crop multiple {m}")
    top, left = (h - new_h) // 2, (w - new_w) // 2
    return img[top : top + new_h, left : left + new_w]


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Centre-crop ``img`` to exactly ``size`` x ``size``.

    Args:
        img: ``uint8`` HWC array.
        size: Target side length. The image must be at least this large in both
            dimensions; this function never upscales or pads.

    Returns:
        A ``(size, size, 3)`` view of ``img``.
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        raise ValueError(f"cannot crop {h}x{w} to {size}x{size} without resizing")
    top, left = (h - size) // 2, (w - size) // 2
    return img[top : top + size, left : left + size]


def residual_int16(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Return ``int16(x) - int16(x_hat)``, the signed reconstruction residual.

    Both inputs are ``uint8``; the subtraction is widened to ``int16`` first so the
    result keeps its sign instead of wrapping.
    """
    if x.shape != x_hat.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {x_hat.shape}")
    if x.dtype != np.uint8 or x_hat.dtype != np.uint8:
        raise TypeError(f"residual expects uint8 inputs, got {x.dtype} and {x_hat.dtype}")
    return x.astype(np.int16) - x_hat.astype(np.int16)
