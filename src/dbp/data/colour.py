"""Greyscale detection for the real-image corpora.

A greyscale photograph in a corpus of colour photographs is a defect for this project
rather than a curiosity. Two reasons, both discovered in P1:

1. **It cannot define a chroma reference.** The JPEG-matched views (CLAUDE.md 6b) draw a
   ``(quality, chroma subsampling)`` pair from an observed real. A greyscale JPEG has no
   chroma channels at all, so Pillow reports its subsampling as ``-1`` and there is no
   honest value to give a colour image. Thirteen generations failed on exactly this.
2. **It is a shortcut.** "Has no colour" is trivially learnable and has nothing to do
   with which decoder produced an image. If one class contains black-and-white
   photographs and the other does not, a detector can score well without ever looking
   at a generative artefact.

Both kinds are caught here: images stored with a single channel, and images stored as
three channels whose channels are equal (a greyscale picture written into an RGB
container, which no format inspection would reveal).
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Largest per-pixel channel spread still considered greyscale. Not zero: a greyscale
#: picture that has been through JPEG comes back with channels that differ by a unit or
#: two from chroma rounding, and it is still a greyscale picture.
GREYSCALE_TOLERANCE = 2

#: Stored with one channel.
SINGLE_CHANNEL = "single_channel"

#: Stored with three channels that carry the same values.
RGB_EQUAL = "rgb_equal"


def channel_spread(img_uint8: np.ndarray) -> int:
    """The largest per-pixel difference between the brightest and dimmest channel.

    Args:
        img_uint8: ``(H, W)``, ``(H, W, 1)`` or ``(H, W, 3)`` ``uint8`` array.

    Returns:
        0 for a single-channel image, otherwise ``max(R,G,B) - min(R,G,B)`` maximised
        over every pixel.

    Raises:
        ValueError: If the array is not ``uint8`` or has an unsupported shape.
    """
    if img_uint8.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {img_uint8.dtype}")
    if img_uint8.ndim == 2 or (img_uint8.ndim == 3 and img_uint8.shape[2] == 1):
        return 0
    if img_uint8.ndim != 3 or img_uint8.shape[2] != 3:
        raise ValueError(f"expected (H, W), (H, W, 1) or (H, W, 3), got {img_uint8.shape}")
    # int16 so the subtraction cannot wrap around.
    channels = img_uint8.astype(np.int16)
    return int((channels.max(axis=2) - channels.min(axis=2)).max())


def classify_colour(
    img_uint8: np.ndarray, tolerance: int = GREYSCALE_TOLERANCE
) -> dict[str, Any]:
    """Decide whether an image is greyscale, and say which kind it is.

    Args:
        img_uint8: The image, as for :func:`channel_spread`.
        tolerance: Largest channel spread still called greyscale.

    Returns:
        ``{"is_greyscale": bool, "kind": str | None, "channel_spread": int}``. ``kind``
        is :data:`SINGLE_CHANNEL`, :data:`RGB_EQUAL`, or None for a colour image.
    """
    if img_uint8.ndim == 2 or (img_uint8.ndim == 3 and img_uint8.shape[2] == 1):
        return {"is_greyscale": True, "kind": SINGLE_CHANNEL, "channel_spread": 0}
    spread = channel_spread(img_uint8)
    if spread <= tolerance:
        return {"is_greyscale": True, "kind": RGB_EQUAL, "channel_spread": spread}
    return {"is_greyscale": False, "kind": None, "channel_spread": spread}
