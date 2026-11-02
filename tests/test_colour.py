"""Tests for greyscale detection (D-039).

Two kinds have to be caught, and they are found in different ways. A single-channel
image is obvious. A greyscale picture written into a three-channel container is not:
its format, its mode and its channel count all look like colour, and only the pixels
give it away. Both reach a corpus of photographs -- eleven of T-real's 2,500 images
were greyscale -- and both are shortcuts a detector could learn instead of learning
anything about decoders.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dbp.data.colour import (
    GREYSCALE_TOLERANCE,
    RGB_EQUAL,
    SINGLE_CHANNEL,
    channel_spread,
    classify_colour,
)


def _colour(size: int = 32) -> np.ndarray:
    """An image whose channels genuinely differ."""
    rng = np.random.default_rng(2026)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _grey_in_rgb(size: int = 32) -> np.ndarray:
    """A greyscale picture stored as three identical channels."""
    rng = np.random.default_rng(7)
    plane = rng.integers(0, 256, (size, size), dtype=np.uint8)
    return np.stack([plane] * 3, axis=2)


def test_single_channel_is_greyscale():
    plane = np.full((8, 8), 120, dtype=np.uint8)
    for array in (plane, plane[:, :, None]):
        verdict = classify_colour(array)
        assert verdict["is_greyscale"] is True
        assert verdict["kind"] == SINGLE_CHANNEL
        assert verdict["channel_spread"] == 0


def test_three_equal_channels_are_greyscale():
    verdict = classify_colour(_grey_in_rgb())
    assert verdict["is_greyscale"] is True
    assert verdict["kind"] == RGB_EQUAL
    assert verdict["channel_spread"] == 0


def test_colour_is_not_greyscale():
    verdict = classify_colour(_colour())
    assert verdict["is_greyscale"] is False
    assert verdict["kind"] is None
    assert verdict["channel_spread"] > GREYSCALE_TOLERANCE


def test_spread_is_the_worst_pixel_not_the_average():
    """One coloured pixel in an otherwise grey image still makes it colour."""
    image = _grey_in_rgb()
    image[0, 0] = (10, 200, 60)
    assert channel_spread(image) == 190
    assert classify_colour(image)["is_greyscale"] is False


@pytest.mark.parametrize("offset", [0, 1, 2])
def test_tolerance_accepts_jpeg_rounding(offset):
    """A greyscale JPEG comes back with channels a unit or two apart, and is still grey."""
    image = _grey_in_rgb()
    image[..., 1] = np.clip(image[..., 1].astype(np.int16) + offset, 0, 255).astype(np.uint8)
    verdict = classify_colour(image)
    assert verdict["is_greyscale"] is True
    assert verdict["channel_spread"] <= GREYSCALE_TOLERANCE


def test_tolerance_rejects_a_spread_of_three():
    image = _grey_in_rgb()
    image[5, 5] = (100, 103, 100)
    assert classify_colour(image)["is_greyscale"] is False


def test_a_real_greyscale_jpeg_round_trip_is_detected():
    """The end-to-end case: a greyscale JPEG, re-read as RGB, is still caught.

    This is the shape the corpus actually had -- COCO stores a few black-and-white
    photographs, and `load_rgb_uint8` converts them to three channels on the way in.
    """
    buffer = io.BytesIO()
    Image.fromarray(_colour(64)).convert("L").save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    reloaded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)
    assert reloaded.shape[2] == 3
    assert classify_colour(reloaded)["is_greyscale"] is True


def test_non_uint8_is_refused():
    with pytest.raises(ValueError, match="uint8"):
        channel_spread(np.zeros((4, 4, 3), dtype=np.float32))


def test_an_unsupported_channel_count_is_refused():
    with pytest.raises(ValueError, match="expected"):
        channel_spread(np.zeros((4, 4, 4), dtype=np.uint8))
