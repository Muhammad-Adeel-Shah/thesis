"""Perceptual-hash tests.

``dbp.data.phash`` is written here rather than taken from ``imagehash``, which exists
on Kaggle but not in the local lock file. Writing it means stating what it must do:

- identical images hash identically;
- an image that survived JPEG recompression stays within a few bits;
- unrelated images sit near the 32-bit separation two random 64-bit strings have;
- the threshold used for deduplication separates those two populations cleanly.

The last point is the one that matters. A threshold that does not separate them would
either miss a train/test leak or discard unrelated images as duplicates, and both
failures are invisible in the final numbers.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from dbp.data.phash import (
    DUPLICATE_THRESHOLD_BITS,
    find_cross_set_matches,
    find_duplicate_groups,
    hamming_distance,
    phash_file,
    phash_from_array,
    phash_from_hex,
    phash_hex,
)

RNG = np.random.default_rng(31337)


def _photo_like(height: int = 256, width: int = 256, seed: int = 0) -> np.ndarray:
    """A smooth, structured image. White noise is a bad pHash fixture: it has no
    low-frequency structure, so its DCT block is dominated by rounding."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    base = (
        80.0
        + 60.0 * np.sin(xs / (12.0 + rng.uniform(0, 8)) + rng.uniform(0, 6))
        + 50.0 * np.cos(ys / (17.0 + rng.uniform(0, 9)) + rng.uniform(0, 6))
        + 30.0 * np.sin((xs + ys) / (25.0 + rng.uniform(0, 10)))
    )
    stack = np.stack(
        [base, base * 0.85 + 20.0, base * 0.7 + 40.0 + rng.normal(0, 3, size=base.shape)],
        axis=-1,
    )
    return np.clip(stack, 0, 255).astype(np.uint8)


def _recompress(img: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def test_identical_images_hash_identically():
    img = _photo_like(seed=1)
    assert phash_from_array(img) == phash_from_array(img.copy())
    assert hamming_distance(phash_from_array(img), phash_from_array(img)) == 0


def test_hash_is_sixty_four_bits():
    value = phash_from_array(_photo_like(seed=2))
    assert 0 <= value < 2**64
    assert len(phash_hex(value)) == 16
    assert phash_from_hex(phash_hex(value)) == value


@pytest.mark.parametrize("quality", [95, 90, 75])
def test_jpeg_recompression_moves_the_hash_only_slightly(quality):
    img = _photo_like(seed=3)
    distance = hamming_distance(phash_from_array(img), phash_from_array(_recompress(img, quality)))
    assert distance <= DUPLICATE_THRESHOLD_BITS, (
        f"q={quality} moved the hash {distance} bits; a recompressed duplicate would "
        f"be missed by the leakage check"
    )


def test_a_small_brightness_change_is_tolerated():
    """Excluding the DC term from the median is what buys this."""
    img = _photo_like(seed=4)
    brighter = np.clip(img.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    assert hamming_distance(phash_from_array(img), phash_from_array(brighter)) <= (
        DUPLICATE_THRESHOLD_BITS
    )


def test_unrelated_images_are_far_apart():
    hashes = [phash_from_array(_photo_like(seed=seed)) for seed in range(12)]
    distances = [
        hamming_distance(a, b) for i, a in enumerate(hashes) for b in hashes[i + 1 :]
    ]
    assert min(distances) > DUPLICATE_THRESHOLD_BITS, (
        "unrelated images fall inside the duplicate threshold; deduplication would "
        "discard distinct images"
    )
    assert float(np.mean(distances)) > 16.0


def test_greyscale_and_rgb_of_the_same_picture_agree():
    img = _photo_like(seed=5)
    grey = np.asarray(Image.fromarray(img, mode="RGB").convert("L"), dtype=np.uint8)
    assert hamming_distance(phash_from_array(img), phash_from_array(grey)) == 0


def test_hashing_from_a_file_matches_hashing_from_the_array(tmp_path):
    img = _photo_like(seed=6)
    path = tmp_path / "x.png"
    Image.fromarray(img, mode="RGB").save(path)
    assert phash_file(path) == phash_from_array(img)


def test_duplicate_groups_collect_the_near_identical_and_nothing_else():
    img = _photo_like(seed=7)
    hashes = {
        "original": phash_from_array(img),
        "same_again": phash_from_array(img.copy()),
        "recompressed": phash_from_array(_recompress(img, 90)),
        "different_a": phash_from_array(_photo_like(seed=8)),
        "different_b": phash_from_array(_photo_like(seed=9)),
    }
    groups = find_duplicate_groups(hashes)
    assert groups == [["original", "recompressed", "same_again"]]


def test_cross_set_matches_find_a_planted_leak():
    shared = _photo_like(seed=10)
    train = {
        "train_0": phash_from_array(_photo_like(seed=11)),
        "train_1": phash_from_array(shared),
    }
    test = {
        "test_0": phash_from_array(_photo_like(seed=12)),
        "test_1": phash_from_array(_recompress(shared, 92)),
    }
    matches = find_cross_set_matches(train, test)
    assert [(left, right) for left, right, _ in matches] == [("train_1", "test_1")]


def test_disjoint_sets_report_no_leakage():
    train = {f"train_{i}": phash_from_array(_photo_like(seed=100 + i)) for i in range(6)}
    test = {f"test_{i}": phash_from_array(_photo_like(seed=200 + i)) for i in range(6)}
    assert find_cross_set_matches(train, test) == []


def test_invalid_shapes_and_sizes_are_rejected():
    with pytest.raises(ValueError, match="hash_size"):
        phash_from_array(_photo_like(), hash_size=0)
    with pytest.raises(ValueError, match="greyscale"):
        phash_from_array(RNG.integers(0, 255, size=(4, 4, 3, 2), dtype=np.uint8))
