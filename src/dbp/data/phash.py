"""Perceptual hashing for deduplication and train/test leakage detection.

The DCT perceptual hash (pHash): greyscale, downsample, 2-D DCT-II, keep the
low-frequency block, threshold at its median. Two images that differ only by
recompression, a small resize or light retouching land within a few bits of each other;
unrelated images land near 32 bits apart in a 64-bit hash.

**Why this is implemented here rather than taken from ``imagehash``.** ``imagehash``
4.3.2 is preinstalled on Kaggle but absent from the local lock file, so depending on it
would mean the local and Kaggle environments could not run the same code. The
algorithm is about twenty lines over ``scipy.fft.dct``, which both environments already
have, so the asymmetry is not worth the dependency. The unit tests state the properties
the implementation must satisfy rather than trusting it by construction.

**The resize here is not a protocol violation.** D-001 forbids resizing images that are
fed to a decoder, because resampling destroys the high-frequency generator traces the
method reads. A perceptual hash is the opposite problem: it must discard exactly those
details in order to recognise the same picture twice. Nothing in this module writes a
pixel; it only reports numbers about images that stay untouched on disk.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct

#: Side length of the low-frequency block kept from the DCT. 8 gives a 64-bit hash.
DEFAULT_HASH_SIZE = 8

#: The image is reduced to ``hash_size * HIGHFREQ_FACTOR`` square before the DCT.
HIGHFREQ_FACTOR = 4

#: Hamming distance at or below which two images are treated as the same picture.
#: Chosen to sit well below the ~32-bit separation of unrelated images while still
#: absorbing recompression; :func:`phash` is measured against both ends in the tests.
DUPLICATE_THRESHOLD_BITS = 6


def _dct2(block: np.ndarray) -> np.ndarray:
    """Orthonormal 2-D DCT-II, applied along both axes."""
    return dct(dct(block, axis=0, norm="ortho"), axis=1, norm="ortho")


def phash_from_array(
    img_uint8: np.ndarray,
    hash_size: int = DEFAULT_HASH_SIZE,
    highfreq_factor: int = HIGHFREQ_FACTOR,
) -> int:
    """Compute the DCT perceptual hash of an in-memory image.

    Args:
        img_uint8: ``(H, W, 3)`` ``uint8`` RGB array, or ``(H, W)`` greyscale.
        hash_size: Side of the kept DCT block. The hash has ``hash_size ** 2`` bits.
        highfreq_factor: The image is reduced to ``hash_size * highfreq_factor`` square
            first, so the DCT block sits in the low-frequency corner.

    Returns:
        The hash as an integer whose bit length is at most ``hash_size ** 2``. Bit 0 is
        the last cell of the block, matching the reading order used by ``phash_hex``.

    Raises:
        ValueError: If ``hash_size`` is not positive.
    """
    if hash_size < 1:
        raise ValueError(f"hash_size must be >= 1, got {hash_size}")

    if img_uint8.ndim == 3:
        image = Image.fromarray(np.ascontiguousarray(img_uint8), mode="RGB").convert("L")
    elif img_uint8.ndim == 2:
        image = Image.fromarray(np.ascontiguousarray(img_uint8), mode="L")
    else:
        raise ValueError(f"expected HWC RGB or HW greyscale, got shape {img_uint8.shape}")

    side = hash_size * highfreq_factor
    reduced = image.resize((side, side), resample=Image.Resampling.LANCZOS)
    coefficients = _dct2(np.asarray(reduced, dtype=np.float64))
    block = coefficients[:hash_size, :hash_size]

    # The DC term carries overall brightness, not structure. Excluding it from the
    # median keeps the hash from flipping wholesale when an image is brightened.
    without_dc = block.flatten()[1:]
    median = float(np.median(without_dc))

    bits = (block.flatten() > median).astype(np.uint8)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash_file(
    path: str | Path,
    hash_size: int = DEFAULT_HASH_SIZE,
    highfreq_factor: int = HIGHFREQ_FACTOR,
) -> int:
    """Compute the perceptual hash of an image on disk.

    The file is opened with Pillow and converted to RGB first, so a palette, greyscale
    or CMYK source hashes the same way a true-colour one does.
    """
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    return phash_from_array(array, hash_size=hash_size, highfreq_factor=highfreq_factor)


def phash_hex(value: int, hash_size: int = DEFAULT_HASH_SIZE) -> str:
    """Render a hash as fixed-width lowercase hex, for manifests and diffs."""
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def phash_from_hex(text: str) -> int:
    """Parse a hash written by :func:`phash_hex`."""
    return int(text, 16)


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return int((a ^ b).bit_count())


def find_duplicate_groups(
    hashes: Mapping[str, int], threshold: int = DUPLICATE_THRESHOLD_BITS
) -> list[list[str]]:
    """Group ids whose hashes are within ``threshold`` bits of each other.

    This is an O(n^2) scan over the ids. At P1 scale -- a few thousand images per set --
    that is a couple of seconds, and it is exact. An approximate index would be faster
    and would risk missing the one leak that matters.

    Args:
        hashes: Mapping of image id to perceptual hash.
        threshold: Maximum Hamming distance for two images to count as the same.

    Returns:
        Groups of two or more ids, each group sorted, the list sorted. Ids that match
        nothing are omitted.
    """
    ids = sorted(hashes)
    parent = {key: key for key in ids}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if hamming_distance(hashes[left], hashes[right]) <= threshold:
                parent[find(right)] = find(left)

    groups: dict[str, list[str]] = {}
    for key in ids:
        groups.setdefault(find(key), []).append(key)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: g[0])


def find_cross_set_matches(
    left: Mapping[str, int],
    right: Mapping[str, int],
    threshold: int = DUPLICATE_THRESHOLD_BITS,
) -> list[tuple[str, str, int]]:
    """Find images that appear in both of two sets -- i.e. train/test leakage.

    Args:
        left: Mapping of id to hash for the first set (normally the training corpus).
        right: Mapping of id to hash for the second set (normally an evaluation set).
        threshold: Maximum Hamming distance for a match.

    Returns:
        ``(left_id, right_id, distance)`` for every match, sorted by distance then by
        id. An empty list means no detected overlap, which is the required outcome for
        every evaluation set.
    """
    matches = [
        (left_id, right_id, hamming_distance(left_hash, right_hash))
        for left_id, left_hash in left.items()
        for right_id, right_hash in right.items()
        if hamming_distance(left_hash, right_hash) <= threshold
    ]
    return sorted(matches, key=lambda m: (m[2], m[0], m[1]))


def hash_many(paths: Iterable[tuple[str, Path]]) -> dict[str, int]:
    """Hash a sequence of ``(id, path)`` pairs into a mapping usable by the finders."""
    return {image_id: phash_file(path) for image_id, path in paths}
