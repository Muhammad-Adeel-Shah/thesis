"""JPEG forensics: what compression an image has already been through.

Two jobs.

**Describing the corpus.** "Fake or JPEG?" [R7] showed that a detector trained where
the reals are JPEG and the fakes are PNG learns the compression history and reports it
as a detection rate. That confound is present, and severe, in E3 as distributed. Every
evaluation set therefore gets a table of ``label x (format, JPEG, quality,
subsampling)`` in its data card, built from the functions here.

**Removing the confound.** Where compression history tracks the label, the never-
compressed fakes are re-encoded to JPEG with quality and chroma subsampling drawn from
that set's *own* real-image distribution, one draw per image, seeded by a hash of the
image id so the view is reproducible from the manifest alone. Real images are never
re-encoded: touching them would replace a real compression history with a synthetic
one, which is the same mistake in the other direction.

Quality is estimated from the quantisation tables using the IJG scaling rule, which is
a property of the format rather than of any particular encoder. The tables below are
the Annex K baseline tables; :mod:`tests.test_jpeg_quality` checks the estimator
recovers Pillow's own quality setting rather than taking the constants on trust.

The encoder itself is not portable, which is why I-010 exists: the same array encoded
at the same quality by two libjpeg builds can differ byte for byte.
:func:`encode_jpeg_bytes` is the single place an encode happens, so the comparison in
``scripts/p1_jpeg_equivalence.py`` measures the encoder and nothing else.
"""

from __future__ import annotations

import hashlib
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, JpegImagePlugin

#: ITU-T T.81 Annex K.1, the baseline luminance quantisation table.
ANNEX_K_LUMINANCE: tuple[int, ...] = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
)  # fmt: skip

#: ITU-T T.81 Annex K.2, the baseline chrominance quantisation table.
ANNEX_K_CHROMINANCE: tuple[int, ...] = (
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
)  # fmt: skip

#: Pillow's ``subsampling`` codes, as human-readable labels.
SUBSAMPLING_LABELS: dict[int, str] = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}

#: Label used when a file is not a JPEG at all, so the column is never blank.
NOT_JPEG = "not-jpeg"


def scaled_table(base: Sequence[int], quality: int) -> tuple[int, ...]:
    """Scale a baseline quantisation table to a quality setting, the IJG way.

    Args:
        base: One of :data:`ANNEX_K_LUMINANCE` / :data:`ANNEX_K_CHROMINANCE`.
        quality: 1..100.

    Returns:
        The 64 scaled, clamped table entries.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in 1..100, got {quality}")
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    return tuple(min(255, max(1, (value * scale + 50) // 100)) for value in base)


def estimate_quality_from_tables(tables: dict[int, Sequence[int]]) -> int | None:
    """Estimate the JPEG quality setting from the quantisation tables.

    Every candidate quality is scored by the total absolute difference between the
    file's tables and the Annex K tables scaled to that quality; the best-scoring
    quality wins. An encoder using non-standard tables will still produce a number, so
    the score is not evidence on its own -- but for the IJG-derived encoders that
    produced these corpora it recovers the setting exactly.

    Args:
        tables: Pillow's ``Image.quantization`` mapping, table index to 64 values.

    Returns:
        The best-scoring quality in 1..100, or None if there are no usable tables.
    """
    if not tables:
        return None

    luminance = tables.get(0)
    chrominance = tables.get(1)
    if luminance is None:
        return None

    best_quality: int | None = None
    best_score: float | None = None
    for quality in range(1, 101):
        score = float(
            np.abs(
                np.asarray(luminance, dtype=np.int64)
                - np.asarray(scaled_table(ANNEX_K_LUMINANCE, quality), dtype=np.int64)
            ).sum()
        )
        if chrominance is not None:
            score += float(
                np.abs(
                    np.asarray(chrominance, dtype=np.int64)
                    - np.asarray(scaled_table(ANNEX_K_CHROMINANCE, quality), dtype=np.int64)
                ).sum()
            )
        if best_score is None or score < best_score:
            best_score, best_quality = score, quality
    return best_quality


def jpeg_metadata(path: str | Path) -> dict[str, Any]:
    """Describe one file's compression history.

    Args:
        path: Any Pillow-readable image.

    Returns:
        A mapping with ``format``, ``is_jpeg``, ``jpeg_quality``, ``subsampling``,
        ``progressive``, ``width`` and ``height``. For a non-JPEG, ``jpeg_quality`` is
        None and ``subsampling`` is :data:`NOT_JPEG`; the row is still complete, so a
        data-card table never has a blank cell to interpret.
    """
    with Image.open(path) as image:
        image_format = (image.format or "").upper()
        width, height = image.size
        is_jpeg = image_format in ("JPEG", "MPO")
        if not is_jpeg:
            return {
                "format": image_format or "UNKNOWN",
                "is_jpeg": False,
                "jpeg_quality": None,
                "subsampling": NOT_JPEG,
                "progressive": False,
                "width": width,
                "height": height,
            }
        quantization = dict(getattr(image, "quantization", {}) or {})
        sampling = JpegImagePlugin.get_sampling(image)
        progressive = bool(image.info.get("progressive", 0) or image.info.get("progression", 0))

    return {
        "format": image_format,
        "is_jpeg": True,
        "jpeg_quality": estimate_quality_from_tables(quantization),
        "subsampling": SUBSAMPLING_LABELS.get(sampling, f"unknown({sampling})"),
        "progressive": progressive,
        "width": width,
        "height": height,
    }


def encode_jpeg_bytes(
    img_uint8: np.ndarray, quality: int, subsampling: int = 0, progressive: bool = False
) -> bytes:
    """Encode an ``uint8`` HWC RGB array to JPEG and return the bytes.

    The single place in the project where a JPEG is written. Keeping it to one call
    site is what lets ``p1_jpeg_equivalence.py`` attribute any local/Kaggle byte
    difference to libjpeg rather than to two slightly different call sites.

    Args:
        img_uint8: ``(H, W, 3)`` ``uint8`` RGB array. Never resized.
        quality: 1..100, passed to Pillow unchanged.
        subsampling: Pillow's code: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0.
        progressive: Whether to write a progressive JPEG.

    Returns:
        The encoded bytes.
    """
    if img_uint8.dtype != np.uint8:
        raise TypeError(f"encode_jpeg_bytes requires uint8, got {img_uint8.dtype}")
    if img_uint8.ndim != 3 or img_uint8.shape[2] != 3:
        raise ValueError(f"encode_jpeg_bytes requires HWC RGB, got shape {img_uint8.shape}")
    if not 1 <= quality <= 100:
        raise ValueError(f"quality must be in 1..100, got {quality}")
    if subsampling not in SUBSAMPLING_LABELS:
        raise ValueError(f"subsampling must be one of {sorted(SUBSAMPLING_LABELS)}")

    buffer = io.BytesIO()
    Image.fromarray(img_uint8, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
        subsampling=int(subsampling),
        progressive=bool(progressive),
    )
    return buffer.getvalue()


def subsampling_code(label: str) -> int:
    """Invert :data:`SUBSAMPLING_LABELS`, raising on anything unrecognised."""
    for code, name in SUBSAMPLING_LABELS.items():
        if name == label:
            return code
    raise ValueError(f"unknown chroma subsampling label {label!r}")


def seeded_rng(image_id: str, salt: str) -> np.random.Generator:
    """A generator whose stream depends only on the image id and a salt.

    Deterministic across machines and across runs, and independent of the order in
    which images are processed, so a rebuilt JPEG-matched view is byte-identical even
    if the file listing comes back in a different order. The seed is derived with
    BLAKE2b rather than :func:`hash`, whose salt changes between processes.
    """
    digest = hashlib.blake2b(f"{salt}:{image_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def sample_matched_pair(
    image_id: str,
    reference: Sequence[tuple[int, str]],
    salt: str,
) -> tuple[int, int, str]:
    """Draw one observed (quality, subsampling) pair from a reference distribution.

    The reference is the list of ``(quality, subsampling)`` actually observed on the
    real images being matched against, **repeats included**, so drawing an index
    uniformly reproduces their distribution rather than a smoothed idea of it.

    The pair is drawn **jointly**, not as two independent draws. Independent draws can
    produce combinations that occur nowhere in the reference -- quality 96 with 4:2:0
    chroma, say, when every high-quality real in the set is 4:4:4 -- and such a
    combination is itself a detectable artefact, which is the thing this whole
    mechanism exists to avoid.

    Args:
        image_id: Stable id of the image being re-encoded; the only source of the seed.
        reference: Observed ``(quality, subsampling_label)`` pairs, with repeats.
        salt: A per-set string, so the same image id in two sets does not draw the same
            parameters.

    Returns:
        ``(index, quality, subsampling_label)``. The index identifies which reference
        image the parameters came from, so the manifest can name it.

    Raises:
        ValueError: If the reference is empty, which means there is no distribution to
            match and the honest answer is to report the confound rather than invent one.
    """
    if not reference:
        raise ValueError(f"{image_id}: no reference (quality, subsampling) pairs to sample from")
    rng = seeded_rng(image_id, salt)
    index = int(rng.integers(len(reference)))
    quality, subsampling = reference[index]
    return index, int(quality), str(subsampling)


def compression_is_confounded_with_label(
    by_label: dict[str, dict[str, int]], min_share: float = 0.05
) -> tuple[bool, str]:
    """Decide whether a set needs a JPEG-matched view.

    Args:
        by_label: ``{label: {"jpeg": n, "non_jpeg": n}}``.
        min_share: A label whose never-compressed share differs from another label's by
            less than this is not treated as confounded.

    Returns:
        ``(confounded, reason)``. The reason is written into the data card either way,
        so a set that does *not* need a matched view still records why.
    """
    shares: dict[str, float] = {}
    for label, counts in by_label.items():
        total = counts.get("jpeg", 0) + counts.get("non_jpeg", 0)
        if total == 0:
            continue
        shares[label] = counts.get("non_jpeg", 0) / total
    if len(shares) < 2:
        return False, "fewer than two labels with images; nothing to compare"
    low_label, low = min(shares.items(), key=lambda item: item[1])
    high_label, high = max(shares.items(), key=lambda item: item[1])
    spread = high - low
    detail = (
        f"never-compressed share: {high_label} {high:.1%} vs {low_label} {low:.1%} "
        f"(spread {spread:.1%})"
    )
    if spread < min_share:
        return False, f"not confounded; {detail}"
    return True, f"CONFOUNDED; {detail}"


#: The subsampling labels that can be turned back into an encoder setting.
#: Anything else -- in practice ``unknown(-1)``, which is what Pillow reports for
#: a greyscale JPEG -- cannot define a chroma setting for a colour image, and
#: drawing one raises. Defined once here because two call sites need it: the
#: training-side matcher and the evaluation matched view (D-039, I-039).
MAPPABLE_SUBSAMPLING = frozenset(SUBSAMPLING_LABELS.values())


#: A single parameter that predicts the label this much better than always guessing the
#: majority class is a shortcut, not a detail. Five points is deliberately low: the case
#: that motivated this rule was +49.9.
SEPARABILITY_LIFT_THRESHOLD = 0.05


def parameter_separability(
    rows: Sequence[Mapping[str, Any]], parameter: str, label_key: str = "label"
) -> dict[str, Any]:
    """How well one encoding parameter alone predicts the label.

    ``compression_is_confounded_with_label`` asks whether an image was compressed. This
    asks the sharper question: *how* was it compressed, and does that give the label
    away. In E7 (DDA-COCO) every FLUX.1 reconstruction is 4:2:0 and every real is 4:4:4,
    so on that subset ``4:2:0 -> fake`` is a perfect classifier -- while the
    compressed-versus-not test reported "not confounded", because both classes are
    entirely JPEG (D-042).

    The measure is the accuracy of the best possible rule that looks only at this
    parameter: map each observed value to its majority label. Comparing it against
    always predicting the majority class gives the *lift*, which is what matters -- on
    an unbalanced set a high accuracy can mean nothing at all.

    Args:
        rows: Manifest rows.
        parameter: The column to test, e.g. ``"subsampling"`` or ``"jpeg_quality"``.
        label_key: The column holding the label.

    Returns:
        ``{"parameter", "n", "accuracy", "baseline", "lift", "separable", "values"}``.
        ``values`` maps each observed value to its per-label counts, so a reader can see
        *which* value gave the game away rather than only that something did.
    """
    counts: dict[str, Counter] = {}
    labels: Counter = Counter()
    for row in rows:
        value = str(row.get(parameter, ""))
        label = str(row.get(label_key, ""))
        counts.setdefault(value, Counter())[label] += 1
        labels[label] += 1

    total = sum(labels.values())
    if total == 0 or len(labels) < 2:
        return {
            "parameter": parameter,
            "n": total,
            "accuracy": None,
            "baseline": None,
            "lift": None,
            "separable": False,
            "reason": "fewer than two labels; separability is undefined",
            "values": {value: dict(per) for value, per in sorted(counts.items())},
        }

    correct = sum(max(per.values()) for per in counts.values())
    accuracy = correct / total
    baseline = max(labels.values()) / total
    lift = accuracy - baseline
    return {
        "parameter": parameter,
        "n": total,
        "accuracy": round(accuracy, 6),
        "baseline": round(baseline, 6),
        "lift": round(lift, 6),
        "separable": lift > SEPARABILITY_LIFT_THRESHOLD,
        "reason": (
            f"{parameter} alone scores {accuracy:.2%} against a {baseline:.2%} "
            f"majority-class baseline (lift {lift:+.2%})"
        ),
        "values": {value: dict(per) for value, per in sorted(counts.items())},
    }


def separability_by_stratum(
    rows: Sequence[Mapping[str, Any]],
    parameter: str,
    stratum_key: str,
    label_key: str = "label",
    positive_label: str = "fake",
) -> dict[str, Any]:
    """Run :func:`parameter_separability` per stratum, each against the shared reals.

    A pooled figure hides exactly the case this exists for. In E7 the pooled lift is
    almost zero, because five of six subsets match the reals; the FLUX.1 subset on its
    own is +49.9 points. Each stratum is therefore scored against the whole real class,
    which is how that subset will actually be evaluated.

    Args:
        rows: Manifest rows.
        parameter: The column to test.
        stratum_key: The column naming the subset, e.g. ``"source_vae"``.
        label_key: The column holding the label.
        positive_label: The label whose rows are split into strata; rows of any other
            label are shared across every stratum.

    Returns:
        ``{stratum: separability_result}``, plus ``"separable_strata"`` listing the
        offenders in descending order of lift.
    """
    shared = [row for row in rows if str(row.get(label_key, "")) != positive_label]
    strata: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get(label_key, "")) == positive_label:
            strata.setdefault(str(row.get(stratum_key, "")), []).append(row)

    results = {
        name: parameter_separability([*subset, *shared], parameter, label_key)
        for name, subset in sorted(strata.items())
    }
    offenders = sorted(
        (name for name, result in results.items() if result["separable"]),
        key=lambda name: results[name]["lift"],
        reverse=True,
    )
    return {"by_stratum": results, "separable_strata": offenders}
