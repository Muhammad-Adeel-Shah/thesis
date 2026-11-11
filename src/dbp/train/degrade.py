"""Real-side degradation chains, applied identically to both members of a pair (P3 §4b).

``P3_PLAN.md`` section 4b, with the values fixed on 2026-10-17 before any candidate
trained (log P3-C of that date):

* **Chain probability p** -- the only tuned value. Candidates 0.7 (the plan default:
  "none | p = 0.3") and 0.35. With probability ``1 - p`` the pair is left clean.
* **One of four operations, 1/4 each**, when a chain applies:

  ========  =========================================================================
  jpeg      quality uniform 50-95 (integer), chroma 4:2:0 or 4:4:4 at 50/50
  webp      quality uniform 50-95 (integer)
  resize    scale uniform 0.5-1.0: down (bicubic), then back up to the original size
  social    resize-and-restore (scale 0.5-1.0), then JPEG quality uniform 70-85 at
            **4:2:0** (the platforms subsample chroma); with probability 0.5 (fixed, not
            tuned) it is a **re-share**: the same two steps applied twice in a row, the
            second round with independently drawn scale and quality
  ========  =========================================================================

  User decisions of 2026-10-19, before any candidate trained: the social-media JPEG is
  4:2:0 only (resolving the chroma ambiguity of the first build); the re-share variant.

**Synchronised, by construction.** As in :mod:`dbp.train.augment`, sampling and
applying are separate: :func:`sample_chain` draws one :class:`DegradeChain` per pair and
:func:`apply_chain` is a pure function of an image and that chain. :func:`degrade_pair`
hands the *same* chain object to every member, so a real and its reconstruction always
receive byte-identical operations and parameters and the degradation can never become a
label cue.

**Stable random stream.** Every :func:`sample_chain` call consumes exactly the same ten
draws whichever branch it takes, so the stream -- and a resumed run -- does not depend
on which operations happened to be chosen.

The chain runs on the full 384 px pair, before the crop and the existing synchronised
augmentation: a photograph is degraded whole and then cropped. Resize-and-restore
restores the original size, so the 320 crop always fits (the reconstruction protocol's
"never resize" governs decoder inputs, not augmentation).
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, features

from dbp.data.jpeg import encode_jpeg_bytes, subsampling_code

OPERATIONS = ("jpeg", "webp", "resize", "social")
JPEG_QUALITY = (50, 95)
WEBP_QUALITY = (50, 95)
RESIZE_SCALE = (0.5, 1.0)
SOCIAL_QUALITY = (70, 85)
SUBSAMPLINGS = ("4:2:0", "4:4:4")
#: The social-media operation's JPEG chroma: the platforms subsample (2026-10-19).
SOCIAL_SUBSAMPLING = "4:2:0"
#: Probability that a social-media operation is a re-share (two rounds); fixed, not tuned.
RESHARE_P = 0.5
#: The two candidates of the tuning rule (fixed 2026-10-17).
CANDIDATE_P = (0.7, 0.35)
#: WebP encoder effort; fixed so a chain's bytes do not depend on a library default.
WEBP_METHOD = 4


class DegradeError(ValueError):
    """A chain or an image the degradation cannot be applied to."""


@dataclass(frozen=True)
class DegradeChain:
    """One draw: ``op`` is None for a clean pair; unused parameters are None.

    ``quality2`` / ``scale2`` are set only for a social-media re-share: the second round.
    """

    op: str | None
    quality: int | None = None
    subsampling: str | None = None
    scale: float | None = None
    quality2: int | None = None
    scale2: float | None = None

    def to_json(self) -> str:
        """Canonical JSON, for logs and for comparing two chains byte for byte."""
        return json.dumps(asdict(self), sort_keys=True)


def sample_chain(rng: np.random.Generator, p: float) -> DegradeChain:
    """Draw one chain. Always consumes the same ten draws from ``rng``."""
    if not 0.0 <= p <= 1.0:
        raise DegradeError(f"chain probability must be in [0, 1], got {p}")
    u_apply = float(rng.random())
    op_index = int(rng.integers(0, len(OPERATIONS)))
    q_jpeg = int(rng.integers(JPEG_QUALITY[0], JPEG_QUALITY[1] + 1))
    q_webp = int(rng.integers(WEBP_QUALITY[0], WEBP_QUALITY[1] + 1))
    q_social = int(rng.integers(SOCIAL_QUALITY[0], SOCIAL_QUALITY[1] + 1))
    chroma = SUBSAMPLINGS[int(rng.integers(0, 2))]
    scale = float(rng.uniform(RESIZE_SCALE[0], RESIZE_SCALE[1]))
    u_reshare = float(rng.random())
    q_social2 = int(rng.integers(SOCIAL_QUALITY[0], SOCIAL_QUALITY[1] + 1))
    scale2 = float(rng.uniform(RESIZE_SCALE[0], RESIZE_SCALE[1]))
    if u_apply >= p:
        return DegradeChain(op=None)
    op = OPERATIONS[op_index]
    if op == "jpeg":
        return DegradeChain(op=op, quality=q_jpeg, subsampling=chroma)
    if op == "webp":
        return DegradeChain(op=op, quality=q_webp)
    if op == "resize":
        return DegradeChain(op=op, scale=scale)
    if u_reshare < RESHARE_P:
        return DegradeChain(op=op, quality=q_social, subsampling=SOCIAL_SUBSAMPLING,
                            scale=scale, quality2=q_social2, scale2=scale2)
    return DegradeChain(op=op, quality=q_social, subsampling=SOCIAL_SUBSAMPLING, scale=scale)


def _resize_restore(img: np.ndarray, scale: float) -> np.ndarray:
    height, width = img.shape[:2]
    small = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray(img, mode="RGB").resize(small, Image.BICUBIC)
    return np.asarray(image.resize((width, height), Image.BICUBIC), dtype=np.uint8)


def _jpeg(img: np.ndarray, quality: int, subsampling: str) -> np.ndarray:
    payload = encode_jpeg_bytes(img, quality, subsampling_code(subsampling))
    with Image.open(io.BytesIO(payload)) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def _webp(img: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(buffer, format="WEBP", quality=quality,
                                          method=WEBP_METHOD, lossless=False)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def apply_chain(img_uint8: np.ndarray, chain: DegradeChain) -> np.ndarray:
    """Apply one chain to one image. Pure; the output has the input's shape.

    Raises:
        DegradeError: On a non-uint8 / non-RGB input or a malformed chain.
    """
    if img_uint8.dtype != np.uint8 or img_uint8.ndim != 3 or img_uint8.shape[2] != 3:
        raise DegradeError(f"expected uint8 HWC RGB, got {img_uint8.dtype} {img_uint8.shape}")
    img = np.ascontiguousarray(img_uint8)
    if chain.op is None:
        return img
    if chain.op == "jpeg" and chain.quality and chain.subsampling:
        return _jpeg(img, chain.quality, chain.subsampling)
    if chain.op == "webp" and chain.quality:
        return _webp(img, chain.quality)
    if chain.op == "resize" and chain.scale:
        return _resize_restore(img, chain.scale)
    if chain.op == "social" and chain.quality and chain.subsampling and chain.scale:
        once = _jpeg(_resize_restore(img, chain.scale), chain.quality, chain.subsampling)
        if chain.quality2 is None and chain.scale2 is None:
            return once
        if chain.quality2 and chain.scale2:            # a re-share: the second round
            return _jpeg(_resize_restore(once, chain.scale2), chain.quality2, chain.subsampling)
    raise DegradeError(f"malformed chain {chain}")


def degrade_pair(members: Sequence[np.ndarray], rng: np.random.Generator,
                 p: float) -> tuple[list[np.ndarray], DegradeChain]:
    """Degrade every member of one pair under a single shared chain."""
    if len(members) < 2:
        raise DegradeError(f"degrade_pair needs at least two members, got {len(members)}")
    chain = sample_chain(rng, p)
    return [apply_chain(member, chain) for member in members], chain


def check_codecs() -> None:
    """Fail at start-up, not mid-run, if this Pillow cannot encode every operation.

    Raises:
        DegradeError: If WebP support is missing or an operation fails on a test image.
    """
    if not features.check("webp"):
        raise DegradeError("this Pillow build has no WebP support; the 'webp' operation "
                           "cannot run")
    probe = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for chain in (DegradeChain("jpeg", 50, "4:2:0"), DegradeChain("webp", 50),
                  DegradeChain("resize", scale=0.5), DegradeChain("social", 70, "4:2:0", 0.5),
                  DegradeChain("social", 70, "4:2:0", 0.5, quality2=80, scale2=0.7)):
        out = apply_chain(probe, chain)
        if out.shape != probe.shape:
            raise DegradeError(f"{chain.op} changed the shape: {out.shape}")
