"""Synchronized augmentation: both members of a pair get the identical treatment.

A content-aligned pair is a real and its own reconstruction. The only difference
between them is what the decoder did, and that is exactly what S2 is meant to learn.
Augmenting the two members independently puts a second difference back in -- one
member compressed harder, one blurred, one greyscale -- and a detector will take the
cheaper signal every time. "Fake or JPEG?" [R12] is the standing demonstration that it
will.

So this module separates **sampling the parameters** from **applying them**:
:func:`sample_params` draws one :class:`AugmentParams` per pair, and :func:`apply` is a
pure function of an image and those parameters. Synchronisation is then structural
rather than a discipline someone has to remember: there is no code path that can
augment one member and not the other, because there is only one parameter object.

The same holds for the noise field, not just the noise sigma. The parameters carry a
``noise_seed`` and both members receive the **same** additive field, so their
difference is unchanged by it. Drawing two fields would leave a per-member noise
signature that is easier to read than any decoder trace.

Order of operations, and why:

1. random 320 crop out of the 384 pair, then horizontal flip -- geometry first;
2. down-then-up resize, then blur, then noise -- the degradations a picture picks up;
3. greyscale (D-041) -- after the noise, so the result really is R == G == B;
4. JPEG -- last, because it is the publishing step and it is what a detector would
   otherwise read instead of the decoder.

Step 4 draws its chroma subsampling from ``{4:4:4, 4:2:0}`` with equal probability, on
**both** classes. That is deliberate: E7's FLUX.1 subset is perfectly separable by
``4:2:0 -> fake`` (D-044), and a model that has seen 4:2:0 on reals during training
cannot use that rule at test time.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from dbp.data.jpeg import encode_jpeg_bytes, subsampling_code

#: Model input size (D-045). A multiple of 16, 32, 8 and 4, so the same crop feeds
#: DINOv3, DC-AE, the KL VAEs and the SR model without a resize anywhere.
CROP_SIZE = 320

#: Size of the stored pair images the crop is taken from.
PAIR_SIZE = 384

#: Probabilities and ranges, exactly as approved in the P2 plan (B2). `greyscale` is
#: D-041's 0.05-0.10 band at its midpoint.
DEFAULTS: dict[str, Any] = {
    "flip_p": 0.5,
    "jpeg_p": 0.5,
    "jpeg_quality_range": (60, 100),
    "jpeg_subsamplings": ("4:4:4", "4:2:0"),
    "blur_p": 0.2,
    "blur_sigma_range": (0.1, 1.5),
    "noise_p": 0.1,
    "noise_sigma_range": (1.0, 5.0),
    "resize_p": 0.2,
    "resize_scale_range": (0.75, 1.0),
    "greyscale_p": 0.075,
}


class AugmentError(ValueError):
    """Raised when an augmentation is asked for something it cannot do."""


@dataclass(frozen=True)
class AugmentParams:
    """One draw, shared by every member of a pair.

    Attributes:
        crop_top: Top edge of the crop box, in pixels.
        crop_left: Left edge of the crop box, in pixels.
        crop_size: Side of the square crop.
        flip: Whether to mirror horizontally.
        resize_scale: Down-then-up scale factor, or ``None`` for no resize.
        blur_sigma: Gaussian blur sigma, or ``None``.
        noise_sigma: Additive noise sigma in grey levels, or ``None``.
        noise_seed: Seed of the **shared** noise field. Meaningless when
            ``noise_sigma`` is ``None``.
        greyscale: Whether to convert to greyscale-in-RGB.
        jpeg_quality: JPEG quality, or ``None`` for no re-encode.
        jpeg_subsampling: Chroma subsampling label, or ``None``.
    """

    crop_top: int
    crop_left: int
    crop_size: int
    flip: bool
    resize_scale: float | None
    blur_sigma: float | None
    noise_sigma: float | None
    noise_seed: int
    greyscale: bool
    jpeg_quality: int | None
    jpeg_subsampling: str | None

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping, for the manifest row that records what a sample received."""
        return asdict(self)

    @property
    def operations(self) -> tuple[str, ...]:
        """Names of the operations this draw actually applies."""
        names = []
        if self.flip:
            names.append("flip")
        if self.resize_scale is not None:
            names.append("resize")
        if self.blur_sigma is not None:
            names.append("blur")
        if self.noise_sigma is not None:
            names.append("noise")
        if self.greyscale:
            names.append("greyscale")
        if self.jpeg_quality is not None:
            names.append("jpeg")
        return tuple(names)


def sample_params(
    rng: np.random.Generator,
    image_size: int = PAIR_SIZE,
    crop_size: int = CROP_SIZE,
    settings: dict[str, Any] | None = None,
) -> AugmentParams:
    """Draw one parameter set, to be applied to every member of a pair.

    Args:
        rng: The generator. Seed it per sample, not per member.
        image_size: Side of the input image.
        crop_size: Side of the crop to take.
        settings: Overrides for :data:`DEFAULTS`.

    Returns:
        The draw.

    Raises:
        AugmentError: If the crop does not fit inside the image.
    """
    if crop_size > image_size:
        raise AugmentError(
            f"crop {crop_size} does not fit in a {image_size}px image; the pairs are "
            f"{PAIR_SIZE}px and the model input is {CROP_SIZE}px"
        )
    config = {**DEFAULTS, **(settings or {})}
    slack = image_size - crop_size

    quality_low, quality_high = config["jpeg_quality_range"]
    blur_low, blur_high = config["blur_sigma_range"]
    noise_low, noise_high = config["noise_sigma_range"]
    scale_low, scale_high = config["resize_scale_range"]
    subsamplings: Sequence[str] = config["jpeg_subsamplings"]

    # Every draw is taken unconditionally, so the stream advances by the same amount
    # whatever the coin flips decide. Two images then differ only in their seed, never
    # in how many numbers were consumed before them.
    crop_top = int(rng.integers(0, slack + 1))
    crop_left = int(rng.integers(0, slack + 1))
    flip_u, resize_u, blur_u, noise_u, grey_u, jpeg_u = (float(x) for x in rng.random(6))
    resize_scale = float(rng.uniform(scale_low, scale_high))
    blur_sigma = float(rng.uniform(blur_low, blur_high))
    noise_sigma = float(rng.uniform(noise_low, noise_high))
    noise_seed = int(rng.integers(0, 2**31 - 1))
    quality = int(rng.integers(quality_low, quality_high + 1))
    subsampling = str(subsamplings[int(rng.integers(0, len(subsamplings)))])

    return AugmentParams(
        crop_top=crop_top,
        crop_left=crop_left,
        crop_size=int(crop_size),
        flip=flip_u < config["flip_p"],
        resize_scale=resize_scale if resize_u < config["resize_p"] else None,
        blur_sigma=blur_sigma if blur_u < config["blur_p"] else None,
        noise_sigma=noise_sigma if noise_u < config["noise_p"] else None,
        noise_seed=noise_seed,
        greyscale=grey_u < config["greyscale_p"],
        jpeg_quality=quality if jpeg_u < config["jpeg_p"] else None,
        jpeg_subsampling=subsampling if jpeg_u < config["jpeg_p"] else None,
    )


def apply(img_uint8: np.ndarray, params: AugmentParams) -> np.ndarray:
    """Apply one draw to one image.

    Pure: the same image and the same ``params`` always give the same bytes, on any
    machine, in any order. That is what lets the two members of a pair be augmented in
    separate calls and still be guaranteed to have received the same treatment.

    Args:
        img_uint8: ``(H, W, 3)`` ``uint8`` RGB.
        params: The shared draw.

    Returns:
        A ``(crop_size, crop_size, 3)`` ``uint8`` RGB array.

    Raises:
        AugmentError: If the input is not uint8 HWC RGB, or the crop falls outside it.
    """
    if img_uint8.dtype != np.uint8:
        raise AugmentError(f"expected uint8, got {img_uint8.dtype}")
    if img_uint8.ndim != 3 or img_uint8.shape[2] != 3:
        raise AugmentError(f"expected HWC RGB, got shape {img_uint8.shape}")

    size = params.crop_size
    bottom, right = params.crop_top + size, params.crop_left + size
    if bottom > img_uint8.shape[0] or right > img_uint8.shape[1]:
        raise AugmentError(
            f"crop box ({params.crop_top}, {params.crop_left}, {size}) falls outside a "
            f"{img_uint8.shape[0]}x{img_uint8.shape[1]} image"
        )

    out = img_uint8[params.crop_top : bottom, params.crop_left : right, :]
    if params.flip:
        out = out[:, ::-1, :]
    out = np.ascontiguousarray(out)

    if params.resize_scale is not None:
        small = max(1, int(round(size * params.resize_scale)))
        image = Image.fromarray(out, mode="RGB")
        image = image.resize((small, small), Image.BICUBIC)
        out = np.asarray(image.resize((size, size), Image.BICUBIC), dtype=np.uint8)

    if params.blur_sigma is not None:
        blurred = Image.fromarray(out, mode="RGB").filter(
            ImageFilter.GaussianBlur(radius=params.blur_sigma)
        )
        out = np.asarray(blurred, dtype=np.uint8)

    if params.noise_sigma is not None:
        field = np.random.default_rng(params.noise_seed).normal(
            0.0, params.noise_sigma, size=out.shape
        )
        out = np.clip(out.astype(np.float32) + field, 0.0, 255.0).round().astype(np.uint8)

    if params.greyscale:
        # ITU-R BT.601 luma, then written back to all three channels, so the tensor
        # shape and the JPEG path stay unchanged and the result satisfies the
        # `rgb_equal` test D-039 uses to detect greyscale.
        luma = (
            0.299 * out[:, :, 0].astype(np.float32)
            + 0.587 * out[:, :, 1].astype(np.float32)
            + 0.114 * out[:, :, 2].astype(np.float32)
        )
        grey = np.clip(luma, 0.0, 255.0).round().astype(np.uint8)
        out = np.repeat(grey[:, :, None], 3, axis=2)

    if params.jpeg_quality is not None:
        if params.jpeg_subsampling is None:
            raise AugmentError("jpeg_quality set without jpeg_subsampling")
        payload = encode_jpeg_bytes(
            out, params.jpeg_quality, subsampling_code(params.jpeg_subsampling)
        )
        with Image.open(io.BytesIO(payload)) as decoded:
            out = np.asarray(decoded.convert("RGB"), dtype=np.uint8)

    return out


def augment_pair(
    members: Sequence[np.ndarray],
    rng: np.random.Generator,
    image_size: int = PAIR_SIZE,
    crop_size: int = CROP_SIZE,
    settings: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], AugmentParams]:
    """Augment every member of one pair under a single shared draw.

    Args:
        members: The pair's images -- typically the real and one reconstruction.
        rng: Generator, seeded per sample.
        image_size: Side of the inputs.
        crop_size: Side of the crop.
        settings: Overrides for :data:`DEFAULTS`.

    Returns:
        ``(augmented_members, params)``.

    Raises:
        AugmentError: If fewer than two members are given. A "pair" of one would make
            the synchronisation vacuous and is always a bug at the call site.
    """
    if len(members) < 2:
        raise AugmentError(
            f"augment_pair needs at least two members, got {len(members)}; use "
            "sample_params + apply for an unpaired row"
        )
    params = sample_params(rng, image_size, crop_size, settings)
    return [apply(member, params) for member in members], params


def centre_crop(img_uint8: np.ndarray, crop_size: int = CROP_SIZE) -> np.ndarray:
    """The evaluation-time counterpart: a fixed centre crop and nothing else.

    Args:
        img_uint8: ``(H, W, 3)`` ``uint8`` RGB.
        crop_size: Side of the crop.

    Returns:
        The centre ``crop_size`` square.

    Raises:
        AugmentError: If the image is smaller than the crop.
    """
    height, width = img_uint8.shape[:2]
    if height < crop_size or width < crop_size:
        raise AugmentError(
            f"image {height}x{width} is smaller than the {crop_size}px centre crop"
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return np.ascontiguousarray(
        img_uint8[top : top + crop_size, left : left + crop_size, :]
    )
