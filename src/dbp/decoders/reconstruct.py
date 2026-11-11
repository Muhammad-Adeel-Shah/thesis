"""The DBP reconstruction protocol -- the single source of truth.

Every reconstruction anywhere in this project goes through this module. Do not
reimplement any part of it elsewhere; a second, slightly different pipeline is the
most likely way for a silent forensic artefact to enter the results.

The protocol (thesis section 6.4):

1. Input is ``uint8`` HWC RGB, loaded with Pillow. Never OpenCV, never BGR.
2. **No resizing, ever.** Centre-crop to a multiple of the member's factor
   (8 for ``vae_kl``, 32 for ``vae_dc``, 4 for ``sr_gan``). Resampling destroys the
   high-frequency generator traces and introduces a train/test bias [R10][R17].
3. Normalise ``x = uint8 / 127.5 - 1`` in float32, NCHW. ``sr_gan`` is the one
   exception: spandrel models expect ``[0, 1]``.
4. Encode/decode:
   - ``vae_kl``: ``z = vae.encode(x).latent_dist.mode()`` then ``vae.decode(z).sample``.
     ``mode()`` and not ``sample()``, so the result is deterministic. ``scaling_factor``
     and ``shift_factor`` are NOT applied: there is no denoising step here and they
     cancel across encode -> decode.
   - ``vae_kl_flux2``: identical to ``vae_kl``. ``AutoencoderKLFlux2`` exposes the same
     ``latent_dist`` / ``sample`` contract; the ``patch_size`` packing and the batch-norm
     in its config are applied by the FLUX.2 pipeline outside ``encode``/``decode``.
   - ``vae_kl_5d``: the same KL contract, but the model's encoder is causal-3D and
     unpacks ``(B, C, T, H, W)``. A single frame is fed as ``T = 1`` and the temporal
     axis is squeezed back out. No frame is duplicated or interpolated.
   - ``vae_dc``: ``z = vae.encode(x).latent`` then ``vae.decode(z).sample``.
   - ``vq``: ``h = model.encode(x).latents`` then ``model.decode(h).sample``. ``decode``
     re-quantises to the codebook by default (``force_not_quantize=False``), which is
     the whole point of probing a de-tokenizer: the residual then carries the
     quantisation lattice. It is deterministic because the codebook lookup is an argmin.
   - ``sr_gan``: bicubic down x4 with PIL, then SR x4 back to the original size.
5. **Never** ``enable_tiling()``: tile seams are artificial forensic traces.
   ``enable_slicing()`` is permitted.
6. Output ``clamp(-1, 1)`` then ``round((x + 1) * 127.5)`` -> ``uint8``, which quantises
   exactly the way a real generator output is. PNG only. Residual is ``int16``.

Attribute names in step 4 were verified against the installed diffusers at P0:
``AutoencoderKLOutput.latent_dist``, ``EncoderOutput.latent``, ``DecoderOutput.sample``.
:func:`verify_diffusers_api` re-checks them at run time so a future diffusers release
cannot silently change the meaning of a reconstruction.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from dbp.utils.io import center_crop_to_multiple

Precision = Literal["fp32", "fp16"]

#: Minimum PSNR in dB between the fp32 and fp16 reconstructions for fp16 to be allowed
#: by the P0 smoke test (decision D-003).
FP16_PSNR_THRESHOLD_DB = 45.0

#: Stricter threshold for running a member in fp16 while *building data* (P1 section
#: 6.2). The smoke test measures one image per crop; a corpus of thousands is a much
#: larger sample of the numerics, and a pair built in fp16 is permanent. A member
#: qualifies only if it clears this at EVERY crop size measured, not on average.
FP16_BUILD_PSNR_THRESHOLD_DB = 55.0


def verify_diffusers_api() -> dict[str, bool]:
    """Check that the output attribute names the protocol depends on still exist.

    Returns:
        A mapping of attribute path to whether it was found. Callers should record
        this in their report and refuse to trust a reconstruction if anything is False.
    """
    import diffusers
    from diffusers.models.autoencoders.autoencoder_dc import EncoderOutput
    from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
    from diffusers.models.autoencoders.vq_model import VQEncoderOutput
    from diffusers.models.modeling_outputs import AutoencoderKLOutput

    kl_fields = AutoencoderKLOutput.__dataclass_fields__
    return {
        "AutoencoderKLOutput.latent_dist": "latent_dist" in kl_fields,
        "DiagonalGaussianDistribution.mode": hasattr(DiagonalGaussianDistribution, "mode"),
        "DecoderOutput.sample": "sample" in DecoderOutput.__dataclass_fields__,
        "EncoderOutput.latent": "latent" in EncoderOutput.__dataclass_fields__,
        "VQEncoderOutput.latents": "latents" in VQEncoderOutput.__dataclass_fields__,
        "diffusers.AutoencoderKLFlux2": hasattr(diffusers, "AutoencoderKLFlux2"),
        "diffusers.AutoencoderKLQwenImage": hasattr(diffusers, "AutoencoderKLQwenImage"),
        "diffusers.VQModel": hasattr(diffusers, "VQModel"),
    }


def fp16_build_allowed(psnr_db_by_crop: dict[int, float | None]) -> tuple[bool, str]:
    """Decide whether a member may be run in fp16 while building P1 data.

    Args:
        psnr_db_by_crop: fp32-vs-fp16 PSNR in dB, keyed by crop size. A ``None`` value
            means the fp16 run failed or was not measured at that crop.

    Returns:
        ``(allowed, reason)``. ``allowed`` is True only when every crop was measured
        and every measurement is at or above :data:`FP16_BUILD_PSNR_THRESHOLD_DB`.
    """
    if not psnr_db_by_crop:
        raise ValueError("fp16_build_allowed needs at least one measured crop")
    missing = sorted(crop for crop, value in psnr_db_by_crop.items() if value is None)
    if missing:
        return False, f"no fp16 parity measurement at crop(s) {missing}"
    failing = {
        crop: value
        for crop, value in psnr_db_by_crop.items()
        if float(value) < FP16_BUILD_PSNR_THRESHOLD_DB
    }
    if failing:
        detail = ", ".join(f"{crop}: {value:.2f} dB" for crop, value in sorted(failing.items()))
        return False, f"below {FP16_BUILD_PSNR_THRESHOLD_DB} dB at {detail}"
    worst = min(float(v) for v in psnr_db_by_crop.values())
    return True, f"min parity {worst:.2f} dB >= {FP16_BUILD_PSNR_THRESHOLD_DB} dB"


def torch_dtype(precision: Precision) -> torch.dtype:
    """Map the protocol's precision label to a torch dtype."""
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"unsupported precision {precision!r}; use 'fp32' or 'fp16'")


def to_tensor(
    img_uint8: np.ndarray,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Convert an ``uint8`` HWC array to a normalised NCHW tensor.

    Args:
        img_uint8: ``(H, W, 3)`` ``uint8`` array.
        device: Target device.
        dtype: Target dtype.
        value_range: ``(-1, 1)`` for the VAEs, ``(0, 1)`` for spandrel SR models.

    Returns:
        A ``(1, 3, H, W)`` tensor on ``device``.
    """
    if img_uint8.dtype != np.uint8:
        raise TypeError(f"to_tensor expects uint8, got {img_uint8.dtype}")
    if img_uint8.ndim != 3 or img_uint8.shape[2] != 3:
        raise ValueError(f"to_tensor expects HWC RGB, got shape {img_uint8.shape}")

    # np.array(..., copy=True) rather than ascontiguousarray: skimage samples are
    # read-only, and torch.from_numpy on a read-only buffer warns about undefined
    # behaviour on write.
    arr = torch.from_numpy(np.array(img_uint8, dtype=np.uint8, copy=True))
    arr = arr.permute(2, 0, 1).unsqueeze(0)
    arr = arr.to(device=device, dtype=torch.float32)
    if value_range == (-1.0, 1.0):
        arr = arr / 127.5 - 1.0
    elif value_range == (0.0, 1.0):
        arr = arr / 255.0
    else:
        raise ValueError(f"unsupported value_range {value_range!r}")
    return arr.to(dtype=dtype)


def to_uint8(x: torch.Tensor, value_range: tuple[float, float] = (-1.0, 1.0)) -> np.ndarray:
    """Quantise a NCHW tensor back to an ``uint8`` HWC array.

    Clamps first, then rounds, matching how a real generator writes its output.

    Args:
        x: ``(1, 3, H, W)`` tensor.
        value_range: The range ``x`` lives in.

    Returns:
        ``(H, W, 3)`` ``uint8`` array.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"to_uint8 expects a (1, 3, H, W) tensor, got {tuple(x.shape)}")
    x = x.detach().to(torch.float32).cpu()
    low, high = value_range
    x = x.clamp(low, high)
    x = (x + 1.0) * 127.5 if value_range == (-1.0, 1.0) else x * 255.0
    arr = x.round().clamp(0, 255).squeeze(0).permute(1, 2, 0).numpy()
    return arr.astype(np.uint8)


def prepare_input(img_uint8: np.ndarray, crop_multiple: int) -> np.ndarray:
    """Centre-crop to the member's multiple. This is the only spatial operation allowed."""
    return center_crop_to_multiple(img_uint8, crop_multiple)


@torch.no_grad()
def reconstruct_vae_kl(vae: Any, x: torch.Tensor) -> torch.Tensor:
    """Encode/decode through a KL autoencoder deterministically.

    ``latent_dist.mode()`` is used rather than ``.sample()`` so that repeated runs on
    the same input produce the same latent. ``scaling_factor``/``shift_factor`` are not
    applied: they cancel between encode and decode when there is no denoising step.
    """
    posterior = vae.encode(x).latent_dist
    z = posterior.mode()
    return vae.decode(z).sample


@torch.no_grad()
def reconstruct_vae_kl_5d(vae: Any, x: torch.Tensor) -> torch.Tensor:
    """Encode/decode a single frame through a KL autoencoder with a temporal axis.

    ``AutoencoderKLQwenImage`` unpacks ``(B, C, T, H, W)`` in ``_encode``, so a 4D
    image tensor has to be lifted to 5D with ``T = 1`` and squeezed back afterwards.
    Nothing is duplicated, padded or interpolated along the temporal axis: one frame
    in, one frame out.

    Args:
        vae: A 5D-input KL autoencoder.
        x: ``(1, 3, H, W)`` tensor.

    Returns:
        ``(1, 3, H, W)`` reconstruction.
    """
    if x.ndim != 4:
        raise ValueError(f"reconstruct_vae_kl_5d expects a 4D NCHW tensor, got {tuple(x.shape)}")
    posterior = vae.encode(x.unsqueeze(2)).latent_dist
    z = posterior.mode()
    y = vae.decode(z).sample
    if y.ndim != 5 or y.shape[2] != 1:
        raise RuntimeError(
            f"expected a (B, C, 1, H, W) decode from a 5D KL autoencoder, got {tuple(y.shape)}"
        )
    return y.squeeze(2)


@torch.no_grad()
def reconstruct_vq(model: Any, x: torch.Tensor) -> torch.Tensor:
    """Encode/decode through a vector-quantised de-tokenizer.

    ``VQModel.encode`` returns plain ``latents`` (no distribution), and ``decode``
    re-quantises them against the codebook because ``force_not_quantize`` defaults to
    False. That quantisation is deliberately kept: it is the forensic signature of the
    de-tokenizer family, and skipping it would turn the member into an ordinary
    continuous autoencoder. The codebook lookup is an argmin, so the path is
    deterministic without any sampling decision.
    """
    h = model.encode(x).latents
    return model.decode(h).sample


@torch.no_grad()
def reconstruct_vae_dc(vae: Any, x: torch.Tensor) -> torch.Tensor:
    """Encode/decode through a deep-compression autoencoder (DC-AE).

    DC-AE is deterministic by construction: ``encode`` returns a plain latent, not a
    distribution.
    """
    z = vae.encode(x).latent
    return vae.decode(z).sample


@torch.no_grad()
def reconstruct_sr_gan(model: Any, img_uint8: np.ndarray, scale: int = 4) -> np.ndarray:
    """Downscale with PIL bicubic, then restore with a GAN super-resolver.

    The downscale is done in PIL on the ``uint8`` image, exactly as an image would be
    prepared for a real upsampling pipeline, so the GAN sees a realistic low-resolution
    input. The output is the same size as the input.

    Args:
        model: A spandrel ``ImageModelDescriptor``, callable on ``(1, 3, H, W)`` in [0, 1].
        img_uint8: ``(H, W, 3)`` ``uint8`` array whose sides are multiples of ``scale``.
        scale: The model's upscaling factor.

    Returns:
        ``(H, W, 3)`` ``uint8`` reconstruction.
    """
    height, width = img_uint8.shape[:2]
    if height % scale or width % scale:
        raise ValueError(f"{height}x{width} is not a multiple of the SR scale {scale}")

    lr = Image.fromarray(img_uint8, mode="RGB").resize(
        (width // scale, height // scale), resample=Image.Resampling.BICUBIC
    )
    device = next(model.model.parameters()).device
    dtype = next(model.model.parameters()).dtype
    lr_t = to_tensor(np.asarray(lr, dtype=np.uint8), device, dtype, value_range=(0.0, 1.0))
    sr_t = model(lr_t)
    out = to_uint8(sr_t, value_range=(0.0, 1.0))

    if out.shape[:2] != (height, width):
        raise RuntimeError(
            f"SR output {out.shape[:2]} does not match input {(height, width)}; "
            f"the protocol requires the reconstruction to keep the input size."
        )
    return out


def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB between two ``uint8`` images. Returns ``inf`` when identical."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(255.0**2 / mse)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> int:
    """Largest absolute per-pixel difference between two ``uint8`` images."""
    return int(np.max(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def fp16_verdict(recon_fp32: np.ndarray, recon_fp16: np.ndarray | None) -> dict[str, Any]:
    """Decide whether fp16 is safe for a member (decision D-003).

    fp16 is allowed only when the fp16 reconstruction matches the fp32 one at
    >= 45 dB PSNR and contains no NaN or Inf. The original SDXL VAE is expected to
    fail this; the outcome is recorded for every member either way.
    """
    if recon_fp16 is None:
        return {"fp16_allowed": False, "fp16_psnr_db": None, "fp16_reason": "fp16 run failed"}
    has_nonfinite = not np.isfinite(recon_fp16.astype(np.float32)).all()
    psnr = psnr_uint8(recon_fp32, recon_fp16)
    allowed = (not has_nonfinite) and psnr >= FP16_PSNR_THRESHOLD_DB
    reason = None
    if has_nonfinite:
        reason = "NaN/Inf in fp16 reconstruction"
    elif psnr < FP16_PSNR_THRESHOLD_DB:
        reason = f"PSNR {psnr:.2f} dB < {FP16_PSNR_THRESHOLD_DB} dB threshold"
    return {
        "fp16_allowed": bool(allowed),
        "fp16_psnr_db": None if math.isinf(psnr) else round(psnr, 3),
        "fp16_reason": reason,
    }
