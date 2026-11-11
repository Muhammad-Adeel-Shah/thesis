"""FIRE (arXiv 2412.07140v3), ImageNet-ADM checkpoint, as frozen in spec sections 1, 3, 7.13-7.14.

Re-implemented from ``Chuchad/FIRE`` @ ``ff1a316d`` (MIT; ``utils/network_utils.py``,
``dataset.py``, ``eval.py``); upstream code is not imported. Module names follow
``FIRE_model`` so the checkpoint loads by name: ``vae.*``, ``resnet.*`` and
``fft_filter_module.{i_mask, r_i_mask, mask_autoencoder.*}``.

* **Input** (``--resize``): ``Resize(256, antialias=True)`` on the PIL image's short
  side, aspect kept, ``ToTensor`` in [0, 1], no normalisation, one image per batch. The
  long side is then centre-cropped down to a multiple of 8 (at most 7 px; spec 7.14).
* **Forward:** an FFT mid-band filter whose mask an ESPCN predicts; the filtered image is
  divided by (max - min) over the batch. Then **two SD VAE round trips** -- of the image,
  then of the filtered copy -- through diffusers' ``retrieve_latents`` (a **sampled**
  latent) in **fp16** on the [0, 1] input. ``abs(recon(x) - x)`` and
  ``abs(recon(filtered) - x)``, concatenated, go to a 6-channel ResNet-50 with instance
  norm. Score: the logit.

**Stated exception to CLAUDE.md section 7** (user, 2026-11-01; spec 7.13): these round
trips run exactly as FIRE published them (fp16, ``latent_dist.sample()``, the filtered
copy), because its ResNet was trained on their residuals; ``reconstruct.py`` (fp32,
``.mode()``) would score a different detector.

**Seed rule:** each image's two draws come from one CPU ``torch.Generator`` seeded with
:func:`dbp.baselines.common.fire_seed` (the image's draw first, then the filtered
copy's), so every score is reproducible.

**VAE:** our pinned ``sd15`` file in fp16 (FIRE's ``runwayml`` id resolves to that
revision); if the checkpoint carries the full ``vae.*`` key set, those weights replace it.
Nothing else is downloaded: the ResNet is built with ``weights=None``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from dbp.baselines.common import BaselineError, fire_seed, load_pth, require_exact_keys
from dbp.utils.io import center_crop_to_multiple

SHORT_SIDE, MULTIPLE = 256, 8
RADIUS_LOW, RADIUS_HIGH = 40, 120  # FIRE_model() defaults, which eval.py uses
MASK_ROWS = MASK_COLS = 256


class ESPCN(nn.Module):
    """FIRE's mask network (``network_utils.ESPCN``), upscale factor 1."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, channels: int = 64,
                 upscale_factor: int = 1) -> None:
        super().__init__()
        hidden = channels // 2
        out = int(out_channels * upscale_factor ** 2)
        self.bn = nn.BatchNorm2d(in_channels)
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)), nn.Tanh(),
            nn.Conv2d(channels, hidden, (3, 3), (1, 1), (1, 1)), nn.Tanh())
        self.sub_pixel_0 = nn.Sequential(nn.Conv2d(hidden, out, (3, 3), (1, 1), (1, 1)),
                                         nn.PixelShuffle(upscale_factor), nn.Sigmoid())
        self.sub_pixel_1 = nn.Sequential(nn.Conv2d(hidden, out, (3, 3), (1, 1), (1, 1)),
                                         nn.PixelShuffle(upscale_factor), nn.Sigmoid())

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """The mid-band mask and its complement."""
        x = self.feature_maps(self.bn(x))
        return self.sub_pixel_0(x), self.sub_pixel_1(x)


def preset_masks(rows: int = MASK_ROWS, cols: int = MASK_COLS, low: int = RADIUS_LOW,
                 high: int = RADIUS_HIGH) -> tuple[Tensor, Tensor]:
    """FIRE's ``init_mask``: buffers kept only so the checkpoint keys match (unused)."""
    mask = torch.ones((1, rows, cols), dtype=torch.float32)
    x, y = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    radius2 = (x - rows // 2) ** 2 + (y - cols // 2) ** 2
    mask[:, radius2 < low * low] = 0
    mask[:, radius2 >= high * high] = 0
    return mask, 1 - mask


class FFTFilter(nn.Module):
    """FIRE's ``fft_filter``: an ESPCN-predicted mid-band filter in the FFT domain."""

    def __init__(self) -> None:
        super().__init__()
        i_mask, r_i_mask = preset_masks()
        self.register_buffer("i_mask", i_mask)
        self.register_buffer("r_i_mask", r_i_mask)
        self.mask_autoencoder = ESPCN()

    @staticmethod
    def _normalised(z: Tensor) -> Tensor:
        """``z / (max - min)`` over the whole batch, as upstream (no subtraction)."""
        return z / (torch.max(z) - torch.min(z))

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """``(middle_freq_image, middle_filtered_image)`` of a [0, 1] batch."""
        freq = torch.fft.fftshift(torch.fft.fftn(image * 255, dim=(-2, -1)), dim=(-2, -1))
        mask_mid, mask_filtered = self.mask_autoencoder(
            (20 * torch.log(torch.abs(freq) + 1e-7)) / 255)
        middle = torch.fft.ifftn(torch.fft.ifftshift(freq * mask_mid.to(freq.dtype),
                                                     dim=(-2, -1)), dim=(-2, -1))
        filtered = torch.fft.ifftn(torch.fft.ifftshift(freq * mask_filtered.to(freq.dtype),
                                                       dim=(-2, -1)), dim=(-2, -1))
        return self._normalised(torch.abs(middle)), self._normalised(torch.abs(filtered))


def fire_resnet() -> nn.Module:
    """``get_frq_resnet_model(mode="frq", norm_layer="instance")``, no download."""
    import torchvision

    model = torchvision.models.resnet50(weights=None, num_classes=1000,
                                        norm_layer=nn.InstanceNorm2d)
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(2048, 1)
    return model


class FIREDetector(nn.Module):
    """``FIRE_model``: ``vae`` (fp16), ``resnet`` and ``fft_filter_module``."""

    def __init__(self, vae: Any) -> None:
        super().__init__()
        self.vae = vae
        self.resnet = fire_resnet()
        self.fft_filter_module = FFTFilter()

    def forward(self, x: Tensor, generator: torch.Generator) -> Tensor:
        """The logit of a ``(1, 3, H, W)`` [0, 1] image; ``generator`` drives both samples."""
        from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
            retrieve_latents,
        )

        dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
        _, filtered = self.fft_filter_module(x)
        recon_x = self.vae.decode(retrieve_latents(self.vae.encode(x.to(dtype)),
                                                   generator=generator).to(dtype),
                                  return_dict=False)[0]
        recon_f = self.vae.decode(retrieve_latents(self.vae.encode(filtered.to(dtype)),
                                                   generator=generator).to(dtype),
                                  return_dict=False)[0]
        residuals = torch.cat([torch.abs(recon_x - x), torch.abs(recon_f - x)], dim=1)
        return self.resnet(residuals)


def vae_keys_held(state: dict[str, Tensor], vae_keys: list[str]) -> str:
    """``"full"``, ``"none"`` or ``"partial"``: how much of the VAE the checkpoint holds."""
    held = {k for k in state if k.startswith("vae.")}
    if not held:
        return "none"
    return "full" if held == {f"vae.{k}" for k in vae_keys} else "partial"


def load_fire(path: Path, sha256: str, vae: Any, device: str) -> tuple[FIREDetector, str]:
    """Build ``FIRE_model`` around the pinned fp16 VAE and load the checkpoint.

    Every ResNet and filter key must be present (spec section 1); the VAE comes from the
    checkpoint when it holds the full set, else stays the pinned one. A partial VAE set
    stops the run. Returns the model and where its VAE weights came from.
    """
    state = load_pth(path, sha256, "FIRE checkpoint")
    if not isinstance(state, dict):
        raise BaselineError(f"FIRE checkpoint is a {type(state).__name__}, not a state dict")
    model = FIREDetector(vae)
    vae_keys = list(vae.state_dict())
    held = vae_keys_held(state, vae_keys)
    if held == "partial":
        raise BaselineError("FIRE checkpoint holds part of the VAE; stop and ask")
    expected = [k for k in model.state_dict() if not k.startswith("vae.")]
    expected += [f"vae.{k}" for k in vae_keys] if held == "full" else []
    require_exact_keys(expected, state.keys(), "FIRE checkpoint")
    model.load_state_dict(state, strict=held == "full")
    return model.to(device).eval(), ("checkpoint" if held == "full" else "pinned sd15")


def fire_input(img: np.ndarray) -> tuple[Tensor, bool]:
    """FIRE's ``--resize`` transform, then the long side cropped to a multiple of 8.

    Returns the ``(3, H, W)`` tensor in [0, 1] and whether the multiple-of-8 crop cut it.
    """
    from PIL import Image
    from torchvision import transforms

    resized = transforms.Resize(SHORT_SIDE, antialias=True)(Image.fromarray(img, "RGB"))
    array = np.asarray(resized, dtype=np.uint8)
    cropped = np.array(center_crop_to_multiple(array, MULTIPLE), dtype=np.uint8, copy=True)
    return transforms.functional.to_tensor(cropped), cropped.shape != array.shape


def score_fire(model: FIREDetector, img: np.ndarray, image_id: str, device: str
               ) -> tuple[float, bool]:
    """One image's logit under the seed rule, and whether it was cropped."""
    x, cropped = fire_input(img)
    generator = torch.Generator(device="cpu").manual_seed(fire_seed(image_id))
    with torch.no_grad():
        logit = model(x.unsqueeze(0).to(device), generator)
    value = float(logit.float().reshape(-1)[0])
    if not math.isfinite(value):
        raise BaselineError(f"FIRE: non-finite logit for {image_id}")
    return value, cropped
