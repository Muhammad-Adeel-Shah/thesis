"""DDA (arXiv 2505.14359v6) as frozen in spec sections 1, 3 and 7.12.

**Architecture.** Upstream builds DINOv2 ViT-L/14 with ``torch.hub.load(...,
pretrained=True)``, which runs code downloaded from an unpinned branch. Here it is
``timm``'s ``vit_large_patch14_dinov2`` (``pretrained=False``, ``timm`` pinned in
``requirements/``), with three things matched to DINOv2's hub code
(``facebookresearch/dinov2`` @ ``7764ea0f``, read 2026-11-01):

* the positional embedding (1 + 37 x 37 tokens) is interpolated as DINOv2 does it --
  bicubic, in fp32, ``scale_factor = ((h0 + 0.1) / 37, (w0 + 0.1) / 37)``, no antialias --
  not with ``timm``'s own resampling;
* an unused ``mask_token`` (1 x 1024) is registered, so the key sets can match;
* the score is ``fc`` on the final-norm class token (DINOv2's ``x_norm_clstoken``).

**LoRA**, exactly as DDA's ``Inference/models/lora.py`` defines it: each ``nn.Linear``
whose name contains ``attn.qkv``, ``attn.proj``, ``mlp.fc1`` or ``mlp.fc2`` is wrapped,
and its output is ``original(x) + (x A^T) B^T * alpha / rank`` (rank 8, alpha 1). Keys
are ``<module>.original_layer.{weight,bias}`` and ``<module>.lora.{lora_A,lora_B}``.

**Loading:** sha256 checked, ``weights_only=True``, ``checkpoint['model']``, strict. Any
missing or unexpected key stops the run.

**Preprocessing** (``Inference/inference.py``): ``CenterCrop(336)`` on the PIL image --
which zero-pads a side under 336 -- then ``ToTensor`` and the CLIP mean/std. The
sensitivity row uses ``CenterCrop(308)``. Score: the logit (upstream's sigmoid is
monotone).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dbp.baselines.common import BaselineError, dryrun, load_pth, require_exact_keys

ARCH = "vit_large_patch14_dinov2"
PATCH = 14
#: DINOv2 hub defaults for ``dinov2_vitl14`` (``dinov2/hub/backbones.py``).
IMG_SIZE = 518
INTERPOLATE_OFFSET = 0.1
LORA_RANK, LORA_ALPHA = 8, 1.0
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
NATIVE_CROP, SENSITIVITY_CROP = 336, 308


class LoRALayer(nn.Module):
    """DDA's ``LoRALayer``: ``(x A^T) B^T * alpha / rank``."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.alpha, self.rank = alpha, rank
        self.lora_A = nn.Parameter(torch.zeros((rank, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, rank)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The low-rank update, scaled by ``alpha / rank``."""
        out = torch.einsum("...d, rd -> ...r", x, self.lora_A)
        out = torch.einsum("...r, or -> ...o", out, self.lora_B)
        return out * (self.alpha / self.rank)


class LoRALinear(nn.Module):
    """DDA's ``LoRALinear``: the frozen layer plus its LoRA update."""

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features,
                              rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``original_layer(x) + lora(x)``."""
        return self.original_layer(x) + self.lora(x)


def apply_lora(model: nn.Module, rank: int = LORA_RANK, alpha: float = LORA_ALPHA,
               targets: tuple[str, ...] = LORA_TARGETS) -> int:
    """Wrap every ``nn.Linear`` whose name contains a target, as DDA's
    ``apply_lora_to_linear_layers`` does. Returns how many were wrapped."""
    wrapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not any(t in name for t in targets):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, rank, alpha))
        wrapped += 1
    return wrapped


def interpolate_pos_encoding(pos_embed: torch.Tensor, height: int, width: int,
                             patch: int = PATCH, offset: float = INTERPOLATE_OFFSET
                             ) -> torch.Tensor:
    """DINOv2's ``interpolate_pos_encoding`` for one class token and no registers.

    ``pos_embed`` is ``(1, 1 + M * M, D)``; the result covers ``(height // patch) x
    (width // patch)`` patches, in fp32 and cast back to the input dtype.
    """
    n = pos_embed.shape[1] - 1
    h0, w0 = height // patch, width // patch
    if h0 * w0 == n and height == width:
        return pos_embed
    m = int(math.sqrt(n))
    if m * m != n:
        raise BaselineError(f"positional embedding of {n} patches is not square")
    previous = pos_embed.dtype
    pos = pos_embed.float()
    dim = pos.shape[-1]
    grid = F.interpolate(pos[:, 1:].reshape(1, m, m, dim).permute(0, 3, 1, 2),
                         mode="bicubic", antialias=False,
                         scale_factor=((h0 + offset) / m, (w0 + offset) / m))
    if tuple(grid.shape[-2:]) != (h0, w0):
        raise BaselineError(f"interpolated grid {tuple(grid.shape[-2:])} != {(h0, w0)}")
    grid = grid.permute(0, 2, 3, 1).reshape(1, -1, dim)
    return torch.cat((pos[:, :1], grid), dim=1).to(previous)


class _Base(nn.Module):
    """DDA's ``DINOv2Model``: ``model`` (the backbone) and ``fc`` (the score head)."""

    def __init__(self, arch: dict[str, Any]) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(ARCH, pretrained=False, num_classes=0,
                                       img_size=IMG_SIZE, dynamic_img_size=True, **arch)
        dim = self.model.embed_dim
        # DINOv2 carries an unused mask token; registered so the key sets can match.
        self.model.register_parameter("mask_token", nn.Parameter(torch.zeros(1, dim)))
        self.fc = nn.Linear(dim, 1)

    def cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """DINOv2's ``x_norm_clstoken`` for ``x`` of shape ``(B, 3, H, W)``."""
        vit = self.model
        batch, _, height, width = x.shape
        if height % PATCH or width % PATCH:
            raise BaselineError(f"{height}x{width} is not a multiple of the {PATCH}-px patch")
        tokens = vit.patch_embed(x)
        tokens = tokens.reshape(batch, -1, tokens.shape[-1])
        tokens = torch.cat((vit.cls_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = tokens + interpolate_pos_encoding(vit.pos_embed, height, width)
        for block in vit.blocks:
            tokens = block(tokens)
        return vit.norm(tokens)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The logit, ``(B, 1)``."""
        return self.fc(self.cls_token(x))


class DDADetector(nn.Module):
    """``DINOv2ModelWithLoRA``: keys ``base_model.model.*`` and ``base_model.fc.*``."""

    def __init__(self, arch: dict[str, Any] | None = None) -> None:
        super().__init__()
        if arch and not dryrun():
            raise BaselineError("a non-default DDA architecture is a dry-run seam only "
                                "(DBP_DRYRUN=1)")
        self.base_model = _Base(dict(arch or {}))
        wrapped = apply_lora(self.base_model.model)
        expected = 4 * len(self.base_model.model.blocks)
        if wrapped != expected:
            raise BaselineError(f"LoRA wrapped {wrapped} linears, expected {expected}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The logit, ``(B, 1)``."""
        return self.base_model(x)


def load_dda(path: Path, sha256: str, device: str, arch: dict[str, Any] | None = None
             ) -> DDADetector:
    """Build the detector and load ``DDA_ckpt.pth`` strictly (spec 7.12)."""
    checkpoint = load_pth(path, sha256, "DDA_ckpt.pth")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise BaselineError("DDA_ckpt.pth has no 'model' entry")
    state = checkpoint["model"]
    model = DDADetector(arch)
    require_exact_keys(model.state_dict().keys(), state.keys(), "DDA_ckpt.pth['model']")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def dda_input(img: np.ndarray, crop: int) -> tuple[torch.Tensor, bool]:
    """DDA's ``test_transform`` on an ``(H, W, 3)`` uint8 image.

    Returns the ``(3, crop, crop)`` tensor and whether ``CenterCrop`` zero-padded it.
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([transforms.CenterCrop(crop), transforms.ToTensor(),
                                    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)])
    padded = img.shape[0] < crop or img.shape[1] < crop
    return transform(Image.fromarray(img, "RGB")), padded
