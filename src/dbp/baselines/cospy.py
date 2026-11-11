"""CO-SPY (arXiv 2503.18286v1), ``sd-v1_4`` weights, as frozen in spec sections 1 and 3.

Re-implemented from ``Megum1/CO-SPY`` @ ``4be0ba51`` (MIT; ``detectors/sd-v1_4/*.py``);
upstream code is not imported. Module names follow upstream so the released weight files
load by name.

* **Outer transform** (``fusion.py``): ``Resize(384)`` on the PIL image's short side, then
  ``CenterCrop(384)`` and ``ToTensor`` -- in [0, 1].
* **Semantic branch:** mean = std = 0.5, then SigLIP ``ViT-SO400M-14-SigLIP-384``
  ``encode_image`` and ``fc`` (1152 -> 1). SigLIP is built by ``open_clip`` from its own
  model config and loaded from the local, sha256-checked ``open_clip_model.safetensors``
  only (spec 7.9); nothing is fetched.
* **Artifact branch:** ``Resize(224, antialias=False)`` on the 384 tensor, identity
  normalisation, then the SD VAE round trip of the [0, 1] tensor (``latent_dist.mean``,
  no scaling factor), the residual ``(x - x_recon) / 7 * 100``, a ResNet stem and its first
  two layers, average pooling, and ``fc`` (512 -> 1).
* **Fusion:** ``fc`` (2 -> 1) on the two logits. Score: the fused logit.

**Stated exception to CLAUDE.md section 7** (user, 2026-11-01): the VAE round trip runs as
CO-SPY wrote it, in fp32, with no tiling, on our pinned ``sd15`` VAE (byte-identical to the
SD-v1.4 VAE CO-SPY loads). It is part of the baseline's learned forward pass, so replacing
it with ``reconstruct.py`` would score a different detector. ``reconstruct.py`` stays the
only path for every reconstruction of ours.

**Weights:** each ``.pth`` is sha256-checked and loaded with ``weights_only=True``.
Upstream loads the artifact file ``strict=False``; here its keys must cover the artifact
branch exactly.

**The VAE in the artifact file** (user decision 2026-11-05, replacing the spec's stop):
the released ``artifact_weights.pth`` holds the full VAE (248 keys; Kaggle 2026-11-05).
Its tensors are compared with the pinned ``sd15`` VAE's. Equal -- exactly, or after a
dtype cast -- the pinned VAE is used; different, the checkpoint's own VAE tensors are
loaded (CO-SPY as published), the max absolute difference of each tensor is recorded, and
the run continues. A partial VAE key set still stops it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from dbp.baselines.common import BaselineError, dryrun, load_pth, require_exact_keys

SIGLIP = "ViT-SO400M-14-SigLIP-384"
OUTER, INNER = 384, 224
SEM_MEAN = SEM_STD = (0.5, 0.5, 0.5)
VAE_PREFIX = "artifact_encoder.vae."


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding, no bias."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                     bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, no bias."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    """CO-SPY's ResNet bottleneck (``artifact.py``)."""

    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual bottleneck."""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class VAEReconEncoder(nn.Module):
    """CO-SPY's artifact encoder: VAE residual -> ResNet stem + layers 1 and 2."""

    def __init__(self, vae: Any) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.vae = vae

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * Bottleneck.expansion,
                                               stride),
                                       nn.BatchNorm2d(planes * Bottleneck.expansion))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers += [Bottleneck(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """512-d artifact feature of a [0, 1] batch."""
        with torch.no_grad():
            recon = self.vae.decode(self.vae.encode(x).latent_dist.mean).sample
        x = (x - recon) / 7.0 * 100.0
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer2(self.layer1(x))
        return self.avgpool(x).flatten(1)


class ArtifactDetector(nn.Module):
    """``artifact_encoder`` + ``fc``."""

    def __init__(self, vae: Any) -> None:
        super().__init__()
        self.artifact_encoder = VAEReconEncoder(vae)
        self.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Artifact logit."""
        return self.fc(self.artifact_encoder(x))


class SemanticDetector(nn.Module):
    """``clip`` (frozen SigLIP) + ``fc``."""

    def __init__(self, clip: nn.Module, dim: int) -> None:
        super().__init__()
        self.clip = clip
        self.fc = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Semantic logit."""
        return self.fc(self.clip.encode_image(x))


class CoSpyFusion(nn.Module):
    """``CoSpyFusionDetector``: per-branch transforms in ``forward``, fused by ``fc``."""

    def __init__(self, clip: nn.Module, clip_dim: int, vae: Any) -> None:
        super().__init__()
        from torchvision import transforms

        self.sem = SemanticDetector(clip, clip_dim)
        self.art = ArtifactDetector(vae)
        self.fc = nn.Linear(2, 1)
        self.sem_transform = transforms.Normalize(SEM_MEAN, SEM_STD)
        self.art_transform = transforms.Compose([
            transforms.Resize(INNER, antialias=False),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fused logit for a ``(B, 3, 384, 384)`` batch in [0, 1]."""
        pred_sem = self.sem(self.sem_transform(x))
        pred_art = self.art(self.art_transform(x))
        return self.fc(torch.cat([pred_sem, pred_art], dim=1))


def build_siglip(weights: Path, sha256: str, model_name: str = SIGLIP,
                 dryrun_config: Path | None = None) -> tuple[nn.Module, int]:
    """SigLIP from its open_clip config and the local, hash-checked weights only.

    Returns the model and its embedding width (1152 for SO400M).
    """
    import open_clip

    from dbp.baselines.common import check_sha256

    if model_name != SIGLIP or dryrun_config is not None:
        if not dryrun():
            raise BaselineError(f"SigLIP must be {SIGLIP}; another model is a dry-run seam")
        if dryrun_config is not None:
            open_clip.add_model_config(dryrun_config)
    check_sha256(weights, sha256, "SigLIP open_clip_model.safetensors")
    config = open_clip.get_model_config(model_name)
    if config is None:
        raise BaselineError(f"open_clip has no model config {model_name!r}")
    model = open_clip.create_model(model_name, pretrained=str(weights), precision="fp32",
                                   device="cpu")
    return model.eval(), int(config["embed_dim"])


def fc_state(state: dict[str, torch.Tensor], what: str) -> dict[str, torch.Tensor]:
    """The two keys upstream's ``load_weights`` reads for the semantic and fusion files."""
    missing = [k for k in ("fc.weight", "fc.bias") if k not in state]
    if missing:
        raise BaselineError(f"{what}: no {missing}")
    return {"fc.weight": state["fc.weight"], "fc.bias": state["fc.bias"]}


def artifact_keys(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """What the artifact file holds: its VAE keys, if any, are reported separately."""
    vae = sorted(k for k in state if k.startswith(VAE_PREFIX))
    return {"keys": len(state), "vae_keys": len(vae)}


def choose_vae(art: dict[str, torch.Tensor], vae: Any) -> dict[str, Any]:
    """Apply the 2026-11-05 rule to the VAE tensors in the artifact file.

    Returns the record: where the VAE in use comes from and how the file's tensors
    compare with the pinned ones. When they differ, the file's tensors are loaded into
    ``vae`` (cast to its dtype) before this returns.
    """
    from dbp.baselines.common import compare_state

    held = {k.removeprefix(VAE_PREFIX): v for k, v in art.items() if k.startswith(VAE_PREFIX)}
    if not held:
        return {"vae_keys": 0, "vae_source": "pinned sd15"}
    pinned = {k: v.detach().cpu() for k, v in vae.state_dict().items()}
    comparison = compare_state(held, pinned)
    record = {"vae_keys": len(held), "vs_pinned_sd15": comparison}
    if comparison["equal"]:
        return {**record, "vae_source": "pinned sd15 (equal to the file's VAE)"}
    if not comparison["same_keys"] or "shape_mismatch" in comparison:
        raise BaselineError(f"CO-SPY artifact_weights.pth holds a VAE that does not fit the "
                            f"pinned architecture ({comparison}); stop and ask")
    dtype = next(iter(pinned.values())).dtype
    vae.load_state_dict({k: v.to(dtype) for k, v in held.items()}, strict=True)
    return {**record, "vae_source": "CO-SPY checkpoint (differs from pinned sd15)"}


def load_cospy(paths: dict[str, Path], sha256: dict[str, str], clip: nn.Module,
               clip_dim: int, vae: Any, device: str) -> tuple[CoSpyFusion, dict[str, Any]]:
    """Build the fused detector and load the three weight files (spec section 1).

    ``paths``/``sha256`` hold ``semantic``, ``artifact`` and ``fusion``. The SigLIP model
    and the pinned VAE are passed in already loaded. Returns the model and the VAE record
    (:func:`choose_vae`).
    """
    states = {k: load_pth(paths[k], sha256[k], f"CO-SPY {k}_weights.pth")
              for k in ("semantic", "artifact", "fusion")}
    model = CoSpyFusion(clip, clip_dim, vae)
    model.sem.fc.load_state_dict({k.removeprefix("fc."): v for k, v in
                                  fc_state(states["semantic"], "semantic").items()})
    model.fc.load_state_dict({k.removeprefix("fc."): v for k, v in
                              fc_state(states["fusion"], "fusion").items()})
    art = states["artifact"]
    expected = [k for k in model.art.state_dict() if not k.startswith(VAE_PREFIX)]
    require_exact_keys(expected, [k for k in art if not k.startswith(VAE_PREFIX)],
                       "CO-SPY artifact_weights.pth")
    vae_record = choose_vae(art, vae)
    non_vae = {k: v for k, v in art.items() if not k.startswith(VAE_PREFIX)}
    missing, unexpected = model.art.load_state_dict(non_vae, strict=False)
    if unexpected or any(not k.startswith(VAE_PREFIX) for k in missing):
        raise BaselineError(f"CO-SPY artifact load: missing {missing[:4]}, unexpected "
                            f"{unexpected[:4]}")
    return model.to(device).eval(), vae_record


def cospy_input(img: np.ndarray) -> tuple[torch.Tensor, bool]:
    """CO-SPY's outer ``test_transform`` on an ``(H, W, 3)`` uint8 image.

    Returns the ``(3, 384, 384)`` tensor in [0, 1] and whether the image was upscaled
    (short side under 384).
    """
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([transforms.Resize(OUTER), transforms.CenterCrop(OUTER),
                                    transforms.ToTensor()])
    return transform(Image.fromarray(img, "RGB")), min(img.shape[:2]) < OUTER
