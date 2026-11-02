"""AEROBLADE (arXiv 2401.17879v2), our own implementation from the paper (spec sections 1, 3).

The repository has no licence, so none of its code is used. Per image:

1. a centre crop of **at most 768 x 768** (AEROBLADE's own data limit; crop, never drop --
   the image is scored and the cut is recorded);
2. for each AE -- our pinned ``sd15`` and ``sd21`` members and the Kandinsky 2.1 MoVQ --
   the round trip through ``reconstruct.py`` (centre crop to a multiple of 8, fp32,
   ``.mode()`` for the KL members, codebook re-quantisation for the MoVQ, no tiling);
3. the LPIPS-VGG16 distance of the **second layer only**, spatially averaged, between
   the cropped input and its reconstruction (the paper's setting, section 3);
4. score = **minus the minimum** distance over the AEs, so larger means more likely fake.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from dbp.baselines.common import BaselineError, center_crop_max
from dbp.decoders.reconstruct import prepare_input, to_tensor

CAP = 768
LPIPS_LAYER = 1  # zero-based: the second VGG16 layer


def build_lpips(device: str) -> Any:
    """LPIPS-VGG16 (v0.1 linear weights shipped with ``lpips``), eval mode."""
    import lpips

    net = lpips.LPIPS(net="vgg", lpips=True, spatial=False, version="0.1", verbose=False)
    return net.to(torch.device(device)).eval()


def layer2_distance(net: Any, a_uint8: np.ndarray, b_uint8: np.ndarray, device: str) -> float:
    """The second LPIPS-VGG layer's spatially averaged distance, inputs in [-1, 1]."""
    with torch.no_grad():
        _, per_layer = net(to_tensor(a_uint8, device), to_tensor(b_uint8, device),
                           retPerLayer=True, normalize=False)
    return float(per_layer[LPIPS_LAYER].reshape(-1)[0])


def score_aeroblade(members: list[Any], net: Any, img: np.ndarray, device: str
                    ) -> tuple[float, dict[str, float], bool]:
    """``(score, distance per AE, whether the 768 cap cut the image)``."""
    capped = center_crop_max(img, CAP)
    distances: dict[str, float] = {}
    for member in members:
        reference = prepare_input(capped, member.spec.crop_multiple)
        reconstruction = member.reconstruct(capped)
        if reconstruction.shape != reference.shape:
            raise BaselineError(f"{member.spec.key}: reconstruction {reconstruction.shape} "
                                f"!= input {reference.shape}")
        distances[member.spec.key] = layer2_distance(net, reference, reconstruction, device)
    if not distances or not all(math.isfinite(v) for v in distances.values()):
        raise BaselineError(f"AEROBLADE: unusable distances {distances}")
    return -min(distances.values()), distances, capped.shape != img.shape
