"""S3: the decoder-bank probe -- explicit, interpretable decoder evidence.

At test time every image goes through all twelve distinct decoders and the residual
``r_k = x - D_k(E_k(x))`` is summarised into a short per-member vector. The bank is
fixed, so these features are computed **once per image** and reused by every protocol's
head; only the head is trained per protocol.

Twenty numbers per member (:data:`FEATURE_NAMES`):

* ``psnr_db``, ``mean_abs_residual``, ``lpips_vgg`` -- how far the decoder had to move
  the picture. AEROBLADE [R8] shows the minimum of the third across VAEs is on its own
  a training-free detector, and it is kept here as a free baseline.
* ``radial_logpower_00..15`` -- the 16-bin radial log-power spectrum of the luma
  residual. A decoder's fingerprint is a spatial-frequency signature; the radial
  average is the rotation-invariant summary of it.
* ``hf_energy_ratio`` -- the share of residual energy in the top quarter of radii,
  which is where upsampling and de-tokenisation artefacts live.

Two heads consume them. **S3-LR** is a logistic regression on the standardised
concatenation and has member-specific weights, so the bank it was fitted on is the
bank it can score. **S3-Set** is permutation-invariant by construction [R10][R11]: a
shared MLP over each member's vector, attention pooling that never sees a member
identity, then a linear layer. It is the only design in which adding a decoder at test
time is well defined, which is the precondition for the P3 plug-in experiment -- and it
is a design constraint, not an accident, so :func:`assert_permutation_invariant` is
part of the module rather than only of the tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

#: Bins in the radial spectrum. Sixteen over a 320-px crop keeps roughly ten pixels of
#: radius per bin, which is enough to be stable on one image.
RADIAL_BINS = 16

#: Radii above this fraction of Nyquist count as high frequency.
HF_RADIUS_FRACTION = 0.75

#: BT.601 luma weights, the same convention the greyscale augmentation uses.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)

FEATURE_NAMES: tuple[str, ...] = (
    "psnr_db",
    "mean_abs_residual",
    "lpips_vgg",
    *(f"radial_logpower_{i:02d}" for i in range(RADIAL_BINS)),
    "hf_energy_ratio",
)

#: Members whose reconstruction is an autoencoder round trip, so AEROBLADE's argument
#: applies to them. The GAN super-resolver is not one and is excluded from that score.
AEROBLADE_EXCLUDED_KINDS = frozenset({"sr_gan"})


class ProbeError(ValueError):
    """Raised when probe features cannot be computed from what was given."""


def luma(img: np.ndarray) -> np.ndarray:
    """BT.601 luma of an HWC array, in the array's own dtype range."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ProbeError(f"expected HWC RGB, got shape {img.shape}")
    weights = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
    return img.astype(np.float64) @ weights


def _radius_grid(height: int, width: int) -> np.ndarray:
    """Normalised spatial-frequency radius, 0 at DC and 1 at Nyquist on the axes.

    Normalising to the **axis** Nyquist rather than to the corner matters: the corner
    is a factor sqrt(2) further out, so dividing by it would leave the outermost bins
    reachable only by diagonal content and empty for a purely horizontal or vertical
    signal. Radii past 1 (the corners) are folded into the last bin, which is where
    they belong.
    """
    ys = np.fft.fftshift(np.fft.fftfreq(height)) * 2.0
    xs = np.fft.fftshift(np.fft.fftfreq(width)) * 2.0
    return np.clip(np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2), 0.0, 1.0)


def radial_log_power_spectrum(plane: np.ndarray, bins: int = RADIAL_BINS) -> np.ndarray:
    """Radially averaged log power spectrum of a single plane.

    Args:
        plane: ``(H, W)`` real array -- here the luma of a residual.
        bins: Number of equal-width radius bins between 0 and Nyquist.

    Returns:
        ``(bins,)`` of ``log1p`` mean power, lowest radius first.

    Raises:
        ProbeError: If the plane is not 2-D or is degenerate.
    """
    if plane.ndim != 2:
        raise ProbeError(f"expected a 2-D plane, got shape {plane.shape}")
    height, width = plane.shape
    if height < 2 or width < 2:
        raise ProbeError(f"plane {height}x{width} is too small for a spectrum")

    power = np.abs(np.fft.fftshift(np.fft.fft2(plane))) ** 2
    radius = _radius_grid(height, width)
    index = np.clip((radius * bins).astype(int), 0, bins - 1)
    out = np.zeros(bins, dtype=np.float64)
    for b in range(bins):
        mask = index == b
        # An empty bin cannot happen for bins <= min(H, W) // 2, but a zero would read
        # as "measured, no energy" rather than "not measured", so it is stated.
        out[b] = np.log1p(power[mask].mean()) if np.any(mask) else 0.0
    return out


def high_frequency_energy_ratio(
    plane: np.ndarray, fraction: float = HF_RADIUS_FRACTION
) -> float:
    """Share of spectral energy above ``fraction`` of the maximum radius."""
    if plane.ndim != 2:
        raise ProbeError(f"expected a 2-D plane, got shape {plane.shape}")
    power = np.abs(np.fft.fftshift(np.fft.fft2(plane))) ** 2
    radius = _radius_grid(*plane.shape)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power[radius >= fraction].sum() / total)


def psnr_db(original: np.ndarray, reconstruction: np.ndarray) -> float:
    """PSNR in dB between two uint8 images, capped where they are identical.

    Returns:
        The PSNR, or ``100.0`` when the two are byte-identical. ``inf`` would
        propagate into every downstream mean and standardisation.
    """
    a = original.astype(np.float64)
    b = reconstruction.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return 100.0
    return float(min(100.0, 10.0 * np.log10(255.0**2 / mse)))


def member_features(
    original: np.ndarray, reconstruction: np.ndarray, lpips_vgg: float
) -> np.ndarray:
    """The twenty numbers for one image through one decoder member.

    Args:
        original: ``(H, W, 3)`` ``uint8`` input crop.
        reconstruction: ``(H, W, 3)`` ``uint8`` output of that member.
        lpips_vgg: LPIPS-vgg between the two, computed by the caller because it needs
            a network this module deliberately does not own.

    Returns:
        ``(20,)`` float32, ordered as :data:`FEATURE_NAMES`.

    Raises:
        ProbeError: If the two images differ in shape or are not uint8 RGB.
    """
    if original.shape != reconstruction.shape:
        raise ProbeError(
            f"original {original.shape} and reconstruction {reconstruction.shape} differ"
        )
    if original.dtype != np.uint8 or reconstruction.dtype != np.uint8:
        raise ProbeError(
            f"probe features are computed on uint8 crops, got {original.dtype} and "
            f"{reconstruction.dtype}"
        )
    if not np.isfinite(lpips_vgg):
        raise ProbeError(f"lpips_vgg must be finite, got {lpips_vgg}")

    residual = original.astype(np.float64) - reconstruction.astype(np.float64)
    residual_luma = luma(residual)
    values = np.concatenate(
        [
            [psnr_db(original, reconstruction)],
            [float(np.mean(np.abs(residual)))],
            [float(lpips_vgg)],
            radial_log_power_spectrum(residual_luma),
            [high_frequency_energy_ratio(residual_luma)],
        ]
    )
    if values.size != len(FEATURE_NAMES):
        raise ProbeError(f"expected {len(FEATURE_NAMES)} features, built {values.size}")
    return values.astype(np.float32)


def stack_members(
    per_member: Mapping[str, np.ndarray], members: Sequence[str]
) -> np.ndarray:
    """Arrange per-member vectors into a fixed ``(n_members, n_features)`` matrix.

    Args:
        per_member: Member key -> its feature vector.
        members: The bank order to impose. Fixing it here is what makes S3-LR's
            coefficients mean the same thing from one run to the next.

    Returns:
        ``(len(members), 20)`` float32.

    Raises:
        ProbeError: If a member is missing. A zero row would be read by the head as a
            perfect reconstruction, which is the opposite of "not measured".
    """
    missing = [key for key in members if key not in per_member]
    if missing:
        raise ProbeError(f"no probe features for {missing}")
    return np.stack([np.asarray(per_member[key], dtype=np.float32) for key in members])


def flatten_members(matrix: np.ndarray) -> np.ndarray:
    """Flatten ``(n_members, n_features)`` to the S3-LR input vector."""
    return np.asarray(matrix, dtype=np.float32).reshape(-1)


def feature_column_names(members: Sequence[str]) -> list[str]:
    """Column names of the flattened vector, for the manifest and the coefficients."""
    return [f"{member}.{name}" for member in members for name in FEATURE_NAMES]


def aeroblade_score(
    lpips_by_member: Mapping[str, float], kinds: Mapping[str, str]
) -> dict[str, Any]:
    """Training-free AEROBLADE-style score: the minimum LPIPS over the VAE members [R8].

    A generated image is easy for the autoencoder that made it, so the smallest
    reconstruction distance across a bank of VAEs is low for fakes and higher for
    reals. No training, no threshold of its own -- a free baseline that says how much
    the learned heads are adding.

    Args:
        lpips_by_member: Member key -> LPIPS-vgg for this image.
        kinds: Member key -> its registry ``kind``.

    Returns:
        ``{"score", "argmin_member", "n_members", "excluded"}``. ``score`` is the
        **negated** minimum, so that larger means more likely fake, like every other
        stream.

    Raises:
        ProbeError: If no eligible member is left.
    """
    eligible = {
        key: float(value)
        for key, value in lpips_by_member.items()
        if kinds.get(key) not in AEROBLADE_EXCLUDED_KINDS
    }
    excluded = sorted(set(lpips_by_member) - set(eligible))
    if not eligible:
        raise ProbeError(
            f"every member was excluded from the AEROBLADE score: {excluded}"
        )
    argmin = min(eligible, key=lambda key: eligible[key])
    return {
        "score": -eligible[argmin],
        "argmin_member": argmin,
        "n_members": len(eligible),
        "excluded": excluded,
    }


def standardiser(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation for standardising S3-LR inputs.

    Fitted on the training subset and then frozen. A zero standard deviation is
    replaced by 1.0, which leaves that column at zero rather than producing inf.

    Args:
        matrix: ``(n_samples, n_features)`` training features.

    Returns:
        ``(mean, scale)``.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ProbeError(f"need at least 2 samples to standardise, got {values.shape}")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def standardise(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a frozen standardiser."""
    return ((np.asarray(matrix, dtype=np.float32) - mean) / scale).astype(np.float32)


def build_set_head(n_features: int = len(FEATURE_NAMES), hidden: int = 64) -> Any:
    """S3-Set: a permutation-invariant head over the bank [R10][R11].

    ``phi`` is shared across members and never sees a member identity, attention
    pooling is a weighted sum, and the final linear layer reads only the pooled
    vector. Every one of those three is required for invariance; the module asserts it
    rather than assuming it.

    Args:
        n_features: Features per member.
        hidden: Width of the shared MLP and of the pooled representation.

    Returns:
        A ``torch.nn.Module`` mapping ``(batch, n_members, n_features)`` to
        ``(batch,)`` logits.
    """
    import torch
    from torch import nn

    class SetHead(nn.Module):
        """Deep-Sets encoder with attention pooling and no member identity."""

        def __init__(self) -> None:
            super().__init__()
            self.phi = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            self.attention = nn.Linear(hidden, 1)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Map per-member features to one logit per sample."""
            if x.ndim != 3:
                raise ProbeError(
                    f"SetHead expects (batch, members, features), got {tuple(x.shape)}"
                )
            embedded = self.phi(x)
            weights = torch.softmax(self.attention(embedded), dim=1)
            pooled = (weights * embedded).sum(dim=1)
            return self.out(pooled).squeeze(-1)

    return SetHead()


def assert_permutation_invariant(head: Any, n_members: int = 12, seed: int = 0) -> float:
    """Check a head's output does not depend on the order of the bank.

    Returns:
        The largest absolute difference found, which a caller may log.

    Raises:
        ProbeError: If the head is not invariant. Without this property the P3
            plug-in experiment -- adding a decoder at test time -- is not defined.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(4, n_members, len(FEATURE_NAMES), generator=generator)
    permutation = torch.randperm(n_members, generator=generator)
    head.eval()
    with torch.no_grad():
        delta = float((head(x) - head(x[:, permutation, :])).abs().max())
    if delta > 1e-5:
        raise ProbeError(
            f"head is not permutation invariant: reordering the bank moved the logit "
            f"by {delta:.3e}. A member-identity input has leaked into it."
        )
    return delta
