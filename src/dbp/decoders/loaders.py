"""Loading bank members and wrapping them behind one uniform interface.

Every member, whatever its architecture, is exposed as a :class:`BankMember` with a
single method ``reconstruct(uint8 HWC) -> uint8 HWC``. Call sites therefore never need
to know whether they are talking to a KL VAE, a DC-AE or a GAN upsampler, which is
what makes the permutation-invariant head in P2 possible.

Safety rules enforced here (guardrail #6):

- ``use_safetensors=True`` for every Hub member.
- ``trust_remote_code`` is never passed, and never True.
- Every Hub load is pinned to the commit SHA from the registry.
- The one unavoidable ``.pth`` (Real-ESRGAN) is loaded through spandrel, which uses
  restricted deserialization, and its SHA-256 is verified against the registry.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dbp.decoders.reconstruct import (
    Precision,
    prepare_input,
    reconstruct_sr_gan,
    reconstruct_vae_dc,
    reconstruct_vae_kl,
    reconstruct_vae_kl_5d,
    reconstruct_vq,
    to_tensor,
    to_uint8,
    torch_dtype,
)
from dbp.decoders.registry import DecoderSpec
from dbp.utils.hf import get_hf_token

#: Reconstruction function for each kind that operates on a normalised NCHW tensor.
#: ``sr_gan`` is absent because it works on the uint8 array directly.
_TENSOR_RECONSTRUCTORS = {
    "vae_kl": reconstruct_vae_kl,
    "vae_kl_flux2": reconstruct_vae_kl,
    "vae_kl_5d": reconstruct_vae_kl_5d,
    "vae_dc": reconstruct_vae_dc,
    "vq": reconstruct_vq,
}


class MemberLoadError(RuntimeError):
    """Raised when a bank member cannot be loaded (access, integrity or API failure)."""


@dataclass
class BankMember:
    """A loaded bank member with a uniform reconstruction interface."""

    spec: DecoderSpec
    model: Any
    device: torch.device
    precision: Precision

    def reconstruct(self, img_uint8: np.ndarray) -> np.ndarray:
        """Run the section 6.4 protocol for this member.

        Args:
            img_uint8: ``(H, W, 3)`` ``uint8`` RGB array. It is centre-cropped to the
                member's multiple first; it is never resized.

        Returns:
            ``(H', W', 3)`` ``uint8`` reconstruction, where ``H'``/``W'`` are the
            cropped dimensions.
        """
        x_uint8 = prepare_input(img_uint8, self.spec.crop_multiple)

        if self.spec.kind == "sr_gan":
            return reconstruct_sr_gan(self.model, x_uint8, scale=int(self.spec.downsample or 4))

        reconstructor = _TENSOR_RECONSTRUCTORS.get(self.spec.kind)
        if reconstructor is None:
            raise MemberLoadError(f"unsupported kind {self.spec.kind!r}")

        dtype = torch_dtype(self.precision)
        x = to_tensor(x_uint8, self.device, dtype)
        return to_uint8(reconstructor(self.model, x))

    def decoder_state_dict_sha256(self) -> str | None:
        """SHA-256 over the decoder sub-module's weights, for lineage de-duplication.

        Two registry entries whose decoders hash identically are the same decoder and
        must not be treated as two lineages in the leave-one-lineage-out protocol.
        Returns None for members with no ``decoder`` attribute.
        """
        decoder = getattr(self.model, "decoder", None)
        if decoder is None:
            return None
        digest = hashlib.sha256()
        for name, tensor in sorted(decoder.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
        return digest.hexdigest()

    def free(self) -> None:
        """Release the model and empty the CUDA cache (protocol step 10)."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _hub_kwargs(spec: DecoderSpec, dtype: torch.dtype) -> dict[str, Any]:
    """Keyword arguments shared by every Hub load.

    ``trust_remote_code`` is never passed, so it can never be True (guardrail #6), and
    the revision is always the pinned commit SHA.
    """
    kwargs: dict[str, Any] = {
        "revision": spec.revision,
        "use_safetensors": True,
        "torch_dtype": dtype,
        "token": get_hf_token(),
    }
    if spec.subfolder:
        kwargs["subfolder"] = spec.subfolder
    return kwargs


def _check_latent_channels(spec: DecoderSpec, reported: int | None, attribute: str) -> None:
    """Fail loudly when the registry and the loaded config disagree."""
    if reported != spec.latent_channels:
        raise MemberLoadError(
            f"{spec.key}: registry declares latent_channels={spec.latent_channels} but the "
            f"loaded model reports {attribute}={reported}. The registry is wrong or the "
            f"pinned revision points at a different model."
        )


def _load_vae_kl(spec: DecoderSpec, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(spec.repo, **_hub_kwargs(spec, dtype))
    _check_latent_channels(spec, vae.config.latent_channels, "latent_channels")
    return vae


def _load_vae_kl_flux2(spec: DecoderSpec, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(spec.repo, **_hub_kwargs(spec, dtype))
    _check_latent_channels(spec, vae.config.latent_channels, "latent_channels")
    return vae


def _load_vae_kl_5d(spec: DecoderSpec, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import AutoencoderKLQwenImage

    vae = AutoencoderKLQwenImage.from_pretrained(spec.repo, **_hub_kwargs(spec, dtype))
    # This family names the latent width `z_dim`, not `latent_channels`.
    _check_latent_channels(spec, vae.config.z_dim, "z_dim")
    ratio = getattr(vae, "spatial_compression_ratio", None)
    if ratio is not None and int(ratio) != int(spec.downsample or 0):
        raise MemberLoadError(
            f"{spec.key}: registry declares downsample={spec.downsample} but the loaded "
            f"model reports spatial_compression_ratio={ratio}."
        )
    return vae


def _load_vq(spec: DecoderSpec, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import VQModel

    model = VQModel.from_pretrained(spec.repo, **_hub_kwargs(spec, dtype))
    _check_latent_channels(spec, model.config.latent_channels, "latent_channels")
    return model


def _load_vae_dc(spec: DecoderSpec, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import AutoencoderDC

    return AutoencoderDC.from_pretrained(spec.repo, **_hub_kwargs(spec, dtype))


def sha256_of_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_sr_weights(spec: DecoderSpec, cache_dir: Path) -> tuple[Path, str]:
    """Download the Real-ESRGAN weights if absent and return the path and its SHA-256.

    The URL is the official GitHub release asset from the registry and is the only
    GitHub host this project contacts. If the registry already pins a ``sha256``, the
    downloaded file must match it or this raises.
    """
    if not spec.url:
        raise MemberLoadError(f"{spec.key}: no url in registry")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / spec.url.rsplit("/", 1)[-1]

    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        # SSL verification is left at its default; it is never disabled.
        with urllib.request.urlopen(spec.url) as response, tmp.open("wb") as out:  # noqa: S310
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)

    digest = sha256_of_file(dest)
    if spec.sha256 is not None and digest != spec.sha256:
        raise MemberLoadError(
            f"{spec.key}: SHA-256 mismatch for {dest.name}.\n"
            f"  expected (registry): {spec.sha256}\n"
            f"  actual   (on disk) : {digest}\n"
            f"Refusing to load. Delete the file and re-download, or investigate."
        )
    return dest, digest


def _load_sr_gan(spec: DecoderSpec, device: torch.device, dtype: torch.dtype, cache_dir: Path):
    from spandrel import ImageModelDescriptor, ModelLoader

    weights_path, _ = fetch_sr_weights(spec, cache_dir)
    # spandrel performs restricted deserialization; basicsr/realesrgan are never used.
    descriptor = ModelLoader().load_from_file(str(weights_path))
    if not isinstance(descriptor, ImageModelDescriptor):
        raise MemberLoadError(
            f"{spec.key}: spandrel returned {type(descriptor).__name__}, expected an "
            f"ImageModelDescriptor."
        )
    if descriptor.scale != spec.downsample:
        raise MemberLoadError(
            f"{spec.key}: registry declares scale {spec.downsample} but the weights are "
            f"x{descriptor.scale}."
        )
    descriptor.to(device)
    if dtype == torch.float16 and descriptor.supports_half:
        descriptor.model.half()
    descriptor.eval()
    return descriptor


#: Hub loader for each kind. ``sr_gan`` is handled separately because it needs a
#: cache directory and returns a spandrel descriptor rather than a diffusers model.
_HUB_LOADERS = {
    "vae_kl": _load_vae_kl,
    "vae_kl_flux2": _load_vae_kl_flux2,
    "vae_kl_5d": _load_vae_kl_5d,
    "vae_dc": _load_vae_dc,
    "vq": _load_vq,
}


def assert_tiling_disabled(model: Any, key: str) -> None:
    """Turn tiling off and then verify it is off (decision D-002).

    Tile seams are an artificial forensic trace: a detector could learn the seam
    instead of the decoder. Several diffusers autoencoders tile automatically once the
    input exceeds a configured size, so it is not enough to simply never call
    ``enable_tiling()`` -- the flag is disabled and then asserted. ``enable_slicing()``
    stays permitted; it only splits the batch.

    Raises:
        MemberLoadError: If the model still reports tiling as enabled.
    """
    if hasattr(model, "disable_tiling"):
        model.disable_tiling()
    if getattr(model, "use_tiling", False):
        raise MemberLoadError(
            f"{key}: use_tiling is still True after disable_tiling(). Tile seams are an "
            f"artificial forensic trace and the protocol forbids them (D-002)."
        )


def load_member(
    spec: DecoderSpec,
    device: torch.device | str = "cpu",
    precision: Precision = "fp32",
    cache_dir: Path | None = None,
) -> BankMember:
    """Load one bank member and wrap it as a :class:`BankMember`.

    Args:
        spec: The validated registry entry.
        device: Torch device to place the model on.
        precision: ``"fp32"`` always for the reference run; ``"fp16"`` only for the
            parity measurement, or afterwards for members recorded as ``fp16_allowed``.
        cache_dir: Where the Real-ESRGAN ``.pth`` is stored. Required for ``sr_gan``.

    Returns:
        A ready, eval-mode :class:`BankMember`.

    Raises:
        MemberLoadError: On gated-access denial, a latent-channel or scale mismatch,
            or a SHA-256 mismatch on the downloaded weights.
    """
    torch_device = torch.device(device)
    dtype = torch_dtype(precision)

    try:
        if spec.kind == "sr_gan":
            if cache_dir is None:
                raise MemberLoadError(f"{spec.key}: cache_dir is required for sr_gan members")
            model = _load_sr_gan(spec, torch_device, dtype, cache_dir)
            return BankMember(spec=spec, model=model, device=torch_device, precision=precision)
        loader = _HUB_LOADERS.get(spec.kind)
        if loader is None:
            raise MemberLoadError(f"{spec.key}: unsupported kind {spec.kind!r}")
        model = loader(spec, torch_device, dtype)
    except MemberLoadError:
        raise
    except Exception as exc:
        raise MemberLoadError(f"{spec.key}: {type(exc).__name__}: {exc}") from exc

    model.to(torch_device)
    model.eval()
    assert_tiling_disabled(model, spec.key)
    return BankMember(spec=spec, model=model, device=torch_device, precision=precision)
