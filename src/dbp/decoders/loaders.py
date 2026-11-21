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


# ---------------------------------------------------------------------------
# P5: video members (configs/video_bank.yaml)
# ---------------------------------------------------------------------------

#: The flags of a diffusers video autoencoder that would split a clip in time or space
#: with overlap blending. Every one is switched off and then asserted off.
_VIDEO_SPLIT_FLAGS = ("use_tiling", "use_framewise_encoding", "use_framewise_decoding")

#: DBP_DRYRUN=1 only (local CPU): tiny random-init instances of the pinned classes, the
#: same code paths at a fraction of the width. Never on CUDA, never on Kaggle data.
DRYRUN_VIDEO_CONFIGS: dict[str, dict[str, Any]] = {
    "AutoencoderKLWan": {"base_dim": 8, "z_dim": 16, "num_res_blocks": 1},
    "AutoencoderKLHunyuanVideo": {"block_out_channels": (8, 8, 8, 8), "layers_per_block": 1,
                                  "norm_num_groups": 4, "mid_block_add_attention": False},
    "AutoencoderKLCogVideoX": {"block_out_channels": (8, 8, 8, 8), "layers_per_block": 1,
                               "norm_num_groups": 4},
    "AutoencoderKLLTXVideo": {"block_out_channels": (8, 8, 8, 8),
                              "decoder_block_out_channels": (8, 8, 8, 8),
                              "layers_per_block": (1, 1, 1, 1, 1),
                              "decoder_layers_per_block": (1, 1, 1, 1, 1)},
    "AutoencoderKLMochi": {"encoder_block_out_channels": (32, 32, 32, 32),
                           "decoder_block_out_channels": (32, 32, 32, 32),
                           "layers_per_block": (1, 1, 1, 1, 1),
                           "add_attention_block": (False, False, False, False, False)},
    "AutoencoderKL": {"block_out_channels": (8, 8, 8, 8), "norm_num_groups": 4,
                      "down_block_types": ("DownEncoderBlock2D",) * 4,
                      "up_block_types": ("UpDecoderBlock2D",) * 4, "layers_per_block": 1},
}
#: Extra dry-run kwargs for a member whose architecture differs from its class default.
DRYRUN_MEMBER_EXTRA: dict[str, dict[str, Any]] = {
    "wan22_5b": {"z_dim": 48, "is_residual": True, "patch_size": 2, "in_channels": 12,
                 "out_channels": 12, "scale_factor_spatial": 16},
}


def module_sha256(module: Any) -> str:
    """SHA-256 over a module's state dict (sorted names, fp32 bytes): lineage identity."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class VideoMember:
    """A loaded video-bank member: ``reconstruct((T, H, W, 3) uint8) -> same shape``."""

    spec: Any  # dbp.decoders.registry.VideoDecoderSpec
    model: Any
    device: torch.device
    precision: Precision
    dry_run_random_init: bool = False

    def reconstruct(self, frames_uint8: np.ndarray) -> np.ndarray:
        """Reconstruct one clip. ``T`` must be native and ``H``/``W`` multiples of the
        member's spatial factor; the caller has already centre-cropped (never resized)."""
        from dbp.decoders.reconstruct import (
            reconstruct_per_frame,
            reconstruct_video_kl,
            video_to_tensor,
            video_to_uint8,
        )

        frames, height, width = frames_uint8.shape[:3]
        if not self.spec.accepts_frames(frames):
            raise MemberLoadError(f"{self.spec.key}: {frames} frames is not native "
                                  f"(1 + k x {self.spec.temporal_factor})")
        if height % self.spec.spatial_factor or width % self.spec.spatial_factor:
            raise MemberLoadError(f"{self.spec.key}: {height}x{width} is not a multiple of "
                                  f"{self.spec.spatial_factor}; crop, never resize")
        x = video_to_tensor(frames_uint8, self.device, torch_dtype(self.precision))
        if self.spec.kind == "vae_kl_per_frame":
            return video_to_uint8(reconstruct_per_frame(self.model, x))
        return video_to_uint8(reconstruct_video_kl(self.model, x))

    def submodule_sha256(self) -> dict[str, str]:
        """State-dict digests of ``encoder``, ``decoder`` and the quant convs, when present."""
        return {name: module_sha256(getattr(self.model, name))
                for name in ("encoder", "decoder", "quant_conv", "post_quant_conv")
                if getattr(self.model, name, None) is not None}

    def free(self) -> None:
        """Release the model and empty the CUDA cache."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assert_video_splitting_disabled(model: Any, key: str) -> dict[str, bool]:
    """Switch every tiling / frame-wise flag off, then assert it is off.

    HunyuanVideo's diffusers default is ``use_framewise_encoding = True``: any clip over
    16 frames would be temporal-tiled with blending, an artificial temporal trace. Native
    causal caches (Wan's ``feat_cache``, CogVideoX's conv cache) are the architecture and
    are not touched.

    Returns:
        Each flag the model has, after the change (all False).
    """
    if hasattr(model, "disable_tiling"):
        model.disable_tiling()
    state = {}
    for flag in _VIDEO_SPLIT_FLAGS:
        if hasattr(model, flag):
            setattr(model, flag, False)
            state[flag] = bool(getattr(model, flag))
    if any(state.values()):
        raise MemberLoadError(f"{key}: splitting flags still on after disabling: {state}")
    return state


def require_diffusers_version(expected: str) -> str:
    """Refuse unless the installed diffusers is exactly the bank's pinned version.

    One version serves the pilot, the AWS reconstruction and every Kaggle scoring (user,
    2026-11-21), so a reconstruction can never come from a different diffusers than the one
    the protocol was checked against.

    Returns:
        The installed version (equal to ``expected``).

    Raises:
        MemberLoadError: On any other version, or when no version is pinned.
    """
    import diffusers

    if not expected:
        raise MemberLoadError("the video bank pins no diffusers version")
    if diffusers.__version__ != expected:
        raise MemberLoadError(f"diffusers {diffusers.__version__} is installed; the video bank "
                              f"is pinned to {expected} (configs/video_bank.yaml). Refused.")
    return diffusers.__version__


def load_video_member(spec: Any, device: torch.device | str = "cpu",
                      precision: Precision = "fp32") -> VideoMember:
    """Load one video-bank member, pinned, safetensors only, no remote code.

    Raises:
        MemberLoadError: For a diffusers version other than the bank's, an excluded member,
            a gated-access denial, a latent width that differs from the registry, or a
            splitting flag that stays on.
    """
    import os

    import diffusers

    from dbp.decoders.registry import VIDEO_CLASSES

    require_diffusers_version(spec.diffusers)
    if spec.role == "excluded":
        raise MemberLoadError(f"{spec.key}: excluded ({spec.excluded_reason.strip()})")
    if spec.cls not in VIDEO_CLASSES:
        raise MemberLoadError(f"{spec.key}: class {spec.cls!r} is not allowed")
    torch_device = torch.device(device)
    dtype = torch_dtype(precision)
    cls = getattr(diffusers, spec.cls)
    dry = os.environ.get("DBP_DRYRUN") == "1"
    try:
        if dry:
            if torch_device.type == "cuda":
                raise MemberLoadError("DBP_DRYRUN=1 builds random-init stand-ins for CPU only")
            torch.manual_seed(0)
            kwargs = {**DRYRUN_VIDEO_CONFIGS[spec.cls], **DRYRUN_MEMBER_EXTRA.get(spec.key, {})}
            if spec.latent_attr == "latent_channels":
                kwargs.setdefault("latent_channels", spec.latent_channels)
            model = cls(**kwargs)
            if dtype != torch.float32:
                model = model.to(dtype)
        else:
            model = cls.from_pretrained(spec.repo, subfolder=spec.subfolder,
                                        revision=spec.revision, use_safetensors=True,
                                        torch_dtype=dtype, token=get_hf_token())
    except MemberLoadError:
        raise
    except Exception as exc:
        raise MemberLoadError(f"{spec.key}: {type(exc).__name__}: {exc}") from exc
    reported = getattr(model.config, spec.latent_attr, None)
    if reported != spec.latent_channels:
        raise MemberLoadError(f"{spec.key}: registry latent_channels={spec.latent_channels} but "
                              f"the model reports {spec.latent_attr}={reported}")
    model.to(torch_device)
    model.eval()
    assert_video_splitting_disabled(model, spec.key)
    return VideoMember(spec, model, torch_device, precision, dry_run_random_init=dry)
