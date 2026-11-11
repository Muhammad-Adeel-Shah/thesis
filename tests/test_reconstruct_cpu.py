"""CPU reconstruction tests.

The pure-tensor parts of the protocol run everywhere and are tested unconditionally.
The one test that touches the Hub is marked ``network`` and uses a 64x64 crop through
the sd15 VAE, which is the smallest thing that still exercises the real encode/decode
path.

Run without network tests:   pytest -m "not network"
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dbp.decoders.reconstruct import (
    FP16_PSNR_THRESHOLD_DB,
    fp16_verdict,
    max_abs_diff,
    prepare_input,
    psnr_uint8,
    to_tensor,
    to_uint8,
    verify_diffusers_api,
)
from dbp.decoders.registry import load_bank_config
from dbp.utils.seed import seed_everything

RNG = np.random.default_rng(2026)


def _random_image(size: int = 64) -> np.ndarray:
    return RNG.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_diffusers_output_attribute_names_are_still_valid():
    """The protocol reads .latent_dist, .latent and .sample by name."""
    checks = verify_diffusers_api()
    missing = [name for name, ok in checks.items() if not ok]
    assert not missing, f"diffusers changed these output attributes: {missing}"


def test_tensor_round_trip_is_lossless_for_uint8():
    """uint8 -> [-1,1] float32 -> uint8 must be the identity."""
    img = _random_image(32)
    restored = to_uint8(to_tensor(img, "cpu"))
    assert np.array_equal(restored, img)


def test_tensor_round_trip_is_lossless_in_unit_range():
    img = _random_image(32)
    tensor = to_tensor(img, "cpu", value_range=(0.0, 1.0))
    assert np.array_equal(to_uint8(tensor, value_range=(0.0, 1.0)), img)


def test_normalisation_endpoints():
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full((2, 2, 3), 255, dtype=np.uint8)
    assert torch.isclose(to_tensor(black, "cpu").min(), torch.tensor(-1.0))
    assert torch.isclose(to_tensor(white, "cpu").max(), torch.tensor(1.0))


def test_to_uint8_clamps_out_of_range_values():
    """Values outside [-1,1] must clamp, not wrap."""
    x = torch.tensor([[[[-5.0]], [[5.0]], [[0.0]]]])
    out = to_uint8(x)
    assert out[0, 0, 0] == 0
    assert out[0, 0, 1] == 255


def test_psnr_of_identical_images_is_infinite():
    img = _random_image(16)
    assert psnr_uint8(img, img) == float("inf")


def test_psnr_decreases_as_error_grows():
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    near = np.full((16, 16, 3), 129, dtype=np.uint8)
    far = np.full((16, 16, 3), 160, dtype=np.uint8)
    assert psnr_uint8(img, near) > psnr_uint8(img, far)


def test_max_abs_diff_does_not_wrap():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert max_abs_diff(a, b) == 255


def test_fp16_verdict_rejects_low_psnr():
    base = np.full((16, 16, 3), 128, dtype=np.uint8)
    degraded = np.full((16, 16, 3), 160, dtype=np.uint8)
    verdict = fp16_verdict(base, degraded)
    assert verdict["fp16_allowed"] is False
    assert str(FP16_PSNR_THRESHOLD_DB) in verdict["fp16_reason"]


def test_fp16_verdict_accepts_identical_output():
    base = _random_image(16)
    assert fp16_verdict(base, base.copy())["fp16_allowed"] is True


def test_fp16_verdict_handles_a_failed_run():
    verdict = fp16_verdict(_random_image(16), None)
    assert verdict["fp16_allowed"] is False
    assert verdict["fp16_reason"] == "fp16 run failed"


@pytest.mark.parametrize(("size", "multiple"), [(70, 8), (70, 32), (70, 4)])
def test_prepare_input_aligns_to_the_members_multiple(size, multiple):
    out = prepare_input(_random_image(size), multiple)
    assert out.shape[0] % multiple == 0
    assert out.shape[1] % multiple == 0


@pytest.mark.network
def test_sd15_vae_reconstructs_64x64_deterministically():
    """Smallest end-to-end check against the real sd15 VAE, pinned to its commit SHA."""
    from pathlib import Path

    from dbp.decoders.loaders import MemberLoadError, load_member

    seed_everything(2026, deterministic=True)
    config = load_bank_config(Path(__file__).resolve().parents[1] / "configs" / "decoder_bank.yaml")
    spec = config.by_key("sd15")

    try:
        member = load_member(spec, device="cpu", precision="fp32")
    except MemberLoadError as exc:
        pytest.skip(f"sd15 VAE unavailable (network or Hub): {exc}")

    img = _random_image(64)
    first = member.reconstruct(img)
    second = member.reconstruct(img)

    assert first.shape == img.shape
    assert first.dtype == np.uint8
    # Determinism: mode() not sample(), so repeated runs must agree within quantisation.
    assert max_abs_diff(first, second) <= 1
    # A real VAE round-trip of random noise is lossy but must not be degenerate.
    assert not np.array_equal(first, img)
    assert first.std() > 0
    member.free()


def _tiny_kl_vae(latent_channels: int = 4):
    """A randomly-initialised, very small AutoencoderKL.

    This exercises the real diffusers encode/decode code path -- ``latent_dist.mode()``,
    ``DecoderOutput.sample``, the f8 downsampling -- with no weights to download, so the
    protocol itself is covered even in an offline runtime.
    """
    from diffusers import AutoencoderKL

    # Four down-blocks, exactly like the real SD KL-f8 VAE: diffusers omits the
    # downsampler on the final block, so N blocks give a factor of 2**(N-1) = 8.
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(4, 4, 8, 8),
        layers_per_block=1,
        latent_channels=latent_channels,
        norm_num_groups=2,
        sample_size=64,
    ).eval()


def test_kl_encode_decode_shapes_and_determinism():
    """encode->decode must preserve size and be deterministic via mode()."""
    import torch as _torch

    from dbp.decoders.reconstruct import reconstruct_vae_kl

    seed_everything(2026, deterministic=True)
    vae = _tiny_kl_vae()
    x = to_tensor(_random_image(64), "cpu")

    with _torch.no_grad():
        first = reconstruct_vae_kl(vae, x)
        second = reconstruct_vae_kl(vae, x)

    assert first.shape == x.shape
    assert _torch.equal(first, second), "mode() must make encode->decode deterministic"
    assert _torch.isfinite(first).all()


def test_kl_latent_is_downsampled_by_eight():
    """The f8 contract: a 64x64 input must produce an 8x8 latent."""
    import torch as _torch

    seed_everything(2026, deterministic=True)
    vae = _tiny_kl_vae(latent_channels=4)
    with _torch.no_grad():
        z = vae.encode(to_tensor(_random_image(64), "cpu")).latent_dist.mode()
    assert z.shape == (1, 4, 8, 8)


def test_kl_sample_is_nondeterministic_but_mode_is_not():
    """Justifies decision D-002: .sample() would inject noise into every residual."""
    import torch as _torch

    seed_everything(2026, deterministic=True)
    vae = _tiny_kl_vae()
    x = to_tensor(_random_image(64), "cpu")
    with _torch.no_grad():
        posterior = vae.encode(x).latent_dist
        assert _torch.equal(posterior.mode(), posterior.mode())
        assert not _torch.equal(posterior.sample(), posterior.sample())


def test_bank_member_reconstruct_returns_uint8_of_the_cropped_size():
    """End-to-end through BankMember with a stand-in model: crop, dtype and shape."""
    from dbp.decoders.loaders import BankMember

    seed_everything(2026, deterministic=True)
    config = load_bank_config(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "decoder_bank.yaml"
    )
    member = BankMember(
        spec=config.by_key("sd15"),
        model=_tiny_kl_vae(),
        device=torch.device("cpu"),
        precision="fp32",
    )
    # 70x70 is not a multiple of 8, so the protocol must crop it to 64x64, not resize.
    out = member.reconstruct(_random_image(70))
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert psnr_uint8(out, out) == float("inf")


# ---------------------------------------------------------------------------
# P1-B3: the three decoder families added to the bank
#
# All three use tiny, randomly-initialised models so the new code paths are covered
# offline. The real pinned weights are exercised by the Kaggle preflight.
# ---------------------------------------------------------------------------


def _tiny_flux2_vae(latent_channels: int = 8):
    """AutoencoderKLFlux2 with the shipped `patch_size`, at a testable size."""
    from diffusers import AutoencoderKLFlux2

    seed_everything(2026, deterministic=True)
    return AutoencoderKLFlux2(
        block_out_channels=(4, 4, 8, 8),
        layers_per_block=1,
        latent_channels=latent_channels,
        norm_num_groups=2,
        sample_size=64,
    ).eval()


def _tiny_qwen_vae(z_dim: int = 4):
    """AutoencoderKLQwenImage: causal-3D encoder, so `_encode` unpacks five dims."""
    from diffusers import AutoencoderKLQwenImage

    seed_everything(2026, deterministic=True)
    return AutoencoderKLQwenImage(
        base_dim=8,
        z_dim=z_dim,
        dim_mult=[1, 2],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False],
        latents_mean=[0.0] * z_dim,
        latents_std=[1.0] * z_dim,
    ).eval()


def _tiny_vq_model():
    """VQModel with `norm_type="spatial"`, as the Kandinsky 2.2 MoVQ uses.

    SpatialNorm hard-codes 32 groups, so the block widths cannot be smaller than 32.
    """
    from diffusers import VQModel

    seed_everything(2026, deterministic=True)
    return VQModel(
        down_block_types=("DownEncoderBlock2D",) * 2,
        up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(32, 32),
        layers_per_block=1,
        latent_channels=2,
        num_vq_embeddings=16,
        norm_num_groups=32,
        vq_embed_dim=2,
        norm_type="spatial",
        sample_size=32,
    ).eval()


def test_flux2_follows_the_same_kl_contract_as_autoencoder_kl():
    """`AutoencoderKLFlux2` reuses `reconstruct_vae_kl`; this is why that is safe."""
    import torch as _torch

    from dbp.decoders.reconstruct import reconstruct_vae_kl

    vae = _tiny_flux2_vae()
    x = to_tensor(_random_image(32), "cpu")
    with _torch.no_grad():
        first = reconstruct_vae_kl(vae, x)
        second = reconstruct_vae_kl(vae, x)
    assert first.shape == x.shape
    assert _torch.equal(first, second), "mode() must make encode->decode deterministic"
    assert _torch.isfinite(first).all()


def test_flux2_patch_size_does_not_change_the_probe_downsampling():
    """`patch_size` is applied by the pipeline, not by encode(); so f8 holds, and 8 is
    the right crop multiple."""
    import torch as _torch

    vae = _tiny_flux2_vae(latent_channels=8)
    with _torch.no_grad():
        z = vae.encode(to_tensor(_random_image(32), "cpu")).latent_dist.mode()
    assert z.shape == (1, 8, 4, 4)  # 32 / 8


def test_qwen_five_dim_round_trip_preserves_size_and_is_deterministic():
    import torch as _torch

    from dbp.decoders.reconstruct import reconstruct_vae_kl_5d

    vae = _tiny_qwen_vae()
    x = to_tensor(_random_image(32), "cpu")
    with _torch.no_grad():
        first = reconstruct_vae_kl_5d(vae, x)
        second = reconstruct_vae_kl_5d(vae, x)
    assert first.shape == x.shape, "the temporal axis must be squeezed back out"
    assert _torch.equal(first, second)
    assert _torch.isfinite(first).all()


def test_qwen_helper_refuses_anything_but_a_single_nchw_frame():
    """A silently accepted 5D input would mean the caller chose the frame layout."""
    from dbp.decoders.reconstruct import reconstruct_vae_kl_5d

    vae = _tiny_qwen_vae()
    x = to_tensor(_random_image(32), "cpu").unsqueeze(2)
    with pytest.raises(ValueError, match="4D NCHW"):
        reconstruct_vae_kl_5d(vae, x)


def test_vq_round_trip_is_deterministic():
    """The codebook lookup is an argmin, so no sampling decision is involved."""
    import torch as _torch

    from dbp.decoders.reconstruct import reconstruct_vq

    model = _tiny_vq_model()
    x = to_tensor(_random_image(32), "cpu")
    with _torch.no_grad():
        first = reconstruct_vq(model, x)
        second = reconstruct_vq(model, x)
    assert first.shape == x.shape
    assert _torch.equal(first, second)
    assert _torch.isfinite(first).all()


def test_vq_decode_actually_quantises():
    """If `force_not_quantize` ever flipped, the member would stop being a
    de-tokenizer and would silently become an ordinary continuous autoencoder."""
    import torch as _torch

    model = _tiny_vq_model()
    x = to_tensor(_random_image(32), "cpu")
    with _torch.no_grad():
        h = model.encode(x).latents
        quantised = model.decode(h).sample
        straight_through = model.decode(h, force_not_quantize=True).sample
    assert not _torch.allclose(quantised, straight_through), (
        "decode() did not quantise; the VQ member would carry no lattice signature"
    )


def test_assert_tiling_disabled_rejects_a_model_with_tiling_on():
    """Tile seams are an artificial forensic trace (D-002), so this must raise."""
    from dbp.decoders.loaders import MemberLoadError, assert_tiling_disabled

    class Tiled:
        use_tiling = True

        def disable_tiling(self):
            pass  # deliberately does nothing, to prove the assertion is the guard

    with pytest.raises(MemberLoadError, match="use_tiling is still True"):
        assert_tiling_disabled(Tiled(), "fake_member")


@pytest.mark.parametrize("factory", [_tiny_qwen_vae, _tiny_flux2_vae, _tiny_kl_vae])
def test_tiling_is_off_after_the_loader_guard_even_if_it_was_enabled(factory):
    """The guard must undo tiling, not merely refrain from switching it on."""
    from dbp.decoders.loaders import assert_tiling_disabled

    model = factory()
    model.enable_tiling()
    assert model.use_tiling is True
    assert_tiling_disabled(model, "under_test")
    assert model.use_tiling is False


def test_the_guard_accepts_a_model_that_has_no_tiling_at_all():
    """`VQModel` implements no tiling: `enable_tiling()` raises and `use_tiling` does
    not exist until `disable_tiling()` creates it. The guard must still pass it."""
    from dbp.decoders.loaders import assert_tiling_disabled

    model = _tiny_vq_model()
    with pytest.raises(NotImplementedError, match="Tiling doesn't seem to be implemented"):
        model.enable_tiling()
    assert_tiling_disabled(model, "movq_like")
    assert model.use_tiling is False


def test_bank_member_dispatches_each_new_kind():
    """End-to-end through BankMember: crop, dtype, size and kind dispatch."""
    from pathlib import Path

    from dbp.decoders.loaders import BankMember

    config = load_bank_config(Path(__file__).resolve().parents[1] / "configs" / "decoder_bank.yaml")
    cases = {
        "flux2_klein": _tiny_flux2_vae,
        "qwen_image": _tiny_qwen_vae,
        "movq_k22": _tiny_vq_model,
    }
    for key, factory in cases.items():
        member = BankMember(
            spec=config.by_key(key),
            model=factory(),
            device=torch.device("cpu"),
            precision="fp32",
        )
        # 70 is not a multiple of 8, so the protocol must crop to 64 and never resize.
        out = member.reconstruct(_random_image(70))
        assert out.shape == (64, 64, 3), f"{key}: wrong output size"
        assert out.dtype == np.uint8, f"{key}: wrong dtype"
