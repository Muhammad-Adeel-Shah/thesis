"""Byte-level regression guard for the reconstruction protocol.

P1 extends ``reconstruct.py`` and ``loaders.py`` with three new decoder families. The
risk that matters is not that the new branches are wrong -- they are tested directly --
but that touching the shared code path silently changes what the *existing* members
produce, which would invalidate every P0 number and every pair built afterwards.

The digests below were captured from the pre-P1 code, on this file's fixtures, before
``registry.py``, ``reconstruct.py`` and ``loaders.py`` were edited (P1-B3). They are
not a statement about the real pretrained weights: they pin the protocol's arithmetic
(crop, normalise, encode, decode, clamp, round, quantise) using small randomly
initialised models, so the whole path is covered with no download and no GPU.

The complementary check against the real weights runs on the T4:
``p0_bank_smoke_test.py --compare-to reports/kaggle_20260920/p0_bank_smoke.csv``
re-measures PSNR/LPIPS/mean-abs-residual for the ten P0 members and fails on any
difference.

If a digest here changes, the reconstruction protocol changed. That may be correct,
but it is never incidental: update the digest in the same commit that changes the
protocol, and say so in the implementation log.

**The digests are only valid in the environment that captured them.** Measured on
Kaggle (diffusers 0.37.1, torch 2.10.0+cu128) the same code produces different bytes,
and locally, changing only ``torch.get_num_threads()`` from 4 to 1 moves the ``dcae``
digest. Neither is a protocol change: a byte-exact digest of a CPU float pipeline
depends on the library version that builds the model and on the kernels the runtime
picks. So the digests are looked up by ``(diffusers, torch, threads)`` and the test
**skips, with the reason printed,** in any environment that has no captured entry.
That is deliberately narrow. A skip here is not "no regression": on other machines the
guarantee comes from ``p0_bank_smoke_test.py --compare-to``, which compares real
pretrained weights against numbers measured on the same hardware, and which passed on
Kaggle over 20 (member, crop) pairs. Adding a second captured environment is a
deliberate act -- measure it, show the values, and record them here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from dbp.decoders.loaders import BankMember
from dbp.decoders.reconstruct import prepare_input, to_tensor, to_uint8
from dbp.decoders.registry import load_bank_config
from dbp.utils.seed import seed_everything

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decoder_bank.yaml"

def current_environment() -> tuple[str, str, int]:
    """The three things a digest turns out to depend on: diffusers, torch, threads."""
    import diffusers

    return (diffusers.__version__, torch.__version__, torch.get_num_threads())


#: The environment the digests below were captured in.
CAPTURE_ENVIRONMENT: tuple[str, str, int] = ("0.40.0", "2.11.0+cpu", 4)

#: Captured 2026-09-20 from the pre-P1 code (commit bd3f61c) by running this file's
#: fixtures through ``BankMember.reconstruct``, in :data:`CAPTURE_ENVIRONMENT`. The
#: values are unchanged since that capture; only the lookup is keyed.
GOLDEN_BY_ENVIRONMENT: dict[tuple[str, str, int], dict[str, str]] = {
    CAPTURE_ENVIRONMENT: {
        "sd15": "4b07486d80ad0a805fc4c7327c618829b5fc2c82be13fe3db3e77dd66ac3a8a7",
        "sd35": "6ae52a2c64f94aeb0783daa72f1856ebfa6d136412c79045c3c494d75fb5b551",
        "dcae": "77a51a0d1547465689693dabad49e971dee70a305d474986e82e8e1a54f68e42",
    },
}

#: The member keys the parametrisation runs over, and the digests used when the running
#: environment is the capture environment.
GOLDEN_RECONSTRUCTIONS: dict[str, str] = GOLDEN_BY_ENVIRONMENT[CAPTURE_ENVIRONMENT]

#: Same capture, for the model-free parts of the protocol.
GOLDEN_HELPERS: dict[str, str] = {
    "prepare_input_8": "1cdfa6672893c53a0f4fdc75d8e3b5ef90fd0eb9b29c922ea89970f26688ce84",
    "prepare_input_32": "1cdfa6672893c53a0f4fdc75d8e3b5ef90fd0eb9b29c922ea89970f26688ce84",
    "prepare_input_4": "0769ff15e3206e1c73eeb5fb54e28887f5909eb90731c00f3f590c6b8d27d974",
    "to_uint8_roundtrip": "8ae349738655bd0f0869a1467e4073cd471939820e8f8a5397cb795ed3004b94",
    "to_uint8_unit_range": "8ae349738655bd0f0869a1467e4073cd471939820e8f8a5397cb795ed3004b94",
}


def _image(size: int = 96) -> np.ndarray:
    """A fixed pseudo-random image. 96 is a multiple of both 8 and 32."""
    return np.random.default_rng(20260920).integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _tiny_kl(latent_channels: int):
    from diffusers import AutoencoderKL

    seed_everything(2026, deterministic=True)
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


def _tiny_dc():
    from diffusers import AutoencoderDC

    seed_everything(2026, deterministic=True)
    return AutoencoderDC(
        in_channels=3,
        latent_channels=8,
        attention_head_dim=4,
        encoder_block_out_channels=(4, 4, 8, 8, 8, 8),
        decoder_block_out_channels=(4, 4, 8, 8, 8, 8),
        encoder_layers_per_block=(1,) * 6,
        decoder_layers_per_block=(1,) * 6,
        encoder_qkv_multiscales=((),) * 6,
        decoder_qkv_multiscales=((),) * 6,
    ).eval()


MODEL_FACTORIES = {
    "sd15": lambda: _tiny_kl(4),
    "sd35": lambda: _tiny_kl(16),
    "dcae": _tiny_dc,
}


@pytest.fixture(scope="module")
def config():
    return load_bank_config(CONFIG_PATH)


@pytest.mark.parametrize("member_key", sorted(GOLDEN_RECONSTRUCTIONS))
def test_existing_members_reconstruct_bit_identically(member_key, config):
    """The P1 refactor must not move a single bit on the pre-existing kinds."""
    environment = current_environment()
    expected = GOLDEN_BY_ENVIRONMENT.get(environment)
    if expected is None:
        pytest.skip(
            f"no digests captured for (diffusers, torch, threads) = {environment}; "
            f"captured: {sorted(GOLDEN_BY_ENVIRONMENT)}. These digests are not portable "
            f"across library versions or thread counts, so a failure here would say "
            f"nothing about the protocol. The cross-environment guard is "
            f"p0_bank_smoke_test.py --compare-to on real weights."
        )
    member = BankMember(
        spec=config.by_key(member_key),
        model=MODEL_FACTORIES[member_key](),
        device=torch.device("cpu"),
        precision="fp32",
    )
    recon = member.reconstruct(_image())
    assert recon.shape == (96, 96, 3)
    assert recon.dtype == np.uint8
    assert _sha(recon) == expected[member_key], (
        f"{member_key}: the reconstruction changed. If that was intended, update the "
        f"digest and record the protocol change in the implementation log."
    )


def test_protocol_helpers_are_unchanged():
    """Crop, normalise and quantise are shared by every member, old and new."""
    measured = {
        "prepare_input_8": _sha(prepare_input(_image(70), 8)),
        "prepare_input_32": _sha(prepare_input(_image(70), 32)),
        "prepare_input_4": _sha(prepare_input(_image(70), 4)),
        "to_uint8_roundtrip": _sha(to_uint8(to_tensor(_image(), "cpu"))),
        "to_uint8_unit_range": _sha(
            to_uint8(to_tensor(_image(), "cpu", value_range=(0.0, 1.0)), value_range=(0.0, 1.0))
        ),
    }
    assert measured == GOLDEN_HELPERS


def test_every_pre_p1_member_still_resolves_to_its_original_lineage(config):
    """Adding kinds must not reshuffle the leave-one-lineage-out groups (D-016)."""
    expected = {
        "sd15": "kl_4ch_sd1",
        "sd21": "kl_4ch_sd1",
        "sd_ft_ema": "kl_4ch_sd1",
        "sd_ft_mse": "kl_4ch_sd1",
        "sdxl": "kl_4ch_sdxl",
        "sdxl_fp16fix": "kl_4ch_sdxl",
        "sd35": "kl_16ch",
        "flux": "kl_16ch",
        "dcae": "dc_ae_f32",
        "realesrgan_x4": "gan_sr",
    }
    assert {key: config.by_key(key).lineage for key in expected} == expected


def test_every_pre_p1_member_keeps_its_pinned_revision(config):
    """A pin that moved would silently change every number measured against it."""
    expected = {
        "sd15": "451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
        "sd21": "bb2154823665391b4fb29b0b9cf82a198964ee05",
        "sd_ft_ema": "f04b2c4b98319346dad8c65879f680b1997b204a",
        "sd_ft_mse": "31f26fdeee1355a5c34592e401dd41e45d25a493",
        "sdxl": "6f5909a7e596173e25d4e97b07fd19cdf9611c76",
        "sdxl_fp16fix": "207b116dae70ace3637169f1ddd2434b91b3a8cd",
        "sd35": "ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f",
        "flux": "741f7c3ce8b383c54771c7003378a50191e9efe9",
        "dcae": "ca69e17e97609e64ce055115a6515215109b1f50",
    }
    assert {key: config.by_key(key).revision for key in expected} == expected
    assert config.by_key("realesrgan_x4").sha256 == (
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
    )
