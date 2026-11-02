"""GPU-only and gated-access smoke tests.

Every test here is skipped with an explicit reason when the resource is unavailable,
so a skip is never mistaken for a pass. On the audited local machine there is no CUDA
device and DINOv3 access is pending, so all of these report as skipped with the
reason printed by ``pytest -rs``.

Run on Kaggle T4 with:  pytest -q -rs tests/test_gpu_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dbp.decoders.registry import load_bank_config
from dbp.utils.hf import get_hf_token

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "decoder_bank.yaml"

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="no CUDA device in this runtime; run on Kaggle T4 (PENDING-GPU)",
)
requires_token = pytest.mark.skipif(
    get_hf_token() is None,
    reason="no HF_TOKEN in this runtime; gated members are BLOCKED-ACCESS",
)

RNG = np.random.default_rng(2026)


@pytest.mark.gpu
@requires_gpu
def test_cuda_device_is_turing_or_newer():
    """PyTorch CUDA-13 builds dropped pre-Turing; a P100 would fail silently later [R19]."""
    major, minor = torch.cuda.get_device_capability(0)
    sm = major * 10 + minor
    assert sm >= 70, f"{torch.cuda.get_device_name(0)} is sm_{sm}; use a T4 (sm_75)"


@pytest.mark.gpu
@requires_gpu
def test_t4_has_fp16_but_no_native_bf16():
    """The precision policy is fp16 autocast, never bf16, because of this [R21].

    ``torch.cuda.is_bf16_supported()`` defaults to ``including_emulation=True`` and
    therefore returns True on a T4: bf16 *tensors* can be created, they just run without
    tensor-core acceleration. That is not the question the precision policy asks, so the
    native-support form is used instead. Measured on a T4 (sm_75, CUDA 12.8):
    ``is_bf16_supported()`` is True while ``is_bf16_supported(False)`` is False, which is
    precisely why bf16 is excluded.
    """
    assert torch.cuda.is_available()
    major, _ = torch.cuda.get_device_capability(0)
    if major >= 8:
        pytest.skip(f"sm_{major}0+ has native bf16; this test is about pre-Ampere GPUs")

    try:
        native = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # torch too old for the keyword
        native = False
    assert not native, "unexpected NATIVE bf16 on a pre-Ampere GPU"


@pytest.mark.gpu
@requires_gpu
def test_sd15_reconstructs_on_gpu_and_reports_peak_memory():
    from dbp.decoders.loaders import load_member
    from dbp.decoders.reconstruct import max_abs_diff

    spec = load_bank_config(CONFIG_PATH).by_key("sd15")
    torch.cuda.reset_peak_memory_stats()
    member = load_member(spec, device="cuda", precision="fp32")

    img = RNG.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    first = member.reconstruct(img)
    second = member.reconstruct(img)

    assert first.shape == (512, 512, 3)
    assert max_abs_diff(first, second) <= 1
    assert torch.cuda.max_memory_allocated() > 0
    member.free()


@pytest.mark.gpu
@requires_gpu
def test_tiling_is_never_enabled():
    """Tile seams are an artificial forensic trace; the loader must leave tiling off."""
    from dbp.decoders.loaders import load_member

    spec = load_bank_config(CONFIG_PATH).by_key("sd15")
    member = load_member(spec, device="cuda", precision="fp32")
    assert getattr(member.model, "use_tiling", False) is False
    member.free()


@pytest.mark.gpu
@pytest.mark.gated
@requires_gpu
@requires_token
@pytest.mark.parametrize("key", ["sd35", "flux"])
def test_sixteen_channel_members_load_and_reconstruct(key):
    """Both 16-channel lineages are the core claim; they must work end to end."""
    from dbp.decoders.loaders import load_member

    spec = load_bank_config(CONFIG_PATH).by_key(key)
    assert spec.latent_channels == 16

    member = load_member(spec, device="cuda", precision="fp32")
    img = RNG.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    out = member.reconstruct(img)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8
    member.free()


@pytest.mark.gpu
@requires_gpu
def test_backbone_forward_under_fp16_autocast():
    from dbp.backbones.loader import forward_once, load_backbone

    backbone = load_backbone("dinov2-base", device="cuda", token=get_hf_token())
    result = forward_once(backbone.model, size=336, device="cuda", use_fp16_autocast=True)
    assert result["fp16_autocast"] is True
    assert result["hidden_size"] == backbone.hidden_size
    assert result["peak_gpu_mb"] is not None and result["peak_gpu_mb"] > 0


@pytest.mark.gated
@requires_token
def test_dinov3_access_status_is_explicit():
    """DINOv3 is gated. Record the outcome rather than letting it fail obscurely later."""
    from dbp.utils.hf import check_access

    status = check_access("facebook/dinov3-vitb16-pretrain-lvd1689m")
    assert status["reachable"] is True
    if not status["access_granted"]:
        pytest.skip(f"DINOv3 BLOCKED-ACCESS: {status['reason']}")
