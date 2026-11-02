"""Offline backbone and LoRA tests.

These use randomly-initialised, tiny DINOv2/DINOv3 configs so the attach logic is
covered in any runtime, including one with no network and no GPU. The real pretrained
backbones are measured by ``scripts/p0_backbone_smoke_test.py`` on the T4.

The LoRA attach is the most fragile part of P2: DINOv2 and DINOv3 do not name their
attention projections the same way, so ``target_modules`` must be discovered from the
loaded model and never hard-coded.
"""

from __future__ import annotations

import pytest
import torch

from dbp.backbones.loader import (
    BACKBONES,
    DEFAULT_INPUT_SIZE,
    BackboneLoadError,
    find_attention_projection_names,
    forward_once,
    lora_dry_run,
)
from dbp.utils.seed import seed_everything


def _tiny_dinov2(patch_size: int = 14, image_size: int = 70):
    from transformers import Dinov2Config, Dinov2Model

    config = Dinov2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=patch_size,
        image_size=image_size,
    )
    return Dinov2Model(config).eval()


def test_default_input_size_is_divisible_by_both_patch_sizes():
    """336 lets DINOv2 (14) and DINOv3 (16) be compared without resizing differently."""
    assert DEFAULT_INPUT_SIZE % 14 == 0
    assert DEFAULT_INPUT_SIZE % 16 == 0


def test_backbone_registry_pins_full_commit_shas():
    for name, entry in BACKBONES.items():
        assert len(entry["revision"]) == 40, f"{name}: revision is not a full commit SHA"
        int(entry["revision"], 16)


def test_exactly_one_backbone_is_the_default():
    assert sum(1 for e in BACKBONES.values() if e["default"]) == 1


def test_unknown_backbone_name_raises():
    with pytest.raises(BackboneLoadError, match="unknown backbone"):
        from dbp.backbones.loader import load_backbone

        load_backbone("resnet50")


def test_attention_projections_are_discovered_not_assumed():
    """DINOv2 uses separate query/key/value Linear layers."""
    found = find_attention_projection_names(_tiny_dinov2())
    assert set(found) == {"query", "key", "value"}


def test_forward_reports_expected_token_count():
    seed_everything(2026, deterministic=True)
    model = _tiny_dinov2(patch_size=14, image_size=70)
    result = forward_once(model, size=70, device="cpu")
    expected_patches = (70 // 14) ** 2
    assert result["num_tokens"] == expected_patches + 1  # + CLS
    assert result["hidden_size"] == 32
    assert result["peak_gpu_mb"] is None  # CPU
    assert result["fp16_autocast"] is False


def test_lora_dry_run_attaches_and_reports_trainable_params():
    seed_everything(2026, deterministic=True)
    model = _tiny_dinov2()
    result = lora_dry_run(model, r=8, lora_alpha=8, size=70, device="cpu")

    assert set(result["target_modules_found"]) == {"query", "key", "value"}
    assert result["params_trainable"] > 0
    assert result["params_total_after"] > result["params_total_before"]
    # LoRA must train only the adapters, never the whole backbone.
    assert result["params_trainable"] < result["params_total_after"]
    assert result["forward_after_attach"]["num_tokens"] == (70 // 14) ** 2 + 1


def test_lora_dry_run_does_not_create_an_optimizer_or_train():
    """P0 attaches adapters only. Training hyperparameters are decided in P2."""
    seed_everything(2026, deterministic=True)
    model = _tiny_dinov2()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = lora_dry_run(model, r=8, lora_alpha=8, size=70, device="cpu")

    assert result["note"] == "dry-run only: no optimizer, no training step"
    for key, value in before.items():
        current = model.state_dict().get(key)
        if current is not None and current.shape == value.shape:
            assert torch.equal(current, value), f"{key} changed during a dry-run"


def test_lora_refuses_to_guess_when_no_projections_exist():
    """Silently attaching to nothing would produce an untrainable model in P2."""

    class NoAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)

    with pytest.raises(BackboneLoadError, match="refusing to guess"):
        lora_dry_run(NoAttention(), r=8, size=70, device="cpu")
