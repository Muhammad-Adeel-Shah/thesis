"""CPU tests for the S1 and S2 stream helpers (B3, B4).

Nothing here downloads a backbone. The parts that can be wrong in a way that still
produces plausible numbers -- the token layout, the register-token exclusion, the LoRA
target discovery, the step budget -- are all pure and are tested here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from dbp.streams.s1_semantic import (
    PROBE_C_GRID,
    SemanticStreamError,
    expected_patch_tokens,
    fit_probe,
    normalise_batch,
    pool_tokens,
    register_token_count,
)
from dbp.streams.s2_artifact import (
    MIN_STEPS_PER_RUN,
    ArtifactStreamError,
    build_head,
    discover_attention_projections,
    pair_batch_plan,
    steps_within_allowance,
)

HIDDEN = 8


# --------------------------------------------------------------------- S1 ---


def test_register_count_is_read_from_the_config() -> None:
    assert register_token_count({"num_register_tokens": 4}) == 4
    assert register_token_count({"num_register_tokens": 0}) == 0


def test_register_count_reads_an_object_config_too() -> None:
    class Cfg:
        num_register_tokens = 4

    assert register_token_count(Cfg()) == 4


def test_a_missing_register_count_is_an_error_not_a_zero() -> None:
    """Defaulting to 0 would average DINOv3's four register tokens into the patch mean."""
    with pytest.raises(SemanticStreamError, match="num_register_tokens"):
        register_token_count({"hidden_size": 768})


def test_expected_patch_tokens_matches_the_measured_dinov3_layout() -> None:
    # P0 measured 446 tokens at input 336: 1 CLS + 4 register + 441 patches.
    assert expected_patch_tokens(336, 16) == 441
    assert 1 + 4 + expected_patch_tokens(336, 16) == 446
    # And the P2 input size.
    assert expected_patch_tokens(320, 16) == 400


def test_expected_patch_tokens_rejects_a_non_multiple() -> None:
    with pytest.raises(SemanticStreamError, match="not a multiple"):
        expected_patch_tokens(330, 16)


def test_pool_excludes_the_register_tokens_from_the_patch_mean() -> None:
    n_register, n_patch = 4, 6
    states = torch.zeros(1, 1 + n_register + n_patch, HIDDEN)
    states[:, 0, :] = 1.0                       # CLS
    states[:, 1 : 1 + n_register, :] = 1000.0   # registers: must not be averaged in
    states[:, 1 + n_register :, :] = 2.0        # patches
    pooled = pool_tokens(states, n_register, n_patch)
    assert pooled.shape == (1, 2 * HIDDEN)
    assert torch.allclose(pooled[:, :HIDDEN], torch.ones(1, HIDDEN))
    assert torch.allclose(pooled[:, HIDDEN:], torch.full((1, HIDDEN), 2.0))


def test_pool_refuses_a_token_count_that_does_not_match_the_layout() -> None:
    with pytest.raises(SemanticStreamError, match="Refusing to guess"):
        pool_tokens(torch.zeros(1, 446, HIDDEN), num_register_tokens=0, num_patch_tokens=441)


def test_pool_rejects_an_unbatched_input() -> None:
    with pytest.raises(SemanticStreamError, match=r"\(batch, tokens, hidden\)"):
        pool_tokens(torch.zeros(446, HIDDEN), 4, 441)


def test_normalise_batch_shape_and_values() -> None:
    images = np.full((2, 4, 4, 3), 255, dtype=np.uint8)
    out = normalise_batch(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    assert out.shape == (2, 3, 4, 4)
    assert torch.allclose(out, torch.ones_like(out))


def test_normalise_batch_rejects_float_input() -> None:
    with pytest.raises(SemanticStreamError, match="uint8"):
        normalise_batch(np.zeros((1, 4, 4, 3), dtype=np.float32), [0.5] * 3, [0.5] * 3)


def test_normalise_batch_rejects_the_wrong_shape() -> None:
    with pytest.raises(SemanticStreamError, match=r"\(n, H, W, 3\)"):
        normalise_batch(np.zeros((4, 4, 3), dtype=np.uint8), [0.5] * 3, [0.5] * 3)


def _separable(n: int, d: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    x = rng.normal(0, 1, (n, d)) + y[:, None] * 0.8
    return x.astype(np.float32), y


def test_probe_picks_c_on_validation_and_reports_the_whole_grid() -> None:
    x_tr, y_tr = _separable(600, 12, 0)
    x_va, y_va = _separable(300, 12, 1)
    result = fit_probe(x_tr, y_tr, x_va, y_va)
    assert result["C"] in PROBE_C_GRID
    assert len(result["grid"]) == len(PROBE_C_GRID)
    assert result["val_auc"] == max(entry["val_auc"] for entry in result["grid"])
    assert result["val_auc"] > 0.8
    assert result["n_train"] == 600 and result["n_val"] == 300


def test_probe_rejects_feature_widths_that_disagree() -> None:
    x_tr, y_tr = _separable(50, 12, 0)
    x_va, y_va = _separable(50, 8, 1)
    with pytest.raises(SemanticStreamError, match="different backbones or crops"):
        fit_probe(x_tr, y_tr, x_va, y_va)


def test_probe_rejects_a_single_class_split() -> None:
    x_tr, y_tr = _separable(50, 6, 0)
    with pytest.raises(SemanticStreamError, match="single class"):
        fit_probe(x_tr, y_tr, x_tr, np.ones(50, dtype=int))


# --------------------------------------------------------------------- S2 ---


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _FakeAttention()
        self.mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN))


def test_lora_targets_are_discovered_from_the_model() -> None:
    assert discover_attention_projections(_FakeBackbone()) == ["k_proj", "q_proj", "v_proj"]


def test_discovery_errors_rather_than_returning_an_empty_target_list() -> None:
    """An empty target list would make S2 silently collapse into S1."""

    class NoAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(HIDDEN, HIDDEN)

    with pytest.raises(ArtifactStreamError, match="collapse into S1"):
        discover_attention_projections(NoAttention())


def test_head_maps_pooled_features_to_one_logit() -> None:
    head = build_head(HIDDEN)
    assert head(torch.randn(5, 2 * HIDDEN)).shape == (5,)


def test_step_budget_is_the_allowance_divided_by_the_measured_step() -> None:
    budget = steps_within_allowance(allowance_hours=2.0, measured_step_seconds=1.8)
    assert budget.steps == 4000
    assert budget.undertrained is False


def test_step_budget_never_exceeds_the_plan_ceiling() -> None:
    budget = steps_within_allowance(allowance_hours=10.0, measured_step_seconds=0.05, ceiling=8000)
    assert budget.steps == 8000


def test_step_budget_flags_an_undertrained_run() -> None:
    """The approved guard: fewer than 2,000 steps invalidates the comparison."""
    budget = steps_within_allowance(allowance_hours=1.0, measured_step_seconds=3.0)
    assert budget.steps < MIN_STEPS_PER_RUN
    assert budget.undertrained is True
    assert budget.to_dict()["min_steps_per_run"] == MIN_STEPS_PER_RUN


def test_step_budget_rejects_a_missing_measurement() -> None:
    with pytest.raises(ArtifactStreamError, match="did not produce a real measurement"):
        steps_within_allowance(2.0, 0.0)
    with pytest.raises(ArtifactStreamError, match="allowance_hours must be positive"):
        steps_within_allowance(0.0, 1.0)


def test_batch_of_sixteen_is_eight_pairs_with_the_gen_matched_share() -> None:
    plan = pair_batch_plan(16, gen_matched_share=0.25)
    assert plan == {
        "pairs": 6, "reals": 8, "reconstruction_fakes": 6, "gen_matched_fakes": 2,
    }
    assert plan["reals"] == plan["reconstruction_fakes"] + plan["gen_matched_fakes"]


def test_batch_plan_without_gen_matched_is_all_pairs() -> None:
    assert pair_batch_plan(16, gen_matched_share=0.0)["pairs"] == 8


def test_batch_plan_rejects_an_odd_batch() -> None:
    with pytest.raises(ArtifactStreamError, match="even number"):
        pair_batch_plan(15)


def test_batch_plan_rejects_an_impossible_share() -> None:
    with pytest.raises(ArtifactStreamError, match=r"\[0, 1\)"):
        pair_batch_plan(16, gen_matched_share=1.0)
