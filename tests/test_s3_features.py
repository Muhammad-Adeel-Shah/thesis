"""Tests for the S3 decoder-bank probe features and heads (B5)."""

from __future__ import annotations

import numpy as np
import pytest

from dbp.streams.s3_probe import (
    FEATURE_NAMES,
    RADIAL_BINS,
    ProbeError,
    aeroblade_score,
    assert_permutation_invariant,
    build_set_head,
    feature_column_names,
    flatten_members,
    high_frequency_energy_ratio,
    luma,
    member_features,
    psnr_db,
    radial_log_power_spectrum,
    stack_members,
    standardise,
    standardiser,
)


def _image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_feature_vector_has_the_documented_twenty_entries() -> None:
    assert len(FEATURE_NAMES) == 3 + RADIAL_BINS + 1 == 20
    assert FEATURE_NAMES[0] == "psnr_db"
    assert FEATURE_NAMES[-1] == "hf_energy_ratio"


def test_member_features_shape_and_dtype() -> None:
    out = member_features(_image(1), _image(2), lpips_vgg=0.25)
    assert out.shape == (20,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))


def test_psnr_of_identical_images_is_capped_not_infinite() -> None:
    """inf would propagate into every downstream mean and standardisation."""
    image = _image(3)
    assert psnr_db(image, image) == 100.0
    assert np.isfinite(member_features(image, image, 0.0)).all()


def test_psnr_falls_as_the_reconstruction_worsens() -> None:
    image = _image(4)
    near = np.clip(image.astype(int) + 1, 0, 255).astype(np.uint8)
    far = np.clip(image.astype(int) + 40, 0, 255).astype(np.uint8)
    assert psnr_db(image, near) > psnr_db(image, far)


def test_luma_rejects_non_rgb() -> None:
    with pytest.raises(ProbeError, match="HWC RGB"):
        luma(np.zeros((8, 8)))


def test_radial_spectrum_has_one_value_per_bin_and_is_finite() -> None:
    plane = np.random.default_rng(0).normal(0, 5, (64, 64))
    out = radial_log_power_spectrum(plane)
    assert out.shape == (RADIAL_BINS,)
    assert np.all(np.isfinite(out))


def test_radial_spectrum_separates_low_from_high_frequency_content() -> None:
    grid = np.arange(64)
    low = np.sin(2 * np.pi * grid / 64.0)[None, :] * np.ones((64, 1))
    high = np.sin(2 * np.pi * grid / 2.5)[None, :] * np.ones((64, 1))
    low_spectrum = radial_log_power_spectrum(low)
    high_spectrum = radial_log_power_spectrum(high)
    assert low_spectrum[:2].sum() > high_spectrum[:2].sum()
    assert high_spectrum[-4:].sum() > low_spectrum[-4:].sum()


def test_radial_spectrum_rejects_a_degenerate_plane() -> None:
    with pytest.raises(ProbeError, match="too small"):
        radial_log_power_spectrum(np.zeros((1, 1)))
    with pytest.raises(ProbeError, match="2-D plane"):
        radial_log_power_spectrum(np.zeros((4, 4, 3)))


def test_hf_ratio_is_between_zero_and_one_and_ranks_correctly() -> None:
    grid = np.arange(64)
    smooth = np.sin(2 * np.pi * grid / 64.0)[None, :] * np.ones((64, 1))
    rough = np.random.default_rng(1).normal(0, 1, (64, 64))
    for plane in (smooth, rough):
        assert 0.0 <= high_frequency_energy_ratio(plane) <= 1.0
    assert high_frequency_energy_ratio(rough) > high_frequency_energy_ratio(smooth)


def test_hf_ratio_of_a_flat_plane_is_zero_not_nan() -> None:
    assert high_frequency_energy_ratio(np.zeros((32, 32))) == 0.0


def test_member_features_reject_mismatched_or_float_inputs() -> None:
    with pytest.raises(ProbeError, match="differ"):
        member_features(_image(1), _image(1, size=32), 0.1)
    with pytest.raises(ProbeError, match="uint8"):
        member_features(_image(1).astype(np.float32), _image(2), 0.1)
    with pytest.raises(ProbeError, match="finite"):
        member_features(_image(1), _image(2), float("nan"))


def test_stack_members_imposes_the_bank_order() -> None:
    per_member = {"b": np.arange(20, dtype=np.float32),
                  "a": np.ones(20, dtype=np.float32)}
    matrix = stack_members(per_member, ["a", "b"])
    assert matrix.shape == (2, 20)
    assert np.array_equal(matrix[0], np.ones(20, dtype=np.float32))


def test_stack_members_refuses_to_fill_a_missing_member_with_zeros() -> None:
    """A zero row reads to the head as a perfect reconstruction, not as missing."""
    with pytest.raises(ProbeError, match="no probe features for"):
        stack_members({"a": np.zeros(20, dtype=np.float32)}, ["a", "b"])


def test_flatten_and_column_names_agree() -> None:
    matrix = stack_members(
        {k: np.zeros(20, dtype=np.float32) for k in ("a", "b", "c")}, ["a", "b", "c"]
    )
    assert flatten_members(matrix).shape == (60,)
    names = feature_column_names(["a", "b", "c"])
    assert len(names) == 60
    assert names[0] == "a.psnr_db"
    assert names[-1] == "c.hf_energy_ratio"


def test_aeroblade_takes_the_minimum_lpips_and_negates_it() -> None:
    lpips = {"sd15": 0.4, "flux": 0.1, "dcae": 0.9}
    kinds = {"sd15": "vae_kl", "flux": "vae_kl", "dcae": "vae_dc"}
    result = aeroblade_score(lpips, kinds)
    assert result["argmin_member"] == "flux"
    assert result["score"] == pytest.approx(-0.1)
    assert result["n_members"] == 3


def test_aeroblade_excludes_the_gan_super_resolver() -> None:
    lpips = {"sd15": 0.4, "realesrgan_x4": 0.01}
    kinds = {"sd15": "vae_kl", "realesrgan_x4": "sr_gan"}
    result = aeroblade_score(lpips, kinds)
    assert result["argmin_member"] == "sd15"
    assert result["excluded"] == ["realesrgan_x4"]


def test_aeroblade_errors_when_nothing_is_eligible() -> None:
    with pytest.raises(ProbeError, match="every member was excluded"):
        aeroblade_score({"realesrgan_x4": 0.1}, {"realesrgan_x4": "sr_gan"})


def test_aeroblade_larger_means_more_likely_fake() -> None:
    """Sign convention: every stream must point the same way before fusion."""
    kinds = {"a": "vae_kl", "b": "vae_kl"}
    easy = aeroblade_score({"a": 0.02, "b": 0.5}, kinds)["score"]   # fake-like
    hard = aeroblade_score({"a": 0.40, "b": 0.5}, kinds)["score"]   # real-like
    assert easy > hard


def test_standardiser_leaves_a_constant_column_at_zero_not_inf() -> None:
    matrix = np.column_stack([np.arange(10.0), np.full(10, 3.0)])
    mean, scale = standardiser(matrix)
    out = standardise(matrix, mean, scale)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:, 1], 0.0)
    assert out[:, 0].std() == pytest.approx(1.0, rel=1e-4)


def test_standardiser_needs_more_than_one_sample() -> None:
    with pytest.raises(ProbeError, match="at least 2 samples"):
        standardiser(np.zeros((1, 20)))


def test_set_head_is_permutation_invariant() -> None:
    """Without this, adding a decoder at test time (P3) is not a defined operation."""
    assert assert_permutation_invariant(build_set_head()) <= 1e-5


def test_set_head_maps_a_batch_to_one_logit_each() -> None:
    import torch

    head = build_set_head()
    out = head(torch.randn(5, 12, len(FEATURE_NAMES)))
    assert out.shape == (5,)


def test_set_head_accepts_a_bank_of_a_different_size() -> None:
    """The shared phi has no member axis, so the member count is free."""
    import torch

    head = build_set_head()
    assert head(torch.randn(2, 13, len(FEATURE_NAMES))).shape == (2,)


def test_set_head_rejects_an_unbatched_input() -> None:
    import torch

    with pytest.raises(ProbeError, match=r"\(batch, members, features\)"):
        build_set_head()(torch.randn(12, len(FEATURE_NAMES)))


def test_the_invariance_check_catches_a_head_that_uses_member_identity() -> None:
    """The guard has to fail on a head that breaks the property, or it proves nothing."""
    import torch
    from torch import nn

    class Positional(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.out = nn.Linear(12 * len(FEATURE_NAMES), 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out(x.flatten(1)).squeeze(-1)

    with pytest.raises(ProbeError, match="not permutation invariant"):
        assert_permutation_invariant(Positional())
