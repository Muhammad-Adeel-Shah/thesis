"""Tests for the synchronized augmentation (B2).

The four the P2 plan requires are here by name: identical parameters for both members,
determinism under a fixed seed, output shape 320, and no operation ever applied to one
member only. The last one is the point of the module, so it is tested by brute force
over many draws rather than by one example.
"""

from __future__ import annotations

import numpy as np
import pytest

from dbp.data.colour import classify_colour
from dbp.train.augment import (
    CROP_SIZE,
    DEFAULTS,
    PAIR_SIZE,
    AugmentError,
    AugmentParams,
    apply,
    augment_pair,
    centre_crop,
    sample_params,
)


def _image(seed: int, size: int = PAIR_SIZE) -> np.ndarray:
    """A textured uint8 RGB image; flat images hide resize and blur differences."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    mixed = 0.5 * base.astype(np.float32) + 0.5 * ramp[None, :, None]
    return mixed.round().astype(np.uint8)


def test_output_is_320_square_uint8_rgb() -> None:
    params = sample_params(np.random.default_rng(0))
    out = apply(_image(1), params)
    assert out.shape == (CROP_SIZE, CROP_SIZE, 3)
    assert out.dtype == np.uint8


def test_both_members_receive_identical_parameters() -> None:
    real, recon = _image(1), _image(2)
    outputs, params = augment_pair([real, recon], np.random.default_rng(7))

    # The same draw applied separately must reproduce each member exactly.
    assert np.array_equal(outputs[0], apply(real, params))
    assert np.array_equal(outputs[1], apply(recon, params))


def test_deterministic_under_a_fixed_seed() -> None:
    real, recon = _image(3), _image(4)
    first, params_a = augment_pair([real, recon], np.random.default_rng(2026))
    second, params_b = augment_pair([real, recon], np.random.default_rng(2026))

    assert params_a == params_b
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)


def test_a_different_seed_gives_a_different_draw() -> None:
    draws = {sample_params(np.random.default_rng(s)) for s in range(40)}
    assert len(draws) > 1, "every seed produced the same parameters"


@pytest.mark.parametrize("seed", range(60))
def test_no_operation_is_ever_applied_to_one_member_only(seed: int) -> None:
    """The structural guarantee: one draw, so the operation set cannot differ."""
    members = [_image(10 + i) for i in range(3)]
    outputs, params = augment_pair(members, np.random.default_rng(seed))

    assert len({out.shape for out in outputs}) == 1
    # Greyscale is the one operation whose effect is visible in the output itself, so
    # it can be checked end to end rather than only through the parameter object.
    greyscale_flags = {classify_colour(out)["is_greyscale"] for out in outputs}
    if params.greyscale:
        assert greyscale_flags == {True}
    else:
        assert len(greyscale_flags) == 1


def test_the_noise_field_is_shared_not_just_its_sigma() -> None:
    """Two members get the same additive field, so their difference is unchanged."""
    params = AugmentParams(
        crop_top=0, crop_left=0, crop_size=CROP_SIZE, flip=False, resize_scale=None,
        blur_sigma=None, noise_sigma=4.0, noise_seed=99, greyscale=False,
        jpeg_quality=None, jpeg_subsampling=None,
    )
    flat_a = np.full((PAIR_SIZE, PAIR_SIZE, 3), 100, dtype=np.uint8)
    flat_b = np.full((PAIR_SIZE, PAIR_SIZE, 3), 100, dtype=np.uint8)
    assert np.array_equal(apply(flat_a, params), apply(flat_b, params))

    # Two flat images 20 levels apart stay 20 levels apart: an independent field would
    # leave a per-member signature in the difference.
    offset = np.full((PAIR_SIZE, PAIR_SIZE, 3), 120, dtype=np.uint8)
    difference = apply(offset, params).astype(int) - apply(flat_a, params).astype(int)
    assert set(np.unique(difference)) == {20}


def test_greyscale_output_passes_the_d039_rgb_equal_test() -> None:
    params = sample_params(np.random.default_rng(0), settings={"greyscale_p": 1.0, "jpeg_p": 0.0})
    assert params.greyscale
    out = apply(_image(5), params)
    verdict = classify_colour(out)
    assert verdict["is_greyscale"] is True
    assert verdict["channel_spread"] == 0


def test_jpeg_draws_both_subsamplings_over_many_samples() -> None:
    """4:2:0 must reach both classes, or E7's chroma shortcut survives training."""
    seen = set()
    for seed in range(200):
        params = sample_params(np.random.default_rng(seed), settings={"jpeg_p": 1.0})
        seen.add(params.jpeg_subsampling)
    assert seen == set(DEFAULTS["jpeg_subsamplings"])


def test_probabilities_are_within_tolerance_of_the_approved_rates() -> None:
    draws = [sample_params(np.random.default_rng(s)) for s in range(4000)]
    rates = {
        "flip_p": sum(d.flip for d in draws) / len(draws),
        "jpeg_p": sum(d.jpeg_quality is not None for d in draws) / len(draws),
        "blur_p": sum(d.blur_sigma is not None for d in draws) / len(draws),
        "noise_p": sum(d.noise_sigma is not None for d in draws) / len(draws),
        "resize_p": sum(d.resize_scale is not None for d in draws) / len(draws),
        "greyscale_p": sum(d.greyscale for d in draws) / len(draws),
    }
    for name, observed in rates.items():
        assert abs(observed - DEFAULTS[name]) < 0.025, (name, observed)


def test_sampled_values_stay_inside_the_approved_ranges() -> None:
    for seed in range(300):
        params = sample_params(np.random.default_rng(seed))
        assert 0 <= params.crop_top <= PAIR_SIZE - CROP_SIZE
        assert 0 <= params.crop_left <= PAIR_SIZE - CROP_SIZE
        if params.jpeg_quality is not None:
            assert 60 <= params.jpeg_quality <= 100
        if params.blur_sigma is not None:
            assert 0.1 <= params.blur_sigma <= 1.5
        if params.noise_sigma is not None:
            assert 1.0 <= params.noise_sigma <= 5.0
        if params.resize_scale is not None:
            assert 0.75 <= params.resize_scale <= 1.0


def test_crop_larger_than_the_image_is_an_error() -> None:
    with pytest.raises(AugmentError, match="does not fit"):
        sample_params(np.random.default_rng(0), image_size=256, crop_size=320)


def test_augment_pair_refuses_a_single_member() -> None:
    with pytest.raises(AugmentError, match="at least two members"):
        augment_pair([_image(1)], np.random.default_rng(0))


def test_apply_rejects_non_uint8_and_non_rgb() -> None:
    params = sample_params(np.random.default_rng(0))
    with pytest.raises(AugmentError, match="uint8"):
        apply(_image(1).astype(np.float32), params)
    with pytest.raises(AugmentError, match="HWC RGB"):
        apply(_image(1)[:, :, 0], params)


def test_centre_crop_is_fixed_and_central() -> None:
    image = _image(6)
    out = centre_crop(image)
    assert out.shape == (CROP_SIZE, CROP_SIZE, 3)
    offset = (PAIR_SIZE - CROP_SIZE) // 2
    assert np.array_equal(
        out, image[offset : offset + CROP_SIZE, offset : offset + CROP_SIZE, :]
    )
    with pytest.raises(AugmentError, match="smaller than"):
        centre_crop(_image(6, size=128))


def test_operations_property_lists_what_was_applied() -> None:
    params = sample_params(
        np.random.default_rng(0),
        settings={"flip_p": 1.0, "jpeg_p": 1.0, "blur_p": 0.0, "noise_p": 0.0,
                  "resize_p": 0.0, "greyscale_p": 0.0},
    )
    assert params.operations == ("flip", "jpeg")
