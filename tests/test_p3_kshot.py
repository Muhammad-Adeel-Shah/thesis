"""The H3-k methods reproduce the numbers frozen in pre-registration section 6.3."""

from __future__ import annotations

import numpy as np
import pytest

from dbp.eval import kshot


@pytest.mark.parametrize(("k", "j", "met", "probability"), [
    (50, 1, True, 0.0203),    # "k = 50 -> j = 1 (P = 0.0203; j = 2 would give 0.1025)"
    (100, 4, True, 0.0526),   # "k = 100 -> j = 4 (P = 0.0526)"
    (10, 1, False, 0.4586),   # "no j meets the bound (j = 1 gives 0.4586 ...)"
    (25, 1, False, 0.1424),   # "... and 0.1424), so j = 1 is used, not met"
])
def test_the_quantile_rule_reproduces_the_preregistration(k: int, j: int, met: bool,
                                                           probability: float) -> None:
    rule = kshot.quantile_rule(k)
    assert (rule.j, rule.bound_met) == (j, met)
    assert round(rule.probability, 4) == probability


def test_j_plus_one_breaks_the_bound_at_k_50() -> None:
    from scipy.stats import binom
    assert round(float(binom.cdf(1, 50, 0.075)), 4) == 0.1025


def test_the_quantile_threshold_is_the_jth_highest() -> None:
    scores = np.arange(100, dtype=float)
    threshold, rule = kshot.quantile_threshold(scores)
    assert rule.j == 4 and threshold == 96.0
    assert kshot.achieved_fpr(scores, threshold) == 0.04


def test_the_logit_shift_moves_the_calibration_median_onto_the_validation_median() -> None:
    shift = kshot.logit_shift([3.0, 5.0, 7.0], [0.0, 1.0, 2.0])
    assert shift == 4.0
    assert kshot.achieved_fpr([5.0, 6.0], threshold=1.5, shift=shift) == 0.5


def test_draws_are_seeded_per_pool_and_k_and_single_at_the_whole_pool() -> None:
    a = kshot.draws("e5_chameleon_pipeline", 500, 50)
    b = kshot.draws("e5_chameleon_pipeline", 500, 50)
    assert len(a) == kshot.DRAWS and all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert all(len(set(d)) == 50 for d in a)
    other = kshot.draws("e4_wildrf_matched", 500, 50)
    assert not all(np.array_equal(x, y) for x, y in zip(a, other, strict=True))
    whole = kshot.draws("e4_wildrf_matched", 100, 100)
    assert len(whole) == 1 and np.array_equal(whole[0], np.arange(100))
    with pytest.raises(kshot.KShotError):
        kshot.draws("e3_matched", 50, 51)
