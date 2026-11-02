"""Tests for the RQ1 complementarity measures (B8)."""

from __future__ import annotations

import numpy as np
import pytest

from dbp.eval.complementarity import (
    disagreement_rate,
    double_fault_rate,
    error_contingency,
    expected_calibration_error,
    mcnemar,
    oracle_correct,
    paired_difference_ci,
    pairwise,
    score_correlation,
    stratified_bootstrap_indices,
    yules_q,
)
from dbp.eval.metrics import MetricError


def test_contingency_counts_the_four_cells() -> None:
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    assert error_contingency(a, b) == {
        "both_right": 1, "a_only": 2, "b_only": 1, "both_wrong": 1, "n": 5,
    }


def test_contingency_rejects_mismatched_lengths() -> None:
    with pytest.raises(MetricError, match="differ"):
        error_contingency([True, False], [True])


def test_contingency_rejects_empty_input() -> None:
    with pytest.raises(MetricError, match="no samples"):
        error_contingency([], [])


def test_double_fault_and_disagreement_are_shares_of_n() -> None:
    table = error_contingency([1, 1, 0, 0], [1, 0, 1, 0])
    assert double_fault_rate(table) == 0.25
    assert disagreement_rate(table) == 0.5


def test_identical_streams_have_no_disagreement_and_q_of_one() -> None:
    a = [True, True, False, False]
    table = error_contingency(a, a)
    assert disagreement_rate(table) == 0.0
    assert yules_q(table) == 1.0


def test_perfectly_complementary_streams_give_q_of_minus_one() -> None:
    table = error_contingency([True, False], [False, True])
    assert yules_q(table) == -1.0


def test_independent_errors_give_q_near_zero() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(4000) < 0.7
    b = rng.random(4000) < 0.7
    q = yules_q(error_contingency(a, b))
    assert q is not None and abs(q) < 0.1


def test_yules_q_is_none_when_undefined_not_zero() -> None:
    """A perfect stream makes Q undefined; 0.0 would read as 'independent'."""
    table = error_contingency([True, True, True], [True, False, True])
    # Stream A never errs, so both_wrong and b_only are both 0 and ad + bc vanishes.
    assert table["both_wrong"] == 0 and table["b_only"] == 0
    assert yules_q(table) is None


def test_mcnemar_uses_the_exact_test_on_few_discordant_pairs() -> None:
    a = [True] * 90 + [True] * 8 + [False] * 2
    b = [True] * 90 + [False] * 8 + [True] * 2
    result = mcnemar(error_contingency(a, b))
    assert result["test"] == "exact binomial"
    assert result["discordant"] == 10
    assert 0.0 <= result["p_value"] <= 1.0


def test_mcnemar_uses_chi_square_on_many_discordant_pairs() -> None:
    a = [True] * 200 + [True] * 80 + [False] * 20
    b = [True] * 200 + [False] * 80 + [True] * 20
    result = mcnemar(error_contingency(a, b))
    assert result["test"] == "chi-square, continuity corrected"
    assert result["discordant"] == 100
    assert result["p_value"] < 0.001


def test_mcnemar_reports_undefined_rather_than_a_p_value() -> None:
    a = [True, False, True]
    result = mcnemar(error_contingency(a, a))
    assert result["test"] == "none"
    assert result["p_value"] is None


def test_mcnemar_is_symmetric_in_magnitude() -> None:
    a = [True] * 50 + [True] * 30 + [False] * 10
    b = [True] * 50 + [False] * 30 + [True] * 10
    forward = mcnemar(error_contingency(a, b))
    backward = mcnemar(error_contingency(b, a))
    assert forward["p_value"] == pytest.approx(backward["p_value"])


def test_spearman_is_one_for_a_monotone_transform() -> None:
    values = np.linspace(0.1, 5.0, 60)
    result = score_correlation(values, np.exp(values))
    assert result["spearman"] == pytest.approx(1.0)


def test_spearman_needs_three_samples() -> None:
    with pytest.raises(MetricError, match="at least 3"):
        score_correlation([1.0, 2.0], [1.0, 2.0])


def test_oracle_is_right_whenever_either_stream_is() -> None:
    assert list(oracle_correct([True, False, False], [False, True, False])) == [
        True, True, False,
    ]


def test_pairwise_covers_every_unordered_pair() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    scores = {name: rng.random(300) for name in ("s1", "s2", "s3")}
    result = pairwise(y, scores, {"s1": 0.5, "s2": 0.5, "s3": 0.5})
    assert set(result) == {"s1|s2", "s1|s3", "s2|s3"}
    for entry in result.values():
        assert entry["oracle_headroom"] >= -1e-12
        assert 0.0 <= entry["double_fault"] <= 1.0


def test_pairwise_refuses_a_stream_without_a_validation_threshold() -> None:
    """Silently substituting 0.5 would replace a calibrated decision with a naive one."""
    with pytest.raises(MetricError, match="no validation threshold"):
        pairwise([0, 1], {"s1": [0.1, 0.9], "s2": [0.2, 0.8]}, {"s1": 0.5})


def test_bootstrap_keeps_every_stratum_at_its_own_size() -> None:
    strata = ["e7"] * 40 + ["e8"] * 10 + ["e5"] * 5
    for index in stratified_bootstrap_indices(strata, resamples=20, seed=3):
        labels = np.asarray(strata)[index]
        assert sum(labels == "e7") == 40
        assert sum(labels == "e8") == 10
        assert sum(labels == "e5") == 5


def test_bootstrap_is_reproducible_under_a_seed() -> None:
    strata = ["a"] * 30 + ["b"] * 30
    first = stratified_bootstrap_indices(strata, resamples=5, seed=7)
    second = stratified_bootstrap_indices(strata, resamples=5, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(first, second, strict=True))


def test_paired_ci_excludes_zero_for_a_real_difference() -> None:
    rng = np.random.default_rng(0)
    better = rng.normal(0.9, 0.05, 600)
    worse = rng.normal(0.6, 0.05, 600)
    strata = ["s"] * 600

    def statistic(index: np.ndarray) -> float:
        return float(better[index].mean() - worse[index].mean())

    result = paired_difference_ci(statistic, strata, resamples=300, seed=1)
    assert result["point"] == pytest.approx(0.3, abs=0.02)
    assert result["excludes_zero"] is True
    assert result["lo"] < result["point"] < result["hi"]


def test_paired_ci_includes_zero_when_the_systems_are_the_same() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(0.8, 0.05, 500)
    noise = values + rng.normal(0.0, 0.001, 500)
    strata = ["a"] * 250 + ["b"] * 250

    def statistic(index: np.ndarray) -> float:
        return float(values[index].mean() - noise[index].mean())

    assert paired_difference_ci(statistic, strata, resamples=300, seed=1)["excludes_zero"] is False


def test_ece_is_zero_for_a_perfectly_calibrated_predictor() -> None:
    rng = np.random.default_rng(5)
    prob = rng.random(40000)
    truth = (rng.random(40000) < prob).astype(int)
    assert expected_calibration_error(truth, prob) < 0.01


def test_ece_is_large_for_a_confidently_wrong_predictor() -> None:
    truth = np.array([0] * 500 + [1] * 500)
    prob = np.array([0.99] * 500 + [0.01] * 500)
    assert expected_calibration_error(truth, prob) > 0.9


def test_ece_rejects_logits() -> None:
    with pytest.raises(MetricError, match=r"probabilities in \[0, 1\]"):
        expected_calibration_error([0, 1], [-3.0, 4.0])


def test_ece_bins_include_the_endpoints() -> None:
    truth = np.array([0, 1])
    assert expected_calibration_error(truth, np.array([0.0, 1.0])) == pytest.approx(0.0)
