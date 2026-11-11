"""Metric tests, checked against scikit-learn as the reference implementation.

``dbp.eval.metrics`` reimplements AUC, average precision and the ROC curve over numpy
so their tie and edge-case behaviour is pinned in this repository rather than inherited
from whatever scikit-learn version a runtime happens to have (1.9.1 locally, 1.6.1 on
Kaggle). That is only safe if the two agree, which is what these tests establish --
including on the cases that usually differ: ties, a constant score, and a single
positive.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as sk

from dbp.eval.metrics import (
    MetricError,
    average_precision,
    confusion_at_threshold,
    equal_error_rate,
    rates_at_threshold,
    roc_auc,
    roc_curve,
    summarise,
    tpr_at_fpr,
)

RNG = np.random.default_rng(2026)


def _separable(n: int = 200, gap: float = 1.5):
    """Scores with a real signal: fakes shifted up, heavily overlapping."""
    y_true = RNG.integers(0, 2, size=n)
    y_score = RNG.normal(0.0, 1.0, size=n) + gap * y_true
    return y_true, y_score


@pytest.mark.parametrize("gap", [0.0, 0.5, 1.5, 4.0])
def test_auc_matches_sklearn(gap):
    y_true, y_score = _separable(gap=gap)
    assert roc_auc(y_true, y_score) == pytest.approx(sk.roc_auc_score(y_true, y_score))


def test_auc_matches_sklearn_with_heavy_ties():
    """Ties must contribute 0.5, not depend on the sort order."""
    y_true = RNG.integers(0, 2, size=300)
    y_score = RNG.integers(0, 4, size=300).astype(float)  # only four distinct scores
    assert roc_auc(y_true, y_score) == pytest.approx(sk.roc_auc_score(y_true, y_score))


def test_auc_of_a_constant_score_is_one_half():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    assert roc_auc(y_true, np.full(6, 0.7)) == pytest.approx(0.5)


def test_auc_is_one_when_perfectly_ranked_and_zero_when_inverted():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y_true, y_score) == pytest.approx(1.0)
    assert roc_auc(y_true, -y_score) == pytest.approx(0.0)


@pytest.mark.parametrize("gap", [0.0, 1.0, 3.0])
def test_average_precision_matches_sklearn(gap):
    y_true, y_score = _separable(gap=gap)
    assert average_precision(y_true, y_score) == pytest.approx(
        sk.average_precision_score(y_true, y_score)
    )


def test_average_precision_matches_sklearn_with_ties_and_one_positive():
    y_true = np.array([0, 0, 0, 1, 0, 0])
    y_score = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1])
    assert average_precision(y_true, y_score) == pytest.approx(
        sk.average_precision_score(y_true, y_score)
    )


def test_roc_curve_matches_sklearn():
    y_true, y_score = _separable()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # drop_intermediate=False: sklearn otherwise prunes collinear points, which is a
    # plotting optimisation. Ours keeps one point per distinct score, so the threshold
    # a calibration picks is always present in the array.
    sk_fpr, sk_tpr, sk_thresholds = sk.roc_curve(y_true, y_score, drop_intermediate=False)
    assert np.allclose(fpr, sk_fpr)
    assert np.allclose(tpr, sk_tpr)
    # sklearn also prepends an infinite threshold; compare the finite tail.
    assert np.allclose(thresholds[1:], sk_thresholds[1:])


def test_roc_curve_starts_at_the_origin():
    y_true, y_score = _separable(n=50)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert np.isinf(thresholds[0])


def test_auc_refuses_a_single_class():
    """Returning 0.5 here would report a meaningless number as a real one."""
    with pytest.raises(MetricError, match="needs both classes"):
        roc_auc(np.ones(10, dtype=int), RNG.normal(size=10))


def test_metrics_reject_mismatched_or_non_finite_input():
    with pytest.raises(MetricError, match="length mismatch"):
        roc_auc([0, 1, 1], [0.1, 0.2])
    with pytest.raises(MetricError, match="NaN or inf"):
        roc_auc([0, 1], [0.1, np.nan])
    with pytest.raises(MetricError, match="must be 0"):
        roc_auc([0, 2], [0.1, 0.2])


def test_confusion_and_accuracy_match_sklearn_at_a_threshold():
    y_true, y_score = _separable()
    threshold = 0.9
    counts = confusion_at_threshold(y_true, y_score, threshold)
    predicted = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = sk.confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    assert counts == {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    rates = rates_at_threshold(y_true, y_score, threshold)
    assert rates["accuracy"] == pytest.approx(sk.accuracy_score(y_true, predicted))
    assert rates["balanced_accuracy"] == pytest.approx(
        sk.balanced_accuracy_score(y_true, predicted)
    )


def test_the_fire_failure_mode_is_visible_in_the_summary():
    """Excellent ranking, useless decisions: AUC ~ 1 while every real is called fake.

    This is the FIRE result that motivates the project -- AUC ~ 100 in-domain, yet at
    threshold 0.5 all 389 real images are labelled fake -- so the summary block must
    show both numbers at once.
    """
    y_true = np.r_[np.zeros(389, dtype=int), np.ones(389, dtype=int)]
    y_score = np.r_[np.linspace(0.90, 0.95, 389), np.linspace(0.96, 0.99, 389)]

    block = summarise(y_true, y_score, threshold=0.5)
    assert block["auc"] == pytest.approx(1.0)
    assert block["fpr"] == pytest.approx(1.0), "every real should be flagged at 0.5"
    assert block["tn"] == 0
    assert block["accuracy"] == pytest.approx(0.5)


def test_rates_report_none_rather_than_zero_for_an_absent_class():
    """A subset with no reals has no false-positive rate; 0.0 would read as perfect."""
    rates = rates_at_threshold(np.ones(5, dtype=int), np.full(5, 0.9), 0.5)
    assert rates["fpr"] is None
    assert rates["balanced_accuracy"] is None
    assert rates["tpr"] == pytest.approx(1.0)


def test_tpr_at_fpr_never_exceeds_the_budget():
    y_true, y_score = _separable(n=500)
    for target in (0.0, 0.01, 0.05, 0.2):
        tpr, threshold = tpr_at_fpr(y_true, y_score, target)
        achieved = rates_at_threshold(y_true, y_score, threshold)
        if np.isfinite(threshold):
            assert achieved["fpr"] <= target + 1e-12
            assert achieved["tpr"] == pytest.approx(tpr)


def test_equal_error_rate_is_where_the_two_error_rates_meet():
    y_true, y_score = _separable(n=1000, gap=2.0)
    eer, threshold = equal_error_rate(y_true, y_score)
    rates = rates_at_threshold(y_true, y_score, threshold)
    assert abs((1.0 - rates["tpr"]) - rates["fpr"]) < 0.05
    assert 0.0 <= eer <= 0.5


def test_summary_block_is_flat_and_json_friendly():
    y_true, y_score = _separable(n=120)
    block = summarise(y_true, y_score)
    assert block["n"] == 120
    assert block["n_fake"] + block["n_real"] == 120
    for key, value in block.items():
        assert isinstance(value, (int, float, type(None))), f"{key} is not a scalar"
