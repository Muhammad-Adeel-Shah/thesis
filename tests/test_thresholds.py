"""Threshold-policy tests.

The properties that matter are not numerical. They are procedural: a threshold must
carry where it came from, it must respect the false-positive budget it claims, and
applying it to the very set it was fitted on must fail loudly rather than produce a
flattering number.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dbp.eval.metrics import MetricError, rates_at_threshold
from dbp.eval.thresholds import (
    NAIVE_THRESHOLD,
    Calibration,
    calibrate_at_fpr,
    calibrate_youden,
    evaluate_with_calibration,
    fixed_threshold,
)

RNG = np.random.default_rng(90210)


def _split(n: int = 400, gap: float = 1.5):
    y_true = RNG.integers(0, 2, size=n)
    y_score = RNG.normal(0.0, 1.0, size=n) + gap * y_true
    return y_true, y_score


@pytest.mark.parametrize("target", [0.01, 0.05, 0.1, 0.25])
def test_calibration_respects_its_false_positive_budget(target):
    y_true, y_score = _split()
    calibration = calibrate_at_fpr(y_true, y_score, target, fitted_on="calib")
    achieved = rates_at_threshold(y_true, y_score, calibration.threshold)
    assert achieved["fpr"] <= target + 1e-12
    assert calibration.achieved_fpr == pytest.approx(achieved["fpr"])
    assert calibration.achieved_tpr == pytest.approx(achieved["tpr"])


def test_calibration_records_its_provenance():
    y_true, y_score = _split(n=250)
    calibration = calibrate_at_fpr(y_true, y_score, 0.05, fitted_on="t_real_holdout")
    assert calibration.rule == "target_fpr"
    assert calibration.fitted_on == "t_real_holdout"
    assert calibration.n_fitted == 250
    assert calibration.target == pytest.approx(0.05)
    assert set(calibration.to_dict()) == {
        "threshold",
        "rule",
        "fitted_on",
        "n_fitted",
        "target",
        "achieved_fpr",
        "achieved_tpr",
    }


def test_applying_a_threshold_to_its_own_calibration_set_is_refused():
    """The easiest way to fake a good table, so it is an error, not a warning."""
    y_true, y_score = _split()
    calibration = calibrate_at_fpr(y_true, y_score, 0.05, fitted_on="e3_raw")
    with pytest.raises(MetricError, match="fitted on"):
        evaluate_with_calibration(calibration, y_true, y_score, set_id="e3_raw")


def test_applying_a_threshold_to_a_different_set_reports_the_provenance():
    y_true, y_score = _split()
    other_true, other_score = _split(n=150)
    calibration = calibrate_at_fpr(y_true, y_score, 0.05, fitted_on="calib")
    block = evaluate_with_calibration(calibration, other_true, other_score, set_id="e4_wildrf")
    assert block["set_id"] == "e4_wildrf"
    assert block["threshold_rule"] == "target_fpr"
    assert block["threshold_fitted_on"] == "calib"
    assert block["threshold_target"] == pytest.approx(0.05)
    assert block["threshold"] == pytest.approx(calibration.threshold)


def test_youden_beats_or_matches_the_naive_threshold_on_its_own_split():
    y_true, y_score = _split(gap=2.0)
    calibration = calibrate_youden(y_true, y_score, fitted_on="calib")
    youden = calibration.achieved_tpr - calibration.achieved_fpr
    naive = rates_at_threshold(y_true, y_score, NAIVE_THRESHOLD)
    assert youden >= (naive["tpr"] - naive["fpr"]) - 1e-12


def test_a_zero_budget_that_cannot_be_met_yields_a_threshold_above_every_score():
    """With a zero FPR budget and overlapping scores, the honest answer is 'flag
    nothing', reported as such rather than as a usable operating point."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.9, 0.9, 0.9, 0.9])
    calibration = calibrate_at_fpr(y_true, y_score, 0.0, fitted_on="calib")
    assert calibration.threshold > y_score.max()
    achieved = rates_at_threshold(y_true, y_score, calibration.threshold)
    assert achieved["tp"] == 0 and achieved["fp"] == 0


def test_fixed_threshold_is_marked_as_never_fitted():
    calibration = fixed_threshold()
    assert isinstance(calibration, Calibration)
    assert calibration.threshold == pytest.approx(NAIVE_THRESHOLD)
    assert calibration.rule == "fixed"
    assert calibration.fitted_on == "none"
    assert calibration.n_fitted == 0
    # It can therefore be applied to any set, including one named "none"-adjacent.
    y_true, y_score = _split(n=60)
    block = evaluate_with_calibration(calibration, y_true, y_score, set_id="e3_raw")
    assert block["threshold_fitted_on"] == "none"


def test_calibration_is_immutable():
    """A threshold that can be edited after the fact is not provenance."""
    calibration = fixed_threshold()
    with pytest.raises(dataclasses.FrozenInstanceError):
        calibration.threshold = 0.9  # type: ignore[misc]


def test_an_impossible_budget_raises_rather_than_returning_nothing():
    y_true = np.array([0, 1])
    y_score = np.array([0.5, 0.5])
    with pytest.raises(MetricError, match="must be in"):
        calibrate_at_fpr(y_true, y_score, 1.5, fitted_on="calib")
