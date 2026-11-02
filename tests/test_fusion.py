"""Tests for validation-only calibration and fusion (B6)."""

from __future__ import annotations

import numpy as np
import pytest

from dbp.eval.complementarity import expected_calibration_error
from dbp.eval.fusion import (
    FittedFusion,
    FusionError,
    fit_mean_fusion,
    fit_stack_fusion,
    fit_temperature,
    oracle_scores,
    sigmoid,
)
from dbp.eval.metrics import MetricError, roc_auc


def _two_stream_val(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Two streams that are individually mediocre and err on different samples."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    half = n // 2
    strong_a = np.zeros(n, dtype=bool)
    strong_a[:half] = True
    a = np.where(strong_a, y * 4.0 - 2.0, rng.normal(0, 1, n)) + rng.normal(0, 0.6, n)
    b = np.where(~strong_a, y * 4.0 - 2.0, rng.normal(0, 1, n)) + rng.normal(0, 0.6, n)
    return y, {"s_a": a, "s_b": b}


def test_sigmoid_is_stable_at_extremes() -> None:
    out = sigmoid(np.array([-800.0, 0.0, 800.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_temperature_improves_calibration_without_touching_auc() -> None:
    # A calibrated stream whose logits are then inflated 3x: the labels really are
    # drawn at the honest probability, so the inflated scores are overconfident and
    # the recoverable temperature is 3.
    rng = np.random.default_rng(3)
    honest = rng.normal(0.0, 1.5, 8000)
    y = (rng.random(8000) < sigmoid(honest)).astype(int)
    overconfident = honest * 3.0

    temperature = fit_temperature(y, overconfident)
    assert temperature == pytest.approx(3.0, rel=0.15)
    before = expected_calibration_error(y, sigmoid(overconfident))
    after = expected_calibration_error(y, sigmoid(overconfident / temperature))

    assert temperature > 1.0, "an overconfident stream should be cooled"
    assert after < before
    assert roc_auc(y, overconfident) == pytest.approx(
        roc_auc(y, overconfident / temperature)
    ), "temperature scaling must not change any ranking"


def test_temperature_needs_both_classes() -> None:
    with pytest.raises(MetricError, match="both classes"):
        fit_temperature([1, 1, 1], [0.1, 0.2, 0.3])


def test_temperature_rejects_non_finite_scores() -> None:
    with pytest.raises(MetricError, match="NaN or inf"):
        fit_temperature([0, 1], [0.1, np.inf])


def test_mean_fusion_beats_neither_stream_on_nothing_but_helps_on_complementary_ones() -> None:
    y, scores = _two_stream_val()
    fusion = fit_mean_fusion(y, scores, fitted_on="val")
    fused = fusion.score(scores, set_id="panel")
    assert roc_auc(y, fused) > max(roc_auc(y, s) for s in scores.values())


def test_stack_fusion_beats_the_best_single_stream_on_complementary_streams() -> None:
    y, scores = _two_stream_val()
    fusion = fit_stack_fusion(y, scores, fitted_on="val")
    fused = fusion.score(scores, set_id="panel")
    assert roc_auc(y, fused) > max(roc_auc(y, s) for s in scores.values())
    assert len(fusion.coefficients) == 2


def test_a_fusion_refuses_to_score_the_split_it_was_fitted_on() -> None:
    """The mistake that would turn a NO-GO into a GO with no visible error."""
    y, scores = _two_stream_val(n=400)
    fusion = fit_stack_fusion(y, scores, fitted_on="val")
    with pytest.raises(FusionError, match="fitted on"):
        fusion.score(scores, set_id="val")


def test_a_fusion_refuses_a_missing_stream() -> None:
    y, scores = _two_stream_val(n=400)
    fusion = fit_stack_fusion(y, scores, fitted_on="val")
    with pytest.raises(FusionError, match="not in scores"):
        fusion.score({"s_a": scores["s_a"]}, set_id="panel")


def test_a_fusion_refuses_streams_of_different_lengths() -> None:
    y, scores = _two_stream_val(n=400)
    fusion = fit_mean_fusion(y, scores, fitted_on="val")
    with pytest.raises(FusionError, match="different lengths"):
        fusion.score({"s_a": scores["s_a"], "s_b": scores["s_b"][:100]}, set_id="panel")


def test_stack_needs_both_classes_in_validation() -> None:
    with pytest.raises(MetricError, match="both classes"):
        fit_stack_fusion([1, 1, 1], {"a": [0.1, 0.2, 0.3]}, fitted_on="val")


def test_fusion_records_what_it_was_fitted_on() -> None:
    y, scores = _two_stream_val(n=500)
    record = fit_stack_fusion(y, scores, fitted_on="val_p_a").to_dict()
    assert record["fitted_on"] == "val_p_a"
    assert record["n_fitted"] == 500
    assert record["kind"] == "stack"
    assert record["streams"] == ["s_a", "s_b"]


def test_stream_order_is_sorted_so_coefficients_are_interpretable() -> None:
    y, scores = _two_stream_val(n=400)
    reversed_order = {"s_b": scores["s_b"], "s_a": scores["s_a"]}
    assert fit_mean_fusion(y, reversed_order, fitted_on="val").streams == ("s_a", "s_b")


def test_oracle_is_right_whenever_any_stream_is() -> None:
    y = np.array([0, 1, 0, 1])
    scores = {"a": np.array([-1.0, 1.0, 1.0, -1.0]),   # right, right, wrong, wrong
              "b": np.array([1.0, -1.0, -1.0, -1.0])}  # wrong, wrong, right, wrong
    out = oracle_scores(y, scores, {"a": 0.0, "b": 0.0})
    # Right on the first three, wrong on the fourth where neither stream is right.
    assert list((out >= 0.5).astype(int) == y) == [True, True, True, False]


def test_oracle_needs_a_threshold_for_every_stream() -> None:
    with pytest.raises(FusionError, match="no threshold"):
        oracle_scores([0, 1], {"a": [0.1, 0.9], "b": [0.2, 0.8]}, {"a": 0.5})


def test_a_hand_built_fusion_round_trips_through_to_dict() -> None:
    fusion = FittedFusion(
        kind="mean", streams=("x",), temperatures=(2.0,), fitted_on="val", n_fitted=10
    )
    assert fusion.to_dict()["temperatures"] == [2.0]
    assert np.allclose(fusion.score({"x": [4.0]}, set_id="panel"), [2.0])
