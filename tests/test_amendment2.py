"""Pre-registration Amendment 2: the governing fusion and the reals-only thresholds.

Two regressions are pinned here:

* **I-062.** ``calibrate_at_fpr`` maximises validation TPR under the FPR budget, so on a
  head that separates validation perfectly the "5% FPR" threshold is the lowest
  validation fake -- P-A's S2 ran at 0.20% FPR on the reals. The threshold must depend
  on the reals alone.
* **F-mean(A2)** learns no weight and sees no fake label: its parameters are the
  validation reals' median and MAD per member, clipped at +-5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbp.eval.fusion import (  # noqa: E402
    GOVERNING_STREAMS,
    MAD_TO_SD,
    FusionError,
    RobustMeanFusion,
    fit_robust_mean,
    fusion_from_dict,
    robust_score_by_id,
)
from dbp.eval.join import JoinError  # noqa: E402
from dbp.eval.metrics import MetricError  # noqa: E402
from dbp.eval.thresholds import calibrate_at_fpr, calibrate_on_reals  # noqa: E402

# ------------------------------------------------------------- I-062 thresholds --


def test_the_threshold_admits_at_most_the_budget_of_reals() -> None:
    reals = np.arange(491, dtype=float)
    five = calibrate_on_reals(reals, 0.05, "val")
    one = calibrate_on_reals(reals, 0.01, "val")
    assert int((reals >= five.threshold).sum()) == 24      # floor(0.05 * 491)
    assert int((reals >= one.threshold).sum()) == 4        # floor(0.01 * 491)
    assert five.achieved_fpr == pytest.approx(24 / 491)
    assert five.achieved_tpr is None and five.rule == "target_fpr_on_reals"


def test_ties_move_the_threshold_up_rather_than_over_the_budget() -> None:
    reals = np.r_[np.zeros(80), np.full(20, 5.0)]          # 20% tied at the top
    threshold = calibrate_on_reals(reals, 0.05, "val").threshold
    assert np.mean(reals >= threshold) <= 0.05
    assert threshold > 5.0                                  # nothing qualifies: flag no real


def test_i062_a_saturated_head_is_thresholded_on_the_reals_not_the_lowest_fake() -> None:
    """The regression: perfect separation used to put the threshold on the fakes."""
    rng = np.random.default_rng(0)
    reals, fakes = rng.normal(0, 1, 491), rng.normal(20, 1, 441)
    y = np.r_[np.zeros(491), np.ones(441)]
    old = calibrate_at_fpr(y, np.r_[reals, fakes], 0.05, "val")
    new = calibrate_on_reals(reals, 0.05, "val")
    assert old.achieved_fpr < 0.01                          # the defect: ~0% FPR on reals
    assert np.mean(reals >= new.threshold) == pytest.approx(24 / 491)
    assert new.threshold < old.threshold


def test_the_reals_only_threshold_ignores_the_fakes_entirely() -> None:
    reals = np.random.default_rng(1).normal(0, 1, 300)
    assert calibrate_on_reals(reals, 0.05, "v").threshold == calibrate_on_reals(
        reals.copy(), 0.05, "v").threshold


@pytest.mark.parametrize("bad", [np.array([]), np.array([0.0, np.nan])])
def test_no_reals_or_a_non_finite_real_raises(bad: np.ndarray) -> None:
    with pytest.raises(MetricError):
        calibrate_on_reals(bad, 0.05, "val")


# ------------------------------------------------------------------- F-mean(A2) --


def _tables(n_real: int = 200, n_fake: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = [f"r{i}" for i in range(n_real)] + [f"f{i}" for i in range(n_fake)]
    tables = {
        name: (ids, np.r_[rng.normal(k, 1.0 + k, n_real), rng.normal(10 + k, 1.0, n_fake)])
        for k, name in enumerate(("s1", "s2", "s3_lr", "s3_set", "aeroblade"))
    }
    return ids, ids[:n_real], tables


def test_parameters_are_the_reals_median_and_scaled_mad() -> None:
    _, real_ids, tables = _tables()
    fusion = fit_robust_mean(tables, real_ids, "val")
    assert fusion.streams == GOVERNING_STREAMS
    for name, median, scale in zip(fusion.streams, fusion.medians, fusion.scales, strict=True):
        reals = tables[name][1][:200]
        assert median == pytest.approx(np.median(reals))
        assert scale == pytest.approx(MAD_TO_SD * np.median(np.abs(reals - np.median(reals))))
    assert fusion.scale_rules == ("mad", "mad", "mad")


def test_no_fake_score_changes_the_fit() -> None:
    _, real_ids, tables = _tables()
    moved = {k: (ids, np.r_[v[:200], v[200:] * 100 - 7]) for k, (ids, v) in tables.items()}
    assert fit_robust_mean(tables, real_ids, "val") == fit_robust_mean(moved, real_ids, "val")


def test_a_shuffled_stream_gives_the_same_scores() -> None:
    ids, real_ids, tables = _tables()
    fusion = fit_robust_mean(tables, real_ids, "val")
    order = np.random.default_rng(3).permutation(len(ids))
    shuffled = dict(tables, s2=([ids[i] for i in order], tables["s2"][1][order]))
    np.testing.assert_allclose(robust_score_by_id(fusion, tables, ids, "panel"),
                               robust_score_by_id(fusion, shuffled, ids, "panel"))


def test_scores_are_the_mean_of_clipped_z() -> None:
    ids, real_ids, tables = _tables()
    fusion = fit_robust_mean(tables, real_ids, "val")
    scores = robust_score_by_id(fusion, tables, ids, "panel")
    assert scores.max() <= 5.0 and scores.min() >= -5.0
    z = [np.clip((tables[n][1] - m) / s, -5, 5)
         for n, m, s in zip(fusion.streams, fusion.medians, fusion.scales, strict=True)]
    np.testing.assert_allclose(scores, np.mean(z, axis=0))


def test_zero_mad_falls_back_to_std_and_constant_reals_raise() -> None:
    ids = [f"r{i}" for i in range(10)]
    mostly_constant = np.r_[np.zeros(9), 4.0]               # MAD 0, std > 0
    tables = {"s1": (ids, mostly_constant), "s2": (ids, np.arange(10.0)),
              "s3_set": (ids, np.arange(10.0))}
    fusion = fit_robust_mean(tables, ids, "val")
    assert fusion.scale_rules[0] == "std"
    assert fusion.scales[0] == pytest.approx(np.std(mostly_constant))
    tables["s1"] = (ids, np.zeros(10))
    with pytest.raises(FusionError, match="constant"):
        fit_robust_mean(tables, ids, "val")


def test_an_absent_member_or_a_missing_id_raises() -> None:
    ids, real_ids, tables = _tables()
    fusion = fit_robust_mean(tables, real_ids, "val")
    without = {k: v for k, v in tables.items() if k != "s3_set"}
    with pytest.raises(JoinError, match="s3_set"):
        robust_score_by_id(fusion, without, ids, "panel")
    with pytest.raises(JoinError, match="s3_set"):
        fit_robust_mean(without, real_ids, "val")
    short = dict(tables, s2=(ids[1:], tables["s2"][1][1:]))
    with pytest.raises(JoinError, match="missing 1"):
        robust_score_by_id(fusion, short, ids, "panel")


def test_it_refuses_score_on_its_own_split_but_transform_is_allowed() -> None:
    ids, real_ids, tables = _tables()
    fusion = fit_robust_mean(tables, real_ids, "val")
    with pytest.raises(FusionError):
        robust_score_by_id(fusion, tables, ids, "val")
    assert robust_score_by_id(fusion, tables, ids, None).shape == (len(ids),)


def test_it_round_trips_through_the_report() -> None:
    _, real_ids, tables = _tables()
    fusion = fit_robust_mean(tables, real_ids, "val")
    spec = fusion.to_dict()
    assert spec["uses_fake_labels"] is False
    assert fusion_from_dict(spec) == fusion
    assert isinstance(fusion_from_dict(spec), RobustMeanFusion)
