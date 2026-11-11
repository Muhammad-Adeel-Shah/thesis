"""Threshold policy: where the decision boundary comes from, and where it may not.

A detection threshold chosen on the set it is then reported on is not a result. It is
the single easiest way to turn a mediocre detector into a publishable table, and it is
easy to do by accident -- ``argmax`` over a test-set sweep looks like analysis.

So a threshold in this project is not a float passed around. It is a
:class:`Calibration`: a value together with the set it was fitted on, the rule used,
how many samples that set had, and what the rule actually achieved there. Every
evaluation result carries the calibration that produced it, and
:func:`evaluate_with_calibration` refuses to score a set whose id matches the
calibration's own -- the mistake is then a loud error rather than a good number.

P1 ships and tests this module; it fits nothing, because P1 produces no scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dbp.eval.metrics import MetricError, rates_at_threshold, roc_curve

#: Threshold reported alongside every calibrated one, because it is what a naive
#: deployment uses and what exposed the FIRE calibration failure.
NAIVE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Calibration:
    """A threshold plus the evidence for it.

    Attributes:
        threshold: Scores at or above this are called fake.
        rule: How it was chosen -- ``"target_fpr"``, ``"youden"`` or ``"fixed"``.
        fitted_on: Id of the split it was fitted on. Never an evaluation set.
        n_fitted: Samples in that split.
        target: The rule's parameter, e.g. the requested FPR. None for ``"fixed"``.
        achieved_fpr: FPR the threshold actually produced on the calibration split.
        achieved_tpr: TPR it produced there.
    """

    threshold: float
    rule: str
    fitted_on: str
    n_fitted: int
    target: float | None = None
    achieved_fpr: float | None = None
    achieved_tpr: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for a JSON report or a CSV row."""
        return asdict(self)


def calibrate_at_fpr(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    target_fpr: float,
    fitted_on: str,
) -> Calibration:
    """Pick the threshold with the best TPR whose FPR does not exceed ``target_fpr``.

    This is the operating point a deployment actually cares about: "how much of the
    fake traffic do we catch, given that we will wrongly flag at most this share of
    genuine images".

    Args:
        y_true: 1 fake, 0 real, from the **calibration** split.
        y_score: Scores for that split.
        target_fpr: The false-positive budget, in [0, 1].
        fitted_on: Id of the calibration split, recorded in the result.

    Returns:
        The :class:`Calibration`.

    Raises:
        MetricError: If no threshold meets the budget, which means even the strictest
            threshold misclassifies more reals than allowed.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise MetricError(f"target_fpr must be in [0, 1], got {target_fpr}")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    allowed = np.where(fpr <= target_fpr)[0]
    if allowed.size == 0:
        raise MetricError(f"no threshold achieves FPR <= {target_fpr} on {fitted_on}")
    best = allowed[int(np.argmax(tpr[allowed]))]
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        # The only allowed point is the trivial one above every score: nothing is
        # flagged. Report it honestly rather than as a usable operating point.
        threshold = float(np.max(np.asarray(y_score, dtype=np.float64))) + 1.0
    return Calibration(
        threshold=threshold,
        rule="target_fpr",
        fitted_on=fitted_on,
        n_fitted=int(np.asarray(y_true).shape[0]),
        target=float(target_fpr),
        achieved_fpr=float(fpr[best]),
        achieved_tpr=float(tpr[best]),
    )


def calibrate_youden(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    fitted_on: str,
) -> Calibration:
    """Pick the threshold maximising Youden's J (``TPR - FPR``).

    Reported alongside the FPR-budget threshold because it is the conventional choice
    in this literature; it is not the one a deployment should use, since it treats a
    missed fake and a wrongly flagged real as equally costly.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        threshold = float(np.max(np.asarray(y_score, dtype=np.float64))) + 1.0
    return Calibration(
        threshold=threshold,
        rule="youden",
        fitted_on=fitted_on,
        n_fitted=int(np.asarray(y_true).shape[0]),
        target=None,
        achieved_fpr=float(fpr[best]),
        achieved_tpr=float(tpr[best]),
    )


def fixed_threshold(threshold: float = NAIVE_THRESHOLD) -> Calibration:
    """The uncalibrated threshold, recorded as such.

    ``fitted_on`` is ``"none"`` because nothing was fitted; that is the point. Reported
    in every table so a detector that ranks well but decides badly cannot hide.
    """
    return Calibration(
        threshold=float(threshold),
        rule="fixed",
        fitted_on="none",
        n_fitted=0,
        target=None,
    )


def evaluate_with_calibration(
    calibration: Calibration,
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    set_id: str,
) -> dict[str, Any]:
    """Apply a calibration to an evaluation set and report the operating point.

    Args:
        calibration: A threshold fitted elsewhere.
        y_true: 1 fake, 0 real, for the evaluation set.
        y_score: Scores for that set.
        set_id: Id of the evaluation set.

    Returns:
        The rates at that threshold, plus the calibration's provenance.

    Raises:
        MetricError: If ``set_id`` is the set the threshold was fitted on. Tuning the
            threshold on the test set is the failure this module exists to prevent, so
            it is refused rather than warned about.
    """
    if calibration.fitted_on == set_id:
        raise MetricError(
            f"threshold was fitted on {set_id!r} and is being applied to {set_id!r}. "
            f"Calibrate on a held-out split, then apply it unchanged."
        )
    block = rates_at_threshold(y_true, y_score, calibration.threshold)
    block["set_id"] = set_id
    block["threshold_rule"] = calibration.rule
    block["threshold_fitted_on"] = calibration.fitted_on
    block["threshold_target"] = calibration.target
    block["threshold_achieved_fpr_on_calibration"] = calibration.achieved_fpr
    return block


def calibrate_on_reals(
    real_scores: Sequence[float] | np.ndarray,
    target_fpr: float,
    fitted_on: str,
) -> Calibration:
    """The threshold at an FPR budget, fixed **on the validation reals alone** (I-062).

    The threshold is the lowest real score ``s`` such that the share of reals scoring
    ``>= s`` is at most ``target_fpr`` (pre-registration Amendment 2, point 3). No fake
    score enters it. :func:`calibrate_at_fpr` maximises validation TPR under the same
    budget, which is the same threshold while the TPR still rises -- but where a stream
    separates validation perfectly it lands on the lowest validation *fake*, far above
    the reals' quantile, and the stream is then judged at a much stricter operating
    point than the budget names.

    Args:
        real_scores: Scores of the calibration split's reals.
        target_fpr: The false-positive budget, in [0, 1).
        fitted_on: Id of the calibration split, recorded in the result.

    Returns:
        The :class:`Calibration`; ``achieved_tpr`` is None because no fake was used.

    Raises:
        MetricError: On no reals, a non-finite score or a budget outside [0, 1).
    """
    if not 0.0 <= target_fpr < 1.0:
        raise MetricError(f"target_fpr must be in [0, 1), got {target_fpr}")
    reals = np.asarray(real_scores, dtype=np.float64).ravel()
    if reals.size == 0:
        raise MetricError(f"no reals to calibrate on in {fitted_on}")
    if not np.all(np.isfinite(reals)):
        raise MetricError(f"non-finite real score in {fitted_on}")
    candidates = np.unique(reals)  # ascending
    at_or_above = reals.size - np.searchsorted(np.sort(reals), candidates, side="left")
    allowed = candidates[at_or_above <= target_fpr * reals.size]
    # Nothing qualifies only when the budget admits no real at all: flag nothing.
    threshold = float(allowed[0]) if allowed.size else float(reals.max()) + 1.0
    return Calibration(
        threshold=threshold,
        rule="target_fpr_on_reals",
        fitted_on=fitted_on,
        n_fitted=int(reals.size),
        target=float(target_fpr),
        achieved_fpr=float(np.mean(reals >= threshold)),
        achieved_tpr=None,
    )
