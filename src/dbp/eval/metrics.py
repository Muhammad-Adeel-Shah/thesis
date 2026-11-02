"""Detection metrics.

The headline number in this literature is AUC, and AUC is exactly the number that hid
the FIRE failure: FIRE reaches AUC ~100 in-domain, yet at threshold 0.5 it labels all
389 real images fake. A ranking metric cannot see that, because it never commits to a
decision. So every summary produced here reports, side by side:

- **ranking** quality, which does not depend on a threshold: AUC and average precision;
- **operating-point** quality at a threshold that was fixed elsewhere: accuracy,
  balanced accuracy, TPR and FPR, and the full confusion count;
- **a cost-free operating point** for reference: TPR at 1% and 5% FPR, and the EER.

Implemented over numpy rather than imported from scikit-learn, so the exact tie and
edge-case behaviour is visible and pinned. ``tests/test_metrics.py`` checks these
against ``sklearn.metrics`` on random and adversarial inputs -- scikit-learn is a
declared dependency precisely to serve as that reference.

Convention throughout: ``y_true`` is 1 for **fake** and 0 for **real**, and ``y_score``
is higher for more-likely-fake. The convention is asserted, not assumed: a silently
inverted label would turn a broken detector into an excellent one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class MetricError(ValueError):
    """Raised when a metric cannot be computed from the inputs as given."""


def _as_arrays(
    y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert the inputs, refusing anything ambiguous."""
    truth = np.asarray(y_true)
    score = np.asarray(y_score, dtype=np.float64)
    if truth.ndim != 1 or score.ndim != 1:
        raise MetricError(f"expected 1-D arrays, got {truth.shape} and {score.shape}")
    if truth.shape != score.shape:
        raise MetricError(f"length mismatch: {truth.shape[0]} labels, {score.shape[0]} scores")
    if truth.size == 0:
        raise MetricError("no samples")
    unique = set(np.unique(truth).tolist())
    if not unique <= {0, 1}:
        raise MetricError(f"y_true must be 0 (real) or 1 (fake); found {sorted(unique)}")
    if not np.isfinite(score).all():
        raise MetricError("y_score contains NaN or inf")
    return truth.astype(np.int64), score


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, as the Mann-Whitney U statistic requires."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.shape[0], dtype=np.float64)

    start = 0
    for index in range(1, sorted_values.shape[0] + 1):
        if index == sorted_values.shape[0] or sorted_values[index] != sorted_values[start]:
            # Ranks are 1-based; a tied block all takes the block's mean rank.
            ranks[order[start:index]] = (start + index + 1) / 2.0
            start = index
    return ranks


def roc_auc(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float:
    """Area under the ROC curve, via the rank-sum identity.

    Ties contribute 0.5, which is what makes this agree with scikit-learn on a
    constant score (AUC 0.5) rather than reporting 0 or 1 depending on sort order.

    Raises:
        MetricError: If either class is absent; AUC is undefined then, and returning
            0.5 would quietly report a meaningless result as a real one.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = int(truth.sum())
    negatives = int(truth.shape[0] - positives)
    if positives == 0 or negatives == 0:
        raise MetricError(
            f"ROC AUC needs both classes; got {positives} fake and {negatives} real"
        )
    ranks = _average_ranks(score)
    rank_sum = float(ranks[truth == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def roc_curve(
    y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve as ``(fpr, tpr, thresholds)``, decreasing in threshold.

    A leading ``(0, 0)`` point at ``threshold = inf`` is included, so the curve starts
    at the origin and the arrays can be plotted directly.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = int(truth.sum())
    negatives = int(truth.shape[0] - positives)
    if positives == 0 or negatives == 0:
        raise MetricError(
            f"an ROC curve needs both classes; got {positives} fake and {negatives} real"
        )

    order = np.argsort(-score, kind="mergesort")
    scores_sorted = score[order]
    truth_sorted = truth[order]

    true_positives = np.cumsum(truth_sorted)
    false_positives = np.cumsum(1 - truth_sorted)

    # One point per distinct score: the curve only turns where the threshold can split.
    distinct = np.where(np.diff(scores_sorted))[0]
    keep = np.r_[distinct, truth_sorted.shape[0] - 1]

    tpr = np.r_[0.0, true_positives[keep] / positives]
    fpr = np.r_[0.0, false_positives[keep] / negatives]
    thresholds = np.r_[np.inf, scores_sorted[keep]]
    return fpr, tpr, thresholds


def average_precision(
    y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> float:
    """Average precision: the step-wise area under the precision-recall curve.

    This is the ``sum((R_n - R_{n-1}) * P_n)`` estimator, not the interpolated one, so
    it matches ``sklearn.metrics.average_precision_score``.
    """
    truth, score = _as_arrays(y_true, y_score)
    positives = int(truth.sum())
    if positives == 0:
        raise MetricError("average precision needs at least one fake sample")

    order = np.argsort(-score, kind="mergesort")
    scores_sorted = score[order]
    truth_sorted = truth[order]

    true_positives = np.cumsum(truth_sorted)
    predicted_positives = np.arange(1, truth_sorted.shape[0] + 1)

    distinct = np.where(np.diff(scores_sorted))[0]
    keep = np.r_[distinct, truth_sorted.shape[0] - 1]

    precision = true_positives[keep] / predicted_positives[keep]
    recall = true_positives[keep] / positives
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


def confusion_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Confusion counts at a fixed threshold. ``score >= threshold`` means fake."""
    truth, score = _as_arrays(y_true, y_score)
    predicted = (score >= threshold).astype(np.int64)
    return {
        "tp": int(np.sum((predicted == 1) & (truth == 1))),
        "fp": int(np.sum((predicted == 1) & (truth == 0))),
        "tn": int(np.sum((predicted == 0) & (truth == 0))),
        "fn": int(np.sum((predicted == 0) & (truth == 1))),
    }


def rates_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, TPR, FPR and the confusion counts at a threshold.

    ``tpr`` and ``fpr`` are None when the corresponding class is absent, rather than 0:
    an evaluation subset with no real images has no false-positive rate, and printing
    0.0 there would read as a perfect result.
    """
    counts = confusion_at_threshold(y_true, y_score, threshold)
    total = sum(counts.values())
    actual_fake = counts["tp"] + counts["fn"]
    actual_real = counts["tn"] + counts["fp"]

    tpr = counts["tp"] / actual_fake if actual_fake else None
    fpr = counts["fp"] / actual_real if actual_real else None
    tnr = counts["tn"] / actual_real if actual_real else None
    balanced = (tpr + tnr) / 2.0 if (tpr is not None and tnr is not None) else None

    return {
        "threshold": float(threshold),
        "accuracy": (counts["tp"] + counts["tn"]) / total,
        "balanced_accuracy": balanced,
        "tpr": tpr,
        "fpr": fpr,
        "n": total,
        **counts,
    }


def tpr_at_fpr(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    target_fpr: float,
) -> tuple[float, float]:
    """Highest TPR reachable without exceeding ``target_fpr``.

    Returns:
        ``(tpr, threshold)``. The threshold is the one that achieves it, so the caller
        can report the operating point rather than only the rate.
    """
    if not 0.0 <= target_fpr <= 1.0:
        raise MetricError(f"target_fpr must be in [0, 1], got {target_fpr}")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    allowed = np.where(fpr <= target_fpr)[0]
    if allowed.size == 0:
        return 0.0, float(np.inf)
    best = allowed[int(np.argmax(tpr[allowed]))]
    return float(tpr[best]), float(thresholds[best])


def equal_error_rate(
    y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray
) -> tuple[float, float]:
    """The point where the false-positive and false-negative rates are closest.

    Returns:
        ``(eer, threshold)``, the EER being the mean of the two rates at that point.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[index] + fpr[index]) / 2.0), float(thresholds[index])


def summarise(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """The full metric block for one evaluation set at one operating point.

    Args:
        y_true: 1 for fake, 0 for real.
        y_score: Higher means more likely fake.
        threshold: A threshold fixed on a calibration split. The default of 0.5 is the
            naive choice and is reported so the FIRE-style failure -- excellent AUC,
            useless decisions -- is visible in the same table as the AUC.

    Returns:
        A flat mapping suitable for a CSV row.
    """
    truth, score = _as_arrays(y_true, y_score)
    block: dict[str, Any] = {
        "n": int(truth.shape[0]),
        "n_fake": int(truth.sum()),
        "n_real": int(truth.shape[0] - truth.sum()),
        "auc": roc_auc(truth, score),
        "average_precision": average_precision(truth, score),
    }
    block.update(rates_at_threshold(truth, score, threshold))

    eer, eer_threshold = equal_error_rate(truth, score)
    block["eer"] = eer
    block["eer_threshold"] = eer_threshold
    for target in (0.01, 0.05):
        rate, operating_threshold = tpr_at_fpr(truth, score, target)
        block[f"tpr_at_fpr_{target:g}"] = rate
        block[f"threshold_at_fpr_{target:g}"] = operating_threshold
    return block
