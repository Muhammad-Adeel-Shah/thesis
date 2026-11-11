"""RQ1: do two streams fail on the *same* images, or on different ones?

Ensemble accuracy is not what RQ1 asks about. Two streams can both score 0.90 AUC and
be worth fusing or worth nothing, depending entirely on whether their mistakes overlap.
So every measure here is computed on **paired errors at a fixed threshold**, not on
scores: for each sample, whether stream A was right and whether stream B was right.

The five measures, and what each one adds [R16][R18]:

* :func:`double_fault_rate` -- the share both get wrong. This is the ceiling on what
  any fusion can fix, and it is the number the go/no-go argument actually rests on.
* :func:`disagreement_rate` -- the share exactly one gets right. Fusion has something
  to work with only here.
* :func:`yules_q` -- disagreement normalised, so slices of different difficulty can be
  compared. +1 means the errors coincide completely, 0 means independent, -1 means
  complementary.
* :func:`mcnemar` -- whether the *asymmetry* in the disagreement is real, i.e. whether
  one stream is genuinely better rather than differently wrong.
* :func:`oracle_headroom` -- what a per-sample oracle would gain. An upper bound used
  for analysis, never reported as a method.
* :func:`jaccard_of_misses` -- of the images either stream misses, the share both miss
  (pre-registration Amendment 2).

:func:`macro_bootstrap` is the gate's interval (Amendment 2, I-061): it resamples each
slice's fakes within the slice and each view's reals within that view's pool, and
recomputes the *macro* M1 and M2 -- the statistic the point estimate is -- on every
replicate.

The thresholds these are computed at always come from validation
(:mod:`dbp.eval.thresholds`); nothing in this module chooses one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from dbp.eval.metrics import MetricError

#: Resamples for every bootstrap interval in the pre-registered plan.
BOOTSTRAP_RESAMPLES = 1000

#: Confidence level of the reported intervals.
CONFIDENCE = 0.95

#: Below this many discordant pairs, McNemar's chi-square approximation is unreliable
#: and the exact binomial test is used instead [R18].
EXACT_MCNEMAR_BELOW = 25


def _correct(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Boolean per-sample correctness at a threshold: score >= threshold means fake."""
    truth = np.asarray(y_true).astype(int).ravel()
    score = np.asarray(y_score, dtype=float).ravel()
    if truth.shape != score.shape:
        raise MetricError(f"y_true {truth.shape} and y_score {score.shape} differ")
    if truth.size == 0:
        raise MetricError("no samples")
    if not set(np.unique(truth)) <= {0, 1}:
        raise MetricError(f"y_true must be 0/1, got {sorted(set(np.unique(truth)))}")
    if not np.all(np.isfinite(score)):
        # Never dropped: a stream with a missing score is an error, not a smaller sample.
        raise MetricError(f"{int(np.sum(~np.isfinite(score)))} non-finite score(s)")
    return (score >= threshold).astype(int) == truth


def error_contingency(
    a_correct: Sequence[bool] | np.ndarray, b_correct: Sequence[bool] | np.ndarray
) -> dict[str, int]:
    """The 2x2 table every other measure in this module is derived from.

    Args:
        a_correct: Per-sample correctness of stream A.
        b_correct: Per-sample correctness of stream B.

    Returns:
        ``{"both_right", "a_only", "b_only", "both_wrong", "n"}``.

    Raises:
        MetricError: If the two arrays differ in length or are empty.
    """
    left = np.asarray(a_correct).astype(bool).ravel()
    right = np.asarray(b_correct).astype(bool).ravel()
    if left.shape != right.shape:
        raise MetricError(f"a_correct {left.shape} and b_correct {right.shape} differ")
    if left.size == 0:
        raise MetricError("no samples")
    return {
        "both_right": int(np.sum(left & right)),
        "a_only": int(np.sum(left & ~right)),
        "b_only": int(np.sum(~left & right)),
        "both_wrong": int(np.sum(~left & ~right)),
        "n": int(left.size),
    }


def double_fault_rate(table: Mapping[str, int]) -> float:
    """Share of samples both streams get wrong -- the ceiling on any fusion gain."""
    return table["both_wrong"] / table["n"]


def disagreement_rate(table: Mapping[str, int]) -> float:
    """Share of samples exactly one stream gets right."""
    return (table["a_only"] + table["b_only"]) / table["n"]


def yules_q(table: Mapping[str, int]) -> float | None:
    """Yule's Q over the error table: +1 coincident errors, 0 independent, -1 complementary.

    Args:
        table: Output of :func:`error_contingency`.

    Returns:
        Q, or ``None`` when it is undefined -- which happens whenever one stream is
        perfect or perfectly wrong on the slice. Returning 0.0 there would read as
        "independent" and would be a fabricated result.
    """
    ad = table["both_right"] * table["both_wrong"]
    bc = table["a_only"] * table["b_only"]
    if ad + bc == 0:
        return None
    return (ad - bc) / (ad + bc)


def jaccard_of_misses(table: Mapping[str, int]) -> float | None:
    """Of the samples either stream gets wrong, the share both get wrong.

    Returns:
        ``|both wrong| / |either wrong|``, or ``None`` when neither stream misses
        anything -- undefined, not 0, for the same reason as :func:`yules_q`.
    """
    either = table["both_wrong"] + table["a_only"] + table["b_only"]
    return None if either == 0 else table["both_wrong"] / either


def mcnemar(table: Mapping[str, int]) -> dict[str, Any]:
    """McNemar's test on the discordant pairs [R18].

    Uses the exact binomial test when there are few discordant pairs, where the
    chi-square approximation is not trustworthy, and the continuity-corrected
    chi-square otherwise. Which one ran is reported, because a p-value whose test is
    not named cannot be checked.

    Args:
        table: Output of :func:`error_contingency`.

    Returns:
        ``{"test", "statistic", "p_value", "discordant", "a_only", "b_only"}``.
    """
    a_only, b_only = table["a_only"], table["b_only"]
    discordant = a_only + b_only
    if discordant == 0:
        return {
            "test": "none",
            "statistic": None,
            "p_value": None,
            "discordant": 0,
            "a_only": a_only,
            "b_only": b_only,
            "note": "the two streams made identical errors; McNemar is undefined",
        }
    if discordant < EXACT_MCNEMAR_BELOW:
        result = stats.binomtest(a_only, discordant, 0.5)
        return {
            "test": "exact binomial",
            "statistic": float(a_only),
            "p_value": float(result.pvalue),
            "discordant": discordant,
            "a_only": a_only,
            "b_only": b_only,
        }
    statistic = (abs(a_only - b_only) - 1) ** 2 / discordant
    return {
        "test": "chi-square, continuity corrected",
        "statistic": float(statistic),
        "p_value": float(stats.chi2.sf(statistic, df=1)),
        "discordant": discordant,
        "a_only": a_only,
        "b_only": b_only,
    }


def score_correlation(
    a_score: Sequence[float] | np.ndarray, b_score: Sequence[float] | np.ndarray
) -> dict[str, Any]:
    """Spearman rank correlation between two streams' scores.

    Rank rather than Pearson, because the streams are on different scales and only
    their ordering is comparable.
    """
    left = np.asarray(a_score, dtype=float).ravel()
    right = np.asarray(b_score, dtype=float).ravel()
    if left.shape != right.shape:
        raise MetricError(f"a_score {left.shape} and b_score {right.shape} differ")
    if left.size < 3:
        raise MetricError("Spearman needs at least 3 samples")
    result = stats.spearmanr(left, right)
    return {"spearman": float(result.statistic), "p_value": float(result.pvalue)}


def oracle_correct(
    a_correct: Sequence[bool] | np.ndarray, b_correct: Sequence[bool] | np.ndarray
) -> np.ndarray:
    """Per-sample correctness of the per-sample oracle: right if either stream is.

    F-oracle is an upper bound for analysis only. It cannot be built, because choosing
    the right stream per sample requires the label.
    """
    return np.asarray(a_correct).astype(bool) | np.asarray(b_correct).astype(bool)


def pairwise(
    y_true: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Every measure, for every unordered pair of streams, on one slice.

    Args:
        y_true: 0 for real, 1 for fake.
        scores: Stream name -> per-sample score.
        thresholds: Stream name -> its validation-fitted threshold.

    Returns:
        A JSON-ready mapping keyed ``"a|b"``.

    Raises:
        MetricError: If a stream has no threshold. Substituting 0.5 would silently
            replace a calibrated decision with a naive one.
    """
    missing = sorted(set(scores) - set(thresholds))
    if missing:
        raise MetricError(
            f"no validation threshold for {missing}; every stream must carry the "
            "Calibration that produced it"
        )
    correct = {
        name: _correct(y_true, score, thresholds[name]) for name, score in scores.items()
    }
    names = sorted(scores)
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            table = error_contingency(correct[a], correct[b])
            accuracy_a = float(np.mean(correct[a]))
            accuracy_b = float(np.mean(correct[b]))
            out[f"{a}|{b}"] = {
                "contingency": table,
                "double_fault": double_fault_rate(table),
                "jaccard_of_misses": jaccard_of_misses(table),
                "disagreement": disagreement_rate(table),
                "yules_q": yules_q(table),
                "mcnemar": mcnemar(table),
                **score_correlation(scores[a], scores[b]),
                "accuracy_a": accuracy_a,
                "accuracy_b": accuracy_b,
                "oracle_accuracy": float(np.mean(oracle_correct(correct[a], correct[b]))),
                "oracle_headroom": float(
                    np.mean(oracle_correct(correct[a], correct[b]))
                    - max(accuracy_a, accuracy_b)
                ),
            }
    return out


def stratified_bootstrap_indices(
    strata: Sequence[Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 2026,
) -> list[np.ndarray]:
    """Index sets for a bootstrap that resamples within each stratum [R17].

    Resampling the pooled set would let a slice vanish from a replicate and make the
    macro-average over slices undefined, so each stratum keeps its own size.

    Args:
        strata: Per-sample stratum label.
        resamples: Number of replicates.
        seed: Base seed.

    Returns:
        One index array per replicate.

    Raises:
        MetricError: If there are no samples.
    """
    labels = np.asarray(strata)
    if labels.size == 0:
        raise MetricError("no samples")
    groups = [np.flatnonzero(labels == value) for value in np.unique(labels)]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        for _ in range(resamples)
    ]


def paired_difference_ci(
    statistic: Callable[[np.ndarray], float],
    strata: Sequence[Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 2026,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """Percentile CI of a paired difference, stratified by slice.

    "Paired" means the two systems are resampled **through the same indices**, so the
    interval is of their difference rather than of two independent estimates. That is
    what the go/no-go criterion needs: an interval on the gain, not two overlapping
    intervals on the levels.

    Args:
        statistic: Given an index array, returns the difference on that replicate.
        strata: Per-sample stratum label.
        resamples: Number of replicates.
        seed: Base seed.
        confidence: Interval level.

    Returns:
        ``{"point", "lo", "hi", "excludes_zero", "resamples", "confidence"}``.
    """
    indices = stratified_bootstrap_indices(strata, resamples, seed)
    all_rows = np.arange(len(np.asarray(strata)))
    point = float(statistic(all_rows))
    draws = np.array([statistic(index) for index in indices], dtype=float)
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        raise MetricError("every bootstrap replicate produced a non-finite statistic")
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(finite, 100 * alpha))
    hi = float(np.percentile(finite, 100 * (1.0 - alpha)))
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "resamples": int(finite.size),
        "confidence": confidence,
    }


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    bins: int = 15,
) -> float:
    """Expected calibration error over equal-width probability bins [R14].

    Args:
        y_true: 0/1 labels.
        y_prob: Predicted probability of the positive class, in ``[0, 1]``.
        bins: Number of equal-width bins.

    Returns:
        The ECE.

    Raises:
        MetricError: If the probabilities fall outside ``[0, 1]``.
    """
    truth = np.asarray(y_true).astype(int).ravel()
    prob = np.asarray(y_prob, dtype=float).ravel()
    if truth.shape != prob.shape:
        raise MetricError(f"y_true {truth.shape} and y_prob {prob.shape} differ")
    if truth.size == 0:
        raise MetricError("no samples")
    if prob.min() < 0.0 or prob.max() > 1.0:
        raise MetricError(
            f"ECE needs probabilities in [0, 1], got [{prob.min():.3f}, {prob.max():.3f}]; "
            "pass calibrated probabilities, not logits"
        )
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed except for the first bin, so 0.0 and 1.0 both land in a bin.
    index = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if not np.any(mask):
            continue
        error += mask.mean() * abs(prob[mask].mean() - truth[mask].mean())
    return float(error)


# --------------------------------------------------------------------------------------
# The gate's bootstrap: macro M1 and M2 over slices that share a real class (I-061).
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GateSlice:
    """One slice of the macro average, as row indices into the score arrays.

    Attributes:
        name: Slice name.
        fakes: Row indices of the slice's fakes.
        fake_units: Resampling unit of each fake (its parent image id).
        fake_group: Resampling group of the fakes; usually the slice itself.
        reals: Row indices of the view's real pool, or None (E8 has no reals).
        real_units: Resampling unit of each real.
        real_group: Resampling group of the reals: the view, so every slice of a view
            shares one draw of its reals.
    """

    name: str
    fakes: np.ndarray
    fake_units: tuple[str, ...]
    fake_group: str
    reals: np.ndarray | None = None
    real_units: tuple[str, ...] | None = None
    real_group: str | None = None


def macro_bootstrap(
    slices: Sequence[GateSlice],
    scores: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 2026,
) -> dict[str, dict[str, np.ndarray]]:
    """Macro M1 (TPR at the threshold) and M2 (AUC) per stream, on the data and on every
    replicate (pre-registration Amendment 2, I-061).

    Each replicate draws, with replacement, the units of every resampling group: a
    slice's fakes within the slice, a view's reals within the view's pool. Every member
    of a group (e.g. E7's reals and their luma companions) must hold exactly the same
    units, and all of them follow the same draw, so a luma row is resampled with its
    parent. The same draw is applied to every stream (paired).

    Args:
        slices: The macro's slices. M1 averages over all of them, M2 over those with
            reals.
        scores: Stream -> one finite score per row.
        thresholds: Stream -> its 5% FPR threshold (from the validation reals).
        resamples: Replicates.
        seed: The one generator's seed.

    Returns:
        ``{"M1": {stream: array}, "M2": {stream: array}}``; index 0 is the estimate on the
        data, indices 1.. the replicates. M2 is NaN throughout if no slice has reals.

    Raises:
        MetricError: On a non-finite score, a missing threshold, a duplicate unit in a
            group member, members of one group covering different units, an empty
            member, or a non-finite replicate.
    """
    from scipy.stats import rankdata

    names = list(scores)
    missing = [n for n in names if n not in thresholds]
    if missing:
        raise MetricError(f"no threshold for {missing}")
    matrix = np.column_stack([np.asarray(scores[n], dtype=float) for n in names])
    if not np.all(np.isfinite(matrix)):
        raise MetricError("non-finite score in the gate rows")
    cut = np.array([thresholds[n] for n in names], dtype=float)
    if not slices:
        raise MetricError("no gate slices")

    members: dict[str, list[tuple[str, np.ndarray, tuple[str, ...]]]] = {}
    for item in slices:
        members.setdefault(item.fake_group, []).append(
            (f"{item.name}:fakes", item.fakes, item.fake_units))
        if item.reals is not None:
            if item.real_units is None or item.real_group is None:
                raise MetricError(f"{item.name}: reals without units or group")
            members.setdefault(item.real_group, []).append(
                (f"{item.name}:reals", item.reals, item.real_units))

    # Every member of a group, as row positions over the group's sorted units.
    positions: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for group, entries in sorted(members.items()):
        reference: list[str] | None = None
        for key, rows, units in entries:
            if len(rows) == 0 or len(rows) != len(units):
                raise MetricError(f"group {group}: member {key} is empty or mis-sized")
            if len(set(units)) != len(units):
                raise MetricError(f"group {group}: a unit appears twice in {key}")
            order = sorted(units)
            if reference is None:
                reference = order
            elif order != reference:
                raise MetricError(f"group {group}: {key} covers different units")
            index = {u: k for k, u in enumerate(units)}
            positions[key] = np.asarray(rows)[[index[u] for u in reference]]
        sizes[group] = len(reference or [])

    def one(draw: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        tpr, auc = [], []
        for item in slices:
            fakes = matrix[positions[f"{item.name}:fakes"][draw[item.fake_group]]]
            tpr.append((fakes >= cut).mean(axis=0))
            if item.reals is not None:
                reals = matrix[positions[f"{item.name}:reals"][draw[item.real_group]]]
                ranks = rankdata(np.vstack([reals, fakes]), axis=0)
                n_r, n_f = len(reals), len(fakes)
                auc.append((ranks[n_r:].sum(axis=0) - n_f * (n_f + 1) / 2) / (n_f * n_r))
        m1 = np.mean(tpr, axis=0)
        m2 = np.mean(auc, axis=0) if auc else np.full(len(names), np.nan)
        return m1, m2

    results = [one({g: np.arange(n) for g, n in sizes.items()})]
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        results.append(one({g: rng.integers(0, n, n) for g, n in sorted(sizes.items())}))
    m1 = np.array([r[0] for r in results])
    m2 = np.array([r[1] for r in results])
    has_m2 = any(item.reals is not None for item in slices)
    if not np.all(np.isfinite(m1)) or (has_m2 and not np.all(np.isfinite(m2))):
        raise MetricError("a bootstrap replicate produced a non-finite macro")
    return {"M1": {n: m1[:, k] for k, n in enumerate(names)},
            "M2": {n: m2[:, k] for k, n in enumerate(names)}}


def percentile_interval(draws: np.ndarray, confidence: float = CONFIDENCE) -> dict[str, Any]:
    """Point (index 0) and percentile CI (indices 1..) of a bootstrapped statistic."""
    values = np.asarray(draws, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise MetricError("percentile_interval needs finite draws beyond the point")
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.percentile(values[1:], 100 * alpha))
    hi = float(np.percentile(values[1:], 100 * (1.0 - alpha)))
    return {"point": float(values[0]), "lo": lo, "hi": hi,
            "excludes_zero": bool(lo > 0.0 or hi < 0.0),
            "resamples": int(values.size - 1), "confidence": confidence}
