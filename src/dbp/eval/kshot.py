"""H3-k few-shot calibration, pre-registration section 6.3 -- fixed before Panel B is scored.

Pre-registration section 5 lists "the k-shot methods of section 6.3" among the values
that must be committed before Panel B is scored; this module is that commitment, and
the final-scoring pre-flight refuses to run unless it reproduces the frozen numbers.

* **Draws.** k reals from a view's calibration pool, 1,000 draws per k, seed 2026. Each
  (pool, k) has its own generator, ``default_rng([2026, k, crc32(pool)])``, so a draw
  does not depend on which other pools or k were drawn first. When k equals the pool
  size there is exactly one draw (the whole pool).
* **(Q) Conformal-style quantile.** The threshold is the j-th highest of the k reals, j
  the largest value with ``P(Binomial(k, 0.075) <= j - 1) <= 0.10``. When no j meets
  the bound (k = 10, 25), j = 1 is used and the bound is reported as not met.
* **(S) Logit shift.** Scores are shifted by ``median(k reals) - median(validation
  reals)``; the k = 0 threshold is then applied.

A score at or above the threshold is called fake, as everywhere in the project
(:func:`dbp.eval.thresholds.calibrate_on_reals`).
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

SEED = 2026
DRAWS = 1000
TARGET_FPR = 0.075
BOUND = 0.10


class KShotError(ValueError):
    """Inputs a k-shot method cannot be applied to."""


@dataclass(frozen=True)
class QuantileRule:
    """The (Q) rule for one k: the rank j and whether the binomial bound is met."""

    k: int
    j: int
    bound_met: bool
    probability: float


def quantile_rule(k: int) -> QuantileRule:
    """j for (Q): the largest j with ``P(Binomial(k, 0.075) <= j - 1) <= 0.10``.

    Raises:
        KShotError: If k < 1.
    """
    if k < 1:
        raise KShotError(f"k must be at least 1, got {k}")
    best = 0
    for j in range(1, k + 1):
        if binom.cdf(j - 1, k, TARGET_FPR) <= BOUND:
            best = j
        else:
            break
    if best == 0:
        return QuantileRule(k, 1, False, float(binom.cdf(0, k, TARGET_FPR)))
    return QuantileRule(k, best, True, float(binom.cdf(best - 1, k, TARGET_FPR)))


def quantile_threshold(k_reals: Sequence[float] | np.ndarray) -> tuple[float, QuantileRule]:
    """(Q): the j-th highest of the k calibration reals, and the rule that chose j."""
    scores = np.sort(np.asarray(k_reals, dtype=np.float64))[::-1]
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise KShotError("(Q) needs at least one finite calibration score")
    rule = quantile_rule(int(scores.size))
    return float(scores[rule.j - 1]), rule


def logit_shift(k_reals: Sequence[float] | np.ndarray,
                validation_reals: Sequence[float] | np.ndarray) -> float:
    """(S): the shift ``median(k reals) - median(validation reals)``, subtracted from scores."""
    k = np.asarray(k_reals, dtype=np.float64)
    v = np.asarray(validation_reals, dtype=np.float64)
    if k.size == 0 or v.size == 0 or not (np.all(np.isfinite(k)) and np.all(np.isfinite(v))):
        raise KShotError("(S) needs finite calibration and validation scores")
    return float(np.median(k) - np.median(v))


def draws(pool: str, pool_size: int, k: int, n: int = DRAWS) -> list[np.ndarray]:
    """The k-subsets of a calibration pool (indices), seeded per (pool, k).

    One draw when ``k == pool_size`` (the whole pool); ``n`` draws without replacement
    within each draw otherwise.

    Raises:
        KShotError: If k is outside ``1..pool_size``.
    """
    if not 1 <= k <= pool_size:
        raise KShotError(f"k = {k} is outside 1..{pool_size} for pool {pool}")
    if k == pool_size:
        return [np.arange(pool_size)]
    rng = np.random.default_rng([SEED, k, zlib.crc32(pool.encode("utf-8"))])
    return [np.sort(rng.choice(pool_size, size=k, replace=False)) for _ in range(n)]


def achieved_fpr(test_reals: Sequence[float] | np.ndarray, threshold: float,
                 shift: float = 0.0) -> float:
    """Share of test reals called fake: ``score - shift >= threshold``."""
    reals = np.asarray(test_reals, dtype=np.float64)
    if reals.size == 0:
        raise KShotError("no test reals")
    return float(np.mean(reals - shift >= threshold))
