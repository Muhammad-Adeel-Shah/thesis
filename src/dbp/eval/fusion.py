"""Calibration and fusion, fitted on validation and nowhere else.

Three fusions, in increasing order of how much they can cheat:

* **F-mean** -- the mean of the temperature-scaled logits. Fits one number per stream
  (the temperature) and nothing else.
* **F-stack** -- logistic regression on the stream logits [R15]. This is the system the
  go/no-go gate is decided on, so its coefficients come from validation only.
* **F-oracle** -- right whenever any stream is right. It is not a method and cannot be
  built: choosing the right stream per sample needs the label. It is reported as the
  headroom that bounds what the other two could ever reach.

Everything here carries the split it was fitted on, in the same spirit as
:class:`dbp.eval.thresholds.Calibration`, and :meth:`FittedFusion.score` refuses a set
whose id matches. A stacker fitted on the set it is then reported on would turn a
NO-GO into a GO with no code change and no obvious error.

**Streams are joined by image id, never by position** (I-056): :func:`fit_by_id` and
:func:`score_by_id` take ``(ids, values)`` per stream and align them through
:func:`dbp.eval.join.join_streams`, and both refuse to run unless exactly the
configured streams (:data:`FUSION_STREAMS`) are present. A fusion records the streams
it was fitted on, and its callers assert that record against the configuration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize

from dbp.eval.join import JoinError, join_streams, require_streams
from dbp.eval.metrics import MetricError

#: The streams every fusion combines -- the five the committed ``p2_fit_heads.py`` fed
#: to both fusions before any panel score existed, made explicit (pre-registration
#: Amendment 2). A fusion over any other set is refused, never silently formed.
FUSION_STREAMS: tuple[str, ...] = ("aeroblade", "s1", "s2", "s3_lr", "s3_set")

#: Temperatures are searched in this band. A temperature outside it means the stream's
#: logits are pathological, which is worth an error rather than a silent clip.
TEMPERATURE_BOUNDS = (0.02, 50.0)


class FusionError(ValueError):
    """Raised when a fusion is fitted or applied in a way that would not be a result."""


def _logits(scores: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float).ravel()
    if values.size == 0:
        raise MetricError("no samples")
    if not np.all(np.isfinite(values)):
        raise MetricError("scores contain NaN or inf")
    return values


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def fit_temperature(
    y_true: Sequence[int] | np.ndarray, logits: Sequence[float] | np.ndarray
) -> float:
    """Temperature that minimises validation NLL [R14].

    Temperature scaling divides the logits by a single positive scalar. It cannot
    change any ranking, so AUC and AP are untouched and only the calibration moves --
    which is why it is safe to fit before comparing streams.

    Args:
        y_true: 0/1 labels from the validation split.
        logits: The stream's raw scores on that split.

    Returns:
        The fitted temperature.

    Raises:
        MetricError: If the split has only one class, which makes NLL degenerate.
    """
    truth = np.asarray(y_true).astype(float).ravel()
    values = _logits(logits)
    if truth.shape != values.shape:
        raise MetricError(f"y_true {truth.shape} and logits {values.shape} differ")
    if len(np.unique(truth)) < 2:
        raise MetricError(
            "temperature scaling needs both classes in the calibration split"
        )

    def nll(log_temperature: float) -> float:
        scaled = values / float(np.exp(log_temperature))
        probability = np.clip(sigmoid(scaled), 1e-12, 1 - 1e-12)
        return float(
            -np.mean(truth * np.log(probability) + (1 - truth) * np.log(1 - probability))
        )

    low, high = (float(np.log(b)) for b in TEMPERATURE_BOUNDS)
    result = optimize.minimize_scalar(nll, bounds=(low, high), method="bounded")
    if not result.success:
        raise MetricError(f"temperature search did not converge: {result.message}")
    return float(np.exp(result.x))


@dataclass(frozen=True)
class FittedFusion:
    """A fusion plus the evidence for it.

    Attributes:
        kind: ``"mean"`` or ``"stack"``.
        streams: Stream names, in the order the coefficients apply.
        temperatures: Per-stream temperature from :func:`fit_temperature`.
        coefficients: Stacker weights; empty for the mean.
        intercept: Stacker intercept; 0 for the mean.
        fitted_on: Id of the split everything above came from.
        n_fitted: Samples in that split.
    """

    kind: str
    streams: tuple[str, ...]
    temperatures: tuple[float, ...]
    fitted_on: str
    n_fitted: int
    coefficients: tuple[float, ...] = field(default=())
    intercept: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the report."""
        return {
            "kind": self.kind,
            "streams": list(self.streams),
            "temperatures": [round(t, 6) for t in self.temperatures],
            "coefficients": [round(c, 6) for c in self.coefficients],
            "intercept": round(self.intercept, 6),
            "fitted_on": self.fitted_on,
            "n_fitted": self.n_fitted,
        }

    def calibrated(self, scores: Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
        """Temperature-scaled logits as an ``(n, n_streams)`` matrix.

        Raises:
            FusionError: If a stream the fusion was fitted on is absent.
        """
        missing = [name for name in self.streams if name not in scores]
        if missing:
            raise FusionError(f"fusion was fitted on {missing}, which are not in scores")
        columns = [
            _logits(scores[name]) / temperature
            for name, temperature in zip(self.streams, self.temperatures, strict=True)
        ]
        widths = {c.shape for c in columns}
        if len(widths) > 1:
            raise FusionError(f"streams have different lengths: {widths}")
        return np.column_stack(columns)

    def score(
        self, scores: Mapping[str, Sequence[float] | np.ndarray], set_id: str
    ) -> np.ndarray:
        """Fused logits for a set the fusion was **not** fitted on.

        Args:
            scores: Stream name -> per-sample score.
            set_id: Id of the set being scored.

        Returns:
            One fused logit per sample.

        Raises:
            FusionError: If ``set_id`` is the split this fusion was fitted on.
        """
        if set_id == self.fitted_on:
            raise FusionError(
                f"this fusion was fitted on {set_id!r}; scoring it there measures the "
                "fit, not the method. Score the test panel instead."
            )
        matrix = self.calibrated(scores)
        if self.kind == "mean":
            return matrix.mean(axis=1)
        return matrix @ np.asarray(self.coefficients, dtype=float) + self.intercept


def fit_mean_fusion(
    y_true: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    fitted_on: str,
    expected_streams: Sequence[str] | None = None,
) -> FittedFusion:
    """F-mean: temperature-scale each stream, then average.

    Args:
        y_true: Validation labels.
        scores: Stream name -> validation scores.
        fitted_on: Id of the validation split.

    Returns:
        The fitted fusion.
    """
    if expected_streams is not None:
        require_streams(scores, expected_streams, "F-mean")
    names = tuple(sorted(scores))
    temperatures = tuple(fit_temperature(y_true, scores[name]) for name in names)
    return FittedFusion(
        kind="mean",
        streams=names,
        temperatures=temperatures,
        fitted_on=fitted_on,
        n_fitted=int(np.asarray(y_true).size),
    )


def fit_stack_fusion(
    y_true: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    fitted_on: str,
    l2: float = 1.0,
    max_iter: int = 1000,
    expected_streams: Sequence[str] | None = None,
) -> FittedFusion:
    """F-stack: L2 logistic regression on the temperature-scaled stream logits [R15].

    Args:
        y_true: Validation labels.
        scores: Stream name -> validation scores.
        fitted_on: Id of the validation split.
        l2: Inverse regularisation strength, scikit-learn's ``C``.
        max_iter: Solver iteration cap.

    Returns:
        The fitted fusion.

    Raises:
        MetricError: If the validation split has only one class.
    """
    from sklearn.linear_model import LogisticRegression

    if expected_streams is not None:
        require_streams(scores, expected_streams, "F-stack")
    truth = np.asarray(y_true).astype(int).ravel()
    if len(np.unique(truth)) < 2:
        raise MetricError("the stacker needs both classes in the validation split")

    names = tuple(sorted(scores))
    temperatures = tuple(fit_temperature(truth, scores[name]) for name in names)
    matrix = np.column_stack(
        [_logits(scores[name]) / t for name, t in zip(names, temperatures, strict=True)]
    )
    model = LogisticRegression(C=l2, max_iter=max_iter)
    model.fit(matrix, truth)
    return FittedFusion(
        kind="stack",
        streams=names,
        temperatures=temperatures,
        coefficients=tuple(float(c) for c in model.coef_.ravel()),
        intercept=float(model.intercept_[0]),
        fitted_on=fitted_on,
        n_fitted=int(truth.size),
    )


def oracle_scores(
    y_true: Sequence[int] | np.ndarray,
    scores: Mapping[str, Sequence[float] | np.ndarray],
    thresholds: Mapping[str, float],
) -> np.ndarray:
    """F-oracle: 1.0 where any stream is right at its threshold, 0.0 elsewhere.

    **Not a method.** It uses the labels, so it can only ever be an upper bound on
    what a fusion of these streams could achieve. Reporting it as a result would be
    fabrication; reporting the gap between it and F-stack is the analysis.

    Args:
        y_true: 0/1 labels.
        scores: Stream name -> per-sample score.
        thresholds: Stream name -> validation-fitted threshold.

    Returns:
        A 0/1 array, one entry per sample.

    Raises:
        FusionError: If a stream has no threshold.
    """
    missing = sorted(set(scores) - set(thresholds))
    if missing:
        raise FusionError(f"no threshold for {missing}")
    truth = np.asarray(y_true).astype(int).ravel()
    any_right = np.zeros(truth.shape, dtype=bool)
    for name, score in scores.items():
        prediction = (_logits(score) >= thresholds[name]).astype(int)
        any_right |= prediction == truth
    # Where the oracle is right it must agree with the label; where no stream is right
    # it must be wrong. Emitting the truth directly would make it trivially perfect.
    return np.where(any_right, truth, 1 - truth).astype(float)


def _aligned(
    tables: Mapping[str, tuple[Sequence[str], Any]],
    target_ids: Sequence[str],
    streams: Sequence[str],
    expected_count: int | None,
    what: str,
) -> dict[str, np.ndarray]:
    """The configured streams of ``tables``, joined to ``target_ids`` by id."""
    absent = [name for name in streams if name not in tables]
    if absent:
        raise JoinError(f"{what}: stream(s) {absent} absent; refusing to fuse without them")
    require_streams(tables, streams, what)
    return join_streams(target_ids, {name: tables[name] for name in streams},
                        expected_count=expected_count, what=what)


def fit_by_id(
    kind: str,
    labels: tuple[Sequence[str], Sequence[int]],
    tables: Mapping[str, tuple[Sequence[str], Any]],
    target_ids: Sequence[str],
    fitted_on: str,
    streams: Sequence[str] = FUSION_STREAMS,
    expected_count: int | None = None,
) -> FittedFusion:
    """Fit ``kind`` ("mean" or "stack") on streams joined to ``target_ids`` by id."""
    aligned = _aligned(tables, target_ids, streams, expected_count, f"fit {kind}")
    truth = join_streams(target_ids, {"labels": labels}, what=f"fit {kind} labels")["labels"]
    fit = {"mean": fit_mean_fusion, "stack": fit_stack_fusion}.get(kind)
    if fit is None:
        raise FusionError(f"unknown fusion kind {kind!r}")
    fusion = fit(truth, aligned, fitted_on, expected_streams=streams)
    require_streams(fusion.streams, streams, f"fit {kind} (recorded)")
    return fusion


def score_by_id(
    fusion: FittedFusion,
    tables: Mapping[str, tuple[Sequence[str], Any]],
    target_ids: Sequence[str],
    set_id: str,
    expected_count: int | None = None,
) -> np.ndarray:
    """Fused logits for ``target_ids``, every stream joined by id first."""
    aligned = _aligned(tables, target_ids, fusion.streams, expected_count,
                       f"score {fusion.kind}")
    return fusion.score(aligned, set_id)


# --------------------------------------------------------------------------------------
# F-mean(A2): the governing fusion of pre-registration Amendment 2.
# --------------------------------------------------------------------------------------

#: The governing fusion's members, fixed by design role (Amendment 2, point 1): S1, S2
#: and S3-Set, the thesis's designated S3 head. S3-LR and AEROBLADE are ablation and
#: baseline heads and never enter it.
GOVERNING_STREAMS: tuple[str, ...] = ("s1", "s2", "s3_set")

#: Robust-z clip and the MAD-to-standard-deviation constant (Amendment 2, point 2).
ROBUST_CLIP = 5.0
MAD_TO_SD = 1.4826


@dataclass(frozen=True)
class RobustMeanFusion:
    """Equal-weight mean of robust z-scores, every parameter from validation reals.

    Attributes:
        streams: Member streams, in order.
        medians: Per-stream median of the validation reals' scores.
        scales: Per-stream scale: ``1.4826 x MAD``, or the standard deviation where the
            MAD is 0.
        scale_rules: ``"mad"`` or ``"std"`` per stream, so a fallback is visible.
        clip: z is clipped to ``[-clip, +clip]``.
        fitted_on: Id of the split the reals came from.
        n_reals: How many reals that was.
    """

    streams: tuple[str, ...]
    medians: tuple[float, ...]
    scales: tuple[float, ...]
    scale_rules: tuple[str, ...]
    fitted_on: str
    n_reals: int
    clip: float = ROBUST_CLIP
    kind: str = "robust_mean"

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the report."""
        return {
            "kind": self.kind,
            "streams": list(self.streams),
            "medians": list(self.medians),
            "scales": list(self.scales),
            "scale_rules": list(self.scale_rules),
            "clip": self.clip,
            "fitted_on": self.fitted_on,
            "n_reals": self.n_reals,
            "uses_fake_labels": False,
        }

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any]) -> RobustMeanFusion:
        """Inverse of :meth:`to_dict`."""
        if spec.get("kind") != "robust_mean":
            raise FusionError(f"not a robust-mean fusion: kind {spec.get('kind')!r}")
        return cls(
            streams=tuple(spec["streams"]),
            medians=tuple(float(v) for v in spec["medians"]),
            scales=tuple(float(v) for v in spec["scales"]),
            scale_rules=tuple(spec["scale_rules"]),
            fitted_on=str(spec["fitted_on"]),
            n_reals=int(spec["n_reals"]),
            clip=float(spec["clip"]),
        )

    def normalised(self, scores: Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
        """Clipped robust z-scores as an ``(n, n_streams)`` matrix."""
        require_streams(scores, self.streams, "F-mean(A2) members")
        columns = [
            np.clip((_logits(scores[name]) - median) / scale, -self.clip, self.clip)
            for name, median, scale in zip(self.streams, self.medians, self.scales,
                                           strict=True)
        ]
        if len({c.shape for c in columns}) > 1:
            raise FusionError(f"streams have different lengths: {[c.shape for c in columns]}")
        return np.column_stack(columns)

    def transform(self, scores: Mapping[str, Sequence[float] | np.ndarray]) -> np.ndarray:
        """The fused score on any set, the calibration split included.

        Unlike :meth:`FittedFusion.score` this is allowed on the split the parameters
        came from, because only the reals' location and scale were taken from it: the
        fusion's own validation threshold has to be computed somewhere. Use
        :meth:`score` for the panel.
        """
        return self.normalised(scores).mean(axis=1)

    def score(
        self, scores: Mapping[str, Sequence[float] | np.ndarray], set_id: str
    ) -> np.ndarray:
        """The fused score for a set other than the calibration split."""
        if set_id == self.fitted_on:
            raise FusionError(f"F-mean(A2) was fitted on {set_id!r}; use transform() there")
        return self.transform(scores)


def fit_robust_mean(
    real_tables: Mapping[str, tuple[Sequence[str], Any]],
    real_ids: Sequence[str],
    fitted_on: str,
    streams: Sequence[str] = GOVERNING_STREAMS,
    expected_count: int | None = None,
) -> RobustMeanFusion:
    """Fit F-mean(A2) on the validation reals, every stream joined to them by id.

    Args:
        real_tables: Stream -> ``(ids, scores)``; may hold more ids than ``real_ids``.
        real_ids: The validation reals. Nothing else -- no fake -- is used.
        fitted_on: Id of the validation split.
        streams: The members; must be exactly :data:`GOVERNING_STREAMS` unless a test
            says otherwise.
        expected_count: If given, the number of reals required.

    Raises:
        JoinError: On a missing stream or id.
        FusionError: If a stream's reals have zero MAD and zero standard deviation.
    """
    absent = [name for name in streams if name not in real_tables]
    if absent:
        raise JoinError(f"F-mean(A2): stream(s) {absent} absent; refusing to fuse without them")
    aligned = join_streams(real_ids, {name: real_tables[name] for name in streams},
                           supersets=tuple(streams), expected_count=expected_count,
                           what="F-mean(A2) reals")
    medians, scales, rules = [], [], []
    for name in streams:
        values = _logits(aligned[name])
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 0:
            scale, rule = MAD_TO_SD * mad, "mad"
        else:
            scale, rule = float(np.std(values)), "std"
            if scale <= 0:
                raise FusionError(f"F-mean(A2): {name} is constant on the validation reals "
                                  "(MAD 0 and std 0); it cannot be normalised")
        medians.append(median)
        scales.append(scale)
        rules.append(rule)
    return RobustMeanFusion(streams=tuple(streams), medians=tuple(medians),
                            scales=tuple(scales), scale_rules=tuple(rules),
                            fitted_on=fitted_on, n_reals=len(real_ids))


def robust_score_by_id(
    fusion: RobustMeanFusion,
    tables: Mapping[str, tuple[Sequence[str], Any]],
    target_ids: Sequence[str],
    set_id: str | None,
    expected_count: int | None = None,
) -> np.ndarray:
    """F-mean(A2) for ``target_ids``, every stream joined by id first.

    ``set_id=None`` means the calibration split itself (:meth:`RobustMeanFusion.transform`).
    ``tables`` normally holds every head; the members are selected from it by name, and
    an absent member raises.
    """
    absent = [name for name in fusion.streams if name not in tables]
    if absent:
        raise JoinError(f"score F-mean(A2): member stream(s) {absent} absent")
    members = {name: tables[name] for name in fusion.streams}
    aligned = _aligned(members, target_ids, fusion.streams, expected_count, "score F-mean(A2)")
    if set_id is None:
        return fusion.transform(aligned)
    return fusion.score(aligned, set_id)


def fusion_from_dict(spec: Mapping[str, Any]) -> FittedFusion | RobustMeanFusion:
    """Rebuild a fusion from its report entry."""
    if spec.get("kind") == "robust_mean":
        return RobustMeanFusion.from_dict(spec)
    return FittedFusion(
        kind=str(spec["kind"]),
        streams=tuple(spec["streams"]),
        temperatures=tuple(float(t) for t in spec["temperatures"]),
        coefficients=tuple(float(c) for c in spec["coefficients"]),
        intercept=float(spec["intercept"]),
        fitted_on=str(spec["fitted_on"]),
        n_fitted=int(spec["n_fitted"]),
    )
