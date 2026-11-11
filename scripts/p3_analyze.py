"""P3 analysis: the frozen pre-registration's hypotheses on the final Panel B scores.

Implements ``docs/P3_PREREGISTRATION.md`` sections 4 and 6, and nothing more in the
confirmatory part. H1-a and H1-c were decided on development data (``p3_select_nstar.py``,
``p3_h1c.py``); this script covers the Panel B ones:

* **H1-b** (6.1, primary): folds 1-3 on Panel B's E7 subsets of the held-out lineage --
  fold 1 sd-vae-ft-ema, sd-vae-ft-mse, sd2-1; fold 2 sdxl-vae; fold 3 SD3.5 and FLUX.1
  luma. Success: each subset's AUC >= 0.85. 97.5% CIs (primary). FLUX.1 standard is
  reported beside fold 3 as an upper bound (section 6d shortcut).
* **H3** (6.2, primary): at k = 0 -- the threshold fixed at 5% FPR by the reals-only rule
  (``calibrate_on_reals``) on the union of the 491 COCO validation reals and the 400
  held-out Open Images reals -- on the Panel B **test** reals of E4 and E5, each counted
  once. Success, on **both** S2-D seeds: the macro achieved FPR <= 15%; neither pool
  above 25%; and a paired relative reduction against P2 ``full_bank`` (same union rule)
  of >= 50% whose 97.5% CI excludes 0. Descriptive: <= 10%, and the three-step table per
  pool (E3 included) and macro.
* **H3-k** (6.3, secondary): the k-shot methods of ``dbp.eval.kshot`` on each view's
  calibration pool; at k = 50, achieved test FPR <= 7.5% in >= 80% of draws, judged on
  E4 and E5, per method. E3 and E7 descriptive.
* **H2** (6.4, secondary): macro AUC over the four unseen slices, S2-D minus P2
  ``full_bank``; non-inferior if the 95% CI's lower bound > -2 points, on both seeds.
  Also without E3 (E4, E5 only); both verdicts are stated.
* **H4** (6.5, secondary): ``BANK-8`` (= S2-B seed 0) minus ``BANK-1`` on the macro AUC
  over the unseen slices; success >= +2 points with the 95% CI excluding 0. Descriptive:
  the curve for both orders with CIs, and each run's held-out-lineage validation AUC.
* **H5** (6.6, secondary, descriptive with a threshold): S2-D (seed 0, user 2026-10-22)
  under JPEG q50 and resize-and-restore 0.5, each within 5 AUC points of clean; all 12
  conditions as curves for S2-D, P2 ``full_bank`` and P2 P-A.

**Metrics (section 4).** A slice is one generator's fakes against the view's whole test
real pool; AUC and AP per slice; the macros average the four unseen slices (E3
GPT-Image2, E3 Nano-Banana2, E4, E5); every FPR rule is over the real pools E4 and E5.
Calibration reals are used only by H3-k. A score at or above the threshold is fake.

**Bootstrap (section 4).** 2,000 resamples, seed 2026, a fresh generator per analysis.
Per resample each real pool involved is drawn once (shared by every slice of its view
and by every model), then each fake slice within itself; groups are drawn in sorted
order. Comparisons are paired (one draw, both models). Thresholds are fixed, not
resampled. The interval is the percentile one: 95%, or 97.5% for the primary H1-b and H3.

**Decisions not spelled out by the frozen text** are the named constants in
``INTERPRETATIONS``, fixed in code before any real score is read.

**Exploratory.** Anything beyond the above goes under ``exploratory`` in the output,
clearly labelled; at this commit that section is empty.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.eval import kshot  # noqa: E402
from dbp.eval.metrics import average_precision, roc_auc, tpr_at_fpr  # noqa: E402
from dbp.eval.perturb import CONDITIONS, H5_DECISION  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

RESAMPLES, SEED = 2000, 2026
PRIMARY, SECONDARY = 0.975, 0.95
TARGET_FPR = 0.05
UNSEEN = (("e3_matched", "GPT-Image2-fake"), ("e3_matched", "Nano-Banana2-fake"),
          ("e4_wildrf_matched", "fake"), ("e5_chameleon_pipeline", "fake"))
UNSEEN_NO_E3 = UNSEEN[2:]
FPR_POOLS = ("e4_wildrf_matched", "e5_chameleon_pipeline")
THREE_STEP_POOLS = ("e3_matched",) + FPR_POOLS
LUMA = "e7_dda_coco_luma"
H1B = {1: (("e7_dda_coco", "sd-vae-ft-ema"), ("e7_dda_coco", "sd-vae-ft-mse"),
           ("e7_dda_coco", "stable-diffusion-2-1")),
       2: (("e7_dda_coco", "sdxl-vae"),),
       3: (("e7_dda_coco", "stable-diffusion-3.5-large"), (LUMA, "FLUX.1"))}
H1B_UPPER_BOUND = {3: ("e7_dda_coco", "FLUX.1")}
H1B_FLOOR, H3_MACRO, H3_POOL, H3_REDUCTION = 0.85, 0.15, 0.25, 0.50
H2_MARGIN, H4_GAIN, H5_POINTS, KSHOT_FPR, KSHOT_SHARE = -0.02, 0.02, 0.05, 0.075, 0.80
KSHOT = {"e3_matched": (10, 25, 50), "e4_wildrf_matched": (10, 50, 100),
         "e5_chameleon_pipeline": (10, 50, 100), "e7_dda_coco": (10, 50, 100)}
KSHOT_JUDGED = FPR_POOLS
S2D, FULL_BANK, P_A = ("s2d_p35_seed0", "s2d_p35_seed1"), "full_bank_seed0", "p_a_sd1_seed0"
H5_MODELS = ("s2d_p35_seed0", FULL_BANK, P_A)
BANK_CURVE = {"near": ((1, 1, "bank_1_seed0"), (2, 5, "bank_2_near_seed0"),
                       (4, 8, "bank_4_near_seed0"), (6, 10, "bank_6_near_seed0"),
                       (8, 12, "s2b_seed0")),
              "far": ((1, 1, "bank_1_seed0"), (2, 4, "bank_2_far_seed0"),
                      (4, 6, "bank_4_far_seed0"), (6, 9, "bank_6_far_seed0"),
                      (8, 12, "s2b_seed0"))}
ECE_BINS = 15

#: Readings of the frozen text where it does not spell a detail out. Fixed before any real
#: Panel B number was computed, and confirmed by the user on 2026-10-28 (before the run).
INTERPRETATIONS = {
    "point_and_interval": "a success criterion stated as a value 'with its CI excluding 0' "
                          "(H3 reduction, H4 gain) needs the point estimate to meet the "
                          "value AND the CI's lower bound > 0",
    "thresholds_on_points": "H1-b (>= 0.85), H3 (<= 15%, <= 25%) and H5 (within 5 points) "
                            "are judged on point estimates; their CIs are reported",
    "h3k_both_seeds": "H3-k is computed for S2-D seeds 0 and 1; a method passes only if it "
                      "passes on both (as H3 and H2 require both seeds)",
    "h3k_shift_reference": "(S)'s 'validation reals' are the 891 reals the k = 0 threshold "
                           "is fitted on (491 COCO validation + 400 held-out Open Images)",
    "h3k_judged_per_pool": "'judged on E4 and E5': each of E4 and E5 must reach the 80% "
                           "share",
    "h5_absolute": "'within 5 AUC points of clean' is |macro AUC(condition) - macro "
                   "AUC(clean)| <= 0.05 on the four unseen slices, Panel B test rows",
    "h3k_ece": "ECE is over the view's test reals and test fakes; (Q) applies no shift, so "
               "its ECE equals k = 0's",
}


class AnalysisError(ValueError):
    """Inputs the frozen analysis cannot be applied to."""


@dataclass
class Panel:
    """One model's test scores, grouped: real pools by view, fake slices by (view, slot)."""

    real: dict[str, np.ndarray]
    fake: dict[tuple[str, str], np.ndarray]
    calibration: dict[str, np.ndarray] = field(default_factory=dict)
    union: np.ndarray | None = None
    coco_val: np.ndarray | None = None


def read_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def scores_for(path: Path, rows: list[dict[str, str]]) -> np.ndarray:
    """A score file's scores, refused unless its ids are the list's, in order, all finite."""
    scored = read_rows(path)
    if [r["image_id"] for r in scored] != [r["image_id"] for r in rows]:
        raise AnalysisError(f"{path}: its image ids are not the list's, in order")
    values = np.array([float(r["score"]) for r in scored])
    if not np.all(np.isfinite(values)):
        raise AnalysisError(f"{path}: non-finite score")
    return values


def build_panel(rows: list[dict[str, str]], scores: np.ndarray,
                luma: tuple[list[dict[str, str]], np.ndarray] | None = None,
                threshold: tuple[list[dict[str, str]], np.ndarray] | None = None) -> Panel:
    """Group a model's scores: test reals by view, test fakes by slot; calibration apart."""
    real: dict[str, list[float]] = {}
    fake: dict[tuple[str, str], list[float]] = {}
    calibration: dict[str, list[float]] = {}
    pairs = list(zip(rows, scores, strict=True))
    if luma is not None:
        pairs += list(zip(*luma, strict=True))
    for row, score in pairs:
        if row["role"] == "calibration":
            calibration.setdefault(row["view"], []).append(score)
        elif row["label"] == "real":
            real.setdefault(row["view"], []).append(score)
        else:
            fake.setdefault((row["view"], row["slot"]), []).append(score)
    panel = Panel({k: np.asarray(v) for k, v in real.items()},
                  {k: np.asarray(v) for k, v in fake.items()},
                  {k: np.asarray(v) for k, v in calibration.items()})
    if threshold is not None:
        t_rows, t_scores = threshold
        panel.union = np.asarray(t_scores)
        panel.coco_val = np.asarray([s for r, s in zip(t_rows, t_scores, strict=True)
                                     if r["view"] == "coco_val"])
    return panel


def load_clean(lists_dir: Path, scores_root: Path, model: str) -> Panel:
    """A model's clean-stage Panel B, luma and threshold-real scores."""
    lists = {n: read_rows(lists_dir / f"{n}.csv") for n in ("panel_b", "luma", "threshold_reals")}
    scored = {n: scores_for(scores_root / model / "clean" / f"{n}.csv", rows)
              for n, rows in lists.items()}
    return build_panel(lists["panel_b"], scored["panel_b"], (lists["luma"], scored["luma"]),
                       (lists["threshold_reals"], scored["threshold_reals"]))


def load_condition(lists_dir: Path, scores_root: Path, model: str, condition: str) -> Panel:
    """A model's Panel B test rows under one H5 condition (clean: the clean stage's)."""
    rows = [r for r in read_rows(lists_dir / "panel_b.csv") if r["role"] == "test"]
    if condition == "clean":
        every = read_rows(lists_dir / "panel_b.csv")
        all_scores = scores_for(scores_root / model / "clean" / "panel_b.csv", every)
        values = np.array([s for r, s in zip(every, all_scores, strict=True)
                           if r["role"] == "test"])
    else:
        values = scores_for(scores_root / model / "robust" / f"{condition}.csv", rows)
    return build_panel(rows, values)


def auc(reals: np.ndarray, fakes: np.ndarray) -> float:
    """AUC of fakes (positive) against reals."""
    return float(roc_auc(np.r_[np.zeros(len(reals)), np.ones(len(fakes))], np.r_[reals, fakes]))


def slice_auc(panel: Panel, key: tuple[str, str]) -> float:
    """One slice: its fakes against its view's whole test real pool."""
    return auc(panel.real[key[0]], panel.fake[key])


def macro_auc(panel: Panel, slices: tuple[tuple[str, str], ...] = UNSEEN) -> float:
    """Mean slice AUC."""
    return float(np.mean([slice_auc(panel, s) for s in slices]))


def fpr(panel: Panel, view: str, threshold: float, shift: float = 0.0) -> float:
    """Achieved FPR on a view's test reals."""
    return float(np.mean(panel.real[view] - shift >= threshold))


def k0_threshold(panel: Panel, which: str = "union") -> float:
    """The k = 0 threshold: 5% FPR by the reals-only rule on the union (or COCO-only) reals."""
    reals = panel.union if which == "union" else panel.coco_val
    if reals is None or reals.size == 0:
        raise AnalysisError(f"no {which} threshold reals")
    return calibrate_on_reals(reals, TARGET_FPR, f"p3 {which} threshold reals").threshold


def groups_of(slices: tuple[tuple[str, str], ...], pools: tuple[str, ...] = ()) -> list[Any]:
    """The resampling groups a statistic touches: its real pools and its fake slices."""
    views = sorted({s[0] for s in slices} | set(pools))
    return [("real", v) for v in views] + [("fake", s) for s in sorted(slices)]


def resample(panel: Panel, idx: dict[Any, np.ndarray]) -> Panel:
    """The panel with every drawn group replaced by its resample."""
    real = dict(panel.real)
    fake = dict(panel.fake)
    for (kind, key), take in idx.items():
        if kind == "real":
            real[key] = panel.real[key][take]
        else:
            fake[key] = panel.fake[key][take]
    return Panel(real, fake, panel.calibration, panel.union, panel.coco_val)


def bootstrap(panels: dict[str, Panel], groups: list[Any],
              statistic: Callable[[dict[str, Panel]], float],
              level: float = SECONDARY, resamples: int | None = None,
              seed: int = SEED) -> dict[str, Any]:
    """Point estimate and percentile CI; one draw per group per resample, shared by models.

    ``resamples`` defaults to the module's ``RESAMPLES`` (2,000) at call time.
    """
    resamples = resamples or RESAMPLES
    point = float(statistic(panels))
    sizes = {}
    for kind, key in groups:
        lengths = {len((p.real if kind == "real" else p.fake)[key]) for p in panels.values()}
        if len(lengths) != 1:
            raise AnalysisError(f"group {kind} {key} differs in size across models")
        sizes[(kind, key)] = lengths.pop()
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        idx = {g: rng.integers(0, sizes[g], sizes[g]) for g in groups}
        values[b] = statistic({m: resample(p, idx) for m, p in panels.items()})
    tail = (1.0 - level) / 2.0 * 100.0
    finite = values[np.isfinite(values)]
    if finite.size < resamples:
        raise AnalysisError(f"{resamples - finite.size} non-finite bootstrap replicate(s)")
    return {"point": point, "lower": float(np.percentile(finite, tail)),
            "upper": float(np.percentile(finite, 100.0 - tail)), "level": level,
            "resamples": resamples, "seed": seed}


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error, ``bins`` equal-width bins on [0, 1]."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(total)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# --------------------------------------------------------------------- descriptive ---


def describe(panel: Panel) -> dict[str, Any]:
    """Section 4 for one model: per slice AUC/AP, TPR at k = 0, slice-calibrated
    TPR@5%FPR; per pool achieved FPR at the union and COCO-only thresholds; E8 TPR."""
    union, coco = k0_threshold(panel, "union"), k0_threshold(panel, "coco")
    slices = {}
    for key, fakes in sorted(panel.fake.items()):
        out: dict[str, Any] = {"fakes": int(len(fakes)),
                               "tpr_k0_union": float(np.mean(fakes >= union))}
        if key[0] in panel.real:
            reals = panel.real[key[0]]
            y = np.r_[np.zeros(len(reals)), np.ones(len(fakes))]
            s = np.r_[reals, fakes]
            out.update({"reals": int(len(reals)), "auc": auc(reals, fakes),
                        "ap": float(average_precision(y, s)),
                        "tpr_at_5pct_slice_calibrated": tpr_at_fpr(y, s, TARGET_FPR)[0]})
        slices[f"{key[0]}/{key[1]}"] = out
    pools = {view: {"reals": int(len(r)), "fpr_union": float(np.mean(r >= union)),
                    "fpr_coco_only": float(np.mean(r >= coco))}
             for view, r in sorted(panel.real.items())}
    return {"threshold_union": union, "threshold_coco_only": coco,
            "macro_auc_unseen": macro_auc(panel), "slices": slices, "pools": pools}


# --------------------------------------------------------------------- hypotheses ---


def h1b(lolo: dict[int, Panel]) -> dict[str, Any]:
    """H1-b: folds 1-3 on their held-out lineage's E7 subsets; each subset AUC >= 0.85."""
    folds = {}
    for fold, subsets in H1B.items():
        panel = lolo[fold]
        rows = {}
        for key in subsets:
            ci = bootstrap({"m": panel}, groups_of((key,)),
                           lambda p, k=key: slice_auc(p["m"], k), level=PRIMARY)
            rows[f"{key[0]}/{key[1]}"] = {**ci, "passes": ci["point"] >= H1B_FLOOR}
        upper = {}
        if fold in H1B_UPPER_BOUND:
            key = H1B_UPPER_BOUND[fold]
            upper[f"{key[0]}/{key[1]}"] = {
                **bootstrap({"m": panel}, groups_of((key,)),
                            lambda p, k=key: slice_auc(p["m"], k), level=PRIMARY),
                "note": "upper bound: E7's FLUX.1 4:2:0-vs-4:4:4 shortcut (section 6d)"}
        folds[str(fold)] = {"subsets": rows, "upper_bound": upper,
                            "verdict": "PASS" if all(r["passes"] for r in rows.values())
                            else "FAIL"}
    passed = sum(f["verdict"] == "PASS" for f in folds.values())
    return {"folds": folds, "folds_passing": passed,
            "claim_needs": "H1-a PASS and H1-b on >= 2 of 3 folds (section 7)",
            "h1b_on_two_of_three": passed >= 2}


def h3(panels: dict[str, Panel]) -> dict[str, Any]:
    """H3: k = 0 union threshold, E4 and E5 test reals; both S2-D seeds vs P2 full_bank."""
    fb = panels[FULL_BANK]
    thr = {m: k0_threshold(p, "union") for m, p in panels.items()}
    coco = k0_threshold(fb, "coco")
    per_seed = {}
    for model in S2D:
        pair = {"s2d": panels[model], "fb": fb}

        def macro(p: dict[str, Panel], who: str, t: dict[str, float]) -> float:
            return float(np.mean([fpr(p[who], v, t[who]) for v in FPR_POOLS]))

        t = {"s2d": thr[model], "fb": thr[FULL_BANK]}
        pools = {v: fpr(panels[model], v, thr[model]) for v in FPR_POOLS}
        macro_s2d = macro(pair, "s2d", t)
        macro_fb = macro(pair, "fb", t)
        if macro_fb == 0:
            raise AnalysisError("full_bank's macro FPR is 0: a relative reduction is undefined")

        def reduction(p: dict[str, Panel], t: dict[str, float] = t) -> float:
            base = macro(p, "fb", t)
            return float("nan") if base == 0 else 1.0 - macro(p, "s2d", t) / base

        ci = bootstrap(pair, groups_of((), FPR_POOLS), reduction, level=PRIMARY)
        criteria = {"macro_fpr_le_15": macro_s2d <= H3_MACRO,
                    "no_pool_above_25": all(v <= H3_POOL for v in pools.values()),
                    "reduction_ge_50": ci["point"] >= H3_REDUCTION,
                    "reduction_ci_excludes_0": ci["lower"] > 0}
        steps = {}
        for view in THREE_STEP_POOLS:
            steps[view] = {"full_bank_coco_only": fpr(fb, view, coco),
                           "full_bank_union": fpr(fb, view, thr[FULL_BANK]),
                           "s2d_union": fpr(panels[model], view, thr[model])}
        steps["macro_e4_e5"] = {k: float(np.mean([steps[v][k] for v in FPR_POOLS]))
                                for k in steps[FPR_POOLS[0]]}
        per_seed[model] = {"threshold_union": thr[model], "fpr": pools, "macro_fpr": macro_s2d,
                           "full_bank_macro_fpr": macro_fb, "relative_reduction": ci,
                           "criteria": criteria, "passes": all(criteria.values()),
                           "descriptive_macro_le_10": macro_s2d <= 0.10,
                           "three_step": steps}
    return {"seeds": per_seed, "full_bank_threshold_union": thr[FULL_BANK],
            "full_bank_threshold_coco_only": coco,
            "verdict": "PASS" if all(s["passes"] for s in per_seed.values()) else "FAIL"}


def kshot_view(panel: Panel, view: str, val_reals: np.ndarray, thr0: float) -> dict[str, Any]:
    """H3-k on one view for one model: (Q) and (S) at every k of the view."""
    cal = panel.calibration[view]
    reals = panel.real[view]
    fakes = np.concatenate([f for (v, _), f in sorted(panel.fake.items()) if v == view])
    labels = np.r_[np.zeros(len(reals)), np.ones(len(fakes))]
    scores = np.r_[reals, fakes]
    base = {"fpr": float(np.mean(reals >= thr0)), "tpr": float(np.mean(fakes >= thr0)),
            "ece": ece(sigmoid(scores), labels)}
    out: dict[str, Any] = {"k0": base, "pool": int(len(cal)), "Q": {}, "S": {}}
    for k in KSHOT[view]:
        draws = kshot.draws(view, len(cal), k)
        for method in ("Q", "S"):
            fprs, tprs, eces = [], [], []
            rule = None
            for draw in draws:
                if method == "Q":
                    thr, rule = kshot.quantile_threshold(cal[draw])
                    shift = 0.0
                else:
                    thr, shift = thr0, kshot.logit_shift(cal[draw], val_reals)
                fprs.append(kshot.achieved_fpr(reals, thr, shift))
                tprs.append(float(np.mean(fakes - shift >= thr)))
                eces.append(ece(sigmoid(scores - shift), labels))
            cell: dict[str, Any] = {"draws": len(draws), "mean_fpr": float(np.mean(fprs)),
                                    "mean_tpr": float(np.mean(tprs)),
                                    "mean_ece": float(np.mean(eces))}
            if method == "Q" and rule is not None:
                cell["j"], cell["bound_met"] = rule.j, rule.bound_met
            if len(draws) == 1:
                draw = draws[0]
                fixed = (kshot.quantile_threshold(cal[draw])[0], 0.0) if method == "Q" else \
                    (thr0, kshot.logit_shift(cal[draw], val_reals))
                cell["single_draw_fpr"] = bootstrap(
                    {"m": panel}, groups_of((), (view,)),
                    lambda p, f=fixed: fpr(p["m"], view, f[0], f[1]))
            else:
                cell["share_fpr_le_7_5"] = float(np.mean(np.asarray(fprs) <= KSHOT_FPR))
            out[method][str(k)] = cell
    return out


def h3k(panels: dict[str, Panel]) -> dict[str, Any]:
    """H3-k for both S2-D seeds; per method, at k = 50, >= 80% of draws <= 7.5% on E4, E5."""
    seeds = {}
    for model in S2D:
        panel = panels[model]
        thr0 = k0_threshold(panel, "union")
        views = {v: kshot_view(panel, v, panel.union, thr0) for v in KSHOT}
        verdict = {m: all(views[v][m]["50"]["share_fpr_le_7_5"] >= KSHOT_SHARE
                          for v in KSHOT_JUDGED) for m in ("Q", "S")}
        seeds[model] = {"views": views, "passes_at_k50": verdict}
    return {"seeds": seeds, "verdict": {m: "PASS" if all(s["passes_at_k50"][m]
                                                        for s in seeds.values()) else "FAIL"
                                        for m in ("Q", "S")}}


def h2(panels: dict[str, Panel]) -> dict[str, Any]:
    """H2: S2-D minus full_bank macro AUC; non-inferior if the 95% lower bound > -2 points."""
    out: dict[str, Any] = {}
    for name, slices in (("four_unseen_slices", UNSEEN), ("without_e3", UNSEEN_NO_E3)):
        seeds = {}
        for model in S2D:
            pair = {"s2d": panels[model], "fb": panels[FULL_BANK]}
            ci = bootstrap(pair, groups_of(slices),
                           lambda p, s=slices: macro_auc(p["s2d"], s) - macro_auc(p["fb"], s))
            seeds[model] = {**ci, "non_inferior": ci["lower"] > H2_MARGIN}
        out[name] = {"seeds": seeds, "verdict": "PASS" if all(
            s["non_inferior"] for s in seeds.values()) else "FAIL"}
    out["verdicts_differ"] = out["four_unseen_slices"]["verdict"] != out["without_e3"]["verdict"]
    return out


def h4(panels: dict[str, Panel], heldout: dict[str, Any]) -> dict[str, Any]:
    """H4: BANK-8 minus BANK-1 macro AUC (>= +2 points, CI excludes 0); the curves."""
    pair = {"b8": panels["s2b_seed0"], "b1": panels["bank_1_seed0"]}
    ci = bootstrap(pair, groups_of(UNSEEN),
                   lambda p: macro_auc(p["b8"]) - macro_auc(p["b1"]))
    criteria = {"gain_ge_2_points": ci["point"] >= H4_GAIN, "ci_excludes_0": ci["lower"] > 0}
    curves = {}
    for order, points in BANK_CURVE.items():
        curves[order] = [{"k": k, "decoders": d, "model": m,
                          "macro_auc": bootstrap({"m": panels[m]}, groups_of(UNSEEN),
                                                 lambda p: macro_auc(p["m"])),
                          "heldout_lineage_val_auc": heldout.get(m)} for k, d, m in points]
    bank_1l = bootstrap({"m": panels["bank_1l_seed0"]}, groups_of(UNSEEN),
                        lambda p: macro_auc(p["m"]))
    return {"bank8_minus_bank1": ci, "criteria": criteria,
            "verdict": "PASS" if all(criteria.values()) else "FAIL", "curves": curves,
            "bank_1l_descriptive": {"macro_auc": bank_1l,
                                    "heldout_lineage_val_auc": heldout.get("bank_1l_seed0")}}


def h5(conditions: dict[str, dict[str, Panel]]) -> dict[str, Any]:
    """H5: S2-D seed 0 within 5 points of clean under JPEG q50 and resize 0.5; curves."""
    curves = {}
    for model, by_condition in conditions.items():
        curves[model] = {c: bootstrap({"m": p}, groups_of(UNSEEN), lambda q: macro_auc(q["m"]))
                         for c, p in by_condition.items()}
    s2d = curves[H5_MODELS[0]]
    decision = {c: {"clean": s2d["clean"]["point"], "condition": s2d[c]["point"],
                    "difference": s2d[c]["point"] - s2d["clean"]["point"],
                    "within_5_points": abs(s2d[c]["point"] - s2d["clean"]["point"])
                    <= H5_POINTS} for c in H5_DECISION}
    return {"model": H5_MODELS[0], "decision": decision, "curves": curves,
            "verdict": "PASS" if all(d["within_5_points"] for d in decision.values())
            else "FAIL"}


def heldout_aucs(heldout_dir: Path) -> dict[str, Any]:
    """Each BANK run's held-out-lineage validation AUC, from its committed sidecar."""
    out = {}
    for path in sorted(heldout_dir.glob("bank_*_seed0_step_*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        out[path.name.split("_step_")[0]] = meta.get("heldout_auc")
    return out


def run(lists_dir: Path, scores_root: Path, heldout_dir: Path, models: list[str],
        robust: bool, log: Any) -> dict[str, Any]:
    """Every section-4 description and every Panel B hypothesis."""
    panels = {m: load_clean(lists_dir, scores_root, m) for m in models}
    log.info("loaded %d models", len(panels))
    result: dict[str, Any] = {
        "what": "P3 Panel B analysis, the frozen pre-registration (sections 4 and 6)",
        "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "primary_level": PRIMARY,
                      "secondary_level": SECONDARY},
        "interpretations": INTERPRETATIONS,
        "descriptive": {m: describe(p) for m, p in panels.items()},
    }
    lolo = {f: panels[f"lolo_f{f}_seed0"] for f in H1B}
    result["confirmatory"] = {"H1b": h1b(lolo), "H3": h3(panels), "H3k": h3k(panels),
                              "H2": h2(panels), "H4": h4(panels, heldout_aucs(heldout_dir))}
    if robust:
        result["confirmatory"]["H5"] = h5({m: {c: load_condition(lists_dir, scores_root, m, c)
                                               for c in CONDITIONS} for m in H5_MODELS})
    else:
        result["confirmatory"]["H5"] = {"verdict": "NOT-RUN",
                                        "reason": "the robust-stage scores are not present"}
    result["exploratory"] = {"note": "nothing exploratory at this commit"}
    return result


def main() -> int:
    """Run the analysis; write the JSON atomically."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lists", type=Path, required=True)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p3_heldout_scores")
    parser.add_argument("--expect", type=Path,
                        default=THESIS_DBP / "env" / "p3_final_expect.json")
    parser.add_argument("--robust", action="store_true",
                        help="Also H5 (needs the robust-stage scores).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log = setup_logging("p3_analyze")
    models = [m["id"] for m in json.loads(args.expect.read_text(encoding="utf-8"))["models"]]
    try:
        result = run(args.lists, args.scores_root, args.heldout_dir, models, args.robust, log)
    except AnalysisError as error:
        log.error("%s", error)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, args.out)
    for name, block in result["confirmatory"].items():
        log.info("%s: %s", name, block.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
