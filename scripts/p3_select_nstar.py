"""P3 nested N* selector and H1-a (pre-registration sections 3 and 6.1, frozen 2026-10-08).

**The frozen rule.** Every LOLO fold trained ``N_max`` = 6,000 steps and evaluated its
held-out lineage every 500 steps. On that 500-step grid:

- ``N*(-f)`` is the step maximising the mean held-out-lineage AUC of the *other seven*
  folds; fold f's own numbers are reported there;
- ``N*`` is the step maximising the mean over all eight folds; every non-LOLO model
  trains for exactly ``N*`` steps.

**Tie-break.** The frozen text specifies none. The earliest tied step is taken (user
decision, 2026-10-17, fixed before this selector first ran on real reports), and every
tie is listed in the output.

**Exact arithmetic.** The training reports carry each AUC rounded to 5 decimals. Each is
converted to an integer number of 1e-5 units and the means are compared as integer sums,
so a tie is a tie and never a floating-point accident. A value with more decimals is
refused rather than silently rounded.

**H1-a.** ``G`` = the mean over the 8 folds of the held-out AUC at ``N*(-f)``. Success:
``G >= 0.90``, the 97.5% CI lower bound ``>= 0.85``, and no fold below ``0.75``.

**The CI** (pre-registration section 4: bootstrap, 2,000 resamples, seed 2026, fakes
resampled within slice, reals within the view's real pool; 97.5% for a primary
hypothesis). It needs per-image scores, which the training reports do not hold, so it is
a separate subcommand (``ci``) that reads the frozen selection from ``select``'s output
and never re-selects. Per resample:

1. the validation reals -- one pool, the same images in every fold -- are resampled once
   with replacement, and the same draw is applied to every fold (paired across folds);
2. each fold's held-out reconstructions (its slice) are resampled with replacement,
   independently per fold, in fold order;
3. each fold's AUC is recomputed and ``G`` is their mean.

The 8 folds are not resampled: they are every lineage of the bank, not a sample, and
``N*(-f)`` is held fixed (it is a section-5 value, chosen once). The interval is the
1.25th-98.75th percentile of ``G`` over the resamples (two-sided 97.5%).

**Run once.** ``select`` refuses to overwrite an existing output, so the real selection
can only be made once; dry-runs write elsewhere.

Usage::

    python scripts/p3_select_nstar.py select --report <fold report> (x8) [--out ...]
    python scripts/p3_select_nstar.py ci --nstar reports/p3_nstar.json \\
        --scores-dir reports/p3_heldout_scores
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.heldout_scores import HeldoutScoreError  # noqa: E402
from dbp.eval.heldout_scores import read_scores as read_heldout_scores  # noqa: E402
from dbp.eval.metrics import roc_auc  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: Frozen values (pre-registration sections 3, 4 and 6.1; set by the user 2026-10-08).
N_MAX = 6000
GRID_EVERY = 500
GRID = tuple(range(GRID_EVERY, N_MAX + 1, GRID_EVERY))
FOLDS = tuple(range(1, 9))
SEED = 0
G_MIN = 0.90
CI_LOWER_MIN = 0.85
FOLD_FLOOR = 0.75
CI_LEVEL = 0.975
RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
TIE_BREAK = "earliest step (frozen text silent; user decision 2026-10-17, before the run)"

#: AUCs in the reports are rounded to 5 decimals; selection runs on integer 1e-5 units.
UNITS = 100_000
#: A rescored fold AUC must reproduce its training-time value within this (CI step).
REPRODUCE_TOL = 1e-3
FOLD_PATTERN = re.compile(r"lolo_f(\d+)")


class SelectionError(ValueError):
    """A fold report or score file that the frozen rule cannot be applied to."""


@dataclass(frozen=True)
class FoldCurve:
    """One LOLO fold's held-out curve on the frozen grid."""

    fold: int
    path: str
    sha256: str
    units: dict[int, int]                 # step -> held-out AUC in 1e-5 units
    points: dict[int, dict[str, Any]]     # step -> the report's curve point


def to_units(value: Any, where: str) -> int:
    """An AUC with at most 5 decimals as an exact integer number of 1e-5 units."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise SelectionError(f"{where}: held-out AUC {value!r} is not a finite number")
    if not 0.0 <= value <= 1.0:
        raise SelectionError(f"{where}: held-out AUC {value} is outside [0, 1]")
    units = round(value * UNITS)
    if abs(units / UNITS - value) > 1e-9:
        raise SelectionError(f"{where}: held-out AUC {value} has more than 5 decimals")
    return int(units)


def expected_lineages(runs_config: Path) -> dict[int, list[str]]:
    """Fold number -> its held-out lineages, from the run config."""
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(runs_config, bank)
    out: dict[int, list[str]] = {}
    for run in runs.runs:
        match = FOLD_PATTERN.fullmatch(run.run_id)
        if match:
            out[int(match.group(1))] = list(run.excluded_lineages)
    return out


def load_fold(path: Path, lineages: dict[int, list[str]]) -> FoldCurve:
    """Read and validate one fold's training report."""
    raw = path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    match = FOLD_PATTERN.fullmatch(str(report.get("protocol", "")))
    if not match:
        raise SelectionError(f"{path}: protocol {report.get('protocol')!r} is not a LOLO fold")
    fold = int(match.group(1))
    checks = {
        "status": (report.get("status"), "OK"),
        "mode": (report.get("mode"), "train"),
        "seed": (report.get("seed"), SEED),
        "fixed_steps": (report.get("fixed_steps"), N_MAX),
        "heldout_every": (report.get("heldout_every"), GRID_EVERY),
        "held_out_lineages": (report.get("held_out_lineages"), lineages.get(fold)),
    }
    for key, (got, want) in checks.items():
        if got != want:
            raise SelectionError(f"{path}: {key} is {got!r}, expected {want!r}")
    points = {int(p["step"]): p for p in report.get("curve", []) if "heldout_auc" in p}
    if tuple(sorted(points)) != GRID:
        raise SelectionError(
            f"{path}: held-out steps {sorted(points)} are not the frozen grid {list(GRID)}"
        )
    units = {step: to_units(points[step]["heldout_auc"], f"{path} step {step}") for step in GRID}
    return FoldCurve(fold=fold, path=path.as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                     units=units, points=points)


def load_folds(paths: list[Path], runs_config: Path) -> dict[int, FoldCurve]:
    """All 8 folds, each exactly once."""
    lineages = expected_lineages(runs_config)
    if sorted(lineages) != list(FOLDS):
        raise SelectionError(f"{runs_config}: folds {sorted(lineages)} are not 1..8")
    folds: dict[int, FoldCurve] = {}
    for path in paths:
        curve = load_fold(path, lineages)
        if curve.fold in folds:
            raise SelectionError(f"fold {curve.fold} given twice: {folds[curve.fold].path}, {path}")
        folds[curve.fold] = curve
    missing = sorted(set(FOLDS) - set(folds))
    if missing:
        raise SelectionError(f"folds {missing} have no report; the rule needs all 8")
    return folds


def argmax_step(folds: dict[int, FoldCurve], members: list[int]) -> dict[str, Any]:
    """The grid step maximising the mean held-out AUC over ``members``; earliest on a tie."""
    sums = {step: sum(folds[f].units[step] for f in members) for step in GRID}
    best = max(sums.values())
    tied = [step for step in GRID if sums[step] == best]
    return {
        "step": tied[0],
        "folds_averaged": members,
        "mean_heldout_auc": best / (UNITS * len(members)),
        "tied_steps": tied,
        "mean_curve": {str(step): sums[step] / (UNITS * len(members)) for step in GRID},
    }


def select(folds: dict[int, FoldCurve]) -> dict[str, Any]:
    """N*(-f) for every fold, N*, and the H1-a table at N*(-f)."""
    nested = {f: argmax_step(folds, [g for g in FOLDS if g != f]) for f in FOLDS}
    overall = argmax_step(folds, list(FOLDS))
    table = []
    for f in FOLDS:
        step = nested[f]["step"]
        point = folds[f].points[step]
        auc = folds[f].units[step] / UNITS
        table.append({
            "fold": f, "n_star_minus_f": step, "heldout_auc": auc,
            "heldout_auc_by_member": point.get("heldout_auc_by_member", {}),
            "below_floor": auc < FOLD_FLOOR,
            "secondary_in_protocol": {k: point.get(k) for k in
                                      ("val_auc", "val_logloss", "val_tpr_at_1pct_fpr")},
        })
    g = sum(folds[f].units[nested[f]["step"]] for f in FOLDS) / (UNITS * len(FOLDS))
    return {"nested": {str(f): nested[f] for f in FOLDS}, "n_star": overall,
            "h1a_table": table, "G": g}


def h1a_verdict(g: float, below: list[int], ci_lower: float | None) -> dict[str, Any]:
    """H1-a as frozen: G >= 0.90, CI lower bound >= 0.85, no fold below 0.75."""
    parts = {"G >= 0.90": g >= G_MIN, "no fold below 0.75": not below,
             "CI lower bound >= 0.85": None if ci_lower is None else ci_lower >= CI_LOWER_MIN}
    if parts["G >= 0.90"] is False or parts["no fold below 0.75"] is False \
            or parts["CI lower bound >= 0.85"] is False:
        verdict = "FAIL"
    elif ci_lower is None:
        verdict = "PENDING-CI"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "criteria": parts, "folds_below_floor": below}


def bootstrap_g(reals: dict[int, np.ndarray], fakes: dict[int, np.ndarray],
                resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED,
                level: float = CI_LEVEL) -> dict[str, Any]:
    """Percentile bootstrap CI of G (see the module docstring for what is resampled).

    Args:
        reals: fold -> real scores, aligned so index i is the same real image in every fold.
        fakes: fold -> that fold's held-out reconstruction scores (its slice).
    """
    order = sorted(reals)
    if sorted(fakes) != order:
        raise SelectionError("reals and fakes cover different folds")
    n_real = {len(reals[f]) for f in order}
    if len(n_real) != 1:
        raise SelectionError(f"the real pool differs in size across folds: {sorted(n_real)}")
    n = n_real.pop()
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        real_idx = rng.integers(0, n, n)
        aucs = []
        for f in order:
            fake_idx = rng.integers(0, len(fakes[f]), len(fakes[f]))
            r, k = reals[f][real_idx], fakes[f][fake_idx]
            aucs.append(roc_auc(np.r_[np.zeros(n), np.ones(len(k))], np.r_[r, k]))
        values[b] = float(np.mean(aucs))
    tail = (1.0 - level) / 2.0 * 100.0
    return {"lower": float(np.percentile(values, tail)),
            "upper": float(np.percentile(values, 100.0 - tail)),
            "level": level, "resamples": resamples, "seed": seed,
            "percentiles": [tail, 100.0 - tail]}


def score_file(scores_dir: Path, fold: int, step: int) -> Path:
    """The per-image file ``p3_score_heldout.py`` writes for fold f at ``N*(-f)``."""
    return scores_dir / f"lolo_f{fold}_seed0_step_{step:05d}.csv"


def read_scores(scores_dir: Path, steps: dict[int, int]) -> tuple[
        dict[int, np.ndarray], dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    """Per-image scores of every fold at its frozen ``N*(-f)`` (D2 (ii) format).

    Each fold's file must exist at exactly that step (its sidecar says so too), be a full
    scoring (no ``--limit``), and score the same real images as every other fold.
    """
    reals: dict[int, dict[str, float]] = {}
    fakes: dict[int, dict[tuple[str, str], float]] = {}
    metas: dict[int, dict[str, Any]] = {}
    for fold, step in sorted(steps.items()):
        path = score_file(scores_dir, fold, step)
        if not path.is_file():
            raise SelectionError(f"fold {fold}: no scores at its N*(-f) = {step} ({path})")
        try:
            rows, meta = read_heldout_scores(path)
        except HeldoutScoreError as error:
            raise SelectionError(str(error)) from error
        if meta.get("protocol") != f"lolo_f{fold}" or int(meta.get("step", -1)) != step:
            raise SelectionError(f"{path}: sidecar says {meta.get('protocol')} step "
                                 f"{meta.get('step')}, its N*(-f) is {step}")
        if meta.get("limit"):
            raise SelectionError(f"{path}: a --limit smoke file, not a full scoring")
        reals[fold] = {r["image_id"]: r["score"] for r in rows if r["label"] == 0}
        fakes[fold] = {(r["image_id"], r["member"]): r["score"] for r in rows if r["label"] == 1}
        metas[fold] = meta
    first = sorted(reals[min(steps)])
    for fold in steps:
        if not reals[fold] or not fakes[fold]:
            raise SelectionError(f"fold {fold} has no reals or no reconstructions")
        if sorted(reals[fold]) != first:
            raise SelectionError(f"fold {fold} was scored on different reals")
    return ({f: np.array([reals[f][i] for i in first]) for f in steps},
            {f: np.array([fakes[f][k] for k in sorted(fakes[f])]) for f in steps}, metas)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def cmd_select(args: argparse.Namespace, log: Any) -> int:
    """Apply the frozen rule to the 8 fold reports, once."""
    if args.out.exists():
        log.error("%s already exists: N* is selected once and never re-selected", args.out)
        return 2
    folds = load_folds(args.report, args.runs_config)
    result = select(folds)
    below = [row["fold"] for row in result["h1a_table"] if row["below_floor"]]
    payload = {
        "what": "P3 nested N* selection and H1-a (pre-registration sections 3, 6.1)",
        "rule": {"n_max": N_MAX, "grid": list(GRID), "tie_break": TIE_BREAK,
                 "n_star_minus_f": "argmax of the mean held-out AUC over the other 7 folds",
                 "n_star": "argmax of the mean held-out AUC over all 8 folds",
                 "precision": "report AUCs (5 decimals) compared as exact integer sums"},
        "inputs": [{"fold": f, "path": folds[f].path, "sha256": folds[f].sha256}
                   for f in FOLDS],
        **result,
        "h1a": {"G": result["G"], "thresholds": {"G": G_MIN, "ci_lower": CI_LOWER_MIN,
                                                 "fold_floor": FOLD_FLOOR, "ci_level": CI_LEVEL},
                "ci": None,
                "ci_status": "NOT-RUN: needs per-image held-out scores at N*(-f); the "
                             "training reports hold AUCs only (run the 'ci' subcommand)",
                **h1a_verdict(result["G"], below, None)},
    }
    write_json(args.out, payload)
    for row in result["h1a_table"]:
        log.info("fold %d: N*(-f) = %5d  held-out AUC %.5f%s", row["fold"],
                 row["n_star_minus_f"], row["heldout_auc"],
                 "  BELOW 0.75" if row["below_floor"] else "")
    log.info("N* = %d (mean %.6f over 8 folds; tied steps %s)", result["n_star"]["step"],
             result["n_star"]["mean_heldout_auc"], result["n_star"]["tied_steps"])
    log.info("G = %.6f; H1-a %s; written %s", result["G"], payload["h1a"]["verdict"], args.out)
    return 0


def cmd_ci(args: argparse.Namespace, log: Any) -> int:
    """Bootstrap CI of G from per-image scores at the frozen N*(-f); never re-selects."""
    selection = json.loads(args.nstar.read_text(encoding="utf-8"))
    steps = {int(f): int(v["step"]) for f, v in selection["nested"].items()}
    reported = {row["fold"]: row["heldout_auc"] for row in selection["h1a_table"]}
    reals, fakes, metas = read_scores(args.scores_dir, steps)
    reproduced = {}
    for f in sorted(steps):
        n = len(reals[f])
        auc = roc_auc(np.r_[np.zeros(n), np.ones(len(fakes[f]))], np.r_[reals[f], fakes[f]])
        reproduced[f] = {"reported": reported[f], "rescored": auc, "diff": auc - reported[f]}
        if abs(auc - reported[f]) > REPRODUCE_TOL:
            log.error("fold %d: rescored AUC %.5f does not reproduce the report's %.5f "
                      "(tolerance %g); refusing to compute the CI", f, auc, reported[f],
                      REPRODUCE_TOL)
            return 1
    ci = bootstrap_g(reals, fakes)
    below = selection["h1a"]["folds_below_floor"]
    payload = {
        "what": "H1-a 97.5% bootstrap CI of G at the frozen N*(-f)",
        "selection": {"path": args.nstar.as_posix(),
                      "sha256": hashlib.sha256(args.nstar.read_bytes()).hexdigest()},
        "scores": {str(f): {"file": score_file(args.scores_dir, f, steps[f]).name,
                            "sha256": metas[f]["scores_sha256"],
                            "snapshot_sha256": metas[f].get("snapshot_sha256")}
                   for f in sorted(steps)},
        "resampled": "reals once per resample, shared by every fold; each fold's fakes "
                     "within its own slice; folds and N*(-f) fixed",
        "reproduction": {str(f): v for f, v in reproduced.items()},
        "G": selection["G"], "ci": ci,
        **h1a_verdict(selection["G"], below, ci["lower"]),
    }
    write_json(args.out, payload)
    log.info("G = %.6f, 97.5%% CI [%.5f, %.5f]; H1-a %s", selection["G"], ci["lower"],
             ci["upper"], payload["verdict"])
    return 0


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_select = sub.add_parser("select", help="apply the frozen N* rule to the 8 fold reports")
    p_select.add_argument("--report", action="append", type=Path, required=True)
    p_select.add_argument("--runs-config", type=Path,
                          default=THESIS_DBP / "configs" / "p3_runs.yaml")
    p_select.add_argument("--out", type=Path, default=THESIS_DBP / "reports" / "p3_nstar.json")
    p_ci = sub.add_parser("ci", help="H1-a CI from per-image scores at the frozen N*(-f)")
    p_ci.add_argument("--nstar", type=Path, default=THESIS_DBP / "reports" / "p3_nstar.json")
    p_ci.add_argument("--scores-dir", type=Path,
                      default=THESIS_DBP / "reports" / "p3_heldout_scores")
    p_ci.add_argument("--out", type=Path, default=THESIS_DBP / "reports" / "p3_h1a_ci.json")
    args = parser.parse_args()
    log = setup_logging("p3_select_nstar")
    try:
        return (cmd_select if args.command == "select" else cmd_ci)(args, log)
    except SelectionError as error:
        log.error("%s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
