"""P3 section 4b: choose the degradation chain probability by the rule committed 2026-10-17.

The rule (log P3-C, 2026-10-17, committed before either candidate trained; unchanged):

* **Candidates:** ``s2b_deg_p70`` (p = 0.7, the plan default) and ``s2b_deg_p35``
  (p = 0.35), each S2-B+deg seed 0 on the P3 recipe.
* **Objective -- development data only, the P2 panel:** the macro achieved FPR over the
  **E4 and E5** dev real pools (each counted once), at the k = 0 threshold fixed at 5%
  FPR on the **491 COCO validation reals only** (P2's reals-only rule, I-062,
  :func:`dbp.eval.thresholds.calibrate_on_reals`).
* **Guard:** a candidate's dev macro AUC over the four unseen slices (E3 GPT-Image2, E3
  Nano-Banana2, E4, E5 -- the first four of ``p2_analyze.P_A_UNSEEN_SLICES``; a slice is
  its fakes against the view's whole real pool) may not fall more than 1 point below
  ``s2b`` seed 0's.
* **Choice:** exactly one candidate passes the guard -> it. Both or neither -> the lower
  macro FPR, except that FPRs within 1 point (|delta| <= 0.01) go to the milder p (0.35).

Every model is scored by ``p2_s2_infer.py`` (centre crop, the run's final weights) on
``panel`` (``p2_panel_images.csv``) and ``val`` (``p2_s1_images.csv``, view ``val``).
**Development data only -- Panel B is never read.**
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.eval.metrics import roc_auc  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

BASELINE = "s2b_seed0"
CANDIDATES = {"s2b_deg_p70_seed0": 0.7, "s2b_deg_p35_seed0": 0.35}
MILDER = "s2b_deg_p35_seed0"
TARGET_FPR = 0.05
GUARD_POINTS = 0.01
TIE_POINTS = 0.01
FPR_POOLS = ("e4_wildrf_matched", "e5_chameleon_pipeline")


class TuneError(ValueError):
    """Inputs the committed rule cannot be applied to."""


def unseen_slices() -> tuple[str, ...]:
    """The four unseen slices, as P2 defined them (not redefined here)."""
    from p2_analyze import P_A_UNSEEN_SLICES

    return tuple(P_A_UNSEEN_SLICES[:4])


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_scores(path: Path) -> dict[str, float]:
    """``image_id -> score`` from a ``p2_s2_infer.py`` file; every score finite."""
    scores = {r["image_id"]: float(r["score"]) for r in read_csv(path)}
    if not scores or not all(np.isfinite(list(scores.values()))):
        raise TuneError(f"{path}: empty or non-finite scores")
    return scores


def panel_rows(panel_list: Path, e3_manifest: Path) -> list[dict[str, str]]:
    """Dev panel rows with the fields ``p2_analyze.slice_key`` reads."""
    generator = {r["image_id"]: r["source_label"] for r in read_csv(e3_manifest)}
    rows = []
    for row in read_csv(panel_list):
        rows.append({**row, "subset": "",
                     "generator": generator.get(row["image_id"], "")
                     if row["view"] == "e3_matched" and row["label"] == "fake" else ""})
    return rows


def evaluate(rows: list[dict[str, str]], panel: dict[str, float],
             val_reals: list[float]) -> dict[str, Any]:
    """One model: threshold on the val reals, FPR per dev pool, AUC per unseen slice."""
    from p2_analyze import slice_key

    missing = [r["image_id"] for r in rows if r["image_id"] not in panel]
    if missing:
        raise TuneError(f"{len(missing)} dev panel row(s) unscored, e.g. {missing[:3]}")
    calibration = calibrate_on_reals(val_reals, TARGET_FPR, "p2 val reals (COCO, 491)")
    threshold = calibration.threshold
    fpr = {}
    for view in FPR_POOLS:
        reals = np.array([panel[r["image_id"]] for r in rows
                          if r["view"] == view and r["label"] == "real"])
        fpr[view] = {"reals": int(reals.size), "achieved_fpr": float(np.mean(reals >= threshold))}
    auc = {}
    for name in unseen_slices():
        view = name.split("/", 1)[0]
        fakes = [panel[r["image_id"]] for r in rows if r["label"] == "fake"
                 and r["view"] == view and slice_key(r) == name]
        reals = [panel[r["image_id"]] for r in rows if r["label"] == "real" and r["view"] == view]
        if not fakes or not reals:
            raise TuneError(f"slice {name}: {len(fakes)} fakes, {len(reals)} reals")
        auc[name] = {"fakes": len(fakes), "reals": len(reals),
                     "auc": float(roc_auc([0] * len(reals) + [1] * len(fakes), reals + fakes))}
    return {"threshold": threshold, "val_reals": len(val_reals),
            "threshold_achieved_fpr_on_val": calibration.achieved_fpr,
            "fpr": fpr, "macro_fpr": float(np.mean([v["achieved_fpr"] for v in fpr.values()])),
            "auc": auc, "macro_auc": float(np.mean([v["auc"] for v in auc.values()]))}


def choose(baseline_auc: float, candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The committed rule, applied to the two candidates' dev numbers."""
    passes = {rid: m["macro_auc"] >= baseline_auc - GUARD_POINTS for rid, m in candidates.items()}
    passing = [rid for rid, ok in passes.items() if ok]
    if len(passing) == 1:
        chosen, why = passing[0], "the only candidate that passes the guard"
    else:
        (a, ma), (b, mb) = sorted(candidates.items())
        delta = abs(ma["macro_fpr"] - mb["macro_fpr"])
        branch = "both" if len(passing) == 2 else "neither"
        if delta <= TIE_POINTS:
            chosen, why = MILDER, (f"{branch} pass the guard; macro FPRs within 1 point "
                                   f"(|delta| = {delta:.4f}) -> the milder p")
        else:
            chosen = min(candidates, key=lambda rid: candidates[rid]["macro_fpr"])
            why = f"{branch} pass the guard; the lower macro FPR (|delta| = {delta:.4f})"
    return {"guard_floor": baseline_auc - GUARD_POINTS, "passes_guard": passes,
            "chosen": chosen, "chosen_p": CANDIDATES[chosen], "reason": why}


def main() -> int:
    """Apply the rule once; write ``reports/p3_degradation_choice.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores-root", type=Path, required=True,
                        help="outputs/p3/s2: <run id>/scores/{panel,val}.csv")
    parser.add_argument("--panel-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--e3-manifest", type=Path, default=THESIS_DBP / "reports"
                        / "p2_panel_sets" / "p2_panel_e3_matched.csv")
    parser.add_argument("--out", type=Path,
                        default=THESIS_DBP / "reports" / "p3_degradation_choice.json")
    args = parser.parse_args()
    log = setup_logging("p3_tune_degradation")
    rows = panel_rows(args.panel_list, args.e3_manifest)
    val_ids = {r["image_id"] for r in read_csv(args.val_list)
               if r.get("view") == "val" and r["label"] == "real"}
    results: dict[str, dict[str, Any]] = {}
    try:
        for rid in (BASELINE, *CANDIDATES):
            panel = load_scores(args.scores_root / rid / "scores" / "panel.csv")
            val = load_scores(args.scores_root / rid / "scores" / "val.csv")
            if not val_ids <= set(val):
                raise TuneError(f"{rid}: {len(val_ids - set(val))} val real(s) unscored")
            results[rid] = evaluate(rows, panel, [val[i] for i in sorted(val_ids)])
    except TuneError as error:
        log.error("%s", error)
        return 1
    decision = choose(results[BASELINE]["macro_auc"],
                      {rid: results[rid] for rid in CANDIDATES})
    payload = {"what": "P3 section 4b degradation tuning, the rule committed 2026-10-17",
               "development_data_only": True, "unseen_slices": list(unseen_slices()),
               "fpr_pools": list(FPR_POOLS), "target_fpr": TARGET_FPR,
               "threshold_rule": "5% FPR on the 491 COCO validation reals only (I-062)",
               "models": results, **decision}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, args.out)
    for rid, m in results.items():
        log.info("%-20s macro AUC %.4f  macro FPR %.4f (E4 %.4f, E5 %.4f)", rid, m["macro_auc"],
                 m["macro_fpr"], m["fpr"][FPR_POOLS[0]]["achieved_fpr"],
                 m["fpr"][FPR_POOLS[1]]["achieved_fpr"])
    log.info("guard floor %.4f; passes %s; CHOSEN %s (p = %s): %s", decision["guard_floor"],
             decision["passes_guard"], decision["chosen"], decision["chosen_p"],
             decision["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
