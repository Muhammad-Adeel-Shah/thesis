"""P3b failure path (``docs/P3b_VITL_PREREG.md`` section 5): may a ViT-L seed be scored?

A seed's training run **fails**, and its Panel B is never scored, if any of these holds:

- its training report is missing, or its status is not ``OK`` / its mode not ``train``;
- it is not the run the pre-registration names: the session's backbone (``dinov3-vitl16``)
  at its pinned revision, ``steps_run`` = the recipe's stop step (the ``--stop-at`` of the
  session's ``train_args``: 5,000);
- the loss is NaN or the run diverged: any recorded loss or validation AUC on its curve is
  missing or not finite;
- its validation AUC at the stop step is **more than 5 points below** S2-D ViT-B's for the
  same seed, both read from training reports (ViT-B's from the committed ones the session's
  expect file names, never typed).

The recipe values come from the session's expect file (``env/p3b_vitl_train_expect.json``),
the file the runbook trained with, so the check judges the run that was made. Nothing here
re-tunes or retrains. The record names every reason for every seed and the seeds cleared
for scoring. Exit status: 0 when every seed passes, 1 otherwise (the runbook calls it from
its report cell, which ignores the status; the scoring pre-flight requires 0).

Usage:
    python scripts/p3b_train_check.py --expect env/p3b_vitl_train_expect.json
        --reports-dir reports --out reports/p3b_train_check.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.backbones.loader import BACKBONES  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import write_record  # noqa: E402

PROTOCOL = "s2d_p35"
#: Section 5: more than 5 AUC points below ViT-B (same seed) fails.
MARGIN = 0.05


def finite(value: Any) -> bool:
    """A real number that is neither NaN nor infinite (None and booleans are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and \
        math.isfinite(value)


def val_auc_at(report: dict[str, Any], step: int) -> float | None:
    """The validation AUC the report's curve records at ``step``, if finite."""
    for point in report.get("curve") or []:
        if point.get("step") == step:
            value = point.get("val_auc")
            return float(value) if finite(value) else None
    return None


def stop_step(train_args: list[str]) -> int:
    """The ``--stop-at`` the session trained with."""
    values = [a.split("=", 1)[1] for a in train_args if a.startswith("--stop-at=")]
    if len(values) != 1:
        raise ValueError(f"train_args name --stop-at {len(values)} times: {train_args}")
    return int(values[0])


def check_seed(report: dict[str, Any] | None, vitb: dict[str, Any], backbone: str,
               revision: str, stop_at: int) -> dict[str, Any]:
    """One seed's verdict, with every reason it fails."""
    reasons: list[str] = []
    vitb_auc = val_auc_at(vitb, stop_at)
    if vitb_auc is None:
        reasons.append(f"the ViT-B report has no finite validation AUC at step {stop_at}")
    if report is None:
        return {"verdict": "FAIL", "reasons": ["no training report", *reasons],
                "vitb_val_auc": vitb_auc}
    if report.get("status") != "OK" or report.get("mode") != "train":
        reasons.append(f"status {report.get('status')!r}, mode {report.get('mode')!r}")
    if report.get("backbone") != backbone or report.get("revision") != revision:
        reasons.append(f"backbone {report.get('backbone')!r} @ {report.get('revision')!r}, "
                       f"expected {backbone} @ {revision}")
    if report.get("steps_run") != stop_at:
        reasons.append(f"steps_run {report.get('steps_run')!r}, expected {stop_at}")
    curve = report.get("curve") or []
    if not curve:
        reasons.append("no validation curve")
    bad = [p.get("step") for p in curve
           if not (finite(p.get("loss")) and finite(p.get("val_auc")))]
    if bad:
        reasons.append(f"NaN or divergence: non-finite loss or validation AUC at step(s) {bad}")
    auc = val_auc_at(report, stop_at)
    floor = None if vitb_auc is None else round(vitb_auc - MARGIN, 10)
    if auc is None:
        reasons.append(f"no finite validation AUC at step {stop_at}")
    elif floor is not None and auc < floor:
        reasons.append(f"validation AUC {auc} at step {stop_at} is more than 5 points below "
                       f"ViT-B's {vitb_auc} (floor {floor})")
    return {"verdict": "FAIL" if reasons else "PASS", "reasons": reasons,
            "val_auc": auc, "vitb_val_auc": vitb_auc, "floor": floor,
            "steps_run": report.get("steps_run"), "status": report.get("status")}


def read(path: Path) -> dict[str, Any] | None:
    """A JSON report, or None when it is not there."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    """Check every seed; write the record; 0 only if every seed passes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True,
                        help="The session's expect file: backbone, train_args, vitb_reports.")
    parser.add_argument("--reports-dir", type=Path, required=True,
                        help="Where the ViT-L training reports p2_s2_<run>.json are.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log = setup_logging("p3b_train_check")
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    backbone, stop_at = expect["backbone"], stop_step(expect["train_args"])
    revision = BACKBONES[backbone]["revision"]
    seeds: dict[str, Any] = {}
    for seed, relative in sorted(expect["vitb_reports"].items()):
        vitb = read(THESIS_DBP / relative)
        if vitb is None:
            seeds[seed] = {"verdict": "FAIL", "reasons": [f"no ViT-B report at {relative}"]}
            continue
        mine = args.reports_dir / f"p2_s2_{PROTOCOL}_seed{seed}.json"
        seeds[seed] = {**check_seed(read(mine), vitb, backbone, revision, stop_at),
                       "report": mine.name, "vitb_report": relative}
    cleared = sorted(s for s, v in seeds.items() if v["verdict"] == "PASS")
    record = {"what": "P3b failure path (docs/P3b_VITL_PREREG.md section 5)",
              "backbone": backbone, "revision": revision, "stop_at": stop_at,
              "margin": MARGIN, "seeds": seeds, "cleared_for_scoring": cleared,
              "all_pass": len(cleared) == len(seeds)}
    write_record(args.out, record)
    for seed, verdict in seeds.items():
        log.info("seed %s: %s %s", seed, verdict["verdict"], "; ".join(verdict["reasons"]))
    log.info("FAILURE PATH: %s -- cleared for Panel B scoring: %s",
             "every seed passes" if record["all_pass"] else "STOP AND REPORT (section 5)",
             cleared or "none")
    return 0 if record["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
