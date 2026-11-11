"""P3 LOLO: which fold(s) this session trains -- the lowest-numbered unfinished ones.

One fold per session (user, 2026-10-14), so no runbook cell is ever edited between
sessions: this script looks at the training reports of every previous session it is
given (``--resume-from``: thesis_dbp roots -- a Kaggle notebook version's output, or a
persistent directory on another platform) and at the local ``reports/``, and picks the
first ``--count`` folds whose report does not say ``OK``. A fold that a previous session
started and did not finish is therefore picked again first, and resumes from its
checkpoint.

**Carry forward.** Every complete fold found in a previous session's output is copied
(its run directory -- checkpoint, final model, snapshots -- and its report) into this
session's tree when absent, so each session's output holds every fold finished so far
and only the latest output ever needs attaching.

``--fold`` overrides the choice (e.g. ``--fold lolo_f5``). Writes
``reports/p3_lolo_plan.json``: ``{"selected": ["lolo_f3:0"], "complete": [...],
"remaining": [...], "all_complete": false}``. Exit 0 when something is selected, 3 when
every fold is complete (then the next step is the N* selection, not more training).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import run_id, s2_run_paths  # noqa: E402

FOLD_PATTERN = re.compile(r"lolo_f(\d+)")
EXIT_ALL_COMPLETE = 3


def fold_ids(runs_config: Path) -> list[str]:
    """The LOLO run ids of the config, in fold order."""
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(runs_config, bank)
    folds = [r.run_id for r in runs.runs if FOLD_PATTERN.fullmatch(r.run_id)]
    return sorted(folds, key=lambda rid: int(FOLD_PATTERN.fullmatch(rid).group(1)))


def is_complete(rid: str, roots: list[Path]) -> bool:
    """Whether any root holds an OK training report for this run."""
    for root in roots:
        report = s2_run_paths(root / "outputs" / "p3" / "s2", root / "reports", rid).report
        if report.is_file():
            payload = json.loads(report.read_text(encoding="utf-8"))
            if payload.get("status") == "OK" and payload.get("mode") == "train":
                return True
    return False


def carry_forward(rid: str, sources: list[Path], here: Path, log) -> str:  # noqa: ANN001
    """Copy a complete run from the first source holding it, unless already here."""
    local = s2_run_paths(here / "outputs" / "p3" / "s2", here / "reports", rid)
    if local.report.is_file() and local.run_dir.is_dir():
        return "already here"
    for source in sources:
        there = s2_run_paths(source / "outputs" / "p3" / "s2", source / "reports", rid)
        if there.report.is_file() and there.run_dir.is_dir() and is_complete(rid, [source]):
            if not local.run_dir.exists():
                shutil.copytree(there.run_dir, local.run_dir)
            local.report.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(there.report, local.report)
            log.info("carried %s forward from %s", rid, source)
            return f"carried from {source}"
    return "not found"


def main() -> int:
    """Pick the session's fold(s). Exit 0 if any, 3 if every fold is complete."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-config", type=Path,
                        default=THESIS_DBP / "configs" / "p3_runs.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=1, help="Folds this session trains.")
    parser.add_argument("--fold", default=None, help="Override: train this fold.")
    parser.add_argument("--resume-from", action="append", type=Path, default=[])
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p3_lolo_next")
    folds = fold_ids(args.runs_config)
    roots = [args.reports.parent, *args.resume_from]
    complete = [f for f in folds if is_complete(run_id(f, args.seed), roots)]
    for fold in complete:
        carry_forward(run_id(fold, args.seed), args.resume_from, args.reports.parent, log)
    remaining = [f for f in folds if f not in complete]
    if args.fold:
        if args.fold not in folds:
            log.error("--fold %s is not a LOLO fold of %s", args.fold, args.runs_config.name)
            return 1
        selected = [args.fold]
    else:
        selected = remaining[:args.count]
    plan = {"folds": folds, "complete": complete, "remaining": remaining,
            "selected": [f"{f}:{args.seed}" for f in selected],
            "all_complete": not remaining, "override": args.fold}
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "p3_lolo_plan.json"
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    log.info("LOLO: complete %s; this session trains %s", complete or "none",
             plan["selected"] or "nothing")
    if not selected:
        log.info("ALL %d FOLDS COMPLETE: the next step is the N* selection", len(folds))
        return EXIT_ALL_COMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
