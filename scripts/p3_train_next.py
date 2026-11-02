"""P3 non-LOLO training: which runs this session trains, and whether the LOLO rescoring is due.

The queue order is committed in the expect file (``queue``: ``run:seed`` in the session
priority of 2026-10-17). This script picks the first ``--count`` runs whose training
report in this session or any attached previous one (``--resume-from``) does not say
``OK``, so a run cut short by a session limit is picked again first and resumes from its
500-step checkpoint -- the same rule as ``p3_lolo_next.py``, whose helpers it uses.

**Carry forward.** Every complete run found in a previous session's output is copied (run
directory, training report, and its per-image held-out score files) into this session's
tree, so the latest output always holds every finished model -- Panel B is scored from
them once, at the end. The LOLO rescoring files (``reports/p3_heldout_scores/lolo_*``)
are carried the same way; ``lolo_rescore_due`` is true until a session has written them.

Writes ``reports/p3_train_plan.json``. Exit 0 when something is selected or the rescoring
is due, 3 when every queued run is complete and the rescoring is done.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import run_id  # noqa: E402
from p3_lolo_next import carry_forward, is_complete  # noqa: E402

EXIT_ALL_COMPLETE = 3
SCORES = Path("reports") / "p3_heldout_scores"
RESCORE_SUMMARY = "p3_lolo_rescore_summary.json"


def parse_run(spec: str) -> tuple[str, int]:
    """``protocol:seed``."""
    protocol, _, seed = spec.partition(":")
    if not protocol or not seed.isdigit():
        raise ValueError(f"queue entry {spec!r} is not PROTOCOL:SEED")
    return protocol, int(seed)


def carry_scores(prefix: str, sources: list[Path], here: Path) -> int:
    """Copy ``reports/p3_heldout_scores/<prefix>*`` files absent here; returns the count."""
    copied = 0
    for source in sources:
        for path in sorted((source / SCORES).glob(f"{prefix}*")):
            target = here / SCORES / path.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
    return copied


def main() -> int:
    """Pick the session's runs. Exit 0 if there is work, 3 if everything is done."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--resume-from", action="append", type=Path, default=[])
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p3_train_next")
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    here = args.reports.parent
    queue = [parse_run(spec) for spec in expect["queue"]]
    roots = [here, *args.resume_from]
    complete, remaining = [], []
    for protocol, seed in queue:
        rid = run_id(protocol, seed)
        if is_complete(rid, roots):
            carry_forward(rid, args.resume_from, here, log)
            carry_scores(f"{rid}_step_", args.resume_from, here)
            complete.append(f"{protocol}:{seed}")
        else:
            remaining.append(f"{protocol}:{seed}")
    carried = carry_scores("lolo_", args.resume_from, here)
    rescore_done = (here / SCORES / RESCORE_SUMMARY).is_file() or any(
        (root / SCORES / RESCORE_SUMMARY).is_file() for root in args.resume_from)
    if rescore_done and not (here / SCORES / RESCORE_SUMMARY).is_file():
        for root in args.resume_from:
            if (root / SCORES / RESCORE_SUMMARY).is_file():
                shutil.copy2(root / SCORES / RESCORE_SUMMARY, here / SCORES / RESCORE_SUMMARY)
                break
    plan = {"queue": [f"{p}:{s}" for p, s in queue], "complete": complete,
            "remaining": remaining, "selected": remaining[: args.count],
            "all_complete": not remaining, "lolo_rescore_due": not rescore_done,
            "lolo_score_files_carried": carried}
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "p3_train_plan.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(out)
    log.info("P3 runs: complete %s; this session trains %s; LOLO rescoring %s",
             complete or "none", plan["selected"] or "nothing",
             "DUE" if plan["lolo_rescore_due"] else "done")
    if not plan["selected"] and not plan["lolo_rescore_due"]:
        log.info("every queued run is complete and the LOLO rescoring is done")
        return EXIT_ALL_COMPLETE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
