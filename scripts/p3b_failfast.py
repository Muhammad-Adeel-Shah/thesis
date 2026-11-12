"""P3b fail fast: project each scoring job from its smoke; refuse if it cannot finish.

The rule of 2026-11-09 for every long job (``docs/P3b_VITL_PREREG.md`` section 2). Run after
a stage's smoke (``p3_final_score.py --limit N``, N > the batch) and before its scoring.
Each smoke summary records the model's ``load_seconds`` and, per list, the images and
seconds **after the first batch** (the warm-up is timed apart and left out). From those:

* a model's warm rate = its smoke's warm seconds / warm images, over every list it scored;
* a model's projection = ``load_seconds + rate x the rows it still has to score`` this
  stage (lists already finished and carried are not counted);
* a job runs its models one after another on one GPU, so its projection is their sum.

The time available is ``min(deadline_hours, session_limit_hours - elapsed - margin_hours)``.
**If any job's projection exceeds it, the stage is refused (exit 1)** and nothing long
starts. Writes ``reports/p3b_projection_<stage>.json`` either way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.utils.logging import setup_logging  # noqa: E402
from p3_final_preflight import CLEAN_LISTS, FINAL, SUMMARIES  # noqa: E402
from p3_s2d_inputs import read_rows, write_record  # noqa: E402


def rows_to_score(plan: dict[str, Any], model: str, here: Path) -> int:
    """Rows this stage still has to score for ``model`` (finished lists left out)."""
    sizes = {name: len(read_rows(Path(plan["lists"][name]["path"]))[1]) for name in CLEAN_LISTS}
    test = sum(1 for r in read_rows(Path(plan["lists"]["panel_b"]["path"]))[1]
               if r["role"] == "test")
    if plan["stage"] == "clean":
        lists = {name: sizes[name] for name in CLEAN_LISTS}
    else:
        lists = {c: test for c in plan["robust"]["conditions"]}
    done = here / FINAL / model / plan["stage"]
    return sum(n for name, n in lists.items() if not (done / f"{name}.done.json").is_file())


def project_model(model: str, plan: dict[str, Any], smoke_root: Path, here: Path
                  ) -> dict[str, Any]:
    """One model's projected seconds from its smoke summary."""
    path = smoke_root / SUMMARIES / f"{model}_{plan['stage']}.json"
    if not path.is_file():
        raise ValueError(f"{model}: no smoke summary at {path}")
    smoke = json.loads(path.read_text(encoding="utf-8"))
    timings = [v.get("timing") or {} for v in smoke.get("lists", {}).values()]
    images = sum(int(t.get("warm_images") or 0) for t in timings)
    seconds = sum(float(t.get("warm_seconds") or 0.0) for t in timings)
    if images <= 0 or seconds <= 0:
        raise ValueError(f"{model}: the smoke timed no image after the warm-up batch "
                         f"(run it with --limit larger than the batch)")
    rate = seconds / images
    rows = rows_to_score(plan, model, here)
    load = float(smoke.get("load_seconds") or 0.0)
    return {"model": model, "rate_s_per_image": rate, "warm_images_timed": images,
            "rows_to_score": rows, "load_seconds": load, "seconds": load + rate * rows}


def project(plan: dict[str, Any], smoke_root: Path, expect: dict[str, Any],
            elapsed_hours: float, here: Path) -> dict[str, Any]:
    """Every job's projection against the time available."""
    available = min(float(expect["deadline_hours"]),
                    float(expect["session_limit_hours"]) - elapsed_hours -
                    float(expect["margin_hours"]))
    jobs = []
    for job in plan["jobs"]:
        device, _, members = job.partition(":")
        units = [project_model(m, plan, smoke_root, here) for m in members.split(",") if m]
        hours = sum(u["seconds"] for u in units) / 3600
        jobs.append({"device": device, "models": units, "hours": round(hours, 3),
                     "fits": hours <= available})
    return {"what": "P3b fail-fast projection (user rule 2026-11-09)", "stage": plan["stage"],
            "elapsed_hours": round(elapsed_hours, 3), "available_hours": round(available, 3),
            "deadline_hours": round(max(available, 0.0), 3), "jobs": jobs,
            "ok": bool(jobs) and all(j["fits"] for j in jobs) and available > 0}


def main() -> int:
    """Project; 0 when every job fits, 1 when any does not (or a smoke is missing)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, default=THESIS_DBP / "reports" /
                        "p3b_score_plan.json")
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--elapsed-hours", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    log = setup_logging("p3b_failfast")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    out = args.out or THESIS_DBP / "reports" / f"p3b_projection_{plan.get('stage')}.json"
    try:
        record = project(plan, args.smoke_root, expect, args.elapsed_hours, THESIS_DBP)
    except (ValueError, KeyError, OSError) as error:
        write_record(out, {"what": "P3b fail-fast projection", "ok": False,
                           "problem": str(error)})
        log.error("FAIL-FAST: cannot project: %s", error)
        return 1
    write_record(out, record)
    for job in record["jobs"]:
        for unit in job["models"]:
            log.info("  GPU %s %-20s %.4f s/image x %d rows + %.0f s load -> %.2f h",
                     job["device"], unit["model"], unit["rate_s_per_image"],
                     unit["rows_to_score"], unit["load_seconds"], unit["seconds"] / 3600)
        log.info("GPU %s: %.2f h projected, %.2f h available -- %s", job["device"],
                 job["hours"], record["available_hours"],
                 "fits" if job["fits"] else "DOES NOT FIT")
    log.info("FAIL-FAST (%s): %s", record["stage"], "START" if record["ok"] else
             "REFUSED (the projection exceeds the remaining deadline)")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
