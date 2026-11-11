"""P4 fail fast: project every job's time from the smoke, refuse if it cannot finish.

User decision 2026-11-09, for every row from now on. Run after the smoke and before the
long job. The smoke (``p4_score.py --limit``) scores one untimed warm-up image per model,
then times its images; from that:

* a whole row is projected as ``load_seconds + s/image x its images`` (the DDA, CO-SPY
  and FIRE rows' time does not depend on image size);
* a shard (AEROBLADE, whose time is linear in capped pixels) as ``load_seconds + s/pixel
  x its pixels still to score + the resume re-score of saved rows``.

A job runs its units one after another on one GPU, so its projection is their sum. The
time available is ``min(deadline_hours, session_limit_hours - elapsed - margin_hours)``.
**If any job's projection exceeds it, the run is refused (exit 1)** and nothing long
starts. Writes ``reports/p4_projection.json`` either way; ``deadline_hours`` there is what
the scoring call should use.
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
from p3_s2d_inputs import read_rows, write_record  # noqa: E402
from p4_preflight import SUMMARIES  # noqa: E402

PROJECTION = Path("reports") / "p4_projection.json"


def list_sizes(plan: dict[str, Any]) -> dict[str, int]:
    """Rows in each scoring list (``panel_b_test`` = Panel B's test rows)."""
    lists = {name: read_rows(Path(path))[1] for name, path in plan["list_files"].items()}
    sizes = {name: len(rows) for name, rows in lists.items()}
    sizes["panel_b_test"] = sum(1 for r in lists["panel_b"] if r["role"] == "test")
    return sizes


def project_unit(unit: str, plan: dict[str, Any], smoke: dict[str, Any],
                 sizes: dict[str, int]) -> dict[str, Any]:
    """One unit's projected seconds from its smoke summary."""
    timing = smoke.get("timing") or {}
    if not timing.get("images") or not timing.get("seconds"):
        raise ValueError(f"{unit}: the smoke recorded no timed image")
    info = plan["units"][unit]
    load = float(smoke.get("load_seconds") or 0.0)
    if info["shard"] is None:
        images = sum(sizes[name] for name in plan["rows"][info["row"]]["lists"])
        rate = timing["seconds"] / timing["images"]
        seconds = load + rate * images
        return {"unit": unit, "basis": "s/image", "rate": rate, "images": images,
                "load_seconds": load, "seconds": seconds}
    spec = plan["shards"][info["row"]]
    shard = spec["units"][unit]
    rate = timing["seconds"] / timing["pixels"]
    remaining = int(shard["pixels"]) - int(shard["saved_pixels"])
    mean_pixels = int(shard["pixels"]) / max(1, int(shard["items"]))
    rescore = min(int(spec["resume_rescore"]), int(shard["saved_rows"])) * mean_pixels * rate
    seconds = load + rate * remaining + rescore
    return {"unit": unit, "basis": "s/pixel", "rate": rate, "pixels_left": remaining,
            "images_left": int(shard["items"]) - int(shard["saved_rows"]),
            "resume_seconds": rescore, "load_seconds": load, "seconds": seconds}


def project(plan: dict[str, Any], smoke_root: Path, expect: dict[str, Any],
            elapsed_hours: float) -> dict[str, Any]:
    """Every job's projection against the time available."""
    sizes = list_sizes(plan)
    available = min(float(expect["deadline_hours"]),
                    float(expect["session_limit_hours"]) - elapsed_hours -
                    float(expect["margin_hours"]))
    jobs = []
    for job in plan["jobs"]:
        device, _, members = job.partition(":")
        units = []
        for unit in [m for m in members.split(",") if m]:
            path = smoke_root / SUMMARIES / f"{unit}.json"
            smoke = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            units.append(project_unit(unit, plan, smoke, sizes))
        hours = sum(u["seconds"] for u in units) / 3600
        jobs.append({"device": device, "units": units, "hours": round(hours, 3),
                     "fits": hours <= available})
    return {"what": "P4 fail-fast projection (user decision 2026-11-09)",
            "elapsed_hours": round(elapsed_hours, 3), "available_hours": round(available, 3),
            "deadline_hours": round(max(available, 0.0), 3), "jobs": jobs,
            "ok": bool(jobs) and all(j["fits"] for j in jobs) and available > 0}


def main() -> int:
    """Project; 0 when every job fits, 1 when any does not (or a smoke is missing)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, default=THESIS_DBP / "reports" / "p4_plan.json")
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--elapsed-hours", type=float, required=True)
    parser.add_argument("--out", type=Path, default=THESIS_DBP / PROJECTION)
    args = parser.parse_args()
    log = setup_logging("p4_failfast")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    try:
        record = project(plan, args.smoke_root, expect, args.elapsed_hours)
    except (ValueError, KeyError, OSError) as error:
        record = {"what": "P4 fail-fast projection", "ok": False, "problem": str(error)}
        write_record(args.out, record)
        log.error("FAIL-FAST: cannot project: %s", error)
        return 1
    write_record(args.out, record)
    for job in record["jobs"]:
        for unit in job["units"]:
            log.info("  GPU %s %-16s %-8s %.4g -> %.2f h", job["device"], unit["unit"],
                     unit["basis"], unit["rate"], unit["seconds"] / 3600)
        log.info("GPU %s: %.2f h projected, %.2f h available -- %s", job["device"],
                 job["hours"], record["available_hours"], "fits" if job["fits"] else "DOES NOT FIT")
    log.info("FAIL-FAST: %s", "START" if record["ok"] else "REFUSED (projection exceeds the "
             "remaining deadline)")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
