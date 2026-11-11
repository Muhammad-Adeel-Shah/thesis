"""P3b: report the ViT-L smoke -- peak memory, s/step, projection. Always exits 0.

Reads each smoke's probe record (``<scratch>/dbp_p3b_smoke_gc_<on|off>/probe.json``, written
by ``p3b_smoke_probe.py`` around the unchanged trainer) and the trainer's own smoke report
beside it, and writes ``--out`` with, per smoke: the trainer's status, peak GPU memory
(allocated and reserved, GiB, of the GPU's total), seconds per step after warm-up, and
whether it ran out of memory.

From the recipe smoke (gradient checkpointing on) it projects one full run:
``steps x s/step x overhead``, where the overhead -- validation every 500 steps and the
checkpoints -- is S2-D ViT-B's MEASURED ratio of GPU-hours to steps x s/step
(``projection.vitb_reference``), so the projection's overhead part is ESTIMATED. Two
seeds train in parallel, one per T4.

A report step must never fail the run (CLAUDE.md section 12, rule 5): every part is
guarded, and the script exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]


def smoke_record(scratch: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """One smoke's measurements, or why there are none."""
    root = scratch / f"dbp_p3b_smoke_gc_{spec['gradient_checkpointing']}"
    record: dict[str, Any] = {"gradient_checkpointing": spec["gradient_checkpointing"],
                              "role": spec["role"]}
    probe = root / "probe.json"
    record["probe"] = json.loads(probe.read_text(encoding="utf-8")) if probe.is_file() else None
    reports = sorted((root / "reports").glob("p2_s2_*.json"))
    trainer = json.loads(reports[-1].read_text(encoding="utf-8")) if reports else {}
    record["trainer_status"] = trainer.get("status", "NO REPORT")
    record["steps"] = trainer.get("steps_run")
    return record


def projection(recipe: dict[str, Any], expect: dict[str, Any]) -> dict[str, Any] | None:
    """Projected GPU hours of one full run from the recipe smoke, beside ViT-B's."""
    step = (recipe.get("probe") or {}).get("step_seconds_after_warmup")
    if not step:
        return None
    plan = expect["projection"]
    out: dict[str, Any] = {"steps": plan["steps"], "step_seconds": step}
    reference = THESIS_DBP / plan["vitb_reference"]
    overhead = 1.0
    if reference.is_file():
        vitb = json.loads(reference.read_text(encoding="utf-8"))
        vitb_step, vitb_hours = vitb["measured_step_seconds_train_loop"], vitb["gpu_hours"]
        overhead = vitb_hours * 3600 / (vitb["steps_run"] * vitb_step)
        out["vitb"] = {"step_seconds": vitb_step, "gpu_hours_per_run": vitb_hours,
                       "overhead_factor": round(overhead, 3), "source": plan["vitb_reference"]}
        out["vitl_over_vitb_step_time"] = round(step / vitb_step, 2)
    hours = step * int(plan["steps"]) * overhead / 3600
    out.update({"gpu_hours_per_run": round(hours, 2),
                "two_seeds_wall_hours_parallel": round(hours, 2),
                "two_seeds_gpu_hours": round(2 * hours, 2),
                "basis": "MEASURED s/step x steps x ViT-B's MEASURED overhead factor "
                         "(the overhead part is ESTIMATED for ViT-L)"})
    return out


def verdict(off: dict[str, Any] | None) -> str | None:
    """Whether gradient checkpointing is needed, from the measurement smoke."""
    if off is None or off.get("probe") is None:
        return None
    probe = off["probe"]
    if probe["out_of_memory"]:
        return "yes: without it the step runs out of memory"
    if probe["exit_code"] == 0 and off["trainer_status"] == "SMOKE-OK":
        return "no: it fits without it (the recipe keeps it on)"
    return f"unclear: the measurement smoke exited {probe['exit_code']}"


def main() -> int:
    """Write and print the smoke report. Always 0."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    try:
        args = parser.parse_args()
    except SystemExit:
        print("(report: bad arguments; nothing reported)", flush=True)
        return 0
    try:
        expect = json.loads(args.expect.read_text(encoding="utf-8"))
        smokes = [smoke_record(args.scratch, s) for s in expect["smoke_runs"]]
        recipe = next(s for s in smokes if s["gradient_checkpointing"] == "on")
        off = next((s for s in smokes if s["gradient_checkpointing"] == "off"), None)
        record = {"what": "P3b ViT-L smoke (no training run)", "backbone": expect["backbone"],
                  "smoke_steps": expect["smoke_steps"], "smokes": smokes,
                  "projection": projection(recipe, expect),
                  "gradient_checkpointing_needed": verdict(off)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        for smoke in smokes:
            p = smoke.get("probe") or {}
            print(f"gradient checkpointing {smoke['gradient_checkpointing']:3s}: trainer "
                  f"{smoke['trainer_status']}, steps {smoke.get('steps')}, out of memory "
                  f"{p.get('out_of_memory')}, peak allocated "
                  f"{p.get('peak_gpu_memory_allocated_gib')} GiB, reserved "
                  f"{p.get('peak_gpu_memory_reserved_gib')} GiB of "
                  f"{p.get('gpu_total_memory_gib')} GiB, {p.get('step_seconds_after_warmup')} "
                  f"s/step after warm-up, GPU {p.get('gpu')}")
        print("PROJECTION:", json.dumps(record["projection"]))
        print("GRADIENT CHECKPOINTING NEEDED:", record["gradient_checkpointing_needed"])
    except Exception as error:  # a report step must never fail the run (rule 5)
        print(f"(report unavailable: {type(error).__name__}: {error})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
