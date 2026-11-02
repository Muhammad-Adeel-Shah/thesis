"""P3 final scoring, step 1: the pre-flight and the session plan (CPU, before any scoring).

Pre-registration sections 2, 5 and 8: **Panel B is scored once per frozen model**, and no
section-5 value may be chosen while a Panel B score exists. This script refuses to plan
any scoring unless every condition below holds, and writes ``reports/p3_final_plan.json``
for the scoring workers (``p3_final_score.py``).

1. **Section 5 is committed** (the code is git's bytes, so present means committed):
   the degradation result (``chosen_p``, and the ``s2b_deg`` and ``s2d`` models are the
   chosen p's); ``N*`` and ``N_max`` (``reports/p3_nstar.json``: every non-LOLO model at
   ``n_star``, every LOLO fold at its ``N*(-f)``, ``n_max`` 6,000); the new real source
   (the Open Images manifest, 3,600 train + 400 held out); the k-shot methods
   (``dbp.eval.kshot`` reproduces the frozen j values). Each file's sha256 must equal the
   expect file's.
2. **Panel B is the frozen one:** the committed manifests have their committed sha256 and
   reproduce the four frozen digests (all, test, calibration, luma).
3. **Every model is present and is the frozen one** (24, the list in the expect file): its
   weights exist, its training report says OK at the right steps, and its hashes equal
   the recorded ones where a value was recorded (snapshot digest, LoRA and head sha256).
   Every hash is written to the plan, so the ones first measured here are recorded too.
4. **Every image to be read** -- Panel B, its luma companion and the 891 threshold reals
   (491 COCO validation + 400 held-out Open Images, the union k = 0 threshold) -- is copied
   once to local disk with its sha256 checked in the same read (``p3_s2d_inputs.stage``).
5. **Once only.** A model whose stage summary exists here or in an attached previous
   version is complete: it is carried forward and never scored again. A model with
   scores but no summary was interrupted (a crash): it is listed in
   ``incomplete_previous`` and scored again, and the log must record why.

The stage is ``clean`` (every model on Panel B + luma + threshold reals) while any model
lacks it, then ``robust`` (H5: the robustness models under the 12 conditions, Panel B
test rows), then ``complete``. Exit 0 when the plan is written, 1 when any check fails.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import file_sha256, read_manifest  # noqa: E402
from dbp.eval import kshot  # noqa: E402
from dbp.eval.heldout_scores import snapshot_digest  # noqa: E402
from dbp.eval.perturb import CONDITIONS  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p2_materialise_panel import panel_digest  # noqa: E402
from p3_s2d_inputs import stage, write_record, write_rows  # noqa: E402

#: Per-image scores live under reports/ so the reports zip carries them back.
FINAL = Path("reports") / "p3_final_scores"
SUMMARIES = Path("reports") / "p3_final"
LISTS = Path("reports") / "p3_final_lists"
PLAN = Path("reports") / "p3_final_plan.json"
#: Corpus-root links the runbook stages: Panel B, the COCO reals, the Open Images reals.
PANEL_LINK, REALS_LINK, OI_LINK = "panel", "reals", "oi_reals"
LIST_COLUMNS = ["image_id", "relative_path", "label", "view", "role", "slot", "sha256"]
CLEAN_LISTS = ("panel_b", "luma", "threshold_reals")


def summary_path(root: Path, model: str, stage_name: str) -> Path:
    """Where a model's stage summary lives under a ``thesis_dbp`` root."""
    return root / SUMMARIES / f"{model}_{stage_name}.json"


def is_complete(root: Path, model: str, stage_name: str) -> bool:
    """A full (not a smoke) stage summary exists."""
    path = summary_path(root, model, stage_name)
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("limit") is None


def check_section5(code: Path, expect: dict[str, Any], models: list[dict[str, Any]]
                   ) -> tuple[dict[str, Any], list[str]]:
    """Every section-5 value committed, and the models built from it."""
    s5, problems = expect["section5"], []
    record: dict[str, Any] = {}
    for key in ("degradation_choice", "nstar", "oi_manifest"):
        path = code / s5[key]
        digest = file_sha256(path) if path.is_file() else None
        record[key] = {"path": s5[key], "sha256": digest}
        if digest != s5[f"{key}_sha256"]:
            problems.append(f"section 5: {s5[key]} is missing or not the committed file")
    if problems:
        return record, problems
    choice = json.loads((code / s5["degradation_choice"]).read_text(encoding="utf-8"))
    if choice.get("chosen_p") != s5["chosen_p"] or choice.get("chosen") != s5["chosen"]:
        problems.append(f"section 5: the degradation result is {choice.get('chosen')} "
                        f"p={choice.get('chosen_p')}, expected {s5['chosen']} p={s5['chosen_p']}")
    tag = f"p{round(100 * s5['chosen_p']):02d}"
    for spec in models:
        if spec["protocol"].startswith(("s2b_deg_", "s2d_")) and not spec["protocol"].endswith(tag):
            problems.append(f"section 5: {spec['id']} is not the chosen p ({tag})")
    nstar = json.loads((code / s5["nstar"]).read_text(encoding="utf-8"))
    n_star, n_max = nstar["n_star"]["step"], nstar["rule"]["n_max"]
    if (n_star, n_max) != (s5["n_star"], s5["n_max"]):
        problems.append(f"section 5: N* {n_star} / N_max {n_max}, expected "
                        f"{s5['n_star']} / {s5['n_max']}")
    for spec in models:
        if spec["kind"] != "snapshot":
            continue
        fold = spec["protocol"].removeprefix("lolo_f")
        want = nstar["nested"][fold]["step"] if spec["protocol"].startswith("lolo_f") else n_star
        if spec["step"] != want:
            problems.append(f"section 5: {spec['id']} at step {spec['step']}, the frozen "
                            f"selection says {want}")
    oi = read_manifest(code / s5["oi_manifest"])
    split = {s: sum(1 for r in oi if r.get("split") == s) for s in ("train", "held_out")}
    if split != {"train": s5["oi_train"], "held_out": s5["oi_held_out"]}:
        problems.append(f"section 5: the Open Images split is {split}")
    rules = {str(k): [kshot.quantile_rule(int(k)).j, kshot.quantile_rule(int(k)).bound_met]
             for k in s5["kshot"]}
    if rules != s5["kshot"]:
        problems.append(f"section 5: the k-shot rule gives {rules}, frozen {s5['kshot']}")
    record.update({"chosen": choice.get("chosen"), "chosen_p": choice.get("chosen_p"),
                   "n_star": n_star, "n_max": n_max, "oi_split": split, "kshot_j": rules,
                   "nested": {f: v["step"] for f, v in nstar["nested"].items()}})
    return record, problems


def check_panel(code: Path, expect: dict[str, Any]) -> tuple[dict[str, Any], list[str],
                                                             list[dict[str, str]],
                                                             list[dict[str, str]]]:
    """The committed Panel B manifests and their four frozen digests."""
    spec, problems = expect["panel_b"], []
    record: dict[str, Any] = {}
    for key in ("images", "luma"):
        digest = file_sha256(code / spec[key]) if (code / spec[key]).is_file() else None
        record[key] = {"path": spec[key], "sha256": digest}
        if digest != spec[f"{key}_sha256"]:
            problems.append(f"Panel B: {spec[key]} is missing or not the committed file")
    if problems:
        return record, problems, [], []
    rows = read_manifest(code / spec["images"])
    luma = read_manifest(code / spec["luma"])
    digests = {"digest": panel_digest(rows), "luma_digest": panel_digest(luma),
               "test_digest": panel_digest([r for r in rows if r["role"] == "test"]),
               "calibration_digest": panel_digest([r for r in rows
                                                   if r["role"] == "calibration"])}
    record.update({"rows": len(rows), "luma_rows": len(luma), **digests})
    for key, value in digests.items():
        if value != spec[key]:
            problems.append(f"Panel B: {key} {value} is not the frozen {spec[key]}")
    return record, problems, rows, luma


def build_lists(code: Path, expect: dict[str, Any], rows: list[dict[str, str]],
                luma: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """The three scoring lists, every path relative to the staged corpus root."""
    def panel(row: dict[str, str]) -> dict[str, str]:
        return {"image_id": row["image_id"], "relative_path": f"{PANEL_LINK}/"
                f"{row['relative_path']}", "label": row["label"], "view": row["view"],
                "role": row["role"], "slot": row["panel_b_slot"], "sha256": row["sha256"]}

    spec = expect["threshold_reals"]
    sha = {r["image_id"]: r["sha256"] for r in read_manifest(code / spec["reals_manifest"])}
    reals = []
    for row in read_manifest(code / spec["coco_val"]):
        reals.append({"image_id": row["image_id"], "relative_path": row["relative_path"],
                      "label": "real", "view": "coco_val", "role": "threshold",
                      "slot": "coco_val", "sha256": sha[row["image_id"]]})
    for row in read_manifest(code / expect["section5"]["oi_manifest"]):
        if row.get("split") == "held_out":
            reals.append({"image_id": row["image_id"],
                          "relative_path": f"{OI_LINK}/t_oi/{row['image_id']}.png",
                          "label": "real", "view": "oi_held_out", "role": "threshold",
                          "slot": "oi_held_out", "sha256": row["sha256"]})
    return {"panel_b": [panel(r) for r in rows], "luma": [panel(r) for r in luma],
            "threshold_reals": reals}


def check_model(spec: dict[str, Any], sources: dict[str, Path], s5: dict[str, Any]
                ) -> tuple[dict[str, Any], list[str]]:
    """One model's weights, training report and hashes."""
    problems: list[str] = []
    if spec["source"] not in sources:
        return {}, [f"{spec['id']}: its source '{spec['source']}' is not attached"]
    root = sources[spec["source"]]
    if spec["kind"] == "snapshot":
        weights = root / "outputs" / "p3" / "s2" / spec["id"] / "snapshots" / \
            f"step_{spec['step']:05d}"
        report_path = root / "reports" / f"p2_s2_{spec['id']}.json"
    else:
        weights, report_path = root / spec["id"], None
    lora, head = weights / "lora", weights / "head.pt"
    if not (lora / "adapter_model.safetensors").is_file() or not head.is_file() \
            or not (lora / "adapter_config.json").is_file():
        return {}, [f"{spec['id']}: no weights at {weights}"]
    record = {"id": spec["id"], "kind": spec["kind"], "protocol": spec["protocol"],
              "seed": spec["seed"], "runs_config": spec["runs_config"], "step": spec.get("step"),
              "source": spec["source"], "weights": str(weights), "lora": str(lora),
              "head": str(head), "lora_sha256": file_sha256(lora / "adapter_model.safetensors"),
              "head_sha256": file_sha256(head)}
    if spec["kind"] == "snapshot":
        record["snapshot_sha256"] = snapshot_digest(weights)
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {})
        lolo = spec["protocol"].startswith("lolo_f")
        ok = report.get("status") == "OK" and report.get("fixed_steps") == s5["n_max"] and (
            (lolo and report.get("steps_run") == s5["n_max"]
             and spec["step"] in report.get("snapshots", []))
            or (not lolo and report.get("steps_run") == spec["step"]
                and report.get("stop_at") == spec["step"]
                and report.get("schedule_steps") == s5["n_max"]))
        if not ok:
            problems.append(f"{spec['id']}: the training report does not say a finished run "
                            f"at step {spec['step']} ({report_path})")
    for key, want in spec.get("expect", {}).items():
        if record.get(key) != want:
            problems.append(f"{spec['id']}: {key} {record.get(key)} is not the recorded {want}")
    record["recorded"] = sorted(spec.get("expect", {}))
    return record, problems


def carry(previous: list[Path], here: Path, models: list[str], stage_name: str,
          log: Any) -> tuple[list[str], list[str]]:
    """Copy complete stages forward; list interrupted ones. Returns (carried, incomplete)."""
    carried, incomplete = [], []
    for model in models:
        if is_complete(here, model, stage_name):
            continue
        for root in previous:
            if is_complete(root, model, stage_name):
                target = here / FINAL / model / stage_name
                if not target.exists():
                    shutil.copytree(root / FINAL / model / stage_name, target)
                summary = summary_path(here, model, stage_name)
                summary.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_path(root, model, stage_name), summary)
                carried.append(model)
                log.info("carried %s (%s) forward from %s", model, stage_name, root)
                break
            if (root / FINAL / model / stage_name).is_dir():
                incomplete.append(model)
    return carried, sorted(set(incomplete) - set(carried))


def main() -> int:
    """Check everything; write the plan. Exit 0 when the plan is written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                        help="lolo, train, tune, s2d (thesis_dbp roots), p2ref (dataset root).")
    parser.add_argument("--root", type=Path, required=True,
                        help="Corpus root with the panel/, reals/ and oi_reals/ links.")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--previous", action="append", type=Path, default=[],
                        help="thesis_dbp roots of this notebook's previous versions.")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, default=THESIS_DBP / PLAN)
    args = parser.parse_args()
    log = setup_logging("p3_final_preflight")
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    sources = {}
    for item in args.source:
        name, _, path = item.partition("=")
        sources[name] = Path(path)
    models = expect["models"]
    record: dict[str, Any] = {"what": "P3 final scoring plan (pre-registration sections 2, 5, 8)"}
    problems: list[str] = []

    record["section5"], found = check_section5(THESIS_DBP, expect, models)
    problems += found
    record["panel_b"], found, rows, luma = check_panel(THESIS_DBP, expect)
    problems += found
    record["models"] = {}
    for spec in models:
        model, found = check_model(spec, sources, expect["section5"])
        record["models"][spec["id"]] = model
        problems += found
    if len(models) != expect["n_models"] or len({m["id"] for m in models}) != len(models):
        problems.append(f"{len(models)} models listed, expected {expect['n_models']} distinct")
    if problems:
        return finish(args.out, {**record, "ok": False, "problems": problems}, log)

    # Every image read once, its sha256 checked in that read, copied to local disk.
    lists = build_lists(THESIS_DBP, expect, rows, luma)
    record["lists"] = {}
    for name, items in lists.items():
        path = THESIS_DBP / LISTS / f"{name}.csv"
        write_rows(path, LIST_COLUMNS, items)
        record["lists"][name] = {"path": str(path), "rows": len(items),
                                 "sha256": file_sha256(path)}
    expected = {r["relative_path"]: r["sha256"] for items in lists.values() for r in items}
    staged = stage(sorted(expected), expected, args.root, args.local_root, args.workers,
                   expect.get("margin_gb", 2.0), log)
    record["staged"] = staged
    if not staged["ok"] or staged["verified"] != len(expected):
        return finish(args.out, {**record, "ok": False, "problems": [
            f"staging FAILED: {staged.get('reason') or 'not every file verified'}"]}, log)

    # Once only: carry complete stages forward; choose this session's stage.
    here = THESIS_DBP
    ids = [m["id"] for m in models]
    robust = expect["robust"]["models"]
    clean_carried, clean_incomplete = carry(args.previous, here, ids, "clean", log)
    robust_carried, robust_incomplete = carry(args.previous, here, robust, "robust", log)
    clean_due = [m for m in ids if not is_complete(here, m, "clean")]
    robust_due = [m for m in robust if not is_complete(here, m, "robust")]
    stage_name = "clean" if clean_due else ("robust" if robust_due else "complete")
    due = clean_due if stage_name == "clean" else robust_due if stage_name == "robust" else []
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    shares = {d: due[i::len(devices)] for i, d in enumerate(devices)}
    record.update({
        "stage": stage_name, "due": due,
        "complete": {"clean": [m for m in ids if m not in clean_due],
                     "robust": [m for m in robust if m not in robust_due]},
        "carried": {"clean": clean_carried, "robust": robust_carried},
        "incomplete_previous": {"clean": clean_incomplete, "robust": robust_incomplete},
        "jobs": [f"{d}:{','.join(share)}" for d, share in shares.items() if share],
        "robust": {"models": robust, "conditions": [c for c in CONDITIONS if c != "clean"],
                   "rows": "panel_b test rows"},
        "local_root": str(args.local_root),
    })
    if clean_incomplete or robust_incomplete:
        log.warning("INTERRUPTED earlier and scored again (record the reason in the log): %s",
                    record["incomplete_previous"])
    log.info("stage %s: %d model(s) due %s; jobs %s", stage_name, len(due), due, record["jobs"])
    return finish(args.out, {**record, "ok": True, "problems": []}, log)


def finish(out: Path, record: dict[str, Any], log: Any) -> int:
    """Write the plan record; 0 when it is ok."""
    write_record(out, record)
    for problem in record["problems"]:
        log.error("%s", problem)
    log.info("final scoring pre-flight: %s", "PASSED" if record["ok"] else "FAILED")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
