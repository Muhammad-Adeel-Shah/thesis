"""P3b scoring, step 1: the pre-flight and plan for the two frozen ViT-L models (CPU).

``docs/P3b_VITL_PREREG.md`` section 2: **Panel B is scored once per frozen ViT-L model**,
on P3's final-scoring path (``p3_final_score.py --backbone dinov3-vitl16``): the clean
stage (Panel B 9,405, its luma companion 700, the 891 threshold reals) and the robust
stage under the two H5 decision conditions only (``jpeg50``, ``resize050``; Panel B's test
rows). It refuses to plan any scoring unless every check holds, and writes
``reports/p3b_score_plan.json`` in the schema the scorer reads. Reused from
``p3_final_preflight.py`` unchanged: section 5, Panel B's frozen digests, the scoring lists,
the staging (every image copied once with its sha256 checked) and the carry-forward.

Per model (the expect file lists ``vitl_s2d_p35_seed0`` and ``vitl_s2d_p35_seed1``, each
naming its training run):

1. the final snapshot ``outputs/p3b/s2/<run>/snapshots/step_<N*>`` in the attached training
   output, and its training report there, which must be **the committed one** (sha256);
2. the **failure path** (section 5) passes on that report (``p3b_train_check.check_seed``);
3. its LoRA, head and snapshot hashes equal **the committed model record**
   (``reports/p3b_results/p3b_model_record.json``, made before any score, its sha256 in the
   expect file). **A mismatch refuses the model**; nothing is scored.

**Once only**, per list (user decision 2026-11-12): a complete stage here or in an attached
previous version is carried forward; an interrupted one has its finished lists carried
(``resumed``) and the scorer, run with ``--resume-lists``, keeps them and scores only the
rest. The stage is ``clean`` while any model lacks it, then ``robust``, then ``complete``.
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

import p3_final_preflight as p3f  # noqa: E402
from dbp.backbones.loader import BACKBONES  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.eval.heldout_scores import snapshot_digest  # noqa: E402
from dbp.eval.perturb import H5_DECISION  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import stage, write_rows  # noqa: E402
from p3b_train_check import check_seed  # noqa: E402

PLAN = Path("reports") / "p3b_score_plan.json"
HASHES = ("lora_sha256", "head_sha256", "snapshot_sha256")


def check_model(spec: dict[str, Any], train_root: Path, expect: dict[str, Any],
                record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """One frozen ViT-L model: weights, committed report, failure path, committed hashes."""
    s5 = expect["section5"]
    run, step = spec["run"], spec["step"]
    weights = train_root / "outputs" / "p3b" / "s2" / run / "snapshots" / f"step_{step:05d}"
    lora, head = weights / "lora", weights / "head.pt"
    if not (lora / "adapter_model.safetensors").is_file() or not head.is_file() \
            or not (lora / "adapter_config.json").is_file():
        return {}, [f"{spec['id']}: no weights at {weights}"]
    problems: list[str] = []
    report_path = train_root / "reports" / f"p2_s2_{run}.json"
    committed = THESIS_DBP / spec["training_report"]
    if not report_path.is_file() or file_sha256(report_path) != file_sha256(committed):
        problems.append(f"{spec['id']}: the training report in the attached output is not "
                        f"the committed {spec['training_report']}")
    report = json.loads(committed.read_text(encoding="utf-8"))
    if report.get("schedule_steps") != s5["n_max"] or report.get("fixed_steps") != s5["n_max"]:
        problems.append(f"{spec['id']}: trained on a {report.get('schedule_steps')}-step "
                        f"schedule, the recipe's is {s5['n_max']}")
    vitb = json.loads((THESIS_DBP / spec["vitb_report"]).read_text(encoding="utf-8"))
    backbone = expect["backbone"]
    verdict = check_seed(report, vitb, backbone, BACKBONES[backbone]["revision"], step)
    if verdict["verdict"] != "PASS":
        problems.append(f"{spec['id']}: the failure path (prereg section 5) does not pass: "
                        f"{verdict['reasons']}")
    model = {"id": spec["id"], "run": run, "kind": "snapshot", "protocol": spec["protocol"],
             "seed": spec["seed"], "runs_config": spec["runs_config"], "step": step,
             "source": "train", "weights": str(weights), "lora": str(lora), "head": str(head),
             "lora_sha256": file_sha256(lora / "adapter_model.safetensors"),
             "head_sha256": file_sha256(head), "snapshot_sha256": snapshot_digest(weights),
             "failure_path": verdict["verdict"]}
    recorded = record.get("runs", {}).get(run)
    if not recorded or not recorded.get("ok"):
        problems.append(f"{spec['id']}: no passing entry for {run} in the committed model record")
    else:
        differ = [k for k in HASHES if model[k] != recorded.get(k)]
        if differ:
            problems.append(f"{spec['id']}: {differ} differ from the committed model record -- "
                            f"refused, not scored")
    model["recorded"] = list(HASHES)
    return model, problems


def carry_lists(previous: list[Path], here: Path, models: list[str], stage_name: str,
                log: Any) -> list[str]:
    """An interrupted stage's finished lists (CSV + record), copied forward to be kept."""
    resumed = []
    for model in models:
        if p3f.is_complete(here, model, stage_name):
            continue
        for root in previous:
            source = root / p3f.FINAL / model / stage_name
            records = sorted(source.glob("*.done.json")) if source.is_dir() else []
            if not records:
                continue
            target = here / p3f.FINAL / model / stage_name
            target.mkdir(parents=True, exist_ok=True)
            for record in records:
                csv_name = record.name.removesuffix(".done.json") + ".csv"
                for name in (record.name, csv_name):
                    if (source / name).is_file() and not (target / name).exists():
                        shutil.copy2(source / name, target / name)
            resumed.append(model)
            log.info("resuming %s (%s): %d finished list(s) carried from %s", model,
                     stage_name, len(records), root)
            break
    return resumed


def main() -> int:
    """Check everything; write the plan. Exit 0 when the plan is written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True,
                        help="The thesis_dbp root of the ViT-L training session's output.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Corpus root with the panel/, reals/ and oi_reals/ links.")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--previous", action="append", type=Path, default=[],
                        help="thesis_dbp roots of this notebook's previous versions.")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, default=THESIS_DBP / PLAN)
    args = parser.parse_args()
    log = setup_logging("p3b_score_preflight")
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    models = expect["models"]
    record: dict[str, Any] = {"what": "P3b scoring plan (docs/P3b_VITL_PREREG.md section 2)",
                              "backbone": expect["backbone"]}
    problems: list[str] = []

    record["section5"], found = p3f.check_section5(THESIS_DBP, expect, models)
    problems += found
    record["panel_b"], found, rows, luma = p3f.check_panel(THESIS_DBP, expect)
    problems += found
    spec = expect["model_record"]
    path = THESIS_DBP / spec["path"]
    if not path.is_file() or file_sha256(path) != spec["sha256"]:
        problems.append(f"the model record {spec['path']} is missing or not the committed one")
        model_record: dict[str, Any] = {}
    else:
        model_record = json.loads(path.read_text(encoding="utf-8"))
    record["model_record"] = {"path": spec["path"], "sha256": spec["sha256"]}
    record["models"] = {}
    for spec_model in models:
        model, found = check_model(spec_model, args.train_root, expect, model_record)
        record["models"][spec_model["id"]] = model
        problems += found
    if len({m["id"] for m in models}) != len(models) or len(models) != expect["n_models"]:
        problems.append(f"{len(models)} models listed, expected {expect['n_models']} distinct")
    if problems:
        return p3f.finish(args.out, {**record, "ok": False, "problems": problems}, log)

    lists = p3f.build_lists(THESIS_DBP, expect, rows, luma)
    record["lists"] = {}
    for name, items in lists.items():
        list_path = THESIS_DBP / p3f.LISTS / f"{name}.csv"
        write_rows(list_path, p3f.LIST_COLUMNS, items)
        record["lists"][name] = {"path": str(list_path), "rows": len(items),
                                 "sha256": file_sha256(list_path)}
    expected = {r["relative_path"]: r["sha256"] for items in lists.values() for r in items}
    staged = stage(sorted(expected), expected, args.root, args.local_root, args.workers,
                   expect.get("margin_gb", 2.0), log)
    record["staged"] = staged
    if not staged["ok"] or staged["verified"] != len(expected):
        return p3f.finish(args.out, {**record, "ok": False, "problems": [
            f"staging FAILED: {staged.get('reason') or 'not every file verified'}"]}, log)

    here = THESIS_DBP
    ids = [m["id"] for m in models]
    carried, resumed = {}, {}
    for stage_name in ("clean", "robust"):
        carried[stage_name], _ = p3f.carry(args.previous, here, ids, stage_name, log)
        resumed[stage_name] = carry_lists(args.previous, here, ids, stage_name, log)
    clean_due = [m for m in ids if not p3f.is_complete(here, m, "clean")]
    robust_due = [m for m in ids if not p3f.is_complete(here, m, "robust")]
    stage_name = "clean" if clean_due else ("robust" if robust_due else "complete")
    due = clean_due if stage_name == "clean" else robust_due if stage_name == "robust" else []
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    shares = {d: due[i::len(devices)] for i, d in enumerate(devices)}
    record.update({
        "stage": stage_name, "due": due,
        "complete": {"clean": [m for m in ids if m not in clean_due],
                     "robust": [m for m in ids if m not in robust_due]},
        "carried": carried, "resumed": resumed,
        "incomplete_previous": {"clean": [], "robust": []},
        "jobs": [f"{d}:{','.join(share)}" for d, share in shares.items() if share],
        "robust": {"models": ids, "conditions": list(H5_DECISION),
                   "rows": "panel_b test rows"},
        "local_root": str(args.local_root),
    })
    log.info("stage %s: %d model(s) due %s; jobs %s; resumed %s", stage_name, len(due), due,
             record["jobs"], resumed)
    return p3f.finish(args.out, {**record, "ok": True, "problems": []}, log)


if __name__ == "__main__":
    raise SystemExit(main())
