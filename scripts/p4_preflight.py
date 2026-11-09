"""P4 baselines, step 1 of the scoring session: pre-flight and plan (CPU, before any score).

Spec ``docs/P4_BASELINES_SPEC.md`` sections 2-4. Writes ``reports/p4_plan.json`` for the
workers (``p4_score.py``) only when every check holds:

1. **The spec is the frozen one**: ``docs/P4_BASELINES_SPEC.md`` has the sha256 recorded
   at the freeze (tag ``p4-baselines-frozen``). The code is git's bytes, so present means
   committed.
2. **The inputs are the P3 final scoring's**: the three committed lists (Panel B 9,405,
   luma 700, threshold reals 891) have their committed sha256 and row counts.
3. **Every weight file is the recorded one**: each file in the attached weights output has
   the sha256 in the expect file -- including FIRE's, which was recorded by the weights
   session and committed before any baseline is scored -- and that session's report says
   OK.
4. **Every image** is copied once to local disk with its sha256 checked in the same read
   (``p3_s2d_inputs.stage``).
5. **Once only.** A row whose summary exists here or in an attached previous version is
   complete: it is carried forward and never scored again. A row with scores but no
   summary was interrupted: it is listed in ``incomplete_previous`` and scored again (the
   log must say why).
6. **Sharded rows** (``expect["shards"]``, AEROBLADE; user decision 2026-11-09): the
   row's images are dealt to its shards by capped pixel count, one shard manifest each
   (``reports/p4_shards/``). Checkpoints of earlier versions are carried forward
   append-only, so a shard resumes where it stopped (the worker re-scores saved rows
   first). The work units are ``<row>#<k>``; when every shard is done, the stage is
   ``merge`` (``p4_merge.py`` builds the row).

Exit 0 when the plan is written, 1 when any check fails.
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

from dbp.baselines.common import BaselineError  # noqa: E402
from dbp.baselines.shards import assign, capped_pixels  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import read_rows, stage, write_record, write_rows  # noqa: E402

SCORES = Path("reports") / "p4_baseline_scores"
SHARDS = Path("reports") / "p4_shards"
SHARD_COLUMNS = ["order", "list", "image_id", "relative_path", "pixels"]
SUMMARIES = Path("reports") / "p4_baselines"
PLAN = Path("reports") / "p4_plan.json"
CLEAN_LISTS = ("panel_b", "luma", "threshold_reals")
#: The weights session's report, inside its attached output.
WEIGHTS_REPORT = Path("Implementation") / "thesis_dbp" / "reports" / "p4_weights.json"
#: The one problem the user ruled is not a stop (2026-11-05): CO-SPY's artifact file
#: holding the VAE. Code at bffdfca reported it as a problem; the session of 2026-11-05
#: found those tensors equal to the pinned sd15 VAE (max abs diff 0.0).
WAIVED = "CO-SPY artifact_weights.pth holds"


def weights_report_ok(report: dict[str, Any]) -> bool:
    """OK, or every problem is the waived CO-SPY VAE notice."""
    if report.get("ok") is True:
        return True
    problems = report.get("problems") or []
    return bool(problems) and all(p.startswith(WAIVED) for p in problems)


def summary_path(root: Path, row: str) -> Path:
    """Where a row's summary lives under a ``thesis_dbp`` root."""
    return root / SUMMARIES / f"{row}.json"


def is_complete(root: Path, row: str) -> bool:
    """A full (not a smoke) summary exists."""
    path = summary_path(root, row)
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("limit") is None


def row_lists(spec: dict[str, Any]) -> list[str]:
    """The lists a row scores: all three when clean, Panel B's test rows under a condition."""
    return list(CLEAN_LISTS) if spec.get("condition", "clean") == "clean" else ["panel_b_test"]


def check_spec(code: Path, expect: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The frozen spec is present, byte for byte."""
    path = code / expect["spec"]["path"]
    digest = file_sha256(path) if path.is_file() else None
    ok = digest == expect["spec"]["sha256"]
    return {"path": expect["spec"]["path"], "sha256": digest}, (
        [] if ok else [f"spec: {expect['spec']['path']} is not the frozen file ({digest})"])


def check_lists(code: Path, expect: dict[str, Any]) -> tuple[dict[str, Any], list[str],
                                                             dict[str, list[dict[str, str]]]]:
    """The committed P3 final lists, by sha256 and row count."""
    record: dict[str, Any] = {}
    problems: list[str] = []
    lists: dict[str, list[dict[str, str]]] = {}
    for name in CLEAN_LISTS:
        spec = expect["lists"][name]
        path = code / expect["lists"]["dir"] / f"{name}.csv"
        digest = file_sha256(path) if path.is_file() else None
        rows = read_rows(path)[1] if digest else []
        record[name] = {"path": str(path), "sha256": digest, "rows": len(rows)}
        if digest != spec["sha256"] or len(rows) != spec["rows"]:
            problems.append(f"list {name}: {digest} / {len(rows)} rows is not the committed "
                            f"{spec['sha256']} / {spec['rows']}")
        lists[name] = rows
    lists["panel_b_test"] = [r for r in lists.get("panel_b", []) if r["role"] == "test"]
    record["panel_b_test"] = {"rows": len(lists["panel_b_test"])}
    if len(lists["panel_b_test"]) != expect["lists"]["panel_b_test_rows"]:
        problems.append(f"Panel B test rows: {len(lists['panel_b_test'])}, expected "
                        f"{expect['lists']['panel_b_test_rows']}")
    return record, problems, lists


def check_weights(root: Path, expect: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Every weight file in the attached weights output is the recorded one."""
    record: dict[str, Any] = {}
    problems: list[str] = []
    report_path = root.parent / WEIGHTS_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8")) \
        if report_path.is_file() else {}
    record["weights_report"] = {"path": str(report_path), "ok": report.get("ok"),
                                "problems": report.get("problems"),
                                "accepted": weights_report_ok(report),
                                "commit": report.get("commit")}
    if not weights_report_ok(report):
        problems.append(f"the weights session's report ({report_path}) is missing or not OK")
    for name, spec in expect["weights"].items():
        path = root / spec["file"]
        if not spec.get("sha256"):
            problems.append(f"weights {name}: no recorded sha256 in the expect file; commit "
                            f"the weights session's value before any scoring")
            continue
        digest = file_sha256(path) if path.is_file() else None
        record[name] = {"path": str(path), "sha256": digest}
        if digest != spec["sha256"]:
            problems.append(f"weights {name}: {path} is missing or not {spec['sha256']}")
    return record, problems


def carry(previous: list[Path], here: Path, rows: list[str], log: Any
          ) -> tuple[list[str], list[str]]:
    """Copy complete rows forward; list interrupted ones. Returns (carried, incomplete)."""
    carried, incomplete = [], []
    for row in rows:
        if is_complete(here, row):
            continue
        for root in previous:
            if is_complete(root, row):
                target = here / SCORES / row
                if not target.exists():
                    shutil.copytree(root / SCORES / row, target)
                summary = summary_path(here, row)
                summary.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_path(root, row), summary)
                carried.append(row)
                log.info("carried %s forward from %s", row, root)
                break
            if (root / SCORES / row).is_dir():
                incomplete.append(row)
    return carried, sorted(set(incomplete) - set(carried))


def shard_dir(root: Path, row: str) -> Path:
    """Where a sharded row's checkpoints live under a ``thesis_dbp`` root."""
    return root / SCORES / row / "_shards"


def shard_done(root: Path, row: str, k: int) -> Path:
    """The record a shard writes when every one of its items is saved."""
    return shard_dir(root, row) / f"shard{k}.done.json"


def carry_checkpoints(previous: list[Path], here: Path, row: str, log: Any) -> int:
    """Copy a sharded row's checkpoint files from earlier versions; never overwrite.

    A file already present must be byte-identical to the earlier one (append-only).
    Returns how many files were copied.
    """
    copied = 0
    for root in previous:
        source = shard_dir(root, row)
        if not source.is_dir():
            continue
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            target = shard_dir(here, row) / path.relative_to(source)
            if target.exists():
                if target.read_bytes() != path.read_bytes():
                    raise BaselineError(f"checkpoint {target} differs from {path}: append-only "
                                        f"checkpoints were changed")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    if copied:
        log.info("carried %d checkpoint file(s) of %s forward", copied, row)
    return copied


def build_shards(row: str, spec: dict[str, Any], lists: dict[str, list[dict[str, str]]],
                 local_root: Path, previous: list[Path], log: Any) -> dict[str, Any]:
    """Deal a row's images to its shards by capped pixels; write the shard manifests.

    The deal depends only on the committed lists and the image sizes (read from the
    staged, hash-checked copies), so every session builds the same manifests; a shard's
    ledger records the manifest's sha256 and the worker refuses a different one.
    """
    from PIL import Image

    items = []
    for name in CLEAN_LISTS:
        for row_ in lists[name]:
            with Image.open(local_root / row_["relative_path"]) as handle:
                width, height = handle.size
            items.append({"order": len(items), "list": name, "image_id": row_["image_id"],
                          "relative_path": row_["relative_path"],
                          "pixels": capped_pixels(width, height)})
    deal = assign(items, int(spec["n"]))
    carried = carry_checkpoints(previous, THESIS_DBP, row, log)
    units: dict[str, Any] = {}
    for k, shard in enumerate(deal):
        manifest = THESIS_DBP / SHARDS / f"{row}_shard{k}.csv"
        write_rows(manifest, SHARD_COLUMNS, [{**i, "order": str(i["order"]),
                                              "pixels": str(i["pixels"])} for i in shard])
        ledger = shard_dir(THESIS_DBP, row) / f"shard{k}" / "ledger.jsonl"
        saved = sum(json.loads(line)["rows"] for line in ledger.read_text("utf-8").splitlines()
                    if line.strip()) if ledger.is_file() else 0
        units[f"{row}#{k}"] = {
            "shard": k, "manifest": str(manifest), "sha256": file_sha256(manifest),
            "items": len(shard), "pixels": sum(i["pixels"] for i in shard),
            "saved_rows": saved, "saved_pixels": sum(i["pixels"] for i in shard[:saved]),
            "done": shard_done(THESIS_DBP, row, k).is_file(),
            "store": str(shard_dir(THESIS_DBP, row) / f"shard{k}")}
    log.info("%s: %d image(s), %.3f Gpx, dealt to %d shard(s): %s", row, len(items),
             sum(i["pixels"] for i in items) / 1e9, len(deal),
             {u: (v["items"], round(v["pixels"] / 1e9, 3), v["saved_rows"]) for u, v in
              units.items()})
    return {**{k: v for k, v in spec.items()}, "units": units, "carried_files": carried,
            "pixels": sum(i["pixels"] for i in items)}


def main() -> int:
    """Check everything; write the plan. Exit 0 when the plan is written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True,
                        help="The weights session's p4_weights/ directory.")
    parser.add_argument("--root", type=Path, required=True,
                        help="Corpus root with the panel/, reals/ and oi_reals/ links.")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--previous", action="append", type=Path, default=[],
                        help="thesis_dbp roots of this notebook's previous versions.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, default=THESIS_DBP / PLAN)
    args = parser.parse_args()
    log = setup_logging("p4_preflight")
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    record: dict[str, Any] = {"what": "P4 baseline scoring plan (spec sections 2-4)"}
    problems: list[str] = []

    record["spec"], found = check_spec(THESIS_DBP, expect)
    problems += found
    record["lists"], found, lists = check_lists(THESIS_DBP, expect)
    problems += found
    record["weights"], found = check_weights(args.weights, expect)
    problems += found
    if problems:
        return finish(args.out, {**record, "ok": False, "problems": problems}, log)

    expected = {r["relative_path"]: r["sha256"] for name in CLEAN_LISTS for r in lists[name]}
    staged = stage(sorted(expected), expected, args.root, args.local_root, args.workers,
                   expect.get("margin_gb", 2.0), log)
    record["staged"] = staged
    if not staged["ok"] or staged["verified"] != len(expected):
        return finish(args.out, {**record, "ok": False, "problems": [
            f"staging FAILED: {staged.get('reason') or 'not every file verified'}"]}, log)

    rows = list(expect["rows"])
    sharded = expect.get("shards", {})
    carried, incomplete = carry(args.previous, THESIS_DBP, rows, log)
    incomplete = [r for r in incomplete if r not in sharded]  # those resume (see below)
    rows_due = [r for r in rows if not is_complete(THESIS_DBP, r)]
    units: dict[str, dict[str, Any]] = {}
    merge_due: list[str] = []
    for row in rows_due:
        if row not in sharded:
            units[row] = {"row": row, "shard": None}
            continue
        try:
            record.setdefault("shards", {})[row] = build_shards(
                row, sharded[row], lists, args.local_root, args.previous, log)
        except (BaselineError, OSError) as error:
            return finish(args.out, {**record, "ok": False, "problems": [str(error)]}, log)
        pending = [u for u, info in record["shards"][row]["units"].items()
                   if not info["done"]]
        for unit in pending:
            units[unit] = {"row": row, "shard": record["shards"][row]["units"][unit]["shard"]}
        if not pending:
            merge_due.append(row)
    due = list(units)
    jobs = []
    for job in expect["jobs"]:
        device, _, members = job.partition(":")
        share = [m for m in members.split(",") if m in due]
        if share:
            jobs.append(f"{device}:{','.join(share)}")
    unplaced = sorted(set(due) - {m for j in jobs for m in j.partition(':')[2].split(',')})
    if unplaced:
        return finish(args.out, {**record, "ok": False, "problems": [
            f"units in no job: {unplaced}"]}, log)
    stage_name = "score" if due else ("merge" if merge_due else "complete")
    record.update({
        "stage": stage_name, "due": due, "units": units, "rows_due": rows_due,
        "merge_due": merge_due,
        "complete": [r for r in rows if r not in rows_due], "carried": carried,
        "incomplete_previous": incomplete, "jobs": jobs,
        "rows": {r: {**expect["rows"][r], "lists": row_lists(expect["rows"][r])}
                 for r in rows},
        "list_files": {name: str(THESIS_DBP / expect["lists"]["dir"] / f"{name}.csv")
                       for name in CLEAN_LISTS},
        "weights_root": str(args.weights), "local_root": str(args.local_root),
        "aeroblade_aes": expect["aeroblade_aes"], "batch": expect["batch"],
        "dryrun_arch": expect.get("dryrun_arch", {}),
    })
    if incomplete:
        log.warning("INTERRUPTED earlier and scored again (record the reason in the log): %s",
                    incomplete)
    log.info("stage %s: %d unit(s) due %s; jobs %s; merge due %s", stage_name, len(due), due,
             jobs, merge_due)
    return finish(args.out, {**record, "ok": True, "problems": []}, log)


def finish(out: Path, record: dict[str, Any], log: Any) -> int:
    """Write the plan record; 0 when it is ok."""
    write_record(out, record)
    for problem in record["problems"]:
        log.error("%s", problem)
    log.info("P4 pre-flight: %s", "PASSED" if record["ok"] else "FAILED")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
