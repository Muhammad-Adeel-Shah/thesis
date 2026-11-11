"""P4: merge a sharded row's checkpoints into the normal row (user decision 2026-11-09).

Run after the scoring. For every sharded row whose shards are all done
(``shard<k>.done.json``) and that has no summary yet:

1. each shard's checkpoint store is loaded -- every part checked against its ledger line,
   the store's ``meta.json`` against the plan's shard manifest;
2. the saved rows are put back into each committed list's order
   (:func:`dbp.baselines.shards.merge`), which stops unless every (list, image id) is
   present exactly once;
3. the row's per-list CSVs are written exactly as an unsharded row's would be
   (``p4_score.write_scores``), then its summary, last.

Exit 0 when every due row was merged (or none was due), 2 when a shard is not done yet
(nothing written; not a failure), 1 on any verification failure.
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

from dbp.baselines.common import BaselineError  # noqa: E402
from dbp.baselines.shards import ShardStore, merge  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import read_rows  # noqa: E402
from p4_preflight import CLEAN_LISTS, SCORES, SUMMARIES, is_complete, shard_done  # noqa: E402
from p4_score import FLAG, PREPROCESSING, write_json, write_scores  # noqa: E402


def merge_row(row: str, plan: dict[str, Any], root: Path, log: Any) -> dict[str, Any] | None:
    """Merge one sharded row; None when a shard is not done yet."""
    spec = plan["shards"][row]
    done = {}
    for unit, shard in spec["units"].items():
        path = shard_done(root, row, int(shard["shard"]))
        if not path.is_file():
            log.info("%s: %s is not done yet; nothing merged", row, unit)
            return None
        done[unit] = json.loads(path.read_text(encoding="utf-8"))
    saved: list[dict[str, str]] = []
    columns: list[str] | None = None
    for unit, shard in spec["units"].items():
        store_root = root / SCORES / row / "_shards" / f"shard{shard['shard']}"
        meta = json.loads((store_root / "meta.json").read_text(encoding="utf-8"))
        if meta["manifest_sha256"] != shard["sha256"]:
            raise BaselineError(f"{unit}: checkpoints of another shard manifest")
        if columns is not None and meta["columns"] != columns:
            raise BaselineError(f"{unit}: columns differ between shards")
        columns = meta["columns"]
        rows = ShardStore(store_root, columns).load()
        if len(rows) != int(shard["items"]):
            raise BaselineError(f"{unit}: {len(rows)} saved rows, the manifest has "
                                f"{shard['items']}")
        if file_sha256(store_root / "ledger.jsonl") != done[unit]["ledger_sha256"]:
            raise BaselineError(f"{unit}: the ledger changed after the shard finished")
        saved += rows
    lists = {name: read_rows(Path(plan["list_files"][name]))[1] for name in CLEAN_LISTS}
    ordered = merge(saved, lists)
    extra = [c for c in columns if c.startswith("lpips2_")]
    first = next(iter(done.values()))
    baseline = plan["rows"][row]["baseline"]
    results: dict[str, Any] = {}
    seconds = sum(d["seconds"] for d in done.values())
    items = sum(d["items"] for d in done.values())
    for name, rows in ordered.items():
        path = root / SCORES / row / f"{name}.csv"
        values = [{"score": float(r["score"]), "flag": int(r["flag"]),
                   **{c: float(r[c]) for c in extra}} for r in rows]
        results[name] = {"rows": len(rows), "file": str(path),
                         "sha256": write_scores(path, lists[name], values, extra),
                         FLAG[baseline]: sum(v["flag"] for v in values),
                         "seconds": None, "seconds_per_image": round(seconds / items, 5)}
    resumes = {}
    for unit, shard in spec["units"].items():
        path = root / SCORES / row / "_shards" / f"shard{shard['shard']}" / "resumes.jsonl"
        resumes[unit] = [json.loads(line) for line in path.read_text("utf-8").splitlines()
                         if line.strip()] if path.is_file() else []
    summary = {
        "row": row, "baseline": baseline, "condition": plan["rows"][row]["condition"],
        "crop": None, "preprocessing": PREPROCESSING[baseline],
        "score": "larger = more likely fake",
        **{k: first[k] for k in ("aes", "lpips", "versions", "device", "dry_run")
           if k in first},
        "limit": None, "lists": results,
        "execution": {"what": "split across GPUs by capped pixels; append-only checkpoints; "
                              "every resume re-scored saved rows first (user 2026-11-09)",
                      "shards": {u: {k: d[k] for k in ("items", "pixels", "parts",
                                                         "ledger_sha256", "manifest_sha256",
                                                         "seconds", "device")}
                                 for u, d in done.items()},
                      "resume_checks": resumes},
    }
    write_json(root / SUMMARIES / f"{row}.json", summary)
    log.info("%s merged: %s", row, {k: v["rows"] for k, v in results.items()})
    return summary


def main() -> int:
    """Merge every due sharded row."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", type=Path, default=THESIS_DBP / "reports" / "p4_plan.json")
    parser.add_argument("--root", type=Path, default=THESIS_DBP)
    args = parser.parse_args()
    log = setup_logging("p4_merge")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    pending = False
    for row in plan.get("shards", {}):
        if is_complete(args.root, row):
            log.info("%s is already complete; not merged again", row)
            continue
        try:
            if merge_row(row, plan, args.root, log) is None:
                pending = True
        except (BaselineError, OSError, KeyError, ValueError) as error:
            log.error("MERGE FAILED for %s: %s", row, error)
            return 1
    return 2 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
