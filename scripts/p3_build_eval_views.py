"""P3 Phase B: rebuild the evaluation views Panel B is drawn from, and prove they are P1's.

Each view is rebuilt by ``p1_build_eval.py`` with **the arguments its committed P1
summary records** (``reports/kaggle_20260928/p1_eval_<set>_summary.json``): globs, fields,
the JPEG-matched view, E5's pipeline-matched view, E8's stratified subsample. Nothing is
retyped.

Then it is checked, and the run stops on any failure:

* **every P2 panel row** of the view (all 12,150 drawn rows) is present in the rebuilt
  manifest with an **identical sha256**. The matched views are seeded re-encodes, so this
  proves byte-identical reproduction (I-010);
* the per-label counts equal P1's.

The rebuilt manifest's own sha256 is compared with P1's and reported. It records absolute
paths, so it matches only when the roots are P1's, as on Kaggle; it is informational.

Usage:
    python scripts/p3_build_eval_views.py --roots roots.json --p2-panel-sets DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

P1_SUMMARIES = THESIS_DBP / "reports" / "kaggle_20260928"

#: fetch set -> (P1 set id, the Panel B / P2 view, the manifest the view is drawn from)
VIEWS = {
    "e3": ("e3_raw", "e3_matched", "p1_eval_e3_raw_jpeg_matched_manifest.csv"),
    "e4": ("e4_wildrf_test", "e4_wildrf_matched",
           "p1_eval_e4_wildrf_test_jpeg_matched_manifest.csv"),
    "e5": ("e5_chameleon", "e5_chameleon_pipeline",
           "p1_eval_e5_chameleon_pipeline_matched_manifest.csv"),
    "e7": ("e7_dda_coco", "e7_dda_coco", "p1_eval_e7_dda_coco_manifest.csv"),
    "e8": ("e8_evalgen", "e8_evalgen", "p1_eval_e8_evalgen_manifest.csv"),
}


class ViewError(RuntimeError):
    """A rebuilt view does not reproduce P1."""


def p1_arguments(summary: dict, root: str) -> list[str]:
    """``p1_build_eval.py`` arguments, from a committed P1 summary."""
    args = ["--set-id", summary["set_id"], "--root", root]
    for pattern in summary["globs"].get("real", []):
        args += ["--real-glob", pattern]
    for pattern in summary["globs"].get("fake", []):
        args += ["--fake-glob", pattern]
    for name, depth in summary.get("fields", {}).items():
        args += ["--field", f"{name}={depth}"]
    if summary.get("jpeg_matched_view"):
        args.append("--jpeg-match")
    pipeline = summary.get("pipeline_matched_view")
    if pipeline:
        args += ["--pipeline-match-width", str(pipeline["target_width"]),
                 "--pipeline-match-quality", str(pipeline["quality"]),
                 "--pipeline-match-subsampling", str(pipeline["subsampling"]),
                 "--pipeline-match-resample", str(pipeline["resample"])]
    subsample = summary.get("subsample") or {}
    if subsample.get("applied"):
        args += ["--subsample", str(subsample["per_stratum"]),
                 "--stratify-by", str(subsample["stratify_by"])]
    return args


def read_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify(view: str, rebuilt: list[dict[str, str]], p2_rows: list[dict[str, str]],
           expected_labels: dict[str, int] | None) -> dict:
    """Every P2 row reproduced byte-identically; label counts as in P1."""
    by_id = {r["image_id"]: r for r in rebuilt}
    missing = [r["image_id"] for r in p2_rows if r["image_id"] not in by_id]
    mismatched = [r["image_id"] for r in p2_rows
                  if r["image_id"] in by_id and by_id[r["image_id"]]["sha256"] != r["sha256"]]
    if missing or mismatched:
        raise ViewError(f"{view}: {len(missing)} P2 row(s) missing (first {missing[:5]}), "
                        f"{len(mismatched)} with a different sha256 (first {mismatched[:5]})")
    labels = dict(Counter(r["label"] for r in rebuilt))
    if expected_labels is not None and labels != expected_labels:
        raise ViewError(f"{view}: label counts {labels} differ from P1's {expected_labels}")
    return {"rows": len(rebuilt), "labels": labels, "p2_rows_reproduced": len(p2_rows)}


def main() -> int:
    """Rebuild and verify every view. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--roots", required=True, type=Path, help="roots.json from the fetch.")
    parser.add_argument("--p2-panel-sets", required=True, type=Path,
                        help="The P2 panel's p2_panel_<view>.csv files (12,150 drawn rows).")
    parser.add_argument("--summaries", type=Path, default=P1_SUMMARIES,
                        help="P1 summaries to take the arguments from (the fixture overrides).")
    parser.add_argument("--no-label-check", action="store_true",
                        help="Dry-run fixture only: skip the P1 label-count comparison.")
    parser.add_argument("--out", type=Path, default=THESIS_DBP / "reports" /
                        "p3_eval_views_rebuild.json")
    args = parser.parse_args()
    log = setup_logging("p3_build_eval_views")
    roots = json.loads(args.roots.read_text(encoding="utf-8"))
    report: dict[str, dict] = {}
    try:
        for name, (set_id, view, manifest_name) in VIEWS.items():
            if name not in roots:
                raise ViewError(f"{name}: no root in {args.roots}")
            summary = json.loads((args.summaries / f"p1_eval_{set_id}_summary.json")
                                 .read_text(encoding="utf-8"))
            command = [sys.executable, "-u", str(THESIS_DBP / "scripts" / "p1_build_eval.py"),
                       *p1_arguments(summary, roots[name])]
            log.info("%s: %s", name, " ".join(command[3:]))
            result = subprocess.run(command, cwd=THESIS_DBP, check=False)
            if result.returncode != 0:
                raise ViewError(f"{name}: p1_build_eval.py exited {result.returncode}")
            manifest = THESIS_DBP / "reports" / manifest_name
            rebuilt = read_rows(manifest)
            p2_rows = read_rows(args.p2_panel_sets / f"p2_panel_{view}.csv")
            expected = None if args.no_label_check else summary.get("by_label")
            entry = verify(view, rebuilt, p2_rows, expected)
            recorded = (summary.get("pipeline_matched_view") or {}).get("manifest_sha256") \
                if "pipeline" in manifest_name else \
                (summary.get("jpeg_matched_view") or {}).get("manifest_sha256") \
                if "jpeg_matched" in manifest_name else summary.get("manifest_sha256")
            actual = file_sha256(manifest)
            entry.update({"manifest": manifest_name, "manifest_sha256": actual,
                          "p1_manifest_sha256": recorded,
                          "identical_to_p1": actual == recorded,
                          "base_manifest_sha256": file_sha256(
                              THESIS_DBP / "reports" / f"p1_eval_{set_id}_manifest.csv")})
            report[view] = entry
            log.info("%s VERIFIED: %d rows, all %d P2 rows reproduced byte-identically; "
                     "manifest identical to P1's: %s", view, entry["rows"],
                     entry["p2_rows_reproduced"], entry["identical_to_p1"])
    except ViewError as exc:
        log.error("VIEW REBUILD FAILED: %s", exc)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".tmp").write_text(json.dumps(report, indent=2) + "\n",
                                            encoding="utf-8")
    args.out.with_suffix(".tmp").replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
