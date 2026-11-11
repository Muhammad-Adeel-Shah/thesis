"""P3: rebuild the P2 development panel on ``muhammadadeelshah``, byte-identical (2026-10-19).

P2 read its panel from the ``phase-2`` notebook output owned by ``adeelshah29``. Every
thesis run uses ``muhammadadeelshah`` only (CLAUDE.md section 13), and the panel holds
WildRF and Chameleon imagery that must not cross accounts (section 6c), so P3 uses only a
copy rebuilt here. **The ``adeelshah29`` copy is never attached again.**

The session re-derives the evaluation views exactly as the Panel B session did
(``p3_fetch_eval_sets.py`` + ``p3_build_eval_views.py``, which write the derived views
where P1 wrote them), stages the corpus for the T-ctrl diagnostic view, and then this
script copies every panel row with P2's own materialiser, **unchanged**
(``p2_materialise_panel.materialise_view``): each file is re-hashed against the sha256
the committed ``reports/p2_panel_sets/p2_panel_<view>.csv`` records, and any mismatch
stops the run.

**Both digests are required** (``p2_materialise_panel.panel_digest``):

* all **12,150** rows -> ``2e1c1920...`` (P2's ``p2_panel_digest.json``, D-050);
* the **11,775** scoreable rows of ``p2_panel_images.csv`` (after I-047's crop filter)
  -> ``bff608c3...`` (pre-registration Amendment 1).

Every scoreable list row must also exist under the panel root at its listed path, so the
rebuilt directory serves the P2 lists unchanged. Exit 0 only when everything matches;
the record ``reports/p3_dev_panel_digest.json`` is written in every case.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import read_manifest  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: The views a P2 panel row can belong to, and the roots.json key of each derived view.
VIEW_ROOTS = {"e3_matched": "e3", "e4_wildrf_matched": "e4", "e5_chameleon_pipeline": "e5",
              "e7_dda_coco": "e7", "e8_evalgen": "e8"}
CORPUS_VIEW = "t_ctrl_diagnostic"
DIGEST_ALL = "2e1c192024ec52e42c4e1761afa6fb2e37c0f7a6983f5ff75d1174305550b769"
DIGEST_SCOREABLE = "bff608c3d8e00ea834532cd541849b4f21de803c8f8e27a55ca5e70424c51116"


def p2_materialiser() -> Any:
    """``p2_materialise_panel.py`` as a module, so its functions are used unchanged."""
    name = "p2_materialise_panel_for_dev"
    spec = importlib.util.spec_from_file_location(
        name, THESIS_DBP / "scripts" / "p2_materialise_panel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_record(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main() -> int:
    """Rebuild, verify and record. Returns 0 only if both digests and every path match."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--p2-panel-sets", type=Path,
                        default=THESIS_DBP / "reports" / "p2_panel_sets")
    parser.add_argument("--roots", type=Path, required=True, help="roots.json from the fetch.")
    parser.add_argument("--corpus-root", type=Path, required=True,
                        help="The staged P1 corpus (T-ctrl reals and their pairs).")
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--scoreable-list", type=Path, required=True,
                        help="p2_panel_images.csv (dbp-p2-s2-inputs): the 11,775 rows.")
    parser.add_argument("--views", default=",".join([*VIEW_ROOTS, CORPUS_VIEW]))
    parser.add_argument("--expect-digest-all", default=DIGEST_ALL)
    parser.add_argument("--expect-digest-scoreable", default=DIGEST_SCOREABLE)
    parser.add_argument("--cleanup-derived", type=Path, default=None,
                        help="After success, delete the derived view images (P1's "
                             "outputs/p1/eval) so they do not bloat the output.")
    parser.add_argument("--out", type=Path,
                        default=THESIS_DBP / "reports" / "p3_dev_panel_digest.json")
    args = parser.parse_args()
    log = setup_logging("p3_rebuild_dev_panel")
    m = p2_materialiser()

    roots = json.loads(args.roots.read_text(encoding="utf-8"))
    source_roots = {view: Path(roots[key]) for view, key in VIEW_ROOTS.items() if key in roots}
    source_roots[CORPUS_VIEW] = args.corpus_root
    views = [v for v in args.views.split(",") if v]
    record: dict[str, Any] = {
        "what": "P2 development panel rebuilt on muhammadadeelshah (P3 uses only this copy)",
        "source_of_truth": "reports/p2_panel_sets/p2_panel_<view>.csv (committed, with sha256)",
        "views": views, "ok": False, "problems": []}
    rows_all: list[dict[str, Any]] = []
    try:
        for view in views:
            rows = read_manifest(args.p2_panel_sets / f"p2_panel_{view}.csv")
            rows_all += m.materialise_view(view, rows, args.panel_root, source_roots, log)
    except (m.MaterialiseError, OSError) as error:
        record["problems"].append(f"materialise: {error}")
        log.error("%s", error)
        write_record(args.out, record)
        return 1

    with args.scoreable_list.open(encoding="utf-8", newline="") as handle:
        listed = list(csv.DictReader(handle))
    by_id = {r["image_id"]: r for r in rows_all}
    scoreable = [by_id[r["image_id"]] for r in listed if r["image_id"] in by_id]
    missing_ids = [r["image_id"] for r in listed if r["image_id"] not in by_id]
    missing_files = [r["relative_path"] for r in listed
                     if not (args.panel_root / r["relative_path"]).is_file()]
    digest_all, digest_scoreable = m.panel_digest(rows_all), m.panel_digest(scoreable)
    problems = []
    if digest_all != args.expect_digest_all:
        problems.append(f"all-rows digest {digest_all} != {args.expect_digest_all}")
    if digest_scoreable != args.expect_digest_scoreable:
        problems.append(f"scoreable digest {digest_scoreable} != {args.expect_digest_scoreable}")
    if missing_ids or missing_files:
        problems.append(f"{len(missing_ids)} listed id(s) not in the panel sets, "
                        f"{len(missing_files)} listed path(s) not under the panel root")
    total_bytes = sum((args.panel_root / r["relative_path"]).stat().st_size for r in rows_all)
    record.update({
        "ok": not problems, "problems": problems,
        "rows_all": len(rows_all), "digest_all": digest_all,
        "rows_scoreable": len(scoreable), "digest_scoreable": digest_scoreable,
        "expected": {"digest_all": args.expect_digest_all,
                     "digest_scoreable": args.expect_digest_scoreable},
        "by_view": dict(sorted(Counter(r["view"] for r in rows_all).items())),
        "by_view_label": {f"{v}/{lab}": n for (v, lab), n in sorted(
            Counter((r["view"], r["label"]) for r in rows_all).items())},
        "panel_bytes": total_bytes,
        "verification": "every file re-hashed against the committed sha256 "
                        "(p2_materialise_panel.materialise_view, unchanged)",
        "licence": "WildRF research-only, Chameleon academic-only: PRIVATE dataset only "
                   "(CLAUDE.md section 6c)",
    })
    write_record(args.out, record)
    if problems:
        for problem in problems:
            log.error("%s", problem)
        return 1
    if args.cleanup_derived is not None:
        target = args.cleanup_derived.resolve()
        if not target.is_relative_to((THESIS_DBP / "outputs").resolve()):
            log.error("refusing to delete %s: not under %s", target, THESIS_DBP / "outputs")
            return 1
        if target.is_dir():
            shutil.rmtree(target)
            log.info("deleted the derived view images under %s", target)
    log.info("DEV PANEL REBUILT: %d rows (digest %s), %d scoreable (digest %s), %.2f GB",
             len(rows_all), digest_all[:12], len(scoreable), digest_scoreable[:12],
             total_bytes / 1e9)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
