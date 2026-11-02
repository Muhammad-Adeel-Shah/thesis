"""P3 Phase B: copy Panel B into one verified directory and freeze it with a digest.

Uses P2's materialiser functions unchanged (``p2_materialise_panel.py``): every image is
copied to ``<panel_root>/<view>/<image_id><suffix>`` and **verified against the sha256
its manifest records**; a mismatch stops the run. E7's FLUX.1 fakes and E7's reals also
get the greyscale-in-RGB luma companion (§6d), as in P2.

Writes:

* ``reports/p3_panel_b_images.csv`` — the frozen list: id, path, label, view, sha256,
  role (test | calibration), slot, and the E3 flag;
* ``reports/p3_panel_b_luma.csv`` — the luma companion;
* ``reports/p3_panel_b_digest.json`` — digests (all rows; test rows; calibration rows;
  luma), counts per view, slot and role. **This is what gets committed before any P3
  model trains.**
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import write_manifest  # noqa: E402
from dbp.eval.panel_b import PANEL_B_LUMA, PANEL_B_SEED  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402


def _p2_materialiser() -> Any:
    name = "p2_materialise_panel_for_b"
    spec = importlib.util.spec_from_file_location(
        name, THESIS_DBP / "scripts" / "p2_materialise_panel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VIEW_ROOTS = {"e3_matched": "e3", "e4_wildrf_matched": "e4", "e5_chameleon_pipeline": "e5",
              "e7_dda_coco": "e7", "e8_evalgen": "e8"}
LIST_COLUMNS = ["image_id", "relative_path", "label", "view", "sha256", "role", "panel_b_slot",
                "e3_number_in_p2"]


def main() -> int:
    """Materialise and freeze Panel B. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--selection", type=Path,
                        default=THESIS_DBP / "reports" / "p3_panel_b_selection.csv")
    parser.add_argument("--roots", required=True, type=Path, help="roots.json from the fetch.")
    parser.add_argument("--panel-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument(
        "--cleanup-derived", type=Path, default=None,
        help="After a successful freeze, delete this directory of derived matched-view "
             "images (P1's outputs/p1/eval): they are rebuilt where P1 wrote them so the "
             "manifests match P1's, but they must not bloat the notebook output.")
    args = parser.parse_args()
    log = setup_logging("p3_materialise_panel_b")
    m = _p2_materialiser()
    roots = json.loads(args.roots.read_text(encoding="utf-8"))
    with args.selection.open(encoding="utf-8", newline="") as handle:
        selection = list(csv.DictReader(handle))
    source_roots = {view: Path(roots[key]) for view, key in VIEW_ROOTS.items() if key in roots}
    args.panel_root.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    try:
        for view in VIEW_ROOTS:
            rows = [r for r in selection if r["panel_b_view"] == view]
            if not rows:
                continue
            materialised = m.materialise_view(view, rows, args.panel_root, source_roots, log)
            by_id = {r["image_id"]: r for r in rows}
            for row in materialised:
                source = by_id[row["image_id"]]
                row.update({"role": source["role"], "panel_b_slot": source["panel_b_slot"],
                            "e3_number_in_p2": source.get("e3_number_in_p2", "")})
            rows_out.extend(materialised)
        source_rows = {r["image_id"]: r for r in selection}
        luma = m.write_luma_variant(rows_out, source_rows,
                                    {v: set(s) for v, s in PANEL_B_LUMA.items()},
                                    args.panel_root, log)
    except m.MaterialiseError as exc:
        log.error("PANEL B MATERIALISE FAILED: %s", exc)
        return 1
    by_id = {r["image_id"]: r for r in rows_out}
    for row in luma:
        row.update({"role": by_id[row["image_id"]]["role"],
                    "panel_b_slot": by_id[row["image_id"]]["panel_b_slot"],
                    "e3_number_in_p2": ""})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(rows_out, args.out_dir / "p3_panel_b_images.csv", LIST_COLUMNS)
    write_manifest(luma, args.out_dir / "p3_panel_b_luma.csv", LIST_COLUMNS)
    digest = {
        "seed": PANEL_B_SEED,
        "rows": len(rows_out),
        "panel_digest": m.panel_digest(rows_out),
        "test_digest": m.panel_digest([r for r in rows_out if r["role"] == "test"]),
        "calibration_digest": m.panel_digest(
            [r for r in rows_out if r["role"] == "calibration"]),
        "luma_rows": len(luma),
        "luma_digest": m.panel_digest(luma),
        "digest_definition": "sha256 of the newline-joined sorted 'image_id<TAB>sha256' lines",
        "by_view_role": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter(
            (r["view"], r["role"]) for r in rows_out).items())},
        "by_slot": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter(
            (r["view"], r["panel_b_slot"]) for r in rows_out).items())},
        "e3_number_in_p2": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter(
            (r["panel_b_slot"], r["e3_number_in_p2"]) for r in rows_out
            if r["view"] == "e3_matched").items())},
        "panel_bytes": sum((args.panel_root / r["relative_path"]).stat().st_size
                           for r in rows_out + luma),
        "licence": "contains research-only (WildRF) and academic-only (Chameleon) imagery: "
                   "PRIVATE Kaggle dataset only (CLAUDE.md section 6c)",
    }
    path = args.out_dir / "p3_panel_b_digest.json"
    path.with_suffix(".tmp").write_text(json.dumps(digest, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".tmp").replace(path)
    log.info("Panel B frozen: %d rows + %d luma; digest %s; %.1f MB", len(rows_out), len(luma),
             digest["panel_digest"][:16], digest["panel_bytes"] / 1e6)
    if args.cleanup_derived is not None:
        import shutil

        target = args.cleanup_derived.resolve()
        if not target.is_relative_to((THESIS_DBP / "outputs").resolve()):
            log.error("refusing to delete %s: not under %s", target, THESIS_DBP / "outputs")
            return 1
        if target.is_dir():
            shutil.rmtree(target)
            log.info("deleted the derived view images under %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
