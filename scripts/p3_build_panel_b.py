"""P3 Phase B: draw Panel B from the rebuilt views (rules in ``dbp.eval.panel_b``).

Inputs: the rebuilt P1 view manifests in ``reports/``, the P2 panel's drawn rows, and
the reference manifests every Panel B row must differ from:

| reference | file | checked by |
|---|---|---|
| P2 panel (12,150 drawn rows) | `p2_panel_<view>.csv` | sha256 + pHash |
| T-real (COCO train2017 crops) | `reports/p1_reals_manifest_t_real.csv` | sha256 + pHash |
| T-ctrl (DIV2K crops) | `reports/p1_reals_manifest_t_ctrl.csv` | sha256 + pHash |
| SD1.5 generations | `p1_gen_manifest.csv` (dbp-p2-s2-inputs) | sha256 + pHash |
| their matched JPEGs (`gen_matched`) | `reports/p1_gen_matched_manifest.csv` | sha256 |
| bank reconstructions | `p1_pairs_manifest.csv` (dbp-p2-s2-inputs) | sha256 only (*) |

(*) The pairs manifest records no pHash; the reconstructions' T-real sources are covered
by pHash above.

Views are drawn in a fixed order (E5, E7, E4, E3, E8), so cross-view near-duplicates are
resolved the same way every time. Output: ``reports/p3_panel_b_selection.csv`` and
``reports/p3_panel_b_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import write_manifest  # noqa: E402
from dbp.eval.panel_b import (  # noqa: E402
    PANEL_B_SEED,
    PANEL_B_VIEWS,
    HashIndex,
    PanelBError,
    draw_view,
)
from dbp.utils.logging import setup_logging  # noqa: E402

DRAW_ORDER = ("e5_chameleon_pipeline", "e7_dda_coco", "e4_wildrf_matched", "e3_matched",
              "e8_evalgen")
EXTRA_COLUMNS = ("panel_b_view", "panel_b_slot", "role", "order_key", "e3_number_in_p2")


def read_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    """Draw Panel B. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--p2-panel-sets", required=True, type=Path)
    parser.add_argument("--gen-manifest", required=True, type=Path)
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument("--slots", type=Path, default=None,
                        help="Dry-run fixture only (needs DBP_DRYRUN=1): JSON of smaller slots.")
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p3_build_panel_b")
    import os

    slots_override: dict[str, Any] = {}
    if args.slots:
        if os.environ.get("DBP_DRYRUN") != "1":
            log.error("--slots changes the frozen Panel B sizes; it is for the dry-run only")
            return 2
        slots_override = json.loads(args.slots.read_text(encoding="utf-8"))
        log.warning("DRY-RUN slots in use: %s", sorted(slots_override))

    references = HashIndex()
    counts: dict[str, int] = {}
    for view in PANEL_B_VIEWS:
        counts[f"p2_panel:{view}"] = references.add_rows(
            "P2 panel", read_rows(args.p2_panel_sets / f"p2_panel_{view}.csv"))
    reports = THESIS_DBP / "reports"
    counts["t_real"] = references.add_rows(
        "T-real", read_rows(reports / "p1_reals_manifest_t_real.csv"))
    counts["t_ctrl"] = references.add_rows(
        "T-ctrl", read_rows(reports / "p1_reals_manifest_t_ctrl.csv"))
    counts["sd15_generations"] = references.add_rows(
        "SD1.5 generations", read_rows(args.gen_manifest))
    counts["gen_matched"] = references.add_rows(
        "gen_matched", read_rows(reports / "p1_gen_matched_manifest.csv"),
        sha_column="jpg_sha256", phash_column=None)
    counts["bank_reconstructions"] = references.add_rows(
        "bank reconstruction", read_rows(args.pairs_manifest), phash_column=None)
    log.info("references: %s (%d sha256, %d pHash)", counts, len(references.sha),
             len(references.phash))

    accepted = HashIndex()
    selection: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    columns: list[str] = []
    try:
        for view in DRAW_ORDER:
            rows = read_rows(args.manifests / PANEL_B_VIEWS[view]["manifest"])
            p2_rows = read_rows(args.p2_panel_sets / f"p2_panel_{view}.csv")
            spec = slots_override.get(view)
            panel, summary = draw_view(view, rows, p2_rows, references, accepted,
                                       slots_spec=[tuple(s) for s in spec] if spec else None)
            summaries[view] = summary
            selection.extend(panel)
            for column in rows[0] if rows else []:
                if column not in columns:
                    columns.append(column)
            log.info("%-24s %5d rows; slots %s; shortfalls %s", view, len(panel),
                     {k: v["filled"] for k, v in summary["slots"].items()},
                     {k: v["shortfall"] for k, v in summary["slots"].items() if v["shortfall"]})
    except PanelBError as exc:
        log.error("PANEL B DRAW FAILED: %s", exc)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(selection, args.out_dir / "p3_panel_b_selection.csv",
                   [*columns, *EXTRA_COLUMNS])
    summary = {"seed": PANEL_B_SEED, "draw_order": list(DRAW_ORDER), "references": counts,
               "dry_run_slots": bool(slots_override), "views": summaries,
               "rows": len(selection),
               "by_role": {role: sum(1 for r in selection if r["role"] == role)
                           for role in ("test", "calibration")}}
    path = args.out_dir / "p3_panel_b_summary.json"
    path.with_suffix(".tmp").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".tmp").replace(path)
    log.info("Panel B: %d rows (%s)", len(selection), summary["by_role"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
