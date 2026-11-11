"""P2-B1: draw the shared evaluation panel from the P1 views.

Every stream is scored on **these** images. That is the whole point: a double-fault
rate computed from S1 on one subset and S3 on another is not a double-fault rate. The
caps come from :data:`dbp.eval.panel.PANEL_CAPS`, which was fixed against the approved
GPU budget; changing one is a plan change, not a tuning knob, so the module refuses a
set it has no approved cap for.

Views and their primary status, per D-043 and D-044:

* **E7 DDA-COCO** -- per source VAE, 500 each. FLUX.1 additionally gets the luma-only
  variant (``--luma-only``), because ``4:2:0 -> fake`` scores 100.00% there.
* **E8 EvalGEN** -- 250 per generator. No reals, so no AUC.
* **E3 matched** -- the matched view, never the raw one.
* **E4 WildRF matched** -- carries the +11.39 quality red flag into every table.
* **E5 Chameleon pipeline-matched** -- the PRIMARY Chameleon view.
* **T-ctrl** -- 200 lossless reals and 200 of their reconstructions, the JPEG-reliance
  diagnostic.

The leak check against the training manifests runs again here rather than being
inherited from P1: the panel is a new artefact, and a set that was clean as a whole is
not automatically clean as a subset of itself once the split exists.

Usage:
    python scripts/p2_build_panel.py --set-id e7_dda_coco \\
        --manifest reports/p1_eval_e7_dda_coco_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256, read_manifest, write_manifest  # noqa: E402
from dbp.eval.panel import PANEL_CAPS, PanelError, draw  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: Extra columns the panel adds on top of whatever the view manifest carried.
PANEL_COLUMNS = ("panel_set_id", "panel_stratum")

#: Subsets whose numbers must be printed beside a trivial-rule baseline (D-044).
FLAGGED_SUBSETS = {
    "e7_dda_coco": "FLUX.1: `4:2:0 -> fake` scores 100.00% (lift +49.85); quality +14.35",
    "e4_wildrf_matched":
        "quality separability +11.39 -- up to ~61% reachable without a generative artefact",
    "e3_matched": "Nano-Banana2 residual +6.47 after matching",
}


def main() -> int:
    """Draw one view's panel. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-id", required=True, choices=sorted(PANEL_CAPS))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--id-key", default="image_id", help="Column holding the image id."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=THESIS_DBP / "reports",
    )
    args = parser.parse_args()
    log = setup_logging("p2_build_panel")

    if not args.manifest.is_file():
        log.error("view manifest not found: %s", args.manifest)
        return 2

    rows = read_manifest(args.manifest)
    log.info("%s: %d rows in the view", args.set_id, len(rows))

    try:
        panel, summary = draw(rows, args.set_id, args.id_key)
    except PanelError as exc:
        log.error("%s", exc)
        return 1

    spec = PANEL_CAPS[args.set_id]
    for row in panel:
        row["panel_set_id"] = args.set_id
        row["panel_stratum"] = "|".join(str(row[c]) for c in spec["by"])

    columns = (*read_manifest(args.manifest)[0].keys(), *PANEL_COLUMNS)
    out = args.out_dir / f"p2_panel_{args.set_id}.csv"
    write_manifest(panel, out, columns)

    labels = Counter(str(row.get("label", "unknown")) for row in panel)
    log.info("panel: %d rows  %s", len(panel), dict(sorted(labels.items())))
    for stratum, counts in sorted(summary["by_stratum"].items()):
        marker = "" if counts["taken"] >= 50 else "   <== thin slice"
        log.info(
            "  %-28s %5d of %6d%s", stratum, counts["taken"], counts["available"], marker
        )

    summary["by_label"] = dict(sorted(labels.items()))
    summary["source_manifest"] = args.manifest.name
    summary["manifest"] = out.name
    summary["manifest_sha256"] = file_sha256(out)
    if args.set_id in FLAGGED_SUBSETS:
        summary["reporting_flag"] = FLAGGED_SUBSETS[args.set_id]
        log.info("D-044 flag carried: %s", summary["reporting_flag"])
    thin = {s: c["taken"] for s, c in summary["by_stratum"].items() if c["taken"] < 50}
    if thin:
        summary["thin_strata"] = thin
        log.info(
            "%d stratum/strata under 50 rows: %s -- present for completeness, not for "
            "a per-slice claim", len(thin), thin,
        )

    summary_path = args.out_dir / f"p2_panel_{args.set_id}_summary.json"
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_path)
    log.info("wrote %s and %s", out.name, summary_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
