"""P3 Phase B step 3, route (ii), KAGGLE stage: dedup the local candidates and accept 4,000.

The local stage (``p3_oi_local_fetch.py``, on the user's PC because Flickr refuses Kaggle's
IPs) walked the seed-2026 order and produced candidates: rows that passed download,
``OriginalMD5``, the ≥ 384 rule and the greyscale rule, each with a native-resolution PNG
crop and the sha256 + pHash of the original and of the crop. Its manifest is committed;
the crops are attached as a private dataset.

Here, on Kaggle, where the references live:

1. every candidate crop is re-hashed and must equal the committed manifest's sha256
   (a mismatch stops the run: the upload is not what was walked);
2. candidates are taken **in walk order (position)** and checked -- original **and**
   crop, sha256 and pHash ≤ 6 -- against Panel B, the P2 panel, the training corpus
   (T-real, T-ctrl, the SD1.5 generations, ``gen_matched``, the bank pairs), all of COCO,
   every E3/E4/E5 real, and the Open Images rows already accepted; each rejection names
   the reference set it matched;
3. the first ``--target`` survivors are accepted, copied to ``<out-root>/reals/t_oi/``,
   and split 3,600 / 400 with the seeded split key.

A shortfall exits non-zero and says where the local walk must resume. Outputs
(``--reports``): ``p3_open_images_manifest.csv``, ``p3_open_images_attribution.csv``,
``p3_oi_accept_log.csv`` and ``p3_oi_accept_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.open_images import assign_split  # noqa: E402
from dbp.utils.disk import df_report  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p1_build_reals import MANIFEST_COLUMNS  # noqa: E402
from p3_fetch_open_images import (  # noqa: E402
    ATTRIBUTION_COLUMNS,
    EXTRA_COLUMNS,
    SET_ID,
    build_references,
    write_csv,
)

DEFAULT_CANDIDATES = THESIS_DBP / "reports" / "p3_open_images" / "p3_oi_candidates_manifest.csv"


def file_sha256(path: Path) -> str:
    """sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:  # noqa: PLR0915 - one linear acceptance pass
    """Dedup and accept. Exit 0 only if the target was met."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidates-manifest", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--candidates-root", required=True, type=Path,
                        help="The attached dataset holding reals/t_oi/*.png.")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--holdout", type=int, default=400)
    parser.add_argument("--coco-index", required=True, type=Path)
    parser.add_argument("--eval-manifest", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--panel-b-selection", type=Path,
                        default=THESIS_DBP / "reports/kaggle_20261011/p3_panel_b_selection.csv")
    parser.add_argument("--p2-panel-sets", type=Path,
                        default=THESIS_DBP / "reports" / "p2_panel_sets")
    parser.add_argument("--t-real", type=Path,
                        default=THESIS_DBP / "reports" / "p1_reals_manifest_t_real.csv")
    parser.add_argument("--t-ctrl", type=Path,
                        default=THESIS_DBP / "reports" / "p1_reals_manifest_t_ctrl.csv")
    parser.add_argument("--gen-matched", type=Path,
                        default=THESIS_DBP / "reports" / "p1_gen_matched_manifest.csv")
    parser.add_argument("--gen-manifest", required=True, type=Path)
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    args = parser.parse_args()
    log = setup_logging("p3_oi_dedup_accept")
    started = time.perf_counter()
    disk = [df_report("start", [args.out_root, args.reports], log)]

    with args.candidates_manifest.open(encoding="utf-8", newline="") as handle:
        candidates = sorted(csv.DictReader(handle), key=lambda r: int(r["position"]))
    log.info("%d candidate(s) from the local walk (positions %s-%s)", len(candidates),
             candidates[0]["position"] if candidates else "-",
             candidates[-1]["position"] if candidates else "-")
    bad = []
    for row in candidates:
        path = args.candidates_root / row["relative_path"]
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            bad.append(row["relative_path"])
    if bad:
        log.error("ACCEPT FAILED: %d candidate crop(s) missing or not the walked bytes "
                  "(first %s); nothing accepted", len(bad), bad[:3])
        return 1
    log.info("all %d candidate crops match the committed manifest's sha256", len(candidates))

    index, ref_counts = build_references(args, log)
    accepted: list[dict[str, str]] = []
    decisions: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in candidates:
        record = {"position": row["position"], "oi_image_id": row["oi_image_id"],
                  "host_group": row["host_group"]}
        if len(accepted) >= args.target:
            decisions.append({**record, "outcome": "not needed (target reached)"})
            continue
        match = index.match(row["original_sha256"], row["source_phash"])
        reason = f"duplicate of a reference ({match})" if match else ""
        if not reason:
            match = index.match(row["sha256"], row["phash"])
            reason = f"crop duplicate of a reference ({match})" if match else ""
        groups[row["host_group"]]["candidates_decided"] += 1
        if reason:
            rejections[reason] += 1
            groups[row["host_group"]]["dedup_rejected"] += 1
            decisions.append({**record, "outcome": reason})
            continue
        index.add("Open Images (accepted)", row["original_sha256"], row["source_phash"])
        index.add("Open Images (accepted crop)", row["sha256"], row["phash"])
        accepted.append(row)
        groups[row["host_group"]]["accepted"] += 1
        decisions.append({**record, "outcome": "accepted"})

    split = assign_split([r["oi_image_id"] for r in accepted],
                         min(args.holdout, len(accepted)))
    crop_dir = args.out_root / "reals" / SET_ID
    crop_dir.mkdir(parents=True, exist_ok=True)
    manifest, attribution = [], []
    for row in accepted:
        shutil.copyfile(args.candidates_root / row["relative_path"],
                        args.out_root / row["relative_path"])
        manifest.append({**row, "split": split[row["oi_image_id"]]})
        attribution.append({"image_id": row["oi_image_id"], "author": row["author"],
                            "original_url": row["original_url"], "licence": row["licence"]})
    attribution.sort(key=lambda r: r["image_id"])
    columns = (*MANIFEST_COLUMNS, *EXTRA_COLUMNS, "position", "host", "host_group",
               "original_sha256", "original_width", "original_height", "decoder_python",
               "decoder_pillow", "decoder_libjpeg")
    write_csv(args.reports / "p3_open_images_manifest.csv", columns, manifest)
    write_csv(args.reports / "p3_open_images_attribution.csv", ATTRIBUTION_COLUMNS, attribution)
    write_csv(args.reports / "p3_oi_accept_log.csv",
              ("position", "oi_image_id", "host_group", "outcome"), decisions)
    disk.append(df_report("end", [args.out_root, args.reports], log))
    last = int(accepted[-1]["position"]) if accepted else 0
    shortfall = max(0, args.target - len(accepted))
    summary = {
        "stage": "p3_oi_dedup_accept", "route": "(ii): local download, Kaggle dedup",
        "candidates": len(candidates), "accepted": len(accepted), "shortfall": shortfall,
        "last_accepted_position": last,
        "last_candidate_position": int(candidates[-1]["position"]) if candidates else 0,
        "split": dict(Counter(split.values())), "rejections": dict(rejections.most_common()),
        "by_host_group": {k: dict(v) for k, v in sorted(groups.items())},
        "references": ref_counts,
        "crop_bytes": sum((args.out_root / m["relative_path"]).stat().st_size
                          for m in manifest),
        "disk": disk, "seconds": round(time.perf_counter() - started, 1),
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "p3_oi_accept_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for reason, n in rejections.most_common():
        log.info("  dedup rejected %5d  %s", n, reason)
    log.info("accepted %d of %d candidate(s) %s", len(accepted), len(candidates),
             summary["split"])
    if shortfall:
        log.error("ACCEPT SHORT by %d: rerun the LOCAL walk with --target-candidates %d; it "
                  "resumes after position %d", shortfall, len(candidates) + 2 * shortfall,
                  summary["last_candidate_position"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
