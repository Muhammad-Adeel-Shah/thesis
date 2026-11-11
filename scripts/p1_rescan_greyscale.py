#!/usr/bin/env python
"""Re-scan a built real corpus for greyscale images and mark them excluded.

Why this exists
---------------
The B4 filter that rejects greyscale sources was added after T-real had already been
built (D-039). Eleven of its 2,500 images are greyscale JPEGs, which is how thirteen
generations came to have no JPEG-matched variant: a greyscale JPEG has no chroma
channels, so there is no ``(quality, subsampling)`` pair to copy from it.

Rather than delete files and rebuild 49,200 reconstruction pairs and 2,488 generations
for eleven images, this script marks the affected rows. Downstream manifests filter on
``excluded_reason``:

* the T-real rows themselves;
* their B5 pair rows, by joining on ``image_id``;
* the matcher's reference set, which refuses them independently (defence in depth).

Nothing is deleted. The images stay on disk, the manifest keeps its row, and the
exclusion is a recorded, reversible decision rather than a hole in the data.

Two kinds are counted separately, because they are found in different ways:

* **single-channel source** -- the source JPEG itself was greyscale. Visible in the
  manifest as ``subsampling == unknown(-1)``, with no pixel access needed.
* **rgb_equal** -- a greyscale picture written into a three-channel container. Format
  inspection cannot see this; only the pixels can.

Usage
-----
    python scripts/p1_rescan_greyscale.py                        # t_real, writes
    python scripts/p1_rescan_greyscale.py --dry-run              # report only
    python scripts/p1_rescan_greyscale.py --instances <path>     # + supercategories
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.colour import GREYSCALE_TOLERANCE, classify_colour  # noqa: E402
from dbp.data.jpeg import SUBSAMPLING_LABELS  # noqa: E402
from dbp.data.manifest import file_sha256, read_manifest, write_manifest  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

#: The reason written into the manifest. One value, so a filter is a string compare.
EXCLUDED_REASON = "greyscale"

MAPPABLE_SUBSAMPLING = frozenset(SUBSAMPLING_LABELS.values())


def coco_image_number(relative_path: str) -> int | None:
    """The integer COCO image id in ``train2017/000000000009.jpg``, or None."""
    stem = Path(relative_path).stem
    return int(stem) if stem.isdigit() else None


def supercategories_by_image(instances_path: Path, log: Any) -> dict[int, set[str]]:
    """Map COCO image id -> the supercategories annotated in it.

    Reads ``instances_train2017.json`` and keeps only the two fields needed, so the
    parsed document can be released immediately; the file is several hundred megabytes.
    """
    log.info("reading %s (this takes a minute and a few GB of RAM)", instances_path)
    document = json.loads(instances_path.read_text(encoding="utf-8"))
    supercategory = {
        int(category["id"]): str(category["supercategory"])
        for category in document["categories"]
    }
    mapping: dict[int, set[str]] = defaultdict(set)
    for annotation in document["annotations"]:
        name = supercategory.get(int(annotation["category_id"]))
        if name is not None:
            mapping[int(annotation["image_id"])].add(name)
    del document
    log.info("supercategories for %d annotated images", len(mapping))
    return dict(mapping)


def scan(
    rows: list[dict[str, str]], reals_root: Path, tolerance: int, log: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify every row's stored crop and return the findings.

    Args:
        rows: Manifest rows.
        reals_root: Root the ``relative_path`` values resolve against.
        tolerance: Channel spread still counted as greyscale.
        log: Logger.

    Returns:
        ``(findings, counts)``. ``findings`` has one entry per greyscale row.

    Raises:
        FileNotFoundError: If a manifest row's image is missing. A silent skip would
            understate the count, which is the one thing this script must not do.
    """
    findings: list[dict[str, Any]] = []
    # Seeded, so every report carries the same keys whether or not the kind occurred.
    # An absent key reads as "not measured"; a zero reads as "measured, none found".
    counts = Counter(
        {
            "scanned": 0,
            "greyscale": 0,
            "single_channel_source": 0,
            "rgb_equal": 0,
            "unmappable_but_colour": 0,
        }
    )
    for index, row in enumerate(rows, start=1):
        path = reals_root.parent / row["relative_path"]
        if not path.is_file():
            path = reals_root / row["set_id"] / f"{row['image_id']}.png"
        if not path.is_file():
            raise FileNotFoundError(f"{row['image_id']}: image not found at {path}")
        verdict = classify_colour(load_rgb_uint8(path), tolerance=tolerance)

        # Only a JPEG can have a chroma subsampling to be unreadable. A PNG corpus
        # reports "not-jpeg" for every row, which is the expected value, not a defect:
        # treating it as unreadable marked 1,594 colour DIV2K rows as greyscale (I-037).
        is_jpeg = str(row.get("is_jpeg", "")).strip().lower() in ("true", "1", "yes")
        unmappable = is_jpeg and str(row.get("subsampling", "")) not in MAPPABLE_SUBSAMPLING

        counts["scanned"] += 1
        entry = {
            "image_id": row["image_id"],
            "source_image_id": row.get("source_image_id", ""),
            "source_relative_path": row.get("source_relative_path", ""),
            "channel_spread": verdict["channel_spread"],
            "subsampling": row.get("subsampling", ""),
            "jpeg_quality": row.get("jpeg_quality", ""),
        }
        if verdict["is_greyscale"]:
            # "single-channel source" is a claim about the source file, and only a JPEG
            # whose subsampling is unreadable supports it. Everything else is reported
            # as what was actually measured: three stored channels carrying one image.
            source_kind = "single_channel_source" if unmappable else "rgb_equal"
            counts[source_kind] += 1
            counts["greyscale"] += 1
            findings.append({**entry, "kind": source_kind, "excluded": True})
        elif unmappable:
            # Colour pixels, but a JPEG whose chroma setting cannot be read. It is a
            # perfectly good training image and is NOT excluded; it simply cannot serve
            # as a chroma reference, which the matcher decides for itself.
            counts["unmappable_but_colour"] += 1
            findings.append({**entry, "kind": "unmappable_but_colour", "excluded": False})
        if index % 500 == 0:
            log.info("%d / %d scanned, %d greyscale so far", index, len(rows), counts["greyscale"])
    return findings, dict(counts)


def stratification_effect(
    rows: list[dict[str, str]],
    excluded_ids: set[str],
    by_image: dict[int, set[str]],
) -> dict[str, dict[str, Any]]:
    """Per-supercategory counts before and after the exclusion.

    Rows whose COCO id carries no instance annotation are grouped under
    ``"(unannotated)"`` rather than dropped, so the totals still add up.
    """
    totals: Counter = Counter()
    removed: Counter = Counter()
    for row in rows:
        number = coco_image_number(row.get("source_relative_path", ""))
        names = by_image.get(number, set()) if number is not None else set()
        for name in names or {"(unannotated)"}:
            totals[name] += 1
            if row["image_id"] in excluded_ids:
                removed[name] += 1
    effect: dict[str, dict[str, Any]] = {}
    for name in sorted(totals):
        kept = totals[name] - removed[name]
        effect[name] = {
            "rows_before": totals[name],
            "rows_excluded": removed[name],
            "rows_after": kept,
            "share_excluded_pct": round(100.0 * removed[name] / totals[name], 3),
        }
    return effect


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark greyscale rows in a built corpus.")
    parser.add_argument("--set-id", default="t_real")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Defaults to reports/p1_reals_manifest_<set-id>.csv.",
    )
    parser.add_argument(
        "--reals-root", type=Path, default=THESIS_DBP / "outputs" / "p1" / "reals"
    )
    parser.add_argument("--tolerance", type=int, default=GREYSCALE_TOLERANCE)
    parser.add_argument(
        "--instances",
        type=Path,
        default=None,
        help="COCO instances_train2017.json, for the per-supercategory effect.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report without touching the manifest."
    )
    args = parser.parse_args()

    log = setup_logging("p1_rescan_greyscale", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    manifest = args.manifest or (
        THESIS_DBP / "reports" / f"p1_reals_manifest_{args.set_id}.csv"
    )
    if not manifest.is_file():
        log.error("manifest not found: %s", manifest)
        return 2

    rows = read_manifest(manifest)
    if not rows:
        log.error("%s has no rows", manifest)
        return 2
    log.info("%d rows in %s, tolerance %d", len(rows), manifest.name, args.tolerance)

    findings, counts = scan(rows, args.reals_root, args.tolerance, log)
    # Only greyscale rows are excluded. A colour row with an unreadable chroma setting
    # is reported and left in the corpus (I-037).
    excluded_ids = {entry["image_id"] for entry in findings if entry["excluded"]}

    log.info(
        "scanned %d: %d greyscale (%d single-channel source, %d rgb_equal), "
        "%d colour with unreadable subsampling",
        counts.get("scanned", 0),
        counts.get("greyscale", 0),
        counts.get("single_channel_source", 0),
        counts.get("rgb_equal", 0),
        counts.get("unmappable_but_colour", 0),
    )
    # Only the exclusions are listed one by one. The informational findings can run to
    # thousands of rows on a PNG corpus, and a log nobody reads hides the ones that matter.
    for entry in findings:
        if entry["excluded"]:
            log.info(
                "  %s  kind=%-22s spread=%-3d subsampling=%s  %s",
                entry["image_id"],
                entry["kind"],
                entry["channel_spread"],
                entry["subsampling"],
                entry["source_relative_path"],
            )
    informational = sum(1 for entry in findings if not entry["excluded"])
    if informational:
        log.info(
            "%d colour row(s) have an unreadable chroma setting. They stay in the "
            "corpus; they are only unusable as a chroma reference, which the matcher "
            "decides for itself. Ids are in the JSON summary.",
            informational,
        )

    effect: dict[str, dict[str, Any]] = {}
    if args.instances is not None:
        if not args.instances.is_file():
            log.error("instances file not found: %s", args.instances)
            return 2
        by_image = supercategories_by_image(args.instances, log)
        effect = stratification_effect(rows, excluded_ids, by_image)
        for name, entry in effect.items():
            if entry["rows_excluded"]:
                log.info(
                    "  supercategory %-16s %d -> %d rows (%.2f%% excluded)",
                    name,
                    entry["rows_before"],
                    entry["rows_after"],
                    entry["share_excluded_pct"],
                )

    # Every row keeps its place; only the reason column changes.
    columns = list(rows[0])
    if "excluded_reason" not in columns:
        columns.append("excluded_reason")
    for row in rows:
        row["excluded_reason"] = EXCLUDED_REASON if row["image_id"] in excluded_ids else ""

    summary: dict[str, Any] = {
        "set_id": args.set_id,
        "decision": "D-039: greyscale reals excluded by flag, nothing deleted",
        "manifest": manifest.name,
        "tolerance": args.tolerance,
        "rows_before": len(rows),
        "rows_excluded": len(excluded_ids),
        "rows_after": len(rows) - len(excluded_ids),
        "counts": counts,
        "excluded": [entry for entry in findings if entry["excluded"]],
        "reported_not_excluded": [entry for entry in findings if not entry["excluded"]],
        "distinct_sources_excluded": len(
            {
                entry["source_image_id"]
                for entry in findings
                if entry["excluded"] and entry["source_image_id"]
            }
        ),
        "supercategory_effect": effect or "NOT-RUN (no --instances given)",
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        log.warning("--dry-run: the manifest was NOT modified")
    else:
        write_manifest(rows, manifest, tuple(columns))
        summary["manifest_sha256"] = file_sha256(manifest)
        log.info("rewrote %s with excluded_reason for %d row(s)", manifest, len(excluded_ids))

    summary_path = THESIS_DBP / "reports" / f"p1_greyscale_rescan_{args.set_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", summary_path)
    log.info(
        "corpus: %d -> %d usable rows. No backfill: the corpus is now this size (D-039).",
        len(rows),
        len(rows) - len(excluded_ids),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
