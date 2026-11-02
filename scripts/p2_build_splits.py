"""P2-B1: draw the train/validation split, once, and prove nothing crosses it.

The split is 80/20, seed 2026, grouped by ``source_image_id`` and stratified by COCO
supercategory (D-046). Its output, ``reports/p2_splits.csv``, is committed: every
threshold, every temperature and every stacker coefficient in P2 is fitted on the
validation side of this file, so re-drawing it later would invalidate all of them
silently.

The group a row follows depends on what the row is:

* a **real** follows its own ``source_image_id`` -- t_ctrl stores two crops per DIV2K
  photograph, and both must land together;
* a **pair row** follows the real it was reconstructed from;
* a **generation** and its ``gen_matched`` variant follow the COCO image whose
  **caption** produced them, never ``source_real_id``, which names the real whose JPEG
  settings were copied and is usually a different picture entirely.

The last one is the subtle case and the reason :func:`dbp.data.splits.caption_source_stem`
exists. Getting it wrong would put a generation in validation while the photograph its
content came from sits in training.

``--instances`` points at COCO's ``instances_train2017.json`` for the supercategories.
Without it the script stops: an unstratified split is a different split, and producing
one under the same file name would be worse than producing none.

Usage:
    python scripts/p2_build_splits.py \\
        --instances /kaggle/input/.../annotations/instances_train2017.json \\
        --pairs-manifest reports/p1_pairs_manifest.csv
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

from dbp.data.manifest import file_sha256, read_manifest, write_manifest  # noqa: E402
from dbp.data.splits import (  # noqa: E402
    NO_CATEGORY,
    SPLIT_SEED,
    TRAIN_FRACTION,
    SplitError,
    assign_group_splits,
    caption_source_stem,
    coco_stem,
    corpus_supercategory_frequency,
    stratum_label,
    summarise_split,
    verify_no_content_crosses,
)
from dbp.utils.logging import setup_logging  # noqa: E402

SPLIT_COLUMNS = ("row_id", "set_id", "kind", "content_id", "group_id", "stratum", "split")


def supercategories_by_image(instances_path: Path, log: Any) -> dict[int, set[str]]:
    """Map COCO image id -> its annotated supercategories.

    Deliberately identical in behaviour to the helper ``p1_rescan_greyscale.py`` uses,
    so the strata here and the per-supercategory exclusion effect recorded in P1 are
    the same partition of the corpus.
    """
    log.info("reading %s (a few hundred MB)", instances_path)
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


def main() -> int:
    """Draw the split, verify it, write it. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", required=True, type=Path)
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument(
        "--pairs-manifest", type=Path, default=None,
        help="Combined pair manifest. Omit to derive pair rows from the reals instead.",
    )
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--out", type=Path, default=THESIS_DBP / "reports" / "p2_splits.csv")
    args = parser.parse_args()
    log = setup_logging("p2_build_splits")

    if not args.instances.is_file():
        log.error(
            "instances_train2017.json not found at %s. The split is stratified by "
            "supercategory (D-046); an unstratified one is a different split and will "
            "not be written under this name.",
            args.instances,
        )
        return 2

    t_real = read_manifest(args.manifests / "p1_reals_manifest_t_real.csv")
    t_ctrl = read_manifest(args.manifests / "p1_reals_manifest_t_ctrl.csv")
    gen_matched = read_manifest(args.manifests / "p1_gen_matched_manifest.csv")
    log.info(
        "manifests: t_real %d, t_ctrl %d, gen_matched %d",
        len(t_real), len(t_ctrl), len(gen_matched),
    )

    by_image = supercategories_by_image(args.instances, log)

    # ---- content id and stratum per group -----------------------------------
    # Content is the COCO stem for t_real and the generations, and the DIV2K source id
    # for t_ctrl, whose photographs are not COCO and carry no supercategory.
    group_content: dict[str, str] = {}
    group_categories: dict[str, set[str]] = {}
    for row in t_real:
        group = str(row["source_image_id"])
        stem = coco_stem(row["source_relative_path"])
        group_content[group] = stem
        numeric = int(stem) if stem.isdigit() else None
        group_categories[group] = by_image.get(numeric, set()) if numeric else set()
    for row in t_ctrl:
        group = str(row["source_image_id"])
        group_content[group] = f"div2k_{coco_stem(row['source_relative_path'])}"
        group_categories[group] = set()

    frequency = corpus_supercategory_frequency(
        {g: c for g, c in group_categories.items() if c}
    )
    strata = {g: stratum_label(c, frequency) for g, c in group_categories.items()}
    log.info("supercategory frequency: %s", dict(sorted(frequency.items())))

    # Generations whose caption has no T-real partner form groups of their own. They
    # must join the draw BEFORE it happens: assigning them afterwards, one at a time,
    # would send every one of them to train, because a stratum of one has no val side.
    group_of_stem = {stem: group for group, stem in group_content.items()}
    orphans: list[str] = []
    for row in gen_matched:
        stem = caption_source_stem(str(row["gen_id"]))
        if stem in group_of_stem:
            continue
        orphans.append(stem)
        group_content[stem] = stem
        categories = by_image.get(int(stem), set()) if stem.isdigit() else set()
        strata[stem] = stratum_label(categories, frequency)
        group_of_stem[stem] = stem
    if orphans:
        log.info(
            "%d generation(s) have no T-real partner for their caption; they join the "
            "draw as their own groups, stratified like every other group",
            len(orphans),
        )

    log.info("strata: %s", dict(sorted(Counter(strata.values()).items())))
    assignment = assign_group_splits(strata, args.train_fraction, args.seed)

    # ---- every row, tagged with its group's side ----------------------------
    records: list[dict[str, Any]] = []

    def record(row_id: str, set_id: str, kind: str, group: str) -> None:
        records.append(
            {
                "row_id": row_id,
                "set_id": set_id,
                "kind": kind,
                "content_id": group_content[group],
                "group_id": group,
                "stratum": strata[group],
                "split": assignment[group],
            }
        )

    for row in t_real:
        record(str(row["image_id"]), "t_real", "real", str(row["source_image_id"]))
    for row in t_ctrl:
        record(str(row["image_id"]), "t_ctrl", "real", str(row["source_image_id"]))

    # Generations follow the caption: the COCO image whose text produced them, never
    # source_real_id, which names the real whose JPEG settings were copied (D-046).
    for row in gen_matched:
        record(
            str(row["gen_id"]),
            "gen_matched",
            "fake",
            group_of_stem[caption_source_stem(str(row["gen_id"]))],
        )

    if args.pairs_manifest and not args.pairs_manifest.is_file():
        # Passing a path and having it ignored is how a 6,588-row split gets written
        # twice while the log says something reassuring. The flag is a statement that
        # the file should be there.
        log.error(
            "--pairs-manifest %s does not exist. Without it the split covers only the "
            "reals and gen_matched (6,588 rows) and every pair row is silently dropped "
            "from the training lists. Stage p1_pairs_manifest.csv and re-run; omit the "
            "flag only if you really mean to build a split without pair rows.",
            args.pairs_manifest,
        )
        return 2

    if args.pairs_manifest:
        pairs = read_manifest(args.pairs_manifest)
        real_group = {
            str(row["image_id"]): str(row["source_image_id"]) for row in (*t_real, *t_ctrl)
        }
        for row in pairs:
            group = real_group.get(str(row["image_id"]))
            if group is None:
                raise SplitError(
                    f"pair row {row['pair_id']} names real {row['image_id']!r}, which "
                    "is in no reals manifest"
                )
            record(str(row["pair_id"]), f"pair_{row['member']}", "fake", group)
        log.info("pair rows tagged: %d", len(pairs))
        if len(pairs) != 49200:
            log.info(
                "  that is not the expected 49,200 -- check the manifest is the "
                "combined one, not a single member's"
            )
    else:
        log.info(
            "no pair manifest given: pair rows inherit their real's split at load time "
            "through source_image_id"
        )

    # ---- the assertion the module exists for --------------------------------
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in records:
        by_set[entry["set_id"]].append(entry)
    leakage = verify_no_content_crosses(by_set)
    log.info(
        "no content crosses the split: %d distinct content ids across %d sets",
        leakage["distinct_content"], len(by_set),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(records, args.out, SPLIT_COLUMNS)

    summary = {
        "decision": "D-046",
        "seed": args.seed,
        "train_fraction": args.train_fraction,
        "grouped_by": "source_image_id",
        "stratified_by": "COCO supercategory, rarest present",
        "generations_follow": "the caption's COCO image, not source_real_id",
        "rows": len(records),
        "by_set": {
            set_id: dict(Counter(e["split"] for e in entries))
            for set_id, entries in sorted(by_set.items())
        },
        "split": summarise_split(strata, assignment),
        "orphan_generations": len(orphans),
        "no_category_stratum": sum(1 for s in strata.values() if s == NO_CATEGORY),
        "leakage": leakage,
        "manifest": args.out.name,
        "manifest_sha256": file_sha256(args.out),
    }
    summary_path = args.out.with_name("p2_splits_summary.json")
    tmp = summary_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.replace(summary_path)

    totals = summary["split"]["totals"]
    log.info("groups: %d train / %d val", totals["train"], totals["val"])
    log.info("wrote %s (%d rows) and %s", args.out, len(records), summary_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
