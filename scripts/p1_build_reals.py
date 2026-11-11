"""P1-B4: build the real-image corpora, T-real (COCO) and T-ctrl (DIV2K).

These are the images every training pair is made from, so three things are settled here
rather than later.

**No resizing, ever (D-001).** An image is centre-cropped to the working size or it is
not used. An image smaller than the working size in either dimension is skipped and the
skip is recorded with its reason; upscaling it would fabricate high-frequency content,
which is precisely the signal the method reads.

**The source's compression history is recorded before we touch it.** COCO is JPEG;
DIV2K is PNG. That difference is a confound waiting to happen if the two are ever mixed
into one training set, so the manifest carries the source format, the estimated JPEG
quality and the chroma subsampling for every image, and the summary prints the table.

**Deduplication happens inside the corpus and is reported, not silently applied.**
Near-duplicate reals would give one picture several votes in training and would leak
across any split. They are found by perceptual hash, dropped, and listed.

**Several crops of one photograph are not several images.** ``--crops-per-image`` takes
more than one **non-overlapping** crop from a source, which is how T-ctrl reaches 1 600
rows from DIV2K's 800 photographs. Every row carries ``source_image_id``, and **any
split must group by that column**: two crops of one photograph straddling a train/test
boundary is leakage, however non-overlapping the pixels are. T-ctrl is diagnostic-only
for the same reason.

The crops are written as PNG. The source files are never modified: everything here
reads from a read-only mount and writes only under ``outputs/p1``.

Usage:
    python scripts/p1_build_reals.py --set t_real --root /kaggle/input/coco-2017-dataset
        --limit 2500
    python scripts/p1_build_reals.py --set t_ctrl --root /tmp/div2k/DIV2K_train_HR
        --limit 1600 --crops-per-image 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.colour import GREYSCALE_TOLERANCE, classify_colour  # noqa: E402
from dbp.data.jpeg import jpeg_metadata  # noqa: E402
from dbp.data.manifest import (  # noqa: E402
    BASE_COLUMNS,
    compression_table,
    file_sha256,
    stable_image_id,
    write_manifest,
)
from dbp.data.phash import find_duplicate_groups, phash_from_array, phash_hex  # noqa: E402
from dbp.utils.io import load_rgb_uint8, save_png_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402
from p0_verify_pins import PinVerificationError, require_pins  # noqa: E402

#: Working crop for every P1 pair. A multiple of 8, 32 and 4, so one crop serves every
#: member's crop multiple without a second, member-specific crop.
DEFAULT_CROP = 384

#: Extensions considered. Deliberately narrow: an unexpected format in a corpus is
#: something to look at, not something to silently include.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

MANIFEST_COLUMNS = (
    *BASE_COLUMNS,
    "source_image_id",
    "crop_index",
    "crop_top",
    "crop_left",
    "source_relative_path",
    "source_width",
    "source_height",
    "source_sha256",
    "crop",
    "phash",
    # Empty for every usable row. Set when a row is kept on disk but must not be used:
    # downstream manifests filter on it rather than deleting anything (D-039).
    "excluded_reason",
)

#: The sets this script knows how to build, with the label recorded for each.
KNOWN_SETS = {
    "t_real": "real",
    "t_ctrl": "real",
}


def list_images(root: Path, log: Any) -> list[Path]:
    """Every image under ``root``, in a deterministic order.

    Sorted by POSIX relative path, so which images a ``--limit`` selects does not
    depend on the filesystem's directory order. Two machines building the same set
    therefore select the same images.
    """
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    found.sort(key=lambda p: p.relative_to(root).as_posix())
    log.info("found %d candidate images under %s", len(found), root)
    return found


def crop_positions(height: int, width: int, crop: int, count: int) -> list[tuple[int, int]]:
    """Top-left corners for up to ``count`` **non-overlapping** crops.

    The crops are laid out along the image's longer axis and the whole block is centred,
    so they come from the middle of the photograph rather than its edges. If the image
    cannot fit ``count`` of them without overlap, fewer are returned -- overlapping crops
    share pixels and are not distinct samples in any useful sense.

    Args:
        height: Source height.
        width: Source width.
        crop: Square crop side.
        count: How many crops are wanted.

    Returns:
        ``(top, left)`` pairs, at most ``count`` of them, in reading order.
    """
    if count < 1:
        raise ValueError(f"crops-per-image must be >= 1, got {count}")
    if height < crop or width < crop:
        return []

    along_width = width >= height
    span = width if along_width else height
    fits = min(count, span // crop)
    if fits < 1:
        return []

    block = fits * crop
    start = (span - block) // 2
    other = ((height if along_width else width) - crop) // 2

    positions = []
    for index in range(fits):
        offset = start + index * crop
        positions.append((other, offset) if along_width else (offset, other))
    return positions


def build_rows(
    source: Path,
    root: Path,
    set_id: str,
    label: str,
    crop: int,
    out_dir: Path,
    crops_per_image: int = 1,
    skip_greyscale: bool = True,
) -> tuple[list[dict[str, Any]], str | None]:
    """Crop one image one or more times, write the crops and describe them.

    Args:
        source: The image to crop.
        root: The read-only root ``source`` sits under, for the recorded relative path.
        set_id: The corpus this row belongs to.
        label: ``real`` or ``fake``, recorded per row.
        crop: Square crop size. The source is never upscaled to reach it (D-001).
        out_dir: Where the crops are written.
        crops_per_image: Non-overlapping crops to take.
        skip_greyscale: Reject greyscale sources, of either kind (D-039).

    Returns:
        ``(rows, skip_reason)``. ``rows`` is empty exactly when ``skip_reason`` is set;
        the reason is ``"too_small"`` or ``"greyscale:<kind>"``. Nothing is written for
        a skipped source.
    """
    relative = source.relative_to(root).as_posix()
    image = load_rgb_uint8(source)
    height, width = image.shape[:2]
    positions = crop_positions(height, width, crop, crops_per_image)
    if not positions:
        return [], "too_small"

    # Judged on the whole source, not on a crop: a crop of a colour photograph can be
    # locally colourless (a grey wall) without the photograph being greyscale.
    if skip_greyscale:
        verdict = classify_colour(image)
        if verdict["is_greyscale"]:
            return [], f"greyscale:{verdict['kind']}"

    source_image_id = stable_image_id(relative, set_id=set_id)
    source_digest = file_sha256(source)
    metadata = jpeg_metadata(source)

    rows: list[dict[str, Any]] = []
    for index, (top, left) in enumerate(positions):
        cropped = image[top : top + crop, left : left + crop]
        # A single crop keeps the bare source id, so a one-crop set's ids are unchanged.
        image_id = source_image_id if len(positions) == 1 else f"{source_image_id}_c{index}"
        destination = out_dir / f"{image_id}.png"
        save_png_uint8(cropped, destination)
        rows.append(
            {
                "image_id": image_id,
                "set_id": set_id,
                "label": label,
                "relative_path": destination.relative_to(out_dir.parent.parent).as_posix(),
                "width": crop,
                "height": crop,
                "format": metadata["format"],
                "is_jpeg": metadata["is_jpeg"],
                "jpeg_quality": metadata["jpeg_quality"],
                "subsampling": metadata["subsampling"],
                "sha256": file_sha256(destination),
                "source_image_id": source_image_id,
                "crop_index": index,
                "crop_top": top,
                "crop_left": left,
                "source_relative_path": relative,
                "source_width": width,
                "source_height": height,
                "source_sha256": source_digest,
                "crop": crop,
                "phash": phash_hex(phash_from_array(cropped)),
                "excluded_reason": "",
            }
        )
    return rows, None


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B4 real-image corpus builder.")
    parser.add_argument("--set", dest="set_id", required=True, choices=sorted(KNOWN_SETS))
    parser.add_argument("--root", type=Path, required=True, help="Read-only source root.")
    parser.add_argument("--limit", type=int, required=True, help="Images to keep.")
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP)
    parser.add_argument(
        "--crops-per-image",
        type=int,
        default=1,
        help=(
            "Non-overlapping crops per source image. >1 means rows share a "
            "source_image_id, and any later split MUST group by that column."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=THESIS_DBP / "outputs" / "p1" / "reals",
        help="Where the crops are written. Never the source root.",
    )
    parser.add_argument("--skip-pin-check", action="store_true", help="Diagnostics only.")
    args = parser.parse_args()

    log = setup_logging(f"p1_build_reals_{args.set_id}", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    if args.crop % 32 or args.crop % 8 or args.crop % 4:
        log.error("--crop %d is not a multiple of 32, 8 and 4; some members could not use it",
                  args.crop)
        return 2

    if not args.root.is_dir():
        log.error("source root not found: %s", args.root)
        return 2
    if args.out_root.resolve() == args.root.resolve():
        log.error("refusing to write into the source root")
        return 2

    # CLAUDE.md 7a: nothing is written until every pin resolves.
    if not args.skip_pin_check:
        try:
            require_pins(printer=log.info)
        except PinVerificationError as exc:
            log.error("%s", exc)
            return 2

    label = KNOWN_SETS[args.set_id]
    out_dir = args.out_root / args.set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = list_images(args.root, log)
    if not candidates:
        log.error("no images found under %s", args.root)
        return 2

    rows: list[dict[str, Any]] = []
    skipped_too_small: list[dict[str, Any]] = []
    skipped_greyscale: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for source in candidates:
        if len(rows) >= args.limit:
            break
        try:
            produced, skip_reason = build_rows(
                source,
                args.root,
                args.set_id,
                label,
                args.crop,
                out_dir,
                args.crops_per_image,
            )
        except Exception as exc:
            failed.append(
                {
                    "path": source.relative_to(args.root).as_posix(),
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            log.warning("could not read %s: %s", source.name, exc)
            continue
        if skip_reason == "too_small":
            skipped_too_small.append({"path": source.relative_to(args.root).as_posix()})
            continue
        if skip_reason is not None and skip_reason.startswith("greyscale"):
            skipped_greyscale.append(
                {
                    "path": source.relative_to(args.root).as_posix(),
                    "kind": skip_reason.split(":", 1)[1],
                }
            )
            continue
        if not produced:
            skipped_too_small.append({"path": source.relative_to(args.root).as_posix()})
            continue
        # Every crop of a source is kept or none is, so a source never contributes a
        # partial set of crops that a grouped split would then have to reason about.
        rows.extend(produced)
        if len(rows) % 250 < len(produced):
            log.info("%d / %d kept", len(rows), args.limit)

    greyscale_kinds = Counter(entry["kind"] for entry in skipped_greyscale)
    log.info(
        "kept %d, skipped %d as smaller than %dpx, skipped %d as greyscale %s, "
        "%d unreadable",
        len(rows),
        len(skipped_too_small),
        args.crop,
        len(skipped_greyscale),
        dict(greyscale_kinds) or "{}",
        len(failed),
    )
    if len(rows) < args.limit:
        log.warning(
            "only %d images met the size requirement; asked for %d. The corpus is smaller "
            "than planned and every later count must say so.",
            len(rows),
            args.limit,
        )

    hashes = {row["image_id"]: int(row["phash"], 16) for row in rows}
    duplicate_groups = find_duplicate_groups(hashes)
    dropped: set[str] = set()
    for group in duplicate_groups:
        dropped.update(group[1:])  # keep the first, by id, drop the rest
    if dropped:
        log.warning(
            "%d near-duplicate image(s) in %d group(s) dropped; a duplicate real would "
            "get several votes in training and leak across any split",
            len(dropped),
            len(duplicate_groups),
        )
        for group in duplicate_groups:
            log.warning("  duplicates: %s", group)
        # Drop the whole source group: keeping one crop of a duplicated photograph
        # while dropping another would leave a row whose source_image_id no longer
        # identifies a complete, self-consistent group.
        dropped_sources = {
            row["source_image_id"] for row in rows if row["image_id"] in dropped
        }
        for row in rows:
            if row["source_image_id"] in dropped_sources:
                (out_dir / f"{row['image_id']}.png").unlink(missing_ok=True)
        rows = [row for row in rows if row["source_image_id"] not in dropped_sources]

    manifest_path = THESIS_DBP / "reports" / f"p1_reals_manifest_{args.set_id}.csv"
    write_manifest(rows, manifest_path, MANIFEST_COLUMNS)
    log.info("wrote %s (%d rows)", manifest_path, len(rows))

    table = compression_table(rows)
    summary = {
        "set_id": args.set_id,
        "label": label,
        "source_root": str(args.root),
        "crop": args.crop,
        "requested": args.limit,
        "kept": len(rows),
        "crops_per_image": args.crops_per_image,
        "distinct_source_images": len({row["source_image_id"] for row in rows}),
        "split_grouping_column": "source_image_id",
        "skipped_too_small": len(skipped_too_small),
        "skipped_greyscale": len(skipped_greyscale),
        "skipped_greyscale_kinds": dict(greyscale_kinds),
        "greyscale_tolerance": GREYSCALE_TOLERANCE,
        "unreadable": failed,
        "near_duplicate_groups": duplicate_groups,
        "near_duplicates_dropped": sorted(dropped),
        "compression_table": table,
        "manifest": manifest_path.name,
        "manifest_sha256": file_sha256(manifest_path),
    }
    summary_path = THESIS_DBP / "reports" / f"p1_reals_manifest_{args.set_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", summary_path)

    for label_name, entry in table.items():
        log.info(
            "compression | %-6s n=%-5d jpeg=%-5d non-jpeg=%-5d quality %s/%s/%s  %s",
            label_name,
            entry["n"],
            entry["jpeg"],
            entry["non_jpeg"],
            entry["quality_min"],
            entry["quality_median"],
            entry["quality_max"],
            entry["formats"],
        )

    distinct_sources = len({row["source_image_id"] for row in rows})
    if args.crops_per_image > 1:
        log.warning(
            "%d rows come from %d distinct source images (%d crops each). ANY split "
            "must group by source_image_id; two crops of one photograph on opposite "
            "sides of a boundary is leakage.",
            len(rows),
            distinct_sources,
            args.crops_per_image,
        )
    if not rows:
        log.error("no images were kept; the corpus is empty")
        return 1
    total_bytes = sum((out_dir / f"{row['image_id']}.png").stat().st_size for row in rows)
    log.info("on disk: %d files, %.2f GB, mean %.1f KB", len(rows),
             total_bytes / 1024**3, total_bytes / len(rows) / 1024)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
