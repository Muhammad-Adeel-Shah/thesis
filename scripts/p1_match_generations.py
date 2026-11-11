"""P1-B6b: give the SD1.5 generations the compression history the reals have.

**The problem this fixes (issue I-023).** T-real is COCO, which is entirely JPEG. The B6
generations are written as PNG and have never been compressed. Train a detector on that
pairing and the easiest separating feature is not a generator artefact at all -- it is
the presence or absence of an 8x8 quantisation lattice. That is the "Fake or JPEG?"
confound [R7], sitting in the *training* data rather than in an evaluation set, where it
is worse: a model can learn it there and then appear to generalise.

``CLAUDE.md`` §6b fixes this on the evaluation side. This script is the same rule applied
to the training side, and it uses the same code path: ``dbp.data.jpeg.sample_matched_pair``
with the same ``blake2b(salt + ':' + image_id)`` seeding.

**What it does.** For each generation, draw one **observed** ``(quality, chroma
subsampling)`` pair from the real images it is matched against -- jointly, from a single
named real, so no combination is produced that occurs nowhere in the reals -- and write a
second variant of the generation encoded with exactly those settings.

**What it does not do.**

- It does not touch the PNG variant. Both variants are kept: P2 trains on ``gen_matched``
  by default and reports raw-vs-matched as an ablation, so the effect of this step is
  measurable rather than assumed.
- It does not resize (D-001). The JPEG is written at the PNG's own pixel dimensions.
- It does not re-encode the reals. Replacing a genuine compression history with a
  synthetic one is the same error in the other direction.

**Pairing.** Both corpora come from COCO, so a generation prompted from caption
``coco_<image_id>`` can often be matched to the T-real row built from that same COCO
image. Where that exact partner is in the reals manifest it is used, and
``pairing`` records ``exact``. Where it is not -- T-real and the generations select
different slices of COCO -- a real is drawn from the same pool by the same seed and
``pairing`` records ``sampled``. Either way ``source_real_id`` names one real image, so
every row can be traced.

Usage:
    python scripts/p1_match_generations.py \\
        --gen-manifest reports/p1_gen_manifest.csv \\
        --reals-manifest reports/p1_reals_manifest_t_real.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.jpeg import (  # noqa: E402
    MAPPABLE_SUBSAMPLING,
    NOT_JPEG,
    encode_jpeg_bytes,
    jpeg_metadata,
    sample_matched_pair,
    seeded_rng,
    subsampling_code,
)
from dbp.data.manifest import file_sha256, read_manifest, write_manifest  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

#: Exactly the columns the P1 review specified, plus the pairing provenance.
MANIFEST_COLUMNS = (
    "gen_id",
    "source_real_id",
    "pairing",
    "matched_quality",
    "subsampling",
    "seed",
    "png_sha256",
    "jpg_sha256",
    "png_path",
    "jpg_path",
    "width",
    "height",
)

#: Salt for the seeded draw. Fixed, so a rebuild reproduces the set byte for byte, and
#: distinct from any evaluation set's salt so the same id never draws the same pair.
SALT = "gen_matched"

#: `coco_000000123456` -> `000000123456`, used to find a generation's exact partner.
_COCO_ID = re.compile(r"(\d{6,})")


def coco_key(text: str) -> str | None:
    """The COCO numeric id inside an id or a path, or None."""
    match = _COCO_ID.search(text)
    return match.group(1).lstrip("0") or "0" if match else None


def build_reference(
    reals: list[dict[str, str]], log: Any
) -> tuple[list[tuple[int, str]], list[dict[str, str]], dict[str, Any]]:
    """The observed (quality, subsampling) pairs of the eligible real JPEGs.

    Repeats are kept: the reference is the empirical distribution, so a setting that
    occurs twice as often in the reals is drawn twice as often.

    Three kinds of row are refused, each counted and logged rather than dropped
    silently:

    * not a JPEG, or missing a quality -- there is nothing to copy;
    * ``excluded_reason`` set, so the corpus itself has ruled the row out (D-039);
    * a subsampling label outside :data:`MAPPABLE_SUBSAMPLING`. A greyscale real cannot
      say how a colour image should be chroma-subsampled, and inventing ``4:2:0`` would
      fabricate a setting no real image showed.

    Args:
        reals: Rows of a reals manifest.
        log: Logger.

    Returns:
        ``(reference, usable_rows, exclusions)``, where ``reference[i]`` is the pair
        observed in ``usable_rows[i]``, so a sampled index identifies a named real.

    Raises:
        ValueError: If no row survives. A matched view built on nothing would be a
            fabricated distribution.
    """
    usable: list[dict[str, str]] = []
    exclusions = {"not_jpeg_or_no_quality": 0, "excluded_reason": 0, "unmappable_subsampling": 0}
    excluded_rows: list[dict[str, str]] = []
    for row in reals:
        if (
            str(row.get("is_jpeg", "")).strip().lower() not in ("true", "1", "yes")
            or row.get("jpeg_quality") in (None, "")
            or row.get("subsampling") in (None, "", NOT_JPEG)
        ):
            exclusions["not_jpeg_or_no_quality"] += 1
            continue
        if str(row.get("excluded_reason", "")).strip():
            exclusions["excluded_reason"] += 1
            excluded_rows.append(row)
            continue
        if str(row["subsampling"]) not in MAPPABLE_SUBSAMPLING:
            exclusions["unmappable_subsampling"] += 1
            excluded_rows.append(row)
            continue
        usable.append(row)

    if not usable:
        raise ValueError(
            "the reals manifest contains no eligible JPEG rows with a usable quality "
            "and subsampling, so there is no distribution to match. Report the "
            f"confound instead of inventing one. Exclusions: {exclusions}"
        )
    reference = [(int(row["jpeg_quality"]), str(row["subsampling"])) for row in usable]
    qualities = [quality for quality, _ in reference]
    log.info(
        "reference: %d real JPEGs, quality %d..%d, subsampling %s",
        len(reference),
        min(qualities),
        max(qualities),
        sorted({subsampling for _, subsampling in reference}),
    )
    for reason, count in exclusions.items():
        if count:
            log.info("reference excluded %d row(s): %s", count, reason)
    return reference, usable, exclusions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="P1-B6b: JPEG-matched variant of the generation set (issue I-023)."
    )
    parser.add_argument(
        "--gen-manifest", type=Path, default=THESIS_DBP / "reports" / "p1_gen_manifest.csv"
    )
    parser.add_argument(
        "--reals-manifest",
        type=Path,
        default=THESIS_DBP / "reports" / "p1_reals_manifest_t_real.csv",
        help="The reals the generations are paired with. T-real, not T-ctrl: T-ctrl is "
        "lossless PNG and has no JPEG distribution to match.",
    )
    parser.add_argument(
        "--gen-root",
        type=Path,
        default=THESIS_DBP / "outputs" / "p1" / "generated",
        help="Root the generation manifest's relative_path values resolve against.",
    )
    parser.add_argument(
        "--out-root", type=Path, default=THESIS_DBP / "outputs" / "p1" / "gen_matched"
    )
    parser.add_argument("--set-id", default="gen_matched")
    args = parser.parse_args()

    log = setup_logging("p1_match_generations", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    for path in (args.gen_manifest, args.reals_manifest):
        if not path.is_file():
            log.error("manifest not found: %s", path)
            return 2

    generations = read_manifest(args.gen_manifest)
    reals = read_manifest(args.reals_manifest)
    log.info("%d generations, %d reals", len(generations), len(reals))
    if not generations:
        log.error("%s has no rows", args.gen_manifest)
        return 2

    try:
        reference, usable_reals, exclusions = build_reference(reals, log)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    # Exact partners, where a generation and a real came from the same COCO image.
    # Built from the ELIGIBLE reals only, so an excluded real is never a partner.
    by_coco: dict[str, dict[str, str]] = {}
    for index, row in enumerate(usable_reals):
        key = coco_key(row.get("source_relative_path", "") or row.get("image_id", ""))
        if key is not None:
            by_coco.setdefault(key, {"row": row, "index": index})  # type: ignore[dict-item]

    # A generation whose own COCO image is in the corpus but excluded. It keeps its
    # matched variant and draws a colour partner by seed instead of being dropped: the
    # generation is a valid fake either way, and losing it would shrink the training
    # set for a reason that has nothing to do with the generator (D-039).
    excluded_keys: set[str] = set()
    for row in reals:
        if str(row.get("excluded_reason", "")).strip() or str(
            row.get("subsampling", "")
        ) not in MAPPABLE_SUBSAMPLING:
            key = coco_key(row.get("source_relative_path", "") or row.get("image_id", ""))
            if key is not None and key not in by_coco:
                excluded_keys.add(key)

    out_dir = args.out_root / args.set_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    exact = 0

    for row in generations:
        gen_id = row["image_id"]
        png_path = args.gen_root.parent / row["relative_path"]
        if not png_path.is_file():
            png_path = args.gen_root / row["set_id"] / f"{gen_id}.png"
        if not png_path.is_file():
            failures.append({"gen_id": gen_id, "error": f"source PNG not found: {png_path}"})
            log.error("source PNG not found for %s", gen_id)
            continue

        key = coco_key(gen_id)
        partner = by_coco.get(key) if key is not None else None
        if partner is not None:
            real_row = partner["row"]  # type: ignore[index]
            quality = int(real_row["jpeg_quality"])
            subsampling = str(real_row["subsampling"])
            pairing = "exact"
            exact += 1
        else:
            index, quality, subsampling = sample_matched_pair(gen_id, reference, salt=SALT)
            real_row = usable_reals[index]
            pairing = (
                "sampled_after_exclusion"
                if key is not None and key in excluded_keys
                else "sampled"
            )

        try:
            image = load_rgb_uint8(png_path)
            payload = encode_jpeg_bytes(image, quality, subsampling_code(subsampling))
        except Exception as exc:
            failures.append({"gen_id": gen_id, "error": f"{type(exc).__name__}: {exc}"[:200]})
            log.error("could not re-encode %s: %s", gen_id, exc)
            continue

        destination = out_dir / f"{gen_id}.jpg"
        destination.write_bytes(payload)

        written = jpeg_metadata(destination)
        if written["jpeg_quality"] != quality:
            log.warning(
                "%s: asked for q=%d, the file reads back as q=%s",
                gen_id,
                quality,
                written["jpeg_quality"],
            )

        rows.append(
            {
                "gen_id": gen_id,
                "source_real_id": real_row["image_id"],
                "pairing": pairing,
                "matched_quality": quality,
                "subsampling": subsampling,
                # The seed the draw used. Recorded even for an exact pairing, where it
                # was not consulted, so a rebuild can be checked without re-deriving it.
                "seed": int(seeded_rng(gen_id, SALT).integers(0, 2**63)),
                "png_sha256": file_sha256(png_path),
                "jpg_sha256": file_sha256(destination),
                "png_path": str(png_path),
                "jpg_path": str(destination),
                "width": image.shape[1],
                "height": image.shape[0],
            }
        )
        if len(rows) % 250 == 0:
            log.info("%d / %d matched", len(rows), len(generations))

    manifest_path = THESIS_DBP / "reports" / "p1_gen_matched_manifest.csv"
    write_manifest(rows, manifest_path, MANIFEST_COLUMNS)
    log.info("wrote %s (%d rows)", manifest_path, len(rows))

    qualities = [row["matched_quality"] for row in rows]
    summary = {
        "set_id": args.set_id,
        "issue": "I-023 (b): training-side compression confound",
        "gen_manifest": args.gen_manifest.name,
        "reals_manifest": args.reals_manifest.name,
        "n_generations": len(generations),
        "n_matched": len(rows),
        "pairing_exact": exact,
        "pairing_sampled": len(rows) - exact,
        "pairing_counts": {
            label: sum(1 for row in rows if row["pairing"] == label)
            for label in sorted({row["pairing"] for row in rows})
        },
        "reference_size": len(reference),
        "reference_excluded": exclusions,
        "reference_subsampling_policy": sorted(MAPPABLE_SUBSAMPLING),
        "sampling": "joint (quality, subsampling) pair drawn from one observed real",
        "salt": SALT,
        "seed_rule": "blake2b(salt + ':' + gen_id), 8 bytes",
        "quality_min": min(qualities) if qualities else None,
        "quality_max": max(qualities) if qualities else None,
        "subsampling_counts": {
            label: sum(1 for row in rows if row["subsampling"] == label)
            for label in sorted({row["subsampling"] for row in rows})
        },
        "png_variant_kept": True,
        "resized": False,
        "failures": failures,
        "manifest_sha256": file_sha256(manifest_path),
    }
    summary_path = THESIS_DBP / "reports" / "p1_gen_matched_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", summary_path)
    log.info(
        "matched %d generations (%d exact partners, %d sampled), %d failed",
        len(rows),
        exact,
        len(rows) - exact,
        len(failures),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
