"""P1-B7: index an evaluation set, describe its compression, and de-confound it.

Test-only. Nothing here trains anything, and no evaluation image is ever modified:
the originals stay exactly as distributed, on a read-only mount where possible.

What a run produces for one set (one *view* of one set, see below):

1. **A manifest** with, per image, its label, its true pixel size, its format, whether
   it is a JPEG, the estimated quality and the chroma subsampling, a SHA-256 and a
   perceptual hash.
2. **The label x compression table** that CLAUDE.md 6b requires of every evaluation
   set, printed as markdown for the data card and stored as JSON.
3. **A verdict on whether compression history is confounded with the label**, and if it
   is, a **JPEG-matched view**: the never-compressed fakes are re-encoded to JPEG with
   quality and chroma subsampling drawn per image from that set's *own* real
   distribution, seeded by the image id. Reals are never re-encoded. The matched view
   is a manifest plus the newly encoded files; every image that already had the right
   compression history is referenced in place.
4. **A leakage report** against the training corpora, by perceptual hash. An evaluation
   set that overlaps the training reals is not an evaluation set.
5. Optionally **a seeded, stratified subsample**, for sets too large to run whole.
   ``--subsample N --stratify-by FIELD`` keeps N per stratum, drawn by a seeded shuffle
   of the ids rather than by file order, and records the sampler, the seed and every
   kept id in the summary. "The first N files" is never acceptable: in a set laid out
   by generator it would return one generator.

Measured for E3 (``zr-zhang/MLLM-Generated-Image-Detection-Dataset``, listing read on
2026-09-20): in the ``Raw`` view every one of the 726 reals is a JPEG and 1361 of the
1452 fakes are PNG that have never been compressed. A detector can reach a high score
there by reading the format alone [R7], which is why the matched view -- not the raw
one -- is the primary view for every claim.

Usage:
    python scripts/p1_build_eval.py --set-id e3_raw --root <root>
        --real-glob "images/Raw/*/real/*" --fake-glob "images/Raw/*/*-fake/*"
        --field subset=2 --field source_label=3 --jpeg-match
        --leak-check-against reports/p1_reals_manifest_t_real.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.jpeg import (  # noqa: E402
    MAPPABLE_SUBSAMPLING,
    NOT_JPEG,
    SEPARABILITY_LIFT_THRESHOLD,
    compression_is_confounded_with_label,
    encode_jpeg_bytes,
    jpeg_metadata,
    parameter_separability,
    sample_matched_pair,
    seeded_rng,
    separability_by_stratum,
    subsampling_code,
)
from dbp.data.manifest import (  # noqa: E402
    BASE_COLUMNS,
    compression_table,
    file_sha256,
    read_manifest,
    stable_image_id,
    summarise_by,
    write_manifest,
)
from dbp.data.phash import (  # noqa: E402
    DUPLICATE_THRESHOLD_BITS,
    find_cross_set_matches,
    find_duplicate_groups,
    phash_file,
    phash_from_array,
    phash_hex,
)
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

MANIFEST_COLUMNS = (*BASE_COLUMNS, "source_path", "phash")

MATCHED_COLUMNS = (
    *BASE_COLUMNS,
    "source_path",
    "phash",
    "view_path",
    "reencoded",
    "matched_quality",
    "matched_subsampling",
    "matched_from",
)


def parse_fields(raw: list[str] | None) -> dict[str, int]:
    """Turn ``name=depth`` arguments into a mapping.

    A set's directory layout carries information -- E3's ``Raw/Texture Images/real/1.png``
    encodes both the perceptual regime and the generator -- and which component means
    what differs per set. Making it an argument keeps that knowledge in the runbook,
    where it is visible, rather than in a per-set branch in this file.
    """
    fields: dict[str, int] = {}
    for item in raw or []:
        if "=" not in item:
            raise ValueError(f"--field expects name=depth, got {item!r}")
        name, _, depth = item.partition("=")
        fields[name.strip()] = int(depth)
    return fields


def collect(
    root: Path, patterns: dict[str, list[str]], log: Any
) -> list[tuple[str, Path]]:
    """Match every glob and return ``(label, path)``, deterministically ordered.

    A file matched by more than one label's globs is a specification error and raises:
    silently taking the first match would mislabel part of the set.
    """
    seen: dict[Path, str] = {}
    for label, globs in patterns.items():
        for pattern in globs:
            for path in root.glob(pattern):
                if not path.is_file():
                    continue
                previous = seen.get(path)
                if previous is not None and previous != label:
                    raise ValueError(
                        f"{path.relative_to(root)} matches both the {previous!r} and "
                        f"{label!r} globs; the labels are ambiguous."
                    )
                seen[path] = label
    items = sorted(seen.items(), key=lambda item: item[0].relative_to(root).as_posix())
    for label, count in summarise_by([{"label": label} for _, label in items], "label").items():
        log.info("matched %5d file(s) labelled %s", count, label)
    return [(label, path) for path, label in items]


def describe(
    path: Path, root: Path, set_id: str, label: str, fields: dict[str, int]
) -> dict[str, Any]:
    """One manifest row for one evaluation image. Never modifies the file."""
    relative = path.relative_to(root).as_posix()
    metadata = jpeg_metadata(path)
    row: dict[str, Any] = {
        "image_id": stable_image_id(relative, set_id=set_id),
        "set_id": set_id,
        "label": label,
        "relative_path": relative,
        "width": metadata["width"],
        "height": metadata["height"],
        "format": metadata["format"],
        "is_jpeg": metadata["is_jpeg"],
        "jpeg_quality": metadata["jpeg_quality"],
        "subsampling": metadata["subsampling"],
        "sha256": file_sha256(path),
        "source_path": str(path),
        "phash": phash_hex(phash_file(path)),
    }
    parts = relative.split("/")
    for name, depth in fields.items():
        row[name] = parts[depth] if 0 <= depth < len(parts) else ""
    return row


def stratified_subsample(
    rows: list[dict[str, Any]],
    per_stratum: int,
    stratify_by: str | None,
    salt: str,
    log: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep at most ``per_stratum`` rows from each stratum, deterministically.

    Selection is by a seeded shuffle of each stratum's **image ids**, not of the file
    listing, so the subset does not depend on directory order and is identical on any
    machine. Every kept id is recorded in the summary, so the subset is a citable object
    rather than "whatever that run happened to pick".

    Args:
        rows: The full manifest rows.
        per_stratum: Rows to keep from each stratum.
        stratify_by: Column to stratify on -- normally the generator, so per-generator
            comparisons stay balanced. None puts everything in one stratum.
        salt: Seed salt, normally the set id.
        log: Logger.

    Returns:
        ``(kept_rows, provenance)``. Strata smaller than ``per_stratum`` are kept whole
        and named in the provenance, because silently returning fewer is how a table
        ends up with an unexplained count.
    """
    if per_stratum < 1:
        raise ValueError(f"--subsample must be >= 1, got {per_stratum}")

    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(stratify_by, "")) if stratify_by else "_all"
        strata.setdefault(key, []).append(row)

    kept: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    short: list[str] = []
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda r: r["image_id"])
        rng = seeded_rng(f"{key}", salt)
        order = rng.permutation(len(members))
        chosen = [members[int(i)] for i in order[:per_stratum]]
        kept.extend(chosen)
        detail[key] = {"available": len(members), "kept": len(chosen)}
        if len(members) < per_stratum:
            short.append(key)
            log.warning(
                "stratum %r has only %d rows, fewer than the requested %d; kept whole",
                key,
                len(members),
                per_stratum,
            )
        else:
            log.info("stratum %-28s %5d available -> %5d kept", key, len(members), len(chosen))

    kept.sort(key=lambda r: r["image_id"])
    provenance = {
        "applied": True,
        "per_stratum": per_stratum,
        "stratify_by": stratify_by,
        "salt": salt,
        "seed_rule": "blake2b(salt + ':' + stratum), 8 bytes; permutation of sorted image_ids",
        "sampler": "stratified_subsample in scripts/p1_build_eval.py",
        "n_before": len(rows),
        "n_after": len(kept),
        "strata": detail,
        "strata_smaller_than_requested": short,
        "kept_image_ids": [row["image_id"] for row in kept],
    }
    log.info("subsample: %d rows -> %d rows across %d strata", len(rows), len(kept), len(strata))
    return kept, provenance


def markdown_table(table: dict[str, dict[str, Any]]) -> str:
    """Render the label x compression table for a data card (CLAUDE.md 6b)."""
    lines = [
        "| label | n | JPEG | never compressed | formats | quality min/median/max | subsampling |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, entry in table.items():
        quality = (
            f"{entry['quality_min']}/{entry['quality_median']}/{entry['quality_max']}"
            if entry["quality_min"] is not None
            else "n/a"
        )
        formats = ", ".join(f"{k or 'UNKNOWN'} {v}" for k, v in entry["formats"].items())
        subsampling = ", ".join(f"{k} {v}" for k, v in entry["subsampling"].items())
        lines.append(
            f"| {label} | {entry['n']} | {entry['jpeg']} | {entry['non_jpeg']} | "
            f"{formats} | {quality} | {subsampling} |"
        )
    return "\n".join(lines)


#: Resampling filters offered for the pipeline-matched view. LANCZOS is the default
#: because it is what a careful thumbnailer uses; BICUBIC exists as the sensitivity
#: check, so a result cannot rest on one filter's ringing signature.
RESAMPLE_FILTERS = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC}


def width_rule_accuracy(rows: Sequence[Mapping[str, Any]], width: int) -> dict[str, Any]:
    """Accuracy of the one-line rule ``width == W -> real``.

    Chameleon's real class is uniformly 500 px wide and its fakes are native
    resolution, so this rule scores 100.00% on the set as distributed. It is the
    sharpest statement of that confound and the number the pipeline-matched view has to
    move; the generic per-value separability measure cannot express it, because width
    takes hundreds of values and a per-value rule would overfit them.
    """
    hit = sum(1 for row in rows if int(row["width"]) == width and row["label"] == "real")
    miss = sum(1 for row in rows if int(row["width"]) == width and row["label"] != "real")
    others = [row for row in rows if int(row["width"]) != width]
    correct = hit + sum(1 for row in others if row["label"] != "real")
    total = len(rows)
    reals = sum(1 for row in rows if row["label"] == "real")
    baseline = max(reals, total - reals) / total if total else 0.0
    return {
        "rule": f"width == {width} -> real",
        "accuracy": round(correct / total, 6) if total else None,
        "baseline": round(baseline, 6),
        "lift": round(correct / total - baseline, 6) if total else None,
        "reals_at_width": hit,
        "fakes_at_width": miss,
        "n": total,
    }


def build_pipeline_matched_view(
    rows: list[dict[str, Any]],
    set_id: str,
    out_dir: Path,
    width: int,
    quality: int,
    subsampling: str,
    resample: str,
    log: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Put the fakes through the pipeline the reals went through.

    The JPEG-matched view (§6b) copies a *compression* setting. That is not enough when
    the two classes differ in how they were **published**: Chameleon's reals are all
    exactly 500 px wide because they were thumbnailed, while its fakes are native
    resolution up to 5376x6720. Re-encoding leaves that untouched, and
    ``width == 500 -> real`` still scores 100.00% (D-043).

    So the fakes are downscaled to the same width with the same aspect ratio and then
    encoded at the reals' own setting. The reals are not touched -- as everywhere else,
    a genuine history is never replaced by a synthetic one -- and the originals stay on
    disk; this is a derived view.

    **This resizes images, deliberately.** CLAUDE.md §7's "never resize" governs the
    reconstruction protocol, where a resize would fabricate a forensic trace. Here the
    resize *is* the trace being matched, and it is applied to one class only because
    only that class is missing it.

    Args:
        rows: Manifest rows of the set as distributed.
        set_id: Set id, used as the sampling salt and the output directory name.
        out_dir: Where the derived images are written.
        width: Target width in pixels; rows already at it are left alone.
        quality: JPEG quality for the re-encode.
        subsampling: Chroma subsampling label, e.g. ``4:2:0``.
        resample: ``lanczos`` or ``bicubic``.
        log: Logger.

    Returns:
        ``(view_rows, summary)``.

    Raises:
        ValueError: If ``resample`` is not one of :data:`RESAMPLE_FILTERS`.
    """
    if resample not in RESAMPLE_FILTERS:
        raise ValueError(f"resample must be one of {sorted(RESAMPLE_FILTERS)}, got {resample!r}")

    out_dir.mkdir(parents=True, exist_ok=True)
    view_rows: list[dict[str, Any]] = []
    resized = 0
    encoded = 0

    for row in rows:
        view = dict(row)
        if row["label"] == "real":
            view.update({"view_path": row["source_path"], "pipeline_applied": False})
            view_rows.append(view)
            continue

        image = load_rgb_uint8(row["source_path"])
        height_in, width_in = image.shape[:2]
        if width_in != width:
            target_height = max(1, round(height_in * width / width_in))
            image = np.asarray(
                Image.fromarray(image).resize(
                    (width, target_height), RESAMPLE_FILTERS[resample]
                ),
                dtype=np.uint8,
            )
            resized += 1

        payload = encode_jpeg_bytes(image, quality, subsampling_code(subsampling))
        destination = out_dir / f"{row['image_id']}.jpg"
        destination.write_bytes(payload)
        encoded += 1

        metadata = jpeg_metadata(destination)
        view.update(
            {
                "view_path": str(destination),
                "pipeline_applied": True,
                "width": metadata["width"],
                "height": metadata["height"],
                "format": metadata["format"],
                "is_jpeg": True,
                "jpeg_quality": metadata["jpeg_quality"],
                "subsampling": metadata["subsampling"],
                "sha256": file_sha256(destination),
            }
        )
        view_rows.append(view)

    summary = {
        "view": "pipeline_matched",
        "decision": "D-043",
        "target_width": width,
        "quality": quality,
        "subsampling": subsampling,
        "resample": resample,
        "resized": resized,
        "encoded": encoded,
        "reals_untouched": sum(1 for row in rows if row["label"] == "real"),
        "width_rule_before": width_rule_accuracy(rows, width),
        "width_rule_after": width_rule_accuracy(view_rows, width),
    }
    log.info(
        "pipeline-matched view: %d fake(s) resized to width %d (%s), %d encoded at "
        "q%d %s; reals untouched",
        resized,
        width,
        resample,
        encoded,
        quality,
        subsampling,
    )
    log.info(
        "  trivial rule '%s': %.2f%% -> %.2f%% (baseline %.2f%%)",
        summary["width_rule_before"]["rule"],
        100 * summary["width_rule_before"]["accuracy"],
        100 * summary["width_rule_after"]["accuracy"],
        100 * summary["width_rule_after"]["baseline"],
    )
    return view_rows, summary


def build_matched_view(
    rows: list[dict[str, Any]],
    set_id: str,
    out_dir: Path,
    log: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-encode the never-compressed images of the confounded labels only.

    The distribution is taken from the real images of this same set, because that is
    what the fakes have to become indistinguishable from. Reals are never re-encoded:
    replacing a genuine compression history with a synthetic one is the same error in
    the other direction.
    """
    real_jpegs: list[dict[str, Any]] = []
    unmappable = 0
    for row in rows:
        if (
            row["label"] != "real"
            or not row["is_jpeg"]
            or row["jpeg_quality"] is None
            or row["subsampling"] == NOT_JPEG
        ):
            continue
        # A greyscale JPEG has no chroma channels, so Pillow reports its subsampling as
        # -1 and there is no setting to copy onto a colour image. Drawing one aborted
        # the whole E4 build (I-039) -- the same defect the training-side matcher had.
        # Two of E4's 1,251 reals were enough.
        if str(row["subsampling"]) not in MAPPABLE_SUBSAMPLING:
            unmappable += 1
            continue
        real_jpegs.append(row)

    if not real_jpegs:
        raise ValueError(
            f"{set_id}: no real JPEG has a usable (quality, subsampling) pair, so there "
            f"is no real distribution to match. Report the confound in the data card "
            f"instead of inventing a distribution. "
            f"({unmappable} row(s) had an unreadable chroma setting.)"
        )
    reference = [(int(row["jpeg_quality"]), str(row["subsampling"])) for row in real_jpegs]
    qualities = [quality for quality, _ in reference]
    log.info(
        "matching against %d real JPEGs: quality %d..%d, subsampling %s",
        len(reference),
        min(qualities),
        max(qualities),
        sorted({subsampling for _, subsampling in reference}),
    )
    if unmappable:
        log.info(
            "%d real(s) excluded from the reference: unreadable chroma setting "
            "(greyscale JPEG). They stay in the set; only the reference is affected.",
            unmappable,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    view_rows: list[dict[str, Any]] = []
    reencoded = 0

    for row in rows:
        view = dict(row)
        if row["label"] == "real" or row["is_jpeg"]:
            view["view_path"] = row["source_path"]
            view["reencoded"] = False
            view["matched_quality"] = row["jpeg_quality"]
            view["matched_subsampling"] = row["subsampling"]
            view["matched_from"] = ""
        else:
            index, quality, subsampling = sample_matched_pair(
                row["image_id"], reference, salt=set_id
            )
            # Native resolution, no resize (D-001): the array is written exactly as read.
            image = load_rgb_uint8(row["source_path"])
            payload = encode_jpeg_bytes(image, quality, subsampling_code(subsampling))
            destination = out_dir / f"{row['image_id']}.jpg"
            destination.write_bytes(payload)

            metadata = jpeg_metadata(destination)
            view["view_path"] = str(destination)
            view["reencoded"] = True
            view["matched_quality"] = quality
            view["matched_subsampling"] = subsampling
            view["matched_from"] = real_jpegs[index]["image_id"]
            view["format"] = metadata["format"]
            view["is_jpeg"] = True
            view["jpeg_quality"] = metadata["jpeg_quality"]
            view["subsampling"] = metadata["subsampling"]
            view["sha256"] = file_sha256(destination)
            view["phash"] = phash_hex(phash_from_array(image))
            reencoded += 1

        # One row per source image, in the same order as the raw manifest.
        view_rows.append(view)

    summary = {
        "reencoded": reencoded,
        "reference_excluded_unmappable_subsampling": unmappable,
        "left_in_place": len(rows) - reencoded,
        "reference_size": len(reference),
        "reference_quality_range": [min(qualities), max(qualities)],
        "reference_subsampling": sorted({subsampling for _, subsampling in reference}),
        "sampling": "joint (quality, subsampling) pair drawn from one observed real",
        "salt": set_id,
        "seed_rule": "blake2b(salt + ':' + image_id), 8 bytes",
        "script": Path(__file__).name,
    }
    log.info("JPEG-matched view: %d re-encoded, %d left in place", reencoded, len(rows) - reencoded)
    return view_rows, summary


def leakage_report(
    rows: list[dict[str, Any]], manifests: list[Path], threshold: int, log: Any
) -> dict[str, Any]:
    """Check this set against the training corpora by perceptual hash."""
    evaluation = {row["image_id"]: int(row["phash"], 16) for row in rows}
    findings: dict[str, Any] = {"threshold_bits": threshold, "against": {}}
    for manifest in manifests:
        if not manifest.is_file():
            findings["against"][manifest.name] = {"status": "NOT-RUN", "reason": "manifest missing"}
            log.warning("leakage check skipped: %s not found", manifest)
            continue
        training = {
            row["image_id"]: int(row["phash"], 16)
            for row in read_manifest(manifest)
            if row.get("phash")
        }
        matches = find_cross_set_matches(training, evaluation, threshold)
        findings["against"][manifest.name] = {
            "status": "CLEAN" if not matches else "LEAKAGE",
            "n_training": len(training),
            "n_evaluation": len(evaluation),
            "matches": [
                {"training_id": a, "evaluation_id": b, "distance_bits": d} for a, b, d in matches
            ],
        }
        if matches:
            log.error("LEAKAGE against %s: %d matching image(s)", manifest.name, len(matches))
            for a, b, d in matches[:20]:
                log.error("  %s ~ %s (%d bits)", a, b, d)
        else:
            log.info("no leakage against %s (%d training images)", manifest.name, len(training))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B7 evaluation-set indexer (test-only).")
    parser.add_argument("--set-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--real-glob", action="append", default=None)
    parser.add_argument("--fake-glob", action="append", default=None)
    parser.add_argument("--field", action="append", default=None, help="name=depth, repeatable.")
    parser.add_argument("--jpeg-match", action="store_true", help="Build the matched view.")
    parser.add_argument(
        "--pipeline-match-width",
        type=int,
        default=0,
        help="Build the pipeline-matched view (D-043): downscale the fakes to this "
        "width, then encode at --pipeline-match-quality. 0 disables it.",
    )
    parser.add_argument("--pipeline-match-quality", type=int, default=75)
    parser.add_argument("--pipeline-match-subsampling", default="4:2:0")
    parser.add_argument(
        "--pipeline-match-resample",
        default="lanczos",
        choices=sorted(RESAMPLE_FILTERS),
        help="lanczos is the default; bicubic is the sensitivity check.",
    )
    parser.add_argument(
        "--leak-check-against",
        default="",
        help="Comma-separated training manifests to check against.",
    )
    parser.add_argument("--phash-threshold", type=int, default=DUPLICATE_THRESHOLD_BITS)
    parser.add_argument(
        "--subsample",
        type=int,
        default=0,
        help=(
            "Keep at most this many rows per stratum, chosen by a seeded shuffle of the "
            "image ids. 0 (the default) keeps everything."
        ),
    )
    parser.add_argument(
        "--stratify-by",
        default=None,
        help=(
            "Column to stratify the subsample on -- normally the generator field, so "
            "per-generator comparisons stay balanced."
        ),
    )
    parser.add_argument(
        "--out-root", type=Path, default=THESIS_DBP / "outputs" / "p1" / "eval"
    )
    args = parser.parse_args()

    log = setup_logging(f"p1_build_eval_{args.set_id}", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    if not args.root.is_dir():
        log.error("root not found: %s", args.root)
        return 2
    if not args.real_glob and not args.fake_glob:
        log.error("give at least one --real-glob or --fake-glob")
        return 2

    patterns = {"real": args.real_glob or [], "fake": args.fake_glob or []}
    try:
        fields = parse_fields(args.field)
        items = collect(args.root, patterns, log)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if not items:
        log.error("no files matched under %s", args.root)
        return 2

    columns = (*MANIFEST_COLUMNS, *fields)
    rows = [describe(path, args.root, args.set_id, label, fields) for label, path in items]

    subsample_provenance: dict[str, Any] = {"applied": False}
    if args.subsample:
        if args.stratify_by and args.stratify_by not in fields:
            log.error(
                "--stratify-by %r is not one of the --field columns %s; stratifying on a "
                "column that does not exist would put everything in one stratum",
                args.stratify_by,
                sorted(fields),
            )
            return 2
        try:
            rows, subsample_provenance = stratified_subsample(
                rows, args.subsample, args.stratify_by, args.set_id, log
            )
        except ValueError as exc:
            log.error("%s", exc)
            return 2

    manifest_path = THESIS_DBP / "reports" / f"p1_eval_{args.set_id}_manifest.csv"
    write_manifest(rows, manifest_path, columns)
    log.info("wrote %s (%d rows)", manifest_path, len(rows))

    table = compression_table(rows)
    confounded, reason = compression_is_confounded_with_label(
        {
            label: {"jpeg": entry["jpeg"], "non_jpeg": entry["non_jpeg"]}
            for label, entry in table.items()
        }
    )
    log.info("compression vs label: %s", reason)
    markdown = markdown_table(table)
    for line in markdown.splitlines():
        log.info("%s", line)

    # D-042. The check above asks whether an image was compressed; this asks how, and
    # whether that alone gives the label away. Run per stratum as well as pooled,
    # because a pooled figure hides the case that motivated the rule: in E7 the pooled
    # lift is +0.03 points while the FLUX.1 subset on its own is +49.9.
    separability: dict[str, Any] = {
        "threshold_lift": SEPARABILITY_LIFT_THRESHOLD,
        "pooled": {
            parameter: parameter_separability(rows, parameter)
            for parameter in ("subsampling", "jpeg_quality")
        },
        "by_field": {},
    }
    for field_name in fields:
        separability["by_field"][field_name] = {
            parameter: separability_by_stratum(rows, parameter, field_name)
            for parameter in ("subsampling", "jpeg_quality")
        }

    offenders: list[str] = []
    for parameter, result in separability["pooled"].items():
        if result["separable"]:
            offenders.append(f"pooled/{parameter}")
            log.warning("SEPARABLE: %s", result["reason"])
        elif result["accuracy"] is not None:
            log.info("separability: %s", result["reason"])
    for field_name, per_parameter in separability["by_field"].items():
        for parameter, result in per_parameter.items():
            for stratum in result["separable_strata"]:
                entry = result["by_stratum"][stratum]
                offenders.append(f"{field_name}={stratum}/{parameter}")
                log.warning(
                    "SEPARABLE: %s=%s -- %s. Values: %s",
                    field_name,
                    stratum,
                    entry["reason"],
                    entry["values"],
                )
    separability["separable"] = bool(offenders)
    separability["offenders"] = offenders
    if offenders:
        log.warning(
            "%d encoding-parameter shortcut(s) found. A detector could score well here "
            "without looking at a generative artefact. This cannot be undone by "
            "re-encoding -- chroma subsampling is destructive -- so the affected "
            "subset(s) must carry the caveat wherever they are reported (D-042).",
            len(offenders),
        )
    else:
        log.info("no encoding parameter predicts the label beyond the majority baseline")

    duplicate_groups = find_duplicate_groups(
        {row["image_id"]: int(row["phash"], 16) for row in rows}, args.phash_threshold
    )
    if duplicate_groups:
        log.warning("%d near-duplicate group(s) inside the set (reported, not removed: an "
                    "evaluation set is used as distributed)", len(duplicate_groups))

    summary: dict[str, Any] = {
        "set_id": args.set_id,
        "root": str(args.root),
        "globs": patterns,
        "fields": fields,
        "n": len(rows),
        "subsample": subsample_provenance,
        "by_label": summarise_by(rows, "label"),
        "compression_table": table,
        "compression_table_markdown": markdown,
        "compression_confounded": confounded,
        "compression_verdict": reason,
        "parameter_separability": separability,
        "near_duplicate_groups_within_set": duplicate_groups,
        "manifest": manifest_path.name,
        "manifest_sha256": file_sha256(manifest_path),
        "jpeg_matched_view": None,
    }
    for name in fields:
        summary[f"by_{name}"] = summarise_by(rows, name)

    if args.jpeg_match:
        if not confounded:
            log.info("not confounded, so no matched view is built; recorded in the summary")
        else:
            try:
                view_rows, view_summary = build_matched_view(
                    rows, args.set_id, args.out_root / f"{args.set_id}_jpeg_matched", log
                )
            except ValueError as exc:
                log.error("%s", exc)
                return 2
            view_columns = (*MATCHED_COLUMNS, *fields)
            view_path = (
                THESIS_DBP / "reports" / f"p1_eval_{args.set_id}_jpeg_matched_manifest.csv"
            )
            write_manifest(view_rows, view_path, view_columns)
            view_summary["manifest"] = view_path.name
            view_summary["manifest_sha256"] = file_sha256(view_path)
            view_summary["compression_table"] = compression_table(view_rows)
            view_summary["compression_table_markdown"] = markdown_table(
                view_summary["compression_table"]
            )
            still_confounded, still_reason = compression_is_confounded_with_label(
                {
                    label: {"jpeg": entry["jpeg"], "non_jpeg": entry["non_jpeg"]}
                    for label, entry in view_summary["compression_table"].items()
                }
            )
            view_summary["compression_confounded"] = still_confounded
            view_summary["compression_verdict"] = still_reason

            # The point of the view is that the shortcut is gone, so measure it on the
            # view as well (D-042). Reporting only "not confounded" would repeat the
            # mistake the parameter check exists to correct: E3's matched view is clean
            # pooled and still lifts +6.5 points on one subset.
            view_rows_measured = [
                {
                    **row,
                    "subsampling": row.get("matched_subsampling", row.get("subsampling")),
                    "jpeg_quality": row.get("matched_quality", row.get("jpeg_quality")),
                }
                for row in view_rows
            ]
            view_separability: dict[str, Any] = {
                "threshold_lift": SEPARABILITY_LIFT_THRESHOLD,
                "pooled": {
                    parameter: parameter_separability(view_rows_measured, parameter)
                    for parameter in ("subsampling", "jpeg_quality")
                },
                "by_field": {
                    field_name: {
                        parameter: separability_by_stratum(
                            view_rows_measured, parameter, field_name
                        )
                        for parameter in ("subsampling", "jpeg_quality")
                    }
                    for field_name in fields
                },
            }
            view_offenders = [
                f"pooled/{parameter}"
                for parameter, result in view_separability["pooled"].items()
                if result["separable"]
            ] + [
                f"{field_name}={stratum}/{parameter}"
                for field_name, per_parameter in view_separability["by_field"].items()
                for parameter, result in per_parameter.items()
                for stratum in result["separable_strata"]
            ]
            view_separability["separable"] = bool(view_offenders)
            view_separability["offenders"] = view_offenders
            view_summary["parameter_separability"] = view_separability
            if view_offenders:
                log.warning(
                    "matched view STILL separable on %s. The view fixes what "
                    "re-encoding can fix; a residual must be reported, not assumed "
                    "away (D-042).",
                    view_offenders,
                )
            else:
                log.info("matched view: no encoding parameter predicts the label")

            summary["jpeg_matched_view"] = view_summary
            log.info("matched view: %s", still_reason)
            log.info("wrote %s (%d rows)", view_path, len(view_rows))

    if args.pipeline_match_width:
        pipeline_rows, pipeline_summary = build_pipeline_matched_view(
            rows,
            args.set_id,
            THESIS_DBP / "outputs" / "p1" / "eval" / f"{args.set_id}_pipeline_matched",
            args.pipeline_match_width,
            args.pipeline_match_quality,
            args.pipeline_match_subsampling,
            args.pipeline_match_resample,
            log,
        )
        pipeline_path = (
            THESIS_DBP / "reports" / f"p1_eval_{args.set_id}_pipeline_matched_manifest.csv"
        )
        write_manifest(
            pipeline_rows,
            pipeline_path,
            (*columns, "view_path", "pipeline_applied"),
        )
        pipeline_table = compression_table(pipeline_rows)
        pipeline_summary["compression_table"] = pipeline_table
        pipeline_summary["compression_table_markdown"] = markdown_table(pipeline_table)
        pipeline_confounded, pipeline_reason = compression_is_confounded_with_label(
            {
                label: {"jpeg": entry["jpeg"], "non_jpeg": entry["non_jpeg"]}
                for label, entry in pipeline_table.items()
            }
        )
        pipeline_summary["compression_confounded"] = pipeline_confounded
        pipeline_summary["compression_verdict"] = pipeline_reason
        pipeline_sep: dict[str, Any] = {
            "threshold_lift": SEPARABILITY_LIFT_THRESHOLD,
            "pooled": {
                parameter: parameter_separability(pipeline_rows, parameter)
                for parameter in ("subsampling", "jpeg_quality")
            },
            "by_field": {
                field_name: {
                    parameter: separability_by_stratum(pipeline_rows, parameter, field_name)
                    for parameter in ("subsampling", "jpeg_quality")
                }
                for field_name in fields
            },
        }
        pipeline_offenders = [
            f"pooled/{parameter}"
            for parameter, result in pipeline_sep["pooled"].items()
            if result["separable"]
        ]
        pipeline_sep["separable"] = bool(pipeline_offenders)
        pipeline_sep["offenders"] = pipeline_offenders
        pipeline_summary["parameter_separability"] = pipeline_sep
        pipeline_summary["manifest"] = pipeline_path.name
        pipeline_summary["manifest_sha256"] = file_sha256(pipeline_path)
        summary["pipeline_matched_view"] = pipeline_summary
        for line in pipeline_summary["compression_table_markdown"].splitlines():
            log.info("%s", line)
        log.info("pipeline-matched view: %s", pipeline_reason)

    manifests = [
        Path(p.strip()) if Path(p.strip()).is_absolute() else THESIS_DBP / p.strip()
        for p in args.leak_check_against.split(",")
        if p.strip()
    ]
    if manifests:
        findings = leakage_report(rows, manifests, args.phash_threshold, log)
        leakage_path = THESIS_DBP / "reports" / "p1_leakage.json"
        existing: dict[str, Any] = {}
        if leakage_path.is_file():
            existing = json.loads(leakage_path.read_text(encoding="utf-8"))
        existing[args.set_id] = findings
        leakage_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
        log.info("wrote %s", leakage_path)
        summary["leakage"] = findings

    summary_path = THESIS_DBP / "reports" / f"p1_eval_{args.set_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
