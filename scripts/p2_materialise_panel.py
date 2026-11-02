"""P2-B1b: freeze the evaluation panel into one self-contained directory.

The panel is drawn once and shared by every stream (D-048). Until now that was true of
the *selection* but not of the *pixels*: each panel row recorded a path relative to the
session that built its view -- `/tmp/e7x/DDA-COCO/...` for E7, an absolute
`/kaggle/working/.../eval/e3_raw_jpeg_matched/...` for E3 -- and those paths mean nothing
anywhere else. 7,750 of the 12,150 rows were unreachable in a fresh session, and the
first thing to discover that would have been a GPU run.

So the panel becomes a directory. Every row's image is copied to
``<panel_root>/<view>/<image_id><suffix>``, **verified against the ``sha256`` the
manifest already records**, and the path column is rewritten relative to the panel root.
After this, a session needs the corpus root and the panel root, and nothing else.

Three things this deliberately does not do:

* **No re-encoding.** The source suffix is preserved, so a PNG stays a PNG. Re-encoding
  would change the pixels the panel was selected on and silently invalidate the hash.
* **No tolerance for a hash mismatch.** A re-derived matched view must be byte-identical
  -- I-010 measured the encoder as identical across libjpeg versions -- so a mismatch
  means something changed, and the run stops rather than recording a new hash over the
  old one.
* **No path outside the panel root.** An ``image_id`` containing a separator or ``..``
  is refused rather than cleaned, because a cleaned path is a path nobody checked.

**The luma-only variant (§6.5, pre-registration §8).** E7's FLUX.1 subset is perfectly
separable by chroma subsampling, so the pre-registration requires its number to be
reported as a pair, standard / luma-only, with the luma-only one entering M1 and M2.
``--luma-subsets`` writes that variant into its own view directory as **PNG for both
classes**: lossless and identical for reals and fakes, so the transform introduces no
new label-correlated difference, and taking luma discards the chroma planes where the
subsampling artefact lives. It is a companion view and is **not** counted in the
panel's 12,150.

**The digest.** SHA-256 over the sorted ``image_id<TAB>sha256`` lines, recorded in
``p2_panel_digest.json``. Session B asserts it before any GPU work starts, which is the
cheapest possible place to find out that the panel is not the panel.

Usage:
    python scripts/p2_materialise_panel.py \\
        --source e7_dda_coco=/tmp/e7x/DDA-COCO \\
        --source e8_evalgen=/tmp/e8x \\
        --source t_ctrl_diagnostic=/kaggle/working/p1 \\
        --luma-subsets "e7_dda_coco:FLUX.1,real"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256, read_manifest, write_manifest  # noqa: E402
from dbp.eval.panel import EVAL_CROP, PANEL_CAPS, filter_by_min_side  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: Columns of the rewritten panel list.
PANEL_LIST_COLUMNS = ("image_id", "relative_path", "label", "view", "sha256")

#: The panel size every later session asserts. Measured in Session A.
#: Rows the panel holds after the evaluation-crop rule (I-047). The caps allocate
#: 12,150; 375 of those are too small to be centre-cropped to 320 and are dropped
#: label-blind, with no backfill. Pre-registration Amendment 1.
EXPECTED_PANEL_ROWS = 11775

#: BT.601 luma weights, the same convention the augmentation and the greyscale
#: detector use, so "greyscale" means one thing across the project.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)


class MaterialiseError(RuntimeError):
    """Raised when the panel cannot be frozen exactly as the manifests describe."""


def safe_name(image_id: str, suffix: str) -> str:
    """File name for a panel image, refusing anything that could escape the root.

    Raises:
        MaterialiseError: If the id carries a path separator, a ``..`` or a drive
            letter. Sanitising it instead would produce a path nobody verified.
    """
    text = str(image_id)
    if not text or text != PurePosixPath(text).name or ".." in text or ":" in text:
        raise MaterialiseError(
            f"image_id {text!r} is not usable as a file name; refusing to sanitise it"
        )
    return f"{text}{suffix}"


def resolve_source(
    row: dict[str, str], view: str, source_roots: dict[str, Path]
) -> Path | None:
    """Where this panel row's image is in *this* session.

    Whether the row is **derived** decides where to look, because the recorded
    ``sha256`` describes a different file in each case.

    A row with a ``view_path`` came from a derived view, and the hash is of that
    derived image. Only two places can hold it: the recorded path, or its basename
    under the root given for this view -- which is what lets an already-built view be
    *attached* instead of rebuilt. ``relative_path`` and ``source_path`` both name the
    image **before** derivation, so neither is tried: E5's rows record a
    pipeline-matched JPEG while their ``relative_path`` and ``source_path`` point at
    the untouched Chameleon original, and returning that original passes a file whose
    hash cannot match and whose pixels are a 4K image rather than a 500-px one.

    A row with no ``view_path`` is undeived -- E7, E8, T-ctrl -- and its hash is of the
    source image, so ``relative_path`` under the view's root comes first and
    ``source_path`` is the fallback.

    Returns:
        The existing path, or ``None``.
    """
    root = source_roots.get(view)
    view_path = str(row.get("view_path") or "").strip()

    if view_path:
        candidate = Path(view_path)
        if candidate.is_file():
            return candidate
        if root is not None and root.is_dir():
            name = PurePosixPath(view_path.replace("\\", "/")).name
            for hit in root.rglob(name):
                if hit.is_file():
                    return hit
        return None

    relative = str(row.get("relative_path") or "").strip()
    if root is not None and relative and not PurePosixPath(relative).is_absolute():
        candidate = root / relative
        if candidate.is_file():
            return candidate

    source_path = str(row.get("source_path") or "").strip()
    if source_path and Path(source_path).is_file():
        return Path(source_path)
    return None


def materialise_view(
    view: str,
    rows: list[dict[str, str]],
    panel_root: Path,
    source_roots: dict[str, Path],
    log: Any,
) -> list[dict[str, Any]]:
    """Copy one view's images under the panel root and return the rewritten rows.

    Raises:
        MaterialiseError: On an unresolvable source, a hash mismatch, or a row with no
            recorded hash to check against.
    """
    destination_dir = panel_root / view
    if not destination_dir.is_dir():
        destination_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict[str, Any]] = []
    copied = reused = 0

    for row in rows:
        image_id = row["image_id"]
        expected = str(row.get("sha256") or "").strip()
        if len(expected) != 64:
            raise MaterialiseError(
                f"{view}/{image_id}: the manifest records no sha256, so the copy "
                "cannot be verified. Refusing to materialise an unchecked image."
            )

        # A file already present under the panel root that hashes to the recorded value
        # IS the panel image, and no source is needed to prove it. Resolving one first
        # would make a re-run depend on corpora that were deleted the moment the panel
        # was frozen -- which is the whole point of freezing it (D-050). It also lets
        # the panel root be read-only, as it is once published as a dataset.
        present = [
            candidate
            for candidate in sorted(destination_dir.glob(f"{safe_name(image_id, '')}.*"))
            if candidate.is_file() and file_sha256(candidate) == expected
        ]
        if present:
            destination = present[0]
            reused += 1
        else:
            source = resolve_source(row, view, source_roots)
            if source is None:
                raise MaterialiseError(
                    f"{view}/{image_id}: not found. Tried view_path "
                    f"{row.get('view_path')!r}, {source_roots.get(view)} / "
                    f"{row.get('relative_path')!r}, and source_path "
                    f"{row.get('source_path')!r}."
                )
            destination = destination_dir / safe_name(image_id, source.suffix.lower())
            resolved = destination.resolve()
            if not resolved.is_relative_to(panel_root.resolve()):
                raise MaterialiseError(f"{destination} would fall outside {panel_root}")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
            copied += 1

            actual = file_sha256(destination)
            if actual != expected:
                raise MaterialiseError(
                    f"{view}/{image_id}: sha256 mismatch after copying from {source}.\n"
                    f"  manifest: {expected}\n  on disk  : {actual}\n"
                    "A re-derived matched view must be byte-identical (I-010). Stopping; "
                    "the recorded hash is not overwritten."
                )

        out.append(
            {
                "image_id": image_id,
                "relative_path": f"{view}/{destination.name}",
                "label": row.get("label", ""),
                "view": view,
                "sha256": expected,
            }
        )

    log.info("%-24s %5d row(s): %d copied, %d already present", view, len(out), copied, reused)
    return out


def write_luma_variant(
    panel_rows: list[dict[str, Any]],
    source_rows: dict[str, dict[str, str]],
    subsets: dict[str, set[str]],
    panel_root: Path,
    log: Any,
) -> list[dict[str, Any]]:
    """Greyscale-in-RGB copies of the named subsets, as a companion view.

    PNG for both classes: lossless and identical either side, so the transform adds no
    label-correlated difference of its own. Taking luma discards the chroma planes,
    which is where E7's 4:2:0-versus-4:4:4 shortcut lives.

    Args:
        panel_rows: The already-materialised panel rows.
        source_rows: ``image_id`` -> its original manifest row, for the subset column.
        subsets: ``view`` -> subset values to convert.
        panel_root: Panel root.
        log: Logger.

    Returns:
        The rewritten luma rows.
    """
    import numpy as np
    from PIL import Image

    from dbp.utils.io import load_rgb_uint8

    out: list[dict[str, Any]] = []
    for view, wanted in sorted(subsets.items()):
        column = {"e7_dda_coco": "source_vae"}.get(view, "label")
        target_view = f"{view}_luma"
        destination_dir = panel_root / target_view
        if not destination_dir.is_dir():
            destination_dir.mkdir(parents=True, exist_ok=True)
        selected = [
            row for row in panel_rows
            if row["view"] == view
            and str(source_rows.get(row["image_id"], {}).get(column, "")) in wanted
        ]
        if not selected:
            raise MaterialiseError(
                f"{view}: no rows matched {sorted(wanted)} on column {column!r}; the "
                "luma-only variant the pre-registration requires would be empty"
            )
        for row in selected:
            destination = destination_dir / safe_name(row["image_id"], ".png")
            # Deterministic from its parent, so an existing companion is the companion.
            if not destination.is_file():
                image = load_rgb_uint8(panel_root / row["relative_path"])
                weights = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
                grey = (
                    np.clip(image.astype(np.float64) @ weights, 0, 255)
                    .round().astype(np.uint8)
                )
                rgb = np.repeat(grey[:, :, None], 3, axis=2)
                temporary = destination.with_suffix(".png.tmp")
                Image.fromarray(rgb, mode="RGB").save(
                    temporary, format="PNG", optimize=False
                )
                temporary.replace(destination)
            out.append(
                {
                    "image_id": row["image_id"],
                    "relative_path": f"{target_view}/{destination.name}",
                    "label": row["label"],
                    "view": target_view,
                    "sha256": file_sha256(destination),
                }
            )
        log.info(
            "%-24s %5d row(s) from %s (greyscale-in-RGB PNG, both classes)",
            target_view, len(selected), sorted(wanted),
        )
    return out


def panel_digest(rows: list[dict[str, Any]]) -> str:
    """SHA-256 over the sorted ``image_id<TAB>sha256`` lines.

    One number that says whether a later session is looking at the same panel. Sorted,
    so it does not depend on the order the views were materialised in.
    """
    lines = sorted(f"{row['image_id']}\t{row['sha256']}" for row in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def parse_mapping(values: list[str], label: str) -> dict[str, Path]:
    """Parse repeated ``view=path`` arguments."""
    out: dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise MaterialiseError(f"--{label} expects view=path, got {item!r}")
        view, path = item.split("=", 1)
        out[view.strip()] = Path(path.strip())
    return out


def main() -> int:
    """Freeze the panel. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", default=[], metavar="VIEW=PATH",
        help="Root that a view's relative_path values resolve against. Repeatable.",
    )
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument(
        "--panel-root", type=Path, default=THESIS_DBP / "outputs" / "p2" / "panel"
    )
    parser.add_argument(
        "--luma-subsets", action="append", default=[], metavar="VIEW:A,B",
        help="Subsets to also write as a greyscale-in-RGB companion view.",
    )
    parser.add_argument(
        "--min-side", type=int, default=EVAL_CROP,
        help="Drop panel rows whose shorter side is below this, using the manifests' "
             "width/height -- no image is re-read. A row below the evaluation crop "
             "cannot be centre-cropped to it and so cannot be scored at all (I-047). "
             "Label-blind and without backfill. 0 disables it.",
    )
    parser.add_argument(
        "--expect-rows", type=int, default=EXPECTED_PANEL_ROWS,
        help="0 disables the count assertion; any other value is enforced.",
    )
    args = parser.parse_args()
    log = setup_logging("p2_materialise_panel")

    try:
        source_roots = parse_mapping(args.source, "source")
    except MaterialiseError as exc:
        log.error("%s", exc)
        return 2
    for view, root in sorted(source_roots.items()):
        log.info("source %-24s %s%s", view, root, "" if root.is_dir() else "   <== MISSING")

    luma_subsets: dict[str, set[str]] = {}
    for item in args.luma_subsets:
        if ":" not in item:
            log.error("--luma-subsets expects VIEW:A,B, got %r", item)
            return 2
        view, values = item.split(":", 1)
        luma_subsets[view.strip()] = {v.strip() for v in values.split(",") if v.strip()}

    panel_root: Path = args.panel_root
    panel_root.mkdir(parents=True, exist_ok=True)

    panel_rows: list[dict[str, Any]] = []
    source_rows: dict[str, dict[str, str]] = {}
    per_view: dict[str, int] = {}
    undersized: dict[str, dict[str, Any]] = {}
    try:
        for view in sorted(PANEL_CAPS):
            path = args.manifests / f"p2_panel_{view}.csv"
            if not path.is_file():
                log.error("%s not found -- run p2_build_panel.py for every view first", path)
                return 2
            rows = read_manifest(path)
            rows, dropped, unmeasured = filter_by_min_side(rows, args.min_side)
            if dropped or unmeasured:
                by_label = Counter(str(r.get("label", "")) for r in dropped)
                undersized[view] = {
                    "dropped": len(dropped),
                    "by_label": dict(sorted(by_label.items())),
                    "unmeasured_kept": unmeasured,
                    "image_ids": sorted(str(r["image_id"]) for r in dropped),
                }
                log.info(
                    "%-24s min-side %d: %d dropped %s, %d row(s) record no size and "
                    "are kept", view, args.min_side, len(dropped),
                    dict(sorted(by_label.items())), unmeasured,
                )
            for row in rows:
                source_rows[row["image_id"]] = row
            rewritten = materialise_view(view, rows, panel_root, source_roots, log)
            panel_rows.extend(rewritten)
            per_view[view] = len(rewritten)

        luma_rows = (
            write_luma_variant(panel_rows, source_rows, luma_subsets, panel_root, log)
            if luma_subsets
            else []
        )
    except MaterialiseError as exc:
        log.error("%s", exc)
        return 1

    if args.expect_rows and len(panel_rows) != args.expect_rows:
        log.error(
            "panel has %d rows, expected %d. Not writing the digest: a panel of the "
            "wrong size is not the panel the budget and the pre-registration describe.",
            len(panel_rows), args.expect_rows,
        )
        return 1

    write_manifest(panel_rows, args.manifests / "p2_panel_images.csv", PANEL_LIST_COLUMNS)
    if luma_rows:
        write_manifest(
            luma_rows, args.manifests / "p2_panel_images_luma.csv", PANEL_LIST_COLUMNS
        )

    digest = panel_digest(panel_rows)
    summary = {
        "decision": "D-050",
        "panel_root": str(panel_root),
        "rows": len(panel_rows),
        "by_view": per_view,
        "panel_digest": digest,
        "digest_definition": (
            "sha256 of the newline-joined sorted 'image_id<TAB>sha256' lines of "
            "p2_panel_images.csv"
        ),
        "luma_variant": {
            "rows": len(luma_rows),
            "views": sorted({row["view"] for row in luma_rows}),
            "digest": panel_digest(luma_rows) if luma_rows else None,
            "format": "greyscale-in-RGB PNG, both classes, not counted in the panel",
        },
        "min_side": {
            "threshold": args.min_side,
            "rule": (
                "I-047: a panel row whose shorter side is below the evaluation crop "
                "cannot be centre-cropped to it. Label-blind, no backfill."
            ),
            "total_dropped": sum(v["dropped"] for v in undersized.values()),
            "by_view": undersized,
        },
        "verification": "every file re-hashed and compared with the manifest's sha256",
        "licence": (
            "contains research-only and evaluation-only imagery (WildRF, Chameleon). "
            "Publish as a PRIVATE Kaggle dataset only -- CLAUDE.md section 6c."
        ),
    }
    path = args.manifests / "p2_panel_digest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

    log.info("panel: %d rows under %s", len(panel_rows), panel_root)
    log.info("panel digest: %s", digest)
    if luma_rows:
        log.info("luma companion: %d rows (not counted in the panel)", len(luma_rows))
    log.info("wrote reports/%s", path.name)
    log.info(
        "PRIVATE dataset only: this directory holds research-only and evaluation-only "
        "imagery (section 6c)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
