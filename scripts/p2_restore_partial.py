"""Restore a partial S3 run so that only the unfinished members are computed.

A bundle is a directory holding ``probe/`` (member caches and their sidecars), ``s1/``
(feature caches) and ``reports/`` (the image lists the caches were built against). It is
copied into the staged repository, and every restored member is then checked with
:func:`p2_probe_features.shard_is_complete` -- the same function the probe run uses to
decide whether to skip it. A member that passes here is therefore a member the run will
skip, and one that fails here would have been silently recomputed there.

Nothing already in ``reports/`` is overwritten with different content: that is how a
stale copy of a manifest replaced the committed one in Session A.

Usage:
    python scripts/p2_restore_partial.py --bundle-root /kaggle/input/.../dbp-p2-s3-partial \\
        --expect-members 9
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

import p2_probe_features as probe  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.train.augment import CROP_SIZE  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

BUNDLE_PARTS = ("probe", "s1", "reports")
PROBE_LIST = "p2_probe_images.csv"


class RestoreError(RuntimeError):
    """Raised when a bundle cannot be restored as it stands."""


def find_bundle(root: Path) -> Path:
    """The single directory at or under ``root`` holding probe/, s1/ and reports/.

    A Kaggle dataset may present a zip's contents at its root or one level down,
    depending on how it was uploaded. Searching is safe only if the answer is unique.

    Raises:
        RestoreError: If no such directory exists, or more than one does.
    """
    if not root.is_dir():
        raise RestoreError(f"bundle root {root} does not exist")
    candidates = [root, *(p for p in sorted(root.rglob("*")) if p.is_dir())]
    hits = [c for c in candidates if all((c / part).is_dir() for part in BUNDLE_PARTS)]
    if len(hits) != 1:
        raise RestoreError(
            f"expected exactly one directory under {root} holding "
            f"{'/, '.join(BUNDLE_PARTS)}/, found {len(hits)}: {[str(h) for h in hits]}"
        )
    return hits[0]


def copy_tree(source: Path, destination: Path, refuse_conflicts: bool) -> dict[str, int]:
    """Copy every file in ``source`` into ``destination``.

    Args:
        source: Directory to copy from (not recursive; bundles are flat).
        destination: Directory to copy into.
        refuse_conflicts: If True, an existing file with different content stops the
            restore instead of being replaced.

    Returns:
        Counts of ``copied`` and ``identical`` files.

    Raises:
        RestoreError: On a conflict when ``refuse_conflicts`` is set.
    """
    destination.mkdir(parents=True, exist_ok=True)
    counts = {"copied": 0, "identical": 0}
    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        target = destination / item.name
        if target.is_file():
            if file_sha256(target) == file_sha256(item):
                counts["identical"] += 1
                continue
            if refuse_conflicts:
                raise RestoreError(
                    f"{target} already exists with different content; the bundle's "
                    f"copy would replace it. Refusing (a stale manifest replaced the "
                    f"committed one this way in Session A)."
                )
        shutil.copy2(item, target)
        counts["copied"] += 1
    return counts


def verify_members(
    probe_dir: Path, rows: list[dict[str, str]], crop: int, log: Any
) -> tuple[list[str], list[str]]:
    """Members whose caches the probe run would accept, and those it would not."""
    digest = probe.image_list_digest(rows)
    good: list[str] = []
    bad: list[str] = []
    for npz in sorted(probe_dir.glob("p2_probe_*.npz")):
        member = npz.stem.removeprefix("p2_probe_")
        if probe.shard_is_complete(npz, len(rows), crop, digest, log):
            good.append(member)
        else:
            bad.append(member)
    return good, bad


def main() -> int:
    """Restore and verify. Returns 0 on success, 1 on any failure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=THESIS_DBP,
                        help="Staged thesis_dbp to restore into.")
    parser.add_argument("--expect-members", type=int, required=True,
                        help="How many completed members the bundle must hold.")
    parser.add_argument("--crop", type=int, default=CROP_SIZE)
    args = parser.parse_args()
    log = setup_logging("p2_restore_partial")

    try:
        bundle = find_bundle(args.bundle_root)
        log.info("bundle: %s", bundle)
        probe_dir = args.repo / "outputs" / "p2" / "probe"
        counts = {
            "probe": copy_tree(bundle / "probe", probe_dir, refuse_conflicts=False),
            "s1": copy_tree(bundle / "s1", args.repo / "outputs" / "p2" / "s1",
                            refuse_conflicts=False),
            "reports": copy_tree(bundle / "reports", args.repo / "reports",
                                 refuse_conflicts=True),
        }
    except RestoreError as exc:
        log.error("%s", exc)
        return 1
    for part, c in counts.items():
        log.info("%-8s %d copied, %d already identical", part, c["copied"], c["identical"])

    list_path = args.repo / "reports" / PROBE_LIST
    if not list_path.is_file():
        log.error("%s is not in the bundle; restored caches cannot be checked", PROBE_LIST)
        return 1
    rows = probe.read_image_list(list_path)
    good, bad = verify_members(probe_dir, rows, args.crop, log)
    digest = probe.image_list_digest(rows)
    log.info("image list: %d rows, digest %s", len(rows), digest)
    log.info("members the probe run will skip (%d): %s", len(good), good)
    if bad:
        log.error("members present but NOT valid for this list (%d): %s", len(bad), bad)

    summary = {
        "bundle": str(bundle),
        "image_list_rows": len(rows),
        "image_list_digest": digest,
        "members_valid": good,
        "members_invalid": bad,
        "expected_members": args.expect_members,
        "copied": counts,
    }
    out = args.repo / "reports" / "p2_restore.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)

    if bad or len(good) != args.expect_members:
        log.error(
            "restore FAILED: %d valid member(s), expected %d", len(good), args.expect_members
        )
        return 1
    log.info("restore PASSED: %d member(s) verified against the image list", len(good))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
