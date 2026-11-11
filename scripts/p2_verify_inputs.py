"""P2-B0: prove the Kaggle inputs are the P1 corpus, before anything trains on them.

The Kaggle copy of the reals was truncated once already -- 1,314 PNGs of 4,100 -- and it
looked exactly like a complete dataset until it was counted. A stream trained on a
truncated corpus produces numbers, and nothing about them says the corpus was wrong.

So this script refuses to pass on a count alone. It checks:

1. every expected file is on disk, per member and per set;
2. every real's **SHA-256** against the committed manifest -- ``sha256`` for the reals,
   ``jpg_sha256`` for ``gen_matched`` (the two manifests name the column differently,
   and a script that guessed would verify nothing);
3. the pair manifests against ``reports/p1_pairs_manifest_digest.json``, because the
   11.3 MB combined manifest is too large to commit (CLAUDE.md section 2) and the
   digest is the tracked reference;
4. the greyscale exclusion flags -- 83 in t_real and 6 in t_ctrl -- because a manifest
   without them is the pre-D-039 build and would put 89 flagged reals back into
   training;
5. ``gen_matched`` at **2,488**. A count of 2,475 is the stale set and is a hard fail.

Any mismatch stops the run. Nothing here writes to the corpus.

**Absolute paths in a manifest.** The reals and pair manifests record paths relative to
the corpus root, but ``p1_gen_matched_manifest.csv`` recorded ``jpg_path`` and
``png_path`` as **absolute** paths inside the session that built it -- including a
``gen_matched/gen_matched`` nesting. Those paths are dead everywhere else, and
``Path(root) / "/abs/path"`` silently discards the root, so a naive join reports all
2,488 rows missing. :func:`resolve_row_path` handles it by falling back to the file's
basename under the expected sub-directory. The manifest is **not** rewritten: its
SHA-256 is part of the P1 provenance chain, and the recorded path is a true record of
where the build wrote the file.

Usage:
    python scripts/p2_verify_inputs.py --root /kaggle/input/.../outputs/p1 \\
        --manifests reports --out reports/p2_input_verification.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: What P1 left behind, every number measured in the P1 close session.
EXPECTED: dict[str, Any] = {
    "reals_total": 4100,
    "reals_t_real": 2500,
    "reals_t_ctrl": 1600,
    "pairs_total": 49200,
    "pairs_per_member": 4100,
    "distinct_members": 12,
    "generated": 2488,
    "gen_matched": 2488,
    "excluded_t_real": 83,
    "excluded_t_ctrl": 6,
    "usable_pairs": 48132,
}

#: Manifest -> the column holding the hash of the file that manifest describes.
HASH_COLUMN = {
    "p1_reals_manifest_t_real.csv": "sha256",
    "p1_reals_manifest_t_ctrl.csv": "sha256",
    "p1_gen_matched_manifest.csv": "jpg_sha256",
}

#: Manifest -> (path column, sub-directory of the corpus root). The sub-directory is
#: what lets an absolutely-recorded path be recovered; see :func:`resolve_row_path`.
PATH_COLUMN = {
    "p1_reals_manifest_t_real.csv": ("relative_path", "reals"),
    "p1_reals_manifest_t_ctrl.csv": ("relative_path", "reals"),
    "p1_gen_matched_manifest.csv": ("jpg_path", "gen_matched"),
}


class InputError(RuntimeError):
    """Raised when the attached corpus is not the one the manifests describe."""


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a manifest, raising if it is absent rather than returning nothing."""
    if not path.is_file():
        raise InputError(f"manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def count_files(root: Path) -> dict[str, Any]:
    """Count what is actually on disk, by set and by member.

    ``gen_matched`` is counted as ``*.jpg``: it is the JPEG-matched variant, and
    counting ``*.png`` there reported 0 during P1 and looked like a failed build.
    """
    counts: dict[str, Any] = {}
    for name, pattern in (
        ("reals", "*.png"),
        ("pairs", "*.png"),
        ("generated", "*.png"),
        ("gen_matched", "*.jpg"),
    ):
        directory = root / name
        counts[name] = (
            sum(1 for _ in directory.rglob(pattern)) if directory.is_dir() else 0
        )
    pairs_dir = root / "pairs"
    counts["per_member"] = (
        {
            member.name: sum(1 for _ in member.glob("*.png"))
            for member in sorted(pairs_dir.iterdir())
            if member.is_dir()
        }
        if pairs_dir.is_dir()
        else {}
    )
    return counts


def resolve_row_path(root: Path, recorded: str, subdir: str) -> Path | None:
    """Locate a manifest row's file under ``root``, tolerating an absolute record.

    Three attempts, in order of how much they trust the manifest:

    1. ``root / recorded`` -- the normal case, a relative path;
    2. the recorded path's own tail from ``subdir`` onwards, which recovers a path that
       was absolute in the session that wrote it;
    3. the basename anywhere under ``root / subdir``, which also absorbs the
       ``gen_matched/gen_matched`` nesting the P1 fix notebook produced.

    Args:
        root: Corpus root.
        recorded: The path as the manifest records it.
        subdir: Sub-directory of ``root`` the file belongs in.

    Returns:
        The existing path, or ``None`` if no attempt found a file. ``None`` is reported
        as missing rather than guessed at -- a wrong file that hashes differently would
        read as corruption, and a wrong file that happens to exist would be worse.
    """
    text = str(recorded).replace("\\", "/")

    direct = root / text
    if not Path(text).is_absolute() and direct.is_file():
        return direct

    marker = f"/{subdir}/"
    if marker in text:
        tail = text[text.rindex(marker) + 1 :]
        candidate = root / tail
        if candidate.is_file():
            return candidate

    name = PurePosixPath(text).name
    base = root / subdir
    if base.is_dir():
        for hit in base.rglob(name):
            if hit.is_file():
                return hit
    return None


def verify_hashes(
    rows: list[dict[str, str]],
    root: Path,
    hash_column: str,
    path_column: str,
    subdir: str,
    limit: int | None,
    log: Any,
) -> dict[str, Any]:
    """Re-hash the files a manifest describes and compare.

    Args:
        rows: Manifest rows.
        root: Corpus root the paths resolve against.
        hash_column: Column holding the expected SHA-256.
        path_column: Column holding the path.
        subdir: Sub-directory of ``root`` the files live in, used to recover a path the
            manifest recorded as absolute.
        limit: Verify only the first N rows, for a smoke run. ``None`` verifies all.
        log: Logger.

    Returns:
        A report with the counts and up to ten offending paths.

    Raises:
        InputError: If the manifest has no hash column at all.
    """
    if rows and hash_column not in rows[0]:
        raise InputError(
            f"manifest has no {hash_column!r} column; it has {sorted(rows[0])}. "
            "Verifying against a column that is not there would pass silently."
        )
    subset = rows if limit is None else rows[:limit]
    missing: list[str] = []
    differing: list[str] = []
    checked = 0
    resolved_elsewhere = 0
    for index, row in enumerate(subset):
        recorded = str(row[path_column])
        path = resolve_row_path(root, recorded, subdir)
        if path is None:
            missing.append(recorded)
            continue
        if Path(recorded.replace("\\", "/")).is_absolute():
            resolved_elsewhere += 1
        checked += 1
        if file_sha256(path) != row[hash_column]:
            differing.append(recorded)
        if index and index % 500 == 0:
            log.info("  hashed %d/%d", index, len(subset))
    return {
        "rows": len(rows),
        "verified": checked,
        "limited_to": limit,
        "missing": len(missing),
        "differing": len(differing),
        "missing_examples": missing[:10],
        "differing_examples": differing[:10],
        "hash_column": hash_column,
        "subdir": subdir,
        "recorded_absolute": resolved_elsewhere,
    }


def verify_pair_digest(digest_path: Path, manifest_dir: Path, log: Any) -> dict[str, Any]:
    """Check the attached pair manifests against the committed digest."""
    if not digest_path.is_file():
        raise InputError(f"pair-manifest digest not found: {digest_path}")
    digest = json.loads(digest_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for name, expected in digest["manifests"].items():
        path = manifest_dir / name
        if not path.is_file():
            results[name] = {"status": "MISSING"}
            continue
        actual = file_sha256(path)
        results[name] = {
            "status": "OK" if actual == expected["sha256"] else "DIFFERS",
            "rows_expected": expected["rows"],
            "precision": expected["precision"],
        }
        log.info("  %-36s %s", name, results[name]["status"])
    return results


def main() -> int:
    """Verify the attached corpus; return 0 only when everything matches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="outputs/p1 on Kaggle")
    parser.add_argument(
        "--manifests", type=Path, default=THESIS_DBP / "reports",
        help="Directory holding the committed manifests.",
    )
    parser.add_argument(
        "--hash-limit", type=int, default=0,
        help="Hash only the first N rows per manifest (0 = all). A smoke value here "
             "must be stated in the report; it is not a full verification.",
    )
    parser.add_argument(
        "--out", type=Path, default=THESIS_DBP / "reports" / "p2_input_verification.json"
    )
    args = parser.parse_args()
    log = setup_logging("p2_verify_inputs")

    root: Path = args.root
    if not root.is_dir():
        log.error("corpus root not found: %s", root)
        return 2

    failures: list[str] = []
    counts = count_files(root)
    log.info("on disk: %s", {k: v for k, v in counts.items() if k != "per_member"})

    for key, expected in (
        ("reals", EXPECTED["reals_total"]),
        ("pairs", EXPECTED["pairs_total"]),
        ("generated", EXPECTED["generated"]),
        ("gen_matched", EXPECTED["gen_matched"]),
    ):
        if counts[key] != expected:
            note = ""
            if key == "gen_matched" and counts[key] == 2475:
                note = " -- this is the STALE set from before the D-039 rebuild"
            failures.append(f"{key}: {counts[key]} on disk, expected {expected}{note}")

    short = {m: n for m, n in counts["per_member"].items() if n != EXPECTED["pairs_per_member"]}
    if short:
        failures.append(f"members short of {EXPECTED['pairs_per_member']}: {short}")
    if len(counts["per_member"]) != EXPECTED["distinct_members"]:
        failures.append(
            f"{len(counts['per_member'])} member directories, expected "
            f"{EXPECTED['distinct_members']}"
        )

    # Exclusion flags: without them, 89 greyscale reals re-enter training.
    exclusions: dict[str, Any] = {}
    for name, expected in (
        ("p1_reals_manifest_t_real.csv", EXPECTED["excluded_t_real"]),
        ("p1_reals_manifest_t_ctrl.csv", EXPECTED["excluded_t_ctrl"]),
    ):
        rows = read_rows(args.manifests / name)
        kinds = Counter(r["excluded_reason"] for r in rows if r.get("excluded_reason"))
        flagged = sum(kinds.values())
        exclusions[name] = {"rows": len(rows), "flagged": flagged, "kinds": dict(kinds)}
        log.info("%s: %d rows, %d flagged", name, len(rows), flagged)
        if flagged != expected:
            failures.append(
                f"{name}: {flagged} rows flagged, expected {expected}"
                + (" -- this is the pre-D-039 build manifest" if flagged == 0 else "")
            )

    limit = args.hash_limit or None
    hashes: dict[str, Any] = {}
    for name, (path_column, subdir) in PATH_COLUMN.items():
        log.info("hashing against %s ...", name)
        hashes[name] = verify_hashes(
            read_rows(args.manifests / name), root, HASH_COLUMN[name],
            path_column, subdir, limit, log,
        )
        result = hashes[name]
        log.info(
            "  verified %d, missing %d, differing %d%s",
            result["verified"], result["missing"], result["differing"],
            f" ({result['recorded_absolute']} recovered from an absolute record)"
            if result["recorded_absolute"] else "",
        )
        if result["missing"] or result["differing"]:
            failures.append(
                f"{name}: {result['missing']} missing, {result['differing']} differing"
            )

    pair_digest = verify_pair_digest(
        args.manifests / "p1_pairs_manifest_digest.json", args.manifests, log
    )
    bad = [n for n, r in pair_digest.items() if r["status"] != "OK"]
    if bad:
        failures.append(f"pair manifests not matching the committed digest: {bad}")

    report = {
        "root": str(root),
        "expected": EXPECTED,
        "counts": counts,
        "exclusions": exclusions,
        "hashes": hashes,
        "pair_manifest_digest": pair_digest,
        "hash_limit": limit,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.out)

    if failures:
        log.error("VERDICT: FAIL")
        for failure in failures:
            log.error("  %s", failure)
        return 1
    log.info("VERDICT: PASS -- the attached corpus is the P1 corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
