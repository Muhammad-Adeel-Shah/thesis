"""P0 protected-asset integrity check.

Re-scans the protected set and diffs it against the B0 baseline
(``reports/p0_protected_manifest_before.json``). Any difference is a critical failure:
it means something outside the P0 write scope changed.

The scanning algorithm below is byte-for-byte the one that produced the baseline.
``--self-test`` proves that by regenerating the baseline and comparing, so the check
cannot pass merely because both sides drifted together.

Exit codes: 0 = clean, 1 = differences found, 2 = the check could not run.

Usage:
    python scripts/p0_verify_integrity.py
    python scripts/p0_verify_integrity.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
REPO_ROOT = THESIS_DBP.parent

ALGO_VERSION = 1
HASH_EXTS = frozenset(
    {".py", ".ipynb", ".sh", ".yaml", ".yml", ".json", ".csv", ".md", ".txt", ".toml", ".cfg"}
)
EXCLUDED_DIRS = (".git", "thesis_dbp")
EXCLUDED_FILES = (".gitignore", "CLAUDE.md")

BASELINE_NOTE = (
    "Protected set = every pre-existing file in REPO_ROOT outside the P0 write scope "
    "(.git/, thesis_dbp/, .gitignore, CLAUDE.md). It is EMPTY at P0 because REPO_ROOT "
    "was a fresh directory; FIRE and the zero-shot results live on Kaggle and are not "
    "mirrored here. Any entry appearing in a later verification that was not added "
    "deliberately is a scope-lock violation. Re-baseline when prior_results/ is added."
)


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan(root: Path) -> list[dict[str, Any]]:
    """Walk ``root`` and describe every protected file, sorted by POSIX relpath."""
    entries: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root)
        at_root = rel_dir == Path(".")
        if at_root:
            dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        else:
            dirnames[:] = sorted(dirnames)
        for name in sorted(filenames):
            rel = name if at_root else (rel_dir / name).as_posix()
            if at_root and rel in EXCLUDED_FILES:
                continue
            path = Path(dirpath) / name
            stat = path.stat()
            mode = "sha256" if path.suffix.lower() in HASH_EXTS else "fast"
            entries.append(
                {
                    "path": rel,
                    "mode": mode,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_of(path) if mode == "sha256" else None,
                }
            )
    entries.sort(key=lambda e: e["path"])
    return entries


def build(root: Path, stamp: str) -> dict[str, Any]:
    """Assemble the full manifest document for ``root``."""
    entries = scan(root)
    return {
        "algo_version": ALGO_VERSION,
        "generated_at_utc": stamp,
        "repo_root_basename": root.name,
        "excluded_top_level_dirs": list(EXCLUDED_DIRS),
        "excluded_files": list(EXCLUDED_FILES),
        "hash_extensions": sorted(HASH_EXTS),
        "note": BASELINE_NOTE,
        "counts": {
            "total": len(entries),
            "sha256": sum(1 for e in entries if e["mode"] == "sha256"),
            "fast": sum(1 for e in entries if e["mode"] == "fast"),
        },
        "entries": entries,
    }


def report_is_materially_unchanged(report: dict[str, Any], dest: Path) -> bool:
    """Whether ``dest`` already holds this report, ignoring only the timestamp.

    Without this the check rewrites `generated_at_utc` on every run and leaves the
    working tree dirty even when the verdict is CLEAN, which trains the reader to
    ignore a modified report file -- exactly the file that must never be ignored.
    """
    if not dest.is_file():
        return False
    try:
        existing = json.loads(dest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not isinstance(existing, dict):
        return False
    ignore = "generated_at_utc"
    return {k: v for k, v in existing.items() if k != ignore} == {
        k: v for k, v in report.items() if k != ignore
    }


def write_report_if_changed(report: dict[str, Any], dest: Path, force: bool = False) -> bool:
    """Write the report unless it would only change the timestamp.

    Args:
        report: The report to write.
        dest: Destination path.
        force: Write unconditionally (used when ``--out`` was given explicitly, where
            the caller has asked for a file at a specific place).

    Returns:
        True if the file was written.
    """
    if not force and report_is_materially_unchanged(report, dest):
        return False
    atomic_write_json(report, dest)
    return True


def atomic_write_json(obj: dict[str, Any], dest: Path) -> None:
    """Write ``obj`` as UTF-8 JSON via a temp file plus os.replace."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(obj, handle, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def diff(baseline: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, list]:
    """Compare two entry lists.

    A hashed file counts as ``content_changed`` only when its digest differs; a changed
    mtime with an identical digest is reported separately as ``mtime_only`` so that a
    harmless touch is not mistaken for tampering. Fast-mode files have no digest, so
    size or mtime change is all there is to go on.
    """
    before = {e["path"]: e for e in baseline}
    after = {e["path"]: e for e in current}

    result: dict[str, list] = {"added": [], "removed": [], "content_changed": [], "mtime_only": []}
    result["added"] = sorted(set(after) - set(before))
    result["removed"] = sorted(set(before) - set(after))

    for path in sorted(set(before) & set(after)):
        old, new = before[path], after[path]
        if old["mode"] == "sha256" and new["mode"] == "sha256":
            if old["sha256"] != new["sha256"]:
                result["content_changed"].append(
                    {"path": path, "before_sha256": old["sha256"], "after_sha256": new["sha256"]}
                )
            elif old["mtime_ns"] != new["mtime_ns"]:
                result["mtime_only"].append(path)
        else:
            if old["size"] != new["size"]:
                result["content_changed"].append(
                    {"path": path, "before_size": old["size"], "after_size": new["size"]}
                )
            elif old["mtime_ns"] != new["mtime_ns"]:
                result["mtime_only"].append(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the protected set is unchanged.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=THESIS_DBP / "reports" / "p0_protected_manifest_before.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Where to write the report. When omitted, the default report is written "
            "only if its content (ignoring the timestamp) actually changed, so a clean "
            "run does not dirty the working tree."
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Also regenerate the baseline with this algorithm and require byte-identity.",
    )
    args = parser.parse_args()

    if not args.baseline.is_file():
        print(f"FATAL: baseline manifest not found: {args.baseline}", file=sys.stderr)
        return 2

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("algo_version") != ALGO_VERSION:
        print(
            f"FATAL: baseline algo_version {baseline.get('algo_version')} != {ALGO_VERSION}. "
            f"The manifest algorithm changed; the comparison would be meaningless.",
            file=sys.stderr,
        )
        return 2

    current = scan(REPO_ROOT)
    differences = diff(baseline["entries"], current)
    changed = (
        len(differences["added"])
        + len(differences["removed"])
        + len(differences["content_changed"])
    )

    algorithm_self_test: dict[str, Any] | None = None
    if args.self_test:
        regenerated = build(REPO_ROOT, baseline["generated_at_utc"])
        expected = json.dumps(regenerated, indent=2, sort_keys=False) + "\n"
        actual = args.baseline.read_text(encoding="utf-8")
        algorithm_self_test = {
            "performed": True,
            "byte_identical_to_baseline": expected == actual,
            "meaning": (
                "True means p0_verify_integrity.py reproduces the B0 baseline exactly, "
                "so the two sides of this comparison cannot have drifted together."
            ),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algo_version": ALGO_VERSION,
        "baseline": args.baseline.name,
        "baseline_generated_at_utc": baseline["generated_at_utc"],
        "files_checked": len(current),
        "files_in_baseline": len(baseline["entries"]),
        "files_changed": changed,
        "differences": differences,
        "algorithm_self_test": algorithm_self_test,
        "verdict": "CLEAN" if changed == 0 else "CRITICAL-PROTECTED-FILES-CHANGED",
    }
    out = args.out if args.out is not None else THESIS_DBP / "reports" / "p0_integrity.json"
    written = write_report_if_changed(report, out, force=args.out is not None)

    print(f"protected files in baseline : {len(baseline['entries'])}")
    print(f"protected files now         : {len(current)}")
    print(f"changed                     : {changed} (expected 0)")
    if differences["mtime_only"]:
        print(f"mtime-only (content intact) : {len(differences['mtime_only'])}")
    if algorithm_self_test:
        print(f"algorithm self-test         : {algorithm_self_test['byte_identical_to_baseline']}")
    print(f"verdict                     : {report['verdict']}")
    print(f"report                      : {out}{'' if written else '  (unchanged, not rewritten)'}")

    if changed:
        for key in ("added", "removed", "content_changed"):
            for item in differences[key]:
                print(f"  {key.upper()}: {item}", file=sys.stderr)
        return 1
    if algorithm_self_test and not algorithm_self_test["byte_identical_to_baseline"]:
        print("FATAL: algorithm self-test failed; this check cannot be trusted.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
