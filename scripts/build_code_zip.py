"""Build the code zip for Kaggle from git's own bytes, so it matches a commit exactly.

The zip used to be built from the working tree. On Windows that tree drifts to CRLF line
endings whenever a file is written in text mode, while git stores LF (``.gitattributes``
says ``eol=lf``). Python does not care. Bash on Kaggle does: a CRLF
``kaggle_bootstrap.sh`` fails on its first line with ``$'\\r': command not found``, and
Git Bash on Windows tolerates CRLF, so no local run would ever have shown it.

Reading every entry as a raw blob (``git cat-file --batch``) makes the zip byte-identical
to the commit it names. ``COMMIT.txt`` is generated into the zip -- a tracked file cannot
contain its own commit hash. The runbook dry-run builds its code fixture with
:func:`extract`, so the dry-run tests the same bytes Kaggle receives.

Usage:
    python scripts/build_code_zip.py --out F:/thesis-2/thesis_dbp_p2.zip
    python scripts/build_code_zip.py --bundle p4-dda-equiv --out F:/thesis-2/kaggle_raw/x.zip
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
PREFIX = "thesis_dbp"


def _git_root() -> Path:
    """The repository root: thesis_dbp/ itself (since 2026-11-05) or its parent (before)."""
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=THESIS_DBP,
                         capture_output=True, text=True, check=True).stdout.strip()
    return Path(out).resolve()


REPO_ROOT = _git_root()
#: How thesis_dbp/ is spelled inside git: "" when it is the repository root. Zip entries
#: always start with ``thesis_dbp/``, whatever the layout, because the runbooks look for
#: ``dbp-code*/thesis_dbp``.
GIT_PREFIX = "" if THESIS_DBP.resolve() == REPO_ROOT else f"{PREFIX}/"


def to_git(path: str) -> str:
    """A zip entry name (``thesis_dbp/...``) as git spells it."""
    return GIT_PREFIX + path.removeprefix(f"{PREFIX}/")

#: Extensions that must reach Kaggle with LF line endings.
LF_ONLY = (".sh", ".py")

#: Generated into the zip, never taken from git: a tracked file cannot hold its own
#: commit hash. An older commit tracked it by mistake, so it is excluded by name.
GENERATED = f"{PREFIX}/COMMIT.txt"

#: Code-only subsets, for a notebook that must see no data and no reports. The isolated
#: DDA equivalence check (user decision 2026-11-01) runs DDA's own code, so its notebook
#: gets only what the check needs: no dataset manifest, no report, no other script.
BUNDLES: dict[str, tuple[str, ...]] = {
    "p4-dda-equiv": (
        "pyproject.toml", "requirements/base.txt", "env/kaggle_bootstrap.sh",
        "env/kaggle_run_p4_dda_equivalence.py", "env/p4_dda_equivalence_expect.json",
        "scripts/p4_dda_equivalence.py", "scripts/p2_preflight.py", "src/dbp/__init__.py",
        "src/dbp/baselines/", "src/dbp/data/__init__.py", "src/dbp/data/manifest.py",
        "src/dbp/utils/__init__.py", "src/dbp/utils/logging.py",
    ),
}


def select(paths: list[str], bundle: str | None) -> list[str]:
    """``paths`` restricted to a bundle's entries (a trailing ``/`` is a directory)."""
    if bundle is None:
        return paths
    entries = [f"{PREFIX}/{e}" for e in BUNDLES[bundle]]
    chosen = [p for p in paths if any(p == e or (e.endswith("/") and p.startswith(e))
                                      for e in entries)]
    missing = [e for e in entries if not any(p == e or p.startswith(e) for p in chosen)]
    if missing:
        raise BuildError(f"bundle {bundle}: not in the tree: {missing}")
    return chosen


class BuildError(RuntimeError):
    """Raised when the zip would not match its commit."""


def git(*args: str) -> str:
    """Run git in the repository root and return stdout."""
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout


def resolve(ref: str) -> tuple[str, str]:
    """``(short, full)`` commit of ``ref``. The index has no commit: ``("index", "")``."""
    if ref == ":":
        return "index", ""
    full = git("rev-parse", ref).strip()
    return full[:7], full


def tracked(ref: str) -> list[str]:
    """Zip names (``thesis_dbp/...``) of the files in a commit, or in the index (``:``)."""
    scope = GIT_PREFIX.rstrip("/") or "."
    if ref == ":":
        out = git("ls-files", "-z", scope)
    else:
        out = git("ls-tree", "-r", "-z", "--name-only", ref, scope)
    names = (f"{PREFIX}/{p.removeprefix(GIT_PREFIX)}" for p in out.split("\0") if p)
    return sorted(p for p in names if p != GENERATED)


def blobs(ref: str, paths: list[str]) -> dict[str, bytes]:
    """Raw stored bytes of each path at ``ref`` (``:`` = the index), no conversion."""
    spec = "".join(f"{ref}{to_git(p)}\n" if ref == ":" else f"{ref}:{to_git(p)}\n"
                   for p in paths)
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=REPO_ROOT,
                          input=spec.encode("utf-8"), capture_output=True, check=True)
    out: dict[str, bytes] = {}
    data, offset = proc.stdout, 0
    for path in paths:
        header_end = data.index(b"\n", offset)
        header = data[offset:header_end].decode("utf-8")
        if header.endswith("missing"):
            raise BuildError(f"{path} is not in {ref!r}")
        size = int(header.rsplit(" ", 1)[1])
        start = header_end + 1
        out[path] = data[start:start + size]
        offset = start + size + 1
    return out


def check_line_endings(contents: dict[str, bytes]) -> list[str]:
    """Paths whose bytes would break on Kaggle: a CR in a shell or Python file."""
    return [p for p, b in contents.items() if p.endswith(LF_ONLY) and b"\r" in b]


def extract(ref: str, destination: Path, bundle: str | None = None) -> str:
    """Write the tree at ``ref`` (or one of its :data:`BUNDLES`) under ``destination``
    (which becomes thesis_dbp).

    Returns:
        The short commit, or ``"index"``.
    """
    short, full = resolve(ref)
    paths = select(tracked(ref), bundle)
    contents = blobs(ref, paths)
    bad = check_line_endings(contents)
    if bad:
        raise BuildError(f"CR line endings would reach Kaggle in: {bad}")
    for path, data in contents.items():
        target = destination / Path(path).relative_to(PREFIX)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / "COMMIT.txt").write_bytes(f"{short}\n{full}\n".encode())
    return short


def main() -> int:
    """Build the zip. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ref", default="HEAD", help="Commit to package; ':' = the index.")
    parser.add_argument("--bundle", choices=sorted(BUNDLES), default=None,
                        help="Package only this code-only subset.")
    args = parser.parse_args()

    short, full = resolve(args.ref)
    paths = select(tracked(args.ref), args.bundle)
    contents = blobs(args.ref, paths)
    bad = check_line_endings(contents)
    if bad:
        print(f"REFUSED: CR line endings would reach Kaggle in {bad}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, data in contents.items():
            archive.writestr(path, data)
        archive.writestr(GENERATED, f"{short}\n{full}\n")
    tmp.replace(args.out)

    with zipfile.ZipFile(args.out) as archive:
        for path in paths:
            if archive.read(path) != contents[path]:
                raise BuildError(f"{path} in the zip differs from {args.ref}")
    print(f"{args.out}")
    print(f"  {len(paths) + 1} files ({len(paths)} from {args.ref} + COMMIT.txt), "
          f"{args.out.stat().st_size // 1024} KB")
    print(f"  commit {short} {full}")
    print("  every entry is byte-identical to the commit; no CR in any .sh/.py")
    for name in ("env/kaggle_bootstrap.sh", "env/kaggle_run_p2_s3_recovery.py",
                 "scripts/p2_run_shards.py", "scripts/p2_preflight.py"):
        key = f"{PREFIX}/{name}"
        if key in contents:
            print(f"  {name:40s} {hashlib.sha256(contents[key]).hexdigest()[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
