"""Stage the P1 corpus root as links to the attached datasets, and count what is there.

The corpus root is a directory of links -- ``reals``, ``generated``, ``pairs`` -- to
read-only inputs, so every image list can resolve ``<root>/<relative_path>`` without
copying 50,000 files. On Kaggle that is a symlink. On Windows an unprivileged process
cannot create one (``WinError 1314``), so the local dry-run falls back to a directory
junction, which needs no privilege and resolves identically for reading. Discovering
that difference on Kaggle instead of here is the failure this script exists to prevent.

Counts are asserted, not printed: a dataset attached at the wrong version has the right
directory names and the wrong number of files.

Usage:
    python scripts/p2_stage_corpus.py --root /kaggle/working/p1 \\
        --link reals=/kaggle/input/.../p1_reals --expect reals=4100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.utils.logging import setup_logging  # noqa: E402


class StageError(RuntimeError):
    """Raised when the corpus cannot be staged as specified."""


def parse_pairs(items: list[str], flag: str) -> dict[str, str]:
    """Parse repeated ``name=value`` arguments, refusing duplicates."""
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise StageError(f"--{flag} expects name=value, got {item!r}")
        name, value = (part.strip() for part in item.split("=", 1))
        if name in out:
            raise StageError(f"--{flag} {name} given twice")
        out[name] = value
    return out


def make_link(link: Path, target: Path) -> str:
    """Point ``link`` at the directory ``target``, replacing an existing link.

    Returns:
        ``"symlink"`` or ``"junction"`` -- which mechanism was used.

    Raises:
        StageError: If ``link`` exists as a real directory or file. Replacing that would
            delete data rather than a link.
    """
    if link.is_symlink() or _is_junction(link):
        _remove_link(link)
    elif link.exists():
        raise StageError(f"{link} exists and is not a link; refusing to replace it")
    try:
        link.symlink_to(target, target_is_directory=True)
        return "symlink"
    except OSError as exc:
        if os.name != "nt":
            raise StageError(f"cannot link {link} -> {target}: {exc}") from exc
    import _winapi

    _winapi.CreateJunction(str(target), str(link))
    return "junction"


def _is_junction(path: Path) -> bool:
    """Whether ``path`` is a Windows directory junction."""
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _remove_link(path: Path) -> None:
    """Remove a symlink or a junction without touching its target."""
    if _is_junction(path):
        os.rmdir(path)
    else:
        path.unlink()


def count_files(root: Path) -> int:
    """Regular files under ``root``, following the link at ``root`` itself."""
    return sum(1 for item in root.rglob("*") if item.is_file())


def main() -> int:
    """Stage the links and check their counts. Returns 0 on success, 1 otherwise."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="Corpus root to create.")
    parser.add_argument("--link", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=COUNT")
    args = parser.parse_args()
    log = setup_logging("p2_stage_corpus")

    try:
        links = parse_pairs(args.link, "link")
        expect = {k: int(v) for k, v in parse_pairs(args.expect, "expect").items()}
    except (StageError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    unknown = set(expect) - set(links)
    if unknown:
        log.error("--expect names %s, which no --link creates", sorted(unknown))
        return 1

    args.root.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, target in sorted(links.items()):
        source = Path(target)
        if not source.is_dir():
            log.error("%-10s source missing: %s", name, source)
            failures += 1
            continue
        try:
            kind = make_link(args.root / name, source)
        except StageError as exc:
            log.error("%-10s %s", name, exc)
            failures += 1
            continue
        found = count_files(args.root / name)
        want = expect.get(name)
        verdict = "" if want is None else ("ok" if found == want else f"EXPECTED {want}")
        log.info("%-10s %7d file(s) via %-8s %s  -> %s", name, found, kind, verdict, source)
        if want is not None and found != want:
            failures += 1

    if failures:
        log.error("corpus staging FAILED: %d problem(s)", failures)
        return 1
    log.info("corpus staged at %s", args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
