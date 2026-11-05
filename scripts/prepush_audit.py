"""Pre-push audit: what a push of ``--ref`` would publish (user decision 2026-11-05).

Read-only. Every blob reachable from ``--ref``, in every commit -- a push sends history,
not only the tip -- is checked for:

* **(a) dataset images.** Any file with an image or video extension, and any blob whose
  bytes start with an image signature (PNG, JPEG, GIF, WebP/RIFF, TIFF, PDF) whatever its
  name, must be a chart of ours: an SVG, PDF or PNG under ``reports/**/figures/``, at most
  1 MB (CLAUDE.md section 2b). Anything else is a violation.
* **(b) tokens and secrets.** Content patterns (Hugging Face, GitHub, AWS, Slack and
  OpenAI-style tokens, private keys, a Kaggle API key) and risky file names (``.env``,
  ``kaggle.json``, ``*.pem``, ``id_rsa*``, ``*token*``, ``*secret*``, ``credentials*``).
  A match is reported redacted, never printed in full.
* **(c) per-image files**, summarised: every CSV at ``--ref`` whose header has an
  image-id or path column (scores and manifests), grouped by directory, with rows and
  bytes. Reported, not judged: whether they may be published depends on the repository
  being private (CLAUDE.md section 6c).

Exit 0 when (a) and (b) are clean, 1 on any violation of either, 2 when the audit itself
cannot run. Paths are checked with or without a leading ``thesis_dbp/``, so it works
whether git is rooted at thesis_dbp/ or at its parent.

Usage:
    python scripts/prepush_audit.py --ref master [--out audit.json]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
                  ".heic", ".avif", ".svg", ".pdf", ".ico", ".mp4", ".avi", ".mov",
                  ".mkv", ".webm"}
CHART_SUFFIXES = {".svg", ".pdf", ".png"}
MAX_CHART_BYTES = 1_000_000
#: Leading bytes of image (and PDF) formats, so a renamed image is still caught.
SIGNATURES = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpeg", b"GIF87a": "gif",
              b"GIF89a": "gif", b"II*\x00": "tiff", b"MM\x00*": "tiff", b"%PDF-": "pdf"}
SECRET = re.compile(
    rb"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|gho_[A-Za-z0-9]{30,}"
    rb"|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
    rb"|sk-[A-Za-z0-9]{32,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    rb"|\"key\"\s*:\s*\"[0-9a-f]{32}\"")
RISKY_NAME = re.compile(r"(^|/)(\.env(\..*)?|kaggle\.json|[^/]*\.pem|id_rsa[^/]*|[^/]*token[^/]*"
                        r"|[^/]*secret[^/]*|credentials[^/]*)$", re.IGNORECASE)
ID_COLUMNS = {"image_id", "id", "member", "pair_id", "source_image_id", "relative_path",
              "source_path"}


class AuditError(RuntimeError):
    """The audit cannot run (not a repository, unknown ref)."""


def git(repo: Path, *args: str) -> bytes:
    """Run git in ``repo``; raise :class:`AuditError` on failure."""
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    if done.returncode != 0:
        raise AuditError(f"git {' '.join(args)}: {done.stderr.decode(errors='replace').strip()}")
    return done.stdout


def relative(path: str) -> str:
    """The path as seen from thesis_dbp/ (a leading ``thesis_dbp/`` is dropped)."""
    return path.removeprefix("thesis_dbp/")


def signature(data: bytes) -> str | None:
    """The image format ``data`` starts with, if any (WebP: a RIFF ... WEBP header)."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return next((kind for magic, kind in SIGNATURES.items() if data.startswith(magic)), None)


def is_allowed_chart(path: str, size: int) -> bool:
    """CLAUDE.md section 2b: SVG/PDF/PNG under reports/**/figures/, at most 1 MB."""
    parts = PurePosixPath(relative(path)).parts
    return (PurePosixPath(path).suffix.lower() in CHART_SUFFIXES and parts[:1] == ("reports",)
            and "figures" in parts[1:-1] and size <= MAX_CHART_BYTES)


def history(repo: Path, ref: str) -> tuple[int, dict[tuple[str, str], int]]:
    """Every (path, blob) in every commit reachable from ``ref``, with the blob's size."""
    commits = git(repo, "rev-list", ref).decode().split()
    entries: dict[tuple[str, str], int] = {}
    for commit in commits:
        for line in git(repo, "ls-tree", "-r", "-l", "-z", commit).decode(
                "utf-8", "replace").split("\0"):
            if not line:
                continue
            meta, path = line.split("\t", 1)
            _, kind, sha, size = meta.split()
            if kind == "blob":
                entries[(path, sha)] = int(size)
    return len(commits), entries


def blob_reader(repo: Path):
    """A function that returns a blob's bytes (one ``git cat-file`` process for all)."""
    proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)

    def read(sha: str) -> bytes:
        proc.stdin.write(f"{sha}\n".encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode().split()
        if len(header) != 3:
            raise AuditError(f"cat-file: {header}")
        data = proc.stdout.read(int(header[2]))
        proc.stdout.read(1)
        return data

    return read, proc


def audit(repo: Path, ref: str) -> dict[str, Any]:
    """Run checks (a), (b) and (c); returns the record (``ok`` is (a) and (b) clean)."""
    n_commits, entries = history(repo, ref)
    read, proc = blob_reader(repo)
    try:
        images, image_violations, secrets, risky = [], [], [], []
        seen: dict[str, bytes] = {}
        for (path, sha), size in sorted(entries.items()):
            if RISKY_NAME.search(path):
                risky.append(path)
            data = seen.get(sha)
            if data is None:
                data = read(sha)
                seen[sha] = data
            kind = signature(data)
            if PurePosixPath(path).suffix.lower() in IMAGE_SUFFIXES or kind:
                entry = {"path": path, "bytes": size, "signature": kind,
                         "allowed_chart": is_allowed_chart(path, size)}
                images.append(entry)
                if not entry["allowed_chart"]:
                    image_violations.append(entry)
            for match in SECRET.finditer(data):
                secrets.append({"path": path, "match": match.group(0)[:4].decode(
                    "ascii", "replace") + "...(redacted)"})
        per_image = []
        for line in git(repo, "ls-tree", "-r", "-l", "-z", ref).decode(
                "utf-8", "replace").split("\0"):
            if not line or not line.endswith(".csv"):
                continue
            meta, path = line.split("\t", 1)
            _, _, sha, size = meta.split()
            data = seen.get(sha) or read(sha)
            header = next(csv.reader(io.StringIO(data[:20000].decode("utf-8", "replace"))), [])
            columns = {c.strip() for c in header}
            if columns & ID_COLUMNS:
                per_image.append({
                    "path": path, "bytes": int(size), "rows": max(0, data.count(b"\n") - 1),
                    "kind": "scores" if "score" in columns or "score" in path else
                    "manifest/list", "id_columns": sorted(columns & ID_COLUMNS)})
    finally:
        proc.stdin.close()
        proc.wait()
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"files": 0, "bytes": 0, "rows": 0,
                                                            "kinds": set()})
    for item in per_image:
        parts = PurePosixPath(relative(item["path"])).parts
        key = "/".join(parts[:2] if len(parts) > 2 else parts)
        group = groups[key]
        group["files"] += 1
        group["bytes"] += item["bytes"]
        group["rows"] += item["rows"]
        group["kinds"].add(item["kind"])
    return {
        "ref": ref, "commit": git(repo, "rev-parse", ref).decode().strip(),
        "commits": n_commits, "paths": len({p for p, _ in entries}),
        "blobs": len({s for _, s in entries}),
        "a_images": {"files": images, "violations": image_violations},
        "b_secrets": {"content_hits": secrets, "risky_names": risky},
        "c_per_image": {"files": len(per_image), "bytes": sum(i["bytes"] for i in per_image),
                        "rows": sum(i["rows"] for i in per_image),
                        "groups": {k: {**v, "kinds": sorted(v["kinds"])}
                                   for k, v in sorted(groups.items())},
                        "list": per_image},
        "ok": not image_violations and not secrets and not risky,
    }


def report(record: dict[str, Any]) -> None:
    """Print the audit for the user."""
    a, b, c = record["a_images"], record["b_secrets"], record["c_per_image"]
    print(f"pre-push audit of {record['ref']} = {record['commit'][:7]}: {record['commits']} "
          f"commit(s), {record['paths']} paths, {record['blobs']} blobs")
    charts = [f for f in a["files"] if f["allowed_chart"]]
    print(f"(a) images: {len(a['files'])} image file(s), {len(charts)} allowed chart(s), "
          f"{len(a['violations'])} VIOLATION(S)")
    for item in a["violations"]:
        print(f"    VIOLATION {item['path']} ({item['bytes']:,} B, signature "
              f"{item['signature']})")
    print(f"(b) secrets: {len(b['content_hits'])} content hit(s), {len(b['risky_names'])} "
          f"risky name(s)")
    for item in b["content_hits"]:
        print(f"    VIOLATION {item['path']}: {item['match']}")
    for path in b["risky_names"]:
        print(f"    VIOLATION file name {path}")
    print(f"(c) per-image files: {c['files']} CSV(s), {c['bytes']:,} B, {c['rows']:,} rows")
    for key, group in sorted(c["groups"].items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"    {key:48s} {group['files']:4d} files {group['bytes'] / 1e6:8.2f} MB "
              f"{group['rows']:9,d} rows  {','.join(group['kinds'])}")
    print(f"PRE-PUSH AUDIT: {'CLEAN' if record['ok'] else 'FAILED'}")


def main() -> int:
    """Audit; 0 clean, 1 on an image or secret violation, 2 when it cannot run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--repo", type=Path, default=THESIS_DBP)
    parser.add_argument("--out", type=Path, default=None, help="Write the full record here.")
    args = parser.parse_args()
    try:
        record = audit(args.repo, args.ref)
    except AuditError as error:
        print(f"PRE-PUSH AUDIT: CANNOT RUN ({error})")
        return 2
    report(record)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
