"""P5 test panel (docs/P5_PLAN.md sections 1.2 and 3.3).

Phase ``list`` (allowed locally, user 2026-11-23: names and sizes only, no video bytes):
read the central directory of every AIGVDBench zip by exact HTTP range reads, and write
one listing JSON. Per zip it reads the last :data:`TAIL` bytes (the end-of-central-directory
records), the ZIP64 end record if there is one, and then exactly the central directory's
byte range -- never a member's data. The bytes read are recorded per zip, so that claim can
be checked: ``bytes_read == TAIL (or more, for a zip comment) + 56 (ZIP64) + cd_size``.

Only names and uncompressed sizes are kept. The full listing (> 10 MB) is written outside the
repository; the tracked summary carries each zip's ``listing_sha256``.

Usage:
    python scripts/p5_fetch_panel.py --phase list
        --out F:/thesis-2/kaggle_raw/p5_panel/aigvdbench_listing.json   (one command line)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

AIGV_REPO = "AIGVDBench/AIGVDBench"
AIGV_REVISION = "bf48acaa4920990af3dd2a511ea88e63354df305"   # docs/P5_PLAN.md section 1.2

#: The first read at the end of a zip: the EOCD record (22 B), the ZIP64 locator (20 B) and a
#: little of the central directory before them. Grown only if a zip comment hides the EOCD.
TAIL = 128
MAX_TAIL = 22 + 0xFFFF + 20
EOCD_SIG, Z64_LOCATOR_SIG, Z64_EOCD_SIG, CD_SIG = (b"PK\x05\x06", b"PK\x06\x07", b"PK\x06\x06",
                                                    b"PK\x01\x02")
CD_FIXED = struct.Struct("<4sHHHHHHIIIHHHHHII")      # 46 bytes


class ZipListError(RuntimeError):
    """A zip whose directory cannot be read as the format says."""


def read_range(handle: BinaryIO, start: int, length: int) -> bytes:
    """Exactly ``length`` bytes at ``start``."""
    handle.seek(start)
    data = handle.read(length)
    if len(data) != length:
        raise ZipListError(f"short read at {start}: {len(data)} of {length} bytes")
    return data


def _zip64_extra(extra: bytes, usize: int, csize: int, offset: int) -> tuple[int, int, int]:
    """Replace the 0xFFFFFFFF fields from the ZIP64 extra field (id 0x0001), in the order
    the format gives: uncompressed size, compressed size, local header offset."""
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4:pos + 4 + size]
        if tag == 0x0001:
            values = list(struct.unpack_from(f"<{len(body) // 8}Q", body))
            fields = [usize, csize, offset]
            for i, value in enumerate(fields):
                if value == 0xFFFFFFFF:
                    if not values:
                        raise ZipListError("ZIP64 extra field shorter than its 0xFFFFFFFF fields")
                    fields[i] = values.pop(0)
            return fields[0], fields[1], fields[2]
        pos += 4 + size
    return usize, csize, offset


def central_directory(handle: BinaryIO, size: int) -> dict[str, Any]:
    """List a zip from its directory alone: members (name, sizes, method) and the bytes read.

    Raises:
        ZipListError: no EOCD record, a multi-disk zip, or a directory that does not parse.
    """
    bytes_read = 0
    tail_len = min(TAIL, size)
    while True:
        tail = read_range(handle, size - tail_len, tail_len)
        bytes_read += tail_len
        at = tail.rfind(EOCD_SIG)
        if at >= 0 or tail_len >= min(MAX_TAIL, size):
            break
        tail_len = min(MAX_TAIL, size)
    if at < 0:
        raise ZipListError("no end-of-central-directory record")
    disk, cd_disk, _, n_total, cd_size, cd_offset, _ = struct.unpack_from("<HHHHIIH", tail, at + 4)
    if disk or cd_disk:
        raise ZipListError("multi-disk zip")
    zip64 = n_total == 0xFFFF or 0xFFFFFFFF in (cd_size, cd_offset)
    if zip64:
        loc = tail[at - 20:at] if at >= 20 else b""
        if loc[:4] != Z64_LOCATOR_SIG:
            raise ZipListError("ZIP64 fields without a ZIP64 locator")
        (record_offset,) = struct.unpack_from("<Q", loc, 8)
        record = read_range(handle, record_offset, 56)
        bytes_read += 56
        if record[:4] != Z64_EOCD_SIG:
            raise ZipListError("bad ZIP64 end-of-central-directory record")
        n_total, cd_size, cd_offset = struct.unpack_from("<QQQ", record, 32)
    directory = read_range(handle, cd_offset, cd_size)
    bytes_read += cd_size
    members, pos = [], 0
    while pos < len(directory):
        (sig, _, _, flags, method, _, _, crc, csize, usize, name_len, extra_len, comment_len,
         _, _, _, offset) = CD_FIXED.unpack_from(directory, pos)
        if sig != CD_SIG:
            raise ZipListError(f"bad central-directory entry at {pos}")
        start = pos + CD_FIXED.size
        raw = directory[start:start + name_len]
        name = raw.decode("utf-8" if flags & 0x800 else "cp437")
        extra = directory[start + name_len:start + name_len + extra_len]
        usize, csize, offset = _zip64_extra(extra, usize, csize, offset)
        members.append({"name": name, "size": usize, "compressed": csize, "method": method,
                        "crc32": f"{crc:08x}", "offset": offset})
        pos = start + name_len + extra_len + comment_len
    if len(members) != n_total:
        raise ZipListError(f"directory lists {len(members)} entries, the EOCD says {n_total}")
    # the end-of-zip read must lie inside the directory and end records, never member data
    tail_in_directory = size - tail_len >= cd_offset
    return {"zip64": zip64, "cd_offset": cd_offset, "cd_size": cd_size,
            "bytes_read": bytes_read, "tail_in_directory": tail_in_directory,
            "members": members}


class RangeReader:
    """seek/read over a remote file where every read is ONE exact HTTP range request
    (``bytes=start-end``), with no read-ahead or cache; downloaded bytes are counted."""

    def __init__(self, remote: Any) -> None:
        self.remote, self.pos, self.downloaded, self.requests = remote, 0, 0, 0

    def seek(self, pos: int) -> None:
        self.pos = pos

    def read(self, length: int) -> bytes:
        data = self.remote._fetch_range(self.pos, self.pos + length)
        self.downloaded += len(data)
        self.requests += 1
        self.pos += len(data)
        return data


def phase_list(args: argparse.Namespace) -> int:
    """Every zip of the pinned AIGVDBench revision, listed from its directory."""
    from huggingface_hub import HfApi, HfFileSystem

    from dbp.utils.hf import get_hf_token

    token = get_hf_token()
    api = HfApi(token=token)
    files = [f for f in api.list_repo_tree(AIGV_REPO, repo_type="dataset",
                                           revision=AIGV_REVISION, recursive=True)
             if getattr(f, "size", None) is not None]
    fs = HfFileSystem(token=token)
    out: dict[str, Any] = {"repo": AIGV_REPO, "revision": AIGV_REVISION,
                           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "method": "exact range reads of each zip's end records and central "
                                     "directory; no member data",
                           "files": {f.path: f.size for f in files}, "zips": {}}
    for f in sorted(files, key=lambda x: x.path):
        if not f.path.endswith(".zip"):
            continue
        path = f"datasets/{AIGV_REPO}@{AIGV_REVISION}/{f.path}"
        with fs.open(path, "rb", block_size=1, cache_type="none") as remote:
            reader = RangeReader(remote)
            listing = central_directory(reader, f.size)
        if reader.downloaded != listing["bytes_read"]:
            raise ZipListError(f"{f.path}: downloaded {reader.downloaded} B, the reader asked "
                               f"for {listing['bytes_read']} B")
        if not listing["tail_in_directory"]:
            raise ZipListError(f"{f.path}: the end-of-zip read reached before the directory")
        files_only = [[m["name"], m["size"]] for m in listing["members"]
                      if not m["name"].endswith("/")]
        out["zips"][f.path] = {
            "zip_bytes": f.size, "n_members": len(files_only), "zip64": listing["zip64"],
            "cd_size": listing["cd_size"], "bytes_read": listing["bytes_read"],
            "range_requests": reader.requests, "bytes_downloaded": reader.downloaded,
            "extensions": sorted({Path(n).suffix.lower() for n, _ in files_only}),
            "listing_sha256": listing_sha256(files_only),
            "members": files_only}                      # [name, uncompressed size] only
        print(f"{f.path}: {len(files_only)} members, directory {listing['cd_size']:,} B, "
              f"read {listing['bytes_read']:,} B", flush=True)
    write_atomic(args.out, json.dumps(out, separators=(",", ":")) + "\n")
    summary = {k: v for k, v in out.items() if k != "zips"}
    summary["zips"] = {z: {k: v for k, v in info.items() if k != "members"}
                       for z, info in out["zips"].items()}
    summary["listing_file"] = args.out.name
    summary["bytes_read_total"] = sum(z["bytes_read"] for z in out["zips"].values())
    write_atomic(args.summary, json.dumps(summary, indent=1) + "\n")
    print(f"wrote {args.out} and {args.summary}: {len(out['zips'])} zips, "
          f"{summary['bytes_read_total']:,} B read in total", flush=True)
    return 0


def listing_sha256(members: list[list[Any]]) -> str:
    """sha256 over ``name<TAB>size<LF>`` lines in name order: one zip's listing identity."""
    text = "".join(f"{name}\t{size}\n" for name, size in sorted(members))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_atomic(path: Path, text: str) -> None:
    """Temp file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", required=True, choices=["list"])
    parser.add_argument("--out", type=Path, required=True,
                        help="the full listing (names + sizes): OUTSIDE the repository, it is "
                             "> 10 MB (F:/thesis-2/kaggle_raw/p5_panel/ locally)")
    parser.add_argument("--summary", type=Path,
                        default=THESIS_DBP / "reports" / "p5_panel"
                        / "aigvdbench_listing_summary.json",
                        help="per-zip counts, bytes read and listing sha256 (tracked)")
    args = parser.parse_args()
    return phase_list(args)


if __name__ == "__main__":
    raise SystemExit(main())
