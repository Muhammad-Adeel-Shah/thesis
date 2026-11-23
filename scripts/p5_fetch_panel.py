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
import csv
import datetime as dt
import hashlib
import json
import re
import shutil
import struct
import sys
import tarfile
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, BinaryIO

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256  # noqa: E402

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


class LocalRemote:
    """A local file with the remote file's ``_fetch_range`` (dry-run sources, tests)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _fetch_range(self, start: int, end: int) -> bytes:
        with self.path.open("rb") as handle:
            handle.seek(start)
            return handle.read(end - start)


class Source:
    """A Hub dataset at a pinned revision (``spec == "hub"``), or a local directory laid
    out as the repository (the dry-run fixture). Range reads, file sizes, small downloads."""

    def __init__(self, spec: str, repo: str, revision: str) -> None:
        self.spec, self.repo, self.revision = spec, repo, revision
        self.hub = spec == "hub"
        if self.hub:
            from huggingface_hub import HfApi, HfFileSystem

            from dbp.utils.hf import get_hf_token

            self.token = get_hf_token()
            self.api = HfApi(token=self.token)
            self.fs = HfFileSystem(token=self.token)
        else:
            self.root = Path(spec)
            if not self.root.is_dir():
                raise SystemExit(f"--source {spec}: not 'hub' and not a directory")

    def files(self) -> dict[str, int]:
        """Every file of the repository at the revision, with its size."""
        if self.hub:
            return {f.path: f.size for f in self.api.list_repo_tree(
                self.repo, repo_type="dataset", revision=self.revision, recursive=True)
                if getattr(f, "size", None) is not None}
        return {p.relative_to(self.root).as_posix(): p.stat().st_size
                for p in sorted(self.root.rglob("*")) if p.is_file()}

    def remote(self, path: str) -> Any:
        """An object with ``_fetch_range(start, end)`` for one file."""
        if self.hub:
            return self.fs.open(f"datasets/{self.repo}@{self.revision}/{path}", "rb",
                                block_size=1, cache_type="none")
        return LocalRemote(self.root / path)

    def download(self, path: str, cache: Path) -> Path:
        """The whole file, locally (small files and the AEGIS tar only)."""
        if not self.hub:
            return self.root / path
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo, path, repo_type="dataset",
                                    revision=self.revision, token=self.token,
                                    cache_dir=str(cache)))


def list_zips(source: Source, full: bool) -> dict[str, Any]:
    """Every zip of the source, listed from its directory (module docstring)."""
    files = source.files()
    out: dict[str, Any] = {"repo": source.repo, "revision": source.revision,
                           "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                           "method": "exact range reads of each zip's end records and central "
                                     "directory; no member data",
                           "files": files, "zips": {}}
    for path, size in sorted(files.items()):
        if not path.endswith(".zip"):
            continue
        remote = source.remote(path)
        reader = RangeReader(remote)
        listing = central_directory(reader, size)
        if hasattr(remote, "close"):
            remote.close()
        if reader.downloaded != listing["bytes_read"]:
            raise ZipListError(f"{path}: downloaded {reader.downloaded} B, the reader asked "
                               f"for {listing['bytes_read']} B")
        if not listing["tail_in_directory"]:
            raise ZipListError(f"{path}: the end-of-zip read reached before the directory")
        members = [m for m in listing["members"] if not m["name"].endswith("/")]
        names_sizes = [[m["name"], m["size"]] for m in members]
        out["zips"][path] = {
            "zip_bytes": size, "n_members": len(members), "zip64": listing["zip64"],
            "cd_size": listing["cd_size"], "bytes_read": listing["bytes_read"],
            "range_requests": reader.requests, "bytes_downloaded": reader.downloaded,
            "extensions": sorted({Path(n).suffix.lower() for n, _ in names_sizes}),
            "listing_sha256": listing_sha256(names_sizes),
            # [name, size] only, unless --full (Kaggle: offsets and CRCs, kept in /tmp)
            "members": members if full else names_sizes}
        print(f"{path}: {len(members)} members, directory {listing['cd_size']:,} B, "
              f"read {listing['bytes_read']:,} B", flush=True)
    return out


def phase_list(args: argparse.Namespace) -> int:
    """List every zip; with ``--reference``, refuse unless each zip's listing equals the
    committed one (CLAUDE.md 12.2: checked against the committed reference)."""
    source = Source(args.source, AIGV_REPO, AIGV_REVISION)
    out = list_zips(source, full=args.full)
    write_atomic(args.out, json.dumps(out, separators=(",", ":")) + "\n")
    summary = {k: v for k, v in out.items() if k != "zips"}
    summary["zips"] = {z: {k: v for k, v in info.items() if k != "members"}
                       for z, info in out["zips"].items()}
    summary["listing_file"] = args.out.name
    summary["bytes_read_total"] = sum(z["bytes_read"] for z in out["zips"].values())
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        want = {z: v["listing_sha256"] for z, v in reference["zips"].items()}
        have = {z: v["listing_sha256"] for z, v in summary["zips"].items()}
        if want != have:
            bad = sorted(z for z in set(want) | set(have) if want.get(z) != have.get(z))
            raise SystemExit(f"the zip listings differ from {args.reference}: {bad}")
        print(f"listings equal the committed reference ({len(want)} zips)", flush=True)
    else:
        write_atomic(args.summary, json.dumps(summary, indent=1) + "\n")
    print(f"wrote {args.out}: {len(out['zips'])} zips, {summary['bytes_read_total']:,} B read",
          flush=True)
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


# ------------------------------------------------------------------------- the draw ---

SOURCE_ID = re.compile(r"^(?P<source>.+)_(?P<k>\d+)_(?P<a>\d+)to(?P<b>\d+)\.mp4$")
ID_KEYS = ("Video_id", "video_id", "videoid", "id")


def source_of(name: str) -> str:
    """The source video of an AIGVDBench real-paired name (``<source>_<k>_<a>to<b>.mp4``);
    a name that does not follow the pattern is its own group."""
    match = SOURCE_ID.match(Path(name).name)
    return match.group("source") if match else Path(name).name


def rank(salt: str, key: str) -> str:
    """The pre-declared order: ``sha256(salt + key)``."""
    return hashlib.sha256((salt + key).encode("utf-8")).hexdigest()


def read_test_ids(path: Path) -> tuple[list[str], str]:
    """The ids of ``Split/test.jsonl`` (basenames, ``.mp4`` added if absent), and the key
    they were read from. Stops if no known id key is present."""
    ids, key = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if key is None:
            key = next((k for k in ID_KEYS if k in row), None)
            if key is None:
                raise SystemExit(f"{path}: no id key among {ID_KEYS}; keys are {sorted(row)}")
        value = Path(str(row[key])).name
        ids.append(value if value.endswith(".mp4") else value + ".mp4")
    return ids, str(key)


def zip_kind(path: str) -> str:
    """``real`` | ``open`` | ``closed`` from the repository layout."""
    if "/Real/" in f"/{path}":
        return "real"
    if "/ClosedSource/" in f"/{path}":
        return "closed"
    if "/OpenSource/" in f"/{path}":
        return "open"
    raise SystemExit(f"{path}: not under Real/, OpenSource/ or ClosedSource/")


def draw_panel(names_by_zip: dict[str, list[str]], test_ids: list[str],
               params: dict[str, Any]) -> dict[str, Any]:
    """The pre-declared AIGVDBench draw (user's principles of 2026-11-23; docs/P5_PLAN.md).

    1. The universe is the names in both ``Split/test.jsonl`` and Real.zip, grouped by
       source video; groups are ordered by ``rank(salt, source)``.
    2. Whole groups fill, in order: the calibration pool (k-shot reals), the closed-source
       real pool, and the open-source pool. A name -- its real and every fake made from
       it -- is in exactly one pool, and no source video spans two pools.
    3. Each open-source zip ranks the open pool by ``rank(salt, zip + ":" + name)``; the
       fetch keeps the first ``per_slice`` names whose real AND fake are eligible
       (content-matched pairs).
    4. The closed-source zips share one order, ``rank(salt, "closed:" + name)``; each keeps
       its first ``per_slice`` eligible fakes, scored against the closed-source real pool.
    """
    salt = params["salt"]
    basenames = {z: [Path(n).name for n in names] for z, names in names_by_zip.items()}
    real_zips = [z for z in basenames if zip_kind(z) == "real"]
    if len(real_zips) != 1:
        raise SystemExit(f"expected one Real zip, found {real_zips}")
    real_names = set(basenames[real_zips[0]])
    universe = sorted(set(test_ids) & real_names)
    groups: dict[str, list[str]] = {}
    for name in universe:
        groups.setdefault(source_of(name), []).append(name)
    order = sorted(groups, key=lambda s: rank(salt, s))
    pools: dict[str, list[str]] = {"calibration": [], "closed_reals": [], "open": []}
    targets = [("calibration", params["calibration"]), ("closed_reals", params["closed_reals"]),
               ("open", params["open_pool"])]
    pool_of_source: dict[str, str] = {}
    current = 0
    for source in order:
        while current < len(targets) and len(pools[targets[current][0]]) >= targets[current][1]:
            current += 1
        if current == len(targets):
            break
        pool = targets[current][0]
        pools[pool].extend(sorted(groups[source]))
        pool_of_source[source] = pool
    short = {p: n for p, n in targets if len(pools[p]) < n}
    if short:
        raise SystemExit(f"the universe ({len(universe)} names) cannot fill the pools: {short}")
    slices: dict[str, dict[str, Any]] = {}
    open_pool = pools["open"]
    closed_union = sorted({n for z, ns in basenames.items() if zip_kind(z) == "closed"
                           for n in ns})
    closed_order = sorted(closed_union, key=lambda n: rank(salt, "closed:" + n))
    for z in sorted(basenames):
        kind = zip_kind(z)
        if kind == "open":
            present = set(basenames[z])
            ranked = sorted((n for n in open_pool if n in present),
                            key=lambda n: rank(salt, f"{z}:{n}"))
            slices[z] = {"kind": "open", "candidates": ranked}
        elif kind == "closed":
            present = set(basenames[z])
            slices[z] = {"kind": "closed", "candidates": [n for n in closed_order if n in present]}
    return {"universe": len(universe), "test_ids": len(test_ids),
            "test_ids_in_real": len(universe), "source_groups": len(groups),
            "pools": pools, "pool_sizes": {p: len(v) for p, v in pools.items()},
            "slices": slices, "params": params}


def phase_draw(args: argparse.Namespace, expect: dict[str, Any]) -> int:
    """The draw from the verified listing and ``Split/test.jsonl``."""
    listing = json.loads(args.listing.read_text(encoding="utf-8"))
    source = Source(args.source, AIGV_REPO, AIGV_REVISION)
    test_path = source.download(expect["aigv"]["test_split"], args.cache)
    test_ids, key = read_test_ids(test_path)
    names = {z: [m["name"] if isinstance(m, dict) else m[0] for m in v["members"]]
             for z, v in listing["zips"].items()}
    draw = draw_panel(names, test_ids, expect["draw"])
    draw.update({"test_split": expect["aigv"]["test_split"], "test_id_key": key,
                 "test_split_sha256": file_sha256(test_path)})
    write_atomic(args.reports / "p5_panel_draw.json", json.dumps(draw, indent=1) + "\n")
    print(json.dumps({k: draw[k] for k in ("test_ids", "test_ids_in_real", "source_groups",
                                           "pool_sizes", "test_id_key")}), flush=True)
    return 0


# ------------------------------------------------------------------------ the fetch ---

LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")       # 30 bytes
LOCAL_SIG = b"PK\x03\x04"


def read_member(remote: Any, member: dict[str, Any]) -> bytes:
    """One zip member's bytes by range reads of its local header and data; the CRC-32 and
    size are checked against the central directory.

    Raises:
        ZipListError: a bad local header, an unsupported method, or a CRC/size mismatch.
    """
    reader = RangeReader(remote)
    reader.seek(member["offset"])
    head = reader.read(LOCAL_HEADER.size)
    sig, _, flags, method, _, _, _, _, _, name_len, extra_len = LOCAL_HEADER.unpack(head)
    if sig != LOCAL_SIG:
        raise ZipListError(f"{member['name']}: bad local header")
    if flags & 0x1:
        raise ZipListError(f"{member['name']}: encrypted member")
    reader.seek(member["offset"] + LOCAL_HEADER.size + name_len + extra_len)
    data = reader.read(member["compressed"])
    if method == 8:
        inflater = zlib.decompressobj(-15)
        data = inflater.decompress(data) + inflater.flush()
    elif method != 0:
        raise ZipListError(f"{member['name']}: compression method {method} is not supported")
    if len(data) != member["size"] or f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != member["crc32"]:
        raise ZipListError(f"{member['name']}: size or CRC-32 differs from the directory")
    return data


def examine(path: Path, clip_key: str, clip: dict[str, Any]) -> dict[str, Any]:
    """Probe, the pre-declared exclusions, the seeded window, and the keyframe pHashes of
    the centre crop (the clip protocol of src/dbp/data/video.py, unchanged)."""
    from dbp.data.phash import phash_from_array, phash_hex
    from dbp.data.video import (
        ShortClipError,
        VideoError,
        centre_crop_clip,
        decode_window,
        exclusion,
        is_static,
        probe,
        window_start,
    )

    record: dict[str, Any] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
    try:
        info = probe(path)
    except VideoError as exc:
        return {**record, "eligible": False, "exclusion": "decode_error", "detail": str(exc)}
    record.update({k: info[k] for k in ("codec", "profile", "pix_fmt", "width", "height", "fps",
                                         "fps_fraction", "frames_reported", "duration_s",
                                         "bit_rate", "rotation", "format")})
    reason = exclusion(info, clip["min_short_side"])
    if reason:
        return {**record, "eligible": False, "exclusion": reason}
    start = window_start(clip_key, int(info["frames_reported"]), clip["frames"])
    record["window_start"] = start
    try:
        frames = centre_crop_clip(decode_window(path, start, clip["frames"]),
                                  clip["min_short_side"])
    except ShortClipError:
        return {**record, "eligible": False, "exclusion": "too_few_frames"}
    except VideoError as exc:
        return {**record, "eligible": False, "exclusion": "decode_error", "detail": str(exc)}
    if is_static(frames):
        return {**record, "eligible": False, "exclusion": "static"}
    keys = [k for k in clip["phash_frames"] if k < len(frames)]
    record.update({"eligible": True, "exclusion": "",
                   "phash": [phash_hex(phash_from_array(frames[k])) for k in keys]})
    return record


class Fetcher:
    """Fetches, examines and records clips; resumable (one JSON line per clip); a clip is
    fetched once even when several slices need it (the open-source reals)."""

    def __init__(self, args: argparse.Namespace, expect: dict[str, Any],
                 members: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.args, self.expect, self.members = args, expect, members
        self.source = Source(args.source, AIGV_REPO, AIGV_REVISION)
        self.records_path = args.reports / "p5_panel_clips.jsonl"
        self.lock = threading.Lock()
        self.done: dict[tuple[str, str], dict[str, Any]] = {}
        self.pending: dict[tuple[str, str], threading.Event] = {}
        if self.records_path.is_file():
            for line in self.records_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.done[(row["zip"], row["name"])] = row

    def target(self, zip_path: str, name: str) -> Path:
        return self.args.out_root / Path(zip_path).with_suffix("").as_posix() / name

    def get(self, zip_path: str, name: str, role: str) -> dict[str, Any]:
        """The record of one clip, fetching and examining it if needed."""
        key = (zip_path, name)
        with self.lock:
            if key in self.done:
                return self.done[key]
            event = self.pending.get(key)
            owner = event is None
            if owner:
                event = self.pending[key] = threading.Event()
        if not owner:
            event.wait()
            with self.lock:
                return self.done[key]
        try:
            row = self._fetch(zip_path, name, role)
            with self.lock:
                self.done[key] = row
                with self.records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            return row
        finally:
            event.set()

    def _fetch(self, zip_path: str, name: str, role: str) -> dict[str, Any]:
        member = self.members[zip_path][name]
        remote = self.source.remote(zip_path)
        try:
            data = read_member(remote, member)
        finally:
            if hasattr(remote, "close"):
                remote.close()
        path = self.target(zip_path, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        row = {"view": "aigvdbench", "zip": zip_path, "name": name, "role": role,
               "path": str(path.relative_to(self.args.out_root)),
               **examine(path, f"{zip_path}:{name}", self.expect["clip"])}
        if not row["eligible"]:
            path.unlink()                         # only eligible clips are kept on disk
        return row


def phase_fetch(args: argparse.Namespace, expect: dict[str, Any]) -> int:
    """Pools, then slices, in the draw's order; per slice until ``per_slice`` eligible, the
    candidates run out, or ``stop_after_ineligible`` consecutive fakes are ineligible."""
    listing = json.loads(args.listing.read_text(encoding="utf-8"))
    draw = json.loads((args.reports / "p5_panel_draw.json").read_text(encoding="utf-8"))
    members = {z: {Path(m["name"]).name: m for m in v["members"]}
               for z, v in listing["zips"].items()}
    real_zip = next(z for z in members if zip_kind(z) == "real")
    fetcher = Fetcher(args, expect, members)
    params = expect["draw"]
    started = time.time()
    with ThreadPoolExecutor(int(expect["fetch_workers"])) as pool:
        rows = list(pool.map(lambda n: fetcher.get(real_zip, n, "real"),
                             draw["pools"]["calibration"] + draw["pools"]["closed_reals"]))
    print(f"pool reals: {sum(r['eligible'] for r in rows)} eligible of {len(rows)} "
          f"({time.time() - started:.0f} s)", flush=True)

    def run_slice(zip_path: str) -> dict[str, Any]:
        spec = draw["slices"][zip_path]
        kept, tried, streak = [], 0, 0
        for name in spec["candidates"]:
            if len(kept) >= params["per_slice"] or streak >= params["stop_after_ineligible"]:
                break
            tried += 1
            fake = fetcher.get(zip_path, name, "fake")
            if not fake["eligible"]:
                streak += 1
                continue
            streak = 0
            if spec["kind"] == "open":
                real = fetcher.get(real_zip, name, "real")
                if not real["eligible"]:
                    continue
            kept.append(name)
        stopped = streak >= params["stop_after_ineligible"]
        status = ("complete" if len(kept) >= params["per_slice"] else
                  "stopped: consecutive ineligible fakes" if stopped
                  else "short: candidates exhausted")
        result = {"zip": zip_path, "kind": spec["kind"], "kept": kept, "tried": tried,
                  "status": status}
        print(f"{zip_path}: {len(kept)} kept of {tried} tried -- {status}", flush=True)
        return result

    with ThreadPoolExecutor(int(expect["slice_workers"])) as pool:
        results = list(pool.map(run_slice, sorted(draw["slices"])))
    write_atomic(args.reports / "p5_panel_slices.json",
                 json.dumps({"slices": {r["zip"]: r for r in results},
                             "seconds": round(time.time() - started, 1)}, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------- AEGIS ---

def phase_aegis(args: argparse.Namespace, expect: dict[str, Any]) -> int:
    """AEGIS hard test (secondary, clean only): the mapping CSV and the video tar, whole;
    every video examined. Labels are joined at the freeze, from the archived mapping."""
    spec = expect["aegis"]
    source = Source(args.aegis_source, spec["repo"], spec["revision"])
    files = source.files()
    pick = {key: one_file(files, spec[key]) for key in ("tar", "mapping")}
    mapping = source.download(pick["mapping"], args.cache)
    shutil.copyfile(mapping, args.reports / "aegis_mapping.csv")
    tar_path = source.download(pick["tar"], args.cache)
    out_dir = args.out_root / "AEGIS"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with tarfile.open(tar_path) as archive:
        for info in archive.getmembers():
            if not info.isfile() or not info.name.lower().endswith(".mp4"):
                continue
            relative = Path(info.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise SystemExit(f"{info.name}: unsafe path in the AEGIS tar")
            target = out_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(info) as handle, target.open("wb") as dest:
                shutil.copyfileobj(handle, dest)
            row = {"view": "aegis", "zip": pick["tar"], "name": relative.as_posix(),
                   "role": "unlabelled", "path": str(target.relative_to(args.out_root)),
                   **examine(target, f"aegis:{relative.as_posix()}", expect["clip"])}
            rows.append(row)
    write_atomic(args.reports / "p5_panel_aegis.jsonl",
                 "".join(json.dumps(r) + "\n" for r in rows))
    print(f"AEGIS: {len(rows)} videos, {sum(r['eligible'] for r in rows)} eligible; mapping "
          f"{pick['mapping']} archived", flush=True)
    if not args.keep_tar and source.hub:
        tar_path.unlink()
    return 0


def one_file(files: dict[str, int], basename: str) -> str:
    """The one repository file with this basename."""
    hits = [p for p in files if Path(p).name == basename]
    if len(hits) != 1:
        raise SystemExit(f"expected one {basename} in the repository, found {hits}")
    return hits[0]


# --------------------------------------------------------------- the final tables ---

NUMERIC_FEATURES = ("fps", "width", "height", "duration_s", "bit_rate", "frames_reported")
CATEGORICAL_FEATURES = ("codec", "profile", "pix_fmt")


def separability_auc(labels: list[int], values: list[Any], categorical: bool) -> float | None:
    """A metadata-only AUC, direction-free (max(AUC, 1 - AUC)). A categorical feature is
    scored by each value's fake share in the slice, as the best single-value rule would."""
    from dbp.eval.metrics import roc_auc

    pairs = [(y, v) for y, v in zip(labels, values, strict=True) if v is not None and v != ""]
    if len({y for y, _ in pairs}) < 2:
        return None
    ys = [y for y, _ in pairs]
    if categorical:
        share: dict[Any, list[int]] = {}
        for y, v in pairs:
            share.setdefault(v, []).append(y)
        scores = [sum(share[v]) / len(share[v]) for _, v in pairs]
    else:
        scores = [float(v) for _, v in pairs]
    auc = roc_auc(ys, scores)
    return round(max(auc, 1.0 - auc), 4)


def near_duplicates(rows: list[dict[str, Any]], bits: int) -> list[tuple[int, int]]:
    """Index pairs whose keyframe pHashes are ALL within ``bits`` (vectorised, exact)."""
    import numpy as np

    hashes = np.array([[int(h, 16) for h in r["phash"]] for r in rows], dtype=np.uint64)
    pairs = []
    for i in range(len(rows) - 1):                 # the last row has nothing after it
        xor = hashes[i + 1:] ^ hashes[i]
        counts = np.unpackbits(xor.view(np.uint8).reshape(len(xor), -1, 8), axis=2).sum(axis=2)
        close = np.nonzero((counts <= bits).all(axis=1))[0]
        pairs += [(i, i + 1 + int(j)) for j in close]
    return pairs


def phase_finalize(args: argparse.Namespace, expect: dict[str, Any]) -> int:
    """Manifest, duplicates, the metadata gate, the panel digest and the archive. Never fails:
    a problem is written into the summary (CLAUDE.md 12.5)."""
    summary: dict[str, Any] = {"stage": "p5_panel", "generated_utc":
                               dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        summary.update(finalize(args, expect))
    except Exception as exc:                       # recorded, never swallowed
        summary["error"] = f"{type(exc).__name__}: {exc}"
    write_atomic(args.reports / "p5_panel_summary.json", json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary.get(k) for k in ("error", "panel_digest", "counts",
                                                  "gate_flags")}, indent=1), flush=True)
    if args.zip:
        try:
            archive = shutil.make_archive(str(args.zip), "zip", args.reports.parent,
                                          args.reports.name)
            print("archive:", archive, flush=True)
        except OSError as exc:
            print(f"ARCHIVE FAILED: {exc}", flush=True)
    return 0


def finalize(args: argparse.Namespace, expect: dict[str, Any]) -> dict[str, Any]:
    draw = json.loads((args.reports / "p5_panel_draw.json").read_text(encoding="utf-8"))
    slices = json.loads((args.reports / "p5_panel_slices.json").read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in (args.reports / "p5_panel_clips.jsonl")
            .read_text(encoding="utf-8").splitlines() if x.strip()]
    aegis_path = args.reports / "p5_panel_aegis.jsonl"
    aegis = [json.loads(x) for x in aegis_path.read_text(encoding="utf-8").splitlines()
             if x.strip()] if aegis_path.is_file() else []
    by_key = {(r["zip"], r["name"]): r for r in rows}
    real_zip = next(z for z, _ in by_key if zip_kind(z) == "real")
    manifest: list[dict[str, Any]] = []
    for pool in ("calibration", "closed_reals"):
        for name in draw["pools"][pool]:
            r = by_key.get((real_zip, name))
            if r and r["eligible"]:
                manifest.append({**r, "pool": pool, "slice": ""})
    for zip_path, result in slices["slices"].items():
        for name in result["kept"]:
            manifest.append({**by_key[(zip_path, name)], "pool": "test", "slice": zip_path})
            if result["kind"] == "open":
                manifest.append({**by_key[(real_zip, name)], "pool": "test",
                                 "slice": zip_path})
    for r in aegis:
        if r["eligible"]:
            manifest.append({**r, "pool": "aegis", "slice": "AEGIS"})
    # duplicates: exact (sha256) and near (every keyframe pHash within the threshold)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for m in manifest:
        unique.setdefault((m["zip"], m["name"]), m)
    clips = list(unique.values())
    by_sha: dict[str, list[str]] = {}
    for c in clips:
        by_sha.setdefault(c["sha256"], []).append(f"{c['zip']}:{c['name']}")
    exact = [v for v in by_sha.values() if len(v) > 1]
    near = []
    for i, j in near_duplicates(clips, int(expect["phash_bits"])):
        a, b = clips[i], clips[j]
        own_pair = (Path(a["name"]).name == Path(b["name"]).name
                    and {zip_kind(a["zip"]), zip_kind(b["zip"])} == {"real", "open"})
        near.append({"a": f"{a['zip']}:{a['name']}", "b": f"{b['zip']}:{b['name']}",
                     "own_pair": own_pair})
    # the metadata gate, per slice: its fakes against its reals
    gate, flags = {}, []
    closed_reals = [m for m in manifest if m["pool"] == "closed_reals"]
    for zip_path, result in slices["slices"].items():
        fakes = [by_key[(zip_path, n)] for n in result["kept"]]
        reals = ([by_key[(real_zip, n)] for n in result["kept"]] if result["kind"] == "open"
                 else closed_reals)
        labels = [1] * len(fakes) + [0] * len(reals)
        table = {}
        for feature in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            values = [c.get(feature) for c in fakes + reals]
            table[feature] = separability_auc(labels, values, feature in CATEGORICAL_FEATURES)
            limit = float(expect["gate"]["auc_threshold"])
            if table[feature] is not None and table[feature] > limit:
                flags.append(f"{zip_path}:{feature}={table[feature]}")
        gate[zip_path] = {"fakes": len(fakes), "reals": len(reals), "auc": table}
    lines = sorted(f"{m['view']},{m['pool']},{m['slice']},{m['role']},{m['zip']},{m['name']},"
                   f"{m['sha256']}" for m in manifest)
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    columns = ["view", "pool", "slice", "role", "zip", "name", "path", "sha256", "bytes",
               "codec", "profile", "pix_fmt", "width", "height", "fps", "fps_fraction",
               "frames_reported", "duration_s", "bit_rate", "window_start", "phash"]
    with (args.reports / "p5_panel_manifest.csv").open("w", encoding="utf-8",
                                                       newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        for m in sorted(manifest, key=lambda m: (m["pool"], m["slice"], m["role"], m["name"])):
            writer.writerow({**m, "phash": " ".join(m.get("phash", []))})
    exclusions: dict[str, dict[str, int]] = {}
    for r in rows + aegis:
        if not r["eligible"]:
            per = exclusions.setdefault(r["zip"], {})
            per[r["exclusion"]] = per.get(r["exclusion"], 0) + 1
    counts = {"pools": {p: sum(m["pool"] == p for m in manifest)
                        for p in ("calibration", "closed_reals", "aegis")},
              "slices": {z: {"kept": len(r["kept"]), "tried": r["tried"], "status": r["status"]}
                         for z, r in slices["slices"].items()},
              "test_clips": sum(m["pool"] == "test" for m in manifest),
              "files_on_disk": len(clips)}
    return {"panel_digest": digest, "counts": counts, "exclusions": exclusions,
            "duplicates": {"exact": exact, "near": near,
                           "near_excluding_own_pairs": sum(not n["own_pair"] for n in near)},
            "gate": gate, "gate_flags": flags, "draw_pool_sizes": draw["pool_sizes"],
            "test_split_sha256": draw["test_split_sha256"], "test_id_key": draw["test_id_key"],
            "manifest": "p5_panel_manifest.csv", "not_done_here": [
                "the training-side pHash cross-check (needs the AWS fetch's frame hashes)",
                "the I2V conditioning-leak check against training reals",
                "the AEGIS label join (from aegis_mapping.csv)",
                "any rebalancing for gate flags (a user decision, before the freeze)"]}


def require_pins(args: argparse.Namespace) -> None:
    """CLAUDE.md 7a: the pin check before any data is written. ``--skip-pins`` is a dry-run
    seam, refused outside DBP_DRYRUN=1."""
    import os

    if args.skip_pins:
        if os.environ.get("DBP_DRYRUN") != "1":
            raise SystemExit("--skip-pins is a dry-run seam; refused outside DBP_DRYRUN=1")
        print("pins: SKIPPED (--skip-pins, dry-run)", flush=True)
        return
    sys.path.insert(0, str(THESIS_DBP / "scripts"))
    from p0_verify_pins import require_video_pins

    require_video_pins()


def phase_preflight(args: argparse.Namespace, expect: dict[str, Any]) -> int:
    """Refuse before anything long starts (CLAUDE.md 12.2): imports, disk, both dataset
    revisions resolve, the committed listing reference exists, and the pins."""
    import importlib
    import os

    problems: list[str] = []
    result: dict[str, Any] = {"imports": {}, "disk_free_gb": {}}
    for name in expect["imports"]:
        try:
            result["imports"][name] = getattr(importlib.import_module(name), "__version__", "?")
        except ImportError as exc:
            problems.append(f"import {name}: {exc}")
    for item in args.disk:
        path, _, need = item.partition("=")
        Path(path).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free / 1e9
        result["disk_free_gb"][path] = round(free, 1)
        if free < float(need):
            problems.append(f"{path}: {free:.1f} GB free < {need} GB needed")
    reference = THESIS_DBP / expect["listing_reference"]
    if not reference.is_file():
        problems.append(f"no committed listing reference {reference}")
    for label, spec, repo, revision in (
            ("aigvdbench", args.source, AIGV_REPO, AIGV_REVISION),
            ("aegis", args.aegis_source, expect["aegis"]["repo"], expect["aegis"]["revision"])):
        try:
            result[f"{label}_files"] = len(Source(spec, repo, revision).files())
        except Exception as exc:                  # recorded as a pre-flight problem
            problems.append(f"{label} {repo}@{revision[:12]}: {type(exc).__name__}: {exc}")
    if expect.get("dryrun_fixture") and os.environ.get("DBP_DRYRUN") != "1":
        problems.append("dryrun_fixture is a dry-run seam; refused outside DBP_DRYRUN=1")
    try:
        require_pins(args)
        result["pins"] = "SKIPPED (dry-run)" if args.skip_pins else "OK"
    except (Exception, SystemExit) as exc:        # recorded as a pre-flight problem
        problems.append(f"pins: {exc}")
    result.update({"problems": problems, "ok": not problems})
    write_atomic(args.reports / "preflight.json", json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1)[:3000], flush=True)
    print("PRE-FLIGHT", "PASSED" if not problems else f"FAILED: {problems}", flush=True)
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", required=True,
                        choices=["preflight", "list", "draw", "fetch", "aegis", "finalize"])
    parser.add_argument("--source", default="hub", help="'hub' or a local AIGVDBench copy")
    parser.add_argument("--aegis-source", default="hub")
    parser.add_argument("--expect", type=Path, default=THESIS_DBP / "env" / "p5_panel_expect.json")
    parser.add_argument("--out", type=Path, help="list: the listing JSON")
    parser.add_argument("--full", action="store_true",
                        help="list: keep offsets/CRCs (needed by fetch; Kaggle /tmp only)")
    parser.add_argument("--summary", type=Path,
                        default=THESIS_DBP / "reports" / "p5_panel"
                        / "aigvdbench_listing_summary.json")
    parser.add_argument("--reference", type=Path, default=None,
                        help="list: refuse unless every zip listing equals this summary's")
    parser.add_argument("--listing", type=Path, help="draw/fetch: the full listing")
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports" / "p5_panel")
    parser.add_argument("--out-root", type=Path, help="fetch/aegis: where kept clips go")
    parser.add_argument("--cache", type=Path, default=Path("/tmp/p5_panel_cache"))
    parser.add_argument("--keep-tar", action="store_true")
    parser.add_argument("--zip", type=Path, default=None, help="finalize: archive base name")
    parser.add_argument("--disk", action="append", default=[], metavar="PATH=GB")
    parser.add_argument("--skip-pins", action="store_true", help="dry-run only")
    args = parser.parse_args()
    if args.phase == "list":
        if args.out is None:
            raise SystemExit("--phase list needs --out")
        return phase_list(args)
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    args.reports.mkdir(parents=True, exist_ok=True)
    if args.phase == "preflight":
        return phase_preflight(args, expect)
    if args.phase == "draw":
        return phase_draw(args, expect)
    if args.phase == "fetch":
        require_pins(args)
        return phase_fetch(args, expect)
    if args.phase == "aegis":
        require_pins(args)
        return phase_aegis(args, expect)
    return phase_finalize(args, expect)


if __name__ == "__main__":
    raise SystemExit(main())
