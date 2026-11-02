"""P3: fetch the frozen TrueFake sample, member by member, from ``TrueFake.zip`` (CPU).

The rule (``P3_PLAN.md``, "TrueFake -- a secondary, descriptive social-media test", fixed
2026-10-20) and the frozen sample ``reports/truefake/truefake_sample.csv`` (4,000 ids x 4
versions) decide **what** is fetched; this script only fetches it, verifiably:

1. **The archive is the one the rule was written against.** Drive's virus-scan page for
   the file gives the confirm form; a range read of the last 64 KiB gives the ZIP64 end
   record, whose total size must equal ``--expect-size``; the central directory (the
   index: names, sizes, offsets, CRC-32s; no image data) is read and its sha256 must
   equal ``--expect-cd-sha256``.
2. **Each member** is read by one byte range covering its local header and data,
   decompressed (stored or deflate), checked against the index's CRC-32 and size, and
   written under ``--out`` at its archive path (minus the ``TrueFake/`` prefix). Its
   sha256 goes into the manifest, appended per member, so a stopped run keeps its work
   and a re-run skips every member already verified on disk.
3. **Any refusal stops the run at once** (user rule, 2026-10-20): an HTTP error status
   (403, 429, 5xx ...), a response that is not a byte range (a quota or sign-in page), or
   a range that does not match the request. Nothing is retried after a refusal and
   nothing works around it; the status file records the first refusal (status, a short
   body excerpt, no cookies). Only plain network failures (timeouts, resets) are retried,
   twice.

**The images stay private** (the rule): they live only under ``--out`` (the notebook's
output, published as a private dataset); the committed report holds names, sizes and
hashes only.

Note on CLAUDE.md section 6a(c): Drive is normally reached through ``gdown``, which can
only fetch whole files (here 647 GB). The approved rule names member-by-member range
reads, so this script uses plain HTTPS range requests to ``drive.usercontent.google.com``
on Kaggle -- recorded in the log as the scoped exception the rule implies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.utils.logging import setup_logging  # noqa: E402

DRIVE = "https://drive.usercontent.google.com/download"
TRUEFAKE_ZIP_ID = "1og1cdczOspkQY3jF1m5sWNBc8X59h2e5"
EXPECT_SIZE = 694_732_361_838
EXPECT_CD_SHA256 = "869a102f65504dc25df759de4a51ae6931cecac7b255b0b5ad270dad2544dc0b"
USER_AGENT = "dbp-thesis/1.0 (academic evaluation; member reads)"
#: Bytes read past a member's name for its extra field, so header and data come in one go.
EXTRA_ALLOWANCE = 1024
MANIFEST_COLUMNS = ("member", "id", "share", "stratum", "bytes", "sha256", "crc_ok")
EXIT_REFUSED = 3


class Refused(RuntimeError):
    """Google Drive refused or rate-limited: stop, report, never work around it."""

    def __init__(self, what: str, status: int | None, excerpt: str) -> None:
        super().__init__(f"{what}: status {status}")
        self.record = {"what": what, "status": status, "excerpt": excerpt[:300]}


class FetchError(ValueError):
    """The archive or a member is not what the rule was written against."""


class Source:
    """Byte ranges of TrueFake.zip, over HTTPS (or a local file in the dry-run)."""

    def __init__(self, file_id: str, local: Path | None, log: Any) -> None:
        self.local, self.log, self.url = local, log, None
        self.reads = 0
        self.lock = threading.Lock()
        self.refuse_after = int(os.environ.get("DBP_FAULT_REFUSE_AFTER", "0") or 0) \
            if os.environ.get("DBP_DRYRUN") == "1" else 0
        if local is None:
            self.url = self._confirm_url(file_id)

    def _confirm_url(self, file_id: str) -> str:
        """The download URL behind Drive's virus-scan page (large files)."""
        page = self._get(f"{DRIVE}?id={file_id}&export=download", None, html_ok=True)
        text = page.decode("utf-8", "replace")
        fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', text))
        if not {"id", "confirm", "uuid"} <= set(fields):
            plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))
            raise Refused("no download form (quota or sign-in page?)", 200, plain)
        return f"{DRIVE}?{urlencode(fields)}"

    def _get(self, url: str, byte_range: tuple[int, int] | None,
             html_ok: bool = False) -> bytes:
        headers = {"User-Agent": USER_AGENT}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                            timeout=120) as response:
                    kind = response.headers.get("content-type", "")
                    if byte_range is None:
                        if not html_ok:
                            raise FetchError("unexpected full-body request")
                        return response.read(500_000)
                    if response.status != 206 or "text/html" in kind:
                        raise Refused("not a byte range (quota or sign-in page?)",
                                      response.status, response.read(2000).decode("utf-8",
                                                                                  "replace"))
                    content_range = response.headers.get("content-range", "")
                    if not content_range.startswith(f"bytes {byte_range[0]}-"):
                        raise Refused("range mismatch", response.status, content_range)
                    return response.read()
            except urllib.error.HTTPError as error:
                body = error.read(2000).decode("utf-8", "replace") if error.fp else ""
                raise Refused("HTTP error", error.code, body) from error
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == 2:
                    raise FetchError(f"network failure after 3 attempts: {error}") from error
                self.log.warning("network failure (%s); retry %d", error, attempt + 1)
                time.sleep(5 * (attempt + 1))
        raise FetchError("unreachable")

    def read(self, start: int, end: int) -> bytes:
        """Bytes ``start..end`` inclusive."""
        with self.lock:
            self.reads += 1
            reads = self.reads
        if self.refuse_after and reads > self.refuse_after:
            raise Refused("DRY-RUN FAULT: simulated rate limit", 429, "")
        if self.local is not None:
            with self.local.open("rb") as handle:
                handle.seek(start)
                return handle.read(end - start + 1)
        assert self.url is not None
        return self._get(self.url, (start, end))


def read_index(source: Source, expect_size: int, expect_cd: str) -> dict[str, dict[str, int]]:
    """The ZIP64 central directory, verified, as ``name -> fields``."""
    if source.local is not None:
        size = source.local.stat().st_size
        tail = source.read(max(0, size - 65536), size - 1)
    else:
        tail = source.read(expect_size - 65536, expect_size - 1)
        size = expect_size
    loc = tail.rfind(b"PK\x06\x06")
    if loc >= 0:                                   # ZIP64 (TrueFake.zip is one)
        _, _, _, _, _, _, _, entries, cd_size, cd_offset = struct.unpack("<4sQ2H2L4Q",
                                                                          tail[loc:loc + 56])
    else:
        end = tail.rfind(b"PK\x05\x06")
        if end < 0:
            raise FetchError("no end-of-central-directory record in the last 64 KiB")
        _, _, _, _, entries, cd_size, cd_offset, _ = struct.unpack("<4s4H2LH",
                                                                   tail[end:end + 22])
    if size != expect_size:
        raise FetchError(f"archive size {size} != {expect_size}")
    cd = source.read(cd_offset, cd_offset + cd_size - 1)
    digest = hashlib.sha256(cd).hexdigest()
    if digest != expect_cd:
        raise FetchError(f"central directory sha256 {digest} != {expect_cd}")
    index, off = {}, 0
    while off < len(cd):
        (_sig, _vm, _vn, _flag, method, _mt, _md, crc, csz, usz, nlen, xlen, clen, _dsk, _ia,
         _ea, lho) = struct.unpack("<4s6H3L5H2L", cd[off:off + 46])
        name = cd[off + 46:off + 46 + nlen].decode("utf-8")
        extra = cd[off + 46 + nlen:off + 46 + nlen + xlen]
        e = 0
        while e + 4 <= len(extra):
            hid, hsz = struct.unpack("<2H", extra[e:e + 4])
            if hid == 1:
                data, k = extra[e + 4:e + 4 + hsz], 0
                if usz == 0xFFFFFFFF:
                    usz = struct.unpack("<Q", data[k:k + 8])[0]
                    k += 8
                if csz == 0xFFFFFFFF:
                    csz = struct.unpack("<Q", data[k:k + 8])[0]
                    k += 8
                if lho == 0xFFFFFFFF:
                    lho = struct.unpack("<Q", data[k:k + 8])[0]
            e += 4 + hsz
        index[name] = {"method": method, "crc": crc, "csize": csz, "usize": usz, "offset": lho}
        off += 46 + nlen + xlen + clen
    if len(index) != entries:
        raise FetchError(f"parsed {len(index)} entries, the end record says {entries}")
    return index


def fetch_member(source: Source, name: str, entry: dict[str, int]) -> bytes:
    """One member's bytes, decompressed and checked against the index."""
    start = entry["offset"]
    blob = source.read(start, start + 30 + len(name.encode()) + EXTRA_ALLOWANCE
                       + entry["csize"] - 1)
    sig, _v, _f, _m, _t, _d, _c, _cs, _us, nlen, xlen = struct.unpack("<4s5H3L2H", blob[:30])
    if sig != b"PK\x03\x04":
        raise FetchError(f"{name}: no local header at {start}")
    data_start = 30 + nlen + xlen
    data = blob[data_start:data_start + entry["csize"]]
    if len(data) < entry["csize"]:            # an unusually long extra field
        more = source.read(start + len(blob), start + data_start + entry["csize"] - 1)
        data = (blob[data_start:] + more)[:entry["csize"]]
    if entry["method"] == 0:
        content = data
    elif entry["method"] == 8:
        content = zlib.decompressobj(-15).decompress(data)
    else:
        raise FetchError(f"{name}: compression method {entry['method']} not supported")
    if len(content) != entry["usize"] or zlib.crc32(content) != entry["crc"]:
        raise FetchError(f"{name}: size or CRC-32 does not match the archive index")
    return content


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main() -> int:
    """Fetch the frozen sample. 0 = complete, 3 = refused (stopped), 1 = failed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=Path,
                        default=THESIS_DBP / "reports" / "truefake" / "truefake_sample.csv")
    parser.add_argument("--out", type=Path, required=True, help="Where the images go.")
    parser.add_argument("--file-id", default=TRUEFAKE_ZIP_ID)
    parser.add_argument("--expect-size", type=int, default=EXPECT_SIZE)
    parser.add_argument("--expect-cd-sha256", default=EXPECT_CD_SHA256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--local-zip", type=Path, default=None,
                        help="Dry-run only (DBP_DRYRUN=1): read a local archive instead.")
    parser.add_argument("--reports", type=Path,
                        default=THESIS_DBP / "reports" / "truefake")
    args = parser.parse_args()
    log = setup_logging("p3_fetch_truefake")
    if args.local_zip is not None and os.environ.get("DBP_DRYRUN") != "1":
        log.error("--local-zip is a dry-run seam and needs DBP_DRYRUN=1")
        return 2
    status_path = args.reports / "truefake_fetch_status.json"
    manifest_path = args.reports / "truefake_fetch_manifest.csv"
    with args.sample.open(encoding="utf-8", newline="") as handle:
        sample = list(csv.DictReader(handle))
    status: dict[str, Any] = {"sample": args.sample.name,
                              "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
                              "members": len(sample), "outcome": "RUNNING", "refusal": None}
    done: dict[str, str] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                path = args.out / row["member"].removeprefix("TrueFake/")
                on_disk = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() \
                    else None
                if on_disk == row["sha256"]:
                    done[row["member"]] = row["sha256"]
    started = time.perf_counter()
    lock = threading.Lock()
    fetched_bytes = [0]
    try:
        source = Source(args.file_id, args.local_zip, log)
        index = read_index(source, args.expect_size, args.expect_cd_sha256)
        status["index"] = {"entries": len(index), "cd_sha256": args.expect_cd_sha256,
                           "size": args.expect_size}
        missing = [r["member"] for r in sample if r["member"] not in index]
        if missing:
            raise FetchError(f"{len(missing)} sample member(s) not in the archive: {missing[:3]}")
        todo = [r for r in sample if r["member"] not in done]
        log.info("index verified (%d entries); %d member(s) to fetch, %d already on disk",
                 len(index), len(todo), len(done))
        new_file = not manifest_path.is_file()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            if new_file:
                writer.writeheader()

            stop = threading.Event()

            def one(row: dict[str, str]) -> None:
                if stop.is_set():          # a refusal elsewhere: send no further request
                    return
                content = fetch_member(source, row["member"], index[row["member"]])
                path = args.out / row["member"].removeprefix("TrueFake/")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(content)
                os.replace(tmp, path)
                with lock:
                    writer.writerow({"member": row["member"], "id": row["id"],
                                     "share": row["share"], "stratum": row["stratum"],
                                     "bytes": len(content),
                                     "sha256": hashlib.sha256(content).hexdigest(),
                                     "crc_ok": True})
                    handle.flush()
                    fetched_bytes[0] += index[row["member"]]["csize"]
                    done[row["member"]] = "ok"
                    if len(done) % 500 == 0:
                        log.info("%d / %d members (%.1f min)", len(done), len(sample),
                                 (time.perf_counter() - started) / 60)

            # The first refusal or failure ends the run: queued members are cancelled and
            # in-flight ones send nothing further (no "with" block -- its exit would wait
            # for every queued member to run).
            pool = ThreadPoolExecutor(max_workers=args.workers)
            futures = [pool.submit(one, row) for row in todo]
            try:
                for future in futures:
                    future.result()
            except BaseException:
                stop.set()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            pool.shutdown(wait=True)
        status["outcome"] = "COMPLETE" if len(done) == len(sample) else "INCOMPLETE"
        code = 0 if status["outcome"] == "COMPLETE" else 1
    except Refused as refusal:
        status["outcome"] = "REFUSED"
        status["refusal"] = refusal.record
        log.error("Google Drive REFUSED (%s); stopping as the rule requires -- no workaround",
                  refusal.record)
        code = EXIT_REFUSED
    except FetchError as error:
        status["outcome"] = "FAILED"
        status["error"] = str(error)
        log.error("%s", error)
        code = 1
    status.update({"members_done": len(done), "bytes_compressed_fetched": fetched_bytes[0],
                   "seconds": round(time.perf_counter() - started, 1)})
    write_json(status_path, status)
    log.info("TrueFake fetch %s: %d / %d members in %.1f min", status["outcome"], len(done),
             len(sample), status["seconds"] / 60)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
