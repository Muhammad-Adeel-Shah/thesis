"""P3 Phase B step 3 diagnostic: does Flickr refuse this session? (user decision 2026-10-13)

Kaggle run 1 of the Open Images fetch got 429 on all 24 requests. This probe tells a rate
limit from a block, and checks the two host groups separately (one may be blocked and the
other not):

* **6 GETs**: the first 3 ``farm*`` rows and the first 3 ``c*`` rows in seed order,
  sent in that combined seed order, **one at a time** (never more than one connection
  open), ``--gap`` seconds apart (30 on Kaggle). Only approved hosts are contacted;
  redirects are recorded, not followed.
* **No image is kept.** Each body is streamed through MD5 only, to report whether it
  matches ``OriginalMD5``.
* Per request: UTC time, host group, host, status, seconds, response headers **without
  cookies**, and for a non-200 the first 200 body bytes.
* The session's public egress IP range -- **first two octets only**; the full address is
  never written or logged -- from one GET to ``--ip-echo-url`` (``api.ipify.org``,
  approved by the user 2026-10-13, and the only echo host accepted).

Writes ``reports/p3_flickr_probe.json`` with a per-group table and a verdict against the
user's pre-decided branches: **A** (mostly 200 at the slow pace: a rate limit) or **B**
(403/429 on both groups: blocked), else **mixed**. A diagnostic: exits 0 whenever the
probe ran, 1 only on bad arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.open_images import (  # noqa: E402
    REFUSAL_CODES,
    USER_AGENT,
    Fetcher,
    LocalFetcher,
    Throttle,
    host_approved,
    host_group,
    host_of,
    load_metadata,
    md5_b64,
)
from dbp.utils.logging import setup_logging  # noqa: E402

IP_ECHO_HOST = "api.ipify.org"
_PRIVATE_HEADERS = frozenset({"set-cookie", "cookie"})


def ip_range(address: str) -> str:
    """The first two octets (IPv4) or hextets (IPv6) of an address; nothing more."""
    address = address.strip()
    if ":" in address:
        parts = address.split(":")
        return f"{parts[0]}:{parts[1]}::/32"
    parts = address.split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return "unrecognised"
    return f"{parts[0]}.{parts[1]}.x.x"


def egress_range(url: str | None, log: Any) -> str:
    """The egress IP range from one echo request, or NOT-MEASURED."""
    if not url:
        return "NOT-MEASURED (no IP-echo URL given)"
    if host_of(url) != IP_ECHO_HOST:
        raise ValueError(f"IP-echo host {host_of(url)!r} is not the approved {IP_ECHO_HOST}")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return ip_range(response.read(64).decode("ascii", "replace"))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        log.warning("IP echo failed: %s", type(exc).__name__)
        return f"NOT-MEASURED ({type(exc).__name__})"


def pick(rows: list[dict[str, str]], per_group: dict[str, int]) -> list[dict[str, str]]:
    """The first ``n`` rows of each host group, kept in the combined seed order."""
    taken: dict[str, int] = dict.fromkeys(per_group, 0)
    picked = []
    for row in rows:
        group = host_group(host_of(row["OriginalURL"]))
        if group in taken and taken[group] < per_group[group]:
            taken[group] += 1
            picked.append(row)
    return picked


def probe_one(row: dict[str, str], fetcher: Fetcher) -> dict[str, Any]:
    """One GET, body streamed through MD5 only; nothing written to disk."""
    url, host = row["OriginalURL"], host_of(row["OriginalURL"])
    record: dict[str, Any] = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "image_id": row["ImageID"], "host": host,
                              "host_group": host_group(host)}
    if not host_approved(host, fetcher.patterns):
        record.update(status="not contacted (host not approved)")
        return record
    started = time.perf_counter()
    try:
        with fetcher._open(url) as response:  # noqa: SLF001 - the same opener the walk uses
            md5 = hashlib.md5(usedforsecurity=False)
            size = 0
            for chunk in iter(lambda: response.read(1 << 20), b""):
                md5.update(chunk)
                size += len(chunk)
            headers = response.headers.items() if hasattr(response, "headers") else []
            record.update(status=200, bytes=size, metadata_bytes=int(row["OriginalSize"]),
                          md5_matches=md5_b64(md5.digest()) == row["OriginalMD5"])
    except urllib.error.HTTPError as exc:
        headers = exc.headers.items() if exc.headers else []
        record.update(status=exc.code, body_head=(exc.read(200) or b"").decode("latin-1"))
        if exc.code in (301, 302, 303, 307, 308):
            record["redirect_to_host"] = host_of(exc.headers.get("Location", ""))
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        headers = []
        record.update(status=f"error: {type(exc).__name__}: {str(exc)[:80]}")
    record["seconds"] = round(time.perf_counter() - started, 2)
    record["headers"] = {k: v for k, v in headers if k.lower() not in _PRIVATE_HEADERS}
    return record


def verdict(records: list[dict[str, Any]]) -> tuple[str, dict[str, dict[str, int]]]:
    """Per-group counts and the branch they point to (user's pre-decided A / B)."""
    table: dict[str, dict[str, int]] = {}
    for rec in records:
        row = table.setdefault(rec["host_group"], {"requests": 0, "ok_200": 0, "refused": 0,
                                                   "other": 0})
        row["requests"] += 1
        if rec["status"] == 200:
            row["ok_200"] += 1
        elif rec["status"] in REFUSAL_CODES:
            row["refused"] += 1
        else:
            row["other"] += 1
    if table and all(r["ok_200"] * 2 > r["requests"] for r in table.values()):
        branch = "A: mostly 200 at the slow pace -- a rate limit"
    elif len(table) >= 2 and all(r["refused"] == r["requests"] for r in table.values()):
        branch = "B: refused (403/429/503) on both host groups -- this session's IPs are blocked"
    else:
        branch = "mixed: see the per-group table; no pre-decided branch applies as such"
    return branch, table


def main() -> int:
    """Run the probe. Returns 0 when it ran, 1 on bad arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--farm", type=int, default=3)
    parser.add_argument("--c", type=int, default=3)
    parser.add_argument("--gap", type=float, default=30.0, help="Seconds between requests.")
    parser.add_argument("--ip-echo-url", default=None)
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--local-source", type=Path, default=None,
                        help="DRY-RUN ONLY (DBP_DRYRUN=1): serve files from this directory.")
    args = parser.parse_args()
    log = setup_logging("p3_probe_flickr")
    if args.local_source and os.environ.get("DBP_DRYRUN") != "1":
        log.error("PROBE: --local-source is for the dry-run only (DBP_DRYRUN=1)")
        return 1
    if args.ip_echo_url and host_of(args.ip_echo_url) != IP_ECHO_HOST:
        log.error("PROBE: the only approved IP-echo host is %s", IP_ECHO_HOST)
        return 1
    fetcher = (LocalFetcher(args.local_source, throttle=Throttle(sleep=lambda s: None))
               if args.local_source else Fetcher(throttle=Throttle(sleep=lambda s: None)))
    rows = pick(load_metadata(args.metadata), {"farm*": args.farm, "c*": args.c})
    log.info("probe: %d request(s), one at a time, %.0f s apart: %s", len(rows), args.gap,
             [host_of(r["OriginalURL"]) for r in rows])
    egress = egress_range(None if args.local_source else args.ip_echo_url, log)
    log.info("egress IP range (first two octets): %s", egress)
    records = []
    for n, row in enumerate(rows):
        if n:
            time.sleep(args.gap)
        record = probe_one(row, fetcher)
        records.append(record)
        log.info("%s  %-24s %-5s -> %s (%.2f s)%s", record["utc"], record["host"],
                 record["host_group"], record["status"], record.get("seconds", 0),
                 f"  Retry-After={record['headers'].get('Retry-After')}"
                 if record.get("headers", {}).get("Retry-After") else "")
    branch, table = verdict(records)
    report = {"stage": "p3_flickr_probe", "egress_ip_range": egress, "gap_s": args.gap,
              "one_connection_at_a_time": True, "images_kept": 0, "requests": records,
              "by_host_group": table, "verdict": branch}
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "p3_flickr_probe.json"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    for group, row in table.items():
        log.info("host group %-5s %s", group, row)
    log.info("PROBE VERDICT: %s", branch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
