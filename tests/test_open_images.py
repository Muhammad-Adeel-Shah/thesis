"""P3 Phase B step 3: Open Images fetch rules (a)-(f), politeness and the disk guard.

User rules 2026-10-11, 2026-10-12 and 2026-10-13 (c1-c8 approved; at most 8 connections;
slow down on repeated 429/503).
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.data import open_images as oi  # noqa: E402
from dbp.utils import disk  # noqa: E402


class _Response(io.BytesIO):
    """A urllib-like response."""


class RecordingFetcher(oi.Fetcher):
    """Answers from a script of (code, location/body) per URL and records every contact."""

    def __init__(self, answers: dict[str, list], **kwargs) -> None:
        kwargs.setdefault("throttle", oi.Throttle(sleep=lambda seconds: None))
        super().__init__(backoff=0.0, **kwargs)
        self.answers = answers
        self.contacted: list[str] = []

    def _open(self, url: str):
        self.contacted.append(oi.host_of(url))
        code, payload = self.answers[url].pop(0)
        if code == 200:
            return _Response(payload)
        headers = ({"Retry-After": payload} if code in (429, 503) else {"Location": payload}) \
            if payload else {}
        raise urllib.error.HTTPError(url, code, "scripted", headers, None)


FARM = "https://farm3.staticflickr.com/1/a_o.jpg"


def test_exactly_farm1_8_and_c1_8_are_approved() -> None:
    for n in range(1, 9):
        assert oi.host_approved(f"farm{n}.staticflickr.com")
        assert oi.host_approved(f"c{n}.staticflickr.com")
    for host in ("c9.staticflickr.com", "c0.staticflickr.com", "farm9.staticflickr.com",
                 "live.staticflickr.com", "farm3.staticflickr.com.evil.io",
                 "xfarm3.staticflickr.com", "c1.staticflickr.com.evil.io",
                 "farm.staticflickr.com", "c.staticflickr.com", ""):
        assert not oi.host_approved(host), host


def test_host_groups() -> None:
    assert oi.host_group("farm7.staticflickr.com") == "farm*"
    assert oi.host_group("c2.staticflickr.com") == "c*"
    assert oi.host_group("live.staticflickr.com") == "other"
    assert oi.host_group("c9.staticflickr.com") == "other"         # not approved, not c*


def test_an_unapproved_host_is_never_contacted(tmp_path: Path) -> None:
    fetcher = RecordingFetcher({})
    result = fetcher.fetch("https://c9.staticflickr.com/1/a_o.jpg", tmp_path / "a.jpg")
    assert result.status == "host" and result.detail == "c9.staticflickr.com"
    assert fetcher.contacted == []


def test_a_redirect_may_land_on_live_but_nowhere_else(tmp_path: Path) -> None:
    live = "https://live.staticflickr.com/1/a_o.jpg"
    fetcher = RecordingFetcher({FARM: [(302, live)], live: [(200, b"abc")]})
    result = fetcher.fetch(FARM, tmp_path / "a.jpg")
    assert result.status == "ok" and result.final_host == "live.staticflickr.com"
    assert (tmp_path / "a.jpg").read_bytes() == b"abc"
    other = RecordingFetcher({FARM: [(302, "https://cdn.example.com/a.jpg")]})
    refused = other.fetch(FARM, tmp_path / "b.jpg")
    assert refused.status == "host" and refused.detail == "redirect:cdn.example.com"
    assert other.contacted == ["farm3.staticflickr.com"]          # the target was not contacted


def test_gone_retry_and_give_up(tmp_path: Path) -> None:
    assert RecordingFetcher({FARM: [(404, None)]}).fetch(FARM, tmp_path / "a").status == "gone"
    assert RecordingFetcher({FARM: [(410, None)]}).fetch(FARM, tmp_path / "a").status == "gone"
    flaky = RecordingFetcher({FARM: [(503, None), (429, None), (200, b"x")]})
    assert flaky.fetch(FARM, tmp_path / "a").status == "ok" and len(flaky.contacted) == 3
    down = RecordingFetcher({FARM: [(503, None)] * 4}, retries=3)
    result = down.fetch(FARM, tmp_path / "b")
    assert result.status == "failed" and "after 4 attempts" in result.detail
    assert not (tmp_path / "b").exists() and not (tmp_path / "b.part").exists()


def test_the_checksum_is_compared_in_the_metadatas_base64_form(tmp_path: Path) -> None:
    data = b"original bytes"
    fetcher = RecordingFetcher({FARM: [(200, data)]})
    result = fetcher.fetch(FARM, tmp_path / "a.jpg")
    assert result.md5_b64 == base64.b64encode(hashlib.md5(data).digest()).decode()
    assert result.sha256 == hashlib.sha256(data).hexdigest()


def test_order_and_split_are_seeded_and_independent(tmp_path: Path) -> None:
    ids = [f"{n:016x}" for n in range(50)]
    assert oi.order_key("abc") == hashlib.blake2b(b"open_images:2026:abc",
                                                  digest_size=16).hexdigest()
    assert oi.order_key("abc") != oi.split_key("abc")
    split = oi.assign_split(ids, 5)
    assert sum(v == "held_out" for v in split.values()) == 5
    assert oi.assign_split(list(reversed(ids)), 5) == split
    with pytest.raises(ValueError):
        oi.assign_split(ids[:3], 5)


def test_metadata_keeps_cc_by_2_only_in_seed_order(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ImageID", "License"])
        writer.writeheader()
        for n in range(20):
            writer.writerow({"ImageID": f"i{n}", "License": oi.CC_BY_2 if n % 4 else "other"})
    rows = oi.load_metadata(path)
    assert len(rows) == 15
    assert [r["ImageID"] for r in rows] == sorted((r["ImageID"] for r in rows), key=oi.order_key)


def test_the_committed_metadata_is_the_cc_by_2_extract() -> None:
    path = ROOT / "reports/p3_open_images/oi_validation_cc_by_2.csv"
    expect = json.loads((ROOT / "env/p3_open_images_expect.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expect["metadata"]["sha256"]
    assert len(oi.load_metadata(path)) == expect["metadata"]["rows"] == 41620


def test_the_disk_guard_reports_a_shortfall(tmp_path: Path) -> None:
    assert disk.require_free({tmp_path: 0.0}) == []
    problems = disk.require_free({tmp_path: 10**9})
    assert len(problems) == 1 and "the run needs" in problems[0]
    record = disk.df_report("stage", [tmp_path / "x"], type("L", (), {"info": print})())
    assert record["stage"] == "stage" and record["filesystems"][0]["free_gb"] > 0


def test_local_source_is_refused_outside_the_dry_run(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "DBP_DRYRUN"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/p3_fetch_open_images.py"), "--metadata", "m",
         "--tmp", str(tmp_path), "--out-root", str(tmp_path), "--coco-index", "c",
         "--gen-manifest", "g", "--pairs-manifest", "p", "--local-source", str(tmp_path)],
        capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 1 and "dry-run only" in result.stdout + result.stderr


def test_more_than_eight_parallel_requests_are_refused(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/p3_fetch_open_images.py"), "--metadata", "m",
         "--tmp", str(tmp_path), "--out-root", str(tmp_path), "--coco-index", "c",
         "--gen-manifest", "g", "--pairs-manifest", "p", "--workers", "9"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 1 and "polite limit" in result.stdout + result.stderr


def test_the_runbook_follows_the_user_rules() -> None:
    text = (ROOT / "env/kaggle_run_p3_open_images.py").read_text(encoding="utf-8")
    expect = json.loads((ROOT / "env/p3_open_images_expect.json").read_text(encoding="utf-8"))
    assert expect["workers"] <= oi.MAX_PARALLEL == 8
    # 2026-10-15: the CVDF S3 copies replace the Flickr originals (Flickr refused every route).
    assert expect["source"] == "s3"
    assert expect["hosts"] == list(oi.S3_HOST_PATTERNS)
    assert '"--source", EXPECT.get("source", "flickr")' in text
    assert expect["local_source"] is None
    assert text.index('py("p2_preflight.py"') < text.index('py("p3_coco_index.py"') \
        < text.index('py("p3_fetch_open_images.py"')
    assert '"--tmp", TMP / "oi_originals"' in text                      # rule (e)


def test_the_run_stops_before_any_download_when_disk_is_short(tmp_path: Path) -> None:
    """User rule 2026-10-12: stop before any download if free space is below the need."""
    reports = tmp_path / "reports"
    env = {**os.environ, "DBP_DRYRUN": "1"}
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/p3_fetch_open_images.py"), "--metadata", "m",
         "--tmp", str(tmp_path / "dl"), "--out-root", str(tmp_path / "out"),
         "--reports", str(reports), "--coco-index", "c", "--gen-manifest", "g",
         "--pairs-manifest", "p", "--local-source", str(tmp_path), "--min-free-tmp-gb", "1e9"],
        capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 1 and "nothing downloaded" in result.stdout + result.stderr
    summary = json.loads((reports / "p3_open_images_summary.json").read_text(encoding="utf-8"))
    assert summary["stopped"].startswith("insufficient disk before any download")
    assert summary["disk"][0]["stage"] == "start"
    assert not any((tmp_path / "dl").iterdir())


# ------------------------------------------------ politeness (user rule 2026-10-13) ----


def test_at_most_eight_connections_are_open_at_once(tmp_path: Path) -> None:
    """The cap holds in the fetcher itself, however many threads call it."""
    gauge = {"open": 0, "peak": 0}
    lock = threading.Lock()

    class Slow(oi.Fetcher):
        def _open(self, url: str):
            with lock:
                gauge["open"] += 1
                gauge["peak"] = max(gauge["peak"], gauge["open"])
            time.sleep(0.05)
            with lock:
                gauge["open"] -= 1
            return _Response(b"x")

    fetcher = Slow(throttle=oi.Throttle(sleep=lambda seconds: None))
    threads = [threading.Thread(target=fetcher.fetch, args=(FARM, tmp_path / f"{i}.jpg"))
               for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert gauge["peak"] <= oi.MAX_PARALLEL == 8
    with pytest.raises(ValueError, match="at most 8"):
        oi.Fetcher(max_parallel=9)


class Clock:
    """A settable monotonic clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_one_parallel_wave_raises_the_delay_once() -> None:
    """Regression, Kaggle run 1 (2026-10-13): 8 simultaneous 429s took 1 s to 60 s in 0.2 s."""
    clock, logged = Clock(), []
    throttle = oi.Throttle(base=1.0, ceiling=60.0, log=logged.append,
                           sleep=lambda seconds: None, clock=clock)
    for _ in range(8):                                   # one wave, same instant
        throttle.refused(429)
    assert throttle.delay == 1.0 and throttle.raises == 1 and throttle.refusals == 8
    assert len(logged) == 1 and "slowing down" in logged[0]
    clock.now += 1.0                                     # the next wave, one delay later
    for _ in range(8):
        throttle.refused(429)
    assert throttle.delay == 2.0 and throttle.raises == 2 and throttle.refusals == 16
    throttle.refused(429, retry_after="30")             # a longer Retry-After always counts
    assert throttle.delay == 30.0


def test_the_walk_stops_only_after_ten_minutes_of_refusals_at_the_ceiling() -> None:
    clock = Clock()
    throttle = oi.Throttle(base=1.0, ceiling=60.0, sleep=lambda seconds: None, clock=clock)
    while True:                                          # escalate wave by wave
        clock.now += max(throttle.delay, 1.0)
        throttle.refused(429)
        if throttle.delay >= 60.0:
            start = clock.now                            # the ceiling is reached here
            break
    clock.now = start + 599
    throttle.refused(429)
    assert not throttle.tripped                          # 599 s is not enough
    throttle.succeeded()                                 # a success resets the clock
    clock.now = start + 700
    throttle.refused(429)
    clock.now = start + 700 + 599
    throttle.refused(429)
    assert not throttle.tripped
    clock.now = start + 700 + 600
    throttle.refused(429)
    assert throttle.tripped and "stopping the walk" in throttle.events[-1]


def test_a_tripped_throttle_contacts_no_one(tmp_path: Path) -> None:
    throttle = oi.Throttle(sleep=lambda seconds: None)
    throttle.tripped = True
    fetcher = RecordingFetcher({FARM: [(200, b"x")]}, throttle=throttle)
    result = fetcher.fetch(FARM, tmp_path / "a.jpg")
    assert result.status == "stopped" and fetcher.contacted == []


def test_the_first_twenty_refusals_are_kept_without_cookies(tmp_path: Path) -> None:
    class Refusing(oi.Fetcher):
        def _open(self, url: str):
            headers = {"Retry-After": "0", "Server": "ATS", "Set-Cookie": "secret=1",
                       "Content-Type": "text/html"}
            raise urllib.error.HTTPError(url, 403, "blocked", headers,
                                         io.BytesIO(b"<html>" + b"x" * 500))

    throttle = oi.Throttle(sleep=lambda seconds: None)
    fetcher = Refusing(throttle=throttle, retries=30, backoff=0.0)
    result = fetcher.fetch(FARM, tmp_path / "a.jpg")
    assert result.status == "failed" and throttle.refusals == 31     # 403 is a refusal
    assert len(throttle.samples) == 20
    sample = throttle.samples[0]
    assert sample["status"] == 403 and sample["host"] == "farm3.staticflickr.com"
    assert "Set-Cookie" not in sample["headers"] and sample["headers"]["Server"] == "ATS"
    assert sample["body_head"].startswith("<html>") and len(sample["body_head"]) == 200
    assert throttle.summary()["refusal_samples"] == throttle.samples


def test_successes_relax_the_delay() -> None:
    clock, logged = Clock(), []
    throttle = oi.Throttle(base=1.0, relax_after=3, log=logged.append,
                           sleep=lambda seconds: None, clock=clock)
    throttle.refused(429, retry_after="8")
    for _ in range(3):
        throttle.succeeded()
    assert throttle.delay == 4.0 and "successes" in logged[-1]


def test_a_429_with_retry_after_is_retried_after_slowing(tmp_path: Path) -> None:
    slept: list[float] = []
    throttle = oi.Throttle(base=1.0, ceiling=60.0, sleep=slept.append)
    fetcher = RecordingFetcher({FARM: [(429, "5"), (200, b"ok")]}, throttle=throttle)
    assert fetcher.fetch(FARM, tmp_path / "a.jpg").status == "ok"
    assert 5.0 in slept and throttle.refusals == 1


# ------------------------------------------------ the Flickr probe (2026-10-13) ----


def _probe_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("p3_probe_test",
                                                  ROOT / "scripts/p3_probe_flickr.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_only_two_octets_of_the_egress_ip_are_kept() -> None:
    probe = _probe_module()
    assert probe.ip_range("34.118.52.7\n") == "34.118.x.x"
    assert probe.ip_range("2600:1900:4000:abcd::1") == "2600:1900::/32"
    assert probe.ip_range("not an ip") == "unrecognised"
    with pytest.raises(ValueError, match="not the approved"):
        probe.egress_range("https://ifconfig.me", None)


def test_the_probe_takes_the_first_three_of_each_group_in_seed_order() -> None:
    probe = _probe_module()
    rows = [{"ImageID": str(n), "OriginalURL": f"https://{h}/1/x.jpg"}
            for n, h in enumerate(["c9.staticflickr.com", "farm1.staticflickr.com",
                                   "c2.staticflickr.com", "farm3.staticflickr.com",
                                   "farm4.staticflickr.com", "farm5.staticflickr.com",
                                   "c3.staticflickr.com", "c4.staticflickr.com",
                                   "c5.staticflickr.com"])]
    picked = probe.pick(rows, {"farm*": 3, "c*": 3})
    assert [r["ImageID"] for r in picked] == ["1", "2", "3", "4", "6", "7"]


def test_the_probe_verdict_follows_the_pre_decided_branches() -> None:
    probe = _probe_module()
    ok = [{"host_group": g, "status": 200} for g in ("farm*", "c*") for _ in range(3)]
    assert probe.verdict(ok)[0].startswith("A:")
    blocked = [{"host_group": g, "status": s} for g, s in
               (("farm*", 429), ("farm*", 403), ("farm*", 429), ("c*", 429), ("c*", 503),
                ("c*", 429))]
    assert probe.verdict(blocked)[0].startswith("B:")
    mixed = [{"host_group": "farm*", "status": 429}] * 3 + [{"host_group": "c*", "status": 200}] * 3
    assert probe.verdict(mixed)[0].startswith("mixed")


def test_colour_modes_are_rejected_never_converted() -> None:
    """User rule 2026-10-14: only 8-bit RGB is kept; each other mode has its own reason."""
    reason = oi.colour_mode_reason
    assert reason("RGB") is None
    assert reason("RGB", {"transparency": (0, 0, 0)}) == "alpha"
    assert reason("CMYK") == "CMYK"
    assert reason("P") == reason("PA") == "palette"
    assert reason("RGBA") == reason("LA") == "alpha"
    assert reason("I;16") == reason("I") == reason("F") == "16-bit or float"
    assert reason("L") == reason("1") == "greyscale (mode L/1)"
    assert set(oi.decoder_versions()) == {"decoder_python", "decoder_pillow", "decoder_libjpeg"}


def test_p1_decoding_applies_neither_exif_orientation_nor_icc() -> None:
    """The local route decodes as P1 decoded COCO: no exif_transpose, no ImageCms."""
    for path in (ROOT / "src/dbp/utils/io.py", ROOT / "scripts/p1_build_reals.py",
                 ROOT / "scripts/p3_fetch_open_images.py", ROOT / "scripts/p3_oi_local_fetch.py"):
        text = path.read_text(encoding="utf-8")
        assert "exif_transpose" not in text and "ImageCms" not in text, path.name



# ------------------------------------------- the CVDF S3 copies (user, 2026-10-15) ----


def test_s3_urls_and_hosts() -> None:
    assert oi.s3_url("fe600639ac5f36c1") == (
        "https://open-images-dataset.s3.amazonaws.com/validation/fe600639ac5f36c1.jpg")
    with pytest.raises(ValueError):
        oi.s3_url("../etc/passwd")
    for host in ("open-images-dataset.s3.amazonaws.com",
                 "open-images-dataset.s3.us-east-1.amazonaws.com",
                 "open-images-dataset.s3-us-west-2.amazonaws.com"):
        assert oi.host_approved(host, oi.S3_HOST_PATTERNS), host
    for host in ("evil.s3.amazonaws.com", "open-images-dataset.s3.amazonaws.com.evil.io",
                 "farm1.staticflickr.com", "live.staticflickr.com"):
        assert not oi.host_approved(host, oi.S3_HOST_PATTERNS), host


def test_s3_mode_skips_original_md5_but_records_sha256(tmp_path: Path) -> None:
    """OriginalMD5 is the Flickr original's; an S3 copy is rescaled, so it is not compared."""
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(ROOT / "scripts"))
    import p3_fetch_open_images as fetch

    image_id = "00000000000000aa"
    served = tmp_path / "served"
    served.mkdir()
    small = np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(small).resize((520, 450), Image.BICUBIC).save(served / f"{image_id}.jpg")
    row = {"ImageID": image_id, "OriginalURL": "https://farm1.staticflickr.com/1/x_o.jpg",
           "OriginalMD5": "AAAAAAAAAAAAAAAAAAAAAA=="}          # would fail a Flickr check
    fetcher = oi.LocalFetcher(served, patterns=oi.S3_HOST_PATTERNS, redirect_hosts=())
    out = fetch.process(row, fetcher, tmp_path / "dl", tmp_path / "crops" / "t_oi", 384,
                        source="s3")
    assert out["outcome"] == "ok", out
    assert out["sha256"] == hashlib.sha256((served / f"{image_id}.jpg").read_bytes()).hexdigest()
    flickr = fetch.process(row, oi.LocalFetcher(served), tmp_path / "dl2",
                           tmp_path / "crops2" / "t_oi", 384, source="flickr")
    assert flickr["outcome"] in ("gone", "md5")          # the Flickr route keeps its checks
