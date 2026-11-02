"""Route (ii), the local stage of the Open Images fetch (user decision 2026-10-13)."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p3_oi_local_fetch.py"
CC = "https://creativecommons.org/licenses/by/2.0/"


def _fixture(tmp: Path, n: int = 8, missing: tuple[int, ...] = (2,)) -> Path:
    """``n`` approved-host rows served from a local folder; ``missing`` rows answer 404."""
    flickr = tmp / "flickr"
    flickr.mkdir()
    rows = []
    for i in range(n):
        name = f"{i:016x}_o.jpg"
        rng = np.random.default_rng(i)
        small = rng.integers(0, 256, (16, 18, 3), dtype=np.uint8)
        Image.fromarray(small).resize((520, 450), Image.BICUBIC).save(flickr / name, quality=90)
        data = (flickr / name).read_bytes()
        rows.append({"ImageID": f"{i:016x}", "OriginalURL": f"https://c{1 + i % 8}"
                     f".staticflickr.com/1/{name}", "License": CC, "Author": f"a{i}",
                     "OriginalSize": str(len(data)),
                     "OriginalMD5": base64.b64encode(hashlib.md5(data).digest()).decode(),
                     "Rotation": ""})
        if i in missing:
            (flickr / name).unlink()
    meta = tmp / "meta.csv"
    with meta.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return meta


def _run(meta: Path, flickr: Path, originals: Path, out: Path, *extra: str,
         dryrun: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONUTF8": "1"}
    env.pop("DBP_DRYRUN", None)
    if dryrun:
        env["DBP_DRYRUN"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--metadata", str(meta), "--originals", str(originals),
         "--out", str(out), "--min-free-gb", "0.01", "--local-source", str(flickr),
         "--batch", "2", *extra],
        capture_output=True, text=True, env=env, check=False)


def _candidates(out: Path) -> list[tuple[str, str]]:
    with (out / "p3_oi_candidates_manifest.csv").open(encoding="utf-8") as handle:
        return [(r["position"], r["oi_image_id"]) for r in csv.DictReader(handle)]


def test_a_resumed_walk_equals_one_uninterrupted_walk(tmp_path: Path) -> None:
    meta = _fixture(tmp_path)
    flickr = tmp_path / "flickr"
    one = _run(meta, flickr, tmp_path / "o1", tmp_path / "c1", "--target-candidates", "5")
    assert one.returncode == 0, one.stdout + one.stderr
    part = _run(meta, flickr, tmp_path / "o2", tmp_path / "c2", "--target-candidates", "5",
                "--max-rows", "3")
    assert part.returncode == 1 and "INCOMPLETE" in part.stdout + part.stderr
    rest = _run(meta, flickr, tmp_path / "o2", tmp_path / "c2", "--target-candidates", "5")
    assert rest.returncode == 0, rest.stdout + rest.stderr
    assert _candidates(tmp_path / "c1") == _candidates(tmp_path / "c2")
    positions = [int(p) for p, _ in _candidates(tmp_path / "c1")]
    assert positions == sorted(positions)
    walk = (tmp_path / "c2" / "p3_oi_local_walk.csv").read_text(encoding="utf-8")
    assert walk.count("download: gone (http 404)") == 1             # decided once, not twice


def test_only_candidates_keep_their_originals(tmp_path: Path) -> None:
    meta = _fixture(tmp_path)
    result = _run(meta, tmp_path / "flickr", tmp_path / "orig", tmp_path / "out",
                  "--target-candidates", "4")
    assert result.returncode == 0, result.stdout + result.stderr
    kept = {p.stem for p in (tmp_path / "orig").iterdir()}
    assert kept == {i for _, i in _candidates(tmp_path / "out")}
    crops = list((tmp_path / "out" / "reals" / "t_oi").glob("*.png"))
    assert len(crops) == 4
    summary = json.loads((tmp_path / "out" / "p3_oi_local_summary.json").read_text("utf-8"))
    assert summary["candidates"] == 4 and summary["max_connections"] == 4


def test_raw_data_inside_the_repository_is_refused(tmp_path: Path) -> None:
    inside = ROOT / ".cache" / "never_created_by_this_test"
    env = {k: v for k, v in os.environ.items() if k != "DBP_DRYRUN"}
    result = subprocess.run(                              # refused before any network use
        [sys.executable, str(SCRIPT), "--metadata", "unused.csv", "--originals", str(inside),
         "--out", str(tmp_path / "out")], capture_output=True, text=True, env=env, check=False)
    assert result.returncode == 1 and "inside REPO_ROOT" in result.stdout + result.stderr
    assert not inside.exists()


def test_the_repository_check_itself() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import p3_oi_local_fetch as local

    assert not local.outside_repo(ROOT / "reports")
    assert local.outside_repo(ROOT.parent.parent / "kaggle_raw" / "p3_oi_originals")


def test_a_refused_row_stays_undecided_and_is_fetched_again_in_order(tmp_path: Path) -> None:
    """Regression, local run 1 (2026-10-14): 15 rows refused with 429 were logged as
    rejected. A refusal says nothing about the image: the row must stay undecided, be
    fetched again on resume, and the final candidates must equal an uninterrupted walk."""
    meta = _fixture(tmp_path)
    flickr = tmp_path / "flickr"
    reference = _run(meta, flickr, tmp_path / "o1", tmp_path / "c1", "--target-candidates", "5")
    assert reference.returncode == 0, reference.stdout + reference.stderr
    expected = _candidates(tmp_path / "c1")
    refused = flickr / f"{int(expected[0][1], 16):016x}_o.jpg.429"
    refused.write_text("10", encoding="utf-8")   # 4 attempts x 2 tries this session, then 2 more
    first = _run(meta, flickr, tmp_path / "o2", tmp_path / "c2", "--target-candidates", "5")
    assert first.returncode == 1 and "INCOMPLETE" in first.stdout + first.stderr
    walk = (tmp_path / "c2" / "p3_oi_local_walk.csv").read_text(encoding="utf-8")
    assert "429" not in walk and expected[0][1] not in walk            # undecided, not rejected
    summary = json.loads((tmp_path / "c2" / "p3_oi_local_summary.json").read_text("utf-8"))
    assert summary["undecided_positions"] == [int(expected[0][0])]
    second = _run(meta, flickr, tmp_path / "o2", tmp_path / "c2", "--target-candidates", "5")
    assert second.returncode == 0, second.stdout + second.stderr
    assert _candidates(tmp_path / "c2") == expected                    # extra one trimmed
    kept = {p.stem for p in (tmp_path / "o2").iterdir()}
    assert kept == {i for _, i in expected}


def test_rows_logged_as_failed_by_the_old_version_are_fetched_again(tmp_path: Path) -> None:
    meta = _fixture(tmp_path)
    flickr = tmp_path / "flickr"
    out = tmp_path / "c"
    out.mkdir()
    (out / "p3_oi_local_walk.csv").write_text(
        "position,oi_image_id,host,outcome\n1,x,c1.staticflickr.com,download failed (http 429)\n",
        encoding="utf-8")
    result = _run(meta, flickr, tmp_path / "o", out, "--target-candidates", "3")
    assert result.returncode == 0, result.stdout + result.stderr
    walk = (out / "p3_oi_local_walk.csv").read_text(encoding="utf-8")
    assert "download failed" not in walk and walk.splitlines()[1].startswith("1,")


# ------------------------------------------- pacing and sessions (user, 2026-10-14) ----


def test_min_interval_spaces_the_starts_of_requests(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from dbp.data import open_images as oi

    clock = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(round(seconds, 6))
        clock["t"] += seconds

    class Instant(oi.Fetcher):
        def _open(self, url: str):
            import io
            return io.BytesIO(b"x")

    fetcher = Instant(throttle=oi.Throttle(sleep=sleep), min_interval=6.0,
                      clock=lambda: clock["t"], max_parallel=1)
    for n in range(3):
        fetcher.fetch("https://farm1.staticflickr.com/1/a_o.jpg", tmp_path / f"pace_{n}.jpg")
    assert slept == [6.0, 6.0] and fetcher.requests_started == 3   # 2 gaps for 3 requests


def test_not_before_refuses_an_early_start(tmp_path: Path) -> None:
    meta = _fixture(tmp_path)
    result = _run(meta, tmp_path / "flickr", tmp_path / "o", tmp_path / "c",
                  "--not-before", "2999-01-01T00:00Z")
    assert result.returncode == 2 and "NOT STARTED" in result.stdout + result.stderr
    assert not (tmp_path / "c").exists()


def test_a_timed_session_ends_cleanly_and_status_reports_it(tmp_path: Path) -> None:
    meta = _fixture(tmp_path)
    flickr = tmp_path / "flickr"
    timed = _run(meta, flickr, tmp_path / "o", tmp_path / "c", "--target-candidates", "5",
                 "--max-minutes", "0")
    assert timed.returncode == 0 and "time limit" in timed.stdout + timed.stderr
    status = subprocess.run([sys.executable, str(SCRIPT), "--originals", str(tmp_path / "o"),
                             "--out", str(tmp_path / "c"), "--status"],
                            capture_output=True, text=True, check=False)
    assert status.returncode == 0 and "rows decided 0" in status.stdout
    assert "ended: session time limit" in status.stdout
    rest = _run(meta, flickr, tmp_path / "o", tmp_path / "c", "--target-candidates", "5")
    assert rest.returncode == 0
    with (tmp_path / "c" / "p3_oi_local_sessions.csv").open(encoding="utf-8") as handle:
        sessions = list(csv.DictReader(handle))
    assert [s["ended"] for s in sessions] == ["session time limit", "target met"]


def test_ctrl_c_saves_the_state_and_the_resume_is_exact(tmp_path: Path, monkeypatch) -> None:
    """An interrupt mid-batch: earlier batches are kept, the batch in progress is undecided,
    its files are swept, exit 130, and the same command resumes to the uninterrupted result."""
    meta = _fixture(tmp_path)
    flickr = tmp_path / "flickr"
    reference = _run(meta, flickr, tmp_path / "o1", tmp_path / "c1", "--target-candidates", "5")
    assert reference.returncode == 0
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import p3_oi_local_fetch as local
    from dbp.data import open_images as oi

    calls = {"n": 0}
    original_open = oi.LocalFetcher._open

    def interrupting_open(self, url):
        calls["n"] += 1
        if calls["n"] == 4:                   # inside the second batch of two
            raise KeyboardInterrupt
        return original_open(self, url)

    monkeypatch.setattr(oi.LocalFetcher, "_open", interrupting_open)
    monkeypatch.setenv("DBP_DRYRUN", "1")
    monkeypatch.setattr(sys, "argv", [
        "p3_oi_local_fetch.py", "--metadata", str(meta), "--originals", str(tmp_path / "o2"),
        "--out", str(tmp_path / "c2"), "--min-free-gb", "0.01", "--local-source", str(flickr),
        "--batch", "2", "--target-candidates", "5"])
    assert local.main() == 130
    saved = _candidates(tmp_path / "c2")
    summary = json.loads((tmp_path / "c2" / "p3_oi_local_summary.json").read_text("utf-8"))
    assert summary["rows_decided"] == 2       # the first batch only; the second is undecided
    assert {p.stem for p in (tmp_path / "o2").iterdir()} == {i for _, i in saved}
    monkeypatch.setattr(oi.LocalFetcher, "_open", original_open)
    rest = _run(meta, flickr, tmp_path / "o2", tmp_path / "c2", "--target-candidates", "5")
    assert rest.returncode == 0, rest.stdout + rest.stderr
    assert _candidates(tmp_path / "c2") == _candidates(tmp_path / "c1")
