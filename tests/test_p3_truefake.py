"""TrueFake fetch (rule of 2026-10-20): verified archive, CRC-checked members, stop on refusal."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p3_fetch_truefake.py"


def _dryrun():
    alias = "_truefake_dryrun"
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / "runbook_dryrun.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp: Path) -> tuple[Path, Path, int, str, list[dict[str, str]]]:
    archive = tmp / "TrueFake.zip"
    rows = _dryrun().truefake_fixture_zip(archive)
    sample = tmp / "sample.csv"
    with sample.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    blob = archive.read_bytes()
    end = blob.rfind(b"PK\x05\x06")
    _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack("<4s4H2LH", blob[end:end + 22])
    cd = hashlib.sha256(blob[cd_offset:cd_offset + cd_size]).hexdigest()
    return archive, sample, len(blob), cd, rows


def _run(tmp: Path, archive: Path, sample: Path, size: int, cd: str, **env) -> tuple[int, dict]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--sample", str(sample), "--out", str(tmp / "out"),
         "--expect-size", str(size), "--expect-cd-sha256", cd, "--local-zip", str(archive),
         "--workers", "2", "--reports", str(tmp / "reports")],
        capture_output=True, text=True, check=False,
        env=dict(os.environ, DBP_DRYRUN="1", **env))
    status = json.loads((tmp / "reports" / "truefake_fetch_status.json").read_text("utf-8"))
    return done.returncode, status


def test_every_sample_member_is_fetched_byte_identical_and_nothing_else(tmp_path: Path) -> None:
    import zipfile

    archive, sample, size, cd, rows = _fixture(tmp_path)
    code, status = _run(tmp_path, archive, sample, size, cd)
    assert code == 0 and status["outcome"] == "COMPLETE" and status["members_done"] == len(rows)
    with zipfile.ZipFile(archive) as source:
        for row in rows:
            path = tmp_path / "out" / row["member"].removeprefix("TrueFake/")
            assert path.read_bytes() == source.read(row["member"])
    assert not (tmp_path / "out" / "PreSocial/Real/FFHQ/99999.png").exists()
    with (tmp_path / "reports" / "truefake_fetch_manifest.csv").open(encoding="utf-8") as h:
        manifest = list(csv.DictReader(h))
    assert {m["member"] for m in manifest} == {r["member"] for r in rows}


def test_a_refusal_stops_the_run_and_nothing_is_retried(tmp_path: Path) -> None:
    archive, sample, size, cd, rows = _fixture(tmp_path)
    # Reads 1 and 2 are the index; the third read is refused.
    code, status = _run(tmp_path, archive, sample, size, cd, DBP_FAULT_REFUSE_AFTER="3")
    assert code == 3 and status["outcome"] == "REFUSED"
    assert status["refusal"]["status"] == 429
    assert status["members_done"] < len(rows)            # stopped; no workaround


def test_a_different_archive_or_index_is_refused_before_any_member(tmp_path: Path) -> None:
    archive, sample, size, cd, _ = _fixture(tmp_path)
    code, status = _run(tmp_path / "a", archive, sample, size + 1, cd)
    assert code == 1 and "archive size" in status["error"] and status["members_done"] == 0
    code, status = _run(tmp_path / "b", archive, sample, size, "0" * 64)
    assert code == 1 and "central directory sha256" in status["error"]


def test_a_rerun_skips_members_already_verified_on_disk(tmp_path: Path) -> None:
    archive, sample, size, cd, rows = _fixture(tmp_path)
    assert _run(tmp_path, archive, sample, size, cd)[0] == 0
    code, status = _run(tmp_path, archive, sample, size, cd, DBP_FAULT_REFUSE_AFTER="2")
    assert code == 0 and status["members_done"] == len(rows)   # only the index was read


def test_the_runbook_and_rule_say_stop_and_keep_private() -> None:
    text = (ROOT / "env" / "kaggle_run_p3_truefake.py").read_text(encoding="utf-8")
    assert "PRIVATE" in text and "assert code != 3" in text
    script = SCRIPT.read_text(encoding="utf-8")
    assert "cancel_futures=True" in script and "stop.set()" in script
    expect = json.loads((ROOT / "env" / "p3_truefake_expect.json").read_text(encoding="utf-8"))
    assert expect["sample_sha256"].startswith("a8954c2f63e81d95") and expect["sample_rows"] == 16000
    assert expect["cd_sha256"].startswith("869a102f65504dc2")
