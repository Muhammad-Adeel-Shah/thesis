"""Regression tests for the runbook tooling (CLAUDE.md section 12).

Each test here exists because the failure it names reached Kaggle:

* a shard died on a missing module (``spandrel``) and the other ran on alone for 3.5
  hours with one GPU idle -- the runner must stop everything at the first failure;
* the persist cell raised ``ModuleNotFoundError`` and marked a ten-hour run as failed --
  the report step must not be able to fail;
* ``gen_matched`` was attached at a stale version (2,475 of 2,488) -- regenerated files
  must be checked against the committed hashes, not against themselves;
* a CRLF bootstrap would have failed on Kaggle's bash while Git Bash ran it happily --
  the code zip must hold git's bytes;
* runbook cells pip-installed single packages and imported ``dbp`` before putting it on
  the path -- the runbook lint forbids both.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

THESIS_DBP = Path(__file__).resolve().parents[1]
SCRIPTS = THESIS_DBP / "scripts"
RUNBOOK = THESIS_DBP / "env" / "kaggle_run_p2_s3_recovery.py"
#: Every runbook the CLAUDE.md section 12 lint applies to.
RUNBOOKS = sorted((THESIS_DBP / "env").glob("kaggle_run_*.py"))
EXPECT = THESIS_DBP / "env" / "p2_s3_recovery_expect.json"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _run(script: str, *args: object, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True, text=True, timeout=timeout, cwd=THESIS_DBP,
    )


# ------------------------------------------------- a fake stage worker ----

FAKE_WORKER = '''
import argparse, sys, time
parser = argparse.ArgumentParser()
parser.add_argument("--members", default="")
args, _ = parser.parse_known_args()
print(f"fake worker for {args.members}", flush=True)
for i in range(40):
    print(f"line {i}", flush=True)
if "bad" in args.members:
    time.sleep(0.5)
    print("Traceback: ModuleNotFoundError: No module named 'spandrel'", flush=True)
    sys.exit(3)
if "slow" in args.members:
    time.sleep(60)
sys.exit(0)
'''


@pytest.fixture
def worker(tmp_path: Path) -> Path:
    path = tmp_path / "fake_worker.py"
    path.write_text(FAKE_WORKER, encoding="utf-8")
    return path


def _shards(tmp_path: Path, worker: Path, *jobs: str, deadline: float = 1.0):
    status = tmp_path / "reports" / "status.json"
    started = time.perf_counter()
    result = _run(
        "p2_run_shards.py", *[a for j in jobs for a in ("--job", j)],
        "--images", tmp_path / "list.csv", "--root", tmp_path,
        "--deadline-hours", deadline, "--poll-seconds", 0.2,
        "--log-dir", tmp_path / "logs", "--probe-dir", tmp_path / "probe",
        "--status", status, "--worker", worker,
    )
    return result, status, time.perf_counter() - started


# ------------------------------------------------------ p2_run_shards ----


def test_one_failing_job_stops_the_others_at_once(tmp_path: Path, worker: Path) -> None:
    """The first S3 commit: one shard died and the other ran 3.5 more hours alone."""
    result, status, elapsed = _shards(tmp_path, worker, "cpu:bad", "cpu:slow")
    assert result.returncode == 1
    assert elapsed < 30, f"took {elapsed:.0f}s -- the healthy job was not stopped"
    record = json.loads(status.read_text())
    assert record["outcome"] == "FAILED"
    assert "bad" in record["failed_job"]
    slow = next(j for j in record["jobs"] if "slow" in j["members"])
    assert slow["returncode"] != 0, "the healthy job was left running"


def test_a_failure_prints_the_failing_jobs_last_lines(tmp_path: Path, worker: Path) -> None:
    result, _, _ = _shards(tmp_path, worker, "cpu:bad", "cpu:slow")
    assert "No module named 'spandrel'" in result.stdout
    assert "last 30 lines" in result.stdout


def test_the_status_and_logs_are_written_on_failure(tmp_path: Path, worker: Path) -> None:
    _, status, _ = _shards(tmp_path, worker, "cpu:bad", "cpu:slow")
    assert status.is_file()
    assert (status.parent / "probe_job0_cpu.log").is_file()


def test_the_deadline_stops_every_job(tmp_path: Path, worker: Path) -> None:
    result, status, elapsed = _shards(tmp_path, worker, "cpu:slow", deadline=0.0005)
    assert result.returncode == 2
    assert json.loads(status.read_text())["outcome"] == "DEADLINE"
    assert elapsed < 30


def test_all_jobs_succeeding_completes(tmp_path: Path, worker: Path) -> None:
    result, status, _ = _shards(tmp_path, worker, "cpu:a", "cpu:b")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(status.read_text())["outcome"] == "COMPLETE"


def test_two_jobs_on_one_gpu_are_refused(tmp_path: Path, worker: Path) -> None:
    result, _, _ = _shards(tmp_path, worker, "0:a", "0:b")
    assert result.returncode == 1
    assert "share a CUDA device" in result.stdout + result.stderr


def test_a_cuda_job_sees_only_its_own_gpu(tmp_path: Path) -> None:
    shards = _load("p2_run_shards")
    job = shards.parse_job("1:qwen_image", 0, tmp_path)
    args = type("A", (), {"worker": "w.py", "images": "i", "root": "r", "panel_root": None,
                          "cache_dir": None})()
    command, env = shards.command_for(job, args)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert command[command.index("--device") + 1] == "cuda"


# ----------------------------------------------------------- p2_report ----


def test_the_report_cannot_fail_even_with_nothing_to_report(tmp_path: Path) -> None:
    """The first S3 commit's persist cell raised and marked ten hours as failed."""
    result = _run("p2_report.py", "--status", tmp_path / "absent.json",
                  "--probe-dir", tmp_path / "absent", "--bank", tmp_path / "absent.yaml",
                  "--reports", tmp_path / "absent", "--zip", tmp_path / "out")
    assert result.returncode == 0
    assert "unavailable" in result.stdout


def test_the_report_survives_bad_arguments() -> None:
    assert _run("p2_report.py", "--no-such-flag").returncode == 0


# -------------------------------------------------------- p2_preflight ----


def test_a_missing_module_fails_the_preflight(tmp_path: Path) -> None:
    result = _run("p2_preflight.py", "--stage", "t", "--imports", "json,no_such_module_xyz",
                  "--reports", tmp_path)
    assert result.returncode == 1
    assert "no_such_module_xyz" in result.stdout + result.stderr


def test_a_missing_input_path_fails_the_preflight(tmp_path: Path) -> None:
    result = _run("p2_preflight.py", "--stage", "t",
                  "--require", f"panel={tmp_path / 'nowhere'}", "--reports", tmp_path)
    assert result.returncode == 1


def _list(tmp_path: Path) -> Path:
    path = tmp_path / "list.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "relative_path", "root"])
        writer.writeheader()
        writer.writerow({"image_id": "a", "relative_path": "x/a.png", "root": "corpus"})
    return path


def test_a_member_that_cannot_run_fails_the_smoke_test(tmp_path: Path, worker: Path) -> None:
    """realesrgan_x4 could not load; the smoke test must find that in minutes."""
    result = _run("p2_preflight.py", "--stage", "t", "--smoke", "bad_member",
                  "--smoke-list", _list(tmp_path), "--root", tmp_path,
                  "--worker", worker, "--reports", tmp_path)
    assert result.returncode == 1
    assert "No module named 'spandrel'" in result.stdout


def test_a_working_member_passes_the_smoke_test(tmp_path: Path, worker: Path) -> None:
    result = _run("p2_preflight.py", "--stage", "t", "--smoke", "fine",
                  "--smoke-list", _list(tmp_path), "--root", tmp_path,
                  "--worker", worker, "--reports", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_models_are_not_loaded_when_a_cheap_check_failed(tmp_path: Path, worker: Path) -> None:
    result = _run("p2_preflight.py", "--stage", "t", "--imports", "no_such_module_xyz",
                  "--smoke", "fine", "--smoke-list", _list(tmp_path), "--root", tmp_path,
                  "--worker", worker, "--reports", tmp_path)
    record = json.loads((tmp_path / "p2_preflight_t.json").read_text())
    smoke = next(c for c in record["checks"] if c["name"] == "smoke fine")
    assert result.returncode == 1 and "skipped" in smoke["detail"]


def _gen_matched(tmp_path: Path) -> tuple[Path, Path]:
    from dbp.data.manifest import file_sha256

    directory = tmp_path / "gen_matched"
    directory.mkdir()
    rows = []
    for i in range(3):
        path = directory / f"coco_{i}.jpg"
        path.write_bytes(f"jpeg {i}".encode())
        rows.append({"gen_id": f"coco_{i}",
                     "jpg_path": f"/kaggle/working/old/gen_matched/coco_{i}.jpg",
                     "jpg_sha256": file_sha256(path)})
    reference = tmp_path / "reference.csv"
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gen_id", "jpg_path", "jpg_sha256"])
        writer.writeheader()
        writer.writerows(rows)
    return reference, directory


def _check_gen_matched(tmp_path: Path, reference: Path, directory: Path, n: int):
    return _run("p2_preflight.py", "--stage", "t", "--gen-matched-reference", reference,
                "--gen-matched-dir", directory, "--expect-gen-matched", n,
                "--reports", tmp_path)


def test_regenerated_files_matching_the_reference_pass(tmp_path: Path) -> None:
    reference, directory = _gen_matched(tmp_path)
    assert _check_gen_matched(tmp_path, reference, directory, 3).returncode == 0


def test_one_changed_byte_in_gen_matched_fails(tmp_path: Path) -> None:
    """The nine restored members were computed on the original files."""
    reference, directory = _gen_matched(tmp_path)
    (directory / "coco_1.jpg").write_bytes(b"different")
    result = _check_gen_matched(tmp_path, reference, directory, 3)
    assert result.returncode == 1 and "differ" in result.stdout + result.stderr


def test_a_missing_gen_matched_file_fails(tmp_path: Path) -> None:
    """2,475 of 2,488 is how the stale copy looked."""
    reference, directory = _gen_matched(tmp_path)
    (directory / "coco_2.jpg").unlink()
    assert _check_gen_matched(tmp_path, reference, directory, 3).returncode == 1


def test_the_wrong_count_fails_even_if_every_hash_matches(tmp_path: Path) -> None:
    reference, directory = _gen_matched(tmp_path)
    assert _check_gen_matched(tmp_path, reference, directory, 4).returncode == 1


def _caches(tmp_path: Path, members: list[str], ids: list[str]) -> tuple[Path, Path]:
    """A list of ``ids`` and one valid probe cache per member, built on it."""
    probe = _load("p2_probe_features")
    from dbp.streams.s3_probe import FEATURE_NAMES

    lst = tmp_path / "probe.csv"
    with lst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "relative_path"])
        writer.writeheader()
        for i in ids:
            writer.writerow({"image_id": i, "relative_path": f"x/{i}.png"})
    digest = probe.image_list_digest(probe.read_image_list(lst))
    out = tmp_path / "probe"
    out.mkdir()
    for member in members:
        npz = out / f"p2_probe_{member}.npz"
        np.savez_compressed(
            npz, features=np.ones((len(ids), len(FEATURE_NAMES)), dtype=np.float32),
            image_ids=np.array(ids), feature_names=np.array(FEATURE_NAMES), member=member,
        )
        probe.write_sidecar(npz, len(ids), 320, digest, member)
    return lst, out


def _verify(lst: Path, out: Path, members: str, tmp_path: Path) -> subprocess.CompletedProcess:
    return _run("p2_preflight.py", "--stage", "v", "--list", lst, "--verify-caches", out,
                "--members", members, "--reports", tmp_path)


def test_valid_caches_pass_and_record_their_hashes(tmp_path: Path) -> None:
    lst, out = _caches(tmp_path, ["sd15", "flux"], ["a", "b", "c"])
    assert _verify(lst, out, "sd15,flux", tmp_path).returncode == 0
    record = json.loads((tmp_path / "p2_preflight_v.json").read_text(encoding="utf-8"))
    check = next(c for c in record["checks"] if c["name"] == "probe caches")
    members = check["data"]["members"]
    assert set(members) == {"sd15", "flux"}
    assert all(len(m["sha256"]) == 64 for m in members.values())


def test_a_cache_changed_after_its_sidecar_fails(tmp_path: Path) -> None:
    lst, out = _caches(tmp_path, ["sd15"], ["a", "b"])
    np.savez_compressed(out / "p2_probe_sd15.npz", features=np.zeros((2, 20), dtype=np.float32),
                        image_ids=np.array(["a", "b"]), feature_names=np.array(["x"] * 20))
    result = _verify(lst, out, "sd15", tmp_path)
    assert result.returncode == 1 and "sidecar sha256" in result.stdout


def test_a_cache_in_another_row_order_fails(tmp_path: Path) -> None:
    lst, out = _caches(tmp_path, ["sd15"], ["a", "b"])
    lst.write_text("image_id,relative_path\nb,x/b.png\na,x/a.png\n", encoding="utf-8")
    result = _verify(lst, out, "sd15", tmp_path)
    assert result.returncode == 1 and "image order" in result.stdout


def test_a_missing_member_fails(tmp_path: Path) -> None:
    lst, out = _caches(tmp_path, ["sd15"], ["a"])
    result = _verify(lst, out, "sd15,movq_k22", tmp_path)
    assert result.returncode == 1 and "movq_k22: cache or sidecar missing" in result.stdout


# --------------------------------------------- restore and the list digest ----


def _bundle(tmp_path: Path, digest_of: list[dict[str, str]] | None = None) -> Path:
    probe = _load("p2_probe_features")
    bundle = tmp_path / "bundle"
    rows = [{"image_id": f"i{n}", "relative_path": f"v/i{n}.png", "label": "fake",
             "view": "v", "root": "panel"} for n in range(3)]
    reports = bundle / "reports"
    reports.mkdir(parents=True)
    with (reports / "p2_probe_images.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (bundle / "s1").mkdir()
    npz = bundle / "probe" / "p2_probe_sd15.npz"
    npz.parent.mkdir()
    np.savez_compressed(npz, features=np.zeros((3, 20), dtype=np.float32))
    digest = probe.image_list_digest(digest_of if digest_of is not None else rows)
    probe.write_sidecar(npz, 3, probe.CROP_SIZE, digest, "sd15")
    return bundle


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    return repo


def test_a_member_built_on_this_list_is_restored(tmp_path: Path) -> None:
    result = _run("p2_restore_partial.py", "--bundle-root", _bundle(tmp_path),
                  "--repo", _repo(tmp_path), "--expect-members", 1)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_member_built_on_another_list_is_refused(tmp_path: Path) -> None:
    other = [{"image_id": "z", "relative_path": "z.png"}]
    result = _run("p2_restore_partial.py", "--bundle-root", _bundle(tmp_path, other),
                  "--repo", _repo(tmp_path), "--expect-members", 1)
    assert result.returncode == 1


def test_the_bundle_is_found_one_level_down(tmp_path: Path) -> None:
    """A zip may be presented at the dataset root or inside a folder."""
    bundle = _bundle(tmp_path)
    nested = tmp_path / "dataset" / "s3_partial"
    nested.parent.mkdir()
    bundle.rename(nested)
    result = _run("p2_restore_partial.py", "--bundle-root", tmp_path / "dataset",
                  "--repo", _repo(tmp_path), "--expect-members", 1)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_report_that_differs_from_the_repos_is_not_overwritten(tmp_path: Path) -> None:
    """Session A: a stale manifest replaced the committed one by being copied over it."""
    repo = _repo(tmp_path)
    (repo / "reports" / "p2_probe_images.csv").write_text("committed\n", encoding="utf-8")
    result = _run("p2_restore_partial.py", "--bundle-root", _bundle(tmp_path),
                  "--repo", repo, "--expect-members", 1)
    assert result.returncode == 1
    assert (repo / "reports" / "p2_probe_images.csv").read_text() == "committed\n"


def test_the_list_digest_is_the_one_the_sidecars_record() -> None:
    """Restore, pre-flight and the run must agree on what 'the same list' means."""
    probe = _load("p2_probe_features")
    rows = [{"image_id": "a", "relative_path": "p/a.png"}]
    import hashlib

    assert probe.image_list_digest(rows) == hashlib.sha256(b"a\tp/a.png").hexdigest()


# ----------------------------------------------------------- stage corpus ----


def test_the_corpus_link_resolves_and_is_counted(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "t_real").mkdir(parents=True)
    for i in range(3):
        (source / "t_real" / f"{i}.png").write_bytes(b"x")
    result = _run("p2_stage_corpus.py", "--root", tmp_path / "p1",
                  "--link", f"reals={source}", "--expect", "reals=3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "p1" / "reals" / "t_real" / "0.png").is_file()


def test_a_wrong_corpus_count_fails(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.png").write_bytes(b"x")
    result = _run("p2_stage_corpus.py", "--root", tmp_path / "p1",
                  "--link", f"reals={source}", "--expect", "reals=2")
    assert result.returncode == 1


def test_a_real_directory_is_never_replaced_by_a_link(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "p1" / "reals").mkdir(parents=True)
    (tmp_path / "p1" / "reals" / "keep.png").write_bytes(b"data")
    result = _run("p2_stage_corpus.py", "--root", tmp_path / "p1", "--link", f"reals={source}")
    assert result.returncode == 1
    assert (tmp_path / "p1" / "reals" / "keep.png").is_file()


# ------------------------------------------------------------ code zip ----


def test_no_shell_or_python_file_in_the_commit_has_a_carriage_return() -> None:
    """A CRLF bootstrap fails on Kaggle's bash; Git Bash on Windows runs it anyway."""
    zipper = _load("build_code_zip")
    paths = zipper.tracked("HEAD")
    assert zipper.check_line_endings(zipper.blobs("HEAD", paths)) == []


def test_the_line_ending_check_catches_a_crlf_script() -> None:
    zipper = _load("build_code_zip")
    bad = {"thesis_dbp/env/x.sh": b"set -e\r\necho hi\r\n", "thesis_dbp/a.csv": b"a\r\n"}
    assert zipper.check_line_endings(bad) == ["thesis_dbp/env/x.sh"]


def test_the_zip_is_byte_identical_to_the_commit(tmp_path: Path) -> None:
    import zipfile

    zipper = _load("build_code_zip")
    out = tmp_path / "code.zip"
    result = _run("build_code_zip.py", "--out", out, "--ref", "HEAD")
    assert result.returncode == 0, result.stdout + result.stderr
    paths = zipper.tracked("HEAD")
    contents = zipper.blobs("HEAD", paths)
    with zipfile.ZipFile(out) as archive:
        for path in paths:
            assert archive.read(path) == contents[path], path
        assert archive.read("thesis_dbp/COMMIT.txt").decode().split()[0] == \
            zipper.resolve("HEAD")[0]


# ---------------------------------------------------------- runbook lint ----


def _cells(runbook: Path = RUNBOOK) -> list:
    return _load("runbook_dryrun").code_cells(
        _load("runbook_dryrun").parse_cells(runbook.read_text(encoding="utf-8")))


def _code(runbook: Path = RUNBOOK) -> str:
    return "\n".join(c.text for c in _cells(runbook))


def test_both_percent_format_runbooks_are_linted() -> None:
    assert {p.name for p in RUNBOOKS} >= {"kaggle_run_p2_s3_recovery.py",
                                          "kaggle_run_p2_s2.py"}


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_the_runbook_parses_into_cells(runbook: Path) -> None:
    assert len(_cells(runbook)) >= 5


def test_code_before_the_first_cell_marker_is_refused() -> None:
    with pytest.raises(ValueError, match="before the first"):
        _load("runbook_dryrun").parse_cells("import os\n# %% Cell 1\nx = 1\n")


def test_markdown_cells_are_not_executed() -> None:
    cells = _load("runbook_dryrun").parse_cells(
        "# %% [markdown]\n# hello\n# %% Cell 1\nx = 1\n")
    assert [c.markdown for c in cells] == [True, False]


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_the_environment_comes_only_from_the_bootstrap(runbook: Path) -> None:
    """lpips was pip-installed by hand, spandrel was not, and one GPU idled 3.5 hours."""
    code = _code(runbook)
    assert "kaggle_bootstrap.sh" in code
    assert "pip install" not in code and "pip uninstall" not in code


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_no_cell_imports_dbp(runbook: Path) -> None:
    """The persist cell imported dbp before putting it on the path."""
    for cell in _cells(runbook):
        for line in cell.text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("from dbp", "import dbp")), (cell.title, line)


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_no_cell_uses_shell_magics(runbook: Path) -> None:
    """``!`` lines cannot run in the local dry-run, so they would go untested."""
    for cell in _cells(runbook):
        for line in cell.text.splitlines():
            assert not line.lstrip().startswith(("!", "%")), (cell.title, line)


def test_every_member_the_run_starts_is_smoke_tested_first() -> None:
    expect = json.loads(EXPECT.read_text(encoding="utf-8"))
    run = {m for _, members in expect["jobs"] for m in members.split(",")}
    assert run == set(expect["smoke_members"])


def test_the_jobs_use_distinct_gpus() -> None:
    expect = json.loads(EXPECT.read_text(encoding="utf-8"))
    devices = [d for d, _ in expect["jobs"] if d != "cpu"]
    assert len(devices) == len(set(devices))


def test_preflight_runs_before_the_long_job() -> None:
    code = _code()
    assert code.index('py("p2_preflight.py"') < code.index('py("p2_run_shards.py"')


def test_the_data_preflight_follows_gen_matched_regeneration() -> None:
    """Checking gen_matched before it is regenerated would check nothing."""
    code = _code()
    matcher = code.index('py("p1_match_generations.py"')
    assert matcher < code.index('"--gen-matched-reference"') < code.index('py("p2_run_shards.py"')


def test_generated_commit_file_is_never_taken_from_git() -> None:
    zipper = _load("build_code_zip")
    assert zipper.GENERATED not in zipper.tracked("HEAD")


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_gen_matched_is_checked_against_the_read_only_committed_manifest(runbook: Path) -> None:
    """Checking a file against the hash its own writer recorded proves nothing.

    Applies to every runbook that regenerates ``gen_matched``; a runbook that never runs
    the matcher (P3 Panel B reads manifests only) has nothing to check.
    """
    code = _code(runbook)
    if 'py("p1_match_generations.py"' not in code:
        assert "--gen-matched-reference" not in code
        return
    reference = code[code.index("--gen-matched-reference"):].split("\n")[0]
    assert 'INPUTS["code"]' in reference


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_the_report_cell_cannot_raise(runbook: Path) -> None:
    """Rule 5. The report cell is the one titled "report" (every runbook has exactly one)."""
    reports = [c for c in _cells(runbook) if "report" in c.title.lower()]
    assert len(reports) == 1, f"{runbook.name}: {len(reports)} report cells"
    assert "assert" not in reports[0].text and "raise" not in reports[0].text


@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.stem)
def test_a_preflight_runs_before_any_long_job(runbook: Path) -> None:
    code = _code(runbook)
    markers = ('py("p2_run_shards.py"', 'code = py("p2_run_queue.py"',
               'codes = {tag: py("p2_extract_s1.py"', 'py("p3_fetch_eval_sets.py"',
               'py("p3_coco_index.py"', 'py("p3_probe_flickr.py"',
               'code = py("p3_fetch_truefake.py"', 'code = py("p4_weights.py"',
               'code = py("p4_dda_equivalence.py"')
    found = [code.index(m) for m in markers if m in code]
    assert found, f"{runbook.name}: no recognised long job (add its marker here)"
    long_job = min(found)
    assert code.index('py("p2_preflight.py"') < long_job


def test_the_long_job_runs_through_the_fail_fast_runner() -> None:
    code = _code()
    assert "p2_run_shards.py" in code
    assert "Popen" not in code.split("def py(")[1], "a cell starts jobs itself"
