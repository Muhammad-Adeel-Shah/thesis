"""I-053: two parallel jobs must never write the same path.

In the S3 recovery both jobs ran as ``shard 0/1`` and both wrote
``reports/p2_probe_timing_shard0.json``; the job that finished last replaced the other's
timing record. The features were unaffected, but the measured cost of two members
survived only in a log. The same pattern was waiting in Session E, where heads, fusions
and analysis outputs were keyed by protocol only, so the second P-A seed would have
overwritten the first seed's fitted fusions.

These tests build the output paths of parallel jobs the way the scripts build them and
assert that no path is shared.
"""

from __future__ import annotations

import importlib.util
import itertools
import re
import sys
from pathlib import Path

import pytest

THESIS_DBP = Path(__file__).resolve().parents[1]
SCRIPTS = THESIS_DBP / "scripts"
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.utils.logging import log_file_name  # noqa: E402
from dbp.utils.run_paths import (  # noqa: E402
    analysis_dir,
    device_tag,
    head_paths,
    run_id,
    s2_run_paths,
    score_paths,
    worker_log_paths,
)

#: The four Session D runs, in queue order.
RUNS = [("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0)]


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _job_paths(protocol: str, seed: int, device: str, root: Path) -> list[Path]:
    """Every path one (run, device) job writes, across Sessions D and E."""
    rid = run_id(protocol, seed)
    paths = s2_run_paths(root / "outputs" / "p2" / "s2", root / "reports", rid).all()
    paths += worker_log_paths(root / "logs", root / "tmp", rid, device).all()
    paths += head_paths(root / "heads", root / "reports", rid).all()
    paths += score_paths(root / "reports" / "p2", rid).all()
    paths.append(analysis_dir(root / "reports" / "p2", rid))
    return paths


# ------------------------------------------------- the S3 regression (I-053) ----

def test_two_probe_jobs_that_are_both_shard_zero_write_different_timing_files(
    tmp_path: Path,
) -> None:
    """The exact S3 recovery layout: two jobs, each ``shard 0/1``, on GPUs 0 and 1."""
    shards = _load("p2_run_shards")
    probe = _load("p2_probe_features")
    jobs = [shards.parse_job("0:movq_k22", 0, tmp_path),
            shards.parse_job("1:qwen_image,realesrgan_x4", 1, tmp_path)]
    args = type("A", (), {"worker": "w.py", "images": "i", "root": "r",
                          "panel_root": None, "cache_dir": None})()
    tags = []
    for job in jobs:
        command, _ = shards.command_for(job, args)
        tags.append(command[command.index("--tag") + 1])
    timing = [probe.timing_report_name(tag) for tag in tags]
    assert len(set(timing)) == 2, timing
    assert len({job.log_path for job in jobs}) == 2
    assert all("gpu" in tag for tag in tags), "the GPU index must be in the name"


def test_the_old_shard_name_would_have_collided() -> None:
    """What the scripts did before the fix: both jobs default to shard 0."""
    probe = _load("p2_probe_features")
    assert probe.timing_report_name("shard0") == probe.timing_report_name("shard0")


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "a b"])
def test_a_timing_tag_must_be_a_plain_name(bad: str) -> None:
    with pytest.raises(ValueError):
        _load("p2_probe_features").timing_report_name(bad)


# ------------------------------------------------------ Session D and E paths ----

def test_two_parallel_s2_jobs_share_no_path(tmp_path: Path) -> None:
    """Seed 0 on GPU 0 and seed 1 on GPU 1 -- the pair that used to collide in E."""
    first = _job_paths("p_a_sd1", 0, "0", tmp_path)
    second = _job_paths("p_a_sd1", 1, "1", tmp_path)
    assert not set(first) & set(second), sorted(set(first) & set(second))


def test_all_four_runs_share_no_path_on_any_device_pairing(tmp_path: Path) -> None:
    for (a, b) in itertools.combinations(RUNS, 2):
        for da, db in (("0", "1"), ("1", "0")):
            shared = set(_job_paths(*a, da, tmp_path)) & set(_job_paths(*b, db, tmp_path))
            assert not shared, (a, b, sorted(shared))


def test_within_one_job_every_path_is_distinct(tmp_path: Path) -> None:
    paths = _job_paths("4ch_only", 0, "1", tmp_path)
    assert len(paths) == len(set(paths))


def test_a_run_resumed_on_the_other_gpu_finds_its_checkpoint(tmp_path: Path) -> None:
    """The checkpoint is keyed by run id alone; only the logs carry the device."""
    rid = run_id("full_bank", 0)
    on0 = s2_run_paths(tmp_path / "s2", tmp_path / "r", rid)
    on1 = s2_run_paths(tmp_path / "s2", tmp_path / "r", rid)
    assert on0.checkpoint == on1.checkpoint
    logs0 = worker_log_paths(tmp_path / "logs", tmp_path / "tmp", rid, "0")
    logs1 = worker_log_paths(tmp_path / "logs", tmp_path / "tmp", rid, "1")
    assert not set(logs0.all()) & set(logs1.all())


def test_heads_are_keyed_by_seed_not_only_protocol(tmp_path: Path) -> None:
    seed0 = head_paths(tmp_path, tmp_path, run_id("p_a_sd1", 0))
    seed1 = head_paths(tmp_path, tmp_path, run_id("p_a_sd1", 1))
    assert seed0.report != seed1.report and seed0.s3_set != seed1.s3_set


def test_two_processes_in_one_second_get_different_log_files() -> None:
    name = "p2_probe_features"
    assert log_file_name(name, pid=101) != log_file_name(name, pid=102)


@pytest.mark.parametrize("bad", ["", "a/b", "p a", "x:y"])
def test_a_protocol_must_be_a_plain_name(bad: str) -> None:
    with pytest.raises(ValueError):
        run_id(bad, 0)


def test_device_tag_rejects_anything_but_cpu_or_an_index() -> None:
    assert device_tag("0") == "gpu0" and device_tag("cpu") == "cpu"
    with pytest.raises(ValueError):
        device_tag("cuda:0")


# --------------------------------------------- no script builds its own names ----

#: An f-string output name that interpolates the protocol directly is how the Session E
#: collision was written. Every output name goes through dbp.utils.run_paths instead.
HAND_BUILT = re.compile(r'f"[^"\n]*\{args\.(protocol|seed)\}[^"\n]*\.(json|npz|pt|csv|log)"')


@pytest.mark.parametrize(
    "script", ["p2_train_s2", "p2_fit_heads", "p2_score", "p2_analyze", "p2_probe_features"]
)
def test_no_script_hand_builds_a_protocol_keyed_output_name(script: str) -> None:
    source = (SCRIPTS / f"{script}.py").read_text(encoding="utf-8")
    assert not HAND_BUILT.findall(source), HAND_BUILT.findall(source)
