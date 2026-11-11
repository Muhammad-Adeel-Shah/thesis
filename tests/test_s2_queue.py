"""Session D tooling: the S2 queue, the trainer's seams, the S2 pre-flight checks.

The queue is tested with a fake trainer and a fake scorer, so these tests prove the
scheduling, failure handling, resume and deadline logic without a model. The real
trainer and scorer are exercised end to end by the runbook dry-run
(``scripts/runbook_dryrun.py --fixture p2_s2``), which kills and resumes every run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

THESIS_DBP = Path(__file__).resolve().parents[1]
SCRIPTS = THESIS_DBP / "scripts"
RUNBOOK = THESIS_DBP / "env" / "kaggle_run_p2_s2.py"
EXPECT = THESIS_DBP / "env" / "p2_s2_expect.json"

# A fake trainer: behaviour chosen by the protocol name; every call is counted.
FAKE_TRAINER = r'''
import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--protocol"); p.add_argument("--seed", type=int)
p.add_argument("--out-dir", type=Path); p.add_argument("--reports-dir", type=Path)
p.add_argument("--smoke", type=int, default=0)
a, _ = p.parse_known_args()
rid = f"{a.protocol}_seed{a.seed}"
calls = Path(os.environ["FAKE_CALLS"]); calls.mkdir(parents=True, exist_ok=True)
(calls / f"{time.time_ns()}_{os.getpid()}.txt").write_text(
    f"train {rid} {os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}")
print(f"training {rid}", flush=True)
for i in range(35):
    print(f"step line {i}", flush=True)
if "bad" in a.protocol:
    print("Traceback: RuntimeError: CUDA out of memory", flush=True); sys.exit(1)
if "under" in a.protocol:
    sys.exit(3)
if "slow" in a.protocol:
    time.sleep(float(os.environ.get("FAKE_SLOW", "3")))
run = a.out_dir / rid; run.mkdir(parents=True, exist_ok=True)
a.reports_dir.mkdir(parents=True, exist_ok=True)
(a.reports_dir / f"p2_s2_{rid}.json").write_text(json.dumps({"status": "OK"}))
'''

FAKE_INFERER = r'''
import argparse, hashlib, json, os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--protocol"); p.add_argument("--seed", type=int)
p.add_argument("--out-dir", type=Path); p.add_argument("--reports-dir", type=Path)
p.add_argument("--list", action="append", default=[]); p.add_argument("--limit", type=int)
a, _ = p.parse_known_args()
rid = f"{a.protocol}_seed{a.seed}"
import time
(Path(os.environ["FAKE_CALLS"]) / f"{time.time_ns()}_{os.getpid()}.txt").write_text(
    f"infer {rid} {os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu')}")
scores = a.out_dir / rid / "scores"; scores.mkdir(parents=True, exist_ok=True)
lists = {}
for item in a.list:
    name = item.split("=")[0]
    path = scores / f"{name}.csv"; path.write_text("image_id,score\nx,0.1\n")
    lists[name] = {"rows": 1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
(a.reports_dir / f"p2_s2_scores_{rid}.json").write_text(
    json.dumps({"run_id": rid, "limit": a.limit, "lists": lists}))
'''


@pytest.fixture
def fakes(tmp_path: Path) -> dict[str, Path]:
    trainer, inferer = tmp_path / "fake_train.py", tmp_path / "fake_infer.py"
    trainer.write_text(FAKE_TRAINER, encoding="utf-8")
    inferer.write_text(FAKE_INFERER, encoding="utf-8")
    return {"trainer": trainer, "inferer": inferer, "calls": tmp_path / "calls"}


def _queue(tmp_path: Path, fakes: dict[str, Path], runs: list[str], devices: str = "0,1",
           *extra: str, deadline: float = 1.0, env: dict[str, str] | None = None):
    root = tmp_path / "w"
    args = [sys.executable, str(SCRIPTS / "p2_run_queue.py"), "--devices", devices,
            "--root", root, "--pairs-manifest", "p.csv", "--splits", "s.csv",
            "--list", "panel=panel.csv", "--list", "val=s1.csv", "--only", "val=val",
            "--out-dir", root / "s2", "--reports-dir", root / "reports",
            "--log-dir", root / "logs", "--scratch", root / "scratch",
            "--trainer", fakes["trainer"], "--inferer", fakes["inferer"],
            "--poll-seconds", "0.2", "--deadline-hours", str(deadline), *extra]
    for run in runs:
        args += ["--run", run]
    started = time.perf_counter()
    result = subprocess.run(
        [str(a) for a in args], capture_output=True, text=True, timeout=180,
        env=dict(os.environ, FAKE_CALLS=str(fakes["calls"]), **(env or {})),
    )
    status_name = "p2_s2_smoke_status.json" if "--smoke" in extra else "p2_s2_status.json"
    status_path = root / "reports" / status_name
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    return result, status, time.perf_counter() - started


def _calls(fakes: dict[str, Path]) -> list[list[str]]:
    """Every fake call, in order. One file per call: appends to one file race on Windows."""
    if not fakes["calls"].is_dir():
        return []
    files = sorted(fakes["calls"].glob("*.txt"), key=lambda f: int(f.stem.split("_")[0]))
    return [f.read_text().split() for f in files]


def _forget_calls(fakes: dict[str, Path]) -> None:
    for f in fakes["calls"].glob("*.txt"):
        f.unlink()


# ----------------------------------------------------------------- scheduling ----

def test_four_runs_on_two_gpus_all_complete(tmp_path: Path, fakes: dict[str, Path]) -> None:
    runs = ["p_a_sd1:0", "p_a_sd1:1", "4ch_only:0", "full_bank:0"]
    result, status, _ = _queue(tmp_path, fakes, runs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert status["outcome"] == "COMPLETE"
    assert [r["state"] for r in status["runs"]] == ["complete"] * 4
    trained = [c for c in _calls(fakes) if c[0] == "train"]
    assert {c[2] for c in trained} == {"0", "1"}, "both GPUs must be used"
    for run in status["runs"]:
        infer = [c for c in _calls(fakes) if c[0] == "infer" and c[1] == run["run_id"]]
        train = [c for c in trained if c[1] == run["run_id"]]
        assert infer[0][2] == train[0][2], "a run is scored on the GPU that trained it"


def test_a_free_gpu_takes_the_next_run_at_once(tmp_path: Path, fakes: dict[str, Path]) -> None:
    """GPU 1 finishes quickly and must take both remaining runs while GPU 0 is busy."""
    runs = ["slow_a:0", "fast_b:0", "fast_c:0", "fast_d:0"]
    result, status, _ = _queue(tmp_path, fakes, runs, env={"FAKE_SLOW": "6"})
    assert result.returncode == 0, result.stdout
    by_run = {r["run_id"]: r["device"] for r in status["runs"]}
    slow_device = by_run["slow_a_seed0"]
    assert all(by_run[f"fast_{x}_seed0"] != slow_device for x in "bcd"), by_run


def test_a_failed_run_is_logged_not_retried_and_the_rest_continue(
    tmp_path: Path, fakes: dict[str, Path]
) -> None:
    runs = ["bad_a:0", "p_a_sd1:0", "4ch_only:0"]
    result, status, _ = _queue(tmp_path, fakes, runs)
    assert result.returncode == 1, "the final cell must see the failure"
    assert status["outcome"] == "RUN-FAILED"
    states = {r["run_id"]: r["state"] for r in status["runs"]}
    assert states == {"bad_a_seed0": "failed", "p_a_sd1_seed0": "complete",
                      "4ch_only_seed0": "complete"}
    assert sum(1 for c in _calls(fakes) if c[:2] == ["train", "bad_a_seed0"]) == 1, \
        "a failed run must not be retried"
    assert "CUDA out of memory" in result.stdout, "the failing run's last lines are printed"


def test_an_undertrained_run_is_named_as_such(tmp_path: Path, fakes: dict[str, Path]) -> None:
    result, status, _ = _queue(tmp_path, fakes, ["under_a:0"], "0")
    assert result.returncode == 1
    assert "STOPPED-UNDERTRAINED" in status["runs"][0]["note"]


def test_the_deadline_stops_every_job(tmp_path: Path, fakes: dict[str, Path]) -> None:
    result, status, elapsed = _queue(tmp_path, fakes, ["slow_a:0", "slow_b:0"],
                                     deadline=0.0005, env={"FAKE_SLOW": "120"})
    assert result.returncode == 2
    assert status["outcome"] == "DEADLINE"
    assert all(r["state"] == "stopped" for r in status["runs"])
    assert elapsed < 60


def test_the_status_is_written_while_runs_are_in_flight(
    tmp_path: Path, fakes: dict[str, Path]
) -> None:
    """The status survives any ending: it exists before the queue returns."""
    root = tmp_path / "w"
    env = dict(os.environ, FAKE_CALLS=str(fakes["calls"]), FAKE_SLOW="8")
    args = [sys.executable, str(SCRIPTS / "p2_run_queue.py"), "--devices", "0",
            "--root", root, "--pairs-manifest", "p", "--splits", "s", "--list", "panel=x",
            "--out-dir", root / "s2", "--reports-dir", root / "reports",
            "--log-dir", root / "logs", "--trainer", fakes["trainer"],
            "--inferer", fakes["inferer"], "--poll-seconds", "0.2", "--run", "slow_a:0"]
    process = subprocess.Popen([str(a) for a in args], env=env, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        status = root / "reports" / "p2_s2_status.json"
        for _ in range(100):
            if status.is_file() and "training" in status.read_text(encoding="utf-8"):
                break
            time.sleep(0.1)
        assert json.loads(status.read_text(encoding="utf-8"))["runs"][0]["state"] == "training"
    finally:
        process.wait(timeout=60)


# --------------------------------------------------------------------- resume ----

def test_a_complete_run_is_skipped(tmp_path: Path, fakes: dict[str, Path]) -> None:
    _queue(tmp_path, fakes, ["p_a_sd1:0"], "0")
    _forget_calls(fakes)
    result, status, _ = _queue(tmp_path, fakes, ["p_a_sd1:0", "4ch_only:0"], "0")
    assert result.returncode == 0
    assert [c[1] for c in _calls(fakes) if c[0] == "train"] == ["4ch_only_seed0"]
    assert status["runs"][0]["note"].startswith("already complete")


def test_a_changed_score_file_makes_a_run_incomplete(
    tmp_path: Path, fakes: dict[str, Path]
) -> None:
    _queue(tmp_path, fakes, ["p_a_sd1:0"], "0")
    (tmp_path / "w" / "s2" / "p_a_sd1_seed0" / "scores" / "panel.csv").write_text("tampered")
    _forget_calls(fakes)
    _queue(tmp_path, fakes, ["p_a_sd1:0"], "0")
    assert [c[1] for c in _calls(fakes) if c[0] == "train"] == ["p_a_sd1_seed0"]


def test_runs_are_restored_from_a_previous_sessions_output(
    tmp_path: Path, fakes: dict[str, Path]
) -> None:
    """The previous version's output, attached as an input: its run dirs come back."""
    previous = tmp_path / "previous" / "thesis_dbp"
    run_dir = previous / "outputs" / "p2" / "s2" / "p_a_sd1_seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint.pt").write_bytes(b"step 500")
    result, status, _ = _queue(tmp_path, fakes, ["p_a_sd1:0"], "0",
                               "--resume-from", str(previous))
    assert result.returncode == 0
    assert (tmp_path / "w" / "s2" / "p_a_sd1_seed0" / "checkpoint.pt").read_bytes() == b"step 500"
    assert "restored from" in status["runs"][0]["note"]


# ---------------------------------------------------------------------- smoke ----

def test_smoke_tests_every_run_on_every_gpu(tmp_path: Path, fakes: dict[str, Path]) -> None:
    result, status, _ = _queue(tmp_path, fakes, ["p_a_sd1:0", "4ch_only:0"], "0,1",
                               "--smoke", "2")
    assert result.returncode == 0, result.stdout
    pairs = {(c[1], c[2]) for c in _calls(fakes) if c[0] == "train"}
    assert pairs == {(r, d) for r in ("p_a_sd1_seed0", "4ch_only_seed0") for d in "01"}
    assert not (tmp_path / "w" / "s2").exists(), "a smoke test never writes real outputs"


def test_a_smoke_failure_stops_everything_at_once(
    tmp_path: Path, fakes: dict[str, Path]
) -> None:
    result, status, elapsed = _queue(tmp_path, fakes, ["bad_a:0", "slow_b:0"], "0",
                                     "--smoke", "2", env={"FAKE_SLOW": "120"})
    assert result.returncode == 1
    assert [c[1] for c in _calls(fakes) if c[0] == "train"] == ["bad_a_seed0"], \
        "after a smoke failure nothing else starts"
    assert elapsed < 60


# ------------------------------------------------------------------ arguments ----

@pytest.mark.parametrize("devices", ["0,0", "cuda:0", ""])
def test_bad_device_lists_are_refused(tmp_path: Path, fakes: dict[str, Path],
                                      devices: str) -> None:
    result, _, _ = _queue(tmp_path, fakes, ["p_a_sd1:0"], devices)
    assert result.returncode == 1 and not _calls(fakes)


def test_a_run_queued_twice_is_refused(tmp_path: Path, fakes: dict[str, Path]) -> None:
    result, _, _ = _queue(tmp_path, fakes, ["p_a_sd1:0", "p_a_sd1:0"], "0")
    assert result.returncode == 1 and not _calls(fakes)


# --------------------------------------------------------- the trainer's seams ----

def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    sys.path.insert(0, str(THESIS_DBP / "src"))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_validation_crop_seed_is_the_same_in_every_process() -> None:
    """``hash()`` of a str is randomised per process; the seed must not be."""
    code = (f"import sys; sys.path.insert(0, r'{THESIS_DBP / 'src'}'); "
            f"sys.path.insert(0, r'{SCRIPTS}'); import p2_train_s2 as t; "
            f"print(t.stable_seed('0031b4853b6babcc_c0__dcae'))")
    seen = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONHASHSEED=seed)).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(seen) == 1 and seen != {""}, seen


def test_dryrun_steps_are_refused_outside_a_dry_run(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k != "DBP_DRYRUN"}
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "p2_train_s2.py"), "--protocol", "p_a_sd1",
         "--root", str(tmp_path), "--pairs-manifest", "x", "--dryrun-steps", "3"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 2 and "DBP_DRYRUN=1" in result.stdout


def test_the_fault_seam_is_ignored_outside_a_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _load("p2_train_s2")
    monkeypatch.setenv("DBP_FAULT_EXIT_AFTER_STEP", "4")
    monkeypatch.delenv("DBP_DRYRUN", raising=False)
    assert trainer.fault_step() is None
    monkeypatch.setenv("DBP_DRYRUN", "1")
    assert trainer.fault_step() == 4


def test_the_dryrun_backbone_refuses_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(THESIS_DBP / "src"))
    from dbp.backbones.loader import BackboneLoadError, load_backbone

    monkeypatch.setenv("DBP_DRYRUN", "1")
    with pytest.raises(BackboneLoadError, match="CPU dry-runs only"):
        load_backbone("dinov3-vitb16", "cuda", None)


def test_the_dryrun_normalisation_says_what_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(THESIS_DBP / "src"))
    from dbp.streams.s1_semantic import processor_normalisation

    monkeypatch.setenv("DBP_DRYRUN", "1")
    assert "DRY-RUN" in processor_normalisation("r", "0" * 40)["source"]


# ------------------------------------------------------- S2 pre-flight checks ----

def _preflight(tmp_path: Path, *args: str, env: dict[str, str] | None = None):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "p2_preflight.py"), "--stage", "t",
         "--reports", str(tmp_path), *args],
        capture_output=True, text=True, timeout=120, env=env,
    )


def test_the_wrong_gpu_count_fails(tmp_path: Path) -> None:
    result = _preflight(tmp_path, "--expect-gpus", "2")
    assert result.returncode == 1 and "expected 2" in result.stdout


def test_a_missing_token_fails_and_a_present_one_is_never_shown(tmp_path: Path) -> None:
    env = {k: v for k, v in os.environ.items() if k not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    missing = _preflight(tmp_path, "--hf-token", env=env)
    assert missing.returncode == 1 and "HF_TOKEN present: false" in missing.stdout
    secret = "hf_dryrunNotARealToken000"
    present = _preflight(tmp_path, "--hf-token", env=dict(env, HF_TOKEN=secret))
    assert present.returncode == 0 and "HF_TOKEN present: true" in present.stdout
    record = (tmp_path / "p2_preflight_t.json").read_text(encoding="utf-8")
    assert secret not in present.stdout + present.stderr + record


def test_an_input_with_another_hash_fails(tmp_path: Path) -> None:
    path = tmp_path / "p2_splits.csv"
    path.write_text("row_id,split\n", encoding="utf-8")
    result = _preflight(tmp_path, "--sha256", f"{path}={'0' * 64}")
    assert result.returncode == 1 and "p2_splits.csv" in result.stdout


# ------------------------------------------------------------ the runbook lint ----

def _cells() -> list[str]:
    sys.path.insert(0, str(SCRIPTS))
    import runbook_dryrun

    return [c.text for c in runbook_dryrun.code_cells(
        runbook_dryrun.parse_cells(RUNBOOK.read_text(encoding="utf-8")))]


def test_the_model_preflight_runs_before_the_queue() -> None:
    code = "\n".join(_cells())
    smoke = code.index('"--smoke", EXPECT["smoke_steps"]')
    assert smoke < code.index('code = py("p2_run_queue.py"')


def test_the_queue_cell_does_not_assert_so_the_report_still_runs() -> None:
    cells = _cells()
    queue = next(c for c in cells if 'code = py("p2_run_queue.py"' in c)
    report = next(c for c in cells if "p2_report.py" in c)
    assert "assert" not in queue and "assert" not in report
    assert cells.index(queue) < cells.index(report) < len(cells) - 1


def test_the_final_cell_fails_the_notebook_on_any_failed_run() -> None:
    assert _cells()[-1].lstrip().startswith("assert code == 0")


def test_every_run_in_the_expect_file_is_a_configured_protocol() -> None:
    sys.path.insert(0, str(THESIS_DBP / "src"))
    from dbp.decoders.registry import load_bank_config
    from dbp.eval.run_config import load_run_configs

    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(THESIS_DBP / "configs" / "p2_runs.yaml", bank)
    expect = json.loads(EXPECT.read_text(encoding="utf-8"))
    for item in expect["runs"]:
        runs.by_id(item.split(":")[0])  # raises on an unknown protocol
    assert expect["gpus"] == 2 and expect["hf_token"] is True
    assert len(expect["devices"].split(",")) == expect["gpus"]


@pytest.mark.parametrize("slug", ["dbp-code", "dbp-code-ed7a8cb"])
def test_cell_1_finds_the_code_dataset_under_either_slug(tmp_path: Path, slug: str) -> None:
    """The upload for ed7a8cb became `dbp-code-ed7a8cb`; Cell 1 looked for `dbp-code`."""
    code = "\n".join(_cells())
    pattern = code.split('"code": find("', 1)[1].split('"', 1)[0]
    (tmp_path / "datasets" / "muhammadadeelshah" / slug / "thesis_dbp").mkdir(parents=True)
    assert len([p for p in tmp_path.glob(pattern) if p.is_dir()]) == 1


@pytest.mark.parametrize("slug", ["p1-data", "report"])
def test_cell_1_finds_the_corpus_under_any_slug(tmp_path: Path, slug: str) -> None:
    """The dataset titled p1-data has the slug `report`; Cell 1 looked for `p1-data`."""
    code = "\n".join(_cells())
    for key, sub in (("reals", "p1_reals"), ("generated", "p1_generated")):
        pattern = code.split(f'"{key}": find("', 1)[1].split('"', 1)[0]
        (tmp_path / "datasets" / "muhammadadeelshah" / slug / sub).mkdir(parents=True)
        assert len([p for p in tmp_path.glob(pattern) if p.is_dir()]) == 1


class _Unkillable:
    """A process that ignores terminate and kill, as one stuck in a hung read does."""

    pid = 4242

    def __init__(self) -> None:
        self.calls: list[str] = []

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            raise AssertionError("waited without a timeout: the queue would hang forever")
        self.calls.append(f"wait {timeout}")
        raise subprocess.TimeoutExpired("x", timeout)


def test_a_process_that_will_not_die_is_abandoned_not_waited_for() -> None:
    """The 9.5 h deadline must hold even if a child ignores SIGKILL."""
    queue = _load("p2_run_queue")
    process = _Unkillable()
    started = time.perf_counter()
    assert queue.stop_process(process, grace=0.01) == "abandoned"
    assert process.calls[0] == "terminate" and "kill" in process.calls
    assert time.perf_counter() - started < 5
