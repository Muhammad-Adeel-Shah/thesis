"""P2 Session D: train and score S2 runs from one queue, one worker per GPU.

**Scheduling.** The runs form one queue in the order given. One worker per device
(``CUDA_VISIBLE_DEVICES`` set per worker) takes the next run as soon as its device is
free, trains it with ``p2_train_s2.py`` and then scores it with ``p2_s2_infer.py`` on the
same device. No device waits while a run is queued.

**Failures.** A failed run is logged with the last lines of its log and recorded in the
status file; it is **not** retried, and the worker moves on to the next run. The runs
are independent, and systematic failures -- a missing module, a bad path, a model that
cannot load or save -- are what ``--smoke`` exists to catch before this starts. The
exit status is non-zero if any run failed, so the notebook's final cell shows it.

**Resume.** Every run checkpoints every 500 steps. ``--resume-from`` names a previous
session's ``thesis_dbp`` root (that notebook version's output, attached as an input); a
run's directory and reports are copied from it when they are not already here, and
``p2_train_s2.py`` continues from the checkpoint. A run already complete -- its report
says ``OK`` and every requested score list is present with the sha256 its summary
records -- is skipped.

**Pre-flight** (``--smoke N``): every run on every device, N training steps (save,
reload, compare) and N rows of every list scored, in a scratch directory. Here the
first failure stops everything at once: a smoke failure is systematic by definition.

**Train only** (``--train-only``, P3 LOLO): no scoring step; a run is complete when its
training report says ``OK``. LOLO folds are scored on Panel B only at ``N*(-f)``, after
all eight folds exist, so nothing is scored in the training session.

**Deadline.** At ``--deadline-hours`` every running job is stopped (it loses at most
the steps since its last checkpoint) and the exit status is 2.

The status is written to ``--status`` after every change, so it survives any ending.

Usage:
    python scripts/p2_run_queue.py --run p_a_sd1:0 --run p_a_sd1:1 --run 4ch_only:0 \\
        --run full_bank:0 --devices 0,1 --root /kaggle/working/p1 ... --deadline-hours 9
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import (  # noqa: E402
    device_tag,
    run_id,
    s2_run_paths,
    worker_log_paths,
)

#: Exit codes. A failure outranks the deadline: both mean "look", a failure means "fix".
EXIT_COMPLETE, EXIT_RUN_FAILED, EXIT_DEADLINE = 0, 1, 2

#: Seconds a stopped job gets to exit before it is killed.
GRACE_SECONDS = 30


@dataclass
class RunState:
    """One run's progress, as the status file records it."""

    run_id: str
    protocol: str
    seed: int
    state: str = "queued"
    device: str | None = None
    train_rc: int | None = None
    infer_rc: int | None = None
    started: float | None = None
    finished: float | None = None
    hours: float | None = None
    logs: list[str] = field(default_factory=list)
    note: str = ""


def parse_run(text: str) -> tuple[str, int]:
    """``PROTOCOL:SEED``."""
    protocol, sep, seed = text.partition(":")
    if not sep or not protocol or not seed.isdigit():
        raise ValueError(f"--run expects PROTOCOL:SEED, got {text!r}")
    return protocol, int(seed)


def parse_devices(text: str) -> list[str]:
    """``0,1`` or ``cpu``; each device at most once."""
    devices = [d.strip() for d in text.split(",") if d.strip()]
    for device in devices:
        device_tag(device)  # validates
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"--devices must name each device once, got {text!r}")
    return devices


def stop_process(process: Any, grace: float = GRACE_SECONDS) -> str:
    """Terminate, then kill, and never wait without a limit.

    A process blocked in an uninterruptible read (a hung ``/kaggle/input`` mount, say)
    ignores even SIGKILL until the read returns. Waiting for it without a timeout would
    hold the queue -- and the notebook -- past its own deadline, so after the kill the
    wait is bounded too and the process is abandoned if it still has not exited.

    Returns:
        ``"terminated"``, ``"killed"`` or ``"abandoned"``.
    """
    process.terminate()
    try:
        process.wait(timeout=grace)
        return "terminated"
    except subprocess.TimeoutExpired:
        process.kill()
    try:
        process.wait(timeout=grace)
        return "killed"
    except subprocess.TimeoutExpired:
        return "abandoned"


def tail(path: Path, lines: int) -> str:
    """The last ``lines`` lines of a text file."""
    if not path.is_file():
        return f"(no log at {path})"
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def is_complete(rid: str, args: argparse.Namespace, lists: list[str]) -> tuple[bool, str]:
    """Whether a run's training report says OK and every requested score list verifies."""
    paths = s2_run_paths(args.out_dir, args.reports_dir, rid)
    if args.train_only:
        if not paths.report.is_file():
            return False, "no training report"
        report = json.loads(paths.report.read_text(encoding="utf-8"))
        ok = report.get("status") == "OK" and report.get("mode") == "train"
        return (True, "complete") if ok else (False, f"training status {report.get('status')}")
    if not paths.report.is_file() or not paths.scores_summary.is_file():
        return False, "no report or no score summary"
    report = json.loads(paths.report.read_text(encoding="utf-8"))
    if report.get("status") != "OK":
        return False, f"training status {report.get('status')}"
    summary = json.loads(paths.scores_summary.read_text(encoding="utf-8"))
    if summary.get("limit"):
        return False, "scores are from a limited (smoke) pass"
    for name in lists:
        entry = summary.get("lists", {}).get(name)
        if entry is None:
            return False, f"no {name} scores"
        scores = paths.scores(name)
        if not scores.is_file() or file_sha256(scores) != entry["sha256"]:
            return False, f"{name} scores missing or changed"
    return True, "complete"


def restore(rid: str, sources: list[Path], args: argparse.Namespace, log: Any) -> str:
    """Copy a run's directory and reports from a previous session, if it has them."""
    here = s2_run_paths(args.out_dir, args.reports_dir, rid)
    try:  # the same place under the previous session's thesis_dbp root (p2 or p3)
        relative = args.out_dir.resolve().relative_to(THESIS_DBP.resolve())
    except ValueError:
        relative = Path("outputs") / "p2" / "s2"
    for source in sources:
        there = s2_run_paths(source / relative, source / "reports", rid)
        if not there.run_dir.is_dir():
            continue
        if here.run_dir.exists():
            return f"kept local {here.run_dir.name}; {source} not used"
        shutil.copytree(there.run_dir, here.run_dir)
        for name in ("report", "scores_summary"):
            src, dst = getattr(there, name), getattr(here, name)
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        log.info("restored %s from %s", rid, source)
        return f"restored from {source}"
    return "nothing to restore"


class Queue:
    """The shared queue, the workers, and the status they report into."""

    def __init__(self, args: argparse.Namespace, log: Any) -> None:
        self.args = args
        self.log = log
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.started = time.time()
        self.runs = [RunState(run_id(p, s), p, s) for p, s in args.runs]
        self.pending: queue.Queue[RunState] = queue.Queue()
        self.outcome = "RUNNING"
        self.device_seconds = 0.0

    # ---- status -------------------------------------------------------------------
    def write_status(self) -> None:
        """Write the status atomically. Called under the lock."""
        payload = {
            "mode": "smoke" if self.args.smoke else "train+score",
            "outcome": self.outcome,
            "devices": self.args.devices,
            "deadline_hours": self.args.deadline_hours,
            "wall_hours": round((time.time() - self.started) / 3600.0, 4),
            "device_hours": round(self.device_seconds / 3600.0, 4),
            "runs": [asdict(r) for r in self.runs],
        }
        self.args.status.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.args.status.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.args.status)

    def update(self, run: RunState, **changes: Any) -> None:
        """Change a run's state and persist the status."""
        with self.lock:
            for key, value in changes.items():
                setattr(run, key, value)
            self.write_status()

    # ---- one subprocess -------------------------------------------------------------
    def execute(self, command: list[str], device: str, log_path: Path) -> int:
        """Run one command on one device, appending to its log; honours the stop event."""
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if device != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = device
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                       env=env, cwd=THESIS_DBP)
            while process.poll() is None:
                if self.stop.wait(timeout=self.args.poll_seconds):
                    how = stop_process(process)
                    handle.write(f"\n[queue] stopped: {how}\n")
                    if how == "abandoned":
                        self.log.error("a process did not exit after kill; abandoned "
                                       "(pid %d) so the queue can end", process.pid)
                    return -15
        return process.returncode

    def train_command(self, run: RunState, device: str, out_dir: Path, reports: Path) -> list[str]:
        """The trainer's command line for one run."""
        command = [
            sys.executable, "-u", str(self.args.trainer),
            "--protocol", run.protocol, "--seed", str(run.seed),
            "--root", str(self.args.root), "--pairs-manifest", str(self.args.pairs_manifest),
            "--splits", str(self.args.splits),
            "--device", "cpu" if device == "cpu" else "cuda",
            "--out-dir", str(out_dir), "--reports-dir", str(reports),
        ]
        if self.args.smoke:
            command += ["--smoke", str(self.args.smoke)]
        return command + self.args.train_arg

    def infer_command(self, run: RunState, device: str, out_dir: Path, reports: Path) -> list[str]:
        """The scorer's command line for one run."""
        command = [
            sys.executable, "-u", str(self.args.inferer),
            "--protocol", run.protocol, "--seed", str(run.seed),
            "--root", str(self.args.root),
            "--device", "cpu" if device == "cpu" else "cuda",
            "--out-dir", str(out_dir), "--reports-dir", str(reports),
        ]
        if self.args.panel_root is not None:
            command += ["--panel-root", str(self.args.panel_root)]
        for item in self.args.list:
            command += ["--list", item]
        for item in self.args.only:
            command += ["--only", item]
        if self.args.smoke:
            command += ["--limit", str(self.args.smoke)]
        return command + self.args.infer_arg

    # ---- workers --------------------------------------------------------------------
    def report_failure(self, run: RunState, device: str, step: str, log_path: Path) -> None:
        """Say which run failed where, with the end of its log."""
        with self.lock:
            print(f"\n----- {run.run_id} FAILED in {step} on {device_tag(device)}: "
                  f"last {self.args.tail} lines of {log_path.name} -----\n"
                  f"{tail(log_path, self.args.tail)}\n", flush=True)

    def work(self, device: str, pending: queue.Queue[RunState]) -> None:
        """Take runs from ``pending`` until it is empty or the queue is stopped."""
        while not self.stop.is_set():
            try:
                run = pending.get_nowait()
            except queue.Empty:
                return
            logs = worker_log_paths(self.args.log_dir, self.args.scratch, run.run_id, device)
            if self.args.smoke:
                out_dir = reports = logs.smoke_dir
                train_log = infer_log = logs.smoke
            else:
                out_dir, reports = self.args.out_dir, self.args.reports_dir
                train_log, infer_log = logs.train, logs.infer
            begun = time.time()
            self.update(run, state="training", device=device, started=begun,
                        logs=sorted({str(train_log.name), str(infer_log.name)}))
            self.log.info("%s -> %s (training)", run.run_id, device_tag(device))
            rc = self.execute(self.train_command(run, device, out_dir, reports), device, train_log)
            self.update(run, train_rc=rc)
            if rc == 0 and self.args.train_only:
                step = "training"
            elif rc == 0 and not self.stop.is_set():
                self.update(run, state="scoring")
                self.log.info("%s on %s: trained, scoring", run.run_id, device_tag(device))
                rc = self.execute(self.infer_command(run, device, out_dir, reports), device,
                                  infer_log)
                self.update(run, infer_rc=rc)
                step = "scoring"
            else:
                step = "training"
            elapsed = time.time() - begun
            with self.lock:
                self.device_seconds += elapsed
            if rc == 0:
                self.update(run, state="complete", finished=time.time(),
                            hours=round(elapsed / 3600.0, 4))
                self.log.info("%s on %s: COMPLETE in %.2f h", run.run_id, device_tag(device),
                              elapsed / 3600.0)
            elif self.stop.is_set() and rc == -15:
                self.update(run, state="stopped", finished=time.time(),
                            hours=round(elapsed / 3600.0, 4),
                            note="stopped at the deadline or by another failure; resumable")
            else:
                note = ("STOPPED-UNDERTRAINED (exit 3): do not raise the ceiling, report"
                        if step == "training" and rc == 3 else f"exit {rc} in {step}")
                self.update(run, state="failed", finished=time.time(),
                            hours=round(elapsed / 3600.0, 4), note=note)
                self.report_failure(run, device, step, train_log if step == "training"
                                    else infer_log)
                if self.args.smoke:
                    self.stop.set()  # a smoke failure is systematic: stop everything

    def run(self) -> int:
        """Restore, skip what is complete, then work the queue to the end."""
        lists = [item.partition("=")[0] for item in self.args.list]
        for state in self.runs:
            if self.args.smoke:
                for _ in self.args.devices:  # one copy per device
                    self.pending.put(RunState(state.run_id, state.protocol, state.seed))
                continue
            note = restore(state.run_id, self.args.resume_from, self.args, self.log)
            done, why = is_complete(state.run_id, self.args, lists)
            if done:
                state.state, state.note = "complete", f"already complete; {note}"
                self.log.info("%s: already complete, skipped", state.run_id)
            else:
                state.note = f"{why}; {note}"
                self.pending.put(state)
        if self.args.smoke:
            # Every (run, device) pair is smoke-tested: each worker drains its own copy.
            self.runs = []
            per_device: dict[str, queue.Queue[RunState]] = {d: queue.Queue() for d in
                                                            self.args.devices}
            for index, state in enumerate(list(self.pending.queue)):
                device = self.args.devices[index % len(self.args.devices)]
                per_device[device].put(state)
                self.runs.append(state)
            return self.drive(per_device)
        return self.drive({d: self.pending for d in self.args.devices})

    def drive(self, queues: dict[str, queue.Queue[RunState]]) -> int:
        """Start one worker per device and wait, enforcing the deadline."""
        with self.lock:
            self.write_status()
        threads = []
        for device, pending in queues.items():
            thread = threading.Thread(target=self.work, args=(device, pending), daemon=True)
            thread.start()
            threads.append(thread)
        deadline = self.started + self.args.deadline_hours * 3600.0
        deadline_hit = False
        while any(t.is_alive() for t in threads):
            if time.time() > deadline and not self.stop.is_set():
                self.log.error("deadline of %.2f h reached: stopping every job",
                               self.args.deadline_hours)
                deadline_hit = True
                self.stop.set()
            for thread in threads:
                thread.join(timeout=1.0)
        failed = [r.run_id for r in self.runs if r.state == "failed"]
        with self.lock:
            if failed:
                self.outcome = "RUN-FAILED"
            elif deadline_hit or any(r.state in ("queued", "stopped") for r in self.runs):
                self.outcome = "DEADLINE"
            else:
                self.outcome = "COMPLETE"
            self.write_status()
        self.copy_logs()
        done = [r.run_id for r in self.runs if r.state == "complete"]
        self.log.info("status %s -> %s", self.outcome, self.args.status)
        self.log.info("complete (%d/%d): %s", len(done), len(self.runs), done)
        if failed:
            self.log.error("FAILED: %s", failed)
            return EXIT_RUN_FAILED
        return EXIT_DEADLINE if self.outcome == "DEADLINE" else EXIT_COMPLETE

    def copy_logs(self) -> None:
        """Copy every log this queue wrote next to the status, for the report archive."""
        target = self.args.status.parent / "s2_logs"
        for path in sorted(self.args.log_dir.glob("s2_*.log")):
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)


def main() -> int:
    """Parse, validate and run the queue. Returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="PROTOCOL:SEED")
    parser.add_argument("--devices", required=True, help="0,1 or cpu")
    parser.add_argument("--root", required=True, type=Path, help="Corpus root.")
    parser.add_argument("--panel-root", type=Path, default=None)
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--list", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--only", action="append", default=[], metavar="NAME=VIEW")
    parser.add_argument("--train-arg", action="append", default=[],
                        help="One extra trainer token, e.g. --train-arg=--ceiling=8000.")
    parser.add_argument("--infer-arg", action="append", default=[])
    parser.add_argument("--smoke", type=int, default=0, help="Pre-flight: N steps per run.")
    parser.add_argument("--resume-from", action="append", type=Path, default=[],
                        help="A previous session's thesis_dbp root to restore runs from.")
    parser.add_argument("--deadline-hours", type=float, default=11.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--tail", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "s2")
    parser.add_argument("--reports-dir", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--log-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "logs")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/dbp_s2_smoke"))
    parser.add_argument("--status", type=Path, default=None)
    parser.add_argument("--trainer", type=Path, default=THESIS_DBP / "scripts" / "p2_train_s2.py")
    parser.add_argument("--inferer", type=Path, default=THESIS_DBP / "scripts" / "p2_s2_infer.py")
    parser.add_argument("--train-only", action="store_true",
                        help="No scoring step; complete = training report OK (P3 LOLO).")
    args = parser.parse_args()
    log = setup_logging("p2_run_queue")

    try:
        args.runs = [parse_run(r) for r in args.run]
        args.devices = parse_devices(args.devices)
    except ValueError as exc:
        log.error("%s", exc)
        return EXIT_RUN_FAILED
    ids = [run_id(p, s) for p, s in args.runs]
    if len(set(ids)) != len(ids):
        log.error("a run is queued twice: %s", ids)
        return EXIT_RUN_FAILED
    if args.status is None:
        name = "p2_s2_smoke_status.json" if args.smoke else "p2_s2_status.json"
        args.status = args.reports_dir / name
    log.info("%s: %d run(s) on %s: %s", "SMOKE" if args.smoke else "QUEUE", len(ids),
             [device_tag(d) for d in args.devices], ids)
    return Queue(args, log).run()


if __name__ == "__main__":
    raise SystemExit(main())
