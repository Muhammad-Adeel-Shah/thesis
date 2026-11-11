"""Run probe-feature jobs in parallel, one per device, and fail fast.

Each ``--job DEVICE:MEMBERS`` starts ``p2_probe_features.py --members MEMBERS`` on one
device. The runner polls them and stops at the first of:

* **a job exits non-zero** -- its last lines are printed, every other job is terminated,
  and the runner exits 1. In the first S3 commit one shard died on a missing module and
  the other ran on alone for 3.5 hours while its GPU sat idle; that cannot recur;
* **the wall-clock deadline** -- every job is terminated and the runner exits 2. A
  member is cached only once it is complete, so a stop loses at most the members in
  flight, never a half-written cache;
* **every job exits 0** -- the runner exits 0.

The status file and the job logs are written in every case, including the two above,
before the runner returns: a later cell failing must never be able to lose them.

**Any member-sharded worker** (P3 step 4a, the reconstruction of the new reals): with
``--worker-arg`` the job runs ``--worker --members MEMBERS --device cpu|cuda`` followed by
exactly the given arguments, and nothing probe-specific; ``--log-prefix`` names its logs.

Usage:
    python scripts/p2_run_shards.py --images reports/p2_probe_images.csv \\
        --root /kaggle/working/p1 --panel-root /kaggle/input/.../panel \\
        --job 0:movq_k22 --job 1:qwen_image,realesrgan_x4 --deadline-hours 3.5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.utils.logging import setup_logging  # noqa: E402

#: Exit codes, so a notebook cell can tell a failure from a deadline.
EXIT_COMPLETE, EXIT_JOB_FAILED, EXIT_DEADLINE = 0, 1, 2

#: Seconds a terminated job gets to exit before it is killed.
GRACE_SECONDS = 30


@dataclass
class Job:
    """One device's share of the members."""

    index: int
    device: str
    members: str
    log_path: Path
    process: subprocess.Popen[bytes] | None = None
    handle: TextIO | None = None
    returncode: int | None = None
    started: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        """Human-readable name for logs."""
        return f"job {self.index} [{self.device}] {self.members}"

    @property
    def tag(self) -> str:
        """Names every file this job writes: its index and its device (I-053)."""
        return f"job{self.index}_{'cpu' if self.device == 'cpu' else 'gpu' + self.device}"


def parse_job(text: str, index: int, log_dir: Path, prefix: str = "probe") -> Job:
    """Parse ``DEVICE:MEMBERS`` -- DEVICE is ``cpu`` or a CUDA index."""
    if ":" not in text:
        raise ValueError(f"--job expects DEVICE:MEMBERS, got {text!r}")
    device, members = (part.strip() for part in text.split(":", 1))
    if not members:
        raise ValueError(f"--job {text!r} names no members")
    if device != "cpu" and not device.isdigit():
        raise ValueError(f"--job device must be 'cpu' or a CUDA index, got {device!r}")
    job = Job(index, device, members, log_dir)
    job.log_path = log_dir / f"{prefix}_{job.tag}.log"
    return job


def command_for(job: Job, args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """The command line and environment for one job."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if job.device != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = job.device
    worker_args = getattr(args, "worker_arg", None) or []
    if worker_args:
        return ([sys.executable, "-u", str(args.worker), "--members", job.members,
                 "--device", "cpu" if job.device == "cpu" else "cuda", *worker_args], env)
    command = [
        sys.executable, "-u", str(args.worker),
        "--images", str(args.images), "--root", str(args.root),
        "--members", job.members,
        "--device", "cpu" if job.device == "cpu" else "cuda",
        "--tag", job.tag,
    ]
    if args.panel_root is not None:
        command += ["--panel-root", str(args.panel_root)]
    if args.cache_dir is not None:
        command += ["--cache-dir", str(args.cache_dir)]
    return command, env


def tail(path: Path, lines: int) -> str:
    """The last ``lines`` lines of a text file."""
    if not path.is_file():
        return "(no log written)"
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(deque(handle, maxlen=lines))


def stop(jobs: list[Job], log: Any) -> None:
    """Terminate every running job, killing any that ignore the request."""
    for job in jobs:
        if job.process is not None and job.process.poll() is None:
            log.info("terminating %s", job.label)
            job.process.terminate()
    deadline = time.time() + GRACE_SECONDS
    for job in jobs:
        if job.process is None:
            continue
        try:
            job.process.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            log.info("killing %s", job.label)
            job.process.kill()
            job.process.wait()


def members_done(probe_dir: Path) -> list[str]:
    """Members with a cache and its sidecar."""
    return sorted(
        npz.stem.removeprefix("p2_probe_")
        for npz in probe_dir.glob("p2_probe_*.npz")
        if npz.with_suffix(".done.json").is_file()
    )


def write_status(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    """Run the jobs. Returns 0 complete, 1 a job failed, 2 the deadline passed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--job", action="append", required=True, metavar="DEVICE:MEMBERS")
    parser.add_argument("--images", type=Path, default=None, help="Probe workers only.")
    parser.add_argument("--root", type=Path, default=None, help="Probe workers only.")
    parser.add_argument("--panel-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--deadline-hours", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--tail", type=int, default=30, help="Log lines shown on failure.")
    parser.add_argument("--log-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "logs")
    parser.add_argument("--probe-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "probe")
    parser.add_argument("--status", type=Path,
                        default=THESIS_DBP / "reports" / "p2_s3_status.json")
    parser.add_argument("--worker", type=Path,
                        default=THESIS_DBP / "scripts" / "p2_probe_features.py",
                        help="The script each job runs (a test seam).")
    parser.add_argument("--worker-arg", action="append", default=[],
                        help="Generic worker mode: passed to the worker after --members and "
                             "--device, instead of the probe arguments.")
    parser.add_argument("--log-prefix", default="probe")
    args = parser.parse_args()
    log = setup_logging("p2_run_shards")
    if not args.worker_arg and (args.images is None or args.root is None):
        log.error("a probe worker needs --images and --root (or give --worker-arg)")
        return EXIT_JOB_FAILED

    args.log_dir.mkdir(parents=True, exist_ok=True)
    try:
        jobs = [parse_job(text, i, args.log_dir, args.log_prefix)
                for i, text in enumerate(args.job)]
    except ValueError as exc:
        log.error("%s", exc)
        return EXIT_JOB_FAILED
    devices = [j.device for j in jobs if j.device != "cpu"]
    if len(devices) != len(set(devices)):
        log.error("two jobs share a CUDA device: %s", devices)
        return EXIT_JOB_FAILED

    started = time.time()
    outcome, failed_job = "RUNNING", None
    try:
        for job in jobs:
            command, env = command_for(job, args)
            job.handle = job.log_path.open("w", encoding="utf-8", buffering=1)
            job.process = subprocess.Popen(
                command, stdout=job.handle, stderr=subprocess.STDOUT, env=env, cwd=THESIS_DBP
            )
            job.started = time.time()
            log.info("started %s (pid %d) -> %s", job.label, job.process.pid, job.log_path.name)

        while True:
            for job in jobs:
                if job.returncode is None and job.process is not None:
                    job.returncode = job.process.poll()
            failed = [j for j in jobs if j.returncode not in (None, 0)]
            if failed:
                failed_job = failed[0]
                outcome = "FAILED"
                log.error("%s exited %d -- stopping every other job now",
                          failed_job.label, failed_job.returncode)
                print(f"\n----- {failed_job.label}: last {args.tail} lines -----\n"
                      f"{tail(failed_job.log_path, args.tail)}", flush=True)
                stop(jobs, log)
                break
            if all(j.returncode == 0 for j in jobs):
                outcome = "COMPLETE"
                break
            hours = (time.time() - started) / 3600
            if hours > args.deadline_hours:
                outcome = "DEADLINE"
                log.error("deadline %.2f h passed at %.2f h -- stopping",
                          args.deadline_hours, hours)
                stop(jobs, log)
                break
            time.sleep(args.poll_seconds)
    finally:
        if outcome == "RUNNING":
            outcome = "INTERRUPTED"
            stop(jobs, log)
        for job in jobs:
            if job.process is not None:
                job.returncode = job.process.poll()
            if job.handle is not None:
                job.handle.close()
        wall = (time.time() - started) / 3600
        status = {
            "outcome": outcome,
            "failed_job": failed_job.label if failed_job else None,
            "jobs": [
                {"index": j.index, "device": j.device, "members": j.members,
                 "returncode": j.returncode, "log": j.log_path.name}
                for j in jobs
            ],
            "members_done": members_done(args.probe_dir),
            "deadline_hours": args.deadline_hours,
            "wall_hours": round(wall, 4),
            "device_hours": round(wall * max(1, len(devices)), 4),
        }
        write_status(args.status, status)
        for job in jobs:
            if job.log_path.is_file():
                shutil.copy2(job.log_path, args.status.parent / job.log_path.name)
        log.info("status %s -> %s", outcome, args.status)
        log.info("members done: %s", status["members_done"])

    return {"COMPLETE": EXIT_COMPLETE, "DEADLINE": EXIT_DEADLINE}.get(outcome, EXIT_JOB_FAILED)


if __name__ == "__main__":
    raise SystemExit(main())
