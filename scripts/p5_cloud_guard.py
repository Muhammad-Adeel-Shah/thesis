"""P5 cloud guard: auto-stop on job end, on idle, and at the spend cap (user rules, 2026-11-21).

Started by ``env/aws_p5_session.sh`` beside every job, never on Kaggle, never locally. It
writes only under ``--state`` (the data volume), never into the repository tree. Every
``--poll`` seconds:

1. **Ledger.** Adds the elapsed wall time to ``ledger.json`` (it survives stops and
   restarts, so the hours span every session). Estimated spend =
   ``instance hours x --rate`` + ``days since --volume-created x --ebs-usd-per-day``. The
   data volume is billed while the instance is stopped, so it is counted in calendar days.
2. **Cap.** At ``spend >= --cap-usd`` (75; the user's AWS Budget action also stops EC2 at
   75) it writes ``ALERT_CAP.txt`` and stops the instance. The user decides what follows.
3. **Job end.** When every ``--watch-pid`` has exited, it stops the instance after
   ``--grace`` seconds. (The session wrapper's exit trap also stops it; whichever is first.)
4. **Idle.** No watched process alive **and** GPU utilisation 0% for ``--idle-minutes``
   (20) -> stop. Without nvidia-smi the GPU counts as idle.

"Stop" runs ``--stop-command`` (AWS: ``sudo shutdown -h now``; the instance's shutdown
behaviour is *stop*, so the volumes and every checkpoint survive).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def gpu_utilisation() -> float | None:
    """Highest GPU utilisation in percent from nvidia-smi, or None without a GPU tool."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"guard: nvidia-smi unavailable ({type(exc).__name__}); GPU counted as idle")
        return None
    values = [float(v) for v in done.stdout.split() if v.strip()]
    return max(values) if values else None


def alive(pid: int) -> bool:
    """Whether a process exists. POSIX: signal 0 (a probe). Windows: ``tasklist`` --
    there ``os.kill(pid, 0)`` is not a probe, it TERMINATES the process (found by
    test_p5_cloud.py, 2026-11-21)."""
    if os.name == "nt":
        done = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True,
                              text=True, check=False)
        return str(pid) in done.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def spend_usd(hours: float, rate: float, volume_created: str, ebs_per_day: float,
              now: dt.date | None = None) -> float:
    """Instance hours x rate + the data volume's calendar days (inclusive) x its daily rate."""
    created = dt.date.fromisoformat(volume_created)
    days = ((now or dt.date.today()) - created).days + 1
    return hours * rate + max(days, 1) * ebs_per_day


def load_ledger(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"hours": 0.0, "sessions": []}


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def stop(args: argparse.Namespace, ledger: dict[str, Any], reason: str) -> None:
    print(f"guard: STOPPING ({reason})", flush=True)
    ledger["sessions"][-1]["stopped"] = f"{time.ctime()} by guard: {reason}"
    save_ledger(args.state / "ledger.json", ledger)
    subprocess.run(shlex.split(args.stop_command), check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True, help="USD per instance hour")
    parser.add_argument("--volume-created", required=True, help="YYYY-MM-DD, the data volume")
    parser.add_argument("--ebs-usd-per-day", type=float, required=True)
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--label", default="session")
    parser.add_argument("--watch-pid", type=int, action="append", default=[])
    parser.add_argument("--grace", type=float, default=60)
    parser.add_argument("--idle-minutes", type=float, default=20)
    parser.add_argument("--poll", type=float, default=60)
    parser.add_argument("--stop-command", default="sudo shutdown -h now")
    parser.add_argument("--max-polls", type=int, default=0, help="tests: return after N polls")
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    ledger_path = args.state / "ledger.json"
    ledger = load_ledger(ledger_path)
    ledger["sessions"].append({"label": args.label, "started": time.ctime(), "rate": args.rate})
    save_ledger(ledger_path, ledger)
    last, idle_since, polls = time.time(), None, 0
    while True:
        time.sleep(args.poll)
        now = time.time()
        ledger["hours"] += (now - last) / 3600
        last = now
        spend = spend_usd(ledger["hours"], args.rate, args.volume_created, args.ebs_usd_per_day)
        ledger["spend_usd"] = round(spend, 2)
        save_ledger(ledger_path, ledger)
        running = [p for p in args.watch_pid if alive(p)]
        util = gpu_utilisation()
        print(f"guard: {ledger['hours']:.2f} h, ${spend:.2f} of cap ${args.cap_usd:.2f}; "
              f"alive {running}; GPU {util}%", flush=True)
        polls += 1
        if spend >= args.cap_usd:
            (args.state / "ALERT_CAP.txt").write_text(
                f"SPEND CAP REACHED: ${spend:.2f} >= ${args.cap_usd:.2f}. Stopped; ask the "
                f"user before any restart.\n", encoding="utf-8")
            stop(args, ledger, "spend cap")
            return 3
        if args.watch_pid and not running:
            time.sleep(args.grace)
            stop(args, ledger, "every watched job ended")
            return 0
        if not running and (util is None or util <= 0.0):
            idle_since = idle_since or now
            if (now - idle_since) / 60 >= args.idle_minutes:
                stop(args, ledger, f"idle {args.idle_minutes} min (GPU 0%, no job)")
                return 0
        else:
            idle_since = None
        if args.max_polls and polls >= args.max_polls:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
