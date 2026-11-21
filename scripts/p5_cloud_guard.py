"""P5 cloud guard (AWS / RunPod): auto-stop on finish or idle, and a spend cap.

Runs beside the jobs on the instance (never on Kaggle, never locally) and writes only under
``--state`` (the data volume), never into the repository tree. Every ``--poll`` seconds:

1. **Spend ledger.** Adds the elapsed wall time to ``ledger.json``, which survives a stop and
   a restart of the instance, so the total spans every session. Spend =
   ``hours x --rate`` plus ``--fixed-usd`` (e.g. the volume's estimate, entered by the user).
2. **Cap.** When the spend reaches ``--cap-usd`` (set below the user's $80 limit), it writes
   ``ALERT_CAP.txt``, prints it, and stops the machine. The user decides what happens next.
3. **Finish.** When every PID in ``--watch-pid`` has exited, it waits ``--grace`` seconds
   and stops the machine (``--keep-on-finish`` to stay up).
4. **Idle.** When no watched PID is alive and GPU utilisation has stayed below 5% for
   ``--idle-minutes``, it stops the machine.

"Stop" runs ``--stop-command`` (AWS: ``sudo shutdown -h now`` with the instance's
shutdown behaviour set to *stop*, so the volume and its checkpoints survive; RunPod:
``runpodctl stop pod $RUNPOD_POD_ID``). ``--dry-run-stop`` prints instead (tests).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


def gpu_utilisation() -> float | None:
    """Mean GPU utilisation in percent from nvidia-smi, or None when there is no GPU."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True,
                              check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"guard: nvidia-smi unavailable ({type(exc).__name__}); idle rule uses PIDs only")
        return None
    values = [float(v) for v in done.stdout.split() if v.strip()]
    return sum(values) / len(values) if values else None


def alive(pid: int) -> bool:
    """Whether a process exists (signal 0)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_ledger(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"hours": 0.0, "sessions": []}


def save_ledger(path: Path, ledger: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def stop(args: argparse.Namespace, reason: str) -> None:
    print(f"guard: STOPPING ({reason})", flush=True)
    (args.state / "last_stop.txt").write_text(f"{time.ctime()} {reason}\n", encoding="utf-8")
    if args.dry_run_stop:
        print(f"guard: dry-run, would run: {args.stop_command}")
        return
    subprocess.run(shlex.split(args.stop_command), check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True, help="USD per instance hour")
    parser.add_argument("--fixed-usd", type=float, default=0.0)
    parser.add_argument("--cap-usd", type=float, required=True)
    parser.add_argument("--watch-pid", type=int, action="append", default=[])
    parser.add_argument("--keep-on-finish", action="store_true")
    parser.add_argument("--grace", type=float, default=120)
    parser.add_argument("--idle-minutes", type=float, default=30)
    parser.add_argument("--poll", type=float, default=60)
    parser.add_argument("--stop-command", default="sudo shutdown -h now")
    parser.add_argument("--dry-run-stop", action="store_true")
    parser.add_argument("--max-polls", type=int, default=0, help="tests: stop after N polls")
    args = parser.parse_args()
    args.state.mkdir(parents=True, exist_ok=True)
    ledger_path = args.state / "ledger.json"
    ledger = load_ledger(ledger_path)
    ledger["sessions"].append({"started": time.ctime(), "rate": args.rate})
    last, idle_since, polls = time.time(), None, 0
    while True:
        time.sleep(args.poll)
        now = time.time()
        ledger["hours"] += (now - last) / 3600
        last = now
        spend = ledger["hours"] * args.rate + args.fixed_usd
        ledger["spend_usd"] = round(spend, 2)
        save_ledger(ledger_path, ledger)
        running = [p for p in args.watch_pid if alive(p)]
        util = gpu_utilisation()
        print(f"guard: {ledger['hours']:.2f} h, ${spend:.2f} of cap ${args.cap_usd:.2f}; "
              f"watched alive {running}; GPU {util}%", flush=True)
        polls += 1
        if spend >= args.cap_usd:
            text = (f"SPEND CAP REACHED: ${spend:.2f} >= ${args.cap_usd:.2f} "
                    f"({ledger['hours']:.2f} h at ${args.rate}/h + ${args.fixed_usd} fixed). "
                    f"Stopped; ask the user before restarting.\n")
            (args.state / "ALERT_CAP.txt").write_text(text, encoding="utf-8")
            stop(args, "spend cap")
            return 3
        if args.watch_pid and not running and not args.keep_on_finish:
            time.sleep(args.grace)
            stop(args, "every watched job finished")
            return 0
        if not running and (util is None or util < 5.0):
            idle_since = idle_since or now
            if (now - idle_since) / 60 >= args.idle_minutes:
                stop(args, f"idle {args.idle_minutes} min")
                return 0
        else:
            idle_since = None
        if args.max_polls and polls >= args.max_polls:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
