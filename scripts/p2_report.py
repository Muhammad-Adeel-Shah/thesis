"""Report on a stage and zip its reports. This script always exits 0.

It runs last in a runbook, after the work is done and persisted. Its job is to show what
happened, never to decide whether it happened: in the first S3 commit an import error in
the final reporting cell marked a ten-hour run as failed. Every step here is guarded and
a failure in one is printed and skipped, so the stage's outcome is whatever the run
cell said it was.

``--stage s3`` (default) reports the probe caches; ``--stage s2`` reports each queued
run's training result and score files; ``--stage final`` lists the P3 final-scoring
summaries (``reports/p3_final/``); ``--stage truefake`` the TrueFake ones
(``reports/p3_truefake/``); ``--stage p4`` the P4 baseline rows (``reports/p4_baselines/``).

Usage:
    python scripts/p2_report.py --status reports/p2_s3_status.json \\
        --zip /kaggle/working/p2_c_reports
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n=== {title} ===", flush=True)


def guarded(title: str, fn: object) -> None:
    """Run one report step; print, never raise, if it fails."""
    section(title)
    try:
        fn()  # type: ignore[operator]
    except Exception as exc:  # reporting must never fail the run (CLAUDE.md section 12)
        print(f"(unavailable: {type(exc).__name__}: {exc})", flush=True)


def main() -> int:
    """Print the stage report and write the zip. Always returns 0."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--status", type=Path, default=THESIS_DBP / "reports" / "p2_s3_status.json")
    parser.add_argument("--probe-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "probe")
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--bank", type=Path, default=THESIS_DBP / "configs" / "decoder_bank.yaml")
    parser.add_argument("--zip", type=Path, default=None, help="Archive path, without .zip.")
    parser.add_argument("--stage", choices=("s3", "s2", "final", "truefake", "p4"),
                        default="s3")
    parser.add_argument("--s2-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "s2")
    try:
        args = parser.parse_args()
    except SystemExit:
        print("(report: bad arguments; nothing reported)", flush=True)
        return 0

    def status() -> None:
        print(json.dumps(json.loads(args.status.read_text(encoding="utf-8")), indent=2))

    def members() -> None:
        sys.path.insert(0, str(THESIS_DBP / "src"))
        from dbp.decoders.registry import load_bank_config

        bank = [spec.key for spec in load_bank_config(args.bank).distinct_members()]
        done = sorted(
            npz.stem.removeprefix("p2_probe_")
            for npz in args.probe_dir.glob("p2_probe_*.npz")
            if npz.with_suffix(".done.json").is_file()
        )
        print(f"complete  {len(done)}/{len(bank)}: {done}")
        print(f"remaining {[m for m in bank if m not in done]}")

    def files() -> None:
        for npz in sorted(args.probe_dir.glob("*.npz")):
            print(f"  {npz.name:32s} {npz.stat().st_size / 1e6:8.2f} MB")

    def archive() -> None:
        if args.zip is None:
            print("(no --zip given)")
            return
        made = shutil.make_archive(str(args.zip), "zip", str(args.reports))
        print(f"{made}  {Path(made).stat().st_size / 1e6:.1f} MB")

    def s2_runs() -> None:
        state = json.loads(args.status.read_text(encoding="utf-8"))
        for run in state["runs"]:
            rid = run["run_id"]
            report = args.reports / f"p2_s2_{rid}.json"
            trained = json.loads(report.read_text(encoding="utf-8")) if report.is_file() else {}
            print(f"  {rid:22s} {run['state']:9s} {run.get('device') or '-':>4s}  "
                  f"steps={trained.get('steps_run')} best_val_auc={trained.get('best_val_auc')} "
                  f"best_step={trained.get('best_step')} "
                  f"resumed_from={trained.get('resumed_from_step')} "
                  f"hours={run.get('hours')}  {run.get('note', '')}")

    def s2_scores() -> None:
        for path in sorted(args.reports.glob("p2_s2_scores_*.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            lists = ", ".join(f"{k}={v['rows']}" for k, v in summary["lists"].items())
            print(f"  {summary['run_id']:22s} {lists}")
        for path in sorted(args.s2_dir.glob("*/scores/*.csv")):
            size = path.stat().st_size / 1e6
            print(f"  {path.parent.parent.name}/{path.name:12s} {size:8.2f} MB")

    def final(folder: str = "p3_final") -> None:
        for path in sorted((args.reports / folder).glob("*.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            lists = ", ".join(f"{k}={v['rows']} ({v['seconds_per_image']} s/img)"
                              for k, v in summary["lists"].items())
            print(f"  {summary['model']:22s} {summary['stage']:6s} {lists}")

    def p4_rows() -> None:
        for path in sorted((args.reports / "p4_baselines").glob("*.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            lists = ", ".join(f"{k}={v['rows']} ({v['seconds_per_image']} s/img)"
                              for k, v in summary["lists"].items())
            print(f"  {summary['row']:14s} {summary['condition']:9s} {lists}")

    guarded("status", status)
    if args.stage == "final":
        guarded("final scoring summaries", final)
    elif args.stage == "p4":
        guarded("P4 baseline row summaries", p4_rows)
    elif args.stage == "truefake":
        guarded("TrueFake scoring summaries", lambda: final("p3_truefake"))
    elif args.stage == "s2":
        guarded("runs", s2_runs)
        guarded("score files", s2_scores)
    else:
        guarded("members", members)
        guarded("probe caches", files)
    guarded("reports archive", archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
