"""Every output path a P2 run writes, built in one place.

Two parallel jobs must never write the same path. In the S3 recovery both jobs ran as
``shard 0/1`` and wrote ``p2_probe_timing_shard0.json``; the later one replaced the
other's record (I-053). The same defect was waiting in Session E: the heads, fusions and
analysis outputs were keyed by protocol only, so the second P-A seed would have
overwritten the first seed's fitted fusions.

The rule, enforced by ``tests/test_output_paths_unique.py``:

* **A run's results are keyed by its run id** (``<protocol>_seed<seed>``). The queue
  hands a run to exactly one worker at a time, so the run id alone is unique among
  concurrent jobs. The GPU index is deliberately *not* in a checkpoint's path: a run
  resumed in a later session may land on the other GPU, and a path containing the
  device would make it start from step 0.
* **Every log and scratch path also carries the GPU index**, so the record of which
  device did what survives, and a smoke test on GPU 0 cannot touch GPU 1's files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def run_id(protocol: str, seed: int) -> str:
    """The id that names everything one ``(protocol, seed)`` run writes."""
    if not protocol or any(c in protocol for c in "/\\ :"):
        raise ValueError(f"protocol must be a plain name, got {protocol!r}")
    return f"{protocol}_seed{int(seed)}"


def device_tag(device: str) -> str:
    """``gpu0`` / ``gpu1`` / ``cpu`` -- the device part of a log name."""
    device = str(device)
    if device == "cpu":
        return "cpu"
    if not device.isdigit():
        raise ValueError(f"device must be 'cpu' or a CUDA index, got {device!r}")
    return f"gpu{device}"


@dataclass(frozen=True)
class S2RunPaths:
    """Files one S2 training run owns."""

    run_dir: Path
    checkpoint: Path
    checkpoint_tmp: Path
    lora: Path
    head: Path
    head_tmp: Path
    report: Path
    report_tmp: Path
    scores_dir: Path
    scores_summary: Path
    scores_summary_tmp: Path

    def all(self) -> list[Path]:
        """Every path, for the uniqueness test."""
        return [getattr(self, name) for name in self.__dataclass_fields__]

    def scores(self, list_name: str) -> Path:
        """Scores of one image list (``panel``, ``luma``, ``val``)."""
        return self.scores_dir / f"{list_name}.csv"


def s2_run_paths(out_root: Path, reports_dir: Path, rid: str) -> S2RunPaths:
    """Where an S2 run keeps its checkpoint, selected model, report and scores."""
    run_dir = Path(out_root) / rid
    reports_dir = Path(reports_dir)
    return S2RunPaths(
        run_dir=run_dir,
        checkpoint=run_dir / "checkpoint.pt",
        checkpoint_tmp=run_dir / "checkpoint.tmp.pt",
        lora=run_dir / "lora",
        head=run_dir / "head.pt",
        head_tmp=run_dir / "head.tmp.pt",
        report=reports_dir / f"p2_s2_{rid}.json",
        report_tmp=reports_dir / f"p2_s2_{rid}.tmp",
        scores_dir=run_dir / "scores",
        scores_summary=reports_dir / f"p2_s2_scores_{rid}.json",
        scores_summary_tmp=reports_dir / f"p2_s2_scores_{rid}.tmp",
    )


@dataclass(frozen=True)
class WorkerLogPaths:
    """Logs of one run on one device."""

    train: Path
    infer: Path
    smoke: Path
    smoke_dir: Path

    def all(self) -> list[Path]:
        """Every path, for the uniqueness test."""
        return [self.train, self.infer, self.smoke, self.smoke_dir]


def worker_log_paths(log_dir: Path, scratch: Path, rid: str, device: str) -> WorkerLogPaths:
    """Per-(run, device) logs, and the scratch directory a smoke test writes into."""
    tag = f"{rid}_{device_tag(device)}"
    log_dir = Path(log_dir)
    return WorkerLogPaths(
        train=log_dir / f"s2_{tag}_train.log",
        infer=log_dir / f"s2_{tag}_infer.log",
        smoke=log_dir / f"s2_{tag}_smoke.log",
        smoke_dir=Path(scratch) / f"smoke_{tag}",
    )


@dataclass(frozen=True)
class HeadPaths:
    """Session E artefacts fitted for one run."""

    s1_probe: Path
    s3_lr: Path
    s3_set: Path
    report: Path
    report_tmp: Path

    def all(self) -> list[Path]:
        """Every path, for the uniqueness test."""
        return [self.s1_probe, self.s3_lr, self.s3_set, self.report, self.report_tmp]


def head_paths(heads_dir: Path, reports_dir: Path, rid: str) -> HeadPaths:
    """Heads, temperatures and fusions are fitted per run, not per protocol.

    The fusions include S2's validation scores, which differ between seeds; keyed by
    protocol, ``p_a_sd1_seed1`` would have replaced ``p_a_sd1_seed0``'s fusions.
    """
    heads_dir, reports_dir = Path(heads_dir), Path(reports_dir)
    return HeadPaths(
        s1_probe=heads_dir / f"s1_probe_{rid}.npz",
        s3_lr=heads_dir / f"s3_lr_{rid}.npz",
        s3_set=heads_dir / f"s3_set_{rid}.pt",
        report=reports_dir / f"p2_heads_{rid}.json",
        report_tmp=reports_dir / f"p2_heads_{rid}.tmp",
    )


@dataclass(frozen=True)
class ScorePaths:
    """The single-shot panel scores of one run."""

    scores: Path
    summary: Path
    summary_tmp: Path

    def all(self) -> list[Path]:
        """Every path, for the uniqueness test."""
        return [self.scores, self.summary, self.summary_tmp]


def score_paths(out_dir: Path, rid: str) -> ScorePaths:
    """Where ``p2_score.py`` writes one run's panel scores."""
    out_dir = Path(out_dir)
    return ScorePaths(
        scores=out_dir / f"scores_{rid}.csv",
        summary=out_dir / f"scores_{rid}_summary.json",
        summary_tmp=out_dir / f"scores_{rid}_summary.tmp",
    )


def analysis_dir(out_dir: Path, rid: str) -> Path:
    """``p2_analyze.py`` writes ``metrics.csv``, ``go_no_go.json`` ... in here, per run."""
    return Path(out_dir) / rid
