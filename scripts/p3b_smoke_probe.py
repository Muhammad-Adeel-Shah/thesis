"""P3b: run the UNCHANGED S2 trainer's smoke in-process and measure it from outside.

The trainer is byte-identical by decision (user, 2026-10-21; ``tests/test_p3_s2d.py``), so
the ViT-L smoke measures it without touching it:

* **step time:** ``torch.optim.AdamW.step`` is wrapped to timestamp each optimiser step;
  the interval between consecutive steps is one training step (batch build, forward,
  backward, update). The trainer reads the loss as a float every step, which synchronises
  the GPU, so the intervals are real. The first ``--warmup`` intervals are left out of the
  mean (kernel loading, allocator growth);
* **peak GPU memory:** ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` after
  the trainer returns, in this process;
* **gradient checkpointing off** (``--gradient-checkpointing off``, a measurement only):
  transformers' ``PreTrainedModel.gradient_checkpointing_enable`` is replaced by a no-op for
  this process, so the trainer's call does nothing. The recipe (``on``) leaves it alone.
  Every call is counted, so the record shows the switch took effect;
* **out of memory:** caught and recorded, not raised.

Usage: ``p3b_smoke_probe.py --gradient-checkpointing on --out probe.json -- <trainer args>``
(the trainer args must include ``--smoke N``). Exits with the trainer's exit status (1 on
out of memory); the runbook decides what a failure means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
TRAINER = THESIS_DBP / "scripts" / "p2_train_s2.py"


def trainer_sha256() -> str:
    """The trainer's sha256 with LF line endings (the guard test's rule)."""
    return hashlib.sha256(TRAINER.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    """Run the trainer's smoke under the probes; write the record."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gradient-checkpointing", choices=("on", "off"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    trainer_args = [a for a in args.trainer_args if a != "--"]
    if "--smoke" not in trainer_args:
        print("the probe runs smokes only: pass --smoke N to the trainer")
        return 2

    import torch
    from transformers import modeling_utils

    stamps: list[float] = []
    original_step = torch.optim.AdamW.step

    def timed_step(self: Any, *a: Any, **k: Any) -> Any:
        result = original_step(self, *a, **k)
        stamps.append(time.perf_counter())
        return result

    torch.optim.AdamW.step = timed_step
    calls = {"gradient_checkpointing_enable": 0}
    original_enable = modeling_utils.PreTrainedModel.gradient_checkpointing_enable

    def enable(self: Any, *a: Any, **k: Any) -> Any:
        calls["gradient_checkpointing_enable"] += 1
        if args.gradient_checkpointing == "on":
            return original_enable(self, *a, **k)
        return None

    modeling_utils.PreTrainedModel.gradient_checkpointing_enable = enable
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    code, out_of_memory, error = 0, False, None
    sys.argv = [str(TRAINER), *trainer_args]
    try:
        runpy.run_path(str(TRAINER), run_name="__main__")
    except SystemExit as exit_:
        code = int(exit_.code or 0) if not isinstance(exit_.code, str) else 1
    except torch.cuda.OutOfMemoryError as oom:
        code, out_of_memory, error = 1, True, str(oom).splitlines()[0]
    intervals = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    steady = intervals[args.warmup:]
    record = {
        "what": "P3b ViT-L smoke, measured from outside the unchanged trainer",
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_enable_calls": calls["gradient_checkpointing_enable"],
        "trainer_sha256": trainer_sha256(), "trainer_args": trainer_args,
        "exit_code": code, "out_of_memory": out_of_memory, "error": error,
        "optimiser_steps": len(stamps), "warmup_intervals_excluded": args.warmup,
        "step_seconds_after_warmup": round(sum(steady) / len(steady), 4) if steady else None,
        "step_seconds_max_after_warmup": round(max(steady), 4) if steady else None,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "peak_gpu_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)
        if cuda else None,
        "peak_gpu_memory_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)
        if cuda else None,
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30,
                                      3) if cuda else None,
        "gpu": torch.cuda.get_device_name() if cuda else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"PROBE {args.gradient_checkpointing}: exit {code}, out of memory {out_of_memory}, "
          f"{len(stamps)} steps, {record['step_seconds_after_warmup']} s/step after warm-up, "
          f"peak allocated {record['peak_gpu_memory_allocated_gib']} GiB of "
          f"{record['gpu_total_memory_gib']} GiB")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
