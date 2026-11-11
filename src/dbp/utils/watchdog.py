"""A hard wall-clock limit for a GPU step that has no deadline of its own.

Added after a Kaggle training session was lost on 2026-10-18 (log P3-C): every long step
of a runbook must be bounded, including scoring steps that run outside the queue.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

#: Exit code when the deadline passes (as p2_run_queue / p2_run_shards).
EXIT_DEADLINE = 2


def start_watchdog(hours: float, log: Any) -> threading.Timer:
    """Kill this process with :data:`EXIT_DEADLINE` once ``hours`` of wall time pass.

    A daemon timer, so it fires even when the main thread is stuck inside a load or a
    forward pass -- a step that hangs can never keep a GPU session running unbounded.
    Whatever the caller had not yet written is not written, so the step stays due and the
    next session redoes it.
    """
    def expire() -> None:
        log.error("DEADLINE: %.2f h of wall time passed -- stopping now", hours)
        logging.shutdown()
        os._exit(EXIT_DEADLINE)

    timer = threading.Timer(hours * 3600.0, expire)
    timer.daemon = True
    timer.start()
    return timer
