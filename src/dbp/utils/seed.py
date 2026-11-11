"""Deterministic seeding for every DBP experiment.

Reproducibility contract (thesis section 6.6): the same input, member and dtype must
produce the same reconstruction on the same hardware. Non-determinism here would be
indistinguishable from a real forensic signal, so it is switched off globally rather
than per-call.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

DEFAULT_SEED = 2026


def seed_everything(seed: int = DEFAULT_SEED, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and torch (CPU + all CUDA devices).

    Args:
        seed: The seed applied to every RNG.
        deterministic: If True, disable cuDNN autotuning and request deterministic
            kernels. This costs throughput but removes run-to-run variation.

    Returns:
        A record of what was applied, suitable for embedding in a report.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only: a few ops have no deterministic kernel. We want a warning in the
        # log, not a crash that hides an otherwise valid smoke test.
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)

    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
