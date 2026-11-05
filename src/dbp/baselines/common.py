"""Helpers shared by the P4 baseline adapters (spec ``docs/P4_BASELINES_SPEC.md``).

Every checkpoint is loaded the same way (spec section 1, CLAUDE.md section 5): its sha256
is checked first, then ``torch.load(..., weights_only=True)``. If ``weights_only`` refuses
a file, the run stops; nothing is ever unpickled. Key sets are compared exactly, so a
checkpoint that does not fit the architecture stops the run instead of loading partially.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from dbp.data.manifest import file_sha256

#: Set only by the local runbook dry-run; enables the tiny-architecture seams.
DRYRUN_ENV = "DBP_DRYRUN"


class BaselineError(RuntimeError):
    """A baseline that may not or cannot be built, loaded or scored."""


def dryrun() -> bool:
    """Whether the dry-run seams may be used (``DBP_DRYRUN=1``, local CPU only)."""
    return os.environ.get(DRYRUN_ENV) == "1"


def check_sha256(path: Path, expected: str, what: str) -> str:
    """Stop unless ``path`` exists and hashes to ``expected``; returns the digest."""
    if not Path(path).is_file():
        raise BaselineError(f"{what}: {path} does not exist")
    digest = file_sha256(path)
    if digest != expected:
        raise BaselineError(f"{what}: sha256 {digest} is not the recorded {expected} ({path})")
    return digest


def load_pth(path: Path, expected_sha256: str, what: str) -> Any:
    """sha256 check, then ``torch.load(weights_only=True)`` on the CPU.

    Raises:
        BaselineError: On a hash mismatch, or when ``weights_only`` refuses the file
            (spec section 1: stop and ask; nothing is unpickled).
    """
    import pickle

    import torch

    check_sha256(path, expected_sha256, what)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise BaselineError(f"{what}: torch.load(weights_only=True) refused {path}: "
                            f"{error}. Stop and ask; the file is not unpickled.") from error


def key_diff(expected: list[str] | set[str], found: list[str] | set[str]) -> dict[str, list[str]]:
    """``missing`` (expected, not found) and ``unexpected`` (found, not expected) keys."""
    expected, found = set(expected), set(found)
    return {"missing": sorted(expected - found), "unexpected": sorted(found - expected)}


def require_exact_keys(expected: list[str] | set[str], found: list[str] | set[str],
                       what: str) -> None:
    """Stop on any missing or unexpected key (spec section 1)."""
    diff = key_diff(expected, found)
    if diff["missing"] or diff["unexpected"]:
        raise BaselineError(f"{what}: {len(diff['missing'])} missing key(s) "
                            f"{diff['missing'][:8]}, {len(diff['unexpected'])} unexpected "
                            f"key(s) {diff['unexpected'][:8]}")


def compare_state(held: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Whether two state dicts hold the same tensors, exactly or after a dtype cast.

    ``equal`` is True when every tensor equals its reference exactly, or exactly once one
    of the two is cast to the other's dtype (fp32 weights saved as fp16, say). When they
    differ, ``per_tensor`` gives the max absolute difference of every tensor that is not
    equal (user decision 2026-11-05: record it and continue).
    """
    import torch

    if set(held) != set(reference):
        return {"same_keys": False, "equal": False,
                "only_held": len(set(held) - set(reference)),
                "only_reference": len(set(reference) - set(held))}
    per_tensor: dict[str, float] = {}
    for key, tensor in held.items():
        ref = reference[key]
        if tensor.shape != ref.shape:
            return {"same_keys": True, "equal": False, "shape_mismatch": key}
        same = torch.equal(tensor, ref) or torch.equal(tensor.to(ref.dtype), ref) or \
            torch.equal(tensor, ref.to(tensor.dtype))
        if not same:
            per_tensor[key] = float((tensor.double() - ref.double()).abs().max())
    worst = max(per_tensor.values(), default=0.0)
    return {"same_keys": True, "equal": not per_tensor, "max_abs_diff": worst,
            **({"per_tensor": per_tensor} if per_tensor else {})}


def fire_seed(image_id: str) -> int:
    """FIRE's per-image seed (spec section 3, item 7.13).

    The first 8 bytes of sha256(image id, UTF-8) as a big-endian integer, masked to 63
    bits so it is a valid ``torch.Generator`` seed.
    """
    digest = hashlib.sha256(image_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def center_crop_max(img: np.ndarray, cap: int) -> np.ndarray:
    """Centre-crop each side to at most ``cap`` pixels; never resamples or pads.

    AEROBLADE's whole-image setting, capped at 768 x 768 (spec section 3, user decision
    2026-10-31: crop, never drop).
    """
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")
    h, w = img.shape[:2]
    new_h, new_w = min(h, cap), min(w, cap)
    top, left = (h - new_h) // 2, (w - new_w) // 2
    return img[top: top + new_h, left: left + new_w]
