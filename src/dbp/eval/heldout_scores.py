"""Held-out-lineage scoring, shared by the S2 trainer and ``scripts/p3_score_heldout.py``.

**One implementation, two callers.** The trainer scores the held-out lineage while it
trains (P3 LOLO, and the final step of every non-LOLO run); the scorer re-scores a saved
snapshot later (the H1-a CI needs per-image scores the LOLO reports did not keep). Both
call :func:`score_items` with the same keys, crop and batch, so a rescored snapshot
reproduces the AUC its training run reported -- which ``p3_select_nstar.py ci`` checks.

**The crop.** Each image gets a fixed crop of ``eval_crop`` pixels at a position drawn
from a seed derived from its key (:func:`stable_seed`) with every augmentation switched
off: the real by its ``image_id``, a reconstruction by its ``pair_id``.

**The per-image file** (user decision 2026-10-17, D2): CSV ``image_id, member, label,
score`` -- a real has an empty ``member`` and label 0, a reconstruction its member key
and label 1, and ``image_id`` is always the real's id, so a reconstruction is
``(image_id, member)`` and two models' files pair up row by row. A JSON sidecar of the
same stem carries the run, the step, the snapshot scored and its sha256
(:func:`snapshot_digest`), and the CSV's own sha256.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dbp.eval.metrics import roc_auc
from dbp.eval.panel import DIAGNOSTIC_ONLY_SETS

SCORE_COLUMNS = ("image_id", "member", "label", "score")
#: Every augmentation probability, switched off for scoring.
NO_AUGMENTATION = ("flip_p", "jpeg_p", "blur_p", "noise_p", "resize_p", "greyscale_p")


class HeldoutScoreError(ValueError):
    """A per-image score file that is malformed or does not match its sidecar."""


def stable_seed(text: str) -> int:
    """A 31-bit seed from a string, identical in every process.

    The validation crop used to be seeded with ``hash(pair_id)``. Python randomises
    ``str`` hashes per process, so every run, every seed and every resumed session
    validated on different crops, and the curves were not comparable.
    """
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(),
                          "big") % (2**31)


def heldout_rows(
    splits: Path, pairs: Path, members: set[str], usable_reals: set[str]
) -> list[dict[str, str]]:
    """Validation-split pair rows of the held-out members.

    The same filters as the trainer's training data: flagged reals (D-039) and the
    diagnostic-only T-ctrl set are excluded, so the held-out reals are the validation
    reals the in-protocol validation draws from.
    """
    side_of = {}
    with splits.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            side_of[row["row_id"]] = row["split"]
    rows = []
    with pairs.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["member"] in members and row["image_id"] in usable_reals
                    and row.get("set_id", "") not in DIAGNOSTIC_ONLY_SETS
                    and side_of.get(row["pair_id"]) == "val"):
                rows.append(row)
    return rows


def score_items(
    items: Sequence[tuple[str, Path]], forward: Callable[[Any], Any],
    normalisation: dict[str, Any], eval_crop: int, device: str, amp: bool, batch_size: int,
) -> list[float]:
    """Score images on their fixed, seeded crop with no augmentation, in batches.

    Args:
        items: ``(key, path)`` pairs; ``key`` seeds the crop position.
        forward: Batch of normalised crops -> logits (model and head, in eval mode).
        normalisation: The backbone processor's ``image_mean`` / ``image_std``.
        eval_crop: Crop side in pixels.
        device: Torch device of the model.
        amp: fp16 autocast on (CUDA only; never bf16, the T4 is sm_75).
        batch_size: Images per forward pass.
    """
    import torch

    from dbp.streams.s1_semantic import normalise_batch
    from dbp.train.augment import PAIR_SIZE, apply, sample_params
    from dbp.utils.io import load_rgb_uint8

    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            crops = []
            for key, path in items[start : start + batch_size]:
                params = sample_params(np.random.default_rng(stable_seed(key)), PAIR_SIZE,
                                       eval_crop, settings={k: 0.0 for k in NO_AUGMENTATION})
                crops.append(apply(load_rgb_uint8(path), params))
            batch = normalise_batch(np.stack(crops), normalisation["image_mean"],
                                    normalisation["image_std"]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out.extend(forward(batch).float().cpu().tolist())
    return out


def score_heldout(
    real_ids: Sequence[str], rows: Sequence[dict[str, str]], root: Path,
    reals_index: dict[str, str], score: Callable[[Sequence[tuple[str, Path]]], list[float]],
) -> list[dict[str, Any]]:
    """Per-image rows for the held-out set: the reals first, then every reconstruction."""
    real_scores = score([(rid, root / reals_index[rid]) for rid in real_ids])
    fake_scores = score([(row["pair_id"], root / row["relative_path"]) for row in rows])
    table = [{"image_id": rid, "member": "", "label": 0, "score": s}
             for rid, s in zip(real_ids, real_scores, strict=True)]
    table += [{"image_id": row["image_id"], "member": row["member"], "label": 1, "score": s}
              for row, s in zip(rows, fake_scores, strict=True)]
    return table


def auc_summary(table: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pooled held-out AUC and the AUC of each member against the same reals."""
    reals = [float(r["score"]) for r in table if int(r["label"]) == 0]
    fakes = [r for r in table if int(r["label"]) == 1]
    if not reals or not fakes:
        return {"heldout_auc": None, "heldout_auc_by_member": {}}

    def auc(scores: list[float]) -> float:
        return round(float(roc_auc([0] * len(reals) + [1] * len(scores), reals + scores)), 5)

    by_member: dict[str, float] = {}
    for member in sorted({r["member"] for r in fakes}):
        by_member[member] = auc([float(r["score"]) for r in fakes if r["member"] == member])
    return {"heldout_auc": auc([float(r["score"]) for r in fakes]),
            "heldout_auc_by_member": by_member}


def file_sha256(path: Path) -> str:
    """sha256 of one file, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_digest(snapshot: Path) -> str:
    """One sha256 for a snapshot directory (LoRA adapters + head).

    The sha256 of the lines ``<relative posix path>\\0<file sha256>\\n`` over every file,
    sorted by path, so a renamed, added or changed file changes it.
    """
    files = sorted(p for p in Path(snapshot).rglob("*") if p.is_file())
    if not files:
        raise HeldoutScoreError(f"{snapshot}: empty snapshot")
    lines = "".join(f"{p.relative_to(snapshot).as_posix()}\0{file_sha256(p)}\n" for p in files)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def write_scores(csv_path: Path, table: Iterable[dict[str, Any]], meta: dict[str, Any]) -> None:
    """Write the per-image CSV and its JSON sidecar, each atomically."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    rows = list(table)
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            score = float(row["score"])
            if not math.isfinite(score):
                raise HeldoutScoreError(f"non-finite score for {row['image_id']} {row['member']}")
            writer.writerow({"image_id": row["image_id"], "member": row["member"],
                             "label": int(row["label"]), "score": f"{score:.6f}"})
    os.replace(tmp, csv_path)
    side = sidecar(csv_path)
    payload = {**meta, "scores_csv": csv_path.name, "scores_sha256": file_sha256(csv_path),
               "rows": len(rows), "reals": sum(int(r["label"]) == 0 for r in rows),
               "reconstructions": sum(int(r["label"]) == 1 for r in rows),
               **auc_summary(rows)}
    side_tmp = side.with_name(side.name + ".tmp")
    side_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(side_tmp, side)


def sidecar(csv_path: Path) -> Path:
    """The JSON sidecar of a per-image score CSV."""
    return Path(csv_path).with_suffix(".json")


def read_scores(csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a per-image CSV and its sidecar; refuse a mismatch or a malformed row."""
    csv_path = Path(csv_path)
    meta = json.loads(sidecar(csv_path).read_text(encoding="utf-8"))
    if meta.get("scores_sha256") != file_sha256(csv_path):
        raise HeldoutScoreError(f"{csv_path}: sha256 differs from its sidecar")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCORE_COLUMNS:
            raise HeldoutScoreError(f"{csv_path}: columns {reader.fieldnames}, "
                                    f"expected {list(SCORE_COLUMNS)}")
        for line, row in enumerate(reader, start=2):
            label = int(row["label"])
            if label not in (0, 1) or (label == 0) != (row["member"] == ""):
                raise HeldoutScoreError(f"{csv_path}:{line}: label {label} with member "
                                        f"{row['member']!r}")
            key = (row["image_id"], row["member"])
            if key in seen:
                raise HeldoutScoreError(f"{csv_path}:{line}: {key} repeated")
            seen.add(key)
            score = float(row["score"])
            if not math.isfinite(score):
                raise HeldoutScoreError(f"{csv_path}:{line}: non-finite score")
            rows.append({"image_id": row["image_id"], "member": row["member"],
                         "label": label, "score": score})
    return rows, meta
