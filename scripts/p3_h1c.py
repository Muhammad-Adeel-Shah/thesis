"""P3 H1-c (SECONDARY): one decoder vs the bank, on lineages neither model saw.

Pre-registration section 6.1 (FROZEN 2026-10-08): for folds f = 2..8, the AUC on lineage
f's held-out validation reconstructions (against the validation reals), **LOLO-f minus
BANK-1** (sd15 alone, the P3 recipe, ``N*`` steps). Neither model trained on lineage f
and both score the same images, so the comparison is paired. **Success:** the mean of
(LOLO-f - BANK-1) over folds 2-8 is >= +2 AUC points **and** its 95% paired bootstrap CI
excludes 0. Every fold is reported. Validation data only -- never Panel B.

**Inputs** (per-image files, ``dbp.eval.heldout_scores`` format):

* ``lolo_f<f>_seed0_step_<N*(-f)>.csv`` -- each fold rescored at its frozen ``N*(-f)``
  (``reports/p3_nstar.json``);
* ``bank_1_seed0_step_<N*>.csv`` -- BANK-1's own held-out scores at its final step,
  which cover every member it did not train on; fold f uses the rows of lineage f's
  members.

**The bootstrap** (section 4: 2,000 resamples, seed 2026, paired; fakes within slice,
reals within the real pool): per resample the validation reals are drawn once, shared by
every fold and both models; each fold's reconstructions are drawn within that fold; the
same draw is applied to both models. The CI is the 2.5th-97.5th percentile of the mean
difference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.heldout_scores import read_scores  # noqa: E402
from dbp.eval.metrics import roc_auc  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

FOLDS = tuple(range(2, 9))
MARGIN = 0.02
RESAMPLES = 2000
SEED = 2026
LEVEL = 0.95


class H1cError(ValueError):
    """Inputs the frozen comparison cannot be computed from."""


def fold_members(runs_config: Path) -> dict[int, list[str]]:
    """Fold f -> the distinct members of its held-out lineage."""
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(runs_config, bank)
    out = {}
    for fold in FOLDS:
        held = set(runs.by_id(f"lolo_f{fold}").excluded_lineages)
        out[fold] = sorted(s.key for s in bank.distinct_members() if s.lineage in held)
    return out


def split_rows(rows: list[dict[str, Any]], members: list[str]) -> tuple[dict, dict]:
    """reals by image id; lineage-f reconstructions by (image id, member)."""
    reals = {r["image_id"]: r["score"] for r in rows if r["label"] == 0}
    fakes = {(r["image_id"], r["member"]): r["score"] for r in rows
             if r["label"] == 1 and r["member"] in members}
    return reals, fakes


def auc(reals: np.ndarray, fakes: np.ndarray) -> float:
    """AUC of fakes against reals."""
    return float(roc_auc(np.r_[np.zeros(len(reals)), np.ones(len(fakes))], np.r_[reals, fakes]))


def paired_bootstrap(data: dict[int, dict[str, np.ndarray]], resamples: int = RESAMPLES,
                     seed: int = SEED, level: float = LEVEL) -> dict[str, Any]:
    """CI of the mean over folds of AUC(LOLO-f) - AUC(BANK-1), paired.

    ``data[f]`` holds ``lolo_real``/``bank_real`` (aligned: index i is the same real in
    every fold and both models) and ``lolo_fake``/``bank_fake`` (aligned within the fold).
    """
    folds = sorted(data)
    n = {len(data[f]["lolo_real"]) for f in folds}
    if len(n) != 1:
        raise H1cError(f"the real pool differs across folds: {sorted(n)}")
    n_real = n.pop()
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        real_idx = rng.integers(0, n_real, n_real)
        diffs = []
        for f in folds:
            d = data[f]
            fake_idx = rng.integers(0, len(d["lolo_fake"]), len(d["lolo_fake"]))
            diffs.append(auc(d["lolo_real"][real_idx], d["lolo_fake"][fake_idx])
                         - auc(d["bank_real"][real_idx], d["bank_fake"][fake_idx]))
        values[b] = float(np.mean(diffs))
    tail = (1.0 - level) / 2.0 * 100.0
    return {"lower": float(np.percentile(values, tail)),
            "upper": float(np.percentile(values, 100.0 - tail)),
            "level": level, "resamples": resamples, "seed": seed,
            "percentiles": [tail, 100.0 - tail]}


def load_inputs(scores_dir: Path, nstar: dict[str, Any], n_star: int,
                members: dict[int, list[str]]) -> tuple[dict, dict]:
    """Aligned score arrays per fold, and a provenance record per file."""
    bank_path = scores_dir / f"bank_1_seed0_step_{n_star:05d}.csv"
    if not bank_path.is_file():
        raise H1cError(f"no BANK-1 scores at N* = {n_star}: {bank_path}")
    bank_rows, bank_meta = read_scores(bank_path)
    if bank_meta.get("run_id") != "bank_1_seed0" or int(bank_meta.get("step", -1)) != n_star:
        raise H1cError(f"{bank_path.name}: sidecar says {bank_meta.get('run_id')} step "
                       f"{bank_meta.get('step')}")
    provenance = {"bank_1": {"file": bank_path.name, "sha256": bank_meta["scores_sha256"],
                             "snapshot_sha256": bank_meta.get("snapshot_sha256")}}
    data, first_reals = {}, None
    for fold in FOLDS:
        step = int(nstar["nested"][str(fold)]["step"])
        path = scores_dir / f"lolo_f{fold}_seed0_step_{step:05d}.csv"
        if not path.is_file():
            raise H1cError(f"fold {fold}: no LOLO scores at its N*(-f) = {step}: {path}")
        rows, meta = read_scores(path)
        if meta.get("limit"):
            raise H1cError(f"{path.name}: a --limit smoke file")
        lolo_reals, lolo_fakes = split_rows(rows, members[fold])
        bank_reals, bank_fakes = split_rows(bank_rows, members[fold])
        if sorted(lolo_reals) != sorted(bank_reals):
            raise H1cError(f"fold {fold}: LOLO and BANK-1 scored different reals")
        if sorted(lolo_fakes) != sorted(bank_fakes) or not lolo_fakes:
            raise H1cError(f"fold {fold}: LOLO and BANK-1 scored different reconstructions "
                           f"of {members[fold]} ({len(lolo_fakes)} vs {len(bank_fakes)})")
        ids = sorted(lolo_reals)
        if first_reals is None:
            first_reals = ids
        elif ids != first_reals:
            raise H1cError(f"fold {fold} was scored on different reals")
        keys = sorted(lolo_fakes)
        data[fold] = {"lolo_real": np.array([lolo_reals[i] for i in ids]),
                      "bank_real": np.array([bank_reals[i] for i in ids]),
                      "lolo_fake": np.array([lolo_fakes[k] for k in keys]),
                      "bank_fake": np.array([bank_fakes[k] for k in keys])}
        provenance[f"lolo_f{fold}"] = {"file": path.name, "step": step,
                                       "sha256": meta["scores_sha256"],
                                       "snapshot_sha256": meta.get("snapshot_sha256")}
    return data, provenance


def main() -> int:
    """Compute H1-c once from the committed per-image files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nstar", type=Path, default=THESIS_DBP / "reports" / "p3_nstar.json")
    parser.add_argument("--scores-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p3_heldout_scores")
    parser.add_argument("--runs-config", type=Path,
                        default=THESIS_DBP / "configs" / "p3_runs.yaml")
    parser.add_argument("--out", type=Path, default=THESIS_DBP / "reports" / "p3_h1c.json")
    args = parser.parse_args()
    log = setup_logging("p3_h1c")
    nstar = json.loads(args.nstar.read_text(encoding="utf-8"))
    n_star = int(nstar["n_star"]["step"])
    members = fold_members(args.runs_config)
    try:
        data, provenance = load_inputs(args.scores_dir, nstar, n_star, members)
    except H1cError as error:
        log.error("%s", error)
        return 1
    table = []
    for fold in FOLDS:
        d = data[fold]
        lolo, bank = auc(d["lolo_real"], d["lolo_fake"]), auc(d["bank_real"], d["bank_fake"])
        table.append({"fold": fold, "members": members[fold], "lolo_auc": round(lolo, 5),
                      "bank_1_auc": round(bank, 5), "difference": round(lolo - bank, 5),
                      "reals": len(d["lolo_real"]), "reconstructions": len(d["lolo_fake"])})
    mean = float(np.mean([lo["difference"] for lo in table]))
    ci = paired_bootstrap(data)
    excludes_zero = ci["lower"] > 0 or ci["upper"] < 0
    verdict = "PASS" if mean >= MARGIN and excludes_zero else "FAIL"
    payload = {
        "what": "H1-c: LOLO-f minus BANK-1 on lineage f's held-out validation "
                "reconstructions, folds 2-8 (pre-registration section 6.1)",
        "nstar": {"path": args.nstar.as_posix(),
                  "sha256": hashlib.sha256(args.nstar.read_bytes()).hexdigest(),
                  "n_star": n_star},
        "inputs": provenance, "folds": table, "mean_difference": mean, "ci": ci,
        "criteria": {"mean >= +2 points": mean >= MARGIN, "95% CI excludes 0": excludes_zero},
        "verdict": verdict,
        "resampled": "reals once per resample, shared by every fold and both models; each "
                     "fold's reconstructions within the fold; the same draw for both models",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, args.out)
    for row in table:
        log.info("fold %d %s: LOLO %.5f  BANK-1 %.5f  diff %+.5f", row["fold"], row["members"],
                 row["lolo_auc"], row["bank_1_auc"], row["difference"])
    log.info("mean %+.5f, 95%% CI [%+.5f, %+.5f]: H1-c %s", mean, ci["lower"], ci["upper"],
             verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
