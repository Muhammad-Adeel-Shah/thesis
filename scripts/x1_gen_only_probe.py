"""EXPLORATORY diagnostic X1: is frozen DINOv3 blind to VAE-only images?

Replicates the question of *Simplicity Prevails* (arXiv 2602.01738, Table 9) in our
pipeline at ViT-B/16. **Exploratory: it changes no P2 number and no part of the gate.**

1. **Generation-only probe.** Frozen DINOv3-B features (the S1 caches) and a logistic
   regression with S1's settings (``fit_probe``, the same ``C`` grid), trained **only**
   on train-split real COCO vs train-split SD1.5 full generations (``gen_matched``). No
   reconstruction is seen in training. ``C`` is chosen on a seeded 20% hold-out of the
   *train* split, drawn by COCO group, so the val split is untouched until testing.
2. **Tests, held out (val split):** (a) ``gen_matched`` vs the val reals; (b) the val
   reals vs their reconstructions by each of the 12 bank decoders; (c) E7 DDA-COCO
   subsets on the panel (existing images) against E7's reals.
3. **S2 beside it** on the same images: P-A seed 0 and ``full_bank`` seed 0.
4. **Metrics:** AUC and the detection rate on fakes at the 5% FPR threshold fixed on the
   491 val reals (``calibrate_on_reals``); 95% bootstrap CIs, 1,000 resamples, seed
   2026. For (b) the bootstrap resamples real/reconstruction *pairs*; elsewhere reals and
   fakes are resampled independently. The threshold is not re-fitted per resample.
5. **One figure:** AUC per decoder, probe vs S2, chance at 0.5.
6. **S1 extracted, not recomputed:** S1's per-decoder validation AUC and its E7
   per-subset results from the committed reports. S1 trained on reconstructions, so it
   is **not** the clean test.

Usage:
    python scripts/x1_gen_only_probe.py --s1-list .../p2_s1_images.csv \\
        --splits .../p2_splits.csv --s1-cache DIR --s2-dir DIR --out-dir reports/p2_x1 \\
        --figure-dir DIR
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.join import join_streams  # noqa: E402
from dbp.eval.metrics import roc_auc  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402
from dbp.streams.s1_semantic import fit_probe  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

LABEL = "EXPLORATORY (X1) -- not part of the P2 gate; no P2 result changes"
RESAMPLES = 1000
SEED = 2026
TUNE_SHARE = 0.20
TARGET_FPR = 0.05
S2_RUNS = ("p_a_sd1_seed0", "full_bank_seed0")
E7_REAL_SUBSET = "real"
CACHE = "p2_s1_dinov3-vitb16_{}.npz"


class SplitError(RuntimeError):
    """The probe's training and test images overlap."""


def group_of(row: dict[str, str]) -> str:
    """``real``, ``gen_matched`` or the decoder of an S1-list row."""
    if row["label"] == "real":
        return "real"
    parts = row["relative_path"].replace("\\", "/").split("/")
    if parts[0] == "gen_matched":
        return "gen_matched"
    if parts[0] == "pairs" and len(parts) > 2:
        return parts[1]
    raise SplitError(f"cannot tell the group of {row['image_id']}")


def in_tune(group_id: str) -> bool:
    """Seeded, group-wise 20% hold-out of the train split, for choosing ``C`` only."""
    digest = hashlib.blake2b(f"x1-tune:{group_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < TUNE_SHARE


def build_split(
    s1_rows: Sequence[dict[str, str]], split_group: dict[str, str]
) -> dict[str, Any]:
    """The probe's fit / tune / test ids.

    Args:
        s1_rows: The S1 list.
        split_group: ``image id -> COCO group id`` from the split file; a reconstruction
            has the group of its source real.

    Returns:
        ``{"fit": [...], "tune": [...], "test_reals": [...], "test_gen": [...],
        "test_recon": {member: [(real id, recon id), ...]}}``.

    Raises:
        SplitError: If any id or COCO group is in both training (fit + tune) and test.
    """
    train = [r for r in s1_rows if r["view"] == "train" and group_of(r) in ("real", "gen_matched")]
    val = [r for r in s1_rows if r["view"] == "val"]
    fit = sorted(r["image_id"] for r in train if not in_tune(split_group[r["image_id"]]))
    tune = sorted(r["image_id"] for r in train if in_tune(split_group[r["image_id"]]))
    reals = sorted(r["image_id"] for r in val if group_of(r) == "real")
    gen = sorted(r["image_id"] for r in val if group_of(r) == "gen_matched")
    recon: dict[str, list[tuple[str, str]]] = {}
    real_set = set(reals)
    for row in val:
        member = group_of(row)
        if member in ("real", "gen_matched"):
            continue
        source = row["image_id"].rsplit("__", 1)[0]
        if source not in real_set:
            raise SplitError(f"reconstruction {row['image_id']} has no val real {source}")
        recon.setdefault(member, []).append((source, row["image_id"]))
    split = {"fit": fit, "tune": tune, "test_reals": reals, "test_gen": gen,
             "test_recon": {m: sorted(v) for m, v in sorted(recon.items())}}
    assert_disjoint(split, split_group)
    return split


def assert_disjoint(split: dict[str, Any], split_group: dict[str, str]) -> None:
    """No image id -- and no COCO group -- in both training and test."""
    training = set(split["fit"]) | set(split["tune"])
    test = set(split["test_reals"]) | set(split["test_gen"]) | {
        i for pairs in split["test_recon"].values() for pair in pairs for i in pair}
    shared = sorted(training & test)
    if shared:
        raise SplitError(f"{len(shared)} id(s) in both training and test: {shared[:10]}")
    groups = {split_group[i] for i in training} & {split_group[i] for i in test}
    if groups:
        raise SplitError(f"{len(groups)} COCO group(s) in both training and test: "
                         f"{sorted(groups)[:10]}")
    if set(split["fit"]) & set(split["tune"]):
        raise SplitError("fit and tune overlap")


def bootstrap(reals: np.ndarray, fakes: np.ndarray, threshold: float,
              paired: bool) -> dict[str, Any]:
    """AUC and detection rate at a fixed threshold, with percentile 95% CIs."""
    def stats(r: np.ndarray, f: np.ndarray) -> tuple[float, float]:
        y = np.r_[np.zeros(len(r)), np.ones(len(f))]
        return roc_auc(y, np.r_[r, f]), float(np.mean(f >= threshold))

    if not (np.all(np.isfinite(reals)) and np.all(np.isfinite(fakes))):
        raise ValueError("non-finite score")
    if paired and len(reals) != len(fakes):
        raise ValueError("paired bootstrap needs one fake per real")
    auc, tpr = stats(reals, fakes)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(RESAMPLES):
        if paired:
            index = rng.integers(0, len(reals), len(reals))
            draws.append(stats(reals[index], fakes[index]))
        else:
            draws.append(stats(reals[rng.integers(0, len(reals), len(reals))],
                               fakes[rng.integers(0, len(fakes), len(fakes))]))
    a = np.array(draws)
    lo, hi = np.percentile(a, [2.5, 97.5], axis=0)
    return {"n_real": int(len(reals)), "n_fake": int(len(fakes)),
            "auc": auc, "auc_ci": [float(lo[0]), float(hi[0])],
            "tpr_at_5pct_fpr": tpr, "tpr_ci": [float(lo[1]), float(hi[1])]}


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_cache(path: Path) -> tuple[list[str], np.ndarray]:
    """``(ids, features)`` of an S1 cache."""
    with np.load(path, allow_pickle=False) as blob:
        return [str(v) for v in blob["image_ids"]], blob["features"]


def figure(rows: list[dict[str, Any]], members: list[str], out: Path) -> list[Path]:
    """AUC per decoder: probe vs S2, chance line at 0.5. PNG and SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heads = [("probe_gen_only", "DINOv3-B gen-only probe (frozen)", "#4c72b0"),
             ("s2_p_a_sd1_seed0", "S2, P-A seed 0", "#dd8452"),
             ("s2_full_bank_seed0", "S2, full_bank seed 0", "#55a868")]
    labels = ["gen_matched\n(SD1.5 full gen)"] + members
    sets = ["gen_matched"] + [f"recon:{m}" for m in members]
    width = 0.27
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for k, (head, name, colour) in enumerate(heads):
        values, errors = [], [[], []]
        for test in sets:
            row = next(r for r in rows if r["test"] == test and r["head"] == head)
            values.append(row["auc"])
            errors[0].append(row["auc"] - row["auc_ci"][0])
            errors[1].append(row["auc_ci"][1] - row["auc"])
        x = np.arange(len(sets)) + (k - 1) * width
        ax.bar(x, values, width, yerr=errors, label=name, color=colour, capsize=2)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(np.arange(len(sets)), labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUC vs the 491 val reals (95% CI)")
    ax.set_title("X1 (EXPLORATORY): is frozen DINOv3-B blind to VAE-only images? "
                 "Probe trained on real COCO vs SD1.5 full generations only")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "x1_auc_by_decoder.png", out / "x1_auc_by_decoder.svg"]
    for path in paths:
        fig.savefig(path, dpi=150)
    plt.close(fig)
    return paths


def main() -> int:
    """Run X1. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--s1-list", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--s1-cache", required=True, type=Path)
    parser.add_argument("--s2-dir", required=True, type=Path)
    parser.add_argument("--panel-scores", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "real" / "scores")
    parser.add_argument("--validation-reports", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "validation")
    parser.add_argument("--panel-analysis", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "real" / "analysis")
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports" / "p2_x1")
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()
    log = setup_logging("x1_gen_only_probe")
    log.info("%s", LABEL)

    members = [s.key for s in load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
               .distinct_members()]
    s1_rows = read_csv(args.s1_list)
    group = {r["row_id"]: r["group_id"] for r in read_csv(args.splits)}
    split_group = {}
    for row in s1_rows:
        source = row["image_id"].rsplit("__", 1)[0] if "__" in row["image_id"] else row["image_id"]
        split_group[row["image_id"]] = group[source]
    split = build_split(s1_rows, split_group)
    log.info("fit %d, tune %d (C only), test: %d reals, %d gen_matched, %d decoders x %s",
             len(split["fit"]), len(split["tune"]), len(split["test_reals"]),
             len(split["test_gen"]), len(split["test_recon"]),
             sorted({len(v) for v in split["test_recon"].values()}))
    labels = {r["image_id"]: 0 if r["label"] == "real" else 1 for r in s1_rows}

    # ---- the probe ------------------------------------------------------------------
    ids, features = load_cache(args.s1_cache / CACHE.format("train"))
    cache = (ids, features)

    def x(target: list[str]) -> np.ndarray:
        return join_streams(target, {"s1": cache}, supersets=("s1",),
                            what="X1 features")["s1"].astype(np.float32)

    probe = fit_probe(x(split["fit"]), [labels[i] for i in split["fit"]],
                      x(split["tune"]), [labels[i] for i in split["tune"]], seed=0)
    log.info("probe: C=%s chosen on the train hold-out (AUC %.4f there); grid %s",
             probe["C"], probe["val_auc"], probe["grid"])
    model = probe["model"]

    def probe_scores(target: list[str], source: tuple[list[str], np.ndarray]) -> np.ndarray:
        joined = join_streams(target, {"s1": source}, supersets=("s1",), what="X1 scoring")
        return model.decision_function(joined["s1"].astype(np.float32))

    # ---- S2 on the val split ------------------------------------------------------------
    s2_val = {}
    for rid in S2_RUNS:
        rows = read_csv(args.s2_dir / rid / "scores" / "val.csv")
        s2_val[rid] = ([r["image_id"] for r in rows], np.array([float(r["score"]) for r in rows]))

    heads: dict[str, Any] = {"probe_gen_only": lambda t: probe_scores(t, cache)}
    for rid in S2_RUNS:
        heads[f"s2_{rid}"] = (lambda t, rid=rid: join_streams(
            t, {"s2": s2_val[rid]}, supersets=("s2",), what=f"S2 {rid}")["s2"])

    results: list[dict[str, Any]] = []
    thresholds: dict[str, dict[str, Any]] = {}
    reals = split["test_reals"]
    for head, scorer in heads.items():
        real_scores = scorer(reals)
        calibration = calibrate_on_reals(real_scores, TARGET_FPR, "x1_val_reals")
        thresholds[head] = calibration.to_dict()
        tests = [("gen_matched", reals, split["test_gen"], False)]
        for member, pairs in split["test_recon"].items():
            tests.append((f"recon:{member}", [p[0] for p in pairs], [p[1] for p in pairs], True))
        for name, real_ids, fake_ids, paired in tests:
            score_real = real_scores if real_ids == reals else scorer(real_ids)
            results.append({"test": name, "head": head,
                            **bootstrap(score_real, scorer(fake_ids),
                                        calibration.threshold, paired)})

    # ---- E7 on the panel ---------------------------------------------------------------
    panel_cache = load_cache(args.s1_cache / CACHE.format("panel"))
    e7_rows = [r for r in read_csv(args.panel_scores / f"scores_{S2_RUNS[0]}.csv")
               if r["view"] == "e7_dda_coco"]
    e7_ids = [r["image_id"] for r in e7_rows]
    e7: dict[str, np.ndarray] = {"probe_gen_only": probe_scores(e7_ids, panel_cache)}
    for rid in S2_RUNS:
        rows = read_csv(args.panel_scores / f"scores_{rid}.csv")
        table = ([r["image_id"] for r in rows], np.array([float(r["s2"]) for r in rows]))
        e7[f"s2_{rid}"] = join_streams(e7_ids, {"s2": table}, supersets=("s2",),
                                       what=f"E7 S2 {rid}")["s2"]
    subsets = sorted({r["subset"] for r in e7_rows} - {E7_REAL_SUBSET})
    is_real = np.array([r["subset"] == E7_REAL_SUBSET for r in e7_rows])
    if not all((r["label"] == "real") == (r["subset"] == E7_REAL_SUBSET) for r in e7_rows):
        raise SplitError("E7 subset 'real' and the label disagree")
    for head, values in e7.items():
        for subset in subsets:
            mask = np.array([r["subset"] == subset for r in e7_rows])
            results.append({"test": f"e7:{subset}", "head": head,
                            "in_bank_p_a": next(r["in_bank"] for r in e7_rows
                                                if r["subset"] == subset),
                            **bootstrap(values[is_real], values[mask],
                                        thresholds[head]["threshold"], False)})

    # ---- S1, extracted from committed reports (not recomputed) --------------------------
    s1_extracted: dict[str, Any] = {"note": "S1 was trained on reconstructions (and "
                                    "gen_matched), so it is NOT the clean test",
                                    "per_decoder_validation_auc": {}, "e7_panel": {}}
    for rid in S2_RUNS:
        path = args.validation_reports / f"p2_heads_{rid}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        s1_extracted["per_decoder_validation_auc"][rid] = {
            "source": path.relative_to(THESIS_DBP).as_posix(),
            "auc": {g: d["s1"] for g, d in report["per_group_val_auc_full_validation"].items()}}
        metrics = args.panel_analysis / rid / "metrics.csv"
        s1_extracted["e7_panel"][rid] = {
            "source": metrics.relative_to(THESIS_DBP).as_posix(),
            "by_slice": {r["slice"]: {"auc": r["auc"], "tpr_at_5pct_fpr": r["tpr_at_5pct_fpr"]}
                         for r in read_csv(metrics)
                         if r["stream"] == "s1" and r["slice"].startswith("e7_dda_coco/")}}

    # ---- write ------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"label": LABEL, "replicates": "Simplicity Prevails, arXiv 2602.01738, Table 9",
               "probe": {"C": probe["C"], "grid": probe["grid"], "c_chosen_on":
                         "seeded 20% COCO-group hold-out of the train split",
                         "n_fit": len(split["fit"]), "n_tune": len(split["tune"]),
                         "train_classes": "train-split real COCO vs train-split gen_matched "
                                          "(SD1.5 full generations); no reconstructions"},
               "thresholds_on_val_reals": thresholds,
               "bootstrap": {"resamples": RESAMPLES, "seed": SEED,
                             "scheme": "pairs for reconstructions; reals and fakes "
                                       "independently otherwise; threshold fixed"},
               "results": results, "s1_extracted": s1_extracted}
    path = args.out_dir / "x1_results.json"
    path.with_suffix(".tmp").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".tmp").replace(path)
    columns = ["test", "head", "n_real", "n_fake", "auc", "auc_lo", "auc_hi",
               "tpr_at_5pct_fpr", "tpr_lo", "tpr_hi"]
    table = args.out_dir / "x1_results.csv"
    with table.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["# " + LABEL])
        writer.writerow(columns)
        for r in results:
            writer.writerow([r["test"], r["head"], r["n_real"], r["n_fake"],
                             f"{r['auc']:.4f}", f"{r['auc_ci'][0]:.4f}", f"{r['auc_ci'][1]:.4f}",
                             f"{r['tpr_at_5pct_fpr']:.4f}", f"{r['tpr_ci'][0]:.4f}",
                             f"{r['tpr_ci'][1]:.4f}"])
    table.with_suffix(".tmp").replace(table)
    for figure_path in figure(results, members, args.figure_dir):
        log.info("figure: %s", figure_path)
    for r in results:
        log.info("%-34s %-20s AUC %.3f [%.3f, %.3f]  TPR@5 %.3f", r["test"], r["head"],
                 r["auc"], *r["auc_ci"], r["tpr_at_5pct_fpr"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
