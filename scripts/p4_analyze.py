"""P4 analysis: the frozen baseline spec's section 5, on the committed scores (descriptive).

Implements ``docs/P4_BASELINES_SPEC.md`` section 5 and nothing more in its main part.
There is no hypothesis, no decision rule and no PASS/FAIL: every value is a point
estimate with a paired bootstrap 95% CI. For every row of section 3:

1. **per slice** (every Panel B test slice: the 4 unseen, E7's 6 VAEs, E7 luma FLUX.1; E8
   fakes-only): AUC and AP with 95% CIs, and TPR at the row's k = 0 union threshold;
2. **per real pool** (E3, E4, E5, E7, E7 luma): achieved FPR at that threshold, with a CI;
3. the **macro AUC over the 4 unseen slices** (the P3 definition) and the **macro FPR over
   E4 and E5**, with CIs;
4. **paired differences against S2-D seed 0, S2-D seed 1 and P2 ``full_bank``** on their
   committed clean scores: macro AUC, each slice's AUC, macro FPR over E4/E5;
5. **DDA under the H5 conditions** (``jpeg50``, ``resize050``): macro AUC with CI, its
   difference from DDA clean, and the paired difference against S2-D seed 0 under the same
   condition (its committed robust scores).

**Bootstrap:** the P3 analysis's own ``bootstrap`` (``p3_analyze.py``), unchanged: 2,000
resamples, seed 2026, a fresh generator per statistic; per resample every real pool is
drawn once and shared by every model in the comparison, every fake slice within itself;
thresholds fixed; percentile 95%. Slices, pools, the macro, the union threshold and the
loaders are imported from it, not re-implemented.

**Labels:** E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d) in every row; E7
is DDA's home benchmark (spec section 6).

**Reference models** (``reference``): the same metrics for S2-D seeds 0 and 1, P2
``full_bank`` and S2-B seed 0 from their committed P3 final scores, so the thesis tables
compare like with like. They restate committed scores; nothing new is decided.

Readings of the frozen text where it leaves a detail open are the named constants in
``INTERPRETATIONS``, fixed in code before the analysis was first run. Anything beyond
section 5 goes under ``exploratory``; at this commit it is empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

import p3_analyze as p3  # noqa: E402
from dbp.eval.metrics import average_precision  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import write_record  # noqa: E402

REPORTS = THESIS_DBP / "reports"
LISTS = REPORTS / "kaggle_20261027" / "p3_final_session1" / "p3_final_lists"
P4_SCORES = REPORTS / "p4_baseline_scores"
P3_SCORES = REPORTS / "p3_final_scores"
OUT = REPORTS / "p4_results" / "p4_analysis.json"
BASELINES = ("dda_native", "dda_crop308", "cospy_native", "fire_native", "aeroblade")
H5_ROWS = {"jpeg50": "dda_jpeg50", "resize050": "dda_resize050"}
COMPARATORS = ("s2d_p35_seed0", "s2d_p35_seed1", "full_bank_seed0")
REFERENCE = (*COMPARATORS, "s2b_seed0")
E7 = tuple(("e7_dda_coco", v) for v in ("FLUX.1", "sd-vae-ft-ema", "sd-vae-ft-mse",
                                         "sdxl-vae", "stable-diffusion-2-1",
                                         "stable-diffusion-3.5-large"))
SLICES = (*p3.UNSEEN, *E7, (p3.LUMA, "FLUX.1"))
E8 = "e8_evalgen"
POOLS = ("e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline", "e7_dda_coco", p3.LUMA)
UPPER_BOUND = ("e7_dda_coco", "FLUX.1")
WORKERS = 8

#: Readings of the frozen text where it leaves a detail open. Fixed before the first run.
INTERPRETATIONS = {
    "rows_items_1_4": "items 1-4 apply to the five clean rows (dda_native, dda_crop308, "
                      "cospy_native, fire_native, aeroblade); the two H5 rows were scored on "
                      "Panel B test rows only (no luma, no threshold reals), so item 5 is "
                      "what is reported for them",
    "difference_sign": "a paired difference is ours minus the baseline (S2-D or full_bank "
                       "minus the row); positive means ours is higher",
    "tpr_point": "TPR at the union threshold is a point value: section 5 gives CIs for AUC, "
                 "AP, the FPRs and the macros, not for TPR (as P3's descriptive section)",
    "slices": "'every Panel B test slice' = the 4 unseen, E7's 6 VAE slices against the E7 "
              "test reals, E7 luma FLUX.1 against the E7 luma test reals; E8 has no reals, "
              "so only its TPR is reported",
    "macro_fpr": "the mean of the E4 and E5 achieved FPRs at the row's own union threshold",
    "h5_clean_reference": "DDA clean under item 5 is dda_native on the same Panel B test rows",
    "h5_versus_s2d": "S2-D seed 0 under the same condition minus DDA under it, paired",
    "bootstrap_calls": "every reported interval is one call of p3_analyze.bootstrap (fresh "
                       "generator, seed 2026); calls are independent, so they may run in "
                       "parallel processes without changing any value",
}


def load_p4(row: str) -> p3.Panel:
    """A clean P4 row: Panel B, luma and threshold reals, from ``reports/p4_baseline_scores``."""
    lists = {n: p3.read_rows(LISTS / f"{n}.csv") for n in ("panel_b", "luma", "threshold_reals")}
    scored = {n: p3.scores_for(P4_SCORES / row / f"{n}.csv", r) for n, r in lists.items()}
    return p3.build_panel(lists["panel_b"], scored["panel_b"], (lists["luma"], scored["luma"]),
                          (lists["threshold_reals"], scored["threshold_reals"]))


def load_p4_condition(row: str) -> p3.Panel:
    """A DDA H5 row: Panel B test rows under its condition."""
    rows = [r for r in p3.read_rows(LISTS / "panel_b.csv") if r["role"] == "test"]
    return p3.build_panel(rows, p3.scores_for(P4_SCORES / row / "panel_b_test.csv", rows))


def test_only(panel: p3.Panel) -> p3.Panel:
    """A clean panel restricted to Panel B's test rows (no luma view), for item 5."""
    return p3.Panel({v: r for v, r in panel.real.items() if v != p3.LUMA},
                    {k: f for k, f in panel.fake.items() if k[0] != p3.LUMA})


def ap(panel: p3.Panel, key: tuple[str, str]) -> float:
    """AP of one slice against its view's whole test real pool."""
    reals, fakes = panel.real[key[0]], panel.fake[key]
    return float(average_precision(np.r_[np.zeros(len(reals)), np.ones(len(fakes))],
                                   np.r_[reals, fakes]))


def macro_fpr(panel: p3.Panel, threshold: float) -> float:
    """Mean achieved FPR over E4 and E5."""
    return float(np.mean([p3.fpr(panel, v, threshold) for v in p3.FPR_POOLS]))


# One task = one bootstrap call; module-level so worker processes can run it.
PANELS: dict[str, p3.Panel] = {}
THRESHOLDS: dict[str, float] = {}


def statistic(kind: str, models: tuple[str, ...], key: Any) -> Any:
    """The statistic a task bootstraps (a picklable description, built in the worker)."""
    def one(name: str):
        if kind == "auc":
            return lambda q: p3.slice_auc(q[name], key)
        if kind == "ap":
            return lambda q: ap(q[name], key)
        if kind == "fpr":
            return lambda q: p3.fpr(q[name], key, THRESHOLDS[name])
        if kind == "macro_auc":
            return lambda q: p3.macro_auc(q[name])
        if kind == "macro_fpr":
            return lambda q: macro_fpr(q[name], THRESHOLDS[name])
        raise ValueError(kind)
    if len(models) == 1:
        return one(models[0])
    first, second = one(models[0]), one(models[1])
    return lambda q: first(q) - second(q)


def groups(kind: str, key: Any) -> list[Any]:
    """The resampling groups a statistic touches."""
    if kind in ("auc", "ap"):
        return p3.groups_of((key,))
    if kind == "fpr":
        return p3.groups_of((), (key,))
    if kind == "macro_auc":
        return p3.groups_of(p3.UNSEEN)
    return p3.groups_of((), p3.FPR_POOLS)


def run_task(task: tuple[str, str, tuple[str, ...], Any]) -> tuple[str, dict[str, Any]]:
    """One bootstrap call: (label, kind, models, key) -> (label, interval)."""
    label, kind, models, key = task
    panels = {m: PANELS[m] for m in models}
    return label, p3.bootstrap(panels, groups(kind, key), statistic(kind, models, key))


def init_worker(panels: dict[str, p3.Panel], thresholds: dict[str, float],
                resamples: int) -> None:
    """Give a worker process the panels, thresholds and resample count (a spawned worker
    re-imports this module, so nothing set in the parent reaches it otherwise)."""
    PANELS.update(panels)
    THRESHOLDS.update(thresholds)
    p3.RESAMPLES = resamples


def name(key: tuple[str, str]) -> str:
    """'view/slot'."""
    return f"{key[0]}/{key[1]}"


def tasks_for(rows: tuple[str, ...]) -> list[tuple[str, str, tuple[str, ...], Any]]:
    """Items 1-3 for each row, item 4 for each (comparator, baseline) pair."""
    tasks = []
    for row in rows:
        for key in SLICES:
            tasks += [(f"{row}|auc|{name(key)}", "auc", (row,), key),
                      (f"{row}|ap|{name(key)}", "ap", (row,), key)]
        tasks += [(f"{row}|fpr|{pool}", "fpr", (row,), pool) for pool in POOLS]
        tasks += [(f"{row}|macro_auc", "macro_auc", (row,), None),
                  (f"{row}|macro_fpr", "macro_fpr", (row,), None)]
    for ours in COMPARATORS:
        for row in BASELINES:
            pair = (ours, row)
            tasks += [(f"{ours}-{row}|macro_auc", "macro_auc", pair, None),
                      (f"{ours}-{row}|macro_fpr", "macro_fpr", pair, None)]
            tasks += [(f"{ours}-{row}|auc|{name(k)}", "auc", pair, k) for k in SLICES]
    return tasks


def h5_block(log: Any) -> dict[str, Any]:
    """Item 5: DDA under jpeg50 and resize050 (Panel B test rows), in-process."""
    clean = test_only(load_p4("dda_native"))
    out: dict[str, Any] = {}
    for condition, row in H5_ROWS.items():
        dda = load_p4_condition(row)
        s2d = p3.load_condition(LISTS, P3_SCORES, "s2d_p35_seed0", condition)
        g = p3.groups_of(p3.UNSEEN)
        out[condition] = {
            "row": row,
            "macro_auc": p3.bootstrap({"m": dda}, g, lambda q: p3.macro_auc(q["m"])),
            "minus_dda_clean": p3.bootstrap({"c": dda, "d": clean}, g, lambda q: p3.macro_auc(
                q["c"]) - p3.macro_auc(q["d"])),
            "s2d_seed0_macro_auc": p3.bootstrap({"m": s2d}, g, lambda q: p3.macro_auc(q["m"])),
            "s2d_seed0_minus_dda": p3.bootstrap({"s": s2d, "c": dda}, g, lambda q: p3.macro_auc(
                q["s"]) - p3.macro_auc(q["c"])),
        }
        log.info("H5 %s: DDA %.4f, minus clean %.4f, S2-D s0 minus DDA %.4f", condition,
                 out[condition]["macro_auc"]["point"], out[condition]["minus_dda_clean"]["point"],
                 out[condition]["s2d_seed0_minus_dda"]["point"])
    return out


def assemble(results: dict[str, dict[str, Any]], panels: dict[str, p3.Panel],
             thresholds: dict[str, float], row: str) -> dict[str, Any]:
    """One row's items 1-3 from the task results."""
    panel = panels[row]
    slices = {}
    for key in SLICES:
        slices[name(key)] = {
            "fakes": int(len(panel.fake[key])), "reals": int(len(panel.real[key[0]])),
            "auc": results[f"{row}|auc|{name(key)}"], "ap": results[f"{row}|ap|{name(key)}"],
            "tpr_k0_union": float(np.mean(panel.fake[key] >= thresholds[row])),
            **({"label": "upper bound (CLAUDE.md section 6d)"} if key == UPPER_BOUND else {})}
    for key in sorted(k for k in panel.fake if k[0] == E8):
        slices[name(key)] = {"fakes": int(len(panel.fake[key])), "reals": 0,
                             "tpr_k0_union": float(np.mean(panel.fake[key] >= thresholds[row]))}
    pools = {pool: {"reals": int(len(panel.real[pool])), "fpr_union": results[f"{row}|fpr|{pool}"]}
             for pool in POOLS}
    return {"threshold_union": thresholds[row], "slices": slices, "pools": pools,
            "macro_auc_unseen": results[f"{row}|macro_auc"],
            "macro_fpr_e4_e5": results[f"{row}|macro_fpr"], "status": "descriptive"}


def run(workers: int, log: Any) -> dict[str, Any]:
    """The whole section 5."""
    panels = {row: load_p4(row) for row in BASELINES}
    panels.update({m: p3.load_clean(LISTS, P3_SCORES, m) for m in REFERENCE})
    sizes = {m: ({v: len(r) for v, r in p.real.items()},
                 {name(k): len(f) for k, f in p.fake.items()}, len(p.union))
             for m, p in panels.items()}
    if len({json.dumps(s, sort_keys=True) for s in sizes.values()}) != 1:
        raise p3.AnalysisError("the models' real pools or fake slices differ in size")
    thresholds = {m: p3.k0_threshold(p, "union") for m, p in panels.items()}
    tasks = tasks_for((*BASELINES, *REFERENCE))
    log.info("%d bootstrap calls (%d resamples each) on %d worker(s)", len(tasks),
             p3.RESAMPLES, workers)
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(panels, thresholds, p3.RESAMPLES)) as pool:
        for done, (label, interval) in enumerate(pool.map(run_task, tasks, chunksize=4), 1):
            results[label] = interval
            if done % 50 == 0:
                log.info("%d/%d done", done, len(tasks))
    paired = {}
    for ours in COMPARATORS:
        for row in BASELINES:
            key = f"{ours}-{row}"
            paired[key] = {"minus": row, "from": ours, "status": "descriptive",
                           "macro_auc_unseen": results[f"{key}|macro_auc"],
                           "macro_fpr_e4_e5": results[f"{key}|macro_fpr"],
                           "slice_auc": {name(k): results[f"{key}|auc|{name(k)}"]
                                         for k in SLICES}}
    return {
        "what": "P4 baselines on Panel B: spec section 5, descriptive only",
        "spec": "docs/P4_BASELINES_SPEC.md (tag p4-baselines-frozen)",
        "bootstrap": {"resamples": p3.RESAMPLES, "seed": p3.SEED, "level": p3.SECONDARY,
                      "function": "p3_analyze.bootstrap (unchanged)"},
        "interpretations": INTERPRETATIONS,
        "labels": {"upper_bound": name(UPPER_BOUND) + " (CLAUDE.md section 6d)",
                   "home_benchmark": "E7 is DDA's home benchmark (DDA-COCO, built by DDA's "
                                     "authors); DDA trained on MSCOCO train, E7's reals are "
                                     "val2017 (spec section 6)"},
        "rows": {row: assemble(results, panels, thresholds, row) for row in BASELINES},
        "reference": {m: assemble(results, panels, thresholds, m) for m in REFERENCE},
        "paired": paired,
        "h5": h5_block(log),
        "exploratory": {"note": "nothing exploratory at this commit"},
    }


def main() -> int:
    """Run section 5 once; write ``reports/p4_results/p4_analysis.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--resamples", type=int, default=None,
                        help="Tests only: fewer resamples (the result is not the analysis).")
    args = parser.parse_args()
    log = setup_logging("p4_analyze")
    if args.resamples:
        p3.RESAMPLES = args.resamples
    try:
        record = run(args.workers, log)
    except (p3.AnalysisError, OSError, KeyError) as error:
        log.error("P4 ANALYSIS FAILED: %s", error)
        return 1
    record["bootstrap"]["resamples"] = p3.RESAMPLES
    write_record(args.out, record)
    for row in BASELINES:
        m = record["rows"][row]["macro_auc_unseen"]
        log.info("%-13s macro AUC %.4f [%.4f, %.4f]", row, m["point"], m["lower"], m["upper"])
    log.info("written %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
