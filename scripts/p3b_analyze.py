"""P3b analysis: ``docs/P3b_VITL_PREREG.md`` section 4 (frozen, tag ``p3b-vitl-prereg-frozen``).

A thin driver: every statistic is computed by the P3 and P4 analysis code, imported and
**unchanged** -- the loaders, slices, pools, the macro over the 4 unseen slices, the k = 0
union threshold, ``p3_analyze.bootstrap`` (2,000 resamples, seed 2026, real pools drawn once
per resample and shared by every model in a comparison, percentile 95%), P3's H3-k
``kshot_view``, P4's bootstrap tasks, labels and ``assemble``. Nothing is re-implemented and
no comparator is re-scored: DDA and CO-SPY are P4's committed scores, S2-D ViT-B is P3's.

**Per seed s** (ViT-L seed s against ViT-B seed s; every difference is ViT-L minus the
comparator), the questions and their pass rules (section 4):

- **Q1** (primary) vs DDA, **Q2** vs CO-SPY: paired macro AUC; *matches* if the 95% CI's
  lower bound > -2 points, *beats* if it is > 0;
- **Q3** vs ViT-B: a *gain* if the point is >= +2 points and the lower bound > 0;
- **Q4** the achieved FPR on the E4 and E5 test reals at the model's own k = 0 union
  threshold: macro <= 15% and neither pool > 25%;
- **Q5** H3-k at k = 50, methods (Q) and (S): >= 80% of draws at FPR <= 7.5% on each of E4
  and E5;
- **Q6** macro AUC under ``jpeg50`` and ``resize050`` within 5 points of clean (test rows).

**Verdicts across seeds:** PASS only if both seeds pass; "seed-dependent" if they disagree
(not passed); "not passed" if both fail; "not answered" if a seed was not scored (the
failure path, section 5). A failed "matches" is not evidence of being worse: only a CI
wholly below 0 shows a deficit (``reading``).

**Descriptive** (no rule; section 4): every slice's AUC and AP, TPR at the k = 0 threshold
(E8 included), each pool's FPR -- P4's ``assemble``; paired macro and slice AUC and macro
FPR (E4/E5, each model at its own threshold) against DDA, CO-SPY and ViT-B; under each H5
condition, ViT-L - DDA (P4's ``dda_jpeg50`` / ``dda_resize050``) and ViT-L - ViT-B where
ViT-B's robust scores exist (seed 0).

**Reproduction first** (``--reproduce``): S2-D ViT-B seeds 0 and 1 are put in ViT-L's
place, and every value the committed P3 and P4 analyses hold for them must come out
**exactly** equal (``reports/p3_results/p3_analysis_full.json``,
``reports/p4_results/p4_analysis.json``). Q3 is then ViT-B against itself: exactly 0.
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
import p4_analyze as p4  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import write_record  # noqa: E402

REPORTS = THESIS_DBP / "reports"
OUT = REPORTS / "p3b_results" / "p3b_analysis.json"
REPRODUCE_OUT = REPORTS / "p3b_results" / "p3b_reproduce.json"
P3_REFERENCE = REPORTS / "p3_results" / "p3_analysis_full.json"
P4_REFERENCE = REPORTS / "p4_results" / "p4_analysis.json"
SEEDS = ("0", "1")
DDA, COSPY = "dda_native", "cospy_native"
VITB = "s2d_p35_seed{}"
#: Section 4's rules, taken from the P3 constants they name.
MATCH_MARGIN = p3.H2_MARGIN        # -2 points (P3 H2)
GAIN = p3.H4_GAIN                  # +2 points (P3 H4)
READING = ("failing Q1's or Q2's 'matches' rule is not evidence that ViT-L is worse than the "
           "comparator: the margin could not be shown. Only a CI wholly below 0 shows a "
           "deficit (prereg section 4).")


def subject_name(seed: str) -> str:
    """The panel name of ViT-L seed ``seed`` (whatever model is put in its place)."""
    return f"vitl_seed{seed}"


def across_seeds(passes: dict[str, bool | None]) -> str:
    """Section 4's seed rule."""
    values = [passes.get(s) for s in SEEDS]
    if any(v is None for v in values):
        return "not answered"
    if all(values):
        return "PASS"
    return "seed-dependent" if any(values) else "not passed"


def paired_tasks(subject: str, comparator: str) -> list[tuple[str, str, tuple[str, ...], Any]]:
    """P4 item 4's tasks (labels as P4's) for one (subject, comparator) pair."""
    pair = (subject, comparator)
    key = f"{subject}-{comparator}"
    return [(f"{key}|macro_auc", "macro_auc", pair, None),
            (f"{key}|macro_fpr", "macro_fpr", pair, None),
            *[(f"{key}|auc|{p4.name(k)}", "auc", pair, k) for k in p4.SLICES]]


def paired_block(results: dict[str, Any], subject: str, comparator: str) -> dict[str, Any]:
    """One pair's results, shaped as P4's ``paired`` entries."""
    key = f"{subject}-{comparator}"
    return {"minus": comparator, "from": subject, "status": "descriptive",
            "macro_auc_unseen": results[f"{key}|macro_auc"],
            "macro_fpr_e4_e5": results[f"{key}|macro_fpr"],
            "slice_auc": {p4.name(k): results[f"{key}|auc|{p4.name(k)}"] for k in p4.SLICES}}


def q4(panel: p3.Panel, threshold: float) -> dict[str, Any]:
    """Q4: achieved FPR on E4 and E5 at the model's own union threshold (P3 H3's rule)."""
    pools = {v: p3.fpr(panel, v, threshold) for v in p3.FPR_POOLS}
    macro = float(np.mean(list(pools.values())))
    criteria = {"macro_fpr_le_15": macro <= p3.H3_MACRO,
                "no_pool_above_25": all(v <= p3.H3_POOL for v in pools.values())}
    return {"threshold_union": threshold, "fpr": pools, "macro_fpr": macro,
            "criteria": criteria, "passes": all(criteria.values())}


def q5(panel: p3.Panel, threshold: float) -> dict[str, Any]:
    """Q5: P3's H3-k per view; per method, k = 50 on E4 and E5 (P3 H3-k's rule)."""
    views = {v: p3.kshot_view(panel, v, panel.union, threshold) for v in p3.KSHOT}
    passes = {m: all(views[v][m]["50"]["share_fpr_le_7_5"] >= p3.KSHOT_SHARE
                     for v in p3.KSHOT_JUDGED) for m in ("Q", "S")}
    return {"views": views, "passes_at_k50": passes}


def macro_ci(panels: dict[str, p3.Panel], first: str, second: str | None = None) -> dict:
    """Macro AUC over the 4 unseen slices, or a paired difference, with its CI."""
    groups = p3.groups_of(p3.UNSEEN)
    if second is None:
        return p3.bootstrap({first: panels[first]}, groups, lambda q: p3.macro_auc(q[first]))
    pair = {first: panels[first], second: panels[second]}
    return p3.bootstrap(pair, groups,
                        lambda q: p3.macro_auc(q[first]) - p3.macro_auc(q[second]))


def robustness(seed: str, root: Path, model: str, log: Any) -> dict[str, Any]:
    """Q6 and the H5-condition descriptives for one seed, on Panel B test rows."""
    subject = subject_name(seed)
    vitb = VITB.format(seed)
    clean = p3.load_condition(p4.LISTS, root, model, "clean")
    out: dict[str, Any] = {"clean": {"macro_auc": macro_ci({subject: clean}, subject)},
                           "conditions": {}}
    decision = {}
    for condition in p3.H5_DECISION:
        panels = {subject: p3.load_condition(p4.LISTS, root, model, condition),
                  "clean": clean, DDA: p4.load_p4_condition(p4.H5_ROWS[condition])}
        block = {"macro_auc": macro_ci(panels, subject),
                 "minus_clean": macro_ci(panels, subject, "clean"),
                 "minus_dda": macro_ci(panels, subject, DDA), "dda_row": p4.H5_ROWS[condition]}
        if (p4.P3_SCORES / vitb / "robust" / f"{condition}.csv").is_file():
            panels[vitb] = p3.load_condition(p4.LISTS, p4.P3_SCORES, vitb, condition)
            block["minus_vitb"] = macro_ci(panels, subject, vitb)
        else:
            block["minus_vitb"] = {"status": "NOT-AVAILABLE",
                                   "reason": f"{vitb} has no committed {condition} scores"}
        out["conditions"][condition] = block
        point = block["macro_auc"]["point"]
        clean_point = out["clean"]["macro_auc"]["point"]
        decision[condition] = {"clean": clean_point, "condition": point,
                               "difference": point - clean_point,
                               "within_5_points": abs(point - clean_point) <= p3.H5_POINTS}
        log.info("seed %s %s: macro AUC %.4f (clean %.4f); minus DDA %.4f", seed, condition,
                 point, clean_point, block["minus_dda"]["point"])
    out["q6"] = {"decision": decision,
                 "passes": all(d["within_5_points"] for d in decision.values())}
    return out


def has_robust(root: Path, model: str) -> bool:
    """Whether the H5 decision conditions are scored for this model."""
    return all((root / model / "robust" / f"{c}.csv").is_file() for c in p3.H5_DECISION)


def run(subjects: dict[str, tuple[Path, str]], workers: int, log: Any) -> dict[str, Any]:
    """Section 4 for every scored seed. ``subjects``: seed -> (scores root, model)."""
    panels = {DDA: p4.load_p4(DDA), COSPY: p4.load_p4(COSPY)}
    for seed, (root, model) in subjects.items():
        panels[subject_name(seed)] = p3.load_clean(p4.LISTS, root, model)
        panels[VITB.format(seed)] = p3.load_clean(p4.LISTS, p4.P3_SCORES, VITB.format(seed))
    sizes = {m: ({v: len(r) for v, r in p.real.items()},
                 {p4.name(k): len(f) for k, f in p.fake.items()}, len(p.union))
             for m, p in panels.items()}
    if len({json.dumps(s, sort_keys=True) for s in sizes.values()}) != 1:
        raise p3.AnalysisError("the models' real pools or fake slices differ in size")
    thresholds = {m: p3.k0_threshold(p, "union") for m, p in panels.items()}
    tasks = []
    for seed in subjects:
        subject = subject_name(seed)
        tasks += [t for t in p4.tasks_for((subject,)) if t[0].startswith(f"{subject}|")]
        for comparator in (DDA, COSPY, VITB.format(seed)):
            tasks += paired_tasks(subject, comparator)
    log.info("%d bootstrap calls (%d resamples each) on %d worker(s)", len(tasks),
             p3.RESAMPLES, workers)
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=p4.init_worker,
                             initargs=(panels, thresholds, p3.RESAMPLES)) as pool:
        for done, (label, interval) in enumerate(pool.map(p4.run_task, tasks, chunksize=4), 1):
            results[label] = interval
            if done % 50 == 0:
                log.info("%d/%d done", done, len(tasks))

    seeds: dict[str, Any] = {}
    for seed, (root, model) in subjects.items():
        subject, vitb = subject_name(seed), VITB.format(seed)
        block: dict[str, Any] = {
            "scores": f"{root.as_posix()}/{model}",
            "descriptive": p4.assemble(results, panels, thresholds, subject),
            "paired": {c: paired_block(results, subject, c) for c in (DDA, COSPY, vitb)},
            "q4": q4(panels[subject], thresholds[subject]),
            "q5": q5(panels[subject], thresholds[subject]),
        }
        block["robust"] = robustness(seed, root, model, log) if has_robust(root, model) else {
            "status": "NOT-AVAILABLE", "reason": f"{model} has no jpeg50/resize050 scores"}
        seeds[seed] = block

    def lower(seed: str, comparator: str) -> float:
        return seeds[seed]["paired"][comparator]["macro_auc_unseen"]["lower"]

    def rule(fn) -> dict[str, Any]:  # noqa: ANN001
        passes = {s: (fn(s) if s in seeds else None) for s in SEEDS}
        return {"per_seed": passes, "verdict": across_seeds(passes)}

    def q3_gain(seed: str) -> bool:
        ci = seeds[seed]["paired"][VITB.format(seed)]["macro_auc_unseen"]
        return ci["point"] >= GAIN and ci["lower"] > 0

    def q6_pass(seed: str) -> bool | None:
        robust = seeds[seed]["robust"]
        return robust["q6"]["passes"] if "q6" in robust else None

    questions = {
        "Q1": {"role": "primary", "comparator": DDA,
               "matches": rule(lambda s: lower(s, DDA) > MATCH_MARGIN),
               "beats": rule(lambda s: lower(s, DDA) > 0)},
        "Q2": {"role": "secondary", "comparator": COSPY,
               "matches": rule(lambda s: lower(s, COSPY) > MATCH_MARGIN),
               "beats": rule(lambda s: lower(s, COSPY) > 0)},
        "Q3": {"role": "secondary", "comparator": "S2-D ViT-B, same seed",
               "gain": rule(q3_gain)},
        "Q4": {"role": "secondary", **rule(lambda s: seeds[s]["q4"]["passes"])},
        "Q5": {"role": "secondary",
               **{m: rule(lambda s, m=m: seeds[s]["q5"]["passes_at_k50"][m])
                  for m in ("Q", "S")}},
        "Q6": {"role": "secondary", **rule(q6_pass)},
    }
    return {
        "what": "P3b ViT-L analysis: docs/P3b_VITL_PREREG.md section 4",
        "prereg": "docs/P3b_VITL_PREREG.md (tag p3b-vitl-prereg-frozen)",
        "bootstrap": {"resamples": p3.RESAMPLES, "seed": p3.SEED, "level": p3.SECONDARY,
                      "function": "p3_analyze.bootstrap (unchanged)"},
        "reading": READING,
        "limitation": "DINOv3 ViT-L/16 against DDA's DINOv2 ViT-L/14: the same size, a "
                      "different pretraining and patch size (prereg section 3)",
        "environment": "ViT-L trained and scored on torch 2.11.0+cu130, as were the P4 "
                       "comparators DDA and CO-SPY (MEASURED); the S2-D ViT-B models and their "
                       "P3 scores on a torch version their records do not hold (2.10.0+cu128 "
                       "is the only earlier Kaggle record: NOT-VERIFIED for them). Recorded, "
                       "not expected to matter; it bears on Q3 only (prereg section 7)",
        "labels": {"upper_bound": p4.name(p4.UPPER_BOUND) + " (CLAUDE.md section 6d)",
                   "home_benchmark": "E7 is DDA's home benchmark"},
        "seeds_scored": sorted(subjects),
        "questions": questions,
        "seeds": seeds,
        "exploratory": {"note": "nothing exploratory at this commit"},
    }


# ------------------------------------------------------------------ reproduction ---


def differences(ours: Any, theirs: Any, path: str = "") -> list[str]:
    """Every place two JSON values differ (after a JSON round trip of ours)."""
    if isinstance(theirs, dict) and isinstance(ours, dict):
        out = [f"{path}/{k}: missing" for k in theirs if k not in ours]
        out += [f"{path}/{k}: extra" for k in ours if k not in theirs]
        for k in theirs:
            if k in ours:
                out += differences(ours[k], theirs[k], f"{path}/{k}")
        return out
    if isinstance(theirs, list) and isinstance(ours, list) and len(ours) == len(theirs):
        out = []
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            out += differences(a, b, f"{path}[{i}]")
        return out
    return [] if ours == theirs else [f"{path}: {ours!r} != {theirs!r}"]


def reproduce(record: dict[str, Any]) -> dict[str, Any]:
    """Compare a run on S2-D ViT-B with the committed P3 and P4 values for it."""
    ours = json.loads(json.dumps(record))
    p3_ref = json.loads(P3_REFERENCE.read_text(encoding="utf-8"))["confirmatory"]
    p4_ref = json.loads(P4_REFERENCE.read_text(encoding="utf-8"))
    pairs: list[tuple[str, Any, Any]] = []
    for seed in SEEDS:
        model, block = VITB.format(seed), ours["seeds"][seed]
        pairs.append((f"seed {seed} descriptive = P4 reference/{model}",
                      block["descriptive"], p4_ref["reference"][model]))
        for comparator in (DDA, COSPY):
            mine = {k: v for k, v in block["paired"][comparator].items() if k != "from"}
            theirs = {k: v for k, v in p4_ref["paired"][f"{model}-{comparator}"].items()
                      if k != "from"}
            pairs.append((f"seed {seed} paired vs {comparator} (Q1/Q2, delta FPR) = P4 "
                          f"paired/{model}-{comparator}", mine, theirs))
        h3 = p3_ref["H3"]["seeds"][model]
        pairs.append((f"seed {seed} Q4 = P3 H3/{model} (threshold, FPR, macro)",
                      {k: block["q4"][k] for k in ("threshold_union", "fpr", "macro_fpr")},
                      {k: h3[k] for k in ("threshold_union", "fpr", "macro_fpr")}))
        pairs.append((f"seed {seed} Q5 = P3 H3k/{model}", block["q5"],
                      p3_ref["H3k"]["seeds"][model]))
        self_pair = block["paired"][model]["macro_auc_unseen"]
        pairs.append((f"seed {seed} Q3 against itself is exactly 0",
                      [self_pair["point"], self_pair["lower"], self_pair["upper"]],
                      [0.0, 0.0, 0.0]))
    robust = ours["seeds"]["0"]["robust"]
    model = VITB.format("0")
    pairs.append(("seed 0 Q6 = P3 H5 decision", robust["q6"]["decision"],
                  p3_ref["H5"]["decision"]))
    pairs.append(("seed 0 clean macro AUC (test rows) = P3 H5 curve",
                  robust["clean"]["macro_auc"], p3_ref["H5"]["curves"][model]["clean"]))
    for condition in p3.H5_DECISION:
        block = robust["conditions"][condition]
        pairs.append((f"seed 0 {condition} macro AUC = P3 H5 curve", block["macro_auc"],
                      p3_ref["H5"]["curves"][model][condition]))
        pairs.append((f"seed 0 {condition} macro AUC = P4 h5 s2d_seed0_macro_auc",
                      block["macro_auc"], p4_ref["h5"][condition]["s2d_seed0_macro_auc"]))
        pairs.append((f"seed 0 {condition} minus DDA = P4 h5 s2d_seed0_minus_dda",
                      block["minus_dda"], p4_ref["h5"][condition]["s2d_seed0_minus_dda"]))
    checks = [{"check": label, "differences": differences(mine, theirs)}
              for label, mine, theirs in pairs]
    return {"what": "p3b_analyze.py on S2-D ViT-B against the committed P3 and P4 values",
            "references": {str(p.relative_to(THESIS_DBP).as_posix()): file_sha256(p)
                           for p in (P3_REFERENCE, P4_REFERENCE)},
            "checks": checks, "checked": len(checks),
            "exact_match": all(not c["differences"] for c in checks)}


def parse_subject(item: str) -> tuple[str, tuple[Path, str]]:
    """``SEED=DIR``: the model's score directory (``<root>/<model>/clean/...``)."""
    seed, sep, path = item.partition("=")
    if not sep or seed not in SEEDS:
        raise ValueError(f"--subject expects SEED=DIR with SEED in {SEEDS}, got {item!r}")
    directory = Path(path)
    return seed, (directory.parent, directory.name)


def main() -> int:
    """Run section 4 once (or the reproduction); write the record atomically."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--subject", action="append", default=[], metavar="SEED=DIR",
                        help="A frozen ViT-L model's score directory, per scored seed.")
    parser.add_argument("--reproduce", action="store_true",
                        help="Put S2-D ViT-B in ViT-L's place; compare with P3 and P4.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=p4.WORKERS)
    parser.add_argument("--resamples", type=int, default=None,
                        help="Tests only: fewer resamples (the result is not the analysis).")
    args = parser.parse_args()
    log = setup_logging("p3b_analyze")
    if args.resamples:
        p3.RESAMPLES = args.resamples
    try:
        if args.reproduce:
            if args.subject:
                raise ValueError("--reproduce takes no --subject: it uses S2-D ViT-B")
            subjects = {s: (p4.P3_SCORES, VITB.format(s)) for s in SEEDS}
        else:
            subjects = dict(parse_subject(item) for item in args.subject)
            if not subjects:
                raise ValueError("no --subject: name each scored ViT-L seed's score directory")
        record = run(subjects, args.workers, log)
        if args.reproduce:
            record = {"reproduction": reproduce(record), "run": record}
    except (p3.AnalysisError, OSError, KeyError, ValueError) as error:
        log.error("P3b ANALYSIS FAILED: %s", error)
        return 1
    out = args.out or (REPRODUCE_OUT if args.reproduce else OUT)
    write_record(out, record)
    if args.reproduce:
        result = record["reproduction"]
        for check in result["checks"]:
            log.info("%s  %s", "MATCH" if not check["differences"] else "DIFFER",
                     check["check"])
            for line in check["differences"][:5]:
                log.info("    %s", line)
        log.info("REPRODUCTION: %s (%d checks); written %s",
                 "EXACT MATCH" if result["exact_match"] else "MISMATCH", result["checked"], out)
        return 0 if result["exact_match"] else 1
    for name, question in record["questions"].items():
        log.info("%s: %s", name, {k: v["verdict"] for k, v in question.items()
                                  if isinstance(v, dict) and "verdict" in v}
                 or question.get("verdict"))
    log.info("written %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
