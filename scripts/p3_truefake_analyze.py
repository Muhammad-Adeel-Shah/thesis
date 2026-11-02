"""P3 TrueFake: the descriptive analysis of the frozen rule (``P3_PLAN.md``, fixed 2026-10-20).

Secondary and **descriptive**: no decision rule, no hypothesis, no PASS/FAIL. For each of
the 4 models (``S2-B`` s0, ``S2-B+deg``, ``S2-D`` s0, P2 ``full_bank``):

* **AUC per generator slice** (its fakes against one real pool), for each of pre-share,
  Facebook, Telegram and Twitter, with **FFHQ and FORLAB reported separately**;
* **Delta = AUC(platform) - AUC(pre-share)**, by a **paired bootstrap**: 2,000 resamples, seed
  2026, the same ids in both versions, reals drawn within their pool, fakes within their
  slice, one draw per group per resample shared by every model and share; 95% percentile
  CIs;
* **macros reported separately**: diffusion (FLUX.1, SD1.5, SD2, SD3, SDXL: lineages in our
  bank) and StyleGAN 1-3 (an unseen generator family);
* **FORLAB's achieved FPR** at the model's committed k = 0 union threshold (5% on the 891
  threshold reals, ``calibrate_on_reals``), **descriptive only, never a claim**. FFHQ: AUC
  only, never an FPR.

**Exclusions:** an id dropped by the disjointness check, with any version too small for
the 320 crop, or with any version that cannot be decoded (a truncated source file), is
excluded from every version (pairing); each is counted. **Caveats** are
written into the output beside the numbers, including the Twitter one: ids whose Twitter
file is byte-identical to the pre-share file are counted per generator (only the fake side
is unchanged there: Delta is near zero, not exactly zero, because the reals were
re-encoded).

Inputs: the scoring list and the per-image scores from the reports zip (kept in
``kaggle_raw``, never committed), the 4 summaries, and the committed threshold-real scores.
Output: aggregates only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_analyze import (  # noqa: E402
    RESAMPLES,
    SECONDARY,
    SEED,
    TARGET_FPR,
    AnalysisError,
    auc,
    read_rows,
    scores_for,
)
from p3_truefake_prepare import scorable  # noqa: E402

MODELS = ("s2b_seed0", "s2b_deg_p35_seed0", "s2d_p35_seed0", "full_bank_seed0")
SHARES = ("PreSocial", "Facebook", "Telegram", "Twitter")
PLATFORMS = SHARES[1:]
POOLS = ("FFHQ", "FORLAB")
DIFFUSION = ("FLUX.1", "StableDiffusion1.5", "StableDiffusion2", "StableDiffusion3",
             "StableDiffusionXL")
STYLEGAN = ("StyleGAN", "StyleGAN2", "StyleGAN3")
GENERATORS = DIFFUSION + STYLEGAN
MACROS = {"diffusion": DIFFUSION, "stylegan": STYLEGAN}
CAVEATS = (
    "Secondary and descriptive: no decision rule, no hypothesis.",
    "Both TrueFake real pools are Flickr, the platform of our training reals, so neither "
    "supports an FPR claim; FORLAB's FPR is reported descriptively only.",
    "StyleGAN fakes are faces, so StyleGAN vs FFHQ is the content-matched comparison; any "
    "comparison against FORLAB also reflects content.",
    "The platforms downscale (720/1200 px wide, <= 800 px tall), so the fixed 320 centre "
    "crop covers a different extent of the scene pre- and post-share.",
    "FFHQ is non-commercial and FORLAB's terms are NOT-VERIFIED: no deployment claim "
    "(CLAUDE.md section 6c).",
    "Twitter left most StableDiffusion1.5 / StableDiffusion2 fakes byte-identical to their "
    "pre-share file (counted in twitter_identical): their Twitter Delta is near zero, not "
    "exactly zero, because the reals were re-encoded.",
)

Table = dict[str, dict[str, np.ndarray]]  # group -> share -> scores, ordered by id


def tables(rows: list[dict[str, str]], scores: dict[str, np.ndarray],
           ) -> tuple[dict[str, Table], dict[str, Any]]:
    """Per model: group -> share -> scores over the kept ids (same id order in every share)."""
    by_id: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_id.setdefault(row["id"], {})[row["share"]] = row
    excluded: dict[str, dict[str, int]] = {"dropped": {}, "too_small": {}, "unreadable": {}}
    kept: dict[str, list[str]] = {}
    for ident, versions in sorted(by_id.items()):
        group = next(iter(versions.values()))["group"]
        if set(versions) != set(SHARES):
            raise AnalysisError(f"{ident}: versions {sorted(versions)}, expected {SHARES}")
        if any(v["dropped"] == "1" for v in versions.values()):
            excluded["dropped"][group] = excluded["dropped"].get(group, 0) + 1
        elif any(v["too_small"] == "1" for v in versions.values()):
            excluded["too_small"][group] = excluded["too_small"].get(group, 0) + 1
        elif any(v["unreadable"] == "1" for v in versions.values()):
            excluded["unreadable"][group] = excluded["unreadable"].get(group, 0) + 1
        else:
            kept.setdefault(group, []).append(ident)
    missing = sorted(set(POOLS + GENERATORS) - set(kept))
    if missing:
        raise AnalysisError(f"no kept ids for {missing}")
    position = {r["image_id"]: k for k, r in enumerate(scored_order(rows))}
    out = {}
    for model, values in scores.items():
        out[model] = {g: {s: np.array([values[position[by_id[i][s]["image_id"]]]
                                       for i in ids]) for s in SHARES}
                      for g, ids in kept.items()}
    identical = {g: sum(1 for i in kept.get(g, []) if by_id[i]["Twitter"]["sha256"] ==
                        by_id[i]["PreSocial"]["sha256"]) for g in GENERATORS + POOLS}
    counts = {"kept_ids": {g: len(v) for g, v in sorted(kept.items())}, "excluded": excluded,
              "twitter_identical": identical}
    return out, counts


def scored_order(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows the scorer scored, in its order (``p3_truefake_score.scored_rows``)."""
    return [r for r in rows if scorable(r)]


def statistics(data: dict[str, Table], idx: dict[str, np.ndarray] | None = None
               ) -> dict[tuple[str, ...], float]:
    """Every AUC, Delta and macro, on the ids ``idx`` selects (None: all of them)."""
    out: dict[tuple[str, ...], float] = {}
    for model, table in data.items():
        def take(group: str, share: str, table: Table = table) -> np.ndarray:
            values = table[group][share]
            return values if idx is None else values[idx[group]]

        for share in SHARES:
            for pool in POOLS:
                reals = take(pool, share)
                for gen in GENERATORS:
                    out[(model, "auc", share, pool, gen)] = auc(reals, take(gen, share))
                for name, gens in MACROS.items():
                    out[(model, "auc", share, pool, name)] = float(np.mean(
                        [out[(model, "auc", share, pool, g)] for g in gens]))
        for share in PLATFORMS:
            for pool in POOLS:
                for item in GENERATORS + tuple(MACROS):
                    out[(model, "delta", share, pool, item)] = \
                        out[(model, "auc", share, pool, item)] - \
                        out[(model, "auc", "PreSocial", pool, item)]
    return out


def paired_bootstrap(data: dict[str, Table], resamples: int = RESAMPLES, seed: int = SEED,
                     level: float = SECONDARY) -> dict[tuple[str, ...], dict[str, float]]:
    """Point estimates and percentile CIs; the same id draw for every share and model."""
    point = statistics(data)
    first = next(iter(data.values()))
    sizes = {g: len(first[g]["PreSocial"]) for g in first}
    rng = np.random.default_rng(seed)
    keys = list(point)
    values = np.empty((resamples, len(keys)))
    for b in range(resamples):
        idx = {g: rng.integers(0, n, n) for g, n in sorted(sizes.items())}
        drawn = statistics(data, idx)
        values[b] = [drawn[k] for k in keys]
    if not np.all(np.isfinite(values)):
        raise AnalysisError("non-finite bootstrap replicate(s)")
    tail = (1.0 - level) / 2.0 * 100.0
    lower = np.percentile(values, tail, axis=0)
    upper = np.percentile(values, 100.0 - tail, axis=0)
    return {k: {"point": point[k], "lower": float(lo), "upper": float(hi)}
            for k, lo, hi in zip(keys, lower, upper, strict=True)}


def forlab_fpr(data: dict[str, Table], thresholds: dict[str, float]) -> dict[str, Any]:
    """FORLAB's achieved FPR per share at each model's k = 0 union threshold (descriptive)."""
    return {model: {"threshold": thresholds[model],
                    **{share: float(np.mean(table["FORLAB"][share] >= thresholds[model]))
                       for share in SHARES}}
            for model, table in data.items()}


def nest(results: dict[tuple[str, ...], dict[str, float]]) -> dict[str, Any]:
    """model -> auc|delta -> share -> pool -> item -> {point, lower, upper}."""
    out: dict[str, Any] = {}
    for (model, kind, share, pool, item), value in results.items():
        out.setdefault(model, {}).setdefault(kind, {}).setdefault(share, {}).setdefault(
            pool, {})[item] = {k: round(v, 6) for k, v in value.items()}
    return out


def run(list_path: Path, scores_root: Path, summaries: Path, threshold_list: Path,
        threshold_root: Path, models: tuple[str, ...] = MODELS,
        resamples: int = RESAMPLES) -> dict[str, Any]:
    """The whole descriptive analysis."""
    rows = read_rows(list_path)
    list_sha = file_sha256(list_path)
    order = scored_order(rows)
    scores, provenance = {}, {}
    for model in models:
        summary = json.loads((summaries / f"{model}.json").read_text(encoding="utf-8"))
        path = scores_root / model / "truefake.csv"
        digest = file_sha256(path)
        if summary.get("limit") is not None or summary["list_sha256"] != list_sha or \
                summary["lists"]["truefake"]["sha256"] != digest or \
                summary["lists"]["truefake"]["rows"] != len(order):
            raise AnalysisError(f"{model}: the summary does not describe this list and these "
                                f"scores")
        values = scores_for(path, order)
        scores[model] = values
        provenance[model] = {"scores_sha256": digest, "rows": len(values),
                             "lora_sha256": summary["lora_sha256"],
                             "head_sha256": summary["head_sha256"],
                             "snapshot_sha256": summary.get("snapshot_sha256")}
    data, counts = tables(rows, scores)
    t_rows = read_rows(threshold_list)
    thresholds = {m: calibrate_on_reals(
        scores_for(threshold_root / m / "clean" / "threshold_reals.csv", t_rows), TARGET_FPR,
        f"p3 union threshold reals ({m})").threshold for m in models}
    results = paired_bootstrap(data, resamples)
    return {
        "what": "P3 TrueFake, descriptive (P3_PLAN.md, rule fixed 2026-10-20)",
        "caveats": list(CAVEATS), "list_sha256": list_sha, "models": provenance,
        "bootstrap": {"resamples": resamples, "seed": SEED, "level": SECONDARY,
                      "paired": "one id draw per group per resample, shared by every share "
                                "and model; reals within their pool, fakes within their "
                                "slice"},
        "counts": counts,
        "macros": {k: list(v) for k, v in MACROS.items()},
        "forlab_fpr_at_k0_union_threshold": {
            "note": "descriptive only, never a claim (both pools are Flickr)",
            "target_fpr": TARGET_FPR, "threshold_reals": len(t_rows),
            "by_model": forlab_fpr(data, thresholds)},
        "results": nest(results),
    }


def main() -> int:
    """Run the analysis and write the aggregate JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", type=Path, required=True, help="p3_truefake_list.csv")
    parser.add_argument("--scores-root", type=Path, required=True,
                        help="The returned reports/p3_truefake_scores (kept in kaggle_raw).")
    parser.add_argument("--summaries", type=Path, required=True,
                        help="The returned reports/p3_truefake.")
    parser.add_argument("--threshold-list", type=Path, default=THESIS_DBP / "reports" /
                        "kaggle_20261027" / "p3_final_session1" / "p3_final_lists" /
                        "threshold_reals.csv")
    parser.add_argument("--threshold-root", type=Path,
                        default=THESIS_DBP / "reports" / "p3_final_scores")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log = setup_logging("p3_truefake_analyze")
    result = run(args.list, args.scores_root, args.summaries, args.threshold_list,
                 args.threshold_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(args.out)
    log.info("TrueFake analysis written: %s (kept ids %s)", args.out,
             result["counts"]["kept_ids"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
