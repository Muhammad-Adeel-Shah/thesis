"""P3 thesis results: tables and figures from the committed results (no new decision).

Reads the committed confirmatory outputs -- ``reports/p3_h1a_ci.json``,
``reports/p3_h1c.json`` and ``reports/p3_results/p3_analysis_full.json`` -- and writes:

* ``reports/p3_results/tables/*.md`` and ``*.csv``:
  - ``t1_verdicts``: every hypothesis, its statistic with CI, criterion and verdict;
  - ``t2_h3_three_step``: the H3 three-step table with TPR beside every FPR;
  - ``t3_h3k_by_k``: H3-k mean FPR, mean TPR and share of draws by k;
  - ``t4_slice_tpr`` (descriptive): per slice AUC, AP, TPR at the k = 0 union threshold
    and slice-calibrated TPR@5%FPR, for every model (CSV) and the main models (Markdown);
  - ``t5_h4_curve``, ``t6_h5_curves`` and ``t7_h3k_ece``;
  - ``t8a_truefake_auc`` and ``t8b_truefake_forlab_fpr`` (descriptive), from
    ``reports/p3_results/p3_truefake_analysis.json``.
* ``reports/p3_results/figures/*.png`` and ``*.svg`` (charts only, <= 1 MB each, CLAUDE.md
  section 2b): H3-k by k, H4 bank size beside the held-out-lineage validation curve, H5
  robustness curves.

**Every row and figure says whether it is a pre-registered decision or descriptive.**

The only values not read from the JSON are the descriptive TPRs of the three-step table
at the COCO-only threshold, which the analysis did not store: they are computed here from
the committed Panel B scores with the analysis's own loader and the committed thresholds,
after the same code reproduces the stored union-threshold FPRs and TPRs exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.utils.logging import setup_logging  # noqa: E402

DECISION, DESCRIPTIVE = "decision (pre-registered)", "descriptive"
REPORTS = THESIS_DBP / "reports"
SOURCES = {"h1a": REPORTS / "p3_h1a_ci.json", "h1c": REPORTS / "p3_h1c.json",
           "full": REPORTS / "p3_results" / "p3_analysis_full.json",
           "truefake": REPORTS / "p3_results" / "p3_truefake_analysis.json"}
PLATFORMS = ("Facebook", "Telegram", "Twitter")
MACRO_LABEL = {"diffusion": "macro: diffusion", "stylegan": "macro: StyleGAN 1-3"}
LISTS = REPORTS / "kaggle_20261027" / "p3_final_session1" / "p3_final_lists"
SCORES = REPORTS / "p3_final_scores"
VIEW = {"e3_matched": "E3", "e4_wildrf_matched": "E4", "e5_chameleon_pipeline": "E5",
        "e7_dda_coco": "E7", "e7_dda_coco_luma": "E7 luma", "e8_evalgen": "E8"}
LABEL = {"s2d_p35_seed0": "S2-D s0", "s2d_p35_seed1": "S2-D s1",
         "full_bank_seed0": "P2 full_bank", "p_a_sd1_seed0": "P2 P-A",
         "s2b_seed0": "S2-B s0", "s2b_seed1": "S2-B s1", "s2b_deg_p35_seed0": "S2-B+deg",
         "s2b_reals_seed0": "S2-B+reals"}
MAIN = ("s2d_p35_seed0", "s2d_p35_seed1", "full_bank_seed0", "s2b_seed0",
        "s2b_deg_p35_seed0", "s2b_reals_seed0")
H5_FAMILIES = (("JPEG quality", ("jpeg95", "jpeg75", "jpeg50", "jpeg30"), "clean"),
               ("WebP quality", ("webp90", "webp70", "webp50"), "clean"),
               ("resize-and-restore scale", ("resize075", "resize050"), "1.0"),
               ("Gaussian blur sigma", ("blur05", "blur1", "blur2"), "0"))
TICK = {"jpeg95": "95", "jpeg75": "75", "jpeg50": "50", "jpeg30": "30", "webp90": "90",
        "webp70": "70", "webp50": "50", "resize075": "0.75", "resize050": "0.5",
        "blur05": "0.5", "blur1": "1", "blur2": "2"}
#: The reference palette's first three categorical slots (validated all-pairs, both modes).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


class FigureError(ValueError):
    """Committed results that do not reproduce or are missing."""


def ci(value: dict[str, Any], digits: int = 4) -> str:
    """'point [lower, upper]'."""
    return (f"{value['point']:.{digits}f} [{value['lower']:.{digits}f}, "
            f"{value['upper']:.{digits}f}]")


def pct(x: float | None) -> str:
    """A rate as a percentage with one decimal."""
    return "n/a" if x is None else f"{100 * x:.1f}%"


# ---------------------------------------------------------------------- tables ---


def verdict_rows(h1a: dict, h1c: dict, full: dict) -> list[dict[str, str]]:
    """Table 1: every hypothesis, as decided."""
    c = full["confirmatory"]
    rows = [{"hypothesis": "H1-a", "role": "primary", "status": DECISION,
             "statistic": "G, mean held-out-lineage AUC over 8 LOLO folds",
             "estimate": f"{h1a['G']:.4f} [{h1a['ci']['lower']:.4f}, {h1a['ci']['upper']:.4f}]",
             "level": f"{h1a['ci']['level']}",
             "criterion": "G >= 0.90, CI lower >= 0.85, no fold < 0.75",
             "verdict": h1a["verdict"], "source": "reports/p3_h1a_ci.json"}]
    for fold, entry in sorted(c["H1b"]["folds"].items()):
        for subset, value in entry["subsets"].items():
            rows.append({"hypothesis": f"H1-b fold {fold}", "role": "primary",
                         "status": DECISION, "statistic": f"AUC, {subset}",
                         "estimate": ci(value), "level": f"{value['level']}",
                         "criterion": "each subset >= 0.85 (point)",
                         "verdict": entry["verdict"], "source": "p3_analysis_full.json"})
        for subset, value in entry.get("upper_bound", {}).items():
            rows.append({"hypothesis": f"H1-b fold {fold}", "role": "primary",
                         "status": DESCRIPTIVE, "statistic": f"AUC, {subset} (upper bound, "
                                                             f"section 6d shortcut)",
                         "estimate": ci(value), "level": f"{value['level']}",
                         "criterion": "not judged", "verdict": "-",
                         "source": "p3_analysis_full.json"})
    rows.append({"hypothesis": "H1-b", "role": "primary", "status": DECISION,
                 "statistic": "folds passing", "estimate": f"{c['H1b']['folds_passing']} / 3",
                 "level": "-", "criterion": ">= 2 of 3 folds (claim, with H1-a)",
                 "verdict": "PASS" if c["H1b"]["h1b_on_two_of_three"] else "FAIL",
                 "source": "p3_analysis_full.json"})
    rows.append({"hypothesis": "H1-c", "role": "secondary", "status": DECISION,
                 "statistic": "mean (LOLO-f - BANK-1) AUC, folds 2-8",
                 "estimate": f"{h1c['mean_difference']:.4f} [{h1c['ci']['lower']:.4f}, "
                             f"{h1c['ci']['upper']:.4f}]", "level": f"{h1c['ci']['level']}",
                 "criterion": ">= +0.02 and CI excludes 0", "verdict": h1c["verdict"],
                 "source": "reports/p3_h1c.json"})
    for seed, s in c["H3"]["seeds"].items():
        rows.append({"hypothesis": "H3", "role": "primary", "status": DECISION,
                     "statistic": f"{LABEL[seed]}: macro FPR E4/E5 at the k = 0 union "
                                  f"threshold (E4 {pct(s['fpr']['e4_wildrf_matched'])}, "
                                  f"E5 {pct(s['fpr']['e5_chameleon_pipeline'])})",
                     "estimate": pct(s["macro_fpr"]), "level": "-",
                     "criterion": "<= 15%; no pool > 25%",
                     "verdict": "met" if s["criteria"]["macro_fpr_le_15"] and
                     s["criteria"]["no_pool_above_25"] else "not met",
                     "source": "p3_analysis_full.json"})
        rows.append({"hypothesis": "H3", "role": "primary", "status": DECISION,
                     "statistic": f"{LABEL[seed]}: relative FPR reduction vs P2 full_bank "
                                  f"({pct(s['full_bank_macro_fpr'])})",
                     "estimate": ci(s["relative_reduction"]),
                     "level": f"{s['relative_reduction']['level']}",
                     "criterion": ">= 0.50 and CI excludes 0",
                     "verdict": "met" if s["criteria"]["reduction_ge_50"] and
                     s["criteria"]["reduction_ci_excludes_0"] else "not met",
                     "source": "p3_analysis_full.json"})
    rows.append({"hypothesis": "H3", "role": "primary", "status": DECISION,
                 "statistic": "all criteria, both seeds", "estimate": "-", "level": "-",
                 "criterion": "all of the above", "verdict": c["H3"]["verdict"],
                 "source": "p3_analysis_full.json"})
    for seed, s in c["H3k"]["seeds"].items():
        for method in ("Q", "S"):
            shares = [f"{VIEW[p]} {s['views'][p][method]['50']['share_fpr_le_7_5']:.3f}"
                      for p in ("e4_wildrf_matched", "e5_chameleon_pipeline")]
            rows.append({"hypothesis": f"H3-k ({method})", "role": "secondary",
                         "status": DECISION,
                         "statistic": f"{LABEL[seed]}: share of k = 50 draws with FPR <= 7.5%",
                         "estimate": ", ".join(shares), "level": "-",
                         "criterion": ">= 0.80 on E4 and on E5, both seeds",
                         "verdict": "met" if s["passes_at_k50"][method] else "not met",
                         "source": "p3_analysis_full.json"})
    for method, verdict in c["H3k"]["verdict"].items():
        rows.append({"hypothesis": f"H3-k ({method})", "role": "secondary",
                     "status": DECISION, "statistic": "both seeds", "estimate": "-",
                     "level": "-", "criterion": "both seeds", "verdict": verdict,
                     "source": "p3_analysis_full.json"})
    for key, name, status in (("four_unseen_slices", "4 unseen slices", DECISION),
                              ("without_e3", "without E3 (E4, E5)",
                               "pre-registered companion test, reported beside the "
                               "decision (section 6.4)")):
        for seed, value in c["H2"][key]["seeds"].items():
            rows.append({"hypothesis": "H2", "role": "secondary", "status": status,
                         "statistic": f"{LABEL[seed]} - P2 full_bank macro AUC, {name}",
                         "estimate": ci(value), "level": f"{value['level']}",
                         "criterion": "CI lower > -0.02 (non-inferior), both seeds",
                         "verdict": "non-inferior" if value["non_inferior"] else "not shown",
                         "source": "p3_analysis_full.json"})
        rows.append({"hypothesis": "H2", "role": "secondary", "status": status,
                     "statistic": f"both seeds, {name}", "estimate": "-", "level": "-",
                     "criterion": "both seeds", "verdict": c["H2"][key]["verdict"],
                     "source": "p3_analysis_full.json"})
    gain = c["H4"]["bank8_minus_bank1"]
    rows.append({"hypothesis": "H4", "role": "secondary", "status": DECISION,
                 "statistic": "BANK-8 - BANK-1 macro AUC, 4 unseen slices",
                 "estimate": ci(gain), "level": f"{gain['level']}",
                 "criterion": ">= +0.02 and CI excludes 0", "verdict": c["H4"]["verdict"],
                 "source": "p3_analysis_full.json"})
    curves = c["H5"]["curves"][c["H5"]["model"]]
    for condition, entry in c["H5"]["decision"].items():
        rows.append({"hypothesis": "H5", "role": "secondary", "status": DECISION,
                     "statistic": f"S2-D s0 macro AUC, {condition} (clean "
                                  f"{entry['clean']:.4f}; difference "
                                  f"{entry['difference']:+.4f})",
                     "estimate": ci(curves[condition]),
                     "level": f"{curves[condition]['level']}",
                     "criterion": "|condition - clean| <= 0.05 (point)",
                     "verdict": "within" if entry["within_5_points"] else "outside",
                     "source": "p3_analysis_full.json"})
    rows.append({"hypothesis": "H5", "role": "secondary", "status": DECISION,
                 "statistic": "both conditions", "estimate": "-", "level": "-",
                 "criterion": "both within 5 points", "verdict": c["H5"]["verdict"],
                 "source": "p3_analysis_full.json"})
    return rows


def three_step_rows(full: dict, coco_tpr: dict[str, dict[str, float]]
                    ) -> list[dict[str, str]]:
    """Table 2: H3 three-step, FPR with TPR beside it (union TPRs from the JSON)."""
    desc = full["descriptive"]
    rows = []
    for seed in ("s2d_p35_seed0", "s2d_p35_seed1"):
        steps = full["confirmatory"]["H3"]["seeds"][seed]["three_step"]
        for pool in ("e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline"):
            s = steps[pool]
            rows.append({
                "seed": LABEL[seed], "pool": VIEW[pool] + (" (descriptive)"
                                                           if pool == "e3_matched" else ""),
                "full_bank_coco_only_fpr": pct(s["full_bank_coco_only"]),
                "full_bank_coco_only_tpr": pct(coco_tpr["full_bank_seed0"][pool]),
                "full_bank_union_fpr": pct(s["full_bank_union"]),
                "full_bank_union_tpr": pct(pool_tpr(desc["full_bank_seed0"], pool)),
                "s2d_union_fpr": pct(s["s2d_union"]),
                "s2d_union_tpr": pct(pool_tpr(desc[seed], pool)),
                "status": (DECISION + " (S2-D union FPR, via the macro)")
                if pool != "e3_matched" else DESCRIPTIVE})
        m = steps["macro_e4_e5"]
        macro = lambda model, rule: float(np.mean(  # noqa: E731
            [rule(model, p) for p in ("e4_wildrf_matched", "e5_chameleon_pipeline")]))
        rows.append({
            "seed": LABEL[seed], "pool": "macro E4/E5",
            "full_bank_coco_only_fpr": pct(m["full_bank_coco_only"]),
            "full_bank_coco_only_tpr": pct(macro("full_bank_seed0",
                                                 lambda mo, p: coco_tpr[mo][p])),
            "full_bank_union_fpr": pct(m["full_bank_union"]),
            "full_bank_union_tpr": pct(macro("full_bank_seed0",
                                             lambda mo, p: pool_tpr(desc[mo], p))),
            "s2d_union_fpr": pct(m["s2d_union"]),
            "s2d_union_tpr": pct(macro(seed, lambda mo, p: pool_tpr(desc[mo], p))),
            "status": DECISION + " (S2-D macro FPR); TPR descriptive"})
    return rows


def pool_tpr(model_desc: dict, pool: str) -> float:
    """TPR at the k = 0 union threshold over a pool's test fakes (all its slices)."""
    slices = [v for k, v in model_desc["slices"].items() if k.split("/")[0] == pool]
    return sum(v["tpr_k0_union"] * v["fakes"] for v in slices) / sum(v["fakes"] for v in slices)


def h3k_rows(full: dict) -> list[dict[str, str]]:
    """Table 3: H3-k by k, both methods, both seeds."""
    rows = []
    for seed, s in full["confirmatory"]["H3k"]["seeds"].items():
        for pool, v in s["views"].items():
            judged = pool in ("e4_wildrf_matched", "e5_chameleon_pipeline")
            rows.append({"seed": LABEL[seed], "pool": VIEW[pool], "method": "k = 0",
                         "k": "0", "mean_fpr": pct(v["k0"]["fpr"]),
                         "mean_tpr": pct(v["k0"]["tpr"]), "share_fpr_le_7_5": "-",
                         "draws": "-", "status": DESCRIPTIVE})
            for method in ("Q", "S"):
                for k, cell in v[method].items():
                    single = cell["draws"] == 1
                    share = (f"single draw: FPR {pct(cell['single_draw_fpr']['point'])} "
                             f"[{pct(cell['single_draw_fpr']['lower'])}, "
                             f"{pct(cell['single_draw_fpr']['upper'])}]"
                             if single else f"{cell['share_fpr_le_7_5']:.3f}")
                    rows.append({"seed": LABEL[seed], "pool": VIEW[pool], "method": method,
                                 "k": k, "mean_fpr": pct(cell["mean_fpr"]),
                                 "mean_tpr": pct(cell["mean_tpr"]),
                                 "share_fpr_le_7_5": share, "draws": str(cell["draws"]),
                                 "status": DECISION if judged and k == "50"
                                 else DESCRIPTIVE})
    return rows


def h3k_ece_rows(full: dict) -> list[dict[str, str]]:
    """Table 7: the pre-registered ECE of H3-k by k, E4 and E5 (reported, never judged)."""
    rows = []
    for seed, s in full["confirmatory"]["H3k"]["seeds"].items():
        for pool in ("e4_wildrf_matched", "e5_chameleon_pipeline"):
            v = s["views"][pool]
            rows.append({"seed": LABEL[seed], "pool": VIEW[pool], "method": "k = 0", "k": "0",
                         "ece": f"{v['k0']['ece']:.4f}", "draws": "-",
                         "status": DESCRIPTIVE})
            for method in ("Q", "S"):
                for k, cell in v[method].items():
                    rows.append({"seed": LABEL[seed], "pool": VIEW[pool], "method": method,
                                 "k": k, "ece": f"{cell['mean_ece']:.4f}",
                                 "draws": str(cell["draws"]), "status": DESCRIPTIVE})
    return rows


def slice_rows(full: dict, models: list[str]) -> list[dict[str, str]]:
    """Table 4 (descriptive): per slice AUC, AP and the two TPRs."""
    rows = []
    for model in models:
        d = full["descriptive"][model]
        for key, v in d["slices"].items():
            view, slot = key.split("/", 1)
            rows.append({"model": LABEL.get(model, model), "view": VIEW[view],
                         "slice": slot, "fakes": str(v["fakes"]),
                         "reals": str(v.get("reals", "-")),
                         "auc": f"{v['auc']:.4f}" if "auc" in v else "-",
                         "ap": f"{v['ap']:.4f}" if "ap" in v else "-",
                         "tpr_k0_union": pct(v["tpr_k0_union"]),
                         "tpr_at_5pct_slice_calibrated":
                             pct(v["tpr_at_5pct_slice_calibrated"])
                             if "tpr_at_5pct_slice_calibrated" in v else "-",
                         "status": DESCRIPTIVE})
    return rows


def h4_rows(full: dict) -> list[dict[str, str]]:
    """Table 5: the H4 curve (descriptive) and BANK-1L."""
    h4 = full["confirmatory"]["H4"]
    rows = []
    for order, points in h4["curves"].items():
        for p in points:
            val = p["heldout_lineage_val_auc"]
            rows.append({"order": order, "k_lineages": str(p["k"]),
                         "decoders": str(p["decoders"]), "model": p["model"],
                         "panel_b_macro_auc": ci(p["macro_auc"]),
                         "heldout_lineage_val_auc": "none held out" if val is None
                         else f"{val:.4f}",
                         "status": DECISION + " (k = 8 - k = 1)" if p["k"] in (1, 8)
                         else DESCRIPTIVE})
    b = h4["bank_1l_descriptive"]
    rows.append({"order": "BANK-1L", "k_lineages": "1", "decoders": "1",
                 "model": "bank_1l_seed0", "panel_b_macro_auc": ci(b["macro_auc"]),
                 "heldout_lineage_val_auc": f"{b['heldout_lineage_val_auc']:.4f}",
                 "status": DESCRIPTIVE})
    return rows


def h5_rows(full: dict) -> list[dict[str, str]]:
    """Table 6: the 12-condition curves of the 3 models."""
    h5 = full["confirmatory"]["H5"]
    rows = []
    for model, curve in h5["curves"].items():
        for condition, value in curve.items():
            decided = model == h5["model"] and condition in h5["decision"]
            rows.append({"model": LABEL[model], "condition": condition,
                         "macro_auc": ci(value),
                         "minus_clean": f"{value['point'] - curve['clean']['point']:+.4f}",
                         "status": DECISION if decided else DESCRIPTIVE})
    return rows


def truefake_rows(tf: dict) -> list[dict[str, str]]:
    """Table 8a (descriptive): TrueFake AUC per generator x platform, Delta vs pre-share."""
    rows = []
    for model, r in tf["results"].items():
        for pool, items in r["auc"]["PreSocial"].items():
            for item in items:
                row = {"model": LABEL.get(model, model), "real_pool": pool,
                       "fakes": MACRO_LABEL.get(item, item),
                       "pre_share_auc": ci(r["auc"]["PreSocial"][pool][item])}
                for share in PLATFORMS:
                    row[f"{share.lower()}_auc"] = ci(r["auc"][share][pool][item])
                    row[f"{share.lower()}_delta"] = ci(r["delta"][share][pool][item])
                row["status"] = DESCRIPTIVE
                rows.append(row)
    return rows


def truefake_fpr_rows(tf: dict) -> list[dict[str, str]]:
    """Table 8b (descriptive): FORLAB's FPR at each model's k = 0 union threshold."""
    by_model = tf["forlab_fpr_at_k0_union_threshold"]["by_model"]
    return [{"model": LABEL.get(model, model), "threshold": f"{v['threshold']:.4f}",
             **{f"{share.lower()}_fpr": pct(v[share]) for share in ("PreSocial",) + PLATFORMS},
             "status": DESCRIPTIVE} for model, v in by_model.items()]


def coco_only_tpr(full: dict, models: list[str]) -> dict[str, dict[str, float]]:
    """TPR over each pool's test fakes at the committed COCO-only threshold (descriptive).

    The same loader first reproduces the stored union FPR and TPR exactly; otherwise the
    committed scores are not the ones the analysis read, and nothing is written.
    """
    from p3_analyze import load_clean

    out = {}
    for model in models:
        panel = load_clean(LISTS, SCORES, model)
        d = full["descriptive"][model]
        for pool in ("e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline"):
            again = float(np.mean(panel.real[pool] >= d["threshold_union"]))
            if abs(again - d["pools"][pool]["fpr_union"]) > 1e-12:
                raise FigureError(f"{model} {pool}: union FPR does not reproduce")
        for key, v in d["slices"].items():
            view, slot = key.split("/", 1)
            again = float(np.mean(panel.fake[(view, slot)] >= d["threshold_union"]))
            if abs(again - v["tpr_k0_union"]) > 1e-12:
                raise FigureError(f"{model} {key}: union TPR does not reproduce")
        out[model] = {}
        for pool in ("e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline"):
            fakes = np.concatenate([s for (v, _), s in panel.fake.items() if v == pool])
            out[model][pool] = float(np.mean(fakes >= d["threshold_coco_only"]))
    return out


def write_table(directory: Path, name: str, title: str, rows: list[dict[str, str]],
                note: str) -> None:
    """One table as CSV and Markdown, atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    for suffix in ("csv", "md"):
        path = directory / f"{name}.{suffix}"
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            if suffix == "csv":
                writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            else:
                handle.write(f"# {title}\n\n{note}\n\n")
                handle.write("| " + " | ".join(columns) + " |\n")
                handle.write("|" + "---|" * len(columns) + "\n")
                for row in rows:
                    handle.write("| " + " | ".join(row[c].replace("|", "\\|")
                                                   for c in columns) + " |\n")
        tmp.replace(path)


# --------------------------------------------------------------------- figures ---


def style(ax: Any) -> None:
    """Recessive chrome: hairline grid, muted axes, ink text."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig: Any, directory: Path, name: str) -> list[Path]:
    """PNG and SVG; each must stay within the 1 MB chart rule."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, facecolor="white", metadata=(
            {"Date": None} if suffix == "svg" else {"Software": None}))
        if path.stat().st_size > 1_000_000:
            raise FigureError(f"{path} is larger than 1 MB")
        paths.append(path)
    return paths


def figure_h3k(full: dict, directory: Path) -> list[Path]:
    """FPR and TPR by k, methods Q and S, both seeds; E4/E5 judged, E3/E7 descriptive."""
    import matplotlib.pyplot as plt

    seeds = full["confirmatory"]["H3k"]["seeds"]
    pools = ("e4_wildrf_matched", "e5_chameleon_pipeline", "e3_matched", "e7_dda_coco")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharey="row")
    for col, pool in enumerate(pools):
        judged = pool in pools[:2]
        for row, metric in enumerate(("fpr", "tpr")):
            ax = axes[row][col]
            style(ax)
            for m, method in enumerate(("Q", "S")):
                for seed, dash in (("s2d_p35_seed0", "-"), ("s2d_p35_seed1", "--")):
                    v = seeds[seed]["views"][pool]
                    ks = [0] + [int(k) for k in v[method]]
                    ys = [v["k0"][metric]] + [v[method][k][f"mean_{metric}"]
                                              for k in v[method]]
                    ax.plot(ks, ys, dash, color=SERIES[m], linewidth=1.5, marker="o",
                            markersize=4.5, label=f"({method}) {LABEL[seed]}")
            if metric == "fpr":
                ax.axhline(0.075, color=MUTED, linestyle=":", linewidth=1)
                ax.text(ks[-1], 0.078, "7.5%", color=INK2, fontsize=7, ha="right",
                        va="bottom")
            ax.set_xticks(ks)
            if row == 0:
                ax.set_title(f"{VIEW[pool]} -- {'judged at k = 50' if judged else 'descriptive'}",
                             fontsize=9, color=INK)
            if row == 1:
                ax.set_xlabel("k target-domain reals", fontsize=8, color=INK2)
        axes[0][0].set_ylabel("mean achieved test FPR", fontsize=8, color=INK2)
        axes[1][0].set_ylabel("mean test TPR", fontsize=8, color=INK2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("H3-k: few-shot calibration of S2-D (decision: share of k = 50 draws with "
                 "FPR <= 7.5% on E4 and E5; every curve shown is descriptive)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    paths = save(fig, directory, "fig_h3k_fpr_tpr_by_k")
    plt.close(fig)
    return paths


def figure_h4(full: dict, directory: Path) -> list[Path]:
    """Panel B macro AUC by bank size beside the held-out-lineage validation AUC."""
    import matplotlib.pyplot as plt

    h4 = full["confirmatory"]["H4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for n, order in enumerate(("near", "far")):
        points = h4["curves"][order]
        ks = [p["k"] + (n - 0.5) * 0.08 for p in points]
        mid = [p["macro_auc"]["point"] for p in points]
        err = [[p["macro_auc"]["point"] - p["macro_auc"]["lower"] for p in points],
               [p["macro_auc"]["upper"] - p["macro_auc"]["point"] for p in points]]
        axes[0].errorbar(ks, mid, yerr=err, color=SERIES[n], linewidth=1.5, marker="o",
                         markersize=4.5, capsize=2, label=f"{order}-first order")
        val = [(p["k"], p["heldout_lineage_val_auc"]) for p in points
               if p["heldout_lineage_val_auc"] is not None]
        axes[1].plot([k for k, _ in val], [v for _, v in val], color=SERIES[n],
                     linewidth=1.5, marker="o", markersize=4.5, label=f"{order}-first order")
    b = h4["bank_1l_descriptive"]
    axes[0].errorbar([1.12], [b["macro_auc"]["point"]],
                     yerr=[[b["macro_auc"]["point"] - b["macro_auc"]["lower"]],
                           [b["macro_auc"]["upper"] - b["macro_auc"]["point"]]],
                     color=SERIES[2], marker="D", markersize=4.5, capsize=2, linestyle="none",
                     label="BANK-1L")
    axes[1].plot([1.12], [b["heldout_lineage_val_auc"]], color=SERIES[2], marker="D",
                 markersize=4.5, linestyle="none", label="BANK-1L")
    gain = h4["bank8_minus_bank1"]
    axes[0].set_title(f"Panel B macro AUC, 4 unseen slices (95% CI)\nH4 decision: "
                      f"BANK-8 - BANK-1 = {ci(gain, 3)}, {h4['verdict']}", fontsize=9,
                      color=INK)
    axes[1].set_title("held-out-lineage validation AUC (reconstructions of the lineages\n"
                      "outside each run; none at k = 8) -- descriptive", fontsize=9, color=INK)
    for ax in axes:
        style(ax)
        ax.set_xticks([1, 2, 4, 6, 8])
        ax.set_xlabel("k = decoder lineages in the bank", fontsize=8, color=INK2)
        ax.set_ylabel("AUC", fontsize=8, color=INK2)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("H4: bank size (the curves are descriptive; separate y-scales, separate "
                 "charts)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    paths = save(fig, directory, "fig_h4_bank_size")
    plt.close(fig)
    return paths


def figure_h5(full: dict, directory: Path) -> list[Path]:
    """Macro AUC under each of the 12 conditions, 3 models, one panel per family."""
    import matplotlib.pyplot as plt

    h5 = full["confirmatory"]["H5"]
    models = ("s2d_p35_seed0", "full_bank_seed0", "p_a_sd1_seed0")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [5, 4, 3, 4]})
    clean_s2d = h5["curves"][h5["model"]]["clean"]["point"]
    for ax, (family, conditions, clean_tick) in zip(axes, H5_FAMILIES, strict=True):
        style(ax)
        names = ("clean",) + conditions
        for m, model in enumerate(models):
            curve = h5["curves"][model]
            xs = np.arange(len(names)) + (m - 1) * 0.08
            mid = [curve[c]["point"] for c in names]
            err = [[curve[c]["point"] - curve[c]["lower"] for c in names],
                   [curve[c]["upper"] - curve[c]["point"] for c in names]]
            ax.errorbar(xs, mid, yerr=err, color=SERIES[m], linewidth=1.5, marker="o",
                        markersize=4.5, capsize=2, label=LABEL[model])
            if model == h5["model"]:
                for c in names:
                    if c in h5["decision"]:
                        x = float(xs[names.index(c)])
                        ax.plot([x], [curve[c]["point"]], marker="o", markersize=11,
                                markerfacecolor="none", markeredgecolor=INK,
                                markeredgewidth=1.2, linestyle="none")
        if any(c in h5["decision"] for c in conditions):
            ax.axhline(clean_s2d - 0.05, color=MUTED, linestyle=":", linewidth=1)
            ax.text(0.5, clean_s2d - 0.047, "S2-D clean - 5 points", color=INK2,
                    fontsize=7, ha="left", va="bottom")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([clean_tick] + [TICK[c] for c in conditions])
        ax.set_xlabel(family, fontsize=8, color=INK2)
    axes[0].set_ylabel("macro AUC, 4 unseen slices (95% CI)", fontsize=8, color=INK2)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"H5: robustness on Panel B test rows. Decision ({h5['verdict']}): the two "
                 "ringed S2-D points within 5 points of clean; everything else descriptive",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths = save(fig, directory, "fig_h5_robustness")
    plt.close(fig)
    return paths


def main() -> int:
    """Write every table and figure; 0 when all were written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPORTS / "p3_results")
    args = parser.parse_args()
    log = setup_logging("p3_thesis_figures")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "dbp"   # stable SVG ids across re-runs
    data = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in SOURCES.items()}
    full = data["full"]
    tables = args.out / "tables"
    coco = coco_only_tpr(full, ["full_bank_seed0"])
    every = list(full["descriptive"])
    tf = data["truefake"]
    boot = tf["bootstrap"]
    kept = ", ".join(f"{g} {n:,}" for g, n in tf["counts"]["kept_ids"].items())
    specs = [
        ("t1_verdicts", "P3 hypothesis verdicts", verdict_rows(data["h1a"], data["h1c"], full),
         "Every row is read from the committed results; `status` says whether it is a "
         "pre-registered decision or descriptive."),
        ("t2_h3_three_step", "H3 three-step table: FPR with TPR beside it",
         three_step_rows(full, coco),
         "k = 0. FPR over each pool's Panel B test reals; TPR over its test fakes (E3: both "
         "slices pooled). Union TPRs are read from the committed analysis; the COCO-only "
         "TPRs are computed from the committed scores after the stored union FPRs and TPRs "
         "were reproduced exactly. Every TPR is descriptive."),
        ("t3_h3k_by_k", "H3-k: FPR and TPR by k", h3k_rows(full),
         "Decision rows: k = 50, E4 and E5. Single-draw cells give one FPR with its "
         "bootstrap CI over the test reals."),
        ("t4_slice_tpr_main", "Per-slice AUC, AP and TPR (descriptive), main models",
         slice_rows(full, [m for m in MAIN if m in full["descriptive"]]),
         "TPR at the k = 0 union threshold and TPR at a 5% FPR threshold calibrated on the "
         "slice's own test reals. E8 is fakes-only (TPR at k = 0 only). E7 FLUX.1 standard "
         "is an upper bound (section 6d shortcut)."),
        ("t4_slice_tpr_all", "Per-slice AUC, AP and TPR (descriptive), all 24 models",
         slice_rows(full, every), "As the main-model table, for every frozen model."),
        ("t5_h4_curve", "H4: bank-size curve", h4_rows(full),
         "The decision is k = 8 minus k = 1 (Table 1); the curve and the held-out-lineage "
         "validation AUC are descriptive."),
        ("t6_h5_curves", "H5: 12-condition robustness curves", h5_rows(full),
         "Macro AUC over the 4 unseen slices, Panel B test rows; 95% CIs."),
        ("t7_h3k_ece", "H3-k: ECE by k (E4, E5)", h3k_ece_rows(full),
         "Expected calibration error, 15 equal-width bins, of sigmoid(score) over the view's "
         "test reals and test fakes (mean over draws; one draw where the pool is used whole). "
         "Pre-registered as reported, never judged. (Q) moves only the threshold, so its ECE "
         "equals k = 0's by construction (interpretation h3k_ece, confirmed 2026-10-28)."),
        ("t8a_truefake_auc", "TrueFake (descriptive): AUC per generator x platform",
         truefake_rows(tf),
         f"Secondary and descriptive: no hypothesis, no decision. Each generator's fakes "
         f"against one real pool, per version; delta = platform AUC - pre-share AUC. Paired "
         f"bootstrap, {boot['resamples']:,} resamples, seed {boot['seed']}, 95% percentile "
         f"CIs; kept ids {kept}. Caveats: " + " ".join(tf["caveats"])),
        ("t8b_truefake_forlab_fpr",
         "TrueFake (descriptive): FORLAB FPR at the k = 0 union threshold",
         truefake_fpr_rows(tf),
         "Share of FORLAB reals at or above each model's committed k = 0 union threshold (5% "
         "target on the 891 threshold reals), per version. Point values only: the analysis "
         "stores no CI for them. Descriptive only, never a claim: both TrueFake real pools "
         "are Flickr, the platform of our training reals."),
    ]
    for name, title, rows, note in specs:
        write_table(tables, name, title, rows, note)
        log.info("table %s: %d rows", name, len(rows))
    figures = args.out / "figures"
    for builder in (figure_h3k, figure_h4, figure_h5):
        for path in builder(full, figures):
            log.info("figure %s (%d bytes)", path.name, path.stat().st_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
