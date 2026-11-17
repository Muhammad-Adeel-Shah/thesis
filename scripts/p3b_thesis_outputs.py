"""P3b thesis outputs: the Q1-Q6 table and two figures, from the committed records only.

Reads ``reports/p3b_results/p3b_analysis.json`` (the analysis, run once on the committed
scores), ``reports/p4_results/p4_analysis.json`` (DDA's and CO-SPY's rows, S2-D ViT-B's
reference rows, DDA under the H5 conditions) and ``reports/p3_results/p3_analysis_full.json``
(S2-D ViT-B seed 0's H5 curve). Nothing is re-scored or re-computed: every number is read
from those records. Writes, in the style of ``reports/p4_results``:

* ``reports/p3b_results/tables/t10_p3b_questions.csv|md``: Q1-Q6 per seed with the
  statistic, its 95% CI, the pre-registered rule and the verdict under the seed rule;
* ``reports/p3b_results/figures/fig_p3b_macro_auc.png|svg``: macro AUC over the 4 unseen
  slices with 95% CI for DDA, CO-SPY, S2-D ViT-B s0/s1 and S2-D ViT-L s0/s1;
* ``reports/p3b_results/figures/fig_p3b_robustness.png|svg``: clean, jpeg50 and resize050
  for DDA, S2-D ViT-B s0 (seed 1 has no robust scores) and S2-D ViT-L s0/s1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.utils.logging import setup_logging  # noqa: E402
from p3_thesis_figures import (  # noqa: E402
    AXIS,
    INK,
    INK2,
    MUTED,
    SERIES,
    FigureError,
    pct,
    save,
    style,
    write_table,
)

REPORTS = THESIS_DBP / "reports"
P3B = REPORTS / "p3b_results" / "p3b_analysis.json"
P4 = REPORTS / "p4_results" / "p4_analysis.json"
P3_FULL = REPORTS / "p3_results" / "p3_analysis_full.json"
OUT = REPORTS / "p3b_results"
SEEDS = ("0", "1")
CONDITIONS = (("clean", "clean"), ("jpeg50", "JPEG q50"), ("resize050", "resize 0.5"))
#: Colour by family (the reference palette's three validated slots); seeds differ by marker.
FAMILY_COLOR = {"baseline": SERIES[1], "vitb": SERIES[2], "vitl": SERIES[0]}
SEED_MARKER = {"0": "o", "1": "s"}
#: Section 4's pass rules, per (question, part).
RULES = {
    ("Q1", "matches"): "CI lower bound > -2 points, on both seeds (P3 H2's margin)",
    ("Q1", "beats"): "CI lower bound > 0, on both seeds",
    ("Q2", "matches"): "CI lower bound > -2 points, on both seeds",
    ("Q2", "beats"): "CI lower bound > 0, on both seeds",
    ("Q3", "gain"): "point >= +2 points and CI lower bound > 0, on both seeds (P3 H4)",
    ("Q4", "pass"): "macro FPR (E4, E5) <= 15% and neither pool > 25%, on both seeds (P3 H3)",
    ("Q5", "method (Q)"): "test FPR <= 7.5% in >= 80% of draws on each of E4 and E5, both "
                          "seeds (P3 H3-k)",
    ("Q5", "method (S)"): "test FPR <= 7.5% in >= 80% of draws on each of E4 and E5, both "
                          "seeds (P3 H3-k)",
    ("Q6", "pass"): "jpeg50 and resize050 both within 5 points of clean, on both seeds "
                    "(P3 H5)",
}
NOTE = ("Every difference is ViT-L minus the comparator, in AUC points, with its 95% "
        "percentile bootstrap CI (B = 2,000, seed 2026; P3's bootstrap, unchanged). Q1 is "
        "primary, Q2-Q6 secondary; all six are reported whatever the outcome (prereg section "
        "4, tag p3b-vitl-prereg-frozen). Seed rule: a question passes only if it passes on "
        "seed 0 and on seed 1; disagreeing seeds are 'seed-dependent' (not passed). Failing a "
        "'matches' rule is not evidence that ViT-L is worse: only a CI wholly below 0 shows a "
        "deficit. Limitation (prereg section 3): DINOv3 ViT-L/16 against DDA's DINOv2 "
        "ViT-L/14, the same size but a different pretraining and patch size. Q3: ViT-L ran on "
        "torch 2.11.0+cu130; ViT-B's torch version is NOT-VERIFIED (prereg section 7).")


def points(value: dict[str, Any]) -> str:
    """A difference in AUC points (x 100) with its CI, signed, two decimals."""
    return (f"{100 * value['point']:+.2f} [{100 * value['lower']:+.2f}, "
            f"{100 * value['upper']:+.2f}]")


def rate(value: dict[str, Any]) -> str:
    """A rate with its CI, in percent."""
    return f"{pct(value['point'])} [{pct(value['lower'])}, {pct(value['upper'])}]"


def mark(passed: bool) -> str:
    """A per-seed rule outcome."""
    return "pass" if passed else "fail"


def question_rows(p3b: dict[str, Any]) -> list[dict[str, str]]:
    """Table 10: Q1-Q6, both seeds, read from the committed analysis."""
    seeds, questions = p3b["seeds"], p3b["questions"]
    rows: list[dict[str, str]] = []

    def row(q: str, part: str, statistic: str, per_seed: dict[str, str],
            outcome: dict[str, Any]) -> None:
        rows.append({"question": q, "role": questions[q]["role"], "rule": part,
                     "statistic": statistic,
                     **{f"seed {s}": per_seed[s] for s in SEEDS},
                     **{f"seed {s} rule": mark(outcome["per_seed"][s]) for s in SEEDS},
                     "verdict": outcome["verdict"]})

    for q, comparator, label in (("Q1", "dda_native", "DDA"), ("Q2", "cospy_native", "CO-SPY")):
        diff = {s: points(seeds[s]["paired"][comparator]["macro_auc_unseen"]) for s in SEEDS}
        for part in ("matches", "beats"):
            row(q, part, f"macro AUC, ViT-L - {label} (points)", diff, questions[q][part])
    row("Q3", "gain", "macro AUC, ViT-L - S2-D ViT-B same seed (points)",
        {s: points(seeds[s]["paired"][f"s2d_p35_seed{s}"]["macro_auc_unseen"]) for s in SEEDS},
        questions["Q3"]["gain"])

    def q4(s: str) -> str:
        pools = seeds[s]["descriptive"]["pools"]
        return (f"E4 {rate(pools['e4_wildrf_matched']['fpr_union'])}; "
                f"E5 {rate(pools['e5_chameleon_pipeline']['fpr_union'])}; "
                f"macro {rate(seeds[s]['descriptive']['macro_fpr_e4_e5'])}")
    row("Q4", "pass", "FPR at k = 0, own union threshold", {s: q4(s) for s in SEEDS},
        questions["Q4"])
    for method in ("Q", "S"):
        def q5(s: str, m: str = method) -> str:
            views = seeds[s]["q5"]["views"]
            return (f"E4 {views['e4_wildrf_matched'][m]['50']['share_fpr_le_7_5']:.3f}; "
                    f"E5 {views['e5_chameleon_pipeline'][m]['50']['share_fpr_le_7_5']:.3f}")
        row("Q5", f"method ({method})", "share of k = 50 draws with test FPR <= 7.5%",
            {s: q5(s) for s in SEEDS}, questions["Q5"][method])

    def q6(s: str) -> str:
        conditions = seeds[s]["robust"]["conditions"]
        return "; ".join(f"{name} {points(conditions[c]['minus_clean'])}"
                         for c, name in CONDITIONS[1:])
    row("Q6", "pass", "macro AUC minus clean, test rows (points)", {s: q6(s) for s in SEEDS},
        questions["Q6"])
    for r in rows:
        r["pass rule (pre-registered)"] = RULES[(r["question"], r["rule"])]
    return rows


def macro_entries(p3b: dict[str, Any], p4: dict[str, Any]) -> list[tuple[str, str, dict]]:
    """(label, family, macro AUC) for figure (a), top to bottom."""
    out = [("DDA", "baseline", p4["rows"]["dda_native"]["macro_auc_unseen"]),
           ("CO-SPY", "baseline", p4["rows"]["cospy_native"]["macro_auc_unseen"])]
    out += [(f"S2-D ViT-B s{s}", "vitb", p4["reference"][f"s2d_p35_seed{s}"]["macro_auc_unseen"])
            for s in SEEDS]
    out += [(f"S2-D ViT-L s{s}", "vitl", p3b["seeds"][s]["descriptive"]["macro_auc_unseen"])
            for s in SEEDS]
    return out


def robustness_series(p3b: dict[str, Any], p4: dict[str, Any],
                      p3_full: dict[str, Any]) -> list[tuple[str, str, str, list[dict]]]:
    """(label, family, seed, [clean, jpeg50, resize050]) for figure (b)."""
    dda = [p4["rows"]["dda_native"]["macro_auc_unseen"],
           *(p4["h5"][c]["macro_auc"] for c, _ in CONDITIONS[1:])]
    curve = p3_full["confirmatory"]["H5"]["curves"]["s2d_p35_seed0"]
    vitb = [curve[c] for c, _ in CONDITIONS]
    if vitb[0] != p4["reference"]["s2d_p35_seed0"]["macro_auc_unseen"]:
        raise FigureError("S2-D ViT-B s0: P3's clean H5 point differs from P4's reference")
    out = [("DDA", "baseline", "0", dda), ("S2-D ViT-B s0", "vitb", "0", vitb)]
    for s in SEEDS:
        robust = p3b["seeds"][s]["robust"]
        out.append((f"S2-D ViT-L s{s}", "vitl", s,
                    [robust["clean"]["macro_auc"],
                     *(robust["conditions"][c]["macro_auc"] for c, _ in CONDITIONS[1:])]))
    return out


def figure_macro(entries: list[tuple[str, str, dict]], directory: Path) -> list[Path]:
    """Figure (a): macro AUC with 95% CI, one row per model."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    style(ax)
    seen: set[str] = set()
    legend = {"baseline": "baselines (P4)", "vitb": "S2-D ViT-B (P3)", "vitl": "S2-D ViT-L (P3b)"}
    for y, (_label, family, m) in enumerate(reversed(entries)):
        color = FAMILY_COLOR[family]
        ax.errorbar(m["point"], y, xerr=[[m["point"] - m["lower"]], [m["upper"] - m["point"]]],
                    fmt="o", color=color, ecolor=color, elinewidth=2, capsize=0, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.5,
                    label=legend[family] if family not in seen else None)
        seen.add(family)
        ax.text(m["upper"] + 0.006, y, f"{m['point']:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(entries)), [label for label, _, _ in reversed(entries)],
                  fontsize=8, color=INK)
    ax.axvline(0.5, color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.503, -0.55, "chance", fontsize=7, color=MUTED, va="center")
    ax.set_ylim(-0.9, len(entries) - 0.4)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("macro AUC over the 4 unseen slices (95% CI)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.25, 0.0))
    ax.set_title("P3b: macro AUC on Panel B (Q1-Q3 are the paired differences, Table 10)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    paths = save(fig, directory, "fig_p3b_macro_auc")
    plt.close(fig)
    return paths


def figure_robustness(series: list[tuple[str, str, str, list[dict]]],
                      directory: Path) -> list[Path]:
    """Figure (b): clean, jpeg50 and resize050 per model, 95% CIs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style(ax)
    offsets = [-0.15, -0.05, 0.05, 0.15]
    for (label, family, seed, values), dx in zip(series, offsets, strict=True):
        xs = [i + dx for i in range(len(CONDITIONS))]
        mid = [v["point"] for v in values]
        err = [[v["point"] - v["lower"] for v in values], [v["upper"] - v["point"] for v in values]]
        ax.errorbar(xs, mid, yerr=err, color=FAMILY_COLOR[family], linewidth=1.5,
                    marker=SEED_MARKER[seed] if family == "vitl" else "o", markersize=5.5,
                    markeredgecolor="white", markeredgewidth=1, capsize=2, label=label)
        nudge = {"S2-D ViT-L s0": 0.012, "S2-D ViT-L s1": -0.012}.get(label, 0.0)
        ax.text(xs[-1] + 0.05, mid[-1] + nudge, label, fontsize=7, color=INK2, va="center")
    ax.set_xticks(range(len(CONDITIONS)), [name for _, name in CONDITIONS], fontsize=8,
                  color=INK)
    ax.set_xlim(-0.4, len(CONDITIONS) - 0.4 + 0.45)
    ax.set_ylabel("macro AUC, 4 unseen slices, test rows (95% CI)", fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("P3b: robustness on Panel B test rows (Q6 rule and verdict: Table 10)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    paths = save(fig, directory, "fig_p3b_robustness")
    plt.close(fig)
    return paths


def main() -> int:
    """Write Table 10 and the two figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    log = setup_logging("p3b_thesis_outputs")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "dbp"   # stable SVG ids across re-runs
    p3b = json.loads(P3B.read_text(encoding="utf-8"))
    p4 = json.loads(P4.read_text(encoding="utf-8"))
    p3_full = json.loads(P3_FULL.read_text(encoding="utf-8"))
    write_table(args.out / "tables", "t10_p3b_questions",
                "Table 10: P3b questions Q1-Q6, S2-D ViT-L seeds 0 and 1 (pre-registered)",
                question_rows(p3b), NOTE)
    paths = figure_macro(macro_entries(p3b, p4), args.out / "figures")
    paths += figure_robustness(robustness_series(p3b, p4, p3_full), args.out / "figures")
    log.info("wrote t10_p3b_questions and %s", [p.name for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
