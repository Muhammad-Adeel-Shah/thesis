"""P4 thesis tables and figure, from the committed analysis (descriptive; no new metric).

Reads ``reports/p4_results/p4_analysis.json`` (spec section 5) and, for S2-D's own H5
change from clean, ``reports/p3_results/p3_analysis_full.json``. Writes
``reports/p4_results/tables/*.csv|md`` and ``reports/p4_results/figures/*.png|svg``:

* ``t9a_unseen_auc``: per-slice AUC [95% CI] on the 4 unseen slices and the macro, for
  DDA, CO-SPY, FIRE, AEROBLADE and S2-D seeds 0 and 1, P2 ``full_bank``, S2-B seed 0;
* ``t9b_fpr_tpr``: achieved FPR on the E4 and E5 real pools [CI], their macro, and TPR on
  the 4 unseen slices, at each model's own k = 0 union threshold;
* ``t9c_paired``: S2-D s0 / S2-D s1 / ``full_bank`` minus each baseline row, macro AUC and
  macro FPR, paired 95% CIs;
* ``t9d_e7_auc`` (E7 per VAE, with DDA's home advantage stated), ``t9d_e8_tpr``,
  ``t9d_crop308`` (DDA's sensitivity row) and ``t9d_h5`` (DDA under JPEG q50 and
  resize 0.5 beside S2-D seed 0's committed H5 numbers);
* ``fig_p4_macro_auc``: macro AUC with 95% CI for every model, baselines beside ours.

Every row is labelled descriptive: P4 has no decision rule (spec, header).
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
    DESCRIPTIVE,
    INK,
    INK2,
    LABEL,
    MUTED,
    SERIES,
    ci,
    pct,
    save,
    style,
    write_table,
)

REPORTS = THESIS_DBP / "reports"
ANALYSIS = REPORTS / "p4_results" / "p4_analysis.json"
P3_FULL = REPORTS / "p3_results" / "p3_analysis_full.json"
OUT = REPORTS / "p4_results"
BASELINE_LABEL = {"dda_native": "DDA", "dda_crop308": "DDA crop-308 (sensitivity)",
                  "cospy_native": "CO-SPY", "fire_native": "FIRE",
                  "aeroblade": "AEROBLADE (ours, from the paper)"}
MAIN_BASELINES = ("dda_native", "cospy_native", "fire_native", "aeroblade")
OURS = ("s2d_p35_seed0", "s2d_p35_seed1", "full_bank_seed0", "s2b_seed0")
UNSEEN = (("e3_matched/GPT-Image2-fake", "GPT-Image-2 (E3)"),
          ("e3_matched/Nano-Banana2-fake", "Nano-Banana-2 (E3)"),
          ("e4_wildrf_matched/fake", "E4 WildRF"), ("e5_chameleon_pipeline/fake", "E5 Chameleon"))
E7 = (("e7_dda_coco/FLUX.1", "FLUX.1 (upper bound)"), ("e7_dda_coco/sd-vae-ft-ema", "sd-ft-ema"),
      ("e7_dda_coco/sd-vae-ft-mse", "sd-ft-mse"), ("e7_dda_coco/sdxl-vae", "SDXL"),
      ("e7_dda_coco/stable-diffusion-2-1", "SD 2.1"),
      ("e7_dda_coco/stable-diffusion-3.5-large", "SD 3.5"),
      ("e7_dda_coco_luma/FLUX.1", "FLUX.1 luma"))
E8 = ("Flux", "GoT", "Infinity", "NOVA", "OmniGen")
CAVEATS = ("Descriptive only: P4 has no hypothesis and no decision rule (spec header). "
           "AEROBLADE is our implementation from the paper (its repository has no licence). "
           "E7 FLUX.1 standard is an upper bound (CLAUDE.md section 6d). Kaggle's Pillow 13.0.0 "
           "is not the 12.3.0 DDA's README pins (spec section 4).")
HOME = ("E7 is DDA's home benchmark: DDA-COCO was built by DDA's authors from MSCOCO val2017 "
        "images reconstructed by VAEs, and DDA trained on MSCOCO train reconstructed by the "
        "SD 2.1 VAE, so its E7 numbers are not a fair cross-generator comparison. CO-SPY's "
        "training reals are MSCOCO too (split NOT-VERIFIED).")


def models(analysis: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """(id, label, row) for the four primary baselines, then ours."""
    out = [(m, BASELINE_LABEL[m], analysis["rows"][m]) for m in MAIN_BASELINES]
    return out + [(m, LABEL[m], analysis["reference"][m]) for m in OURS]


def group(model: str) -> str:
    """'baseline' or 'ours'."""
    return "ours" if model in OURS else "baseline"


def t9a(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """Per-slice AUC on the 4 unseen slices and the macro."""
    rows = []
    for model, label, row in models(analysis):
        rows.append({"model": label, "group": group(model),
                     **{name: ci(row["slices"][key]["auc"], 3) for key, name in UNSEEN},
                     "macro AUC (4 unseen)": ci(row["macro_auc_unseen"], 3),
                     "status": DESCRIPTIVE})
    return rows


def t9b(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """FPR on E4/E5 and TPR on the unseen slices at each model's union threshold."""
    rows = []
    for model, label, row in models(analysis):
        fpr = {v: row["pools"][v]["fpr_union"] for v in ("e4_wildrf_matched",
                                                          "e5_chameleon_pipeline")}
        rows.append({"model": label, "group": group(model),
                     "threshold (k = 0 union)": f"{row['threshold_union']:.4f}",
                     "FPR E4": ci_pct(fpr["e4_wildrf_matched"]),
                     "FPR E5": ci_pct(fpr["e5_chameleon_pipeline"]),
                     "macro FPR E4/E5": ci_pct(row["macro_fpr_e4_e5"]),
                     **{f"TPR {name}": pct(row["slices"][key]["tpr_k0_union"])
                        for key, name in UNSEEN},
                     "status": DESCRIPTIVE})
    return rows


def ci_pct(value: dict[str, Any]) -> str:
    """A rate with its CI, in percent."""
    return f"{pct(value['point'])} [{pct(value['lower'])}, {pct(value['upper'])}]"


def ci_points(value: dict[str, Any]) -> str:
    """A difference in AUC points (x 100) with its CI, signed."""
    return (f"{100 * value['point']:+.1f} [{100 * value['lower']:+.1f}, "
            f"{100 * value['upper']:+.1f}]")


def excludes_zero(value: dict[str, Any]) -> str:
    """Whether the CI excludes 0 (stated, not judged)."""
    return "yes" if value["lower"] > 0 or value["upper"] < 0 else "no"


def t9c(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """Paired differences, ours minus each baseline row."""
    rows = []
    for ours in ("s2d_p35_seed0", "s2d_p35_seed1", "full_bank_seed0"):
        for base in BASELINE_LABEL:
            pair = analysis["paired"][f"{ours}-{base}"]
            rows.append({"ours": LABEL[ours], "minus baseline": BASELINE_LABEL[base],
                         "delta macro AUC, points": ci_points(pair["macro_auc_unseen"]),
                         "AUC CI excludes 0": excludes_zero(pair["macro_auc_unseen"]),
                         "delta macro FPR E4/E5, points": ci_points(pair["macro_fpr_e4_e5"]),
                         "FPR CI excludes 0": excludes_zero(pair["macro_fpr_e4_e5"]),
                         "status": DESCRIPTIVE})
    return rows


def all_rows(analysis: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Every baseline row (crop-308 included), then ours."""
    out = [(m, BASELINE_LABEL[m], analysis["rows"][m]) for m in BASELINE_LABEL]
    return out + [(m, LABEL[m], analysis["reference"][m]) for m in OURS]


def t9d_e7(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """E7 per-VAE AUC (and E7 luma FLUX.1)."""
    return [{"model": label, "group": group(model),
             **{name: ci(row["slices"][key]["auc"], 3) for key, name in E7},
             "status": DESCRIPTIVE} for model, label, row in all_rows(analysis)]


def t9d_e8(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """E8 TPR per generator at the union threshold (E8 has no reals)."""
    return [{"model": label, "group": group(model),
             **{g: pct(row["slices"][f"e8_evalgen/{g}"]["tpr_k0_union"]) for g in E8},
             "status": DESCRIPTIVE} for model, label, row in all_rows(analysis)]


def t9d_crop308(analysis: dict[str, Any]) -> list[dict[str, str]]:
    """DDA native beside its crop-308 sensitivity row."""
    rows = []
    for model in ("dda_native", "dda_crop308"):
        row = analysis["rows"][model]
        rows.append({"row": BASELINE_LABEL[model],
                     "crop": "CenterCrop(336), DDA's own" if model == "dda_native"
                     else "CenterCrop(308), sensitivity",
                     **{name: ci(row["slices"][key]["auc"], 3) for key, name in UNSEEN},
                     "macro AUC (4 unseen)": ci(row["macro_auc_unseen"], 3),
                     "macro FPR E4/E5": ci_pct(row["macro_fpr_e4_e5"]), "status": DESCRIPTIVE})
    return rows


def t9d_h5(analysis: dict[str, Any], p3_full: dict[str, Any]) -> list[dict[str, str]]:
    """DDA under the two H5 conditions beside S2-D seed 0 (committed P3 numbers)."""
    decision = p3_full["confirmatory"]["H5"]["decision"]
    rows = []
    for condition, label in (("jpeg50", "JPEG q50"), ("resize050", "resize-and-restore 0.5")):
        block = analysis["h5"][condition]
        rows.append({"condition": label,
                     "DDA macro AUC": ci(block["macro_auc"], 3),
                     "DDA minus DDA clean, points": ci_points(block["minus_dda_clean"]),
                     "S2-D s0 macro AUC": ci(block["s2d_seed0_macro_auc"], 3),
                     "S2-D s0 minus its clean, points (P3 H5)":
                         f"{100 * decision[condition]['difference']:+.1f}",
                     "S2-D s0 minus DDA, points": ci_points(block["s2d_seed0_minus_dda"]),
                     "status": DESCRIPTIVE})
    return rows


def figure(analysis: dict[str, Any], directory: Path) -> list[Path]:
    """Macro AUC with 95% CI for every model, baselines above ours."""
    import matplotlib.pyplot as plt

    entries = all_rows(analysis)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    colors = {"baseline": SERIES[1], "ours": SERIES[0]}
    for y, (model, _label, row) in enumerate(reversed(entries)):
        m = row["macro_auc_unseen"]
        kind = group(model)
        ax.errorbar(m["point"], y, xerr=[[m["point"] - m["lower"]], [m["upper"] - m["point"]]],
                    fmt="o", color=colors[kind], ecolor=colors[kind], elinewidth=2, capsize=0,
                    markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                    label=kind if kind not in ax.get_legend_handles_labels()[1] else None)
        ax.text(m["upper"] + 0.006, y, f"{m['point']:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(entries)), [label for _, label, _ in reversed(entries)],
                  fontsize=8, color=INK)
    ax.axvline(0.5, color=AXIS, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.503, -0.55, "chance", fontsize=7, color=MUTED, va="center")
    ours_rows = sum(1 for model, _, _ in entries if group(model) == "ours")
    ax.axhline(ours_rows - 0.5, color=AXIS, linewidth=0.8)
    ax.set_ylim(-0.9, len(entries) - 0.4)
    ax.set_xlabel("macro AUC over the 4 unseen slices (95% CI)", fontsize=9, color=INK2)
    ax.set_xlim(0.45, 0.85)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("ours", "baseline")]
    ax.legend([handles[i] for i in order], ["ours", "baselines"], frameon=False, fontsize=8,
              loc="upper left", bbox_to_anchor=(0.0, 0.99))
    ax.set_title("P4: baselines vs ours on Panel B (descriptive)", fontsize=10, color=INK,
                 loc="left")
    fig.tight_layout()
    paths = save(fig, directory, "fig_p4_macro_auc")
    plt.close(fig)
    return paths


def main() -> int:
    """Write t9a-t9d and the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    args = parser.parse_args()
    log = setup_logging("p4_thesis_tables")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "dbp"   # stable SVG ids across re-runs
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    p3_full = json.loads(P3_FULL.read_text(encoding="utf-8"))
    tables = args.out / "tables"
    specs = [
        ("t9a_unseen_auc", "Table 9a: AUC on the 4 unseen slices and their macro, 95% CI",
         t9a(analysis), CAVEATS),
        ("t9b_fpr_tpr", "Table 9b: FPR on the E4/E5 real pools and TPR on the unseen slices at "
         "each model's k = 0 union threshold", t9b(analysis),
         CAVEATS + " Each model's own threshold: 5% FPR by the reals-only rule on the 891 "
         "threshold reals (491 COCO validation + 400 held-out Open Images). TPR is a point "
         "value (spec section 5)."),
        ("t9c_paired", "Table 9c: paired differences, ours minus each baseline (AUC and FPR "
         "points, 95% CI)", t9c(analysis),
         CAVEATS + " A positive macro-AUC difference favours ours; a negative macro-FPR "
         "difference means ours has the lower FPR. 'CI excludes 0' is stated, not judged."),
        ("t9d_e7_auc", "Table 9d (i): E7 per-VAE AUC, 95% CI", t9d_e7(analysis),
         CAVEATS + " " + HOME),
        ("t9d_e8_tpr", "Table 9d (ii): E8 TPR at the k = 0 union threshold (fakes only)",
         t9d_e8(analysis), CAVEATS),
        ("t9d_crop308", "Table 9d (iii): DDA crop sensitivity, CenterCrop 336 vs 308",
         t9d_crop308(analysis), CAVEATS),
        ("t9d_h5", "Table 9d (iv): DDA under the two H5 conditions beside S2-D seed 0",
         t9d_h5(analysis, p3_full),
         CAVEATS + " Panel B test rows. S2-D seed 0's macro AUC under each condition "
         "reproduces the committed P3 H5 curve; its change from clean is P3's H5 decision "
         "value."),
    ]
    for name, title, rows, note in specs:
        write_table(tables, name, title, rows, note)
    paths = figure(analysis, args.out / "figures")
    log.info("wrote %d tables and %s", len(specs), [p.name for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
