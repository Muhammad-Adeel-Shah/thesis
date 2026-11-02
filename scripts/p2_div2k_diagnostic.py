"""P2 section 7: the DIV2K lossless diagnostic -- the JPEG-reliance check. CPU only.

T-ctrl is 200 DIV2K crops -- 2K photography stored as lossless PNG, **never JPEG** -- and
200 reconstructions of DIV2K crops by the bank's decoders, also lossless PNG. It is in
the frozen panel as ``t_ctrl_diagnostic``, withheld from every training list (D-051),
and every run has already scored it once (Session E). This script reads those committed
scores; it scores nothing and changes no P2 number.

Two questions per run and per score:

* **Does the detector call clean, never-compressed photographs fake?** FPR on the
  DIV2K reals at the 5% FPR threshold fixed on the validation reals (COCO, JPEG history),
  beside the FPR on E7's COCO reals at the same threshold, and their paired difference.
  A detector that learned "no JPEG trace -> fake" flags lossless reals.
* **Can it still separate real from fake when neither has a JPEG trace?** AUC and
  slice-calibrated TPR@5%FPR of DIV2K reals vs the DIV2K reconstructions.

Caveat carried into every output: DIV2K also differs from COCO in content and
resolution, so a high DIV2K FPR is consistent with JPEG reliance but does not isolate it.

Usage:
    python scripts/p2_div2k_diagnostic.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.eval.complementarity import GateSlice  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

RUNS = ("p_a_sd1_seed0", "p_a_sd1_seed1", "4ch_only_seed0", "full_bank_seed0")
SCORES = ("s1", "s2", "s3_set", "s3_lr", "aeroblade", "f_mean_rz", "f_stack")
VIEW = "t_ctrl_diagnostic"
REFERENCE_VIEW = "e7_dda_coco"
CAVEAT = ("DIV2K differs from COCO in content and resolution as well as compression, so "
          "a high DIV2K FPR is consistent with JPEG reliance but does not isolate it")


def _load(name: str, filename: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, THESIS_DBP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Write ``reports/p2_div2k/``. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "real")
    parser.add_argument("--validation-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "validation")
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports" / "p2_div2k")
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args()
    log = setup_logging("p2_div2k_diagnostic")
    x = _load("p2_exploratory_x_div2k", "p2_exploratory_x.py")
    analyze = x.analyze

    rows_out, payload = [], {"label": "P2 section 7 DIV2K lossless diagnostic",
                             "caveat": CAVEAT, "runs": {}}
    tex_rows = []
    for rid in RUNS:
        rows = analyze.read_scores(args.panel_dir / "scores" / f"scores_{rid}.csv")
        scores = analyze.validate_rows(rows)
        heads = json.loads((args.validation_dir / f"p2_heads_{rid}.json")
                           .read_text(encoding="utf-8"))
        thresholds = np.array([heads["calibrations"][s]["fpr_5"]["threshold"] for s in SCORES])
        matrix = np.column_stack([scores[s] for s in SCORES])
        units = [r["parent_id"] for r in rows]
        pick = {(view, label): [k for k, r in enumerate(rows)
                                if r["view"] == view and r["label"] == label]
                for view in (VIEW, REFERENCE_VIEW) for label in ("real", "fake")}
        div2k_reals, div2k_fakes = pick[(VIEW, "real")], pick[(VIEW, "fake")]
        coco_reals = pick[(REFERENCE_VIEW, "real")]
        if (len(div2k_reals), len(div2k_fakes)) != (200, 200) or len(coco_reals) != 486:
            raise RuntimeError(f"{rid}: unexpected T-ctrl / E7 counts "
                               f"{len(div2k_reals)}, {len(div2k_fakes)}, {len(coco_reals)}")
        # One structure, two resampling groups: DIV2K (reals and fakes within themselves)
        # and E7's reals; the same draw for every score (paired).
        div2k = GateSlice("div2k", np.asarray(div2k_fakes),
                          tuple(units[k] for k in div2k_fakes), "f:div2k",
                          np.asarray(div2k_reals), tuple(units[k] for k in div2k_reals),
                          "r:div2k")
        coco = GateSlice("coco", np.asarray(div2k_fakes), tuple(units[k] for k in div2k_fakes),
                         "f:div2k", np.asarray(coco_reals),
                         tuple(units[k] for k in coco_reals), "r:coco")
        stats: dict[str, list] = {k: [] for k in ("fpr_div2k", "fpr_coco", "auc", "tpr_slice",
                                                  "tpr_val")}
        for replicate in x.draws([div2k, coco], args.resamples):
            (fakes, reals_d), (_, reals_c) = replicate
            f, rd, rc = matrix[fakes], matrix[reals_d], matrix[reals_c]
            stats["fpr_div2k"].append((rd >= thresholds).mean(axis=0))
            stats["fpr_coco"].append((rc >= thresholds).mean(axis=0))
            stats["auc"].append(x.auc_columns(rd, f))
            stats["tpr_slice"].append((f >= x.slice_thresholds(rd)).mean(axis=0))
            stats["tpr_val"].append((f >= thresholds).mean(axis=0))
        arrays = {k: np.array(v) for k, v in stats.items()}
        arrays["fpr_gap"] = arrays["fpr_div2k"] - arrays["fpr_coco"]
        payload["runs"][rid] = {}
        for k, name in enumerate(SCORES):
            cells = {key: x.interval(arrays[key][:, k]) for key in arrays}
            row = {"run": x.RUN_NAMES[rid], "score": x.DISPLAY[name],
                   "fpr_div2k_reals_at_val_threshold": x.fmt(*cells["fpr_div2k"]),
                   "fpr_coco_reals_at_val_threshold": x.fmt(*cells["fpr_coco"]),
                   "fpr_gap_points": x.fmt(*(100 * np.array(cells["fpr_gap"])), digits=1),
                   "auc_div2k_real_vs_recon": x.fmt(*cells["auc"]),
                   "tpr_slice_calibrated": x.fmt(*cells["tpr_slice"]),
                   "tpr_at_val_threshold": x.fmt(*cells["tpr_val"])}
            rows_out.append(row)
            payload["runs"][rid][name] = {key: list(v) for key, v in cells.items()}
            if name in ("s1", "s2", "s3_set", "f_mean_rz"):
                tex_rows.append({"run": row["run"], "score": row["score"],
                                 "FPR DIV2K": x.fmt(cells["fpr_div2k"][0]),
                                 "FPR COCO": x.fmt(cells["fpr_coco"][0]),
                                 "gap (pts)": x.fmt(*(100 * np.array(cells["fpr_gap"])),
                                                    digits=1),
                                 "AUC": x.fmt(*cells["auc"])})
        log.info("%s done", rid)
    columns = list(rows_out[0])
    x.write_table(args.out_dir, "div2k_diagnostic", columns, rows_out,
                  "P2 section 7, DIV2K lossless diagnostic. FPR on 200 never-JPEG DIV2K reals "
                  "and on E7's 486 COCO reals at the 5% FPR validation threshold, their paired "
                  "gap, and AUC / TPR of DIV2K reals vs 200 DIV2K reconstructions; 95% bootstrap "
                  "CI. " + CAVEAT + ".")
    x.write_table(args.out_dir, "div2k_diagnostic_main", list(tex_rows[0]), tex_rows,
                  "P2 section 7, DIV2K lossless diagnostic (main modules and F-mean(A2)). "
                  + CAVEAT + ".")
    path = args.out_dir / "div2k_diagnostic.json"
    path.with_suffix(".tmp").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".tmp").replace(path)
    for row in rows_out:
        log.info("%-9s %-12s FPR DIV2K %s | COCO %s | gap %s | AUC %s", row["run"],
                 row["score"][:12], row["fpr_div2k_reals_at_val_threshold"],
                 row["fpr_coco_reals_at_val_threshold"], row["fpr_gap_points"],
                 row["auc_div2k_real_vs_recon"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
