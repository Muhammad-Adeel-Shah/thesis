"""P2-B8: the pre-registered analysis and the go / no-go verdict.

This script implements ``docs/P2_PREREGISTRATION.md`` **as amended by Amendment 2**
(D-069) and nothing else. It has no flag that changes the criterion, the metrics, the
slices or the fusion, and it refuses to run if the amendment is absent.

The governing verdict (Amendment 2):

* **M1** = macro-average TPR@5%FPR over the unseen slices; **M2** = macro-average AUC
  over the unseen slices that have reals. Thresholds come from the validation reals.
* ``Δ_M = M(F-mean(A2)) - max over the five single heads of M(head)``, the maximum
  recomputed in every bootstrap replicate.
* **GO** if ``Δ >= 2.0 points`` on M1 or M2 and the 95% paired bootstrap CI excludes 0.
  Under P-A the run-level verdicts of both seeds are combined by ``--combine``: GO only
  if both pass.

**Analysis A** -- the original section 6: F-stack minus the best single stream fixed on
validation AUC -- is computed through the same slices and bootstrap and reported beside
it, with no role in the verdict.

**Slices (I-061).** A slice is one generator's fakes against its view's whole real
class: E7's subsets share E7's reals, E3's generators share E3's reals, the FLUX.1 luma
slice uses the luma reals, E4 and E5 are one slice each, E8's generators have no reals
(M1 only). Under P-A the twelve unseen slices of section 3 are asserted by name. A slice
the macro needs but cannot compute is an error; a non-finite score is an error.

**The bootstrap** (:func:`dbp.eval.complementarity.macro_bootstrap`) resamples every
slice's fakes within the slice and every view's reals within its pool -- one draw per
view, shared by all its slices, a luma row moving with its parent -- applies the same
draw to every head and fusion, and recomputes the macros.

Every slice is reported beside its **trivial-rule baseline** (D-044).

Usage:
    python scripts/p2_analyze.py --scores DIR/scores_p_a_sd1_seed0.csv \\
        --heads DIR/p2_heads_p_a_sd1_seed0.json --out-dir DIR
    python scripts/p2_analyze.py --combine DIR/p_a_sd1_seed0 DIR/p_a_sd1_seed1 \\
        --verdict-out DIR/p2_verdict_p_a_sd1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256, write_manifest  # noqa: E402
from dbp.eval.complementarity import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    GateSlice,
    _correct,
    expected_calibration_error,
    macro_bootstrap,
    pairwise,
    percentile_interval,
)
from dbp.eval.fusion import GOVERNING_STREAMS, sigmoid  # noqa: E402
from dbp.eval.metrics import average_precision, roc_auc  # noqa: E402
from dbp.eval.panel import LUMA_VIEWS, gate_slices_replaced_by_luma  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import analysis_dir  # noqa: E402

#: The gate, verbatim from the pre-registration. Not a flag.
GATE_MARGIN_POINTS = 2.0
GATE_CONFIDENCE = 0.95
BOOTSTRAP_SEED = 2026

SINGLE_STREAMS = ("s1", "s2", "s3_lr", "s3_set", "aeroblade")
GOVERNING_FUSION = "f_mean_rz"
FUSIONS = (GOVERNING_FUSION, "f_mean", "f_stack")
ALL_STREAMS = SINGLE_STREAMS + FUSIONS

#: Pre-registration section 3 with Amendments 1 and 2: the P-A slices M1 averages over.
#: Asserted, so a slice that went missing or appeared cannot silently re-weight it.
P_A_UNSEEN_SLICES = (
    "e3_matched/GPT-Image2-fake",
    "e3_matched/Nano-Banana2-fake",
    "e4_wildrf_matched",
    "e5_chameleon_pipeline",
    "e7_dda_coco/sdxl-vae",
    "e7_dda_coco/stable-diffusion-3.5-large",
    "e7_dda_coco_luma/FLUX.1",
    "e8_evalgen/Flux",
    "e8_evalgen/GoT",
    "e8_evalgen/Infinity",
    "e8_evalgen/NOVA",
    "e8_evalgen/OmniGen",
)
P_A_M2_SLICES = 7

#: Trivial-rule baselines measured in P1, printed beside every number from these
#: slices (D-044). Lift is in percentage points over the majority-class baseline.
TRIVIAL_BASELINES = {
    "e7_dda_coco/FLUX.1": "`4:2:0 -> fake` scores 100.00% (lift +49.85); quality +14.35",
    "e4_wildrf_matched":
        "quality separability +11.39 (~61% reachable without a generative artefact)",
    # E3's source_label carries the label suffix, so the baseline is matched on a prefix.
    "e3_matched/Nano-Banana2": "residual +6.47 after matching",
}

METRIC_COLUMNS = (
    "slice", "n", "n_real", "n_fake", "stream", "auc", "ap",
    "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "fpr_at_5pct_threshold", "balanced_accuracy",
    "ece", "in_gate", "trivial_baseline", "caveat",
)


class AnalysisError(RuntimeError):
    """The scores or the configuration cannot support the pre-registered analysis."""


# ------------------------------------------------------------------ helpers kept ---


def load_panel_hashes(path: Path) -> dict[str, str]:
    """``image_id -> sha256`` from the frozen panel list, or empty if it is absent."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["image_id"]): str(row.get("sha256", ""))
            for row in csv.DictReader(handle)
            if row.get("sha256")
        }


def exact_duplicate_groups(
    subset: list[dict[str, str]], hashes: dict[str, str]
) -> list[list[str]]:
    """Image ids sharing byte-identical content, as groups of two or more.

    WildRF distributes the same bytes under more than one path, and an evaluation set is
    used as distributed (P1), so these rows stay in the primary number. They are
    reported because a duplicated image is weighted twice in every metric computed over
    that slice.
    """
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in subset:
        digest = hashes.get(str(row["image_id"]), "")
        if digest:
            by_hash[digest].append(str(row["image_id"]))
    return [sorted(ids) for ids in by_hash.values() if len(ids) > 1]


def deduplicate(
    subset: list[dict[str, str]], groups: list[list[str]]
) -> list[dict[str, str]]:
    """``subset`` with each duplicate group reduced to its lexicographically first id."""
    drop = {image_id for group in groups for image_id in group[1:]}
    return [row for row in subset if str(row["image_id"]) not in drop]


def read_scores(path: Path) -> list[dict[str, str]]:
    """Panel scores written by ``p2_score.py``."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


#: Views whose subsets are each a slice in their own right. A luma companion is listed
#: through its parent: `e7_dda_coco_luma` must divide into `/FLUX.1` exactly as E7 does,
#: or the section 8 substitution has no slice to substitute.
SUBSET_SLICED_VIEWS = frozenset(
    {"e7_dda_coco", "e8_evalgen"}
    | {name for name, spec in LUMA_VIEWS.items() if spec["parent"] in ("e7_dda_coco", "e8_evalgen")}
)


def parent_view(view: str) -> str:
    """The view a luma companion derives from, or ``view`` itself."""
    spec = LUMA_VIEWS.get(view)
    return str(spec["parent"]) if spec else view


def slice_key(row: dict[str, str]) -> str:
    """The slice a **fake** row belongs to: the view, sub-divided where the plan says to.

    Reals do not belong to a slice: they are the view's real pool, shared by every
    slice of the view (I-061).
    """
    view = row["view"]
    if view in SUBSET_SLICED_VIEWS and row["subset"]:
        return f"{view}/{row['subset']}"
    if parent_view(view) == "e3_matched" and row["generator"]:
        return f"{view}/{row['generator']}"
    return view


def baseline_for(name: str) -> str:
    """The trivial-rule note for a slice, if one was measured in P1.

    Matched on a prefix, because a slice name carries whatever the set's own column
    said (``e3_matched/Nano-Banana2-fake``). A luma slice is looked up under its parent
    too: D-044 wants the baseline beside the luma number, which exists to defuse it.
    """
    view, _, rest = name.partition("/")
    candidates = [name]
    base = parent_view(view)
    if base != view:
        candidates.append(f"{base}/{rest}" if rest else base)
    for candidate in candidates:
        for key, note in TRIVIAL_BASELINES.items():
            if candidate == key or candidate.startswith(key):
                return note
    return ""


# ----------------------------------------------------------------------- checks ---


def validate_rows(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    """Every row has a unique id, a real/fake label and a finite score for every stream.

    Returns:
        Stream -> scores, in row order.

    Raises:
        AnalysisError: Listing the first ten offending ids. Nothing is dropped.
    """
    if not rows:
        raise AnalysisError("no score rows")
    ids = Counter(r["image_id"] for r in rows)
    duplicated = sorted(i for i, n in ids.items() if n > 1)
    if duplicated:
        raise AnalysisError(f"{len(duplicated)} duplicate id(s): {duplicated[:10]}")
    odd = [r["image_id"] for r in rows if r["label"] not in ("real", "fake")]
    if odd:
        raise AnalysisError(f"{len(odd)} row(s) without a real/fake label: {odd[:10]}")
    missing_columns = [s for s in ALL_STREAMS if s not in rows[0]]
    if missing_columns:
        raise AnalysisError(f"score file lacks stream column(s) {missing_columns}")
    scores: dict[str, np.ndarray] = {}
    for stream in ALL_STREAMS:
        values = np.array([float(r[stream]) if r[stream] != "" else np.nan for r in rows])
        bad = [rows[k]["image_id"] for k in np.flatnonzero(~np.isfinite(values))]
        if bad:
            raise AnalysisError(f"{stream}: {len(bad)} row(s) without a finite score, "
                                f"first 10: {bad[:10]}")
        scores[stream] = values
    return scores


def build_slices(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Slice name -> its fake rows, its view's real pool and its in-bank status."""
    fakes: dict[str, list[int]] = defaultdict(list)
    pools: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["label"] == "fake":
            fakes[slice_key(row)].append(index)
        else:
            pools[row["view"]].append(index)
    slices: dict[str, dict[str, Any]] = {}
    for name, members in sorted(fakes.items()):
        view = name.split("/", 1)[0]
        statuses = {rows[k]["in_bank"] for k in members}
        if len(statuses) != 1:
            raise AnalysisError(f"{name}: fakes disagree on in_bank status {sorted(statuses)}")
        slices[name] = {"view": view, "fakes": members, "reals": pools.get(view, []),
                        "in_bank": statuses.pop()}
    orphan = sorted(set(pools) - {s["view"] for s in slices.values()})
    if orphan:
        raise AnalysisError(f"real pool(s) {orphan} belong to no slice")
    return slices


def gate_slice_names(slices: dict[str, dict[str, Any]], log: Any) -> list[str]:
    """The unseen slices, with the section 8 luma substitution applied."""
    candidates = sorted(
        name for name, spec in slices.items()
        if spec["in_bank"] != "in-bank" and spec["view"] != "t_ctrl_diagnostic"
    )
    displaced = gate_slices_replaced_by_luma()
    present = set(candidates)
    for name in sorted(displaced & present):
        subset = name.split("/", 1)[1]
        replacement = [s for s in present if s.split("/", 1)[0] in LUMA_VIEWS
                       and s.endswith(f"/{subset}")]
        if not replacement:
            raise AnalysisError(
                f"section 8 displaces {name} from the gate, but no luma slice replaces it: "
                "the luma companion was not scored, and an unchecked FLUX.1 number is not "
                "claimable")
        log.info("section 8 substitution: %s -> %s", name, ", ".join(sorted(replacement)))
    return [name for name in candidates if name not in displaced]


def thresholds_from(fitted: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per stream and fusion, the validation-reals thresholds (Amendment 2 point 3)."""
    out: dict[str, dict[str, float]] = {}
    for stream in ALL_STREAMS:
        entry = fitted["calibrations"].get(stream)
        if entry is None:
            raise AnalysisError(f"no calibration for {stream} in the heads report")
        for key in ("fpr_1", "fpr_5"):
            if entry[key].get("rule") != "target_fpr_on_reals":
                raise AnalysisError(f"{stream} {key} was not fitted on the validation reals "
                                    f"alone (rule {entry[key].get('rule')!r}; I-062)")
        out[stream] = {key: float(entry[key]["threshold"]) for key in ("fpr_1", "fpr_5")}
    return out


# ---------------------------------------------------------------------- metrics ---


def slice_metrics(
    labels: np.ndarray, scores: np.ndarray, thresholds: dict[str, float], stream: str
) -> dict[str, Any]:
    """Every pre-registered metric for one stream on one slice.

    A slice with a single class has no AUC, AP, ECE or balanced accuracy; those come
    back ``None``, because E8 has no reals and a 0.5 written there would be read as a
    measurement. F-mean(A2) is a robust z, not a logit, so it has no ECE.
    """
    if labels.size == 0 or not np.all(np.isfinite(scores)):
        raise AnalysisError(f"{stream}: empty slice or non-finite score")
    both = len(np.unique(labels)) == 2
    reals, fakes = scores[labels == 0], scores[labels == 1]
    out: dict[str, Any] = {
        "n": int(labels.size), "n_real": int(reals.size), "n_fake": int(fakes.size),
        "auc": roc_auc(labels, scores) if both else None,
        "ap": average_precision(labels, scores) if both else None,
        "ece": (expected_calibration_error(labels, sigmoid(scores))
                if both and stream != GOVERNING_FUSION else None),
        "tpr_at_1pct_fpr": float(np.mean(fakes >= thresholds["fpr_1"])) if fakes.size else None,
        "tpr_at_5pct_fpr": float(np.mean(fakes >= thresholds["fpr_5"])) if fakes.size else None,
        "fpr_at_5pct_threshold": (float(np.mean(reals >= thresholds["fpr_5"]))
                                  if reals.size else None),
    }
    out["balanced_accuracy"] = (
        0.5 * (out["tpr_at_5pct_fpr"] + 1.0 - out["fpr_at_5pct_threshold"]) if both else None)
    return out


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    return round(value, 6) if isinstance(value, float) else value


# ------------------------------------------------------------------------- gate ---


def gate_slices(
    rows: list[dict[str, str]], slices: dict[str, dict[str, Any]], names: Sequence[str]
) -> list[GateSlice]:
    """The macro's slices as resampling structures (Amendment 2, I-061 b)."""
    out = []
    for name in names:
        spec = slices[name]
        view, _, rest = name.partition("/")
        canonical = f"{parent_view(view)}/{rest}" if rest else parent_view(view)
        reals = spec["reals"]
        out.append(GateSlice(
            name=name,
            fakes=np.asarray(spec["fakes"]),
            fake_units=tuple(rows[k]["parent_id"] for k in spec["fakes"]),
            fake_group=f"fake:{canonical}",
            reals=np.asarray(reals) if reals else None,
            real_units=tuple(rows[k]["parent_id"] for k in reals) if reals else None,
            real_group=f"real:{parent_view(view)}" if reals else None,
        ))
    return out


def gate_result(
    boot: dict[str, dict[str, np.ndarray]], fusion: str, against: Sequence[str] | str,
) -> dict[str, Any]:
    """Δ per metric, in points: ``fusion`` minus the max over ``against`` (re-maximised
    per replicate), or minus one fixed stream; with the percentile CI and the rule."""
    result: dict[str, Any] = {}
    passes = []
    for metric in ("M1", "M2"):
        fused = boot[metric][fusion]
        if isinstance(against, str):
            best = boot[metric][against]
            which: dict[str, Any] = {"fixed": against}
        else:
            stacked = np.vstack([boot[metric][s] for s in against])
            best = stacked.max(axis=0)
            winners = Counter(against[k] for k in stacked.argmax(axis=0)[1:])
            which = {"max_over": list(against),
                     "max_on_data": against[int(stacked[:, 0].argmax())],
                     "max_share_in_replicates": {k: v / (len(best) - 1)
                                                 for k, v in sorted(winners.items())}}
        delta = 100.0 * (fused - best)
        interval = percentile_interval(delta, GATE_CONFIDENCE)
        ok = interval["point"] >= GATE_MARGIN_POINTS and interval["excludes_zero"]
        if ok:
            passes.append(metric)
        result[metric] = {
            "fusion": fusion, "fusion_macro": float(fused[0]),
            "best_single_macro": float(best[0]), "best_single": which,
            "difference_points": interval["point"], "ci_points": [interval["lo"],
                                                                   interval["hi"]],
            "ci_excludes_zero": interval["excludes_zero"], "passes": ok,
        }
    m1, m2 = result["M1"]["difference_points"], result["M2"]["difference_points"]
    result["passes_on"] = passes
    result["verdict"] = "GO" if passes else "NO-GO"
    result["m1_m2_disagree_in_sign"] = bool(np.sign(m1) != np.sign(m2))
    return result


# ------------------------------------------------------------- complementarity ---


def oracle_rows(labels: np.ndarray, scores: dict[str, np.ndarray],
                thresholds: dict[str, dict[str, float]]) -> dict[str, Any]:
    """F-oracle for the governing members and for all five heads: an upper bound only."""
    out: dict[str, Any] = {}
    for name, members in (("governing_members", GOVERNING_STREAMS),
                          ("all_single_heads", SINGLE_STREAMS)):
        correct = {s: _correct(labels, scores[s], thresholds[s]["fpr_5"]) for s in members}
        any_right = np.any(np.vstack(list(correct.values())), axis=0)
        accuracy = {s: float(np.mean(c)) for s, c in correct.items()}
        best = max(accuracy, key=lambda s: accuracy[s])
        out[name] = {
            "streams": list(members),
            "oracle_accuracy": float(np.mean(any_right)),
            "oracle_tpr": float(np.mean(any_right[labels == 1])) if (labels == 1).any() else None,
            "oracle_tnr": float(np.mean(any_right[labels == 0])) if (labels == 0).any() else None,
            "best_single": best, "best_single_accuracy": accuracy[best],
            "headroom": float(np.mean(any_right)) - accuracy[best],
            "note": "F-oracle uses the label: an upper bound, never a method or a result",
        }
    return out


# ------------------------------------------------------------------------- main ---


def combine_seeds(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Amendment 2, point 6: the P-A verdict is GO only if every seed's run passes."""
    runs = {}
    for directory in run_dirs:
        payload = json.loads((Path(directory) / "go_no_go.json").read_text(encoding="utf-8"))
        runs[payload["run_id"]] = {
            "verdict": payload["governing"]["verdict"],
            "passes_on": payload["governing"]["passes_on"],
            "M1": payload["governing"]["M1"], "M2": payload["governing"]["M2"],
            "synthetic": payload.get("synthetic", False),
        }
    protocols = {rid.rsplit("_seed", 1)[0] for rid in runs}
    if len(protocols) != 1 or len(runs) < 2:
        raise AnalysisError(f"--combine needs >= 2 seeds of one protocol, got {sorted(runs)}")
    verdict = "GO" if all(r["verdict"] == "GO" for r in runs.values()) else "NO-GO"
    return {"protocol": protocols.pop(), "rule": "Amendment 2 point 6: GO only if every "
            "seed passes", "runs": runs, "verdict": verdict,
            "synthetic": any(r["synthetic"] for r in runs.values())}


def write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    """Run the pre-registered analysis, or combine seeds. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--heads", type=Path)
    parser.add_argument("--preregistration", type=Path,
                        default=THESIS_DBP / "docs" / "P2_PREREGISTRATION.md")
    parser.add_argument("--panel-hashes", type=Path, default=None,
                        help="The frozen panel list, whose sha256 column is how exact "
                             "duplicates are found.")
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports" / "p2")
    parser.add_argument("--synthetic", action="store_true",
                        help="Stamp every output SYNTHETIC (pipeline dry-run).")
    parser.add_argument("--combine", nargs="+", type=Path, default=None,
                        help="Per-seed analysis directories to combine (Amendment 2 point 6).")
    parser.add_argument("--verdict-out", type=Path, default=None)
    args = parser.parse_args()
    log = setup_logging("p2_analyze")
    try:
        if args.combine:
            if args.verdict_out is None:
                raise AnalysisError("--combine needs --verdict-out")
            verdict = combine_seeds(args.combine)
            write_json(args.verdict_out, verdict)
            log.info("COMBINED VERDICT %s: %s %s", verdict["protocol"], verdict["verdict"],
                     {k: v["verdict"] for k, v in verdict["runs"].items()})
            return 0
        if args.scores is None or args.heads is None:
            raise AnalysisError("--scores and --heads are required")
        return analyse(args, log)
    except AnalysisError as exc:
        log.error("%s", exc)
        return 1


def analyse(args: argparse.Namespace, log: Any) -> int:
    """The per-run analysis."""
    if not args.preregistration.is_file():
        raise AnalysisError(f"{args.preregistration} is missing")
    prereg = args.preregistration.read_text(encoding="utf-8")
    if "## Amendment 2" not in prereg:
        raise AnalysisError("pre-registration Amendment 2 is not in the committed plan")

    fitted = json.loads(args.heads.read_text(encoding="utf-8"))
    rid = fitted["run_id"]
    if args.scores.stem != f"scores_{rid}":
        raise AnalysisError(f"{args.scores.name} does not belong to {rid}")
    out_dir = analysis_dir(args.out_dir, rid)
    rows = read_scores(args.scores)
    scores = validate_rows(rows)
    thresholds = thresholds_from(fitted)
    labels = np.array([1 if r["label"] == "fake" else 0 for r in rows])
    log.info("%s: %d rows, protocol %s%s", rid, len(rows), fitted["protocol"],
             " (SYNTHETIC)" if args.synthetic else "")

    slices = build_slices(rows)
    unseen = gate_slice_names(slices, log)
    m2_slices = [s for s in unseen if slices[s]["reals"]]
    if fitted["protocol"] == "p_a_sd1" and tuple(unseen) != P_A_UNSEEN_SLICES:
        raise AnalysisError(f"P-A unseen slices {unseen} differ from section 3 "
                            f"{list(P_A_UNSEEN_SLICES)}")
    if fitted["protocol"] == "p_a_sd1" and len(m2_slices) != P_A_M2_SLICES:
        raise AnalysisError(f"P-A M2 has {len(m2_slices)} slices, expected {P_A_M2_SLICES}")
    log.info("gate slices: %d for M1, %d for M2", len(unseen), len(m2_slices))

    # ---- per-slice metrics ------------------------------------------------------------
    hashes = load_panel_hashes(args.panel_hashes) if args.panel_hashes else {}
    metric_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for name, spec in slices.items():
        members = spec["fakes"] + spec["reals"]
        subset = [rows[k] for k in members]
        groups = exact_duplicate_groups(subset, hashes)
        caveat = ""
        if groups:
            caveat = (f"{len(groups)} byte-identical group(s), "
                      f"{sum(len(g) for g in groups)} of {len(subset)} rows; used as "
                      "distributed, so those images are weighted twice")
        variants = [(members, caveat, metric_rows)]
        if groups:
            keep = {r["image_id"] for r in deduplicate(subset, groups)}
            reduced = [k for k in members if rows[k]["image_id"] in keep]
            variants.append((reduced, f"each of {len(groups)} duplicate group(s) counted "
                             f"once: {len(reduced)} of {len(members)} rows retained",
                             sensitivity_rows))
        for chosen, note, target in variants:
            index = np.asarray(chosen)
            for stream in ALL_STREAMS:
                result = slice_metrics(labels[index], scores[stream][index],
                                       thresholds[stream], stream)
                target.append({"slice": name, "stream": stream,
                               **{k: _cell(v) for k, v in result.items()},
                               "in_gate": "M1+M2" if name in m2_slices else
                               "M1" if name in unseen else "",
                               "trivial_baseline": baseline_for(name), "caveat": note})
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(metric_rows, out_dir / "metrics.csv", METRIC_COLUMNS)
    if sensitivity_rows:
        write_manifest(sensitivity_rows, out_dir / "metrics_deduplicated.csv", METRIC_COLUMNS)

    # ---- the gate ----------------------------------------------------------------------
    boot = macro_bootstrap(gate_slices(rows, slices, unseen), scores,
                           {s: thresholds[s]["fpr_5"] for s in ALL_STREAMS},
                           resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    macro = {stream: {metric: {**percentile_interval(boot[metric][stream]),
                               "points": 100.0 * float(boot[metric][stream][0])}
                      for metric in ("M1", "M2")}
             for stream in ALL_STREAMS}
    governing = gate_result(boot, GOVERNING_FUSION, SINGLE_STREAMS)
    val_auc = {s: fitted["streams"][s]["val_auc"] for s in SINGLE_STREAMS}
    best_val = max(SINGLE_STREAMS, key=lambda s: (val_auc[s], -SINGLE_STREAMS.index(s)))
    ties = sorted(s for s in SINGLE_STREAMS if val_auc[s] == val_auc[best_val])
    analysis_a = gate_result(boot, "f_stack", best_val)
    analysis_a.update({"best_single_fixed_on_validation": best_val,
                       "validation_auc": val_auc, "validation_ties": ties,
                       "role": "SECONDARY (Amendment 2): reported unchanged, no gate role"})
    gate = {
        "run_id": rid, "protocol": fitted["protocol"], "seed": fitted["seed"],
        "synthetic": bool(args.synthetic),
        "criterion": (f"Amendment 2: GO if F-mean(A2) minus the maximum over the five "
                      f"single heads is >= {GATE_MARGIN_POINTS} points on M1 or M2 and the "
                      f"{int(GATE_CONFIDENCE * 100)}% paired bootstrap CI excludes 0"),
        "preregistration_sha256": file_sha256(args.preregistration),
        "scores_sha256": file_sha256(args.scores),
        "heads_report_sha256": file_sha256(args.heads),
        "gate_slices_m1": unseen, "gate_slices_m2": m2_slices,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                      "scheme": "fakes within slice, reals within view pool, luma rows "
                                "with their parent; same draw for every stream"},
        "governing": governing,
        "analysis_a": analysis_a,
        "macro_per_stream": macro,
        "note": ("A NO-GO is a result: the thesis then studies the best single stream plus "
                 "calibration for deployment. The criterion was fixed before any panel "
                 "score was produced; it is reported unchanged either way."),
    }

    # ---- complementarity ---------------------------------------------------------------
    complementarity: dict[str, Any] = {"thresholds": "validation reals, 5% FPR",
                                       "by_slice": {}, "synthetic": bool(args.synthetic)}
    for name, spec in slices.items():
        index = np.asarray(spec["fakes"] + spec["reals"])
        y = labels[index]
        single = {s: scores[s][index] for s in SINGLE_STREAMS}
        complementarity["by_slice"][name] = {
            "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
            "in_gate": name in unseen,
            "pairs": pairwise(y, single, {s: thresholds[s]["fpr_5"] for s in SINGLE_STREAMS}),
            "f_oracle": oracle_rows(y, single, thresholds),
        }
    pooled = np.unique(np.concatenate([np.asarray(slices[n]["fakes"] + slices[n]["reals"])
                                       for n in unseen]))
    y = labels[pooled]
    single = {s: scores[s][pooled] for s in SINGLE_STREAMS}
    complementarity["pooled_gate_slices"] = {
        "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "pairs": pairwise(y, single, {s: thresholds[s]["fpr_5"] for s in SINGLE_STREAMS}),
        "f_oracle": oracle_rows(y, single, thresholds),
    }

    write_json(out_dir / "complementarity.json", complementarity)
    write_json(out_dir / "go_no_go.json", gate)
    write_json(out_dir / "headline.json",
               {"run_id": rid, "protocol": fitted["protocol"], "synthetic": bool(args.synthetic),
                "macro_points": {s: {m: macro[s][m]["points"] for m in ("M1", "M2")}
                                 for s in ALL_STREAMS}})

    log.info("=" * 72)
    log.info("%s VERDICT (governing, Amendment 2): %s%s", rid, governing["verdict"],
             f" on {governing['passes_on']}" if governing["passes_on"] else "")
    for metric in ("M1", "M2"):
        g, a = governing[metric], analysis_a[metric]
        log.info("  %s: F-mean(A2) %.4f vs max single %.4f (%s) -> %+.2f [%+.2f, %+.2f]",
                 metric, g["fusion_macro"], g["best_single_macro"],
                 g["best_single"]["max_on_data"], g["difference_points"], *g["ci_points"])
        log.info("  %s (A, secondary): F-stack %.4f vs %s %.4f -> %+.2f [%+.2f, %+.2f]",
                 metric, a["fusion_macro"], best_val, a["best_single_macro"],
                 a["difference_points"], *a["ci_points"])
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
