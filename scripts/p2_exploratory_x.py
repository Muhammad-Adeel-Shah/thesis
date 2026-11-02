"""EXPLORATORY (post-verdict) analyses X1-X3 for the supervisor report. CPU only.

Nothing here changes the P2 verdict, retrains a P2 model or rewrites a P2 output. It
reads the committed panel scores and validation reports and the local caches, and
writes to ``reports/exploratory_x/<analysis>/`` (CSV + LaTeX booktabs + JSON) and the
figures to ``--figure-dir`` (outside the repo; CLAUDE.md section 2).

General rules (user instruction 2026-10-06):

* every row of a table uses the same images and slices (one scores file per run, rows
  selected by id; the same slice definitions as Session E);
* metrics: AUC (primary), AP, and TPR at 5% FPR with the threshold set on **the slice's
  own reals** ("slice-calibrated", I-064), plus the real-image FPR at the validation
  threshold as a separate column;
* 95% bootstrap CIs, 1,000 resamples, seed 2026 -- fakes within slice, reals within the
  view's pool, a luma row with its parent; the slice-calibrated threshold is re-fitted
  inside every replicate; every gain over the best single module is paired and
  re-maximised per replicate;
* runs: P-A seed 0, P-A seed 1, 4ch_only, full_bank.

Analyses:

* **X1** -- the generation-only DINOv3-B probe of ``x1_gen_only_probe.py``, in this
  format, beside S2 for all four runs.
* **X2** -- module ablation: single modules, references, equal-weight pairs on the
  Amendment 2 robust-z scheme, F-mean(A2), F-stack, and F-oracle (an upper bound, not a
  method): macro, per slice, complementarity per pair, and who wins where.
* **X3a** -- training-free min-LPIPS with sd15 only, the SD1 family, or every VAE member.
* **X3b** -- S3-Set and S3-LR retrained on features from k decoders (k = 1, 2, 4, 6, 8,
  12; two member orders fixed in advance), training images fixed to full_bank's.
* **X3c** -- S2 of the four runs on the same panel slices.

Usage:
    python scripts/p2_exploratory_x.py all --inputs F:/thesis-2/kaggle_raw ...
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.complementarity import (  # noqa: E402
    GateSlice,
    error_contingency,
    jaccard_of_misses,
    mcnemar,
)
from dbp.eval.join import join_streams  # noqa: E402
from dbp.eval.metrics import average_precision, roc_auc  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402
from dbp.streams.s3_probe import (  # noqa: E402
    AEROBLADE_EXCLUDED_KINDS,
    FEATURE_NAMES,
    build_set_head,
    standardise,
    standardiser,
)
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

LABEL = "EXPLORATORY (post-verdict)"
RUNS = ("p_a_sd1_seed0", "p_a_sd1_seed1", "4ch_only_seed0", "full_bank_seed0")
RUN_NAMES = {"p_a_sd1_seed0": "P-A s0", "p_a_sd1_seed1": "P-A s1",
             "4ch_only_seed0": "4ch_only", "full_bank_seed0": "full_bank"}
MODULES = ("s1", "s2", "s3_set")
REFERENCES = ("s3_lr", "aeroblade")
PAIRS = (("s1", "s2"), ("s1", "s3_set"), ("s2", "s3_set"))
PAIR_NAMES = tuple(f"{a}+{b}" for a, b in PAIRS)
ORACLE = "f_oracle"
X2_SCORES = (*MODULES, *REFERENCES, *PAIR_NAMES, "f_mean_rz", "f_stack", ORACLE)
DISPLAY = {"s1": "S1", "s2": "S2", "s3_set": "S3-Set", "s3_lr": "S3-LR",
           "aeroblade": "AEROBLADE", "s1+s2": "S1+S2", "s1+s3_set": "S1+S3-Set",
           "s2+s3_set": "S2+S3-Set", "f_mean_rz": "F-mean(A2)", "f_stack": "F-stack (A)",
           ORACLE: "F-oracle (UPPER BOUND, not a method)"}
RESAMPLES = 1000
SEED = 2026
TARGET_FPR = 0.05

#: X3b member orders, fixed before any X3b number existed. Both start at sd15 and add one
#: new lineage per step until all eight lineages are in (k = 8), then the remaining
#: members. "near" adds 4-channel latents first; "far" the reverse lineage order.
X3B_FILL = ("sd21", "sd_ft_ema", "sdxl_fp16fix", "flux")
X3B_ORDERS = {
    "near_first": ("sd15", "sdxl", "movq_k22", "sd35", "qwen_image", "flux2_klein", "dcae",
                   "realesrgan_x4", *X3B_FILL),
    "far_first": ("sd15", "realesrgan_x4", "dcae", "flux2_klein", "qwen_image", "sd35",
                  "movq_k22", "sdxl", *X3B_FILL),
}
X3B_KS = (1, 2, 4, 6, 8, 12)
SET_HEAD_EPOCHS = 200
SET_HEAD_LR = 1e-3


def _load_script(name: str, filename: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, THESIS_DBP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_script("p2_analyze_x", "p2_analyze.py")


class ExploratoryError(RuntimeError):
    """A hard check failed."""


# ===================================================================== metrics ===


def slice_thresholds(reals: np.ndarray) -> np.ndarray:
    """Per column, :func:`calibrate_on_reals`'s 5% threshold, vectorised.

    The lowest real score ``s`` with at most ``floor(0.05 n)`` reals at or above it: the
    smallest real value strictly above the ``(n - k)``-th smallest (``k`` = budget), or
    max + 1 if no real value qualifies.
    """
    values = np.sort(np.asarray(reals, dtype=float), axis=0)
    n = values.shape[0]
    budget = int(np.floor(TARGET_FPR * n + 1e-12))
    if budget >= n:
        return values[0] - 1.0
    pivot = values[n - budget - 1]
    out = np.empty(values.shape[1])
    for column in range(values.shape[1]):
        above = values[:, column][values[:, column] > pivot[column]]
        out[column] = above[0] if above.size else values[-1, column] + 1.0
    return out


def auc_columns(reals: np.ndarray, fakes: np.ndarray) -> np.ndarray:
    """AUC per column (Mann-Whitney with average ranks)."""
    from scipy.stats import rankdata

    n_r, n_f = len(reals), len(fakes)
    ranks = rankdata(np.vstack([reals, fakes]), axis=0)
    return (ranks[n_r:].sum(axis=0) - n_f * (n_f + 1) / 2) / (n_f * n_r)


def ap_columns(reals: np.ndarray, fakes: np.ndarray) -> np.ndarray:
    """Average precision per column."""
    y = np.r_[np.zeros(len(reals)), np.ones(len(fakes))]
    both = np.vstack([reals, fakes])
    return np.array([average_precision(y, both[:, k]) for k in range(both.shape[1])])


def slice_metrics(reals: np.ndarray | None, fakes: np.ndarray, with_ap: bool) -> dict:
    """AUC, AP and slice-calibrated TPR per column; None without reals."""
    if reals is None or len(reals) == 0:
        return {}
    threshold = slice_thresholds(reals)
    out = {"auc": auc_columns(reals, fakes),
           "tpr_slice": (fakes >= threshold).mean(axis=0)}
    if with_ap:
        out["ap"] = ap_columns(reals, fakes)
    return out


def draws(slices: Sequence[GateSlice], resamples: int = RESAMPLES,
          seed: int = SEED) -> Iterator[list[tuple[np.ndarray, np.ndarray | None]]]:
    """Per replicate (0 = the data), per slice: ``(fake rows, real rows)``.

    The resampling scheme of :func:`dbp.eval.complementarity.macro_bootstrap`: every
    member of a group holds the same units and follows one draw.
    """
    members: dict[str, list[tuple[str, np.ndarray, tuple[str, ...]]]] = {}
    for item in slices:
        members.setdefault(item.fake_group, []).append(
            (f"{item.name}:f", item.fakes, item.fake_units))
        if item.reals is not None:
            members.setdefault(item.real_group, []).append(
                (f"{item.name}:r", item.reals, item.real_units))
    positions, sizes = {}, {}
    for group, entries in sorted(members.items()):
        reference = None
        for key, rows, units in entries:
            if len(set(units)) != len(units):
                raise ExploratoryError(f"{group}: duplicate unit in {key}")
            order = sorted(units)
            if reference is None:
                reference = order
            elif order != reference:
                raise ExploratoryError(f"{group}: {key} covers different units")
            index = {u: k for k, u in enumerate(units)}
            positions[key] = np.asarray(rows)[[index[u] for u in reference]]
        sizes[group] = len(reference)
    rng = np.random.default_rng(seed)
    for replicate in range(resamples + 1):
        pick = ({g: np.arange(n) for g, n in sizes.items()} if replicate == 0 else
                {g: rng.integers(0, n, n) for g, n in sorted(sizes.items())})
        yield [(positions[f"{s.name}:f"][pick[s.fake_group]],
                positions[f"{s.name}:r"][pick[s.real_group]] if s.reals is not None else None)
               for s in slices]


def bootstrap(slices: Sequence[GateSlice], matrix: np.ndarray, with_ap: bool = False,
              resamples: int = RESAMPLES) -> dict[str, np.ndarray]:
    """Metric -> array ``(replicates + 1, n_slices, n_scores)``; NaN where undefined."""
    out: dict[str, list] = {"auc": [], "tpr_slice": [], **({"ap": []} if with_ap else {})}
    for replicate in draws(slices, resamples):
        per = {k: [] for k in out}
        for fakes, reals in replicate:
            result = slice_metrics(None if reals is None else matrix[reals], matrix[fakes],
                                   with_ap)
            for key in out:
                per[key].append(result.get(key, np.full(matrix.shape[1], np.nan)))
        for key in out:
            out[key].append(per[key])
    return {k: np.array(v) for k, v in out.items()}


def interval(values: np.ndarray) -> tuple[float, float, float]:
    """``(point, lo, hi)`` from replicate 0 and the 2.5/97.5 percentiles of the rest."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values[0]):
        return (float("nan"),) * 3
    lo, hi = np.percentile(values[1:], [2.5, 97.5])
    return float(values[0]), float(lo), float(hi)


def single_slice(name: str, fakes: Sequence[int], reals: Sequence[int] | None,
                 units: Sequence[str]) -> GateSlice:
    """A slice resampled on its own: fakes within the slice, reals within the slice."""
    return GateSlice(name=name, fakes=np.asarray(fakes),
                     fake_units=tuple(units[k] for k in fakes), fake_group=f"f:{name}",
                     reals=None if not reals else np.asarray(reals),
                     real_units=None if not reals else tuple(units[k] for k in reals),
                     real_group=None if not reals else f"r:{name}")


# ====================================================================== output ===


def tex_escape(text: Any) -> str:
    """Escape LaTeX specials in a cell."""
    import re

    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\~{}", "^": r"\^{}",
               "†": r"\textsuperscript{\dag}"}
    return re.sub("|".join(re.escape(k) for k in special), lambda m: special[m.group(0)],
                  str(text))


def write_table(directory: Path, name: str, columns: Sequence[str],
                rows: Sequence[dict[str, Any]], caption: str,
                tex_columns: Sequence[str] | None = None) -> None:
    """``<name>.csv`` (every column) and ``<name>.tex`` (booktabs, ``tex_columns``)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.csv"
    with path.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {LABEL}: {caption}\n")
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows({c: row.get(c, "") for c in columns} for row in rows)
    path.with_suffix(".tmp").replace(path)
    shown = list(tex_columns or columns)
    lines = [
        f"% {LABEL} -- generated by scripts/p2_exploratory_x.py; needs booktabs",
        "\\begin{table}[htbp]", "\\centering", "\\scriptsize",
        f"\\caption{{\\textbf{{{tex_escape(LABEL)}.}} {tex_escape(caption)}}}",
        f"\\label{{tab:{name.replace('_', '-')}}}",
        "\\begin{tabular}{" + "l" * min(2, len(shown)) + "r" * (len(shown) - min(2, len(shown)))
        + "}", "\\toprule",
        " & ".join(tex_escape(c) for c in shown) + " \\\\", "\\midrule"]
    lines += [" & ".join(tex_escape(row.get(c, "")) for c in shown) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    tex = directory / f"{name}.tex"
    tex.with_suffix(".tmp").write_text("\n".join(lines), encoding="utf-8")
    tex.with_suffix(".tmp").replace(tex)


def fmt(point: float, lo: float | None = None, hi: float | None = None, digits: int = 3) -> str:
    """``0.803 [0.790, 0.816]``, or ``n/a``."""
    if point is None or not np.isfinite(point):
        return "n/a"
    if lo is None:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def write_json(path: Path, payload: Any) -> None:
    """Atomic JSON write, labelled."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_text(json.dumps({"label": LABEL, **payload}, indent=2,
                                                   default=float) + "\n", encoding="utf-8")
    path.with_suffix(".tmp").replace(path)


# ================================================================= panel data ===


class Run:
    """One run's panel rows, scores (incl. derived ones), slices and thresholds."""

    def __init__(self, args: argparse.Namespace, rid: str) -> None:
        self.rid = rid
        self.rows = analyze.read_scores(args.panel_dir / "scores" / f"scores_{rid}.csv")
        base = analyze.validate_rows(self.rows)
        self.ids = [r["image_id"] for r in self.rows]
        self.units = [r["parent_id"] for r in self.rows]
        self.labels = np.array([1 if r["label"] == "fake" else 0 for r in self.rows])
        self.heads = json.loads((args.validation_dir / f"p2_heads_{rid}.json")
                                .read_text(encoding="utf-8"))
        self.gate = json.loads((args.panel_dir / "analysis" / rid / "go_no_go.json")
                               .read_text(encoding="utf-8"))
        robust = self.heads["fusions"]["f_mean_rz"]
        if tuple(robust["streams"]) != MODULES:
            raise ExploratoryError(f"{rid}: F-mean(A2) streams {robust['streams']}")
        self.robust = robust
        z = {name: self.z(name, base[name]) for name in MODULES}
        recomputed = np.mean([z[m] for m in MODULES], axis=0)
        gap = float(np.max(np.abs(recomputed - base["f_mean_rz"])))
        if gap > 1e-4:
            raise ExploratoryError(f"{rid}: F-mean(A2) not reproduced from S1/S2/S3-Set "
                                   f"(max gap {gap})")
        self.scores = dict(base)
        for (a, b), name in zip(PAIRS, PAIR_NAMES, strict=True):
            self.scores[name] = (z[a] + z[b]) / 2.0
        stacked = np.vstack([z[m] for m in MODULES])
        # Upper bound only: the label picks the most favourable module per image.
        self.scores[ORACLE] = np.where(self.labels == 1, stacked.max(axis=0),
                                       stacked.min(axis=0))
        self.val_thresholds = {s: float(self.heads["calibrations"][s]["fpr_5"]["threshold"])
                               for s in (*MODULES, *REFERENCES, "f_mean_rz", "f_stack")}
        self.slices = analyze.build_slices(self.rows)
        self.add_content_type_slices()
        self.unseen = list(self.gate["gate_slices_m2"])

    def z(self, name: str, values: np.ndarray) -> np.ndarray:
        """Amendment 2's clipped robust z for a module."""
        k = self.robust["streams"].index(name)
        return np.clip((values - self.robust["medians"][k]) / self.robust["scales"][k],
                       -self.robust["clip"], self.robust["clip"])

    def add_content_type_slices(self) -> None:
        """E3 by content type: that type's fakes vs that type's reals."""
        types = sorted({r["content_type"] for r in self.rows if r["view"] == "e3_matched"})
        for kind in types:
            rows = [k for k, r in enumerate(self.rows)
                    if r["view"] == "e3_matched" and r["content_type"] == kind]
            self.slices[f"e3_matched/content:{kind}"] = {
                "view": "e3_matched", "in_bank": "",
                "fakes": [k for k in rows if self.labels[k] == 1],
                "reals": [k for k in rows if self.labels[k] == 0]}

    def matrix(self, names: Sequence[str]) -> np.ndarray:
        """Scores as ``(rows, len(names))``."""
        return np.column_stack([self.scores[n] for n in names])

    def gate_slices(self, names: Sequence[str]) -> list[GateSlice]:
        """Joint resampling structures (shared real pools), as in Session E."""
        return analyze.gate_slices(self.rows, self.slices, names)

    def own_slice(self, name: str) -> GateSlice:
        """The slice resampled on its own."""
        spec = self.slices[name]
        return single_slice(name, spec["fakes"], spec["reals"], self.units)


def validation_real_scores(args: argparse.Namespace, rid: str, members: list[str],
                           val_real_ids: list[str]) -> dict[str, np.ndarray]:
    """S1, S2 and S3-Set on the 491 validation reals, re-scored from the frozen heads.

    Needed only for the pairs' validation thresholds; checked against the thresholds in
    the committed heads report before use.
    """
    import torch

    with np.load(args.heads_dir / f"s1_probe_{rid}.npz", allow_pickle=False) as blob:
        coef, intercept = blob["coef"].ravel(), float(blob["intercept"].ravel()[0])
    with np.load(args.s1_cache / "p2_s1_dinov3-vitb16_train.npz", allow_pickle=False) as blob:
        table = ([str(v) for v in blob["image_ids"]], blob["features"])
    s1x = join_streams(val_real_ids, {"s1": table}, supersets=("s1",), what="val S1")["s1"]
    s1 = s1x.astype(np.float32).astype(np.float64) @ coef.astype(np.float64) + intercept
    rows = analyze.read_scores(args.s2_dir / rid / "scores" / "val.csv")
    s2 = join_streams(val_real_ids, {"s2": ([r["image_id"] for r in rows],
                                            np.array([float(r["score"]) for r in rows]))},
                      supersets=("s2",), what="val S2")["s2"]
    caches = {}
    for member in members:
        with np.load(args.probe_dir / f"p2_probe_{member}.npz", allow_pickle=False) as blob:
            caches[member] = ([str(v) for v in blob["image_ids"]], blob["features"])
    joined = join_streams(val_real_ids, caches, supersets=members, what="val S3")
    matrix = np.stack([joined[m] for m in members], axis=1).astype(np.float32)
    with np.load(args.heads_dir / f"s3_lr_{rid}.npz", allow_pickle=False) as blob:
        mean, scale = blob["mean"], blob["scale"]
    flat = standardise(matrix.reshape(len(matrix), -1), mean, scale)
    head = build_set_head()
    head.load_state_dict(torch.load(args.heads_dir / f"s3_set_{rid}.pt", weights_only=True))
    head.eval()
    with torch.no_grad():
        s3 = head(torch.from_numpy(flat.reshape(matrix.shape))).numpy().astype(np.float64)
    return {"s1": s1, "s2": s2, "s3_set": s3}


# ========================================================================== X2 ===


def run_x2(args: argparse.Namespace, log: Any, runs: dict[str, Run], context: dict) -> None:
    """Module ablation."""
    out = args.out_dir / "x2"
    macro_rows, slice_rows, comp_rows, win_rows, tex_slices = [], [], [], [], []
    payload: dict[str, Any] = {"runs": {}}
    for rid, run in runs.items():
        # Validation thresholds for the pairs, from the 491 validation reals.
        val = validation_real_scores(args, rid, context["members"], context["val_reals"])
        for module in MODULES:
            again = calibrate_on_reals(val[module], TARGET_FPR, "x").threshold
            if abs(again - run.val_thresholds[module]) > 1e-3:
                raise ExploratoryError(f"{rid} {module}: validation threshold not reproduced "
                                       f"({again} vs {run.val_thresholds[module]})")
        zval = {m: run.z(m, val[m]) for m in MODULES}
        for (a, b), name in zip(PAIRS, PAIR_NAMES, strict=True):
            run.val_thresholds[name] = calibrate_on_reals((zval[a] + zval[b]) / 2,
                                                          TARGET_FPR, "x").threshold
        names = list(X2_SCORES)
        matrix = run.matrix(names)

        # ---- (a) macro over unseen slices with reals ---------------------------------
        boot = bootstrap(run.gate_slices(run.unseen), matrix, with_ap=True)
        macro = {k: np.nanmean(v, axis=1) for k, v in boot.items()}       # (R+1, S)
        module_idx = [names.index(m) for m in MODULES]
        best = {k: macro[k][:, module_idx].max(axis=1) for k in ("auc", "tpr_slice")}
        payload["runs"][rid] = {"unseen_slices": run.unseen, "macro": {}}
        for k, name in enumerate(names):
            row = {"run": RUN_NAMES[rid], "score": DISPLAY[name],
                   "role": ("upper bound -- NOT a method" if name == ORACLE else
                            "reference" if name in REFERENCES else
                            "module" if name in MODULES else "fusion")}
            for metric in ("auc", "ap", "tpr_slice"):
                p, lo, hi = interval(macro[metric][:, k])
                row[f"{metric}"], row[f"{metric}_lo"], row[f"{metric}_hi"] = p, lo, hi
                row[f"{metric}_ci"] = fmt(p, lo, hi)
            for metric in ("auc", "tpr_slice"):
                if name in MODULES:
                    row[f"gain_{metric}"] = ""
                    continue
                p, lo, hi = interval(100 * (macro[metric][:, k] - best[metric]))
                row[f"gain_{metric}"] = fmt(p, lo, hi, 2)
            macro_rows.append(row)
            payload["runs"][rid]["macro"][name] = {c: row[c] for c in row}
        win_on_data = {k: MODULES[int(np.argmax(macro[k][0, module_idx]))]
                       for k in ("auc", "tpr_slice")}
        payload["runs"][rid]["best_single_module_on_data"] = win_on_data

        # ---- (b) per slice -------------------------------------------------------------
        ordered = sorted(run.slices, key=lambda s: (s.split("/")[0], s))
        for name in ordered:
            spec = run.slices[name]
            single = bootstrap([run.own_slice(name)], matrix)
            fakes, reals = spec["fakes"], spec["reals"]
            tex_row = {"run": RUN_NAMES[rid], "slice": name,
                       "n f/r": f"{len(fakes)}/{len(reals)}"}
            for k, score in enumerate(names):
                p, lo, hi = interval(single["auc"][:, 0, k])
                tp = interval(single["tpr_slice"][:, 0, k])
                thr = run.val_thresholds.get(score)
                fpr_val = (float(np.mean(matrix[reals, k] >= thr))
                           if reals and thr is not None else None)
                tpr_val = float(np.mean(matrix[fakes, k] >= thr)) if thr is not None else None
                slice_rows.append({
                    "run": RUN_NAMES[rid], "slice": name, "in_gate": name in run.unseen,
                    "n_fake": len(fakes), "n_real": len(reals), "score": DISPLAY[score],
                    "auc": p, "auc_lo": lo, "auc_hi": hi,
                    "ap": (float(average_precision(np.r_[np.zeros(len(reals)),
                                                         np.ones(len(fakes))],
                                                   np.r_[matrix[reals, k], matrix[fakes, k]]))
                           if reals else ""),
                    "tpr_slice": tp[0], "tpr_slice_lo": tp[1], "tpr_slice_hi": tp[2],
                    "fpr_reals_at_val_threshold": "" if fpr_val is None else fpr_val,
                    "tpr_at_val_threshold": "" if tpr_val is None else tpr_val})
                if score in (*MODULES, "f_mean_rz", "f_stack", ORACLE):
                    tex_row[DISPLAY[score].split(" ")[0]] = (fmt(p) if reals else
                                                             f"({fmt(tpr_val)}†)")
            tex_slices.append(tex_row)

            # ---- (d) who wins where -------------------------------------------------
            if reals:
                aucs = single["auc"][:, 0, :]
                point = aucs[0, module_idx]
                order = np.argsort(point)[::-1]
                winner, runner = MODULES[order[0]], MODULES[order[1]]
                others = [module_idx[i] for i in order[1:]]
                margin = interval(100 * (aucs[:, module_idx[order[0]]]
                                         - aucs[:, others].max(axis=1)))
                overall = names[int(np.nanargmax(np.where(
                    np.array([n == ORACLE for n in names]), -np.inf, aucs[0])))]
                win_rows.append({"run": RUN_NAMES[rid], "slice": name,
                                 "best_module": DISPLAY[winner], "best_auc": fmt(point[order[0]]),
                                 "runner_up": DISPLAY[runner],
                                 "margin_points": fmt(*margin, digits=2),
                                 "best_of_all_non_oracle": DISPLAY[overall]})

        # ---- (c) complementarity per pair, slice-calibrated thresholds ----------------
        summed = {pair: {"both_right": 0, "a_only": 0, "b_only": 0, "both_wrong": 0, "n": 0}
                  for pair in PAIRS}
        for name in run.unseen:
            spec = run.slices[name]
            rows = np.asarray(spec["fakes"] + spec["reals"])
            y = run.labels[rows]
            correct = {}
            for module in MODULES:
                values = run.scores[module]
                threshold = slice_thresholds(values[spec["reals"]][:, None])[0]
                correct[module] = (values[rows] >= threshold).astype(int) == y
            for a, b in PAIRS:
                table = error_contingency(correct[a], correct[b])
                for key in summed[(a, b)]:
                    summed[(a, b)][key] += table[key]
                comp_rows.append(complementarity_row(rid, name, a, b, table, correct))
        for a, b in PAIRS:
            comp_rows.append(complementarity_row(rid, "SUM over unseen slices", a, b,
                                                 summed[(a, b)], None))
        log.info("X2 %s done", rid)

    macro_cols = ["run", "score", "role", "auc", "auc_lo", "auc_hi", "ap", "ap_lo", "ap_hi",
                  "tpr_slice", "tpr_slice_lo", "tpr_slice_hi", "gain_auc", "gain_tpr_slice",
                  "auc_ci", "ap_ci", "tpr_slice_ci"]
    write_table(out, "x2a_macro", macro_cols, macro_rows,
                "X2a. Module ablation, macro over each run's unseen panel slices with reals "
                "(P-A 7, 4ch_only 6, full_bank 4). AUC, AP and TPR@5%FPR with the threshold "
                "set on each slice's own reals; 95% bootstrap CI; gain in points vs the best "
                "single module (S1, S2, S3-Set), paired and re-maximised per replicate. "
                "F-oracle uses the labels: an upper bound, not a method.",
                ["run", "score", "auc_ci", "tpr_slice_ci", "ap_ci", "gain_auc", "gain_tpr_slice"])
    write_table(out, "x2b_per_slice", ["run", "slice", "in_gate", "n_fake", "n_real", "score",
                                       "auc", "auc_lo", "auc_hi", "ap", "tpr_slice",
                                       "tpr_slice_lo", "tpr_slice_hi",
                                       "fpr_reals_at_val_threshold", "tpr_at_val_threshold"],
                slice_rows, "X2b. Per-slice metrics, every score.")
    shown = ["run", "slice", "n f/r", "S1", "S2", "S3-Set", "F-mean(A2)", "F-stack", "F-oracle"]
    write_table(out, "x2b_per_slice_auc", shown, tex_slices,
                "X2b. Per-slice AUC. E8 has no reals: shown is the detection rate at the "
                "validation threshold (†), whose real FPR is unknown (I-064). F-oracle: upper "
                "bound, not a method.")
    write_table(out, "x2c_complementarity",
                ["run", "slice", "pair", "double_fault", "jaccard_of_misses", "mcnemar_test",
                 "mcnemar_p", "a_only_right", "b_only_right", "oracle_headroom"], comp_rows,
                "X2c. Complementarity per module pair at slice-calibrated 5% FPR thresholds. "
                "SUM rows add the per-slice 2x2 tables (a real shared by two slices counts in "
                "each).")
    write_table(out, "x2d_who_wins", ["run", "slice", "best_module", "best_auc", "runner_up",
                                      "margin_points", "best_of_all_non_oracle"], win_rows,
                "X2d. Best single module per slice by AUC, and its margin in points over the "
                "better of the other two (paired 95% CI).")
    write_json(out / "x2.json", payload)


def complementarity_row(rid: str, name: str, a: str, b: str, table: dict,
                        correct: dict | None) -> dict[str, Any]:
    """One row of X2c."""
    test = mcnemar(table)
    acc_a = (table["both_right"] + table["a_only"]) / table["n"]
    acc_b = (table["both_right"] + table["b_only"]) / table["n"]
    oracle = (table["n"] - table["both_wrong"]) / table["n"]
    jac = jaccard_of_misses(table)
    return {"run": RUN_NAMES[rid], "slice": name, "pair": f"{DISPLAY[a]} / {DISPLAY[b]}",
            "double_fault": f"{table['both_wrong'] / table['n']:.3f}",
            "jaccard_of_misses": "undefined" if jac is None else f"{jac:.3f}",
            "mcnemar_test": test["test"],
            "mcnemar_p": "n/a" if test["p_value"] is None else f"{test['p_value']:.2g}",
            "a_only_right": table["a_only"], "b_only_right": table["b_only"],
            "oracle_headroom": f"{oracle - max(acc_a, acc_b):.3f}"}


# ========================================================================== X1 ===


def run_x1(args: argparse.Namespace, log: Any, runs: dict[str, Run], context: dict) -> None:
    """The generation-only probe in the X-format, S2 of every run beside it."""
    x1 = _load_script("x1_probe_x", "x1_gen_only_probe.py")
    s1_rows = analyze.read_scores(args.s1_list)
    group = {r["row_id"]: r["group_id"] for r in analyze.read_scores(args.splits)}
    split_group = {r["image_id"]: group[r["image_id"].rsplit("__", 1)[0]
                                        if "__" in r["image_id"] else r["image_id"]]
                   for r in s1_rows}
    split = x1.build_split(s1_rows, split_group)
    labels = {r["image_id"]: 0 if r["label"] == "real" else 1 for r in s1_rows}
    with np.load(args.s1_cache / "p2_s1_dinov3-vitb16_train.npz", allow_pickle=False) as blob:
        cache = ([str(v) for v in blob["image_ids"]], blob["features"])
    with np.load(args.s1_cache / "p2_s1_dinov3-vitb16_panel.npz", allow_pickle=False) as blob:
        panel_cache = ([str(v) for v in blob["image_ids"]], blob["features"])
    from dbp.streams.s1_semantic import fit_probe

    def feats(ids, source):
        return join_streams(ids, {"f": source}, supersets=("f",), what="X1")["f"].astype(np.float32)

    probe = fit_probe(feats(split["fit"], cache), [labels[i] for i in split["fit"]],
                      feats(split["tune"], cache), [labels[i] for i in split["tune"]], seed=0)
    model = probe["model"]
    reals = split["test_reals"]
    s2_val = {}
    for rid in RUNS:
        rows = analyze.read_scores(args.s2_dir / rid / "scores" / "val.csv")
        s2_val[rid] = ([r["image_id"] for r in rows], np.array([float(r["score"]) for r in rows]))
    heads = ["probe_gen_only", *[f"s2_{rid}" for rid in RUNS]]
    head_name = {"probe_gen_only": "gen-only probe",
                 **{f"s2_{r}": f"S2 {RUN_NAMES[r]}" for r in RUNS}}

    def val_scores(ids: list[str]) -> np.ndarray:
        columns = [model.decision_function(feats(ids, cache))]
        for rid in RUNS:
            columns.append(join_streams(ids, {"s2": s2_val[rid]}, supersets=("s2",),
                                        what="X1 S2")["s2"])
        return np.column_stack(columns)

    real_matrix = val_scores(reals)
    val_thr = slice_thresholds(real_matrix)
    rows_out, tex_rows = [], []
    tests = [("gen_matched (SD1.5 full)", split["test_gen"], None)]
    tests += [(f"recon {m}", [p[1] for p in pairs], [p[0] for p in pairs])
              for m, pairs in split["test_recon"].items()]
    for name, fake_ids, pair_reals in tests:
        fake_matrix = val_scores(fake_ids)
        matrix = np.vstack([real_matrix, fake_matrix])
        n_r = len(reals)
        units = list(reals) + (list(pair_reals) if pair_reals else list(fake_ids))
        if pair_reals:
            if pair_reals != reals:
                raise ExploratoryError(f"{name}: pairs not aligned with the val reals")
            item = GateSlice(name, np.arange(n_r, 2 * n_r), tuple(pair_reals), "pair",
                             np.arange(n_r), tuple(reals), "pair")
        else:
            item = single_slice(name, list(range(n_r, len(matrix))), list(range(n_r)), units)
        boot = bootstrap([item], matrix, with_ap=True)
        tex = {"test": name}
        for k, head in enumerate(heads):
            a, t, p = (interval(boot[m][:, 0, k]) for m in ("auc", "tpr_slice", "ap"))
            rows_out.append({"test": name, "head": head_name[head], "n_fake": len(fake_ids),
                             "n_real": n_r, "auc": fmt(*a), "ap": fmt(*p),
                             "tpr_slice": fmt(*t),
                             "note": "slice = the 491 val reals (= validation threshold)"})
            tex[head_name[head]] = fmt(a[0])
        tex_rows.append(tex)
    # E7 on the panel: slice-calibrated on E7's reals, and the FPR at the val threshold.
    run0 = runs["p_a_sd1_seed0"]
    e7 = [n for n in run0.slices if n.startswith("e7_dda_coco/")]
    e7_rows = sorted({k for n in e7 for k in run0.slices[n]["fakes"] + run0.slices[n]["reals"]})
    e7_ids = [run0.ids[k] for k in e7_rows]
    probe_e7 = np.full(len(run0.ids), np.nan)
    probe_e7[e7_rows] = model.decision_function(feats(e7_ids, panel_cache))
    columns = [probe_e7] + [runs[r].scores["s2"] if runs[r].ids == run0.ids else
                            join_streams(run0.ids, {"s2": (runs[r].ids, runs[r].scores["s2"])},
                                         what="X1 E7 S2")["s2"] for r in RUNS]
    matrix = np.column_stack(columns)
    for name in sorted(e7):
        spec = run0.slices[name]
        boot = bootstrap([run0.own_slice(name)], matrix, with_ap=True)
        tex = {"test": f"E7 {name.split('/')[1]}"}
        for k, head in enumerate(heads):
            a, t, p = (interval(boot[m][:, 0, k]) for m in ("auc", "tpr_slice", "ap"))
            fpr = float(np.mean(matrix[spec["reals"], k] >= val_thr[k]))
            rows_out.append({"test": tex["test"], "head": head_name[head],
                             "n_fake": len(spec["fakes"]), "n_real": len(spec["reals"]),
                             "auc": fmt(*a), "ap": fmt(*p), "tpr_slice": fmt(*t),
                             "fpr_reals_at_val_threshold": f"{fpr:.3f}",
                             "note": "FLUX.1: chroma shortcut, upper bound" if "FLUX" in name
                             else ""})
            tex[head_name[head]] = fmt(a[0])
        tex_rows.append(tex)
    out = args.out_dir / "x1"
    write_table(out, "x1_results", ["test", "head", "n_fake", "n_real", "auc", "ap",
                                    "tpr_slice", "fpr_reals_at_val_threshold", "note"], rows_out,
                "X1. Generation-only DINOv3-B probe (trained on real COCO vs SD1.5 full "
                "generations only) vs S2, AUC / AP / slice-calibrated TPR@5%FPR, 95% CI.")
    write_table(out, "x1_auc", ["test", *[head_name[h] for h in heads]], tex_rows,
                "X1. AUC: generation-only frozen DINOv3-B probe vs S2 of each run. Val "
                "reconstructions are paired with their own real; E7 on the panel vs E7's reals. "
                "E7 FLUX.1 carries the chroma shortcut (upper bound).")
    write_json(out / "x1.json", {"probe_C": probe["C"], "n_fit": len(split["fit"]),
                                 "n_tune": len(split["tune"]), "rows": rows_out})
    log.info("X1 done (probe C=%s)", probe["C"])


# ========================================================================== X3 ===


def load_probe_caches(args: argparse.Namespace, members: list[str]) -> dict:
    """Member -> ``(ids, features)``."""
    caches = {}
    for member in members:
        with np.load(args.probe_dir / f"p2_probe_{member}.npz", allow_pickle=False) as blob:
            caches[member] = ([str(v) for v in blob["image_ids"]], blob["features"])
    return caches


def probe_matrix(caches: dict, ids: list[str], members: Sequence[str]) -> np.ndarray:
    """``(n, len(members), 20)`` S3 features joined by id."""
    joined = join_streams(ids, {m: caches[m] for m in members}, supersets=tuple(members),
                          what="X3 probe features")
    return np.stack([joined[m] for m in members], axis=1).astype(np.float32)


def run_x3a(args: argparse.Namespace, log: Any, runs: dict[str, Run], context: dict) -> None:
    """Training-free min-LPIPS: sd15 only, SD1 family, every VAE member."""
    bank = context["bank"]
    members = context["members"]
    kinds = {s.key: s.kind for s in bank.distinct_members()}
    lineage = {s.key: s.lineage for s in bank.distinct_members()}
    variants = {"sd15 only": ["sd15"],
                "SD1 family": [m for m in members if lineage[m] == lineage["sd15"]],
                "all VAE members": [m for m in members
                                    if kinds[m] not in AEROBLADE_EXCLUDED_KINDS]}
    caches = load_probe_caches(args, members)
    lp = FEATURE_NAMES.index("lpips_vgg")

    def scores(ids: list[str]) -> np.ndarray:
        feats = probe_matrix(caches, ids, members)[:, :, lp]
        return np.column_stack([-feats[:, [members.index(m) for m in v]].min(axis=1)
                                for v in variants.values()])

    val_rows = context["probe_val"]
    reals = [r["image_id"] for r in val_rows if r["label"] == "real"]
    rows_out, tex_rows, figure_data = [], [], {}
    real_matrix = scores(reals)
    for member in members:
        fakes = [r["image_id"] for r in val_rows if context["member_of"](r) == member]
        matrix = np.vstack([real_matrix, scores(fakes)])
        units = reals + fakes
        item = single_slice(f"val {member}", list(range(len(reals), len(units))),
                            list(range(len(reals))), units)
        boot = bootstrap([item], matrix, with_ap=True)
        tex = {"test": f"val reals vs {member} recon", "n f/r": f"{len(fakes)}/{len(reals)}"}
        figure_data[member] = {}
        for k, variant in enumerate(variants):
            a, t, p = (interval(boot[m][:, 0, k]) for m in ("auc", "tpr_slice", "ap"))
            rows_out.append({"test": tex["test"], "variant": variant, "n_fake": len(fakes),
                             "n_real": len(reals), "auc": fmt(*a), "ap": fmt(*p),
                             "tpr_slice": fmt(*t),
                             "decoder_in_variant": member in variants[variant]})
            tex[variant] = fmt(a[0]) + ("*" if member in variants[variant] else "")
            figure_data[member][variant] = a
        tex_rows.append(tex)
    run = runs["p_a_sd1_seed0"]
    panel = [n for n in run.slices if n.split("/")[0] in ("e7_dda_coco", "e7_dda_coco_luma",
                                                          "e4_wildrf_matched",
                                                          "e5_chameleon_pipeline")
             or (n.startswith("e3_matched/") and "content:" not in n)]
    panel = [n for n in sorted(panel) if run.slices[n]["reals"]]
    used = sorted({k for n in panel for k in run.slices[n]["fakes"] + run.slices[n]["reals"]})
    matrix = np.full((len(run.ids), len(variants)), np.nan)
    matrix[used] = scores([run.ids[k] for k in used])
    for name in panel:
        boot = bootstrap([run.own_slice(name)], matrix, with_ap=True)
        tex = {"test": name, "n f/r": f"{len(run.slices[name]['fakes'])}/"
                                      f"{len(run.slices[name]['reals'])}"}
        for k, variant in enumerate(variants):
            a, t, p = (interval(boot[m][:, 0, k]) for m in ("auc", "tpr_slice", "ap"))
            rows_out.append({"test": name, "variant": variant,
                             "n_fake": len(run.slices[name]["fakes"]),
                             "n_real": len(run.slices[name]["reals"]),
                             "auc": fmt(*a), "ap": fmt(*p), "tpr_slice": fmt(*t),
                             "decoder_in_variant": "",
                             "note": "FLUX.1 standard: chroma shortcut" if name.endswith(
                                 "e7_dda_coco/FLUX.1") else ""})
            tex[variant] = fmt(a[0])
        tex_rows.append(tex)
    out = args.out_dir / "x3a"
    write_table(out, "x3a_min_lpips", ["test", "variant", "n_fake", "n_real", "auc", "ap",
                                       "tpr_slice", "decoder_in_variant", "note"], rows_out,
                "X3a. Training-free min-LPIPS score with sd15 only, the SD1 family or all 11 "
                "VAE members; AUC / AP / slice-calibrated TPR@5%FPR with 95% CI.")
    write_table(out, "x3a_auc", ["test", "n f/r", *variants], tex_rows,
                "X3a. AUC of the training-free min-LPIPS score. * = the decoder is one of the "
                "variant's members. Validation: the probe list's val reconstructions (100-130 "
                "per decoder) vs the 491 val reals.")
    write_json(out / "x3a.json", {"variants": variants, "rows": rows_out})
    x3a_figure(figure_data, members, list(variants), args.figure_dir)
    log.info("X3a done")


def x3a_figure(data: dict, members: list[str], variants: list[str], directory: Path) -> None:
    """Bars per validation decoder: min-LPIPS with 1 vs the SD1 family vs all members."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.27
    for k, (variant, colour) in enumerate(zip(variants, ("#4c72b0", "#8172b3", "#55a868"),
                                              strict=True)):
        values = [data[m][variant][0] for m in members]
        errors = [[data[m][variant][0] - data[m][variant][1] for m in members],
                  [data[m][variant][2] - data[m][variant][0] for m in members]]
        ax.bar(np.arange(len(members)) + (k - 1) * width, values, width, yerr=errors,
               capsize=2, color=colour, label=f"min-LPIPS, {variant}")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(np.arange(len(members)), members, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUC: val reals vs reconstructions (95% CI)")
    ax.set_title(f"X3a ({LABEL}): training-free min-LPIPS, 1 decoder vs the bank")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"x3a_min_lpips_by_decoder.{suffix}", dpi=150)
    plt.close(fig)


def run_x3b(args: argparse.Namespace, log: Any, runs: dict[str, Run], context: dict) -> None:
    """S3-Set and S3-LR retrained on k decoders, training images fixed (full_bank)."""
    import torch
    from sklearn.linear_model import LogisticRegression

    fit = _load_script("p2_fit_heads_x", "p2_fit_heads.py")
    bank, members = context["bank"], context["members"]
    for order in X3B_ORDERS.values():
        if sorted(order) != sorted(members) or order[0] != "sd15":
            raise ExploratoryError(f"X3b order {order} is not the bank starting at sd15")
    lineage = {s.key: s.lineage for s in bank.distinct_members()}
    run_config = context["run_configs"].by_id("full_bank")
    allowed = set(run_config.active_members(bank)) | {"gen_matched"}
    s1_rows = analyze.read_scores(args.s1_list)
    reals, fakes = fit.s3_training_ids(s1_rows, allowed, "full_bank", 2026)
    train_ids = sorted(set(reals) | set(fakes))
    label_of = {r["image_id"]: 0 if r["label"] == "real" else 1 for r in s1_rows}
    y_train = np.array([label_of[i] for i in train_ids])
    val_rows = context["probe_val"]
    val_ids = [r["image_id"] for r in val_rows]
    y_val = np.array([0 if r["label"] == "real" else 1 for r in val_rows])
    groups = [context["member_of"](r) for r in val_rows]
    caches = load_probe_caches(args, members)
    x_train_all = probe_matrix(caches, train_ids, members)
    x_val_all = probe_matrix(caches, val_ids, members)
    run = runs["full_bank_seed0"]
    unseen = list(run.unseen)
    pa_slices = list(runs["p_a_sd1_seed0"].unseen)
    used = sorted({k for n in set(unseen) | set(pa_slices)
                   for k in run.slices[n]["fakes"] + run.slices[n]["reals"]})
    x_panel_used = probe_matrix(caches, [run.ids[k] for k in used], members)
    rows_out, curve = [], {}
    for order_name, order in X3B_ORDERS.items():
        for k in X3B_KS:
            chosen = list(order[:k])
            cols = [members.index(m) for m in chosen]
            xt = x_train_all[:, cols, :].reshape(len(train_ids), -1)
            mean, scale = standardiser(xt)
            xt_std = standardise(xt, mean, scale)
            xv_std = standardise(x_val_all[:, cols, :].reshape(len(val_ids), -1), mean, scale)
            xp_std = standardise(x_panel_used[:, cols, :].reshape(len(used), -1), mean, scale)
            lr = LogisticRegression(C=1.0, max_iter=2000, random_state=0).fit(xt_std, y_train)
            seed_everything(0)
            head = build_set_head()
            optimiser = torch.optim.AdamW(head.parameters(), lr=SET_HEAD_LR)
            loss_fn = torch.nn.BCEWithLogitsLoss()
            train_t = torch.from_numpy(xt_std.reshape(len(train_ids), k, -1))
            target = torch.from_numpy(y_train.astype(np.float32))
            for _ in range(SET_HEAD_EPOCHS):
                head.train()
                optimiser.zero_grad(set_to_none=True)
                loss_fn(head(train_t), target).backward()
                optimiser.step()
            head.eval()
            with torch.no_grad():
                set_val = head(torch.from_numpy(xv_std.reshape(len(val_ids), k, -1))).numpy()
                set_panel = head(torch.from_numpy(xp_std.reshape(len(used), k, -1))).numpy()
            heads = {"S3-LR": (lr.decision_function(xv_std), lr.decision_function(xp_std)),
                     "S3-Set": (set_val, set_panel)}
            for head_name, (val_scores, panel_used) in heads.items():
                per_decoder = {}
                for member in members:
                    mask = np.array([g in ("real", member) for g in groups])
                    per_decoder[member] = roc_auc(y_val[mask], val_scores[mask])
                held_out = [m for m in members if m not in chosen]
                panel = np.full(len(run.ids), np.nan)
                panel[used] = panel_used
                boot_fb = bootstrap(run.gate_slices(unseen), panel[:, None])
                boot_pa = bootstrap(runs["p_a_sd1_seed0"].gate_slices(pa_slices),
                                    panel[:, None]) if runs["p_a_sd1_seed0"].ids == run.ids \
                    else None
                fb = interval(np.nanmean(boot_fb["auc"][:, :, 0], axis=1))
                row = {"order": order_name, "k": k, "members": " ".join(chosen),
                       "lineages": len({lineage[m] for m in chosen}), "head": head_name,
                       "val_auc_all": f"{roc_auc(y_val, val_scores):.4f}",
                       "val_macro_auc_in_bank_decoders":
                           f"{np.mean([per_decoder[m] for m in chosen]):.4f}",
                       "val_macro_auc_held_out_decoders": (
                           f"{np.mean([per_decoder[m] for m in held_out]):.4f}"
                           if held_out else "n/a (k = 12)"),
                       "panel_macro_auc_full_bank_unseen": fmt(*fb),
                       "panel_macro_auc_p_a_gate_slices": (
                           fmt(*interval(np.nanmean(boot_pa["auc"][:, :, 0], axis=1)))
                           if boot_pa is not None else "n/a")}
                rows_out.append(row)
                curve.setdefault((order_name, head_name), []).append(
                    (k, float(row["val_macro_auc_held_out_decoders"])
                     if held_out else np.nan, fb))
            log.info("X3b %s k=%d done", order_name, k)
    out = args.out_dir / "x3b"
    columns = ["order", "k", "lineages", "head", "val_auc_all",
               "val_macro_auc_in_bank_decoders", "val_macro_auc_held_out_decoders",
               "panel_macro_auc_full_bank_unseen", "panel_macro_auc_p_a_gate_slices", "members"]
    write_table(out, "x3b_decoder_count", columns, rows_out,
                "X3b. S3-LR and S3-Set retrained on features from k bank decoders, training "
                "images fixed to full_bank's (1,000 reals, 1,000 fakes of every lineage). "
                "Held-out decoders: those whose features are not used (their reconstructions "
                "are still among the training fakes). Panel macro AUC with 95% CI.",
                columns[:-1])
    write_json(out / "x3b.json", {"orders": X3B_ORDERS, "ks": X3B_KS, "rows": rows_out,
                                  "training": "full_bank S3 subset, C=1.0, set head 200 epochs, "
                                              "AdamW 1e-3, seed 0, no validation selection"})
    x3b_figure(curve, args.figure_dir)


def x3b_figure(curve: dict, directory: Path) -> None:
    """AUC vs k: held-out-decoder validation AUC and panel macro AUC."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True)
    styles = {"near_first": "-", "far_first": "--"}
    colours = {"S3-LR": "#dd8452", "S3-Set": "#4c72b0"}
    for (order, head), points in sorted(curve.items()):
        ks = [p[0] for p in points]
        axes[0].plot(ks, [p[1] for p in points], styles[order], marker="o",
                     color=colours[head], label=f"{head}, {order}")
        axes[1].errorbar(ks, [p[2][0] for p in points],
                         yerr=[[p[2][0] - p[2][1] for p in points],
                               [p[2][2] - p[2][0] for p in points]],
                         linestyle=styles[order], marker="o", capsize=2, color=colours[head],
                         label=f"{head}, {order}")
    for ax, title in zip(axes, ("validation: macro AUC on held-out decoders",
                                "panel: macro AUC, full_bank unseen slices (95% CI)"),
                         strict=True):
        ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
        ax.set_xticks(list(X3B_KS))
        ax.set_xlabel("k = decoders whose features the head sees")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("AUC")
    fig.suptitle(f"X3b ({LABEL}): decoder count, training images fixed (full_bank)",
                 fontsize=11)
    fig.tight_layout()
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(directory / f"x3b_auc_vs_k.{suffix}", dpi=150)
    plt.close(fig)


def run_x3c(args: argparse.Namespace, log: Any, runs: dict[str, Run], context: dict) -> None:
    """S2 of the four runs on the same panel slices."""
    base = runs[RUNS[0]]
    for rid in RUNS[1:]:
        if runs[rid].ids != base.ids:
            raise ExploratoryError(f"{rid}: panel rows differ from {RUNS[0]}")
    matrix = np.column_stack([runs[r].scores["s2"] for r in RUNS])
    rows_out, tex_rows = [], []
    for name in sorted(base.slices, key=lambda s: (s.split("/")[0], s)):
        spec = base.slices[name]
        boot = bootstrap([base.own_slice(name)], matrix) if spec["reals"] else None
        status = {r: runs[r].slices[name]["in_bank"] for r in RUNS}
        tex = {"slice": name, "n f/r": f"{len(spec['fakes'])}/{len(spec['reals'])}"}
        for k, rid in enumerate(RUNS):
            thr = runs[rid].val_thresholds["s2"]
            tpr_val = float(np.mean(matrix[spec["fakes"], k] >= thr))
            if boot is not None:
                a = interval(boot["auc"][:, 0, k])
                t = interval(boot["tpr_slice"][:, 0, k])
                fpr = float(np.mean(matrix[spec["reals"], k] >= thr))
            else:
                a = t = (float("nan"),) * 3
                fpr = None
            rows_out.append({"slice": name, "run": RUN_NAMES[rid], "n_fake": len(spec["fakes"]),
                             "n_real": len(spec["reals"]),
                             "protocol_trained_on_this_generator": status[rid] or "n/a",
                             "auc": fmt(*a), "tpr_slice": fmt(*t),
                             "fpr_reals_at_val_threshold": "" if fpr is None else f"{fpr:.3f}",
                             "tpr_at_val_threshold": f"{tpr_val:.3f}"})
            tex[RUN_NAMES[rid]] = (fmt(a[0]) + ("*" if status[rid] == "in-bank" else "")
                                   if boot is not None else f"({tpr_val:.3f}†)")
        tex_rows.append(tex)
    out = args.out_dir / "x3c"
    write_table(out, "x3c_s2_by_protocol", ["slice", "run", "n_fake", "n_real",
                                            "protocol_trained_on_this_generator", "auc",
                                            "tpr_slice", "fpr_reals_at_val_threshold",
                                            "tpr_at_val_threshold"], rows_out,
                "X3c. S2 of each run on the same panel slices.")
    write_table(out, "x3c_s2_auc", ["slice", "n f/r", *[RUN_NAMES[r] for r in RUNS]], tex_rows,
                "X3c. S2 AUC per slice, same images for every run. * = the run trained on that "
                "E7 VAE's lineage (in-bank). full_bank saw every bank lineage, so its E7 numbers "
                "are not an unseen test; the fair unseen test is P4 leave-one-lineage-out. "
                "† E8 has no reals: detection rate at the validation threshold (I-064).")
    write_json(out / "x3c.json", {"rows": rows_out})
    log.info("X3c done")


# ======================================================================== main ===


def main() -> int:
    """Run the requested analyses."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("analysis", choices=("x1", "x2", "x3a", "x3b", "x3c", "all"))
    parser.add_argument("--s1-list", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--probe-list", required=True, type=Path)
    parser.add_argument("--s1-cache", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--s2-dir", required=True, type=Path)
    parser.add_argument("--heads-dir", required=True, type=Path)
    parser.add_argument("--panel-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "real")
    parser.add_argument("--validation-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p2_session_e" / "validation")
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "reports" / "exploratory_x")
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    log = setup_logging("p2_exploratory_x")
    log.info("%s", LABEL)
    globals()["RESAMPLES"] = args.resamples

    from dbp.eval.run_config import load_run_configs

    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    fit = _load_script("p2_fit_heads_x", "p2_fit_heads.py")
    probe_rows = analyze.read_scores(args.probe_list)
    probe_val = [r for r in probe_rows if r["view"] == "val"]
    context = {"bank": bank, "members": [s.key for s in bank.distinct_members()],
               "run_configs": load_run_configs(THESIS_DBP / "configs" / "p2_runs.yaml", bank),
               "probe_val": probe_val, "member_of": fit.member_of,
               "val_reals": sorted(r["image_id"] for r in probe_val if r["label"] == "real")}
    if len(context["val_reals"]) != 491:
        raise ExploratoryError(f"{len(context['val_reals'])} val reals, expected 491")
    runs = {rid: Run(args, rid) for rid in RUNS}
    steps = {"x1": run_x1, "x2": run_x2, "x3a": run_x3a, "x3b": run_x3b, "x3c": run_x3c}
    for name, step in steps.items():
        if args.analysis in (name, "all"):
            step(args, log, runs, context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
