"""P2-B5/B6: fit every head, threshold and fusion -- on train and validation only.

Nothing in this script touches the test panel. It produces, for one run
(``<protocol>_seed<seed>``):

* **S1** -- the linear probe on frozen DINOv3 features, ``C`` chosen on validation;
* **S3-LR** -- logistic regression on the standardised twelve-member probe features;
* **S3-Set** -- the permutation-invariant set head, its invariance asserted before and
  after training;
* **AEROBLADE** -- training-free, so only its validation threshold;
* **S2** -- trained by ``p2_train_s2.py``; here only its validation scores are
  calibrated;
* per stream and per fusion: thresholds at 1% and 5% FPR fixed on the validation
  **reals alone** (Amendment 2 point 3, I-062), and a temperature per stream;
* **F-mean(A2)** (``f_mean_rz``), the governing fusion of Amendment 2: S1, S2 and S3-Set,
  robust-z-scored on the validation reals, equal weights, no fake label;
* Analysis A's fusions, F-mean and F-stack, over exactly :data:`FUSION_STREAMS`.

**The protocol decides which fakes every fitted thing may see** (pre-registration §2,
Amendment 2). S1 trains on the S1 list's training reals and the training fakes of the
protocol's lineages (plus ``gen_matched``). S3 trains on the 1,000 reals and the
protocol's own 1,000-fake pick of the S3 subset -- recomputed here with the same seeded
draw ``p2_build_image_lists.py`` made, and required to lie inside the probe list. The
**validation set** is the probe list's validation rows (all 491 validation reals and the
fakes of every lineage) restricted to the protocol's lineages: everything fitted on it
-- temperatures, thresholds, the S1 ``C``, the set head's epoch, both fusions, and the
choice of the best single stream -- sees no fake from an unseen lineage. The first
version fitted all of these on every lineage, which would have let F-stack learn the
unseen families the gate is about.

**Every array is joined by image id** (I-056) through :mod:`dbp.eval.join`: duplicates,
missing ids, a stream holding ids it should not, and labels that differ between the S1
list, the probe list and the split all raise, naming the first ten ids.

**The fusion check** (before the one-time panel analysis): the F-stack weight of each
stream on standardised validation logits, any solver or separation warning, and the
validation AUC of every single stream and of F-mean(A2) per decoder group on the *full*
probe validation set. It stopped P-A (I-060); Amendment 2 resolved that by making
F-stack secondary, so the check is still computed and reported for Analysis A but no
longer blocks scoring.

Everything written records ``fitted_on``, and
:meth:`dbp.eval.fusion.FittedFusion.score` refuses to be applied to that split.

Usage:
    python scripts/p2_fit_heads.py --protocol p_a_sd1 --seed 0 \\
        --s1-list .../p2_s1_images.csv --probe-list .../p2_probe_images.csv \\
        --splits .../p2_splits.csv --s1-cache .../s1 --probe-dir .../probe --s2-dir .../s2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.fusion import (  # noqa: E402
    FUSION_STREAMS,
    GOVERNING_STREAMS,
    fit_by_id,
    fit_robust_mean,
    fit_temperature,
    robust_score_by_id,
)
from dbp.eval.join import agreed_labels, join_streams, require_streams  # noqa: E402
from dbp.eval.metrics import roc_auc  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402
from dbp.streams.s1_semantic import fit_probe  # noqa: E402
from dbp.streams.s3_probe import (  # noqa: E402
    FEATURE_NAMES,
    aeroblade_score,
    assert_permutation_invariant,
    build_set_head,
    feature_column_names,
    standardise,
    standardiser,
)
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import head_paths, run_id, s2_run_paths  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

#: FPR targets every stream is calibrated at.
FPR_TARGETS = (0.01, 0.05)

#: Set-head training, on the same subset S3-LR uses.
SET_HEAD_EPOCHS = 200
SET_HEAD_LR = 1e-3

#: The single streams, in the order reports list them.
SINGLE_STREAMS = ("s1", "s2", "s3_lr", "s3_set", "aeroblade")

#: The fusion check's stop rule (user instruction 2026-10-05, before any panel score).
S2_WEIGHT_SHARE_STOP = 0.90

#: Fusions, in the order reports list them: the governing one first (Amendment 2).
FUSIONS = ("f_mean_rz", "f_mean", "f_stack")

#: Validation reals every run must hold (the same 491 images in every protocol).
EXPECTED_VAL_REALS = 491

#: Rows expected in each input, so a truncated or substituted file fails here.
EXPECTED_S2_VAL_ROWS = 6883

S1_CACHE_NAME = "p2_s1_dinov3-vitb16_train.npz"


def member_of(row: dict[str, str]) -> str:
    """``real``, ``gen_matched`` or the bank member a corpus row was reconstructed by."""
    if row["label"] == "real":
        return "real"
    parts = row["relative_path"].replace("\\", "/").split("/")
    if parts[0] == "pairs" and len(parts) > 2:
        return parts[1]
    if parts[0] == "gen_matched":
        return "gen_matched"
    raise ValueError(f"cannot tell the member of fake row {row['image_id']}: "
                     f"{row['relative_path']}")


def read_rows(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """``(image_ids, features)`` from a cache."""
    with np.load(path, allow_pickle=False) as blob:
        return [str(v) for v in blob["image_ids"]], blob["features"]


def load_build_lists() -> Any:
    """``p2_build_image_lists.py``, for the seeded S3 draw it made."""
    name = "p2_build_image_lists_for_heads"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, THESIS_DBP / "scripts" / "p2_build_image_lists.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def s3_training_ids(
    s1_rows: list[dict[str, str]], allowed: set[str], protocol: str, seed: int
) -> tuple[list[str], list[str]]:
    """The S3 training reals and this protocol's own fake pick, re-drawn exactly.

    The candidates are the S1 list's training rows -- the population the list builder
    drew from -- and the draw uses its seed and salt.
    """
    lists = load_build_lists()
    if protocol not in lists.S3_PROTOCOLS:
        raise ValueError(f"no S3 training subset was drawn for protocol {protocol!r}")
    reals = sorted(r["image_id"] for r in s1_rows if r["view"] == "train" and r["label"] == "real")
    by_member: dict[str, list[str]] = {}
    for row in s1_rows:
        if row["view"] == "train" and row["label"] == "fake" and member_of(row) in allowed:
            by_member.setdefault(member_of(row), []).append(row["image_id"])
    chosen_reals = lists.balanced_take({"real": reals}, lists.S3_TRAIN_REALS, seed, "s3_reals")
    chosen_fakes = lists.balanced_take(by_member, lists.S3_TRAIN_FAKES_PER_PROTOCOL, seed,
                                       f"s3_fakes:{protocol}")
    return sorted(chosen_reals), sorted(chosen_fakes)


def digest(ids: list[str]) -> str:
    """sha256 of the newline-joined ids, so a report names exactly which rows it used."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> int:
    """Fit every head and fusion for one run. Returns 0, or 1 on a failed check."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--s1-list", required=True, type=Path)
    parser.add_argument("--probe-list", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--s1-cache", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--s2-dir", required=True, type=Path,
                        help="Root of the S2 run directories (outputs/p2/s2).")
    parser.add_argument("--list-seed", type=int, default=2026,
                        help="Seed the image lists were drawn with (summary: 2026).")
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "heads")
    parser.add_argument("--reports-dir", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p2_fit_heads")
    rid = run_id(args.protocol, args.seed)
    out = head_paths(args.out_dir, args.reports_dir, rid)

    import torch

    seed_everything(args.seed)
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    run = load_run_configs(THESIS_DBP / "configs" / "p2_runs.yaml", bank).by_id(args.protocol)
    log.info("%s", run.report_header(bank))
    members = [spec.key for spec in bank.distinct_members()]
    kinds = {spec.key: spec.kind for spec in bank.distinct_members()}
    allowed = set(run.active_members(bank)) | {"gen_matched"}
    log.info("%s: fakes allowed from %s", rid, sorted(allowed))

    # ---- the lists and their labels -------------------------------------------------
    s1_rows = read_rows(args.s1_list)
    probe_rows = read_rows(args.probe_list)
    split_rows = read_rows(args.splits)
    s1_by_id = {r["image_id"]: r for r in s1_rows}
    probe_by_id = {r["image_id"]: r for r in probe_rows}
    label_sources = {
        "s1_list": {r["image_id"]: r["label"] for r in s1_rows},
        "probe_list": {r["image_id"]: r["label"] for r in probe_rows},
        "splits": {r["row_id"]: r["kind"] for r in split_rows},
    }
    split_of = {r["row_id"]: r["split"] for r in split_rows}
    disagreeing = [r["image_id"] for r in s1_rows if split_of.get(r["image_id"]) != r["view"]]
    if disagreeing:
        log.error("S1 list view differs from the split for %d row(s), first 10: %s",
                  len(disagreeing), disagreeing[:10])
        return 1

    # ---- training rows, protocol by protocol ------------------------------------------
    s1_train = [r["image_id"] for r in s1_rows
                if r["view"] == "train" and (r["label"] == "real" or member_of(r) in allowed)]
    s3_reals, s3_fakes = s3_training_ids(s1_rows, allowed, args.protocol, args.list_seed)
    s3_train = sorted(set(s3_reals) | set(s3_fakes))
    outside = [i for i in s3_train if probe_by_id.get(i, {}).get("view") != "train"]
    if outside:
        log.error("%d re-drawn S3 training id(s) are not training rows of the probe list "
                  "(first 10: %s): the draw does not reproduce the list", len(outside),
                  outside[:10])
        return 1
    probe_train_reals = sorted(r["image_id"] for r in probe_rows
                               if r["view"] == "train" and r["label"] == "real")
    if s3_reals != probe_train_reals:
        log.error("the re-drawn S3 training reals differ from the probe list's")
        return 1
    leaked = [i for i in s1_train + s3_train
              if s1_by_id[i]["label"] == "fake" and member_of(s1_by_id[i]) not in allowed]
    if leaked:
        log.error("training fakes from outside the protocol: %s", leaked[:10])
        return 1

    # ---- the validation set: probe validation rows, restricted to the protocol --------
    probe_val = [r["image_id"] for r in probe_rows if r["view"] == "val"]
    val_ids = [i for i in probe_val
               if probe_by_id[i]["label"] == "real" or member_of(probe_by_id[i]) in allowed]
    s1_val_reals = {r["image_id"] for r in s1_rows if r["view"] == "val" and r["label"] == "real"}
    val_reals = {i for i in val_ids if probe_by_id[i]["label"] == "real"}
    if val_reals != s1_val_reals:
        log.error("the validation set holds %d reals; every one of the %d S1 validation "
                  "reals must be in it", len(val_reals), len(s1_val_reals))
        return 1
    y_val = agreed_labels(val_ids, label_sources, what="validation labels")
    y_s1_train = agreed_labels(s1_train, {k: v for k, v in label_sources.items()
                                          if k != "probe_list"}, what="S1 training labels")
    y_s3_train = agreed_labels(s3_train, label_sources, what="S3 training labels")
    composition = Counter(member_of(probe_by_id[i]) for i in val_ids)
    log.info("validation: %d rows (%d real, %d fake) %s", len(val_ids), int((y_val == 0).sum()),
             int((y_val == 1).sum()), dict(sorted(composition.items())))
    log.info("training: S1 %d rows (%s), S3 %d rows (%d real, %d fake)", len(s1_train),
             dict(sorted(Counter(member_of(s1_by_id[i]) for i in s1_train).items())),
             len(s3_train), len(s3_reals), len(s3_fakes))

    # ---- the streams, joined by id --------------------------------------------------
    s1_ids, s1_features = load_npz(args.s1_cache / S1_CACHE_NAME)
    join_streams([r["image_id"] for r in s1_rows], {"s1_cache": (s1_ids, np.zeros(len(s1_ids)))},
                 what="S1 cache = S1 list")
    s1_tr = join_streams(s1_train, {"s1": (s1_ids, s1_features)}, supersets=("s1",),
                         what="S1 train")["s1"].astype(np.float32)
    s1_va = join_streams(val_ids, {"s1": (s1_ids, s1_features)}, supersets=("s1",),
                         what="S1 val")["s1"].astype(np.float32)

    s2_csv = s2_run_paths(args.s2_dir, args.reports_dir, rid).scores("val")
    s2_rows = read_rows(s2_csv)
    s2_all_ids = [r["image_id"] for r in s2_rows]
    s1_val_ids = [r["image_id"] for r in s1_rows if r["view"] == "val"]
    join_streams(s1_val_ids, {"s2_val": (s2_all_ids, np.zeros(len(s2_all_ids)))},
                 expected_count=EXPECTED_S2_VAL_ROWS, what="S2 val = S1 validation rows")
    s2_values = np.array([float(r["score"]) for r in s2_rows])

    probe_cache: dict[str, tuple[list[str], np.ndarray]] = {}
    for member in members:
        ids, features = load_npz(args.probe_dir / f"p2_probe_{member}.npz")
        probe_cache[member] = (ids, features)

    def s3_matrix(target: list[str], what: str) -> np.ndarray:
        joined = join_streams(target, probe_cache, supersets=members, what=what)
        return np.stack([joined[m] for m in members], axis=1).astype(np.float32)

    s3_tr = s3_matrix(s3_train, "S3 train")
    s3_va = s3_matrix(val_ids, "S3 val")
    s3_full_val = s3_matrix(probe_val, "S3 full validation")

    captured: list[str] = []

    def record(caught: list[warnings.WarningMessage], where: str) -> None:
        for item in caught:
            captured.append(f"{where}: {item.category.__name__}: {item.message}")

    # ---- S1 --------------------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    y_s1_val = y_val
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = fit_probe(s1_tr, y_s1_train, s1_va, y_s1_val, seed=args.seed)
    record(caught, "S1 probe")
    log.info("S1 probe: C=%s  val AUC %.4f", probe["C"], probe["val_auc"])
    np.savez_compressed(out.s1_probe, coef=probe["model"].coef_,
                        intercept=probe["model"].intercept_)

    # ---- S3-LR -----------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression

    mean, scale = standardiser(s3_tr.reshape(len(s3_tr), -1))
    s3_tr_std = standardise(s3_tr.reshape(len(s3_tr), -1), mean, scale)
    s3_va_std = standardise(s3_va.reshape(len(s3_va), -1), mean, scale)
    s3_full_std = standardise(s3_full_val.reshape(len(s3_full_val), -1), mean, scale)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s3_lr = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed)
        s3_lr.fit(s3_tr_std, y_s3_train)
    record(caught, "S3-LR")
    np.savez_compressed(out.s3_lr, coef=s3_lr.coef_, intercept=s3_lr.intercept_, mean=mean,
                        scale=scale, members=np.array(members))

    # ---- S3-Set ----------------------------------------------------------------------
    set_head = build_set_head()
    assert_permutation_invariant(set_head, len(members))
    shape = (-1, len(members), len(FEATURE_NAMES))
    train_t = torch.from_numpy(s3_tr_std.reshape(shape))
    val_t = torch.from_numpy(s3_va_std.reshape(shape))
    labels_t = torch.from_numpy(y_s3_train.astype(np.float32))
    optimiser = torch.optim.AdamW(set_head.parameters(), lr=SET_HEAD_LR)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_auc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(SET_HEAD_EPOCHS):
        set_head.train()
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(set_head(train_t), labels_t)
        loss.backward()
        optimiser.step()
        if (epoch + 1) % 20 == 0:
            set_head.eval()
            with torch.no_grad():
                auc = roc_auc(y_val, set_head(val_t).numpy())
            if auc > best_auc:
                best_auc, best_epoch = auc, epoch + 1
                best_state = {k: v.clone() for k, v in set_head.state_dict().items()}
    if best_state is not None:
        set_head.load_state_dict(best_state)
    delta = assert_permutation_invariant(set_head, len(members))
    set_head.eval()
    torch.save(set_head.state_dict(), out.s3_set)

    # ---- every single stream on the validation set (and the full one, diagnostic) -------
    def aeroblade(matrix: np.ndarray) -> np.ndarray:
        column = FEATURE_NAMES.index("lpips_vgg")
        return np.array([aeroblade_score(dict(zip(members, row[:, column], strict=True)),
                                         kinds)["score"] for row in matrix])

    def streams_on(ids: list[str], s3_raw: np.ndarray, s3_std: np.ndarray) -> dict[str, np.ndarray]:
        s1_x = join_streams(ids, {"s1": (s1_ids, s1_features)}, supersets=("s1",),
                            what="S1 scoring")["s1"].astype(np.float32)
        with torch.no_grad():
            set_scores = set_head(torch.from_numpy(s3_std.reshape(shape))).numpy()
        return {
            "s1": probe["model"].decision_function(s1_x),
            "s2": join_streams(ids, {"s2": (s2_all_ids, s2_values)}, supersets=("s2",),
                               what="S2 scoring")["s2"],
            "s3_lr": s3_lr.decision_function(s3_std),
            "s3_set": set_scores,
            "aeroblade": aeroblade(s3_raw),
        }

    val_scores = streams_on(val_ids, s3_va, s3_va_std)
    require_streams(val_scores, SINGLE_STREAMS, "validation streams")
    stream_report = {
        name: {"val_auc": roc_auc(y_val, values)} for name, values in val_scores.items()
    }
    stream_report["s1"].update({"C": probe["C"], "grid": probe["grid"],
                                "n_train": probe["n_train"], "n_val": probe["n_val"]})
    stream_report["s3_lr"].update({"n_features": int(s3_tr_std.shape[1]),
                                   "columns": feature_column_names(members)[:5] + ["..."]})
    stream_report["s3_set"].update({"best_epoch": best_epoch, "epochs": SET_HEAD_EPOCHS,
                                    "permutation_delta_after_training": delta})
    stream_report["aeroblade"].update({"trained": False, "excluded_kinds": ["sr_gan"]})
    for name in SINGLE_STREAMS:
        log.info("%-9s validation AUC %.5f", name, stream_report[name]["val_auc"])

    # ---- calibration and fusion, all on validation, all joined by id -------------------
    fitted_on = f"val_{rid}"
    real_ids = [i for i, y in zip(val_ids, y_val, strict=True) if y == 0]
    if len(real_ids) != EXPECTED_VAL_REALS:
        log.error("validation holds %d reals, expected %d", len(real_ids), EXPECTED_VAL_REALS)
        return 1
    is_real = y_val == 0

    def calibrate(scores: np.ndarray) -> dict[str, Any]:
        """Thresholds on the validation reals alone (Amendment 2 point 3, I-062)."""
        out: dict[str, Any] = {}
        for target in FPR_TARGETS:
            entry = calibrate_on_reals(scores[is_real], target, fitted_on=fitted_on).to_dict()
            # Reported for reading only: the threshold above was fixed without the fakes.
            entry["val_tpr_at_threshold"] = float(np.mean(scores[~is_real] >= entry["threshold"]))
            out[f"fpr_{int(target * 100)}"] = entry
        return out

    calibrations: dict[str, Any] = {}
    for name, scores in val_scores.items():
        calibrations[name] = calibrate(scores)
        calibrations[name]["temperature"] = fit_temperature(y_val, scores)

    tables = {name: (val_ids, values) for name, values in val_scores.items()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fusions: dict[str, Any] = {
            "f_mean_rz": fit_robust_mean(tables, real_ids, fitted_on,
                                         expected_count=EXPECTED_VAL_REALS),
            "f_mean": fit_by_id("mean", (val_ids, y_val), tables, val_ids, fitted_on),
            "f_stack": fit_by_id("stack", (val_ids, y_val), tables, val_ids, fitted_on),
        }
    record(caught, "fusions")
    require_streams(fusions["f_mean_rz"].streams, GOVERNING_STREAMS, "f_mean_rz (recorded)")
    for name in ("f_mean", "f_stack"):
        require_streams(fusions[name].streams, FUSION_STREAMS, f"{name} (recorded)")
    for name, fusion in fusions.items():
        log.info("%s fitted on %s over %s", name, fusion.fitted_on, list(fusion.streams))
    def in_sample(fusion: Any) -> np.ndarray:
        # A FittedFusion on the split it was fitted on, for its threshold only.
        matrix = fusion.calibrated(val_scores)
        if fusion.kind == "mean":
            return matrix.mean(axis=1)
        return matrix @ np.asarray(fusion.coefficients) + fusion.intercept

    fused_on_val = {
        "f_mean_rz": robust_score_by_id(fusions["f_mean_rz"], tables, val_ids, None),
        "f_mean": in_sample(fusions["f_mean"]),
        "f_stack": in_sample(fusions["f_stack"]),
    }
    for name, scores in fused_on_val.items():
        calibrations[name] = calibrate(scores)
        stream_report[name] = {"val_auc": roc_auc(y_val, scores),
                               "note": "in-sample for f_mean/f_stack (fitted on this set); "
                                       "f_mean_rz used the reals' location and scale only"}
        log.info("%-9s validation AUC %.5f", name, stream_report[name]["val_auc"])
    log.info("F-mean(A2) parameters: %s", fusions["f_mean_rz"].to_dict())

    # ---- the fusion check -------------------------------------------------------------
    stack = fusions["f_stack"]
    calibrated = stack.calibrated({n: val_scores[n] for n in stack.streams})
    spread = calibrated.std(axis=0)
    weighted = np.abs(np.asarray(stack.coefficients) * spread)
    shares = dict(zip(stack.streams, (weighted / weighted.sum()).tolist(), strict=True))
    fused_val = calibrated @ np.asarray(stack.coefficients) + stack.intercept
    separation: list[str] = []
    for name in SINGLE_STREAMS:
        if stream_report[name]["val_auc"] >= 1.0:
            separation.append(f"{name} separates the validation set perfectly (AUC 1.0)")
    fused_auc = roc_auc(y_val, fused_val)
    if fused_auc >= 1.0:
        separation.append("F-stack separates the validation set perfectly (in-sample AUC 1.0)")
    separation += [w for w in captured if "Convergence" in w or "separat" in w.lower()]
    stop_reasons = []
    if shares.get("s2", 0.0) > S2_WEIGHT_SHARE_STOP:
        stop_reasons.append(f"S2 holds {shares['s2']:.1%} of the standardised F-stack "
                            f"weight (> {S2_WEIGHT_SHARE_STOP:.0%})")
    stop_reasons += separation

    full_scores = streams_on(probe_val, s3_full_val, s3_full_std)
    full_tables = {name: (probe_val, values) for name, values in full_scores.items()}
    full_scores["f_mean_rz"] = robust_score_by_id(fusions["f_mean_rz"], full_tables,
                                                  probe_val, f"full_validation_{rid}")
    full_labels = agreed_labels(probe_val, label_sources, what="full validation labels")
    groups = [member_of(probe_by_id[i]) for i in probe_val]
    real_mask = np.array([g == "real" for g in groups])
    per_group: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups) - {"real"}):
        mask = np.array([g == group for g in groups])
        y = np.r_[np.zeros(real_mask.sum(), int), np.ones(mask.sum(), int)]
        per_group[group] = {
            "n_fake": int(mask.sum()),
            "in_protocol": group in allowed,
            **{name: round(roc_auc(y, np.r_[full_scores[name][real_mask],
                                             full_scores[name][mask]]), 5)
               for name in (*SINGLE_STREAMS, "f_mean_rz")},
        }
    if int(full_labels.sum()) != int((~real_mask).sum()):
        log.error("full validation: labels and member groups disagree on the fake count")
        return 1

    # ---- exploratory (Amendment 2): reconstruction fidelity vs S2 transfer -------------
    from scipy import stats

    psnr_column = FEATURE_NAMES.index("psnr_db")
    mean_psnr = {member: float(np.mean(s3_full_val[real_mask, k, psnr_column]))
                 for k, member in enumerate(members)}
    in_both = [m for m in members if m in per_group]
    rho = stats.spearmanr([mean_psnr[m] for m in in_both], [per_group[m]["s2"] for m in in_both])
    unseen = [m for m in in_both if not per_group[m]["in_protocol"]]
    rho_unseen = (stats.spearmanr([mean_psnr[m] for m in unseen],
                                  [per_group[m]["s2"] for m in unseen])
                  if len(unseen) >= 3 else None)
    exploratory = {
        "label": "EXPLORATORY -- not part of the gate (Amendment 2)",
        "mean_psnr_db_on_validation_reals": {m: round(v, 4) for m, v in mean_psnr.items()},
        "spearman_psnr_vs_s2_val_auc": {
            "decoders": in_both, "n": len(in_both),
            "rho": float(rho.statistic), "p_value": float(rho.pvalue),
        },
        "spearman_psnr_vs_s2_val_auc_unseen_only": None if rho_unseen is None else {
            "decoders": unseen, "n": len(unseen),
            "rho": float(rho_unseen.statistic), "p_value": float(rho_unseen.pvalue),
        },
    }
    log.info("EXPLORATORY Spearman(mean PSNR, S2 val AUC) over %d decoders: %.3f (p %.3g)",
             len(in_both), rho.statistic, rho.pvalue)

    fusion_check = {
        "stack_coefficients": dict(zip(stack.streams, stack.coefficients, strict=True)),
        "stack_intercept": stack.intercept,
        "standardised_weight": {k: round(v, 5) for k, v in shares.items()},
        "standardisation": "|coefficient| x standard deviation of the temperature-scaled "
                           "validation logit, normalised to sum to 1",
        "stack_val_auc_in_sample": fused_auc,
        "warnings": captured,
        "separation": separation,
        "s2_share_threshold": S2_WEIGHT_SHARE_STOP,
        "stop": bool(stop_reasons),
        "stop_reasons": stop_reasons,
        "rule": "user instruction 2026-10-05: STOP before scoring the panel if S2 holds "
                ">90% of the standardised F-stack weight or the fit shows separation",
        "resolution": "Amendment 2 (D-069, commit 5a0b250): F-stack is secondary "
                      "(Analysis A); the verdict is decided by F-mean(A2), which learns no "
                      "weight, so this check is reported but no longer blocks scoring",
    }
    log.info("F-stack standardised weight: %s", {k: f"{v:.1%}" for k, v in shares.items()})
    for reason in stop_reasons:
        log.warning("FUSION CHECK: %s", reason)

    report = {
        "run_id": rid,
        "protocol": args.protocol,
        "seed": args.seed,
        "fitted_on": fitted_on,
        "allowed_fakes": sorted(allowed),
        "validation": {
            "rows": len(val_ids), "reals": int((y_val == 0).sum()),
            "fakes": int((y_val == 1).sum()), "by_group": dict(sorted(composition.items())),
            "ids_sha256": digest(val_ids),
            "definition": "probe-list validation rows whose label is real or whose member "
                          "is one of the protocol's lineages or gen_matched (Amendment 2)",
        },
        "training": {
            "s1_rows": len(s1_train), "s1_ids_sha256": digest(s1_train),
            "s3_rows": len(s3_train), "s3_reals": len(s3_reals), "s3_fakes": len(s3_fakes),
            "s3_ids_sha256": digest(s3_train),
        },
        "streams": stream_report,
        "calibrations": calibrations,
        "fusions": {name: f.to_dict() for name, f in fusions.items()},
        "fusion_streams_configured": list(FUSION_STREAMS),
        "governing_fusion": {"name": "f_mean_rz", "streams": list(GOVERNING_STREAMS),
                             "rule": "pre-registration Amendment 2"},
        "exploratory": exploratory,
        "fusion_check": fusion_check,
        "per_group_val_auc_full_validation": per_group,
        "n_val": len(val_ids),
        "note": ("Every number here comes from train and validation. The test panel is "
                 "untouched; p2_score.py scores it once against these frozen artefacts."),
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out.report_tmp.write_text(json.dumps(report, indent=2, default=float) + "\n",
                              encoding="utf-8")
    out.report_tmp.replace(out.report)
    log.info("wrote %s; fusion check: %s", out.report.name,
             "STOP" if fusion_check["stop"] else "pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
