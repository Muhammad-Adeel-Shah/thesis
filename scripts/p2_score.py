"""P2-B7: score the test panel once, against frozen artefacts.

Single shot. Every head, threshold and fusion was fitted by ``p2_fit_heads.py`` on
train and validation; this script only applies them. There is no flag here that
changes a model, because the moment a panel score can be re-run with a different
setting after it has been seen, the panel stops being a test set. A re-run is permitted
for a crash and for nothing else (B7): ``--crash-rerun REASON`` is required to overwrite
an existing score file, and the reason is written into the summary.

**What it reads, and how it checks it (I-056).** Every input is joined by image id
through :mod:`dbp.eval.join`; a missing, duplicate or unexpected id raises with the
first ten ids, and nothing becomes a silent NaN:

* the **frozen panel list** (``p2_panel_images_frozen.csv``, 11,775 rows) and the
  **frozen luma list** (``p2_panel_images_luma.csv``, 978 rows), each checked against
  the digest committed in pre-registration Amendment 1;
* the luma rows' stream ids (``p2_luma_images.csv``, ``<id>__luma``), required to equal
  the frozen luma list's parents with the suffix added;
* the **per-set panel metadata** (``p2_panel_<view>.csv``, the pre-I-047 draw, so a
  superset), joined by id, its ``sha256`` required to match the frozen list's;
* S1's panel and luma caches (exact), S2's panel and luma score files (exact), the
  twelve S3 probe caches (supersets: they also hold the training and validation rows);
* the run's heads and its ``p2_heads_<run id>.json``.

**``--check-only``** runs every path, id, count, digest and shape check above and
stops: no label column is read, no head is applied, no score is written. It is how the
real panel is dry-run (user instruction 2026-10-05).

Output: one row per panel and luma image, with the label, the slice metadata,
``in_bank`` (from :meth:`dbp.eval.run_config.RunConfig.subset_status`), the five single
heads and the three fusions -- F-mean(A2) (``f_mean_rz``, the governing fusion of
Amendment 2) and Analysis A's ``f_mean`` and ``f_stack``.

Usage:
    python scripts/p2_score.py --protocol p_a_sd1 --seed 0 --panel-list .../frozen.csv \\
        --luma-list .../luma.csv --luma-ids .../p2_luma_images.csv --panel-sets DIR \\
        --s1-cache DIR --probe-dir DIR --s2-dir DIR --heads DIR --heads-reports DIR \\
        --out-dir DIR [--check-only]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256, write_manifest  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.fusion import (  # noqa: E402
    FUSION_STREAMS,
    GOVERNING_STREAMS,
    fusion_from_dict,
    robust_score_by_id,
    score_by_id,
)
from dbp.eval.join import JoinError, agreed_labels, check_unique, join_streams  # noqa: E402
from dbp.eval.panel import LUMA_VIEWS, luma_image_id  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.streams.s3_probe import (  # noqa: E402
    FEATURE_NAMES,
    aeroblade_score,
    build_set_head,
    standardise,
)
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import head_paths, run_id, s2_run_paths, score_paths  # noqa: E402

#: The frozen panel (pre-registration Amendment 1). Not flags: a different panel is a
#: different experiment. ``--synthetic`` replaces them for the pipeline dry-run only,
#: and stamps every output SYNTHETIC.
PANEL_ROWS = 11775
LUMA_ROWS = 978
PANEL_DIGEST = "bff608c3d8e00ea834532cd541849b4f21de803c8f8e27a55ca5e70424c51116"
LUMA_DIGEST = "190279cc889da9c4d879c4d7ffd18afae42949c58896a983a94f1bd6ec9c2303"

SINGLE_STREAMS = ("s1", "s2", "s3_lr", "s3_set", "aeroblade")
FUSIONS = ("f_mean_rz", "f_mean", "f_stack")

SCORE_COLUMNS = (
    "image_id", "parent_id", "view", "subset", "generator", "content_type", "label",
    "in_bank", *SINGLE_STREAMS, *FUSIONS,
)

#: Per view, the metadata column that names the subset, the generator and the content
#: type. Each evaluation set names them differently, and writing a plausible name
#: instead of the real one is how the E3 panel failed the first time it ran.
SUBSET_COLUMN = {
    "e7_dda_coco": "source_vae",
    "e8_evalgen": "generator",
    "e3_matched": "source_label",
    "e4_wildrf_matched": "platform",
    "t_ctrl_diagnostic": "member",
}
GENERATOR_COLUMN = {"e8_evalgen": "generator", "e3_matched": "source_label",
                    "e7_dda_coco": "source_vae"}
CONTENT_TYPE_COLUMN = {"e3_matched": "subset"}

#: E7 subset name -> the bank member whose VAE produced it. Asserted, not inferred: a
#: silent miss would mark an in-bank subset unseen and inflate every generalisation claim.
E7_SUBSET_TO_MEMBER = {
    "sd-vae-ft-ema": "sd_ft_ema",
    "sd-vae-ft-mse": "sd_ft_mse",
    "stable-diffusion-2-1": "sd21",
    "sdxl-vae": "sdxl",
    "stable-diffusion-3.5-large": "sd35",
    "FLUX.1": "flux",
    "real": None,
}

S1_CACHES = {"panel": "p2_s1_dinov3-vitb16_panel.npz", "luma": "p2_s1_dinov3-vitb16_panel_luma.npz"}


class ScoreError(RuntimeError):
    """A check before scoring failed."""


def read_columns(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Only ``columns`` of a CSV -- so ``--check-only`` never holds a label in memory."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ScoreError(f"{path.name}: column(s) {missing} absent")
        return [{c: row[c] for c in columns} for row in reader]


def list_digest(rows: list[dict[str, str]]) -> str:
    """Amendment 1's digest: sha256 of the sorted ``image_id<TAB>sha256`` lines."""
    lines = sorted(f"{r['image_id']}\t{r['sha256']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def load_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """``(image_ids, features)`` from a cache."""
    with np.load(path, allow_pickle=False) as blob:
        return [str(v) for v in blob["image_ids"]], blob["features"]


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Every array in a head file."""
    with np.load(path, allow_pickle=False) as blob:
        return {key: blob[key] for key in blob.files}


def main() -> int:
    """Check, then (unless ``--check-only``) score the panel once. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--panel-list", required=True, type=Path)
    parser.add_argument("--luma-list", required=True, type=Path)
    parser.add_argument("--luma-ids", required=True, type=Path,
                        help="p2_luma_images.csv: the luma rows' stream ids (<id>__luma).")
    parser.add_argument("--panel-sets", required=True, type=Path,
                        help="Directory of the per-set panel metadata p2_panel_<view>.csv.")
    parser.add_argument("--s1-cache", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--s2-dir", required=True, type=Path)
    parser.add_argument("--heads", required=True, type=Path)
    parser.add_argument("--heads-reports", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true",
                        help="Paths, ids, counts, digests and shapes only: no label is "
                             "read, no head applied, no score written.")
    parser.add_argument("--synthetic", nargs=4, metavar=("ROWS", "DIGEST", "LUMA_ROWS",
                                                         "LUMA_DIGEST"),
                        help="Pipeline dry-run on a synthetic panel: its counts and digests "
                             "replace the frozen panel's, and every output says SYNTHETIC.")
    parser.add_argument("--crash-rerun", default=None, metavar="REASON",
                        help="Overwrite an existing score file after a crash (B7). The "
                             "reason is recorded.")
    args = parser.parse_args()
    log = setup_logging("p2_score")
    rid = run_id(args.protocol, args.seed)
    try:
        return run(args, rid, log)
    except (ScoreError, JoinError) as exc:
        log.error("%s: %s", rid, exc)
        return 1


def run(args: argparse.Namespace, rid: str, log: Any) -> int:
    """The checks and the scoring; raises :class:`ScoreError` / ``JoinError``."""
    if args.synthetic:
        rows_expected, digest_expected = int(args.synthetic[0]), args.synthetic[1]
        luma_expected, luma_digest_expected = int(args.synthetic[2]), args.synthetic[3]
        log.warning("SYNTHETIC panel: a pipeline dry-run, not a result")
    else:
        rows_expected, digest_expected = PANEL_ROWS, PANEL_DIGEST
        luma_expected, luma_digest_expected = LUMA_ROWS, LUMA_DIGEST
    fitted_paths = head_paths(args.heads, args.heads_reports, rid)
    out_paths = score_paths(args.out_dir, rid)
    s2_paths = s2_run_paths(args.s2_dir, args.heads_reports, rid)

    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    run_config = load_run_configs(THESIS_DBP / "configs" / "p2_runs.yaml", bank).by_id(
        args.protocol)
    members = [spec.key for spec in bank.distinct_members()]
    kinds = {spec.key: spec.kind for spec in bank.distinct_members()}

    # ---- every path, before anything is read ------------------------------------------
    paths = {
        "panel_list": args.panel_list, "luma_list": args.luma_list, "luma_ids": args.luma_ids,
        "s1_panel": args.s1_cache / S1_CACHES["panel"],
        "s1_luma": args.s1_cache / S1_CACHES["luma"],
        "s2_panel": s2_paths.scores("panel"), "s2_luma": s2_paths.scores("luma"),
        "s1_probe": fitted_paths.s1_probe, "s3_lr": fitted_paths.s3_lr,
        "s3_set": fitted_paths.s3_set, "heads_report": fitted_paths.report,
        **{f"probe_{m}": args.probe_dir / f"p2_probe_{m}.npz" for m in members},
    }
    missing = [f"{name}: {path}" for name, path in paths.items() if not Path(path).is_file()]
    if missing:
        raise ScoreError(f"{len(missing)} input(s) missing: {missing}")
    if not args.check_only and out_paths.scores.exists() and not args.crash_rerun:
        raise ScoreError(f"{out_paths.scores} exists. The panel is scored once (B7); pass "
                         "--crash-rerun REASON only after a crash recorded in the log")

    # ---- the frozen lists ---------------------------------------------------------------
    columns = ("image_id", "view", "sha256") + (() if args.check_only else ("label",))
    panel = read_columns(args.panel_list, columns)
    luma = read_columns(args.luma_list, columns)
    for name, rows, count, expected_digest in (
        ("panel", panel, rows_expected, digest_expected),
        ("luma", luma, luma_expected, luma_digest_expected),
    ):
        check_unique(f"{name} list", [r["image_id"] for r in rows])
        if len(rows) != count:
            raise ScoreError(f"{name} list holds {len(rows)} rows, expected {count}")
        found = list_digest(rows)
        if found != expected_digest:
            raise ScoreError(f"{name} list digest {found} != the committed {expected_digest}")
        log.info("%s list: %d rows, digest %s OK", name, len(rows), found[:16])
    panel_ids = [r["image_id"] for r in panel]
    panel_by_id = {r["image_id"]: r for r in panel}
    for row in luma:
        spec = LUMA_VIEWS.get(row["view"])
        if spec is None:
            raise ScoreError(f"luma row {row['image_id']}: view {row['view']!r} is not a "
                             "declared luma view")
        parent = panel_by_id.get(row["image_id"])
        if parent is None or parent["view"] != spec["parent"]:
            raise ScoreError(f"luma row {row['image_id']} has no parent in {spec['parent']}")
    luma_ids = [luma_image_id(r["image_id"]) for r in luma]
    stream_luma_ids = [r["image_id"] for r in read_columns(args.luma_ids, ("image_id",))]
    join_streams(luma_ids,
                 {"p2_luma_images.csv": (stream_luma_ids, np.zeros(len(stream_luma_ids)))},
                 expected_count=luma_expected, what="luma stream ids = frozen luma list")

    # ---- per-set metadata, joined by id ------------------------------------------------
    views = sorted({r["view"] for r in panel})
    metadata: dict[str, dict[str, str]] = {}
    label_from_metadata: dict[str, str] = {}
    for view in views:
        path = args.panel_sets / f"p2_panel_{view}.csv"
        if not path.is_file():
            raise ScoreError(f"panel metadata for {view} missing: {path}")
        wanted = ("image_id", "sha256", *{c for c in (SUBSET_COLUMN.get(view),
                                                      GENERATOR_COLUMN.get(view),
                                                      CONTENT_TYPE_COLUMN.get(view)) if c})
        rows = read_columns(path, wanted + (() if args.check_only else ("label",)))
        ids = [r["image_id"] for r in panel if r["view"] == view]
        meta_ids = [r["image_id"] for r in rows]
        joined = join_streams(ids, {f"metadata {view}": (meta_ids, np.arange(len(rows)))},
                              supersets=(f"metadata {view}",), what=f"metadata {view}")
        for image_id, position in zip(ids, joined[f"metadata {view}"], strict=True):
            row = rows[int(position)]
            if row["sha256"] != panel_by_id[image_id]["sha256"]:
                raise ScoreError(f"{view}: sha256 of {image_id} differs between the frozen "
                                 "list and the panel metadata")
            metadata[image_id] = row
            if not args.check_only:
                label_from_metadata[image_id] = row["label"]
        log.info("metadata %-24s %5d row(s) joined of %d", view, len(ids), len(rows))

    # The luma companion must be exactly its parent view's declared subsets: the gate's
    # bootstrap resamples luma rows with their parents, and a mismatch found only there
    # would be found after the panel was scored. Metadata only; no label, no score.
    for luma_view, spec in LUMA_VIEWS.items():
        parent_view = str(spec["parent"])
        column = SUBSET_COLUMN[parent_view]
        expected = {r["image_id"] for r in panel if r["view"] == parent_view
                    and metadata[r["image_id"]][column] in spec["subsets"]}
        found = {r["image_id"] for r in luma if r["view"] == luma_view}
        if found != expected:
            raise ScoreError(
                f"{luma_view}: luma parents differ from the {parent_view} rows with "
                f"{column} in {list(spec['subsets'])} (missing "
                f"{sorted(expected - found)[:10]}, unexpected {sorted(found - expected)[:10]})")
        log.info("%s: %d luma rows = the %s rows of %s %s", luma_view, len(found),
                 parent_view, column, list(spec["subsets"]))

    # ---- heads ----------------------------------------------------------------------------
    fitted = json.loads(fitted_paths.report.read_text(encoding="utf-8"))
    if fitted.get("run_id") != rid:
        raise ScoreError(f"{fitted_paths.report.name} is for {fitted.get('run_id')!r}")
    for name in FUSIONS:
        if name not in fitted.get("fusions", {}) or name not in fitted.get("calibrations", {}):
            raise ScoreError(f"{fitted_paths.report.name} has no fitted {name}")
    fusions = {name: fusion_from_dict(fitted["fusions"][name]) for name in FUSIONS}
    if set(fusions["f_mean_rz"].streams) != set(GOVERNING_STREAMS):
        raise ScoreError(f"f_mean_rz was fitted on {fusions['f_mean_rz'].streams}")
    for name in ("f_mean", "f_stack"):
        if set(fusions[name].streams) != set(FUSION_STREAMS):
            raise ScoreError(f"{name} was fitted on {fusions[name].streams}")
    probe = load_arrays(fitted_paths.s1_probe)
    lr = load_arrays(fitted_paths.s3_lr)
    if [str(m) for m in lr["members"]] != members:
        raise ScoreError(f"S3-LR was fitted on members {list(lr['members'])}, "
                         f"the bank is {members}")

    # ---- stream inputs, joined by id ----------------------------------------------------
    all_ids = panel_ids + luma_ids
    s1_panel_ids, s1_panel = load_npz(paths["s1_panel"])
    s1_luma_ids, s1_luma = load_npz(paths["s1_luma"])
    s1_x = np.concatenate([
        join_streams(panel_ids, {"S1 panel cache": (s1_panel_ids, s1_panel)},
                     expected_count=rows_expected, what="S1 panel")["S1 panel cache"],
        join_streams(luma_ids, {"S1 luma cache": (s1_luma_ids, s1_luma)},
                     expected_count=luma_expected, what="S1 luma")["S1 luma cache"],
    ])
    if s1_x.shape[1] != probe["coef"].shape[-1]:
        raise ScoreError(f"S1 features have {s1_x.shape[1]} dims, the probe expects "
                         f"{probe['coef'].shape[-1]}")

    def s2_table(path: Path) -> tuple[list[str], np.ndarray]:
        rows = read_columns(path, ("image_id", "score"))
        return [r["image_id"] for r in rows], np.array([float(r["score"]) for r in rows])

    s2 = np.concatenate([
        join_streams(panel_ids, {"S2 panel": s2_table(paths["s2_panel"])},
                     expected_count=rows_expected, what="S2 panel")["S2 panel"],
        join_streams(luma_ids, {"S2 luma": s2_table(paths["s2_luma"])},
                     expected_count=luma_expected, what="S2 luma")["S2 luma"],
    ])
    probe_caches = {m: load_npz(paths[f"probe_{m}"]) for m in members}
    joined = join_streams(all_ids, probe_caches, supersets=members, what="S3 probe caches")
    matrix = np.stack([joined[m] for m in members], axis=1).astype(np.float32)
    if matrix.shape[1:] != (len(members), len(FEATURE_NAMES)):
        raise ScoreError(f"S3 matrix {matrix.shape}, expected (n, {len(members)}, "
                         f"{len(FEATURE_NAMES)})")
    if lr["mean"].shape[-1] != len(members) * len(FEATURE_NAMES):
        raise ScoreError("S3-LR standardiser does not match the bank")
    for name, values in (("S1", s1_x), ("S2", s2), ("S3", matrix)):
        if not np.all(np.isfinite(values)):
            raise ScoreError(f"{name} inputs hold non-finite values")
    shapes = {"s1": list(s1_x.shape), "s2": list(s2.shape), "s3": list(matrix.shape)}
    log.info("inputs joined by id for %d rows (%d panel + %d luma): %s", len(all_ids),
             len(panel_ids), len(luma_ids), shapes)

    inputs_sha256 = {name: file_sha256(Path(path)) for name, path in sorted(paths.items())}
    check = {
        "run_id": rid, "synthetic": bool(args.synthetic),
        "panel_rows": len(panel_ids), "luma_rows": len(luma_ids),
        "panel_digest": list_digest(panel), "luma_digest": list_digest(luma),
        "views": dict(sorted(Counter(r["view"] for r in panel).items())),
        "shapes": shapes, "inputs_sha256": inputs_sha256,
        "labels_read": not args.check_only,
    }
    if args.check_only:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / f"check_{rid}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({**check, "result": "PASSED", "note":
                                   "ids, counts, digests, paths and shapes only; no label "
                                   "read, no head applied, no score written"}, indent=2)
                       + "\n", encoding="utf-8")
        tmp.replace(path)
        log.info("CHECK PASSED for %s (no label read, nothing scored) -> %s", rid, path.name)
        return 0

    # ---- labels: the frozen lists and the metadata must agree ---------------------------
    luma_label = {luma_image_id(r["image_id"]): r["label"] for r in luma}
    labels = agreed_labels(panel_ids, {
        "frozen list": {r["image_id"]: r["label"] for r in panel},
        "panel metadata": label_from_metadata,
    }, what="panel labels")
    luma_labels = agreed_labels(luma_ids, {
        "frozen luma list": luma_label,
        "parent's label": {luma_image_id(r["image_id"]): panel_by_id[r["image_id"]]["label"]
                           for r in luma},
    }, what="luma labels")
    y = np.r_[labels, luma_labels]

    # ---- the heads ---------------------------------------------------------------------
    import torch

    s1 = s1_x.astype(np.float64) @ probe["coef"].ravel().astype(np.float64) + float(
        probe["intercept"].ravel()[0])
    flat = standardise(matrix.reshape(len(matrix), -1), lr["mean"], lr["scale"])
    s3_lr = flat.astype(np.float64) @ lr["coef"].ravel().astype(np.float64) + float(
        lr["intercept"].ravel()[0])
    set_head = build_set_head()
    set_head.load_state_dict(torch.load(fitted_paths.s3_set, weights_only=True))
    set_head.eval()
    with torch.no_grad():
        s3_set = set_head(torch.from_numpy(flat.reshape(matrix.shape))).numpy().astype(np.float64)
    lpips = FEATURE_NAMES.index("lpips_vgg")
    aeroblade = np.array([aeroblade_score(dict(zip(members, row[:, lpips], strict=True)),
                                          kinds)["score"] for row in matrix])
    singles = {"s1": s1, "s2": s2, "s3_lr": s3_lr, "s3_set": s3_set, "aeroblade": aeroblade}
    tables = {name: (all_ids, values) for name, values in singles.items()}
    set_id = f"panel_{rid}" + ("_synthetic" if args.synthetic else "")
    fused = {
        "f_mean_rz": robust_score_by_id(fusions["f_mean_rz"], tables, all_ids, set_id),
        "f_mean": score_by_id(fusions["f_mean"], tables, all_ids, set_id),
        "f_stack": score_by_id(fusions["f_stack"], tables, all_ids, set_id),
    }
    for name, values in {**singles, **fused}.items():
        if values.shape != (len(all_ids),) or not np.all(np.isfinite(values)):
            raise ScoreError(f"{name}: scores are not one finite value per row")

    # ---- rows ----------------------------------------------------------------------------
    out_rows = []
    for position, image_id in enumerate(all_ids):
        is_luma = position >= len(panel_ids)
        parent_id = luma[position - len(panel_ids)]["image_id"] if is_luma else image_id
        parent = panel_by_id[parent_id]
        base_view = parent["view"]
        view = luma[position - len(panel_ids)]["view"] if is_luma else base_view
        meta = metadata[parent_id]
        subset = meta[SUBSET_COLUMN[base_view]] if base_view in SUBSET_COLUMN else ""
        in_bank = ""
        if base_view == "e7_dda_coco":
            if subset not in E7_SUBSET_TO_MEMBER:
                raise ScoreError(f"E7 subset {subset!r} is not in E7_SUBSET_TO_MEMBER; "
                                 "refusing to mark it unseen by default")
            member = E7_SUBSET_TO_MEMBER[subset]
            in_bank = "real" if member is None else run_config.subset_status(bank, member)
        out_rows.append({
            "image_id": image_id, "parent_id": parent_id, "view": view, "subset": subset,
            "generator": meta[GENERATOR_COLUMN[base_view]]
            if base_view in GENERATOR_COLUMN else "",
            "content_type": meta[CONTENT_TYPE_COLUMN[base_view]]
            if base_view in CONTENT_TYPE_COLUMN else "",
            "label": "fake" if y[position] == 1 else "real",
            "in_bank": in_bank,
            **{name: f"{values[position]:.6f}" for name, values in {**singles, **fused}.items()},
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(out_rows, out_paths.scores, SCORE_COLUMNS)  # atomic
    summary = {
        **check,
        "rows": len(out_rows),
        "streams": list(SINGLE_STREAMS), "fusions": list(FUSIONS),
        "fusion_streams": {name: list(f.streams) for name, f in fusions.items()},
        "artefacts_fitted_on": fitted["fitted_on"],
        "heads_report_sha256": file_sha256(fitted_paths.report),
        "single_shot": "scored once; a re-run is permitted only for a crash (B7)",
        "crash_rerun_reason": args.crash_rerun,
        "scores": out_paths.scores.name,
        "scores_sha256": file_sha256(out_paths.scores),
    }
    if args.synthetic:
        summary["SYNTHETIC"] = "pipeline dry-run on validation rows; not a result"
    out_paths.summary_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    out_paths.summary_tmp.replace(out_paths.summary)
    log.info("wrote %s (%d rows)", out_paths.scores.name, len(out_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
