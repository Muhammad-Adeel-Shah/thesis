"""EXPLORATORY X1 format control: does the generation-only probe read the file format?

X1 found that a probe on frozen DINOv3-B features, trained on real COCO vs SD1.5 full
generations, separates the generations almost perfectly (AUC 0.999). The reals are
COCO JPEGs stored as PNG, ``gen_matched`` are JPEGs format-matched to COCO, the
reconstructions are PNG. This control gives **every** X1 image the same history -- a
JPEG round-trip at quality 95, 4:4:4 (``dbp.utils.io.reencode_jpeg``) -- re-extracts the
features with the S1 extractor, re-fits the same probe on the same split, and reports
the result beside the original.

Subcommands:

``build-list``
    The X1 image list (train fit + tune, val reals, val ``gen_matched``, the val
    reconstructions of all 12 decoders, E7 on the panel), written to
    ``reports/p2_x1/x1_format_control_images.csv`` with a summary. Local.
``summarise``
    On Kaggle, after extraction: what was written, with counts. Always exits 0.
``analyse``
    Local, after the Kaggle session: three arms side by side -- the original cached
    features, the same images re-extracted as-is in that session (reproducibility), and
    re-encoded -- with 95% bootstrap CIs and the paired difference re-encoded minus
    original.

Caveat carried into the output: re-encoding at 4:4:4 cannot restore chroma that a
4:2:0 source already lost (CLAUDE.md section 6d), so E7 FLUX.1 keeps its shortcut.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

LIST_COLUMNS = ["image_id", "relative_path", "label", "view", "root"]
DEFAULT_LIST = THESIS_DBP / "reports" / "p2_x1" / "x1_format_control_images.csv"
TAGS = {"asis": "x1_asis", "jpeg": "x1_jpeg95"}
QUALITY = 95
CACHE = "p2_s1_dinov3-vitb16_{}.npz"
LABEL = "EXPLORATORY (X1 format control)"


def _load(name: str, filename: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, THESIS_DBP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def split_groups(s1_rows: list[dict[str, str]], splits: Path) -> dict[str, str]:
    """``image id -> COCO group``; a reconstruction takes its source real's group."""
    group = {r["row_id"]: r["group_id"] for r in read_csv(splits)}
    return {r["image_id"]: group[r["image_id"].rsplit("__", 1)[0]
                                 if "__" in r["image_id"] else r["image_id"]]
            for r in s1_rows}


def build_list(s1_rows: list[dict[str, str]], group: dict[str, str],
               panel_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    """Every image X1 touches, once, with the root it is read from."""
    x1 = _load("x1_probe_fc", "x1_gen_only_probe.py")
    split = x1.build_split(s1_rows, group)
    by_id = {r["image_id"]: r for r in s1_rows}
    corpus_ids = (split["fit"] + split["tune"] + split["test_reals"] + split["test_gen"]
                  + [p[1] for pairs in split["test_recon"].values() for p in pairs])
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ValueError("an X1 image appears twice")
    rows = [{"image_id": i, "relative_path": by_id[i]["relative_path"],
             "label": by_id[i]["label"], "view": by_id[i]["view"], "root": "corpus"}
            for i in corpus_ids]
    e7 = [r for r in panel_rows if r["view"] == "e7_dda_coco"]
    if any(r.get("root") != "panel" for r in e7):
        raise ValueError("E7 panel rows must carry root=panel")
    rows += [{c: r[c] for c in LIST_COLUMNS} for r in e7]
    rows.sort(key=lambda r: (r["root"], r["image_id"]))
    counts = {"fit": len(split["fit"]), "tune": len(split["tune"]),
              "val_reals": len(split["test_reals"]), "val_gen_matched": len(split["test_gen"]),
              "val_reconstructions": sum(len(v) for v in split["test_recon"].values()),
              "e7_panel": len(e7), "total": len(rows)}
    return rows, counts


def cmd_build_list(args: argparse.Namespace) -> int:
    """Write the list and its summary."""
    s1_rows = read_csv(args.s1_list)
    rows, counts = build_list(s1_rows, split_groups(s1_rows, args.splits),
                              read_csv(args.panel_list))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.with_suffix(".tmp").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.out.with_suffix(".tmp").replace(args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    summary = {"label": LABEL, "list": args.out.name, "rows": len(rows), "sha256": digest,
               "counts": counts, "by_root": dict(Counter(r["root"] for r in rows)),
               "reencode": {"format": "JPEG", "quality": QUALITY, "subsampling": "4:4:4"}}
    path = args.out.with_name(args.out.stem + "_summary.json")
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_summarise(args: argparse.Namespace) -> int:
    """Kaggle report cell: never fails the run."""
    for arm, tag in TAGS.items():
        cache = args.cache_dir / CACHE.format(tag)
        report = args.reports / f"p2_s1_extract_{tag}.json"
        try:
            with np.load(cache, allow_pickle=False) as blob:
                shape = blob["features"].shape
            meta = json.loads(report.read_text(encoding="utf-8"))
            print(f"{arm:5s} {cache.name}: {shape}, {meta.get('seconds_per_image')} s/image "
                  f"on {meta.get('device')}, reencode {meta.get('reencode')}")
        except (OSError, KeyError, ValueError) as exc:
            print(f"{arm:5s} NOT AVAILABLE: {type(exc).__name__}: {exc}")
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Original vs as-is vs re-encoded, same probe, same split, same tests."""
    from dbp.eval.complementarity import GateSlice
    from dbp.eval.join import join_streams
    from dbp.streams.s1_semantic import fit_probe

    x = _load("p2_exploratory_x_fc", "p2_exploratory_x.py")
    x1 = _load("x1_probe_fc", "x1_gen_only_probe.py")
    s1_rows = read_csv(args.s1_list)
    split = x1.build_split(s1_rows, split_groups(s1_rows, args.splits))
    labels = {r["image_id"]: 0 if r["label"] == "real" else 1 for r in s1_rows}
    e7_rows = [r for r in read_csv(args.panel_scores) if r["view"] == "e7_dda_coco"]

    def load(path: Path) -> tuple[list[str], np.ndarray]:
        with np.load(path, allow_pickle=False) as blob:
            return [str(v) for v in blob["image_ids"]], blob["features"]

    original_train = load(args.original_cache / CACHE.format("train"))
    original_panel = load(args.original_cache / CACHE.format("panel"))
    arms = {"original (P2 caches)": (original_train, original_panel)}
    for arm, tag in (("as-is (this session)", TAGS["asis"]),
                     (f"JPEG q{QUALITY} 4:4:4", TAGS["jpeg"])):
        cache = load(args.session_dir / CACHE.format(tag))
        arms[arm] = (cache, cache)

    def feats(ids: list[str], source) -> np.ndarray:
        return join_streams(ids, {"f": source}, supersets=("f",),
                            what="X1 format control")["f"].astype(np.float32)

    e7_ids = [r["image_id"] for r in e7_rows]
    reals = split["test_reals"]
    tests = [("gen_matched (SD1.5 full)", split["test_gen"], None)]
    tests += [(f"recon {m}", [p[1] for p in pairs], [p[0] for p in pairs])
              for m, pairs in split["test_recon"].items()]
    scored: dict[str, dict[str, np.ndarray]] = {}
    probes = {}
    for arm, (corpus, panel) in arms.items():
        probe = fit_probe(feats(split["fit"], corpus), [labels[i] for i in split["fit"]],
                          feats(split["tune"], corpus), [labels[i] for i in split["tune"]],
                          seed=0)
        probes[arm] = {"C": probe["C"], "tune_auc": probe["val_auc"]}
        model = probe["model"]
        scored[arm] = {"reals": model.decision_function(feats(reals, corpus)),
                       "e7": model.decision_function(feats(e7_ids, panel))}
        for name, fake_ids, _ in tests:
            scored[arm][name] = model.decision_function(feats(fake_ids, corpus))
    names = list(arms)

    # Reproducibility of the as-is arm against the original caches.
    reproduce = {}
    for which, ids, original in (("corpus", split["fit"][:2000], original_train),
                                 ("panel", e7_ids[:2000], original_panel)):
        a = feats(ids, original)
        b = feats(ids, arms["as-is (this session)"][0])
        reproduce[which] = {"rows": len(ids), "max_abs_diff": float(np.max(np.abs(a - b))),
                            "mean_cosine": float(np.mean(np.sum(a * b, axis=1) / (
                                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))))}

    rows_out, tex_rows = [], []

    def report(test: str, item: GateSlice, matrix: np.ndarray, n_fake: int, n_real: int,
               note: str) -> None:
        boot = x.bootstrap([item], matrix, with_ap=True)
        tex = {"test": test}
        for k, arm in enumerate(names):
            a, t, p = (x.interval(boot[m][:, 0, k]) for m in ("auc", "tpr_slice", "ap"))
            rows_out.append({"test": test, "arm": arm, "n_fake": n_fake, "n_real": n_real,
                             "auc": x.fmt(*a), "ap": x.fmt(*p), "tpr_slice": x.fmt(*t),
                             "note": note})
            tex[arm] = x.fmt(a[0])
        delta = x.interval(100 * (boot["auc"][:, 0, 2] - boot["auc"][:, 0, 0]))
        rows_out.append({"test": test, "arm": "JPEG minus original, AUC points (paired)",
                         "n_fake": n_fake, "n_real": n_real, "auc": x.fmt(*delta, digits=2),
                         "ap": "", "tpr_slice": "", "note": note})
        tex["JPEG - orig. (pts)"] = x.fmt(*delta, digits=2)
        tex_rows.append(tex)

    n_r = len(reals)
    for name, fake_ids, pair_reals in tests:
        matrix = np.vstack([np.column_stack([scored[a]["reals"] for a in names]),
                            np.column_stack([scored[a][name] for a in names])])
        if pair_reals:
            if pair_reals != reals:
                raise ValueError(f"{name}: pairs not aligned with the val reals")
            item = GateSlice(name, np.arange(n_r, 2 * n_r), tuple(reals), "pair",
                             np.arange(n_r), tuple(reals), "pair")
        else:
            units = list(reals) + list(fake_ids)
            item = x.single_slice(name, list(range(n_r, len(units))), list(range(n_r)), units)
        report(name, item, matrix, len(fake_ids), n_r, "val split; threshold = val reals")
    e7_matrix = np.column_stack([scored[a]["e7"] for a in names])
    e7_reals = [k for k, r in enumerate(e7_rows) if r["label"] == "real"]
    for subset in sorted({r["subset"] for r in e7_rows} - {"real"}):
        fakes = [k for k, r in enumerate(e7_rows) if r["subset"] == subset]
        item = x.single_slice(subset, fakes, e7_reals, e7_ids)
        report(f"E7 {subset}", item, e7_matrix, len(fakes), len(e7_reals),
               "panel; threshold = E7 reals" + ("; 4:2:0 chroma loss survives re-encoding "
                                                "(section 6d): upper bound"
                                                if subset == "FLUX.1" else ""))
    out = args.out_dir
    x.write_table(out, "x1_format_control", list(rows_out[0]), rows_out,
                  f"X1 format control. The generation-only DINOv3-B probe on the original "
                  f"features, on the same images re-extracted as-is, and with every image "
                  f"re-encoded as JPEG q{QUALITY} 4:4:4 (train and test alike); AUC / AP / "
                  f"slice-calibrated TPR@5%FPR with 95% bootstrap CI.")
    x.write_table(out, "x1_format_control_auc", list(tex_rows[0]), tex_rows,
                  f"X1 format control, AUC. Every image re-encoded as JPEG q{QUALITY} 4:4:4 "
                  f"vs the original; paired difference in points.")
    (out / "x1_format_control.json").write_text(json.dumps(
        {"label": LABEL, "probes": probes, "reproducibility_as_is_vs_original": reproduce,
         "rows": rows_out}, indent=2) + "\n", encoding="utf-8")
    for row in rows_out:
        print(f"{row['test'][:30]:30s} {row['arm'][:40]:40s} {row['auc']}")
    print("reproducibility:", json.dumps(reproduce))
    return 0


def main() -> int:
    """Dispatch."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-list")
    build.add_argument("--s1-list", required=True, type=Path)
    build.add_argument("--splits", required=True, type=Path)
    build.add_argument("--panel-list", required=True, type=Path,
                       help="p2_panel_images.csv (rows carry root=panel).")
    build.add_argument("--out", type=Path, default=DEFAULT_LIST)
    summarise = sub.add_parser("summarise")
    summarise.add_argument("--cache-dir", required=True, type=Path)
    summarise.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    analyse = sub.add_parser("analyse")
    analyse.add_argument("--s1-list", required=True, type=Path)
    analyse.add_argument("--splits", required=True, type=Path)
    analyse.add_argument("--original-cache", required=True, type=Path)
    analyse.add_argument("--session-dir", required=True, type=Path)
    analyse.add_argument("--panel-scores", type=Path, default=THESIS_DBP / "reports" /
                         "p2_session_e" / "real" / "scores" / "scores_p_a_sd1_seed0.csv")
    analyse.add_argument("--out-dir", type=Path,
                         default=THESIS_DBP / "reports" / "p2_x1" / "format_control")
    args = parser.parse_args()
    return {"build-list": cmd_build_list, "summarise": cmd_summarise,
            "analyse": cmd_analyse}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
