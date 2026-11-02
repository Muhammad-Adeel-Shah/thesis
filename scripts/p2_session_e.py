"""Session E, end to end, on CPU: score the panel once and analyse it once.

Three modes, run in this order (user instruction 2026-10-05):

``synthetic``
    The **pipeline dry-run**. A synthetic panel is built from the probe list's
    validation rows -- whose S1 features, S2 scores and S3 probe features all exist --
    laid out with the real panel's views, slice names, luma companion and real pools,
    and the full scoring and analysis are run on it for every run, then the P-A seeds
    are combined. Every output says SYNTHETIC: the heads were fitted on these rows, so
    the numbers measure nothing.
``check-real``
    ``p2_score.py --check-only`` on the **real** panel for every run: paths, ids,
    counts, digests and shapes. No label is read, no head is applied, no score written.
``real``
    The one-time analysis: score every run, analyse every run, combine the P-A seeds
    (Amendment 2 point 6). Refused if it has started before, unless ``--crash-rerun
    REASON`` (B7), and refused unless the pre-registration with Amendment 2 and every
    tracked script and module are committed, so the result is tied to a commit.

Usage:
    python scripts/p2_session_e.py synthetic --inputs F:/thesis-2/kaggle_raw ...
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.panel import luma_image_id  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: The runs Session E scores, in order; P-A's two seeds are combined.
RUNS = (("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0))
COMBINED = ("p_a_sd1", ("p_a_sd1_seed0", "p_a_sd1_seed1"))

#: Synthetic layout: which validation member's fakes play which panel slice. Chosen so
#: every slice name the real panel has exists, with P-A's in-bank / unseen split.
SYNTHETIC_FAKES = {
    "sdxl": ("e7_dda_coco", {"source_vae": "sdxl-vae"}),
    "sd35": ("e7_dda_coco", {"source_vae": "stable-diffusion-3.5-large"}),
    "flux": ("e7_dda_coco", {"source_vae": "FLUX.1"}),
    "sd_ft_ema": ("e7_dda_coco", {"source_vae": "sd-vae-ft-ema"}),
    "sd21": ("e7_dda_coco", {"source_vae": "stable-diffusion-2-1"}),
    "sd15": ("e7_dda_coco", {"source_vae": "sd-vae-ft-mse"}),
    "qwen_image": ("e3_matched", {"source_label": "GPT-Image2-fake", "subset": "Hybrid Images"}),
    "flux2_klein": ("e3_matched", {"source_label": "Nano-Banana2-fake",
                                   "subset": "Texture Images"}),
    "dcae": ("e4_wildrf_matched", {"platform": "reddit"}),
    "movq_k22": ("e5_chameleon_pipeline", {}),
    "sdxl_fp16fix": ("e8_evalgen", {"generator": "Flux"}),
    "realesrgan_x4": ("e8_evalgen", {"generator": "GoT"}),
}
#: gen_matched fakes are split three ways for E8's remaining generators, and the t_ctrl
#: view takes every fifth real and the last of them.
SYNTHETIC_E8_SPLIT = ("Infinity", "NOVA", "OmniGen")
REAL_VIEWS = ("e7_dda_coco", "e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline",
              "t_ctrl_diagnostic")
METADATA_COLUMNS = {
    "e7_dda_coco": ("source_vae",), "e3_matched": ("source_label", "subset"),
    "e4_wildrf_matched": ("platform",), "e5_chameleon_pipeline": (),
    "e8_evalgen": ("generator",), "t_ctrl_diagnostic": ("member",),
}
LIST_COLUMNS = ["image_id", "relative_path", "label", "view", "sha256"]


class SessionError(RuntimeError):
    """A Session E step failed or was refused."""


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Atomic CSV write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: row.get(c, "") for c in columns} for row in rows)
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    """All rows of a CSV."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(rows: list[dict[str, Any]]) -> str:
    """Amendment 1's list digest."""
    lines = sorted(f"{r['image_id']}\t{r['sha256']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def member_of(row: dict[str, str]) -> str:
    """The bank member (or ``gen_matched`` / ``real``) of a probe-list row."""
    if row["label"] == "real":
        return "real"
    parts = row["relative_path"].replace("\\", "/").split("/")
    if parts[0] == "pairs" and len(parts) > 2:
        return parts[1]
    if parts[0] == "gen_matched":
        return "gen_matched"
    raise SessionError(f"cannot tell the member of {row['image_id']}")


def build_synthetic(args: argparse.Namespace, out: Path, log: Any) -> list[str]:
    """Write a synthetic panel shaped like the real one; return p2_score's --synthetic."""
    probe_rows = [r for r in read_csv(args.probe_list) if r["view"] == "val"]
    reals = sorted(r["image_id"] for r in probe_rows if r["label"] == "real")
    by_member: dict[str, list[str]] = defaultdict(list)
    for row in probe_rows:
        if row["label"] == "fake":
            by_member[member_of(row)].append(row["image_id"])
    panel: list[dict[str, Any]] = []
    metadata: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(image_id: str, view: str, label: str, meta: dict[str, str]) -> None:
        row = {"image_id": image_id, "view": view, "label": label,
               "relative_path": f"{view}/{image_id}.png",
               "sha256": hashlib.sha256(f"synthetic:{image_id}".encode()).hexdigest()}
        panel.append(row)
        metadata[view].append({**row, **meta})

    for index, image_id in enumerate(reals):
        view = REAL_VIEWS[index % len(REAL_VIEWS)]
        meta = {"e7_dda_coco": {"source_vae": "real"}, "e3_matched": {"source_label": "real",
                "subset": "Structure Images"}, "e4_wildrf_matched": {"platform": "twitter"},
                "t_ctrl_diagnostic": {"member": ""}}.get(view, {})
        add(image_id, view, "real", meta)
    for member, (view, meta) in SYNTHETIC_FAKES.items():
        ids = sorted(by_member[member])
        if member == "sd15":  # half to E7's sd-vae-ft-mse, half to the T-ctrl view
            for image_id in ids[1::2]:
                add(image_id, "t_ctrl_diagnostic", "fake", {"member": "sd15"})
            ids = ids[::2]
        for image_id in ids:
            add(image_id, view, "fake", meta)
    for k, image_id in enumerate(sorted(by_member["gen_matched"])):
        add(image_id, "e8_evalgen", "fake",
            {"generator": SYNTHETIC_E8_SPLIT[k % len(SYNTHETIC_E8_SPLIT)]})
    luma = [dict(r, view="e7_dda_coco_luma",
                 sha256=hashlib.sha256(f"synthetic-luma:{r['image_id']}".encode()).hexdigest())
            for r in panel if r["view"] == "e7_dda_coco"
            and metadata_value(metadata, r["image_id"]) in ("FLUX.1", "real")]
    luma_ids = [luma_image_id(r["image_id"]) for r in luma]

    lists = out / "lists"
    write_csv(lists / "frozen.csv", panel, LIST_COLUMNS)
    write_csv(lists / "luma.csv", luma, LIST_COLUMNS)
    write_csv(lists / "luma_ids.csv", [{"image_id": i} for i in luma_ids], ["image_id"])
    for view, rows in metadata.items():
        write_csv(out / "sets" / f"p2_panel_{view}.csv", rows,
                  LIST_COLUMNS + list(METADATA_COLUMNS[view]))

    panel_ids = [r["image_id"] for r in panel]
    parents = [r["image_id"] for r in luma]
    with np.load(args.s1_cache / "p2_s1_dinov3-vitb16_train.npz", allow_pickle=False) as blob:
        index = {str(v): k for k, v in enumerate(blob["image_ids"])}
        features = blob["features"]
        (out / "s1").mkdir(parents=True, exist_ok=True)
        np.savez(out / "s1" / "p2_s1_dinov3-vitb16_panel.npz", image_ids=np.array(panel_ids),
                 features=features[[index[i] for i in panel_ids]])
        np.savez(out / "s1" / "p2_s1_dinov3-vitb16_panel_luma.npz",
                 image_ids=np.array(luma_ids), features=features[[index[i] for i in parents]])
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    (out / "probe").mkdir(parents=True, exist_ok=True)
    for spec in bank.distinct_members():
        with np.load(args.probe_dir / f"p2_probe_{spec.key}.npz", allow_pickle=False) as blob:
            index = {str(v): k for k, v in enumerate(blob["image_ids"])}
            rows = [index[i] for i in panel_ids + parents]
            np.savez(out / "probe" / f"p2_probe_{spec.key}.npz",
                     image_ids=np.array(panel_ids + luma_ids), features=blob["features"][rows])
    for protocol, seed in RUNS:
        rid = f"{protocol}_seed{seed}"
        scores = {r["image_id"]: r["score"]
                  for r in read_csv(args.s2_dir / rid / "scores" / "val.csv")}
        target = out / "s2" / rid / "scores"
        write_csv(target / "panel.csv", [{"image_id": i, "score": scores[i]} for i in panel_ids],
                  ["image_id", "score"])
        write_csv(target / "luma.csv", [{"image_id": luma_image_id(i), "score": scores[i]}
                                        for i in parents], ["image_id", "score"])
    log.info("SYNTHETIC panel: %d rows (%d real), %d luma rows, from validation", len(panel),
             len(reals), len(luma))
    return [str(len(panel)), digest(panel), str(len(luma)), digest(luma)]


def metadata_value(metadata: dict[str, list[dict[str, Any]]], image_id: str) -> str:
    """E7 ``source_vae`` of a synthetic row."""
    for row in metadata["e7_dda_coco"]:
        if row["image_id"] == image_id:
            return str(row["source_vae"])
    return ""


def run_step(command: list[str], log_path: Path, log: Any) -> None:
    """Run one step, its output to ``log_path``; raise on a non-zero exit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    seconds = time.monotonic() - started
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-30:]
        raise SessionError(f"{log_path.name} exited {result.returncode} after {seconds:.0f}s:\n"
                           + "\n".join(tail))
    log.info("%-40s OK (%.0fs)", log_path.name, seconds)


def require_committed(log: Any) -> str:
    """HEAD, after checking the pre-registration and every tracked script are committed."""
    def git(*command: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *command], cwd=THESIS_DBP, capture_output=True,
                              text=True, check=False)

    head = git("rev-parse", "HEAD").stdout.strip()
    dirty = git("status", "--porcelain", "--", "docs/P2_PREREGISTRATION.md", "scripts",
                "src", "configs").stdout.strip()
    if dirty:
        raise SessionError(f"uncommitted changes would make the result untraceable:\n{dirty}")
    committed = git("show", "HEAD:thesis_dbp/docs/P2_PREREGISTRATION.md").stdout
    if "## Amendment 2" not in committed:
        raise SessionError("Amendment 2 is not in the committed pre-registration")
    log.info("committed at %s; pre-registration with Amendment 2 committed", head)
    return head


def main() -> int:
    """Run one Session E mode. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("synthetic", "check-real", "real"))
    parser.add_argument("--probe-list", required=True, type=Path)
    parser.add_argument("--s1-cache", required=True, type=Path)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--s2-dir", required=True, type=Path)
    parser.add_argument("--heads", required=True, type=Path)
    parser.add_argument("--heads-reports", required=True, type=Path)
    parser.add_argument("--panel-list", required=True, type=Path)
    parser.add_argument("--luma-list", required=True, type=Path)
    parser.add_argument("--luma-ids", required=True, type=Path)
    parser.add_argument("--panel-sets", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path,
                        help="Session E output root; each mode writes <out>/<mode>/.")
    parser.add_argument("--crash-rerun", default=None, metavar="REASON")
    args = parser.parse_args()
    log = setup_logging("p2_session_e")
    out = args.out / args.mode
    try:
        return run(args, out, log)
    except SessionError as exc:
        log.error("%s", exc)
        return 1


def run(args: argparse.Namespace, out: Path, log: Any) -> int:
    """The mode's steps."""
    python = sys.executable
    scripts = THESIS_DBP / "scripts"
    inputs = {"panel_list": args.panel_list, "luma_list": args.luma_list,
              "luma_ids": args.luma_ids, "panel_sets": args.panel_sets,
              "s1_cache": args.s1_cache, "probe_dir": args.probe_dir, "s2_dir": args.s2_dir}
    synthetic: list[str] = []
    head = ""
    if args.mode == "synthetic":
        synthetic = build_synthetic(args, out / "panel", log)
        root = out / "panel"
        inputs = {"panel_list": root / "lists" / "frozen.csv",
                  "luma_list": root / "lists" / "luma.csv",
                  "luma_ids": root / "lists" / "luma_ids.csv", "panel_sets": root / "sets",
                  "s1_cache": root / "s1", "probe_dir": root / "probe", "s2_dir": root / "s2"}
    elif args.mode == "real":
        head = require_committed(log)
        marker = out / "SESSION_E_REAL_STARTED.json"
        if marker.exists() and not args.crash_rerun:
            raise SessionError(f"{marker} exists: the panel is analysed once (B7). Pass "
                               "--crash-rerun REASON only after a crash recorded in the log")
        out.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "commit": head, "crash_rerun": args.crash_rerun},
                                     indent=2) + "\n", encoding="utf-8")
    scores_dir = out / "scores"
    analysis = out / "analysis"
    for protocol, seed in RUNS:
        rid = f"{protocol}_seed{seed}"
        command = [python, str(scripts / "p2_score.py"), "--protocol", protocol,
                   "--seed", str(seed), "--panel-list", str(inputs["panel_list"]),
                   "--luma-list", str(inputs["luma_list"]), "--luma-ids", str(inputs["luma_ids"]),
                   "--panel-sets", str(inputs["panel_sets"]),
                   "--s1-cache", str(inputs["s1_cache"]), "--probe-dir", str(inputs["probe_dir"]),
                   "--s2-dir", str(inputs["s2_dir"]), "--heads", str(args.heads),
                   "--heads-reports", str(args.heads_reports), "--out-dir", str(scores_dir)]
        if args.mode == "check-real":
            command.append("--check-only")
        if synthetic:
            command += ["--synthetic", *synthetic]
        if args.crash_rerun:
            command += ["--crash-rerun", args.crash_rerun]
        run_step(command, out / "logs" / f"score_{rid}.log", log)
    if args.mode == "check-real":
        log.info("CHECK-REAL PASSED for %d runs: ids, counts, digests, paths and shapes; "
                 "no label read, nothing scored", len(RUNS))
        return 0
    for protocol, seed in RUNS:
        rid = f"{protocol}_seed{seed}"
        command = [python, str(scripts / "p2_analyze.py"),
                   "--scores", str(scores_dir / f"scores_{rid}.csv"),
                   "--heads", str(args.heads_reports / f"p2_heads_{rid}.json"),
                   "--out-dir", str(analysis)]
        if synthetic:
            command.append("--synthetic")
        else:
            command += ["--panel-hashes", str(args.panel_list)]
        run_step(command, out / "logs" / f"analyze_{rid}.log", log)
    protocol, seeds = COMBINED
    run_step([python, str(scripts / "p2_analyze.py"), "--combine",
              *[str(analysis / s) for s in seeds],
              "--verdict-out", str(analysis / f"p2_verdict_{protocol}.json")],
             out / "logs" / f"combine_{protocol}.log", log)
    verdict = json.loads((analysis / f"p2_verdict_{protocol}.json").read_text(encoding="utf-8"))
    log.info("%s%s verdict: %s %s", "SYNTHETIC " if synthetic else "", protocol,
             verdict["verdict"], {k: v["verdict"] for k, v in verdict["runs"].items()})
    if args.mode == "real":
        done = out / "SESSION_E_REAL_DONE.json"
        done.write_text(json.dumps({"commit": head, "verdict": verdict["verdict"]}, indent=2)
                        + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
