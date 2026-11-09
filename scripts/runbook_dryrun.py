"""Dry-run a Kaggle runbook locally, on CPU, executing its cells verbatim.

A runbook is a percent-format ``.py`` file: each ``# %%`` line starts a cell, and
``# %% [markdown]`` cells are skipped. The code cells are concatenated **exactly as
written** -- no substitution, no rewriting -- into one script and run with this
interpreter, which is what a notebook kernel does with consecutive cells. The only
differences from Kaggle are environment variables the runbook reads for its roots
(``DBP_INPUT_ROOT``, ``DBP_WORKING_ROOT``, ``DBP_TMP_ROOT``), the device
(``DBP_DEVICE=cpu``), the expectations file (``DBP_EXPECT``) and ``DBP_DRYRUN=1``, which
tells the bootstrap not to touch the local venv.

The fixture reproduces the shape of ``/kaggle/input`` -- the same dataset slugs at the
same depth -- in miniature, so a wrong path, a missing module, a misordered import or a
script called with the wrong flag fails here. Every one of those reached Kaggle at least
once (CLAUDE.md section 12).

A fixture may run the runbook in **several sessions**. Between sessions the working
directory becomes an attached notebook output under ``input/notebooks/...`` -- what
Kaggle does when the previous version's output is attached -- and the next session
starts from an empty working directory. Each session declares the exit status it must
end with, and the fixture's checks then inspect what the sessions left behind. The S2
fixture uses this to kill every run after step 4 in session 1 and prove, in session 2,
that each resumes at step 5 from the saved state.

Usage:
    python scripts/runbook_dryrun.py --runbook env/kaggle_run_p2_s3_recovery.py \\
        --fixture p2_s3_recovery
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
REPO_ROOT = THESIS_DBP.parent
CELL_MARKER = "# %%"


@dataclass(frozen=True)
class Cell:
    """One runbook cell."""

    index: int
    title: str
    text: str
    markdown: bool

    @property
    def sha256(self) -> str:
        """Hash of the cell text, so the hand-over can be checked against the dry-run."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def parse_cells(text: str) -> list[Cell]:
    """Split a percent-format file into cells.

    Raises:
        ValueError: If code appears before the first ``# %%`` marker -- it would run in
            the dry-run but belong to no cell the user can paste.
    """
    cells: list[Cell] = []
    title, body, started = "", [], False
    preamble: list[str] = []

    def flush() -> None:
        if started:
            cells.append(Cell(len(cells), title, "\n".join(body).strip("\n") + "\n",
                              "[markdown]" in title))

    for line in text.splitlines():
        if line.startswith(CELL_MARKER):
            flush()
            title, body, started = line[len(CELL_MARKER):].strip(), [], True
        elif started:
            body.append(line)
        elif line.strip():
            preamble.append(line)
    flush()
    if preamble:
        raise ValueError(f"text before the first {CELL_MARKER!r} marker: {preamble[:2]}")
    return cells


def code_cells(cells: list[Cell]) -> list[Cell]:
    """The cells a notebook would execute."""
    return [c for c in cells if not c.markdown]


# ---------------------------------------------------------------- fixtures ---


@dataclass
class Session:
    """One execution of the runbook: extra environment, and the exit it must end with."""

    label: str
    env: dict[str, str] = field(default_factory=dict)
    expect_exit: str = "zero"  # "zero" or "nonzero"


@dataclass
class Fixture:
    """What a fixture builder returns."""

    env: dict[str, str]
    commit: str
    sessions: list[Session]
    carry_to: Path | None = None
    check: Callable[[Path], list[str]] | None = None


def copy_code(destination: Path, ref: str) -> str:
    """Write the code dataset from git's own bytes, exactly as the zip will hold it.

    Built with :func:`build_code_zip.extract`, the function that builds the upload, so
    the dry-run tests the bytes Kaggle receives -- including their line endings, which
    the working tree on Windows does not reliably preserve.
    """
    sys.path.insert(0, str(THESIS_DBP / "scripts"))
    import build_code_zip

    return build_code_zip.extract(ref, destination)


def write_png(path: Path, seed: int, size: int = 384) -> None:
    """A random RGB PNG, large enough for the 320 px evaluation crop."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB").save(path)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write a CSV with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_script(path: Path, name: str) -> ModuleType:
    """Import a script from the fixture's code copy, not from the live repository."""
    sys.path.insert(0, str(path.parent.parent / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def fixture_p2_s3_recovery(work: Path, ref: str) -> dict[str, str]:
    """The S3 recovery runbook's inputs, in miniature.

    One finished member (``sd15``) stands for the nine; one member to run
    (``realesrgan_x4``, the one whose missing module stopped the first commit) stands
    for the three. The probe list has the real one's shape: panel rows first, then
    corpus rows, including a pair and a ``gen_matched`` row the run must regenerate.
    """
    inputs = work / "input"
    code = inputs / "datasets/adeelshah29/dbp-code/thesis_dbp"
    bundle = inputs / "datasets/adeelshah29/dbp-p2-s3-partial"
    panel = inputs / "notebooks/adeelshah29/phase-2/panel"
    p1_data = inputs / "datasets/adeelshah29/p1-data"
    pairs = (inputs / "datasets/muhammadadeelshah/thesis-dbp-1-dataset"
             "/Implementation/thesis_dbp/outputs/p1/pairs")

    commit = copy_code(code, ref)

    # --- pixels -------------------------------------------------------------------
    write_png(panel / "e3_matched" / "fx_panel_0.png", 1)
    write_png(panel / "e3_matched" / "fx_panel_1.png", 2)
    real_id = "fx_real_c0"
    write_png(p1_data / "p1_reals" / "t_real" / f"{real_id}.png", 3)
    write_png(pairs / "sd15" / f"{real_id}.png", 4)
    generations = ["coco_000000000009", "coco_000000000025"]
    for seed, gen_id in enumerate(generations, start=10):
        write_png(p1_data / "p1_generated" / "g_sd15" / f"{gen_id}.png", seed)

    # --- the generation manifest the matcher reads (columns it actually uses) --------
    (bundle / "reports").mkdir(parents=True, exist_ok=True)
    write_csv(bundle / "reports" / "p1_gen_manifest.csv",
              ["image_id", "set_id", "label", "relative_path"],
              [{"image_id": g, "set_id": "g_sd15", "label": "fake",
                "relative_path": f"generated/g_sd15/{g}.png"} for g in generations])

    # --- the committed reference: run the real matcher once, from the fixture copy -----
    scratch = work / "reference_build"
    staged = scratch / "p1"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copytree(p1_data / "p1_generated", staged / "generated")
    result = subprocess.run(
        [sys.executable, str(code / "scripts" / "p1_match_generations.py"),
         "--gen-manifest", str(bundle / "reports" / "p1_gen_manifest.csv"),
         "--reals-manifest", str(code / "reports" / "p1_reals_manifest_t_real.csv"),
         "--gen-root", str(staged / "generated"), "--out-root", str(staged),
         "--set-id", "gen_matched"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fixture matcher failed:\n{result.stdout}\n{result.stderr}")
    # The matcher wrote its manifest into the fixture code copy's reports/: that file is
    # now the committed reference the runbook verifies against, exactly as on Kaggle.

    # --- the bundle: one finished member, the image list, a stand-in S1 cache ----------
    rows = [
        {"image_id": "fx_panel_0", "relative_path": "e3_matched/fx_panel_0.png",
         "label": "fake", "view": "e3_matched", "root": "panel"},
        {"image_id": "fx_panel_1", "relative_path": "e3_matched/fx_panel_1.png",
         "label": "real", "view": "e3_matched", "root": "panel"},
        {"image_id": f"{real_id}__sd15", "relative_path": f"pairs/sd15/{real_id}.png",
         "label": "fake", "view": "train", "root": "corpus"},
        {"image_id": generations[0], "relative_path": f"gen_matched/{generations[0]}.jpg",
         "label": "fake", "view": "train", "root": "corpus"},
    ]
    write_csv(bundle / "reports" / "p2_probe_images.csv",
              ["image_id", "relative_path", "label", "view", "root"], rows)

    probe = load_script(code / "scripts" / "p2_probe_features.py", "fixture_probe")
    digest = probe.image_list_digest(rows)
    finished = bundle / "probe" / "p2_probe_sd15.npz"
    finished.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(finished, features=np.zeros((len(rows), len(probe.FEATURE_NAMES)),
                                                    dtype=np.float32),
                        image_ids=np.array([r["image_id"] for r in rows]),
                        feature_names=np.array(probe.FEATURE_NAMES), member="sd15")
    probe.write_sidecar(finished, len(rows), probe.CROP_SIZE, digest, "sd15")
    (bundle / "s1").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bundle / "s1" / "p2_s1_dinov3-vitb16_train.npz",
                        features=np.zeros((2, 4), dtype=np.float16),
                        image_ids=np.array(["a", "b"]))

    expect = {
        "stage": "s3_recovery (dry-run fixture)",
        "imports": ["spandrel", "lpips", "peft", "diffusers", "transformers"],
        "corpus": {"reals": 1, "generated": len(generations), "pairs": 1},
        "restored_members": 1,
        "probe_rows": len(rows),
        "list_digest": digest,
        "gen_matched": len(generations),
        "smoke_members": ["realesrgan_x4"],
        "smoke_images": 2,
        "jobs": [["cpu", "realesrgan_x4"]],
        "deadline_hours": 0.5,
        "poll_seconds": 2,
        "expected": {"setup_wall_hours": 0, "s3_wall_hours": 0, "quota_hours": 0,
                     "basis": "dry-run fixture"},
    }
    expect_path = work / "expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1")])


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The S2 dry-run: every run is killed right after its step-4 checkpoint in session 1,
#: and must resume at step 5 in session 2 and finish at step 6.
S2_STEPS, S2_KILL_AFTER = 6, 4
S2_RUNS = ["p_a_sd1:0", "4ch_only:0"]


def fixture_p2_s2(work: Path, ref: str) -> Fixture:
    """Session D's inputs, in miniature, run in two sessions: killed, then resumed.

    Six real reals from the committed ``t_real`` manifest (four train, two val), each
    with an ``sd15`` and an ``sd21`` reconstruction; two generations that the runbook
    turns into ``gen_matched``; a three-row panel and a one-row luma companion. The
    backbone is the random-init dry-run DINOv3 (``DBP_DRYRUN=1``).
    """
    inputs = work / "input"
    # The real upload is a per-commit slug (dbp-code-ed7a8cb); the fixture mirrors it.
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/adeelshah29/dbp-p2-s2-inputs"
    panel = inputs / "notebooks/adeelshah29/phase-2/panel"
    p1_data = inputs / "datasets/muhammadadeelshah/report"  # as on Kaggle (I-058)
    pairs = (inputs / "datasets/muhammadadeelshah/thesis-dbp-1-dataset"
             "/Implementation/thesis_dbp/outputs/p1/pairs")
    commit = copy_code(code, ref)

    # --- reals from the committed manifest, so the trainer's D-039 filter keeps them ---
    with (code / "reports" / "p1_reals_manifest_t_real.csv").open(encoding="utf-8") as handle:
        reals = [r for r in csv.DictReader(handle) if not r.get("excluded_reason")][:6]
    sides = ["train"] * 4 + ["val"] * 2
    members = ["sd15", "sd21"]
    split_rows, pair_rows, s1_rows = [], [], []
    for seed, (real, side) in enumerate(zip(reals, sides, strict=True)):
        rid = real["image_id"]
        write_png(p1_data / "p1_reals" / Path(real["relative_path"]).relative_to("reals"), seed)
        split_rows.append({"row_id": rid, "set_id": "t_real", "kind": "real",
                           "content_id": rid, "group_id": rid, "stratum": "x", "split": side})
        s1_rows.append({"image_id": rid, "relative_path": real["relative_path"],
                        "label": "real", "view": side, "root": "corpus"})
        for member in members:
            pair_id = f"{rid}__{member}"
            rel = f"pairs/{member}/{rid}.png"
            write_png(pairs / member / f"{rid}.png", 100 + seed)
            pair_rows.append({"pair_id": pair_id, "image_id": rid, "set_id": "t_real",
                              "member": member, "relative_path": rel})
            split_rows.append({"row_id": pair_id, "set_id": "t_real", "kind": "fake",
                               "content_id": rid, "group_id": rid, "stratum": "x",
                               "split": side})
            s1_rows.append({"image_id": pair_id, "relative_path": rel, "label": "fake",
                            "view": side, "root": "corpus"})

    generations = ["coco_000000000009", "coco_000000000025"]
    for seed, gen_id in enumerate(generations, start=10):
        write_png(p1_data / "p1_generated" / "g_sd15" / f"{gen_id}.png", seed)
        split_rows.append({"row_id": gen_id, "set_id": "g_sd15", "kind": "fake",
                           "content_id": gen_id, "group_id": gen_id, "stratum": "x",
                           "split": "train"})

    # --- the panel and its luma companion -----------------------------------------------
    write_png(panel / "e3_matched" / "fx_p0.png", 30)
    write_png(panel / "e3_matched" / "fx_p1.png", 31)
    write_png(panel / "e7_dda_coco" / "fx_p2.png", 32)
    write_png(panel / "e7_dda_coco_luma" / "fx_p2.png", 33)
    panel_rows = [
        {"image_id": "fx_p0", "relative_path": "e3_matched/fx_p0.png", "label": "fake",
         "view": "e3_matched", "root": "panel"},
        {"image_id": "fx_p1", "relative_path": "e3_matched/fx_p1.png", "label": "real",
         "view": "e3_matched", "root": "panel"},
        {"image_id": "fx_p2", "relative_path": "e7_dda_coco/fx_p2.png", "label": "fake",
         "view": "e7_dda_coco", "root": "panel"},
    ]
    luma_rows = [{"image_id": "fx_p2__luma", "relative_path": "e7_dda_coco_luma/fx_p2.png",
                  "label": "fake", "view": "e7_dda_coco_luma", "root": "panel"}]

    # --- the inputs bundle ----------------------------------------------------------------
    list_cols = ["image_id", "relative_path", "label", "view", "root"]
    write_csv(s2in / "p2_panel_images.csv", list_cols, panel_rows)
    write_csv(s2in / "p2_luma_images.csv", list_cols, luma_rows)
    write_csv(s2in / "p2_s1_images.csv", list_cols, s1_rows)
    write_csv(s2in / "p2_splits.csv", ["row_id", "set_id", "kind", "content_id", "group_id",
                                       "stratum", "split"], split_rows)
    write_csv(s2in / "p1_pairs_manifest.csv",
              ["pair_id", "image_id", "set_id", "member", "relative_path"], pair_rows)
    (s2in / "p2_splits_summary.json").write_text('{"fixture": true}\n', encoding="utf-8")
    write_csv(s2in / "p1_gen_manifest.csv", ["image_id", "set_id", "label", "relative_path"],
              [{"image_id": g, "set_id": "g_sd15", "label": "fake",
                "relative_path": f"generated/g_sd15/{g}.png"} for g in generations])

    # --- the committed gen_matched reference: the real matcher, run once ------------------
    staged = work / "reference_build" / "p1"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copytree(p1_data / "p1_generated", staged / "generated")
    result = subprocess.run(
        [sys.executable, str(code / "scripts" / "p1_match_generations.py"),
         "--gen-manifest", str(s2in / "p1_gen_manifest.csv"),
         "--reals-manifest", str(code / "reports" / "p1_reals_manifest_t_real.csv"),
         "--gen-root", str(staged / "generated"), "--out-root", str(staged),
         "--set-id", "gen_matched"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fixture matcher failed:\n{result.stdout}\n{result.stderr}")

    counted = {"p1_pairs_manifest.csv": pair_rows, "p2_splits.csv": split_rows,
               "p1_gen_manifest.csv": generations, "p2_panel_images.csv": panel_rows,
               "p2_luma_images.csv": luma_rows, "p2_s1_images.csv": s1_rows}
    expect = {
        "stage": "p2_s2 (dry-run fixture)",
        "imports": ["peft", "transformers", "safetensors", "huggingface_hub", "PIL"],
        "gpus": 0,
        "hf_token": False,
        "corpus": {"reals": len(reals), "generated": len(generations),
                   "pairs": len(pair_rows)},
        "gen_matched": len(generations),
        "inputs": {
            name: {"sha256": sha256_file(s2in / name),
                   **({"rows": len(counted[name])} if name in counted else {})}
            for name in sorted(p.name for p in s2in.iterdir())
        },
        "resolve": ["p2_panel_images.csv", "p2_luma_images.csv", "p2_s1_images.csv",
                    "p1_pairs_manifest.csv"],
        "runs": S2_RUNS,
        "devices": "cpu",
        "lists": {"panel": ["p2_panel_images.csv", None], "luma": ["p2_luma_images.csv", None],
                  "val": ["p2_s1_images.csv", "val"]},
        "train_args": [f"--dryrun-steps={S2_STEPS}", "--checkpoint-every=2", "--eval-every=2",
                       "--val-pairs=4", "--calibrate=1"],
        "infer_args": ["--batch=4"],
        "smoke_steps": 2,
        "smoke_deadline_hours": 0.5,
        "deadline_hours": 1.0,
        "poll_seconds": 2,
        "expected": {"total_wall_hours": 0, "basis": "dry-run fixture"},
    }
    expect_path = work / "expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    carry_to = inputs / "notebooks/adeelshah29/dbp-p2-s2-session-1"
    val_rows = sum(1 for r in s1_rows if r["view"] == "val")

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError) as exc:  # reported as a failed check
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        first = carry_to / "Implementation/thesis_dbp/reports/p2_s2_status.json"
        state = json.loads(first.read_text(encoding="utf-8"))
        if state["outcome"] != "RUN-FAILED" or any(r["state"] != "failed" for r in state["runs"]):
            problems.append(f"session 1 should have failed every run: {state['outcome']}")
        reports = work_dir / "working/Implementation/thesis_dbp/reports"
        state = json.loads((reports / "p2_s2_status.json").read_text(encoding="utf-8"))
        if state["outcome"] != "COMPLETE":
            problems.append(f"session 2 outcome {state['outcome']}")
        for run in S2_RUNS:
            protocol, seed = run.split(":")
            rid = f"{protocol}_seed{seed}"
            trained = json.loads((reports / f"p2_s2_{rid}.json").read_text(encoding="utf-8"))
            if trained.get("resumed_from_step") != S2_KILL_AFTER + 1:
                problems.append(f"{rid} resumed from {trained.get('resumed_from_step')}, "
                                f"expected step {S2_KILL_AFTER + 1}")
            if trained.get("steps_run") != S2_STEPS or trained.get("status") != "OK":
                problems.append(f"{rid}: status {trained.get('status')}, "
                                f"steps {trained.get('steps_run')} (expected {S2_STEPS})")
            scored = json.loads((reports / f"p2_s2_scores_{rid}.json").read_text(encoding="utf-8"))
            rows = {k: v["rows"] for k, v in scored["lists"].items()}
            want = {"panel": len(panel_rows), "luma": len(luma_rows), "val": val_rows}
            if rows != want:
                problems.append(f"{rid} scored {rows}, expected {want}")
            print(f"  check {rid}: killed after step {S2_KILL_AFTER}, resumed at "
                  f"{trained.get('resumed_from_step')}, finished at {trained.get('steps_run')}; "
                  f"scored {rows}")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
        [Session("session 1: every run killed after its step-4 checkpoint",
                 {"DBP_FAULT_EXIT_AFTER_STEP": str(S2_KILL_AFTER)}, "nonzero"),
         Session("session 2: the previous output attached; every run resumes")],
        carry_to=carry_to, check=check,
    )


def fixture_x1_format(work: Path, ref: str) -> Fixture:
    """The X1 format-control session in miniature, on Session D's fixture inputs.

    The image list mixes every root and kind X1 uses: corpus reals and reconstructions
    (train and val), the ``gen_matched`` JPEGs the runbook regenerates, and an E7 row
    from the panel. Both arms run on the random-init dry-run backbone; the check is that
    both caches hold every row and that re-encoding changed the features.
    """
    base = fixture_p2_s2(work, ref)
    s2in = work / "input/datasets/adeelshah29/dbp-p2-s2-inputs"
    s2_expect = json.loads((work / "expect.json").read_text(encoding="utf-8"))
    with (s2in / "p2_s1_images.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (s2in / "p1_gen_manifest.csv").open(encoding="utf-8") as handle:
        generations = [r["image_id"] for r in csv.DictReader(handle)]
    rows += [{"image_id": g, "relative_path": f"gen_matched/{g}.jpg", "label": "fake",
              "view": "train", "root": "corpus"} for g in generations]
    rows.append({"image_id": "fx_p2", "relative_path": "e7_dda_coco/fx_p2.png",
                 "label": "fake", "view": "e7_dda_coco", "root": "panel"})
    images = work / "x1_format_control_images.csv"
    write_csv(images, ["image_id", "relative_path", "label", "view", "root"], rows)
    expect = {
        "stage": "x1_format_control (dry-run fixture)",
        "imports": ["transformers", "safetensors", "huggingface_hub", "PIL"],
        "hf_token": False,
        "corpus": s2_expect["corpus"],
        "gen_matched": s2_expect["gen_matched"],
        "gen_manifest": {"name": "p1_gen_manifest.csv",
                         "sha256": sha256_file(s2in / "p1_gen_manifest.csv")},
        "images": str(images),
        "images_rows": len(rows),
        "images_sha256": sha256_file(images),
        "quality": 95,
        "batch": 4,
        "smoke_images": 2,
        "expected": {"total_wall_hours_t4": 0, "basis": "dry-run fixture"},
    }
    expect_path = work / "x1_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        out = work_dir / "working/Implementation/thesis_dbp/outputs/x1_format"
        problems: list[str] = []
        try:
            arms = {}
            for tag in ("x1_asis", "x1_jpeg95"):
                with np.load(out / f"p2_s1_dinov3-vitb16_{tag}.npz") as blob:
                    arms[tag] = (list(blob["image_ids"]), blob["features"].astype(np.float32))
        except OSError as exc:
            return [f"an arm's cache is missing: {exc}"]
        for tag, (ids, features) in arms.items():
            if len(ids) != len(rows) or features.shape[0] != len(rows):
                problems.append(f"{tag}: {len(ids)} rows, expected {len(rows)}")
        if arms["x1_asis"][0] != arms["x1_jpeg95"][0]:
            problems.append("the two arms hold different ids")
        changed = float(np.max(np.abs(arms["x1_asis"][1] - arms["x1_jpeg95"][1])))
        if not changed > 0:
            problems.append("re-encoding left every feature unchanged")
        print(f"  check: both arms hold {len(rows)} rows; max |as-is - jpeg95| = {changed:.4f}")
        return problems

    return Fixture({**base.env, "DBP_EXPECT": str(expect_path)}, base.commit,
                   [Session("session 1")], check=check)


def write_image(path: Path, seed: int, size: tuple[int, int], fmt: str,
                quality: int = 85) -> None:
    """A random RGB image of ``size`` (w, h), PNG or JPEG."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(array, "RGB")
    if fmt == "JPEG":
        image.save(path, format="JPEG", quality=quality, subsampling=2)
    else:
        image.save(path, format="PNG")


def fixture_p3_panel_b(work: Path, ref: str) -> Fixture:
    """The Panel B session in miniature, on CPU, with no network.

    Five tiny evaluation sets laid out exactly as P1's recorded globs expect (E3
    `images/Raw/<folder>/<label>/<n>`, E4 `test/<platform>/{0_real,1_fake}`, E5
    `test/{0_real,1_fake}`, E7 `<vae>/val2017/`, E8 `<top>/<generator>/`),
    passed to the fetch step with ``--local`` instead of being downloaded. E7's reals live
    in a separate COCO `val2017` input, as on Kaggle: DDA-COCO ships none, and the fetch
    step must copy them into `real/` (the first Kaggle run, 2026-10-11, failed on exactly
    that, because this fixture had put them in `real/` itself). The fixture runs
    ``p1_build_eval.py`` once itself to learn the rebuilt rows, takes a few of them as the
    "P2 panel" (including an E7 COCO id and an E3 number, so both exclusions are
    exercised), and gives the draw small slots.
    """
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/muhammadadeelshah/dbp-p2-s2-inputs"
    chameleon = inputs / "datasets/muhammadadeelshah/chameleon-dataset/Chameleon"
    coco_val = inputs / "datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    sources = work / "sources"
    commit = copy_code(code, ref)
    seed = iter(range(10_000, 10**6))

    # --- E3: reals JPEG, fakes PNG (confounded, so the matched view is built) ------------
    for folder in ("Texture Images", "Structure Images", "Hybrid Images"):
        for n in range(1, 5):
            write_image(sources / "e3/images/Raw" / folder / "real" / f"{n}.jpg", next(seed),
                        (480, 400), "JPEG")
            for gen in ("GPT-Image2-fake", "Nano-Banana2-fake"):
                write_image(sources / "e3/images/Raw" / folder / gen / f"{n}.png", next(seed),
                            (480, 400), "PNG")
    # --- E4: reals JPEG, fakes PNG -----------------------------------------------------------
    for platform in ("reddit", "twitter", "facebook"):
        for n in range(4):
            write_image(sources / "wildrf/WildRF/test" / platform / "0_real" / f"r{n}.jpg",
                        next(seed), (448, 400), "JPEG")
            write_image(sources / "wildrf/WildRF/test" / platform / "1_fake" / f"f{n}.png",
                        next(seed), (448, 400), "PNG")
    # --- E5: reals 500 px wide q75 JPEG, fakes larger PNG ------------------------------------
    for n in range(12):
        write_image(chameleon / "test/0_real" / f"r{n}.jpg", next(seed), (500, 375), "JPEG", 75)
        write_image(chameleon / "test/1_fake" / f"f{n}.png", next(seed), (800, 600), "PNG")
    # --- E7: one COCO id set across the reals and six VAE subsets -----------------------------
    vaes = ("FLUX.1", "sd-vae-ft-ema", "sd-vae-ft-mse", "sdxl-vae", "stable-diffusion-2-1",
            "stable-diffusion-3.5-large")
    for n in range(16):
        cid = f"{n:012d}"
        write_image(coco_val / f"{cid}.jpg", next(seed), (512, 384), "JPEG", 96)
        for vae in vaes:
            write_image(sources / "e7x/DDA-COCO" / vae / "val2017" / f"{cid}.jpg", next(seed),
                        (512, 384), "JPEG", 96)
    # --- E8: fakes only ----------------------------------------------------------------------
    for gen in ("Flux", "GoT", "Infinity", "NOVA", "OmniGen"):
        for n in range(4):
            write_image(sources / "e8x/EvalGEN" / gen / f"{n}.jpg", next(seed), (512, 512),
                        "JPEG", 96)

    roots = {"e3": sources / "e3", "e4": sources / "wildrf/WildRF", "e5": chameleon,
             "e7": sources / "e7x/DDA-COCO", "e8": sources / "e8x"}

    # --- learn the rebuilt rows by running P1's builder once, from the fixture code ---------
    # E7's reals go into real/ only for this learning pass and are removed again, so the
    # runbook's fetch step has to put them there itself.
    learn_real = sources / "e7x/DDA-COCO/real"
    shutil.copytree(coco_val, learn_real)
    views = load_script(code / "scripts" / "p3_build_eval_views.py", "fixture_views")
    for name, (set_id, _view, _manifest) in views.VIEWS.items():
        summary_path = code / "reports/kaggle_20260928" / f"p1_eval_{set_id}_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result = subprocess.run(
            [sys.executable, str(code / "scripts" / "p1_build_eval.py"),
             *views.p1_arguments(summary, str(roots[name]))],
            capture_output=True, text=True, cwd=code)
        if result.returncode != 0:
            raise RuntimeError(f"fixture p1_build_eval {set_id} failed:\n{result.stdout[-2000:]}"
                               f"\n{result.stderr[-2000:]}")
    shutil.rmtree(learn_real)

    # --- the fixture "P2 panel": a few real rebuilt rows per view -----------------------------
    p2_dir = work / "p2_panel_sets"
    p2_dir.mkdir(parents=True, exist_ok=True)
    for _name, (_set_id, view, manifest) in views.VIEWS.items():
        with (code / "reports" / manifest).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        picked = [r for r in rows if r["label"] == "real"][:2] + \
            [r for r in rows if r["label"] == "fake"][:2]
        write_csv(p2_dir / f"p2_panel_{view}.csv", list(rows[0]), picked)
    write_csv(p2_dir / "p2_panel_t_ctrl_diagnostic.csv", ["image_id", "sha256", "phash"], [])

    # --- dedup reference manifests (the real ones are much larger) ---------------------------
    write_csv(s2in / "p1_gen_manifest.csv", ["image_id", "sha256", "phash"],
              [{"image_id": "g0", "sha256": "0" * 64, "phash": "0123456789abcdef"}])
    write_csv(s2in / "p1_pairs_manifest.csv", ["pair_id", "sha256"],
              [{"pair_id": "p0", "sha256": "1" * 64}])

    slots = {
        "e3_matched": [[gen, "fake", "source_label", gen, 4, "test"]
                       for gen in ("GPT-Image2-fake", "Nano-Banana2-fake")] + [
                       ["real-calibration", "real", None, None, 2, "calibration"],
                       ["real-test", "real", None, None, None, "test"]],
        "e4_wildrf_matched": [["fake", "fake", None, None, None, "test"],
                              ["real-calibration", "real", None, None, 2, "calibration"],
                              ["real-test", "real", None, None, None, "test"]],
        "e5_chameleon_pipeline": [["fake", "fake", None, None, 5, "test"],
                                  ["real-calibration", "real", None, None, 3, "calibration"],
                                  ["real-test", "real", None, None, 5, "test"]],
        "e7_dda_coco": [*[[v, "fake", "source_vae", v, 2, "test"] for v in vaes],
                        ["real-calibration", "real", None, None, 1, "calibration"],
                        ["real-test", "real", None, None, 1, "test"]],
        "e8_evalgen": [[g, "fake", "generator", g, 2, "test"]
                       for g in ("Flux", "GoT", "Infinity", "NOVA", "OmniGen")],
    }
    slots_path = work / "p3_slots.json"
    slots_path.write_text(json.dumps(slots, indent=2) + "\n", encoding="utf-8")
    expect = json.loads((code / "env" / "p3_panel_b_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_panel_b (dry-run fixture)",
        "inputs": {name: {"sha256": sha256_file(s2in / name), "rows": 1}
                   for name in ("p1_gen_manifest.csv", "p1_pairs_manifest.csv")},
        "local": {name: str(path) for name, path in roots.items() if name != "e5"},
        "p2_panel_sets": str(p2_dir),
        "label_check": False,
        "coco_val_count": 16,
        "slots": str(slots_path),
        "expected": {"wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        reports = work_dir / "working/Implementation/thesis_dbp/reports"
        problems: list[str] = []
        try:
            digest = json.loads((reports / "p3_panel_b_digest.json").read_text(encoding="utf-8"))
            with (reports / "p3_panel_b_images.csv").open(encoding="utf-8") as handle:
                listed = list(csv.DictReader(handle))
        except (OSError, ValueError) as exc:
            return [f"Panel B was not frozen: {exc}"]
        p2_ids = set()
        for path in p2_dir.glob("p2_panel_*.csv"):
            with path.open(encoding="utf-8") as handle:
                p2_ids |= {r["image_id"] for r in csv.DictReader(handle)}
        if p2_ids & {r["image_id"] for r in listed}:
            problems.append("a P2 panel row entered Panel B")
        panel_root = work_dir / "working/panel_b"
        missing = [r["relative_path"] for r in listed
                   if not (panel_root / r["relative_path"]).is_file()]
        if missing:
            problems.append(f"{len(missing)} listed image(s) not materialised")
        if (work_dir / "working/Implementation/thesis_dbp/outputs/p1/eval").exists():
            problems.append("the derived view images were not cleaned up")
        roles = Counter(r["role"] for r in listed)
        print(f"  check: Panel B {digest['rows']} rows {dict(roles)}, luma {digest['luma_rows']}, "
              f"digest {digest['panel_digest'][:16]}, {digest['panel_bytes']} bytes")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1")], check=check)


def write_photo(path: Path, seed: int, size: tuple[int, int], quality: int = 90,
                grey: bool = False) -> None:
    """A smooth, photo-like JPEG (low-pass noise), so pHash is stable across re-encodes."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    small = rng.integers(0, 256, (h // 32 + 2, w // 32 + 2, 1 if grey else 3), dtype=np.uint8)
    image = Image.fromarray(small.squeeze(-1) if grey else small).resize((w, h), Image.BICUBIC)
    image.convert("RGB").save(path, format="JPEG", quality=quality)


def fixture_p3_open_images(work: Path, ref: str, source: str = "s3") -> Fixture:
    """The Open Images fetch in miniature, on CPU, with no network.

    The image host is a local directory served by ``--local-source`` (dry-run only). With
    ``source="s3"`` (the CVDF copies, 2026-10-15) files are served by ``<ImageID>.jpg`` and
    every walk path runs once: a 404, a greyscale image, one too small for 320 and one in
    the 320-383 band, the four rejected colour modes, an exact copy of a COCO image and of
    an E5 real, a re-encode of an accepted image (pHash), a 429 slow-down and a row after
    the target. ``source="flickr"`` (route (ii), closed) also walks an unapproved host and
    an MD5 mismatch, and serves files by their Flickr name.
    """
    import base64

    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/muhammadadeelshah/dbp-p2-s2-inputs"
    pb_reports = inputs / "datasets/muhammadadeelshah/dbp-p3-panel-b/Implementation/" \
        "thesis_dbp/reports"
    coco = inputs / "datasets/awsaf49/coco-2017-dataset/coco2017"
    flickr = work / "flickr"
    commit = copy_code(code, ref)
    seed = iter(range(20_000, 10**6))

    # --- COCO: a few images per split ------------------------------------------------------
    for split, n in (("train2017", 3), ("val2017", 2)):
        for i in range(n):
            write_photo(coco / split / f"{i:012d}.jpg", next(seed), (640, 480))

    # --- E3/E4/E5 "rebuilt manifests": reals with sha256 + pHash --------------------------
    sys.path.insert(0, str(code / "src"))
    from dbp.data.phash import phash_file, phash_hex

    e5_real = work / "e5_real.jpg"
    write_photo(e5_real, next(seed), (500, 400))
    rebuild = {}
    for name, view, n in (("e3", "e3_matched", 2), ("e4", "e4_wildrf_matched", 2),
                          ("e5", "e5_chameleon_pipeline", 2)):
        rows = []
        for i in range(n):
            img = e5_real if (name == "e5" and i == 0) else work / f"{name}_{i}.jpg"
            if img != e5_real:
                write_photo(img, next(seed), (500, 400))
            rows.append({"image_id": f"{name}{i}", "label": "real", "sha256": sha256_file(img),
                         "phash": phash_hex(phash_file(img))})
        manifest = f"fixture_{name}_manifest.csv"
        write_csv(pb_reports / manifest, ["image_id", "label", "sha256", "phash"], rows)
        rebuild[view] = {"manifest": manifest, "rows": n,
                         "manifest_sha256": sha256_file(pb_reports / manifest)}
    rebuild_path = work / "rebuild.json"
    rebuild_path.write_text(json.dumps(rebuild, indent=2) + "\n", encoding="utf-8")
    write_csv(s2in / "p1_gen_manifest.csv", ["image_id", "sha256", "phash"],
              [{"image_id": "g0", "sha256": "0" * 64, "phash": "0123456789abcdef"}])
    write_csv(s2in / "p1_pairs_manifest.csv", ["pair_id", "sha256"],
              [{"pair_id": "p0", "sha256": "1" * 64}])

    # --- the "Flickr" originals and their metadata -----------------------------------------
    cc = "https://creativecommons.org/licenses/by/2.0/"
    meta: list[dict[str, str]] = []
    expected: dict[str, int] = {"accepted": 0}
    plan: list[tuple[str, Path | None, str, str | None, int]] = []

    def add(kind: str, source: Path | None, host: str = "farm1.staticflickr.com",
            md5: str | None = None, refusals: int = 0) -> None:
        """Queue one row; rows are walked in the order they are queued (ids chosen so)."""
        plan.append((kind, source, host, md5, refusals))
        expected[kind] = expected.get(kind, 0) + 1

    good = []
    for _ in range(7):
        path = work / f"good_{len(good)}.jpg"
        write_photo(path, next(seed), (512, 448))
        good.append(path)
    if source == "flickr":
        for _ in range(3):
            add("host not approved (not contacted)", good[0], host="c9.staticflickr.com")
    add("download: gone (http 404)", None)
    if source == "flickr":
        add("checksum mismatch (re-encode served)", good[1], md5="AAAAAAAAAAAAAAAAAAAAAA==")
    grey = work / "grey.jpg"
    write_photo(grey, next(seed), (512, 448), grey=True)
    add("greyscale", grey)
    small = work / "small.jpg"
    write_photo(small, next(seed), (300, 300))
    add("shorter side < 320", small)
    band = work / "band.jpg"
    write_photo(band, next(seed), (500, 350))
    add("shorter side 320-383 (below the 384 COCO crop)", band)
    # Colour modes are rejected, never converted (user rule 2026-10-14), one row each.
    from PIL import Image as _Image

    base_rgb = work / "mode_base.jpg"
    write_photo(base_rgb, next(seed), (512, 448))
    with _Image.open(base_rgb) as rgb:
        rgb.convert("CMYK").save(work / "mode_cmyk.jpg", format="JPEG", quality=90)
        rgb.convert("P").save(work / "mode_palette.png", format="PNG")
        rgb.convert("RGBA").save(work / "mode_alpha.png", format="PNG")
        (np.asarray(rgb.convert("L"), dtype=np.uint16) * 257).astype(np.uint16).tofile(
            work / "mode_16.raw")
    _Image.frombytes("I;16", (512, 448), (work / "mode_16.raw").read_bytes()).save(
        work / "mode_16bit.png", format="PNG")
    for kind, name in (("CMYK", "mode_cmyk.jpg"), ("palette", "mode_palette.png"),
                       ("alpha", "mode_alpha.png"), ("16-bit or float", "mode_16bit.png")):
        add(f"colour mode: {kind} (rejected, not converted)", work / name)
    add("COCO duplicate", coco / "val2017" / f"{0:012d}.jpg")
    add("E5 duplicate", e5_real)
    from PIL import Image

    add("accepted", good[2], host="c2.staticflickr.com")
    reenc = work / "reencoded.jpg"
    with Image.open(good[2]) as image:
        image.save(reenc, format="JPEG", quality=75)
    add("pHash duplicate of an accepted image", reenc)
    add("accepted", good[0])
    add("accepted", good[1], host="c5.staticflickr.com")
    add("accepted", good[3], refusals=2)                  # answers 429 twice, then the file
    add("accepted", good[4], host="farm8.staticflickr.com")
    add("accepted", good[5], host="c8.staticflickr.com")
    add("not needed (target reached)", good[6])          # walked after the target is met

    from dbp.data.open_images import order_key

    ids = sorted((f"{n:016x}" for n in range(len(plan))), key=order_key)
    for image_id, (_kind, origin, host, md5, refusals) in zip(ids, plan, strict=True):
        name = f"{image_id}_o.jpg"
        served = f"{image_id}.jpg" if source == "s3" else name
        if origin is not None:
            flickr.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, flickr / served)
            if refusals:
                (flickr / f"{served}.429").write_text(str(refusals), encoding="utf-8")
        data = (flickr / served).read_bytes() if origin is not None else b""
        meta.append({"ImageID": image_id, "OriginalURL": f"https://{host}/1/{name}",
                     "License": cc, "Author": f"fixture author {len(meta)}",
                     "OriginalSize": str(len(data)),
                     "OriginalMD5": md5 or base64.b64encode(hashlib.md5(data).digest()).decode(),
                     "Rotation": ""})
    meta_path = work / "oi_meta.csv"
    write_csv(meta_path, ["ImageID", "OriginalURL", "License", "Author", "OriginalSize",
                          "OriginalMD5", "Rotation"], meta)

    expect = json.loads((code / "env" / "p3_open_images_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_open_images (dry-run fixture)",
        "inputs": {name: {"sha256": sha256_file(s2in / name), "rows": 1}
                   for name in ("p1_gen_manifest.csv", "p1_pairs_manifest.csv")},
        "metadata": {"path": str(meta_path), "sha256": sha256_file(meta_path), "rows": len(meta)},
        "rebuild_record": str(rebuild_path),
        "coco_counts": {"train2017": 3, "val2017": 2},
        "target": expected["accepted"],
        "holdout": 2,
        "min_free_gb": {"tmp": 0.01, "out": 0.01},
        "local_source": str(flickr),
        "source": source,
        **({"hosts": [r"farm[1-8]\.staticflickr\.com", r"c[1-8]\.staticflickr\.com"]}
           if source == "flickr" else {}),
        "expected": {"wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_oi_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        reports = work_dir / "working/Implementation/thesis_dbp/reports"
        problems: list[str] = []
        try:
            summary = json.loads((reports / "p3_open_images_summary.json").read_text("utf-8"))
            coco_summary = json.loads((reports / "p3_coco_index_summary.json").read_text("utf-8"))
            with (reports / "p3_open_images_manifest.csv").open(encoding="utf-8") as handle:
                manifest = list(csv.DictReader(handle))
            with (reports / "p3_open_images_attribution.csv").open(encoding="utf-8") as handle:
                attribution = list(csv.DictReader(handle))
        except (OSError, ValueError) as exc:
            return [f"the draw did not finish: {exc}"]
        rej = summary["rejections"]
        wants = {
            "host not approved (not contacted)": 3 if source == "flickr" else 0,
            "download: gone (http 404)": 1,
            "checksum mismatch (re-encode served)": 1 if source == "flickr" else 0,
            "shorter side < 320": 1,
            "shorter side 320-383 (below the 384 COCO crop)": 1,
            "colour mode: CMYK (rejected, not converted)": 1,
            "colour mode: palette (rejected, not converted)": 1,
            "colour mode: alpha (rejected, not converted)": 1,
            "colour mode: 16-bit or float (rejected, not converted)": 1,
        }
        wants["not needed (target reached)"] = 0          # not a rejection
        walk_log = (reports / "p3_open_images_walk.csv").read_text(encoding="utf-8")
        if walk_log.count("not needed (target reached)") != 1:
            problems.append("the row after the target was not logged as not needed")
        for reason, n in wants.items():
            if n == 0:
                continue
            if rej.get(reason) != n:
                problems.append(f"rejection {reason!r}: {rej.get(reason)} (want {n})")
        if source == "s3" and any(k.startswith(("host not approved", "checksum mismatch"))
                                  for k in rej):
            problems.append(f"host or checksum rejections in s3 mode: {rej}")
        def count(prefix: str) -> int:
            return sum(n for k, n in rej.items() if k.startswith(prefix))
        if count("greyscale") != 1:
            problems.append(f"greyscale rejections {count('greyscale')} (want 1)")
        if count("duplicate of a reference (sha:COCO val2017)") != 1:
            problems.append("the COCO copy was not rejected against COCO val2017")
        if count("duplicate of a reference (sha:e5 reals)") != 1:
            problems.append("the E5-real copy was not rejected against the E5 reals")
        if count("duplicate of a reference (phash:Open Images (accepted))") != 1:
            problems.append("the re-encode was not caught as a pHash duplicate")
        if len(manifest) != expected["accepted"] or len(attribution) != expected["accepted"]:
            problems.append(f"{len(manifest)} manifest / {len(attribution)} attribution rows")
        if list(attribution[0]) != ["image_id", "author", "original_url", "licence"]:
            problems.append(f"attribution columns {list(attribution[0])}")
        if summary["split"] != {"train": expected["accepted"] - 2, "held_out": 2} and \
                summary["split"] != {"held_out": 2, "train": expected["accepted"] - 2}:
            problems.append(f"split {summary['split']}")
        from PIL import Image as _Image

        for row in manifest:
            path = work_dir / "working/oi" / row["relative_path"]
            if not path.is_file():
                problems.append(f"crop missing: {row['relative_path']}")
                continue
            with _Image.open(path) as image:
                if image.size != (384, 384) or image.format != "PNG":
                    problems.append(f"crop {path.name}: {image.size} {image.format}")
        crops = list((work_dir / "working/oi/reals/t_oi").glob("*.png"))
        if len(crops) != expected["accepted"]:
            problems.append(f"{len(crops)} crop files on disk (rejected crops not removed?)")
        originals = Path(os.environ["DBP_DRYRUN_TMP"]) / "oi_originals"
        left = list(originals.glob("*")) if originals.exists() else []
        if left:
            problems.append(f"{len(left)} original(s) left in the download directory")
        if len(summary["disk"]) < 3 or len(coco_summary.get("disk", [])) < 2:
            problems.append("df was not logged at every stage")
        groups = summary.get("by_host_group", {})
        if source == "s3":
            if set(groups) != {"s3"} or groups["s3"]["accepted"] != expected["accepted"] \
                    or groups["s3"]["gone"] != 1:
                problems.append(f"s3 group counts {groups}")
            if summary.get("source", {}).get("copy") != "cvdf_s3_rescaled_max1024px":
                problems.append(f"source {summary.get('source')}")
            if any(r["source_copy"] != "cvdf_s3_rescaled_max1024px" or r["md5_checked"] != "False"
                   or not r["fetch_url"].startswith("https://open-images-dataset.s3.amazonaws"
                                                    ".com/validation/") for r in manifest):
                problems.append("manifest rows do not record the S3 copy")
            if summary["hosts"]["redirect_hosts"]:
                problems.append(f"redirects allowed in s3 mode: {summary['hosts']}")
        elif set(groups) != {"farm*", "c*", "other"}:
            problems.append(f"host groups {sorted(groups)}")
        else:
            if groups["c*"]["accepted"] != 3 or groups["farm*"]["accepted"] != 3:
                problems.append(f"accepted per group: farm* {groups['farm*']['accepted']}, "
                                f"c* {groups['c*']['accepted']} (want 3 and 3)")
            if groups["other"]["not_contacted"] != 3 or groups["farm*"]["gone"] != 1 \
                    or groups["farm*"]["checksum_mismatch"] != 1:
                problems.append(f"group counts {groups}")
            if not groups["farm*"]["median_width"]:
                problems.append("no median width for farm*")
        throttle = summary.get("throttle", {})
        if throttle.get("refusals") != 2 or throttle.get("tripped"):
            problems.append(f"throttle {throttle}")
        if summary.get("fetch_policy", {}).get("max_parallel_connections") != 8:
            problems.append("the connection cap is not 8")
        if "e3_reals_with_coco_phash_match" not in coco_summary:
            problems.append("the E3-vs-COCO count is missing")
        print(f"  check: {len(manifest)} accepted {summary['split']}, rejections "
              f"{sum(rej.values())}, disk stages {len(summary['disk'])}, coco rows "
              f"{coco_summary['index_rows']}")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1")], check=check)


def fixture_p3_flickr_probe(work: Path, ref: str) -> Fixture:
    """The Flickr probe in miniature: 3 farm* + 3 c* rows served locally, no network.

    One c* file answers 429 (with a cookie header that must not be recorded), so the
    report carries a refusal; a row on an unapproved host sorts first and must be skipped
    by the per-group pick; no image may be written anywhere.
    """
    import base64

    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    flickr = work / "flickr"
    commit = copy_code(code, ref)
    sys.path.insert(0, str(code / "src"))
    from dbp.data.open_images import order_key

    hosts = ["c9.staticflickr.com"] + ["farm2.staticflickr.com"] * 4 + \
        ["c4.staticflickr.com"] * 4
    ids = sorted((f"{n:016x}" for n in range(len(hosts))), key=order_key)
    meta = []
    for n, (image_id, host) in enumerate(zip(ids, hosts, strict=True)):
        name = f"{image_id}_o.jpg"
        write_photo(flickr / name, 30_000 + n, (512, 448))
        data = (flickr / name).read_bytes()
        meta.append({"ImageID": image_id, "OriginalURL": f"https://{host}/1/{name}",
                     "License": "https://creativecommons.org/licenses/by/2.0/",
                     "Author": f"a{n}", "OriginalSize": str(len(data)),
                     "OriginalMD5": base64.b64encode(hashlib.md5(data).digest()).decode(),
                     "Rotation": ""})
        if host == "c4.staticflickr.com" and n == len(hosts) - 4:
            (flickr / f"{name}.429").write_text("1", encoding="utf-8")
    meta_path = work / "oi_meta.csv"
    write_csv(meta_path, list(meta[0]), meta)
    expect = json.loads((code / "env" / "p3_flickr_probe_expect.json").read_text("utf-8"))
    expect.update({"stage": "p3_flickr_probe (dry-run fixture)",
                   "metadata": {"path": str(meta_path), "sha256": sha256_file(meta_path),
                                "rows": len(meta)},
                   "gap_s": 0, "local_source": str(flickr),
                   "expected": {"wall_minutes": 0, "basis": "dry-run fixture"}})
    expect_path = work / "p3_probe_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        reports = work_dir / "working/Implementation/thesis_dbp/reports"
        try:
            report = json.loads((reports / "p3_flickr_probe.json").read_text("utf-8"))
        except (OSError, ValueError) as exc:
            return [f"no probe report: {exc}"]
        problems = []
        groups = [r["host_group"] for r in report["requests"]]
        if sorted(groups) != ["c*"] * 3 + ["farm*"] * 3:
            problems.append(f"requests per group {groups}")
        if any(r["host"] == "c9.staticflickr.com" for r in report["requests"]):
            problems.append("the unapproved host was contacted")
        refused = [r for r in report["requests"] if r["status"] == 429]
        if len(refused) != 1 or "Set-Cookie" in json.dumps(refused[0]["headers"]):
            problems.append(f"refusal record {refused}")
        if not all(r.get("md5_matches") for r in report["requests"] if r["status"] == 200):
            problems.append("an MD5 did not match")
        images = [p for p in (work_dir / "working").rglob("*")
                  if p.suffix.lower() in (".jpg", ".png") and "thesis_dbp" not in p.parts]
        if images or report["images_kept"] != 0:
            problems.append(f"images were kept: {images[:3]}")
        if not report["egress_ip_range"].startswith("NOT-MEASURED"):
            problems.append("the dry-run must not contact the IP echo")
        print(f"  check: {len(groups)} requests {report['by_host_group']}; "
              f"verdict {report['verdict'][:40]}")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1")], check=check)


def fixture_p3_oi_accept(work: Path, ref: str) -> Fixture:
    """Route (ii) end to end: the LOCAL stage on the Open Images fixture, then the Kaggle
    dedup/accept runbook on its output.

    The local stage (``p3_oi_local_fetch.py``, no network) walks the same 18 rows: it
    rejects what it can check locally (unapproved host, 404, MD5, greyscale, size) and
    keeps 10 candidates, three of which only the Kaggle references can reject (a COCO
    copy, an E5-real copy, a re-encode of an earlier candidate). Its manifest is "committed"
    into the fixture's code copy and its crops are the uploaded dataset, one folder deep.
    """
    fixture_p3_open_images(work, ref, source="flickr")    # inputs, flickr dir, metadata
    oi_expect = json.loads((work / "p3_oi_expect.json").read_text(encoding="utf-8"))
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    local_out, originals = work / "local_candidates", work / "local_originals"
    env = {**os.environ, "DBP_DRYRUN": "1", "PYTHONUTF8": "1"}
    local = subprocess.run(
        [sys.executable, str(code / "scripts" / "p3_oi_local_fetch.py"),
         "--metadata", oi_expect["metadata"]["path"], "--originals", str(originals),
         "--out", str(local_out), "--target-candidates", "10", "--min-free-gb", "0.01",
         "--local-source", oi_expect["local_source"]],
        cwd=code, env=env, capture_output=True, text=True)
    if local.returncode != 0:
        raise RuntimeError(f"fixture local stage failed:\n{local.stdout[-3000:]}"
                           f"\n{local.stderr[-3000:]}")
    manifest = code / "reports" / "p3_open_images" / "p3_oi_candidates_manifest.csv"
    shutil.copyfile(local_out / "p3_oi_candidates_manifest.csv", manifest)
    shutil.copytree(local_out / "reals",
                    inputs / "datasets/muhammadadeelshah/dbp-p3-oi-candidates/"
                    "p3_oi_candidates/reals")
    with manifest.open(encoding="utf-8") as handle:
        rows = sum(1 for _ in csv.DictReader(handle))
    expect = json.loads((code / "env" / "p3_oi_accept_expect.json").read_text("utf-8"))
    expect.update({
        "stage": "p3_oi_accept (dry-run fixture)",
        "inputs": oi_expect["inputs"], "rebuild_record": oi_expect["rebuild_record"],
        "coco_counts": oi_expect["coco_counts"],
        "candidates": {"path": "reports/p3_open_images/p3_oi_candidates_manifest.csv",
                       "sha256": sha256_file(manifest), "rows": rows},
        "target": 6, "holdout": 2,
        "expected": {"wall_minutes": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_accept_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        problems: list[str] = []
        local_summary = json.loads((local_out / "p3_oi_local_summary.json").read_text("utf-8"))
        if local_summary["candidates"] != 10:
            problems.append(f"local candidates {local_summary['candidates']} (want 10)")
        kept = sorted(p.name for p in originals.iterdir())
        if len(kept) != 10:
            problems.append(f"{len(kept)} originals kept (want the 10 candidates only)")
        if local_summary["rejections"].get("host not approved (not contacted)") != 3:
            problems.append("the unapproved host was not refused locally")
        reports = work_dir / "working/Implementation/thesis_dbp/reports"
        try:
            summary = json.loads((reports / "p3_oi_accept_summary.json").read_text("utf-8"))
            with (reports / "p3_open_images_manifest.csv").open(encoding="utf-8") as handle:
                accepted = list(csv.DictReader(handle))
            log = (reports / "p3_oi_accept_log.csv").read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            return [*problems, f"the accept step did not finish: {exc}"]
        want = {"duplicate of a reference (sha:COCO val2017)": 1,
                "duplicate of a reference (sha:e5 reals)": 1,
                "duplicate of a reference (phash:Open Images (accepted))": 1}
        if summary["rejections"] != want:
            problems.append(f"dedup rejections {summary['rejections']}")
        if summary["accepted"] != 6 or summary["split"] not in ({"train": 4, "held_out": 2},
                                                              {"held_out": 2, "train": 4}):
            problems.append(f"accepted {summary['accepted']} {summary['split']}")
        if log.count("not needed (target reached)") != 1:
            problems.append("the candidate after the target was not logged as not needed")
        for row in accepted:
            path = work_dir / "working/oi" / row["relative_path"]
            if not path.is_file() or sha256_file(path) != row["sha256"]:
                problems.append(f"accepted crop missing or changed: {row['relative_path']}")
        print(f"  check: local {local_summary['candidates']} candidates, {len(kept)} originals "
              f"kept; Kaggle accepted {summary['accepted']} {summary['split']}, dedup "
              f"{sum(summary['rejections'].values())}")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)},
                   (code / "COMMIT.txt").read_text(encoding="utf-8").split()[0],
                   [Session("session 1")], check=check)


LOLO_STEPS, LOLO_KILL_AFTER = 6, 4


def fixture_p3_lolo(work: Path, ref: str) -> Fixture:
    """LOLO, two folds per session, in three sessions on Session D's fixture inputs.

    The second reconstruction member becomes ``sdxl``, and ``sd35`` (kl_16ch) and
    ``dcae`` (dc_ae_f32) are added, so folds 1-4 each hold out a lineage that has
    validation reconstructions. Session 1 picks folds 1 and 2 and both are killed after
    their step-4 checkpoints; session 2 (session 1's output attached) resumes both to the
    end; session 3 sees only session 2's output, carries folds 1-2 forward and trains
    folds 3 and 4. On CPU the queue has one worker, so a session's two folds run in turn;
    on T4 x2 they run in parallel, one per GPU.
    """
    base = fixture_p2_s2(work, ref)
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/adeelshah29/dbp-p2-s2-inputs"
    pairs = (inputs / "datasets/muhammadadeelshah/thesis-dbp-1-dataset"
             "/Implementation/thesis_dbp/outputs/p1/pairs")
    os.replace(pairs / "sd21", pairs / "sdxl")
    for name in ("p1_pairs_manifest.csv", "p2_splits.csv", "p2_s1_images.csv"):
        path = s2in / name
        text = path.read_text(encoding="utf-8")
        text = text.replace("__sd21", "__sdxl").replace("pairs/sd21/", "pairs/sdxl/")
        text = text.replace(",sd21,", ",sdxl,")
        path.write_text(text, encoding="utf-8")
    # Two more lineages, so folds 3 and 4 have something to hold out.
    with (s2in / "p1_pairs_manifest.csv").open(encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    with (s2in / "p2_splits.csv").open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    side = {r["row_id"]: r for r in split_rows}
    extra_pairs, extra_splits = [], []
    for row in [r for r in pair_rows if r["member"] == "sd15"]:
        for member in ("sd35", "dcae"):
            pair_id = f"{row['image_id']}__{member}"
            (pairs / member).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pairs / "sd15" / f"{row['image_id']}.png",
                            pairs / member / f"{row['image_id']}.png")
            extra_pairs.append({**row, "pair_id": pair_id, "member": member,
                                "relative_path": f"pairs/{member}/{row['image_id']}.png"})
            extra_splits.append({**side[row["pair_id"]], "row_id": pair_id})
    write_csv(s2in / "p1_pairs_manifest.csv", list(pair_rows[0]), pair_rows + extra_pairs)
    write_csv(s2in / "p2_splits.csv", list(split_rows[0]), split_rows + extra_splits)
    old = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    expect = json.loads((code / "env" / "p3_lolo_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_lolo (dry-run fixture)",
        "imports": old["imports"], "gpus": 0, "hf_token": False,
        "corpus": {**old["corpus"], "pairs": old["corpus"]["pairs"] + len(extra_pairs)},
        "gen_matched": old["gen_matched"],
        "inputs": {name: {**spec, "sha256": sha256_file(s2in / name),
                          **({"rows": spec["rows"] + len(extra_pairs)}
                             if name in ("p1_pairs_manifest.csv", "p2_splits.csv") else {})}
                   for name, spec in old["inputs"].items()
                   if name in ("p1_pairs_manifest.csv", "p2_splits.csv",
                               "p2_splits_summary.json", "p1_gen_manifest.csv")},
        "devices": "cpu", "folds_per_session": 2,
        "train_args": ["--runs-config=configs/p3_runs.yaml", f"--fixed-steps={LOLO_STEPS}",
                       "--heldout-every=2", "--checkpoint-every=2", "--val-pairs=4"],
        "smoke_deadline_hours": 0.5, "deadline_hours": 1.0, "poll_seconds": 2,
        "expected": {"per_fold_wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_lolo_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    carry_to = inputs / "notebooks/muhammadadeelshah/dbp-p3-lolo"

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError) as exc:
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        plan = json.loads((dbp / "reports" / "p3_lolo_plan.json").read_text(encoding="utf-8"))
        if plan["complete"] != ["lolo_f1", "lolo_f2"] or \
                plan["selected"] != ["lolo_f3:0", "lolo_f4:0"]:
            problems.append(f"session 3 plan {plan['complete']} -> {plan['selected']}")
        for fold, resumed in (("lolo_f1", LOLO_KILL_AFTER + 1), ("lolo_f2", LOLO_KILL_AFTER + 1),
                              ("lolo_f3", None), ("lolo_f4", None)):
            report = json.loads((dbp / "reports" / f"p2_s2_{fold}_seed0.json")
                                .read_text(encoding="utf-8"))
            steps = [p["step"] for p in report["curve"]]
            if report["status"] != "OK" or report["steps_run"] != LOLO_STEPS:
                problems.append(f"{fold}: {report['status']} at {report['steps_run']}")
            if steps != [2, 4, 6] or any(p.get("heldout_auc") is None for p in report["curve"]):
                problems.append(f"{fold}: curve steps {steps} or held-out AUC missing")
            if report.get("resumed_from_step") != resumed:
                problems.append(f"{fold}: resumed from {report.get('resumed_from_step')}, "
                                f"expected {resumed}")
            snaps = sorted(p.name for p in (dbp / "outputs/p3/s2" / f"{fold}_seed0"
                                             / "snapshots").iterdir())
            if snaps != ["step_00002", "step_00004", "step_00006"]:
                problems.append(f"{fold}: snapshots {snaps}")
            print(f"  check {fold}: {report['status']}, steps {steps}, resumed "
                  f"{report.get('resumed_from_step')}, held-out "
                  f"{[p['heldout_auc'] for p in report['curve']]}, members "
                  f"{report['held_out_members']}, gen_matched {report['gen_matched_used']}")
        f1 = json.loads((dbp / "reports" / "p2_s2_lolo_f1_seed0.json").read_text("utf-8"))
        if f1["gen_matched_used"] or f1["batch_plan"]["gen_matched_fakes"] != 0:
            problems.append("fold 1 trained with gen_matched")
        if "sd15" not in f1["held_out_members"]:
            problems.append(f"fold 1 held out {f1['held_out_members']}")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, base.commit,
        [Session("session 1: folds 1-2 picked, both killed after their step-4 checkpoints",
                 {"DBP_FAULT_EXIT_AFTER_STEP": str(LOLO_KILL_AFTER)}, "nonzero"),
         Session("session 2: session 1's output attached; folds 1-2 resume and complete"),
         Session("session 3: only session 2's output attached; folds 1-2 carried, folds "
                 "3-4 trained")],
        carry_to=carry_to, check=check,
    )


TRAIN_STEPS, TRAIN_STOP, TRAIN_KILL_AFTER = 6, 5, 4
#: One fixture reconstruction member per lineage the LOLO fixture does not already have.
TRAIN_EXTRA_MEMBERS = ("realesrgan_x4", "flux2_klein", "qwen_image", "movq_k22")


def fixture_p3_train(work: Path, ref: str) -> Fixture:
    """The P3 non-LOLO runbook in three sessions, with a real (tiny) LOLO output to rescore.

    Built on the LOLO fixture's inputs, plus one reconstruction member for each of the
    four lineages it lacks, so every LOLO fold and ``BANK-1`` has held-out validation
    rows. The builder stages the corpus and trains the eight LOLO folds itself (2 steps
    each, the committed trainer, random-init backbone) into an attached dataset
    ``dbp-p3-lolo``, and writes a fixture ``N*`` selection (every ``N*(-f)`` = 2).

    Session 1 trains ``BANK-1`` + ``S2-B`` seed 0 on the P3 recipe in miniature (schedule
    6, stop at 5) and both are killed after their step-4 checkpoints; the LOLO rescoring
    still runs. Session 2 resumes both to step 5 and skips the rescoring (carried).
    Session 3 sees only session 2's output: it carries everything forward and trains the
    next two queued runs.
    """
    base = fixture_p3_lolo(work, ref)
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    # Since 0e1945d the code holds the real LOLO rescoring, which tells the chooser it is
    # done; the fixture is the state before session 1, so the committed scores go.
    committed_scores = code / "reports" / "p3_heldout_scores"
    if committed_scores.is_dir():
        shutil.rmtree(committed_scores)
    s2in = inputs / "datasets/adeelshah29/dbp-p2-s2-inputs"
    pairs = (inputs / "datasets/muhammadadeelshah/thesis-dbp-1-dataset"
             "/Implementation/thesis_dbp/outputs/p1/pairs")
    with (s2in / "p1_pairs_manifest.csv").open(encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    with (s2in / "p2_splits.csv").open(encoding="utf-8", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    side = {r["row_id"]: r for r in split_rows}
    extra_pairs, extra_splits = [], []
    for row in [r for r in pair_rows if r["member"] == "sd15"]:
        for member in TRAIN_EXTRA_MEMBERS:
            pair_id = f"{row['image_id']}__{member}"
            (pairs / member).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pairs / "sd15" / f"{row['image_id']}.png",
                            pairs / member / f"{row['image_id']}.png")
            extra_pairs.append({**row, "pair_id": pair_id, "member": member,
                                "relative_path": f"pairs/{member}/{row['image_id']}.png"})
            extra_splits.append({**side[row["pair_id"]], "row_id": pair_id})
    write_csv(s2in / "p1_pairs_manifest.csv", list(pair_rows[0]), pair_rows + extra_pairs)
    write_csv(s2in / "p2_splits.csv", list(split_rows[0]), split_rows + extra_splits)
    lolo_expect = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    corpus = {**lolo_expect["corpus"],
              "pairs": lolo_expect["corpus"]["pairs"] + len(extra_pairs)}
    specs = {name: {**spec, "sha256": sha256_file(s2in / name),
                    **({"rows": spec["rows"] + len(extra_pairs)}
                       if name in ("p1_pairs_manifest.csv", "p2_splits.csv") else {})}
             for name, spec in lolo_expect["inputs"].items()}

    # The LOLO output to rescore: stage the corpus as the runbook does, then train the
    # eight folds for 2 steps each with the committed trainer.
    build = work / "lolo_build"
    lolo = inputs / "datasets/muhammadadeelshah/dbp-p3-lolo/Implementation/thesis_dbp"
    env = dict(os.environ, DBP_DRYRUN="1", PYTHONUTF8="1",
               HF_HOME=str(build / "hf_home"), TORCH_HOME=str(build / "torch_home"))

    def run(*command: str | Path) -> None:
        done = subprocess.run([sys.executable, *[str(c) for c in command]], cwd=code, env=env,
                              capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"fixture build step failed: {command[:2]}\n"
                               f"{(done.stdout + done.stderr)[-3000:]}")

    links = []
    for name, pattern in (("reals", "datasets/*/*/p1_reals"),
                          ("generated", "datasets/*/*/p1_generated"),
                          ("pairs", "datasets/*/thesis-dbp-1-dataset/Implementation/"
                                    "thesis_dbp/outputs/p1/pairs")):
        links += ["--link", f"{name}={next(inputs.glob(pattern))}",
                  "--expect", f"{name}={corpus[name]}"]
    run("scripts/p2_stage_corpus.py", "--root", build / "p1", *links)
    run("scripts/p1_match_generations.py", "--gen-manifest", s2in / "p1_gen_manifest.csv",
        "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
        "--gen-root", build / "p1" / "generated", "--out-root", build / "p1",
        "--set-id", "gen_matched")
    folds = {}
    for fold in range(1, 9):
        run("scripts/p2_train_s2.py", "--runs-config", "configs/p3_runs.yaml",
            "--protocol", f"lolo_f{fold}", "--seed", "0", "--root", build / "p1",
            "--pairs-manifest", s2in / "p1_pairs_manifest.csv", "--splits",
            s2in / "p2_splits.csv", "--device", "cpu", "--fixed-steps", "2",
            "--heldout-every", "2", "--checkpoint-every", "2", "--val-pairs", "4",
            "--out-dir", lolo / "outputs/p3/s2", "--reports-dir", lolo / "reports")
        report = json.loads((lolo / "reports" / f"p2_s2_lolo_f{fold}_seed0.json")
                            .read_text(encoding="utf-8"))
        folds[fold] = report["curve"][-1]["heldout_auc"]
    nstar = work / "p3_nstar_fixture.json"
    nstar.write_text(json.dumps({
        "what": "dry-run fixture selection: every N*(-f) = 2",
        "nested": {str(f): {"step": 2} for f in folds},
        "h1a_table": [{"fold": f, "heldout_auc": a} for f, a in folds.items()],
        "G": sum(folds.values()) / len(folds),
        "h1a": {"folds_below_floor": [f for f, a in folds.items() if a < 0.75]},
    }, indent=2) + "\n", encoding="utf-8")

    expect = json.loads((code / "env" / "p3_train_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_train (dry-run fixture)",
        "imports": lolo_expect["imports"], "gpus": 0, "hf_token": False,
        "corpus": corpus, "gen_matched": lolo_expect["gen_matched"], "inputs": specs,
        "nstar": str(nstar), "nstar_sha256": sha256_file(nstar),
        "queue": ["bank_1:0", "s2b:0", "bank_2_near:0", "bank_2_far:0"],
        "devices": "cpu",
        "train_args": ["--runs-config=configs/p3_runs.yaml", f"--fixed-steps={TRAIN_STEPS}",
                       f"--stop-at={TRAIN_STOP}", "--heldout-every=1", "--checkpoint-every=2",
                       "--val-pairs=4"],
        "smoke_deadline_hours": 0.5, "deadline_hours": 1.0, "poll_seconds": 2,
        "expected": {"per_session_wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_train_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    carry_to = inputs / "notebooks/muhammadadeelshah/dbp-p3-train"

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError, StopIteration) as exc:
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def lr_reference(step: int) -> float:
        """LEARNING_RATE x the TRAIN_STEPS schedule after ``step`` scheduler steps."""
        warm = min(1.0, (step + 1) / 300)
        return 1e-4 * warm * 0.5 * (1.0 + math.cos(math.pi * min(step / TRAIN_STEPS, 1.0)))

    def check_results(work_dir: Path) -> list[str]:
        sys.path.insert(0, str(code / "src"))
        from dbp.eval.heldout_scores import read_scores, snapshot_digest

        problems: list[str] = []
        session2 = carry_to / "Implementation/thesis_dbp"
        dbp = work_dir / "working/Implementation/thesis_dbp"
        plan = json.loads((dbp / "reports" / "p3_train_plan.json").read_text(encoding="utf-8"))
        if plan["complete"] != ["bank_1:0", "s2b:0"] or \
                plan["selected"] != ["bank_2_near:0", "bank_2_far:0"] or \
                plan["lolo_rescore_due"]:
            problems.append(f"session 3 plan {plan}")
        for rid in ("bank_1_seed0", "s2b_seed0", "bank_2_near_seed0", "bank_2_far_seed0"):
            report = json.loads((dbp / "reports" / f"p2_s2_{rid}.json").read_text("utf-8"))
            steps = [p["step"] for p in report["curve"]]
            resumed = TRAIN_KILL_AFTER + 1 if rid in ("bank_1_seed0", "s2b_seed0") else None
            if report["status"] != "OK" or report["steps_run"] != TRAIN_STOP or \
                    report["schedule_steps"] != TRAIN_STEPS or report["stop_at"] != TRAIN_STOP:
                problems.append(f"{rid}: {report['status']} at {report['steps_run']}, "
                                f"schedule {report.get('schedule_steps')}")
            if steps != list(range(1, TRAIN_STOP + 1)):
                problems.append(f"{rid}: curve steps {steps}")
            if report.get("resumed_from_step") != resumed:
                problems.append(f"{rid}: resumed from {report.get('resumed_from_step')}")
            # D1: the LR every step trained with is the 6-step schedule's, not a 5-step one.
            for point in report["curve"]:
                if not math.isclose(point["lr"], lr_reference(point["step"] - 1),
                                    rel_tol=1e-9, abs_tol=1e-15):
                    problems.append(f"{rid}: step {point['step']} lr {point['lr']}")
            held = [p["step"] for p in report["curve"] if "heldout_auc" in p]
            snaps = sorted(p.name for p in (dbp / "outputs/p3/s2" / rid / "snapshots").iterdir())
            if snaps != [f"step_{TRAIN_STOP:05d}"]:
                problems.append(f"{rid}: snapshots {snaps}")
            if rid == "s2b_seed0":
                if held or report["heldout_scores"] is not None or report["held_out_members"]:
                    problems.append(f"s2b: held-out {held} {report['heldout_scores']}")
            else:
                if held != [TRAIN_STOP]:
                    problems.append(f"{rid}: held-out evaluated at {held}, expected the end only")
                rows, meta = read_scores(dbp / "reports" / report["heldout_scores"])
                snapshot = dbp / "outputs/p3/s2" / rid / "snapshots" / f"step_{TRAIN_STOP:05d}"
                if meta["snapshot_sha256"] != snapshot_digest(snapshot) or \
                        meta["step"] != TRAIN_STOP or meta["heldout_auc"] != \
                        report["curve"][-1]["heldout_auc"]:
                    problems.append(f"{rid}: score sidecar {meta}")
                print(f"  check {rid}: {len(rows)} per-image rows, members "
                      f"{sorted(meta['heldout_auc_by_member'])}")
            print(f"  check {rid}: {report['status']}, steps {steps}, resumed "
                  f"{report.get('resumed_from_step')}, held-out at {held}, lr "
                  f"{[p['lr'] for p in report['curve']]}")
        bank_1 = json.loads((dbp / "reports" / "p2_s2_bank_1_seed0.json").read_text("utf-8"))
        if bank_1["active_members"] != ["sd15"] or "sdxl" not in bank_1["held_out_members"]:
            problems.append(f"bank_1 trained on {bank_1['active_members']}")
        # The LOLO rescoring: written in session 1, carried to sessions 2 and 3.
        summary = json.loads((dbp / "reports/p3_heldout_scores/p3_lolo_rescore_summary.json")
                             .read_text(encoding="utf-8"))
        for fold in summary["folds"]:
            if fold["diff"] is None or abs(fold["diff"]) > 1e-6:
                problems.append(f"rescored {fold['protocol']}: diff {fold['diff']}")
        print(f"  check rescoring: {[(f['protocol'], f['diff']) for f in summary['folds']]}")
        if not (session2 / "reports/p3_heldout_scores/lolo_f8_seed0_step_00002.csv").is_file():
            problems.append("session 2 did not carry the LOLO rescoring")
        # The scorer reproduces the trainer image by image: rescore BANK-1's final
        # snapshot and compare every per-image score with the file written in training.
        rescored = work_dir / "bank_1_rescored.csv"
        tmp = Path(os.environ.get("DBP_DRYRUN_TMP", work_dir))
        done = subprocess.run(
            [sys.executable, str(dbp / "scripts" / "p3_score_heldout.py"),
             "--run-dir", str(dbp / "outputs/p3/s2/bank_1_seed0"), "--protocol", "bank_1",
             "--step", str(TRAIN_STOP), "--out", str(rescored),
             "--root", str(work_dir / "working/p1"),
             "--pairs-manifest", str(s2in / "p1_pairs_manifest.csv"),
             "--splits", str(s2in / "p2_splits.csv"), "--device", "cpu"],
            capture_output=True, text=True, cwd=dbp, check=False,
            env=dict(os.environ, DBP_DRYRUN="1", HF_HOME=str(tmp / "hf_home"),
                     TORCH_HOME=str(tmp / "torch_home")))
        if done.returncode != 0:
            problems.append(f"rescoring BANK-1: {(done.stdout + done.stderr)[-800:]}")
        else:
            trained = {(r["image_id"], r["member"]): r["score"] for r in read_scores(
                dbp / "reports/p3_heldout_scores" / f"bank_1_seed0_step_{TRAIN_STOP:05d}.csv")[0]}
            again = {(r["image_id"], r["member"]): r["score"] for r in read_scores(rescored)[0]}
            worst = max(abs(trained[k] - again[k]) for k in trained) if \
                set(trained) == set(again) else float("inf")
            if worst > 1e-5:
                problems.append(f"BANK-1 rescored per image: max |diff| {worst}")
            print(f"  check per-image reproduction (BANK-1, {len(trained)} rows): "
                  f"max |diff| {worst:.2e}")
        ci = subprocess.run(
            [sys.executable, str(code / "scripts" / "p3_select_nstar.py"), "ci",
             "--nstar", str(nstar), "--scores-dir", str(dbp / "reports/p3_heldout_scores"),
             "--out", str(work_dir / "p3_h1a_ci_fixture.json")],
            capture_output=True, text=True, check=False)
        if ci.returncode != 0:
            problems.append(f"H1-a CI on the rescored files: {(ci.stdout + ci.stderr)[-800:]}")
        else:
            result = json.loads((work_dir / "p3_h1a_ci_fixture.json").read_text("utf-8"))
            print(f"  check H1-a CI (fixture): G {result['G']:.4f}, CI "
                  f"[{result['ci']['lower']:.4f}, {result['ci']['upper']:.4f}], "
                  f"{result['verdict']}")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, base.commit,
        [Session("session 1: BANK-1 + S2-B s0 picked, killed after their step-4 checkpoints; "
                 "the LOLO rescoring runs", {"DBP_FAULT_EXIT_AFTER_STEP": str(TRAIN_KILL_AFTER)},
                 "nonzero"),
         Session("session 2: session 1's output attached; both resume to step 5; rescoring "
                 "carried"),
         Session("session 3: only session 2's output attached; everything carried, the next "
                 "two runs trained")],
        carry_to=carry_to, check=check,
    )


RECON_REALS, RECON_SAMPLE = 6, 2
#: Members whose pinned weights are already cached locally (P0/P2): the dry-run decodes
#: on CPU with real weights and downloads nothing new.
RECON_JOBS = ("cpu:sd15", "cpu:realesrgan_x4")


def fixture_p3_reconstruct(work: Path, ref: str) -> Fixture:
    """Step 4a in miniature: six synthetic 384 px reals through two real bank members.

    The reals are synthetic (smooth gradients plus noise, seeded), laid out as the
    Open Images dataset is (under another name), with a six-row manifest in the committed
    manifest's columns. The members are ``sd15`` (a VAE) and ``realesrgan_x4`` (the GAN
    path, its ``.pth`` from the cache dir), both decoded on CPU at fp32 with their pinned
    weights, copied into the dry-run cache from the local caches when absent.
    """
    import numpy as np
    from PIL import Image

    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    commit = copy_code(code, ref)
    # A slug that is not "dbp-p3-open-images" (the user's upload is "dbp-p3-lolo_data"):
    # the runbook must find the reals by layout.
    reals = inputs / "datasets/muhammadadeelshah/dbp-p3-lolo-data/oi/reals/t_oi"
    reals.mkdir(parents=True)
    committed = code / "reports/kaggle_20261017/p3_open_images/p3_open_images_manifest.csv"
    with committed.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, rows = list(reader.fieldnames or []), list(reader)[:RECON_REALS]
    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:384, 0:384] / 383.0
    for index, row in enumerate(rows):
        image = np.stack([xx, yy, (xx + yy) / 2], axis=-1) * 200 + rng.normal(0, 12, (384, 384, 3))
        path = reals / f"{row['image_id']}.png"
        Image.fromarray(np.clip(image + index * 5, 0, 255).astype(np.uint8)).save(path)
        row["sha256"] = sha256_file(path)
    manifest = work / "p3_oi_manifest_fixture.csv"
    write_csv(manifest, columns, rows)
    throughput = work / "p1_throughput_fixture.csv"
    with (code / "reports/kaggle_20260926/p1_throughput.csv").open(
            encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        bench = [{**r, "fp16_build_allowed": "False"} for r in reader]   # CPU: fp32
        write_csv(throughput, list(bench[0]), bench)

    cache = THESIS_DBP / ".cache" / "dryrun" / "tmp"           # the harness's default
    sd15 = "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    if not (cache / "hf_home" / "hub" / sd15).exists() and \
            (THESIS_DBP / ".cache" / "hf" / "hub" / sd15).exists():
        shutil.copytree(THESIS_DBP / ".cache" / "hf" / "hub" / sd15,
                        cache / "hf_home" / "hub" / sd15)
    pth = cache / "dbp_weights" / "RealESRGAN_x4plus.pth"
    if not pth.exists() and (THESIS_DBP / ".cache" / "weights" / pth.name).exists():
        pth.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(THESIS_DBP / ".cache" / "weights" / pth.name, pth)

    expect = json.loads((code / "env" / "p3_reconstruct_expect.json").read_text("utf-8"))
    expect.update({
        "stage": "p3_reconstruct (dry-run fixture)", "gpus": 0, "hf_token": False,
        "reals_manifest": str(manifest), "reals_manifest_sha256": sha256_file(manifest),
        "throughput_csv": str(throughput), "throughput_csv_sha256": sha256_file(throughput),
        "n_reals": RECON_REALS, "jobs": list(RECON_JOBS), "sample_reals": RECON_SAMPLE,
        "margin_gb": 0.5, "sample_deadline_hours": 0.5, "deadline_hours": 1.0,
        "poll_seconds": 2, "expected": {"wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_reconstruct_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError) as exc:
            return [f"the session did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        final = json.loads((dbp / "reports/p3_oi_pairs_final.json").read_text("utf-8"))
        gate = json.loads((dbp / "reports/p3_oi_size_gate.json").read_text("utf-8"))
        if not final["ok"] or final["rows"] != RECON_REALS * len(RECON_JOBS):
            problems.append(f"final check {final}")
        if set(final["precision_rule"][m] for m in ("sd15", "realesrgan_x4")) != {"fp32"}:
            problems.append(f"precision rule {final['precision_rule']}")
        if not gate["ok"] or set(gate["members"]) != {"sd15", "realesrgan_x4"} or any(
                v["sample_images"] != RECON_SAMPLE for v in gate["members"].values()):
            problems.append(f"size gate {gate}")
        merged = dbp / "reports/p3_oi_pairs_manifest.csv"
        with merged.open(encoding="utf-8", newline="") as handle:
            pair_rows = list(csv.DictReader(handle))
        if {r["precision"] for r in pair_rows} != {"fp32"} or \
                {r["relative_path"].split("/")[0] for r in pair_rows} != {"oi_pairs"}:
            problems.append("merged manifest precision or paths")
        if (dbp / "reports/p1_pairs_manifest.csv").exists() != \
                (code / "reports/p1_pairs_manifest.csv").exists():
            problems.append("step 4a wrote a P1-named manifest")
        sample = sorted(p.name for p in (dbp / "reports").glob("p3_oi_pairs_sample_manifest_*"))
        print(f"  check: {final['rows']} pairs, {final['gb']} GB measured; gate projected "
              f"{gate['projected_gb']} GB for {gate['n_reals']} reals "
              f"({ {m: v['mean_bytes'] for m, v in gate['members'].items()} } B/PNG); "
              f"sample manifests kept apart: {sample}")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1: sample + size gate, then both members, merged and "
                            "verified")], check=check)


TUNE_STEPS, TUNE_STOP, TUNE_KILL_AFTER = 6, 5, 4


def fixture_p3_tune(work: Path, ref: str) -> Fixture:
    """Degradation tuning in miniature, on Session D's fixture inputs, in two sessions.

    The dev panel is rebuilt with the views the rule reads: E3 (real E3 ids from the
    committed panel manifest, two GPT-Image2 and two Nano-Banana2 fakes, two reals), E4 and
    E5 (three fakes and three reals each). The builder stages the corpus and trains a tiny
    S2-B seed 0 with the committed trainer into an attached ``dbp-p3-train`` output.
    Session 1 scores it, then both candidates are killed after their step-4 checkpoints;
    session 2 resumes them, scores them and applies the rule.
    """
    base = fixture_p2_s2(work, ref)
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/adeelshah29/dbp-p2-s2-inputs"
    panel = inputs / "notebooks/adeelshah29/phase-2/panel"
    with (code / "reports/p2_panel_sets/p2_panel_e3_matched.csv").open(
            encoding="utf-8", newline="") as handle:
        e3 = list(csv.DictReader(handle))
    picks = ([r for r in e3 if r["source_label"] == "GPT-Image2-fake"][:2]
             + [r for r in e3 if r["source_label"] == "Nano-Banana2-fake"][:2]
             + [r for r in e3 if r["label"] == "real"][:2])
    rows = []
    for seed, row in enumerate(picks, start=40):
        write_png(panel / "e3_matched" / f"{row['image_id']}.png", seed)
        rows.append({"image_id": row["image_id"], "relative_path":
                     f"e3_matched/{row['image_id']}.png", "label": row["label"],
                     "view": "e3_matched", "root": "panel"})
    for offset, view in ((50, "e4_wildrf_matched"), (60, "e5_chameleon_pipeline")):
        for k in range(6):
            image_id = f"fx_{view[:2]}_{k}"
            write_png(panel / view / f"{image_id}.png", offset + k)
            rows.append({"image_id": image_id, "relative_path": f"{view}/{image_id}.png",
                         "label": "fake" if k < 3 else "real", "view": view, "root": "panel"})
    # Rows the rule never reads, with no image on disk: tuning must neither resolve nor
    # score them (user, 2026-10-21: only the rule's rows are scored).
    unread = [{"image_id": f"fx_unread_{k}", "relative_path": f"{view}/fx_unread_{k}.png",
               "label": label, "view": view, "root": "panel"}
              for k, (view, label) in enumerate((("e7_dda_coco", "fake"), ("e7_dda_coco", "real"),
                                                 ("e8_evalgen", "fake"),
                                                 ("t_ctrl_diagnostic", "real")))]
    write_csv(s2in / "p2_panel_images.csv",
              ["image_id", "relative_path", "label", "view", "root"], rows + unread)

    # A tiny trained S2-B seed 0, in a dbp-p3-train notebook output.
    old = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    build = work / "tune_build"
    train_out = inputs / "notebooks/muhammadadeelshah/dbp-p3-train/Implementation/thesis_dbp"
    env = dict(os.environ, DBP_DRYRUN="1", PYTHONUTF8="1",
               HF_HOME=str(build / "hf_home"), TORCH_HOME=str(build / "torch_home"))

    def run(*command: str | Path) -> None:
        done = subprocess.run([sys.executable, *[str(c) for c in command]], cwd=code, env=env,
                              capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"fixture build step failed: {command[:2]}\n"
                               f"{(done.stdout + done.stderr)[-3000:]}")

    links = []
    for name, pattern in (("reals", "datasets/*/*/p1_reals"),
                          ("generated", "datasets/*/*/p1_generated"),
                          ("pairs", "datasets/*/thesis-dbp-1-dataset/Implementation/"
                                    "thesis_dbp/outputs/p1/pairs")):
        links += ["--link", f"{name}={next(inputs.glob(pattern))}",
                  "--expect", f"{name}={old['corpus'][name]}"]
    run("scripts/p2_stage_corpus.py", "--root", build / "p1", *links)
    run("scripts/p1_match_generations.py", "--gen-manifest", s2in / "p1_gen_manifest.csv",
        "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
        "--gen-root", build / "p1" / "generated", "--out-root", build / "p1",
        "--set-id", "gen_matched")
    run("scripts/p2_train_s2.py", "--runs-config", "configs/p3_runs.yaml", "--protocol", "s2b",
        "--seed", "0", "--root", build / "p1", "--pairs-manifest", s2in / "p1_pairs_manifest.csv",
        "--splits", s2in / "p2_splits.csv", "--device", "cpu", "--fixed-steps", str(TUNE_STEPS),
        "--stop-at", str(TUNE_STOP), "--heldout-every", "1", "--checkpoint-every", "2",
        "--val-pairs", "4", "--out-dir", train_out / "outputs/p3/s2",
        "--reports-dir", train_out / "reports")

    names = ("p1_pairs_manifest.csv", "p2_splits.csv", "p2_splits_summary.json",
             "p1_gen_manifest.csv", "p2_panel_images.csv", "p2_s1_images.csv")
    counted = {"p2_panel_images.csv": len(rows) + len(unread)}
    lists_ref = work / "tune_lists_reference"
    run("scripts/p3_tune_lists.py", "--panel-list", s2in / "p2_panel_images.csv",
        "--val-list", s2in / "p2_s1_images.csv", "--out-dir", lists_ref)
    derived = {p.name: {"rows": sum(1 for _ in p.open(encoding="utf-8")) - 1,
                        "sha256": sha256_file(p)} for p in sorted(lists_ref.glob("*.csv"))}
    with (s2in / "p2_s1_images.csv").open(encoding="utf-8", newline="") as handle:
        val_reals = sum(1 for r in csv.DictReader(handle)
                        if r["view"] == "val" and r["label"] == "real")
    expect = json.loads((code / "env" / "p3_tune_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_tune (dry-run fixture)", "imports": old["imports"], "gpus": 0,
        "hf_token": False, "corpus": old["corpus"], "gen_matched": old["gen_matched"],
        "inputs": {n: {**{k: v for k, v in old["inputs"][n].items() if k == "rows"},
                       "sha256": sha256_file(s2in / n),
                       **({"rows": counted[n]} if n in counted else {})} for n in names},
        "devices": "cpu", "base_stop_at": TUNE_STOP,
        "train_args": ["--runs-config=configs/p3_runs.yaml", f"--fixed-steps={TUNE_STEPS}",
                       f"--stop-at={TUNE_STOP}", "--heldout-every=1", "--checkpoint-every=2",
                       "--val-pairs=4"],
        "infer_args": ["--batch=4", "--runs-config=configs/p3_runs.yaml"],
        "smoke_deadline_hours": 0.5, "deadline_hours": 1.0, "poll_seconds": 2,
        "derived_lists": derived,
        "expected": {"wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_tune_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    carry_to = inputs / "notebooks/muhammadadeelshah/dbp-p3-tune"

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError, TypeError) as exc:
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        choice = json.loads((dbp / "reports/p3_degradation_choice.json").read_text("utf-8"))
        models = set(choice["models"])
        if models != {"s2b_seed0", "s2b_deg_p70_seed0", "s2b_deg_p35_seed0"} or \
                choice["chosen"] not in ("s2b_deg_p70_seed0", "s2b_deg_p35_seed0"):
            problems.append(f"choice {choice.get('chosen')} over {sorted(models)}")
        for rid, model in choice["models"].items():
            if set(model["auc"]) != set(choice["unseen_slices"]) or \
                    set(model["fpr"]) != {"e4_wildrf_matched", "e5_chameleon_pipeline"}:
                problems.append(f"{rid}: slices {sorted(model['auc'])}, "
                                f"pools {sorted(model['fpr'])}")
        summary = json.loads((dbp / "reports/p2_s2_scores_s2b_seed0.json").read_text("utf-8"))
        if summary.get("limit") is not None or set(summary["lists"]) != {"panel", "val"}:
            problems.append(f"S2-B seed 0 scoring summary {summary.get('limit')} "
                            f"{sorted(summary['lists'])}")
        # Only the rule's rows were scored, for every model; the unread rows never.
        scored = {"panel": len(rows), "val": val_reals}
        for rid in ("s2b_seed0", "s2b_deg_p70_seed0", "s2b_deg_p35_seed0"):
            for name, want in scored.items():
                path = dbp / "outputs/p3/s2" / rid / "scores" / f"{name}.csv"
                with path.open(encoding="utf-8", newline="") as handle:
                    ids = [r["image_id"] for r in csv.DictReader(handle)]
                if len(ids) != want or any(i.startswith("fx_unread_") for i in ids):
                    problems.append(f"{rid} {name}: {len(ids)} rows scored, expected {want}")
        print(f"  check scored rows per model: {scored} (of {len(rows) + len(unread)} panel rows)")
        for rid, p in (("s2b_deg_p70_seed0", 0.7), ("s2b_deg_p35_seed0", 0.35)):
            report = json.loads((dbp / f"reports/p2_s2_{rid}.json").read_text("utf-8"))
            degrade = report["degrade"]
            if report["status"] != "OK" or report["steps_run"] != TUNE_STOP or \
                    report.get("resumed_from_step") != TUNE_KILL_AFTER + 1:
                problems.append(f"{rid}: {report['status']} at {report['steps_run']}, "
                                f"resumed {report.get('resumed_from_step')}")
            if degrade["p"] != p or degrade["pairs"] <= 0 or degrade["ms_per_pair"] is None:
                problems.append(f"{rid}: degrade {degrade}")
            print(f"  check {rid}: {report['status']}, resumed {report['resumed_from_step']}, "
                  f"degrade {degrade}")
        base = json.loads((dbp / "reports/p2_s2_s2b_seed0.json").read_text("utf-8"))
        if base["degrade"]["pairs"] != 0:
            problems.append(f"S2-B seed 0 was degraded: {base['degrade']}")
        tmp = Path(os.environ.get("DBP_DRYRUN_TMP", work_dir))
        smoke = sorted((tmp / "dbp_p3_tune_smoke").rglob("p2_s2_s2b_deg_*_seed0.json"))
        costs = [json.loads(p.read_text("utf-8"))["degrade"]["ms_per_pair"] for p in smoke]
        if not smoke or any(c is None for c in costs):
            problems.append(f"the smoke did not measure the degradation cost: {costs}")
        print(f"  check smoke degradation cost (ms per pair): {costs}")
        print(f"  check choice: {choice['chosen']} (p = {choice['chosen_p']}): {choice['reason']}")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, base.commit,
        [Session("session 1: S2-B s0 scored; both candidates killed after their step-4 "
                 "checkpoints", {"DBP_FAULT_EXIT_AFTER_STEP": str(TUNE_KILL_AFTER)}, "nonzero"),
         Session("session 2: the previous output attached; candidates resume, are scored, "
                 "and the rule is applied")],
        carry_to=carry_to, check=check,
    )


def fixture_p3_dev_panel(work: Path, ref: str) -> Fixture:
    """The dev-panel rebuild in miniature, on the Panel B fixture, with no network.

    As on Kaggle, a derived P2 row is found at its recorded ``view_path`` only because the
    session's view rebuild writes there again: the fixture deletes its own learning-pass
    copies and points those rows at the session's code tree, so every derived file the
    rebuild copies was produced in the session and re-hashed. Two T-ctrl reals and their
    reconstructions come from a tiny staged corpus; one row is left out of the scoreable
    list (the crop filter), so the two digests differ as on Kaggle.
    """
    base = fixture_p3_panel_b(work, ref)
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/muhammadadeelshah/dbp-p2-s2-inputs"
    old = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    p2_dir = Path(old["p2_panel_sets"])
    session_dbp = work / "working" / "Implementation" / "thesis_dbp"
    learned = code / "outputs" / "p1" / "eval"
    if learned.exists():
        shutil.rmtree(learned)
    rows_by_view: dict[str, list[dict[str, str]]] = {}
    for path in sorted(p2_dir.glob("p2_panel_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, rows = list(reader.fieldnames or []), list(reader)
        for row in rows:
            recorded = row.get("view_path") or ""
            # E3/E5 point into the code tree's derived views; E4 at its local source.
            if recorded and Path(recorded).is_relative_to(code):
                row["view_path"] = str(session_dbp / Path(recorded).relative_to(code))
        if rows:
            write_csv(path, columns, rows)
        rows_by_view[path.stem.removeprefix("p2_panel_")] = rows

    reals = inputs / "datasets/muhammadadeelshah/report/p1_reals"
    pairs = (inputs / "datasets/muhammadadeelshah/thesis-dbp-1-dataset"
             "/Implementation/thesis_dbp/outputs/p1/pairs")
    tctrl = []
    for n in range(2):
        write_png(reals / "t_ctrl" / f"tc{n}_c0.png", 70 + n)
        write_png(pairs / "sd15" / f"tc{n}_c0.png", 80 + n)
        for label, rel in (("real", f"reals/t_ctrl/tc{n}_c0.png"),
                           ("fake", f"pairs/sd15/tc{n}_c0.png")):
            source = reals.parent / "p1_reals" / rel.removeprefix("reals/") if label == "real" \
                else pairs / rel.removeprefix("pairs/")
            tctrl.append({"image_id": f"{'pairs_' if label == 'fake' else ''}tc{n}_c0",
                          "relative_path": rel, "label": label, "view": "t_ctrl_diagnostic",
                          "sha256": sha256_file(source)})
    write_csv(p2_dir / "p2_panel_t_ctrl_diagnostic.csv", list(tctrl[0]), tctrl)
    rows_by_view["t_ctrl_diagnostic"] = tctrl

    views = load_script(code / "scripts" / "p2_materialise_panel.py", "fixture_materialiser")
    all_rows, listed = [], []
    for view, rows in sorted(rows_by_view.items()):
        for row in rows:
            source = row.get("view_path") or row["relative_path"]
            suffix = Path(source).suffix.lower()
            all_rows.append(row)
            listed.append({"image_id": row["image_id"],
                           "relative_path": f"{view}/{views.safe_name(row['image_id'], suffix)}",
                           "label": row["label"], "view": view, "root": "panel"})
    listed = listed[1:]                                   # one row "dropped by the crop rule"
    scoreable = [r for r in all_rows if r["image_id"] in {x["image_id"] for x in listed}]
    write_csv(s2in / "p2_panel_images.csv", ["image_id", "relative_path", "label", "view", "root"],
              listed)
    expect = json.loads((code / "env" / "p3_dev_panel_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_dev_panel (dry-run fixture)",
        "inputs": {"p2_panel_images.csv": {"sha256": sha256_file(s2in / "p2_panel_images.csv"),
                                           "rows": len(listed)},
                   "p1_pairs_manifest.csv": {"sha256": sha256_file(s2in / "p1_pairs_manifest.csv"),
                                             "rows": 1}},
        "local": old["local"], "p2_panel_sets": str(p2_dir), "label_check": False,
        "coco_val_count": old["coco_val_count"],
        "corpus": {"reals": 2, "pairs": 2},
        "digest_all": views.panel_digest(all_rows),
        "digest_scoreable": views.panel_digest(scoreable),
        "expected": {"wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_dev_panel_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        try:
            record = json.loads((dbp / "reports/p3_dev_panel_digest.json").read_text("utf-8"))
        except (OSError, ValueError) as exc:
            return [f"no dev panel record: {exc}"]
        if not record["ok"] or record["rows_all"] != len(all_rows) or \
                record["rows_scoreable"] != len(listed):
            problems.append(f"record {record.get('ok')} {record.get('problems')}")
        panel = work_dir / "working/dev_panel"
        missing = [r["relative_path"] for r in listed if not (panel / r["relative_path"]).is_file()]
        if missing:
            problems.append(f"{len(missing)} listed file(s) not under the panel root")
        if (dbp / "outputs/p1/eval").exists():
            problems.append("the derived view images were not cleaned up")
        print(f"  check: {record['rows_all']} rows (digest {record['digest_all'][:12]}), "
              f"{record['rows_scoreable']} scoreable (digest {record['digest_scoreable'][:12]}), "
              f"{record['panel_bytes']} bytes, by view {record['by_view']}")
        return problems

    return Fixture(base.env | {"DBP_EXPECT": str(expect_path)}, base.commit,
                   [Session("session 1: views rebuilt, every row copied and re-hashed, both "
                            "digests required")], check=check)


def truefake_fixture_zip(path: Path) -> list[dict[str, str]]:
    """A small archive laid out like TrueFake.zip, and its sample rows (2 ids x 4 shares)."""
    import zipfile

    rows = []
    with zipfile.ZipFile(path, "w") as archive:
        for k, (stratum, stem) in enumerate((("Fake/StyleGAN/images-psi-0.5", "00007"),
                                             ("Real/FORLAB", "00042"))):
            for share in ("PreSocial", "Facebook", "Telegram", "Twitter"):
                suffix = "png" if share == "PreSocial" and stratum.startswith("Fake") else "jpg"
                ident = f"{stratum}/{stem}"
                member = f"TrueFake/{share}/{ident}.{suffix}"
                payload = bytes((k * 37 + i * 11 + len(share)) % 256 for i in range(5000))
                method = zipfile.ZIP_DEFLATED if share != "Twitter" else zipfile.ZIP_STORED
                archive.writestr(member, payload, compress_type=method)
                info = archive.getinfo(member)
                rows.append({"id": ident, "stratum": stratum, "share": share, "member": member,
                             "compressed": str(info.compress_size),
                             "uncompressed": str(info.file_size)})
        archive.writestr("TrueFake/PreSocial/Real/FFHQ/99999.png", b"not in the sample")
    return rows


def fixture_p3_truefake(work: Path, ref: str) -> Fixture:
    """The TrueFake fetch in miniature: a local archive with TrueFake's layout, no network.

    The archive mixes deflated and stored members and holds one member outside the
    sample. The session verifies the archive size and the central-directory sha256, then
    fetches every sample member and checks its CRC-32. The refusal path (stop at once, no
    workaround) is covered by tests/test_p3_truefake.py.
    """
    import struct

    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    commit = copy_code(code, ref)
    archive = work / "TrueFake_fixture.zip"
    rows = truefake_fixture_zip(archive)
    sample = work / "truefake_sample_fixture.csv"
    write_csv(sample, list(rows[0]), rows)
    blob = archive.read_bytes()
    end = blob.rfind(b"PK\x05\x06")
    _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack("<4s4H2LH", blob[end:end + 22])
    expect = json.loads((code / "env" / "p3_truefake_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_truefake (dry-run fixture)", "sample": str(sample),
        "sample_sha256": sha256_file(sample), "sample_rows": len(rows), "sample_gb": 0.001,
        "zip_size": len(blob),
        "cd_sha256": hashlib.sha256(blob[cd_offset:cd_offset + cd_size]).hexdigest(),
        "local_zip": str(archive), "workers": 2,
        "expected": {"wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_truefake_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        import zipfile

        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        try:
            status = json.loads((dbp / "reports/truefake/truefake_fetch_status.json")
                                .read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [f"no fetch status: {exc}"]
        if status["outcome"] != "COMPLETE" or status["members_done"] != len(rows):
            problems.append(f"status {status['outcome']} {status['members_done']}")
        out = work_dir / "working/truefake_sample"
        with zipfile.ZipFile(archive) as source:
            for row in rows:
                path = out / row["member"].removeprefix("TrueFake/")
                if not path.is_file() or path.read_bytes() != source.read(row["member"]):
                    problems.append(f"{row['member']} missing or different")
        if (out / "PreSocial/Real/FFHQ/99999.png").exists():
            problems.append("a member outside the sample was fetched")
        print(f"  check: {status['outcome']}, {status['members_done']} members, "
              f"index {status['index']['entries']} entries, every member byte-identical")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1: archive and index verified, every sample member "
                            "fetched and CRC-checked")], check=check)


S2D_OI = {"o_s2d_0": "train", "o_s2d_1": "train", "o_s2d_2": "train", "o_s2d_3": "held_out"}


def fixture_p3_s2d(work: Path, ref: str) -> Fixture:
    """S2-D and S2-B+reals in two sessions: Session D's inputs plus four Open Images reals.

    Three Open Images reals are ``train`` and one is ``held_out``; each has an ``sd15`` and
    an ``sd21`` reconstruction in a step-4a-shaped notebook output. **The held-out real and
    its reconstructions do not exist on disk** although both manifests list them, so any
    read of them fails the session. The tuning result (``chosen_p`` 0.35) is written into
    the fixture's code tree, as a commit made after tuning will hold it. Session 1 trains
    ``s2d_p35`` s0 + ``s2b_reals`` on the P3 recipe in miniature; session 2 sees session
    1's output, carries both forward and trains ``s2d_p35`` s1.
    """
    base = fixture_p2_s2(work, ref)
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    reals_dir = inputs / "datasets/muhammadadeelshah/dbp-p3-open-images/oi/reals/t_oi"
    recon = inputs / "notebooks/muhammadadeelshah/dbp-p3-reconstruct"
    members = ("sd15", "sd21")
    committed = code / "reports/kaggle_20261017/p3_open_images/p3_open_images_manifest.csv"
    with committed.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns, template = list(reader.fieldnames or []), next(reader)
    oi_rows, pair_rows = [], []
    for seed, (image_id, split) in enumerate(S2D_OI.items(), start=40):
        path = reals_dir / f"{image_id}.png"
        if split == "train":
            write_png(path, seed)
        oi_rows.append({**template, "image_id": image_id, "split": split,
                        "relative_path": f"reals/t_oi/{image_id}.png",
                        "sha256": sha256_file(path) if split == "train" else "0" * 64})
        for k, member in enumerate(members):
            target = recon / "oi_pairs" / member / f"{image_id}.png"
            if split == "train":
                write_png(target, 100 * seed + k)
            pair_rows.append({"pair_id": f"{image_id}__{member}", "image_id": image_id,
                              "set_id": "t_oi", "member": member,
                              "relative_path": f"oi_pairs/{member}/{image_id}.png",
                              "sha256": sha256_file(target) if split == "train" else "0" * 64})
    write_csv(recon / "Implementation/thesis_dbp/reports/p3_oi_pairs_manifest.csv",
              ["pair_id", "image_id", "set_id", "member", "relative_path", "sha256"], pair_rows)
    fixture_dir = code / "reports" / "dryrun_fixture"
    oi_manifest = fixture_dir / "p3_oi_manifest.csv"
    write_csv(oi_manifest, columns, oi_rows)
    builder = load_script(code / "scripts" / "p3_s2d_inputs.py", "_s2d_fixture_builder")
    digest = fixture_dir / "p3_oi_pairs_digest.json"
    digest.write_text(json.dumps({"manifest": {
        "rows": len(pair_rows), "rows_per_member": {m: len(S2D_OI) for m in sorted(members)},
        "content_sha256": builder.content_digest(pair_rows)}}, indent=2) + "\n",
        encoding="utf-8")
    (code / "reports" / "p3_degradation_choice.json").write_text(json.dumps(
        {"chosen": "s2b_deg_p35", "chosen_p": 0.35, "reason": "dry-run fixture"}) + "\n",
        encoding="utf-8")

    old = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    expect = json.loads((code / "env" / "p3_s2d_expect.json").read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_s2d (dry-run fixture)", "imports": old["imports"], "gpus": 0,
        "hf_token": False, "corpus": old["corpus"], "gen_matched": old["gen_matched"],
        "inputs": {name: old["inputs"][name] for name in expect["inputs"]},
        "oi": {**expect["oi"], "reals_manifest": oi_manifest.relative_to(code).as_posix(),
               "reals_manifest_sha256": sha256_file(oi_manifest),
               "pairs_digest": digest.relative_to(code).as_posix(),
               "pairs_digest_sha256": sha256_file(digest),
               "train_reals": 3, "held_out_reals": 1, "hash_workers": 4},
        "devices": "cpu",
        "train_args": ["--runs-config=configs/p3_runs.yaml", f"--fixed-steps={TRAIN_STEPS}",
                       f"--stop-at={TRAIN_STOP}", "--heldout-every=1", "--checkpoint-every=2",
                       "--val-pairs=4"],
        "smoke_deadline_hours": 0.5, "deadline_hours": 1.0, "poll_seconds": 2,
        "expected": {"per_session_wall_hours": 0, "basis": "dry-run fixture"},
    })
    expect_path = work / "p3_s2d_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    carry_to = inputs / "notebooks/muhammadadeelshah/dbp-p3-s2d"

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError, StopIteration) as exc:
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        session1 = carry_to / "Implementation/thesis_dbp"
        plan = json.loads((dbp / "reports" / "p3_train_plan.json").read_text("utf-8"))
        if plan["complete"] != ["s2d_p35:0", "s2b_reals:0"] or \
                plan["selected"] != ["s2d_p35:1"]:
            problems.append(f"session 2 plan {plan}")
        built = json.loads((dbp / "reports" / "p3_s2d_inputs.json").read_text("utf-8"))
        checked = json.loads((dbp / "reports" / "p3_s2d_inputs_check.json").read_text("utf-8"))
        if built["queue"] != ["s2d_p35:0", "s2b_reals:0", "s2d_p35:1"]:
            problems.append(f"queue {built['queue']}")
        if not checked["ok"] or any(checked["held_out_rows_found"].values()):
            problems.append(f"held-out check {checked}")
        if not (dbp / "reports" / "p3_s2d_hardware.txt").read_text("utf-8").startswith("CPU:"):
            problems.append("no hardware log")
        # Staging: every file read once, the 3 + 3 x 2 Open Images train files hash-checked.
        staged = built["staged"]
        if not staged["ok"] or staged["verified"] != 3 + 3 * len(members):
            problems.append(f"staging {staged}")
        print(f"  check staging: {staged['files']} files, {staged['bytes']} bytes, "
              f"{staged['verified']} hash-verified, {staged['files_per_s']} files/s, "
              f"by source {staged['by_source']}")
        for rid, p, tree in (("s2d_p35_seed0", 0.35, session1),
                             ("s2b_reals_seed0", 0.0, session1),
                             ("s2d_p35_seed1", 0.35, dbp)):
            report = json.loads((dbp / "reports" / f"p2_s2_{rid}.json").read_text("utf-8"))
            if report["status"] != "OK" or report["steps_run"] != TRAIN_STOP or \
                    report["schedule_steps"] != TRAIN_STEPS:
                problems.append(f"{rid}: {report['status']} at {report['steps_run']}")
            if (report.get("degrade") or {}).get("p") != p:
                problems.append(f"{rid}: degradation {report.get('degrade')}, expected p {p}")
            if report["heldout_scores"] is not None or report["held_out_members"]:
                problems.append(f"{rid}: held-out {report['held_out_members']}")
            # The trainer's own pool equals the builder's: COCO and Open Images reals pooled.
            pool = built["pools"][rid.rsplit("_seed", 1)[0]]
            log = next((tree / "outputs/p3/logs").glob(f"s2_{rid}_*train.log"))
            lines = log.read_text("utf-8").splitlines()
            if "dbp_s2d_corpus" not in lines[0]:   # the trainer reads the local copy
                problems.append(f"{rid}: trainer root not the local copy: {lines[0][-200:]}")
            line = next(x for x in lines if "protocol data" in x)
            if f"over {pool['reals']} reals" not in line or pool["oi_reals"] != 3:
                problems.append(f"{rid}: trainer '{line.strip()}' vs pool {pool}")
            print(f"  check {rid}: {report['status']}, steps {report['steps_run']}, degrade p "
                  f"{(report.get('degrade') or {}).get('p')}, pool {pool['coco_reals']} COCO + "
                  f"{pool['oi_reals']} Open Images; trainer: {line.split('|')[-1].strip()}")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, base.commit,
        [Session("session 1: S2-D s0 + S2-B+reals picked and trained; the held-out Open "
                 "Images files do not exist"),
         Session("session 2: session 1's output attached; both carried, S2-D s1 trained")],
        carry_to=carry_to, check=check,
    )


FINAL_STEP, FINAL_N_MAX = 2, 3


def fixture_p3_final(work: Path, ref: str) -> Fixture:
    """The final scoring in miniature: a tiny Panel B, 24 models, two sessions.

    One S2-B run is trained (schedule 3, stop at 2) with the committed trainer and its
    snapshot is copied to all 24 model ids in the sources the runbook finds by content:
    the LOLO dataset, the train, tune and S2-D notebook outputs, and the P2 reference
    dataset (``<id>/lora`` + ``head.pt``). Panel B, the threshold reals and the section-5
    records are fixture files with their own digests. Session 1 scores every model clean;
    session 2 sees session 1's output, carries the clean stage forward and scores H5 for
    the three robustness models. Rows the robust stage must not read (calibration, luma)
    are present; the check proves a complete model is never scored again.
    """
    base = fixture_p2_s2(work, ref)
    inputs = work / "input"
    # The final session attaches no development panel; the base fixture's would make two
    # directories holding e7_dda_coco_luma, which the runbook rightly refuses.
    shutil.rmtree(inputs / "notebooks/adeelshah29/phase-2")
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    s2in = inputs / "datasets/adeelshah29/dbp-p2-s2-inputs"
    reals_root = inputs / "datasets/muhammadadeelshah/report/p1_reals"
    fixture_dir = code / "reports" / "dryrun_fixture"
    sys.path.insert(0, str(code / "scripts"))
    sys.path.insert(0, str(code / "src"))
    from p2_materialise_panel import panel_digest

    # --- Panel B and its luma companion --------------------------------------------------
    panel = inputs / "datasets/muhammadadeelshah/dbp-p3-panel-b/panel_b"
    slots = [("e3_matched", "fake", "test", "GPT-Image2-fake"),
             ("e3_matched", "fake", "test", "Nano-Banana2-fake"),
             ("e3_matched", "real", "test", "real-test"),
             ("e3_matched", "real", "calibration", "real-calibration"),
             ("e4_wildrf_matched", "fake", "test", "fake"),
             ("e4_wildrf_matched", "real", "test", "real-test"),
             ("e4_wildrf_matched", "real", "calibration", "real-calibration"),
             ("e5_chameleon_pipeline", "fake", "test", "fake"),
             ("e5_chameleon_pipeline", "real", "test", "real-test"),
             ("e5_chameleon_pipeline", "real", "calibration", "real-calibration"),
             ("e7_dda_coco", "fake", "test", "sdxl-vae"),
             ("e7_dda_coco", "real", "test", "real-test"),
             ("e8_evalgen", "fake", "test", "Flux")]
    luma_slots = [("e7_dda_coco_luma", "fake", "test", "FLUX.1"),
                  ("e7_dda_coco_luma", "real", "test", "real-test"),
                  ("e7_dda_coco_luma", "real", "calibration", "real-calibration")]
    columns = ["image_id", "relative_path", "label", "view", "sha256", "role",
               "panel_b_slot", "e3_number_in_p2"]

    def panel_rows(spec: list[tuple[str, str, str, str]], offset: int) -> list[dict[str, str]]:
        rows = []
        for k, (view, label, role, slot) in enumerate(spec):
            image_id = f"pb_{offset + k:03d}"
            path = panel / view / f"{image_id}.png"
            write_png(path, 300 + offset + k)
            rows.append({"image_id": image_id, "relative_path": f"{view}/{image_id}.png",
                         "label": label, "view": view, "sha256": sha256_file(path),
                         "role": role, "panel_b_slot": slot, "e3_number_in_p2": ""})
        return rows

    rows, luma = panel_rows(slots, 0), panel_rows(luma_slots, 100)
    write_csv(fixture_dir / "p3_panel_b_images.csv", columns, rows)
    write_csv(fixture_dir / "p3_panel_b_luma.csv", columns, luma)

    # --- the threshold reals: the fixture's COCO val reals and one held-out OI real -------
    with (s2in / "p2_s1_images.csv").open(encoding="utf-8", newline="") as handle:
        val = [r for r in csv.DictReader(handle) if r["view"] == "val" and r["label"] == "real"]
    write_csv(fixture_dir / "val_reals.csv", ["image_id", "relative_path", "label", "view",
                                              "root"], val)
    write_csv(fixture_dir / "reals_manifest.csv", ["image_id", "relative_path", "sha256"],
              [{"image_id": r["image_id"], "relative_path": r["relative_path"],
                "sha256": sha256_file(reals_root / Path(r["relative_path"]).relative_to(
                    "reals"))} for r in val])
    oi_dir = inputs / "datasets/muhammadadeelshah/dbp-p3-open-images/oi/reals/t_oi"
    oi_rows = []
    for k, split in enumerate(("train", "train", "train", "held_out")):
        image_id = f"oi_final_{k}"
        write_png(oi_dir / f"{image_id}.png", 400 + k)
        oi_rows.append({"image_id": image_id, "split": split,
                        "sha256": sha256_file(oi_dir / f"{image_id}.png")})
    write_csv(fixture_dir / "oi_manifest.csv", ["image_id", "split", "sha256"], oi_rows)

    # --- section 5: the tuning result, N* / N_max ----------------------------------------
    (code / "reports" / "p3_degradation_choice.json").write_text(json.dumps(
        {"chosen": "s2b_deg_p35_seed0", "chosen_p": 0.35, "reason": "dry-run fixture"}) + "\n",
        encoding="utf-8")
    nstar = fixture_dir / "p3_nstar.json"
    nstar.write_text(json.dumps({
        "n_star": {"step": FINAL_STEP}, "rule": {"n_max": FINAL_N_MAX},
        "nested": {str(f): {"step": FINAL_STEP} for f in range(1, 9)}}) + "\n",
        encoding="utf-8")

    # --- one trained snapshot, copied to every model in its source ------------------------
    build = work / "final_build"
    env = dict(os.environ, DBP_DRYRUN="1", PYTHONUTF8="1",
               HF_HOME=str(build / "hf_home"), TORCH_HOME=str(build / "torch_home"))
    links = []
    old = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    for name, pattern in (("reals", "datasets/*/*/p1_reals"),
                          ("generated", "datasets/*/*/p1_generated"),
                          ("pairs", "datasets/*/thesis-dbp-1-dataset/Implementation/"
                                    "thesis_dbp/outputs/p1/pairs")):
        links += ["--link", f"{name}={next(inputs.glob(pattern))}"]
    for command in (
            ["scripts/p2_stage_corpus.py", "--root", build / "p1", *links],
            ["scripts/p1_match_generations.py", "--gen-manifest", s2in / "p1_gen_manifest.csv",
             "--reals-manifest", "reports/p1_reals_manifest_t_real.csv", "--gen-root",
             build / "p1" / "generated", "--out-root", build / "p1", "--set-id", "gen_matched"],
            ["scripts/p2_train_s2.py", "--runs-config", "configs/p3_runs.yaml", "--protocol",
             "s2b", "--seed", "0", "--root", build / "p1", "--pairs-manifest",
             s2in / "p1_pairs_manifest.csv", "--splits", s2in / "p2_splits.csv", "--device",
             "cpu", "--fixed-steps", str(FINAL_N_MAX), "--stop-at", str(FINAL_STEP),
             "--heldout-every", "1", "--checkpoint-every", "2", "--val-pairs", "4",
             "--out-dir", build / "s2", "--reports-dir", build / "reports"]):
        done = subprocess.run([sys.executable, *[str(c) for c in command]], cwd=code, env=env,
                              capture_output=True, text=True, check=False)
        if done.returncode != 0:
            raise RuntimeError(f"fixture build step failed: {command[0]}\n"
                               f"{(done.stdout + done.stderr)[-3000:]}")
    trained = build / "s2" / "s2b_seed0"
    report = json.loads((build / "reports" / "p2_s2_s2b_seed0.json").read_text("utf-8"))
    roots = {"lolo": inputs / "datasets/muhammadadeelshah/dbp-p3-lolo-output",
             "train": inputs / "notebooks/muhammadadeelshah/dbp-p3-train",
             "tune": inputs / "notebooks/muhammadadeelshah/dbp-p3-tune",
             "s2d": inputs / "notebooks/muhammadadeelshah/dbp-p3-s2d"}
    real = json.loads((code / "env" / "p3_final_expect.json").read_text(encoding="utf-8"))
    models = []
    for spec in real["models"]:
        spec = {**spec, "expect": {}}
        if spec["kind"] == "snapshot":
            spec["step"] = FINAL_STEP
            tree = roots[spec["source"]] / "Implementation/thesis_dbp"
            shutil.copytree(trained, tree / "outputs/p3/s2" / spec["id"])
            lolo = spec["protocol"].startswith("lolo_f")
            (tree / "reports").mkdir(parents=True, exist_ok=True)
            (tree / "reports" / f"p2_s2_{spec['id']}.json").write_text(json.dumps({
                **report, "run_id": spec["id"], "protocol": spec["protocol"],
                "seed": spec["seed"], **({"steps_run": FINAL_N_MAX,
                                          "snapshots": [FINAL_STEP]} if lolo else {})}),
                encoding="utf-8")
        else:
            target = inputs / "datasets/muhammadadeelshah/dbp-p2-reference-models" / spec["id"]
            snapshot = trained / "snapshots" / f"step_{FINAL_STEP:05d}"
            shutil.copytree(snapshot / "lora", target / "lora")
            shutil.copyfile(snapshot / "head.pt", target / "head.pt")
        models.append(spec)
    # The recorded hashes are checked where they exist: two models carry them here.
    snapshot = trained / "snapshots" / f"step_{FINAL_STEP:05d}"
    from dbp.eval.heldout_scores import snapshot_digest
    models[0]["expect"] = {"snapshot_sha256": snapshot_digest(snapshot)}
    models[-1]["expect"] = {"lora_sha256": sha256_file(snapshot / "lora" /
                                                      "adapter_model.safetensors"),
                            "head_sha256": sha256_file(snapshot / "head.pt")}

    rel = lambda p: p.relative_to(code).as_posix()  # noqa: E731
    expect = {**real, "imports": old["imports"], "gpus": 0, "hf_token": False,
              "section5": {**real["section5"],
                           "degradation_choice_sha256": sha256_file(
                               code / "reports" / "p3_degradation_choice.json"),
                           "nstar": rel(nstar), "nstar_sha256": sha256_file(nstar),
                           "n_star": FINAL_STEP, "n_max": FINAL_N_MAX,
                           "oi_manifest": rel(fixture_dir / "oi_manifest.csv"),
                           "oi_manifest_sha256": sha256_file(fixture_dir / "oi_manifest.csv"),
                           "oi_train": 3, "oi_held_out": 1},
              "panel_b": {"images": rel(fixture_dir / "p3_panel_b_images.csv"),
                          "images_sha256": sha256_file(fixture_dir / "p3_panel_b_images.csv"),
                          "luma": rel(fixture_dir / "p3_panel_b_luma.csv"),
                          "luma_sha256": sha256_file(fixture_dir / "p3_panel_b_luma.csv"),
                          "digest": panel_digest(rows), "luma_digest": panel_digest(luma),
                          "test_digest": panel_digest([r for r in rows if r["role"] == "test"]),
                          "calibration_digest": panel_digest(
                              [r for r in rows if r["role"] == "calibration"])},
              "threshold_reals": {"coco_val": rel(fixture_dir / "val_reals.csv"),
                                  "reals_manifest": rel(fixture_dir / "reals_manifest.csv")},
              "models": models, "devices": "cpu", "batch": 4, "workers": 4,
              "smoke_deadline_hours": 0.5, "deadline_hours": 1.0, "poll_seconds": 2,
              "expected": {"wall_hours": 0, "basis": "dry-run fixture"}}
    expect_path = work / "p3_final_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    carry_to = inputs / "notebooks/muhammadadeelshah/dbp-p3-final"
    test_rows = sum(1 for r in rows if r["role"] == "test")

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError, StopIteration) as exc:
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        session1 = carry_to / "Implementation/thesis_dbp"
        plan1 = json.loads((session1 / "reports/p3_final_plan.json").read_text("utf-8"))
        plan2 = json.loads((dbp / "reports/p3_final_plan.json").read_text("utf-8"))
        if plan1["stage"] != "clean" or len(plan1["due"]) != 24:
            problems.append(f"session 1 plan {plan1['stage']} {len(plan1['due'])}")
        if plan2["stage"] != "robust" or plan2["due"] != expect["robust"]["models"] or \
                len(plan2["carried"]["clean"]) != 24:
            problems.append(f"session 2 plan {plan2['stage']} {plan2['due']} "
                            f"carried {len(plan2['carried']['clean'])}")
        if plan1["panel_b"]["digest"] != expect["panel_b"]["digest"] or \
                plan1["staged"]["verified"] != len(rows) + len(luma) + len(val) + 1:
            problems.append(f"session 1 staging/digest {plan1['staged']}")
        for model in plan1["due"]:
            summary = json.loads((dbp / "reports/p3_final" / f"{model}_clean.json")
                                 .read_text("utf-8"))
            sizes = {k: v["rows"] for k, v in summary["lists"].items()}
            if sizes != {"panel_b": len(rows), "luma": len(luma), "threshold_reals":
                         len(val) + 1} or summary["lora_sha256"] != \
                    plan1["models"][model]["lora_sha256"]:
                problems.append(f"{model} clean: {sizes}")
        for model in plan2["due"]:
            summary = json.loads((dbp / "reports/p3_final" / f"{model}_robust.json")
                                 .read_text("utf-8"))
            if sorted(summary["lists"]) != sorted(plan2["robust"]["conditions"]) or \
                    any(v["rows"] != test_rows for v in summary["lists"].values()):
                problems.append(f"{model} robust: {sorted(summary['lists'])}")
        # Once only: scoring a complete model again is refused.
        again = subprocess.run(
            [sys.executable, str(dbp / "scripts/p3_final_score.py"), "--members",
             plan2["due"][0], "--device", "cpu", "--plan", str(dbp / "reports/p3_final_plan.json")],
            capture_output=True, text=True, cwd=dbp, check=False,
            env=dict(os.environ, DBP_DRYRUN="1"))
        if again.returncode == 0 or "scored once per frozen model" not in again.stdout + \
                again.stderr:
            problems.append(f"a second scoring was not refused: {again.returncode}")
        print(f"  check: session 1 clean {len(plan1['due'])} models, session 2 robust "
              f"{plan2['due']} x {len(plan2['robust']['conditions'])} conditions x "
              f"{test_rows} test rows; a re-score refused (exit {again.returncode})")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, base.commit,
        [Session("session 1: section 5, Panel B and 24 models verified; every model clean"),
         Session("session 2: session 1 attached; clean carried; H5 for the 3 models")],
        carry_to=carry_to, check=check,
    )


TF_GROUPS = (("Real", "FFHQ", ""), ("Real", "FORLAB", ""), ("Fake", "FLUX.1", "animals"),
             ("Fake", "StableDiffusion1.5", "faces"), ("Fake", "StableDiffusion2", "general"),
             ("Fake", "StableDiffusion3", "landscapes"), ("Fake", "StableDiffusionXL", "faces"),
             ("Fake", "StyleGAN", "images-psi-0.5"), ("Fake", "StyleGAN2", "conf-f-psi-1"),
             ("Fake", "StyleGAN3", "conf-t-psi-0.7"))
TF_SHARES = ("PreSocial", "Facebook", "Telegram", "Twitter")


def fixture_p3_truefake_score(work: Path, ref: str) -> Fixture:
    """The TrueFake scoring in miniature: the final fixture's models, a tiny sample, COCO.

    The 4 models come from the final-scoring fixture (one trained snapshot copied to every
    id, found by content). The sample has 3 ids per real pool and 2 per generator in all 4 versions,
    in ``truefake_sample/`` of the fetch notebook's output, which is attached as is (beside
    that notebook's own code tree and reports). Planted cases: a FORLAB real
    whose pre-share file is byte-identical to a COCO original behind T-real (dropped by
    sha256), an FFHQ real whose Telegram version is an Open Images train crop's pHash
    (dropped by pHash), a fake whose Facebook version is 300 px tall (too small for the
    crop), and an SD1.5 Twitter file byte-identical to its pre-share one. The check runs
    the analysis on what the session returned and proves a re-score is refused.
    """
    base = fixture_p3_final(work, ref)
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    # As on Kaggle (2026-10-31): the tune output also carries s2b_seed0, so a runbook that
    # finds the train output by s2b_seed0 sees two outputs and must not.
    runs = "Implementation/thesis_dbp/outputs/p3/s2"
    shutil.copytree(inputs / "notebooks/muhammadadeelshah/dbp-p3-train" / runs / "s2b_seed0",
                    inputs / "notebooks/muhammadadeelshah/dbp-p3-tune" / runs / "s2b_seed0")
    # Kaggle would not publish the 8 GB sample as a dataset (user, 2026-10-29): it is the
    # fetch notebook's output, attached as is -- truefake_sample/ beside the fetch's own
    # Implementation/thesis_dbp tree and reports.
    fetch_output = inputs / "notebooks/muhammadadeelshah/dbp-truefake-fetch"
    sample_root = fetch_output / "truefake_sample"
    fetch_reports = fetch_output / "Implementation/thesis_dbp/reports/truefake"
    fetch_reports.mkdir(parents=True, exist_ok=True)
    (fetch_reports / "truefake_fetch_status.json").write_text(
        json.dumps({"outcome": "COMPLETE", "note": "dry-run fixture"}) + "\n", encoding="utf-8")
    coco = inputs / "datasets/awsaf49/coco-2017-dataset/coco2017"
    sys.path.insert(0, str(code / "src"))
    from dbp.data.phash import phash_file, phash_hex

    # --- the training reals: two COCO originals behind T-real, two Open Images train ------
    t_rows = []
    for k in range(2):
        original = coco / "train2017" / f"{k + 9:012d}.jpg"
        write_photo(original, 900 + k, (640, 480))
        t_rows.append({"sha256": f"{k:064x}", "source_sha256": sha256_file(original),
                       "source_relative_path": original.name,
                       "phash": phash_hex(phash_file(original)), "excluded_reason": ""})
    write_csv(work / "tf_t_real.csv", list(t_rows[0]), t_rows)
    oi_image = work / "tf_oi_crop.jpg"
    write_photo(oi_image, 950, (384, 384))
    oi_rows = [{"sha256": f"{k + 7:064x}", "source_sha256": f"{k + 70:064x}",
                "phash": phash_hex(phash_file(oi_image)) if k == 0 else "0f0f0f0f0f0f0f0f",
                "source_phash": "00ff00ff00ff00ff", "split": "train"} for k in range(2)]
    write_csv(work / "tf_oi.csv", list(oi_rows[0]), oi_rows)

    # --- the sample: 3 ids per real pool, 2 per generator, x 4 shares ---------------------
    # Two files are truncated as one is in TrueFake itself (Kaggle, 2026-10-31): a real's
    # Twitter version and a fake's Telegram version, each hashed as it is (so the manifest
    # check passes, as it did on Kaggle) and undecodable.
    truncated = {("FORLAB", 2, "Twitter"), ("StyleGAN3", 0, "Telegram")}
    sample_rows, manifest_rows = [], []
    seed = iter(range(30_000, 10**6))
    for kind, group, category in TF_GROUPS:
        stratum = "/".join(p for p in (kind, group, category) if p)
        for k in range(3 if kind == "Real" else 2):
            ident = f"{stratum}/{k:05d}"
            for share in TF_SHARES:
                suffix = "png" if share == "PreSocial" and kind == "Fake" else "jpg"
                member = f"TrueFake/{share}/{ident}.{suffix}"
                path = sample_root / member.removeprefix("TrueFake/")
                path.parent.mkdir(parents=True, exist_ok=True)
                if group == "FORLAB" and k == 0 and share == "PreSocial":
                    shutil.copyfile(coco / "train2017" / t_rows[0]["source_relative_path"],
                                    path)
                elif group == "FFHQ" and k == 0 and share == "Telegram":
                    shutil.copyfile(oi_image, path)
                elif group == "StableDiffusion1.5" and k == 0 and share == "Twitter":
                    shutil.copyfile(sample_root / "PreSocial" / f"{ident}.png", path)
                elif group == "StyleGAN2" and k == 1 and share == "Facebook":
                    write_photo(path, next(seed), (400, 300))
                elif suffix == "png":
                    write_image(path, next(seed), (400, 360), "PNG")
                else:
                    write_photo(path, next(seed), (420, 360))
                if (group, k, share) in truncated:
                    path.write_bytes(path.read_bytes()[:-400])
                size = path.stat().st_size
                sample_rows.append({"id": ident, "stratum": stratum, "share": share,
                                    "member": member, "compressed": str(size),
                                    "uncompressed": str(size)})
                manifest_rows.append({"member": member, "id": ident, "share": share,
                                      "stratum": stratum, "bytes": str(size),
                                      "sha256": sha256_file(path), "crc_ok": "True"})
    write_csv(work / "tf_sample.csv", list(sample_rows[0]), sample_rows)
    write_csv(work / "tf_manifest.csv", list(manifest_rows[0]), manifest_rows)

    old = json.loads(Path(base.env["DBP_EXPECT"]).read_text(encoding="utf-8"))
    expect = json.loads((code / "env" / "p3_truefake_score_expect.json")
                        .read_text(encoding="utf-8"))
    expect.update({
        "stage": "p3_truefake_score (dry-run fixture)", "imports": old["imports"], "gpus": 0,
        "hf_token": False, "final_expect": base.env["DBP_EXPECT"],
        "sample": str(work / "tf_sample.csv"),
        "sample_sha256": sha256_file(work / "tf_sample.csv"),
        "manifest": str(work / "tf_manifest.csv"),
        "manifest_sha256": sha256_file(work / "tf_manifest.csv"),
        "members": len(manifest_rows), "t_real_manifest": str(work / "tf_t_real.csv"),
        "oi_manifest": str(work / "tf_oi.csv"), "devices": "cpu", "batch": 4, "workers": 4,
        "smoke_deadline_hours": 0.5, "deadline_hours": 1.0, "poll_seconds": 2,
        "expected": {"wall_hours": 0, "basis": "dry-run fixture"}})
    expect_path = work / "p3_truefake_score_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    scored = len(manifest_rows) - 2 * len(TF_SHARES) - 1 - len(truncated)

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError, StopIteration) as exc:
            return [f"the session did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        plan = json.loads((dbp / "reports/p3_truefake_plan.json").read_text("utf-8"))
        if plan["stage"] != "truefake" or plan["due"] != expect["models"]:
            problems.append(f"plan {plan['stage']} {plan['due']}")
        if plan["staged"]["verified"] != len(manifest_rows) or \
                Path(plan["staged"]["sample_root"]) != sample_root:
            problems.append(f"staging {plan['staged']} (expected the fetch notebook's "
                            f"{sample_root})")
        pools = plan["disjointness"]["pools"]
        if pools != {"Real/FFHQ": {"ids": 3, "dropped_ids": 1},
                     "Real/FORLAB": {"ids": 3, "dropped_ids": 1}}:
            problems.append(f"disjointness {pools}")
        how = {i: [m["sha256_match"] for m in v["members"]]
               for i, v in plan["disjointness"]["matched"].items()}
        if True not in how.get("Real/FORLAB/00000", []) or \
                "Real/FFHQ/00000" not in how:
            problems.append(f"matched {how}")
        coco_record = plan["disjointness"]["references"]["coco_originals"]
        if coco_record["hashed"] != 2:
            problems.append(f"COCO originals {coco_record}")
        broken = sorted(u["member"] for u in plan["list"]["unreadable"])
        if plan["list"]["too_small_members"] != 1 or \
                plan["list"]["scored_members"] != scored or len(broken) != len(truncated) or \
                any("truncated" not in u["error"] for u in plan["list"]["unreadable"]):
            problems.append(f"list {plan['list']}")
        for model in expect["models"]:
            summary = json.loads((dbp / "reports/p3_truefake" / f"{model}.json")
                                 .read_text("utf-8"))
            if summary["lists"]["truefake"]["rows"] != scored or summary["crop"] != 320 or \
                    summary["lora_sha256"] != plan["models"][model]["lora_sha256"]:
                problems.append(f"{model}: {summary['lists']}")
        again = subprocess.run(
            [sys.executable, str(dbp / "scripts/p3_truefake_score.py"), "--members",
             expect["models"][0], "--device", "cpu", "--plan",
             str(dbp / "reports/p3_truefake_plan.json")],
            capture_output=True, text=True, cwd=dbp, check=False,
            env=dict(os.environ, DBP_DRYRUN="1"))
        if again.returncode == 0 or "scored once per frozen model" not in again.stdout + \
                again.stderr:
            problems.append(f"a second scoring was not refused: {again.returncode}")
        analysis = load_script(dbp / "scripts" / "p3_truefake_analyze.py", "_tf_fixture_ana")
        lists = dbp / "reports/kaggle_20261027/p3_final_session1/p3_final_lists"
        result = analysis.run(dbp / "reports/p3_truefake_list.csv",
                              dbp / "reports/p3_truefake_scores", dbp / "reports/p3_truefake",
                              lists / "threshold_reals.csv", dbp / "reports/p3_final_scores",
                              resamples=20)
        counts = result["counts"]
        if counts["excluded"] != {"dropped": {"FFHQ": 1, "FORLAB": 1},
                                  "too_small": {"StyleGAN2": 1},
                                  "unreadable": {"FORLAB": 1, "StyleGAN3": 1}} or \
                counts["twitter_identical"]["StableDiffusion1.5"] != 1:
            problems.append(f"analysis counts {counts}")
        print(f"  check: {len(manifest_rows)} members staged and hash-checked; dropped "
              f"{ {k: v['dropped_ids'] for k, v in pools.items()} }; too small 1; "
              f"undecodable {broken}; "
              f"{len(expect['models'])} models x {scored} rows; re-score refused (exit "
              f"{again.returncode}); analysis ran: excluded {counts['excluded']}, Twitter "
              f"identical SD1.5 {counts['twitter_identical']['StableDiffusion1.5']}")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, base.commit,
        [Session("session 1: sample, 4 models, staging and disjointness verified; scored")],
        check=check)


P4_TINY_DDA = {"embed_dim": 32, "depth": 2, "num_heads": 2}
P4_SIGLIP = "dbp-dryrun-siglip"
#: A tiny open_clip config (its own ViT, not SigLIP), so build_siglip's local-file path runs.
P4_SIGLIP_CONFIG = {"embed_dim": 16,
                    "vision_cfg": {"image_size": 384, "layers": 1, "width": 32,
                                   "head_width": 16, "patch_size": 32},
                    "text_cfg": {"context_length": 8, "vocab_size": 64, "width": 16,
                                 "heads": 1, "layers": 1}}
P4_HUB = ("models--stable-diffusion-v1-5--stable-diffusion-v1-5",
          "models--kandinsky-community--kandinsky-2-2-decoder")


def p4_hub_cache() -> None:
    """Copy the pinned sd15 and Kandinsky 2.2 caches into the dry-run cache when absent."""
    cache = THESIS_DBP / ".cache" / "dryrun" / "tmp" / "hf_home" / "hub"
    for repo in P4_HUB:
        local = THESIS_DBP / ".cache" / "hf" / "hub" / repo
        if not (cache / repo).exists() and local.exists():
            shutil.copytree(local, cache / repo)


def p4_tiny_weights(code: Path, work: Path) -> dict[str, Path]:
    """Tiny stand-ins for the baselines' weight files, in the real files' layouts.

    DDA: ``{"model": state}`` of the tiny ``DDADetector``; SigLIP: a tiny open_clip model
    as ``.safetensors`` (with its config directory); CO-SPY: the three ``.pth`` files in a
    zip under ``pretrained/sd-v1_4/``; FIRE: ``FIRE_model``'s state without VAE keys (so the
    pinned ``sd15`` is used). Built with the fixture's own code copy.
    """
    import io
    import zipfile

    import torch

    sys.path.insert(0, str(code / "src"))
    os.environ["DBP_DRYRUN"] = "1"  # the tiny architectures are dry-run seams
    import open_clip
    from safetensors.torch import save_file

    from dbp.baselines import cospy, dda, fire

    source = work / "p4_local_weights"
    source.mkdir(parents=True)
    torch.manual_seed(2026)
    torch.save({"model": dda.DDADetector(P4_TINY_DDA).state_dict(), "epoch": 0},
               source / "DDA_ckpt.pth")
    config_dir = work / "p4_siglip_config"
    config_dir.mkdir()
    (config_dir / f"{P4_SIGLIP}.json").write_text(json.dumps(P4_SIGLIP_CONFIG), "utf-8")
    open_clip.add_model_config(config_dir)
    clip = open_clip.create_model(P4_SIGLIP, pretrained=None)
    save_file({k: v.contiguous() for k, v in clip.state_dict().items()},
              str(source / "open_clip_model.safetensors"))
    dim = P4_SIGLIP_CONFIG["embed_dim"]
    states = {"semantic": {"fc.weight": torch.randn(1, dim), "fc.bias": torch.randn(1)},
              "artifact": cospy.ArtifactDetector(None).state_dict(),
              "fusion": {"fc.weight": torch.randn(1, 2), "fc.bias": torch.randn(1)}}
    with zipfile.ZipFile(source / "sd-v1_4.zip", "w") as archive:
        for name, state in states.items():
            buffer = io.BytesIO()
            torch.save(state, buffer)
            archive.writestr(f"pretrained/sd-v1_4/{name}_weights.pth", buffer.getvalue())
            (source / f"{name}_weights.pth").write_bytes(buffer.getvalue())
    torch.save(fire.FIREDetector(None).state_dict(), source / "fire_imagenet_adm.pth")
    return {"source": source, "config_dir": config_dir}


def fixture_p4_weights(work: Path, ref: str) -> Fixture:
    """The weights session in miniature: tiny stand-ins served by ``--local-source``.

    The fetch, the size and sha256 checks, the CO-SPY extraction, FIRE's recorded hash and
    every ``weights_only=True`` inspection run on the stand-ins; the pinned ``sd15`` VAE is
    the real one (the inspection compares against it).
    """
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    commit = copy_code(code, ref)
    p4_hub_cache()
    made = p4_tiny_weights(code, work)
    source = made["source"]
    expect = json.loads((code / "env" / "p4_weights_expect.json").read_text("utf-8"))
    for spec in expect["hf"].values():
        path = source / spec["file"]
        spec.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    expect.update({"stage": "p4_weights (dry-run fixture)", "local_source": str(source),
                   "dryrun_dda_arch": P4_TINY_DDA, "siglip_model": P4_SIGLIP,
                   "dryrun_siglip_config": str(made["config_dir"]),
                   "expected": {"wall": "dry-run fixture", "quota": "none"}})
    expect_path = work / "p4_weights_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        problems: list[str] = []
        try:
            dbp = work_dir / "working/Implementation/thesis_dbp"
            report = json.loads((dbp / "reports/p4_weights.json").read_text("utf-8"))
            weights = work_dir / "working/p4_weights"
            fire_sha = sha256_file(source / "fire_imagenet_adm.pth")
            if not report["ok"] or report["files"]["fire"]["sha256"] != fire_sha:
                problems.append(f"report {report.get('problems')} fire "
                                f"{report['files'].get('fire')}")
            names = sorted(p.name for p in (weights / "cospy_zip").glob("*_weights.pth"))
            if names != ["artifact_weights.pth", "fusion_weights.pth", "semantic_weights.pth"]:
                problems.append(f"CO-SPY extraction {names}")
            checks = report["checks"]
            if checks["fire"]["vae"] != "none" or checks["cospy"]["artifact_vae_keys"] or \
                    not checks["dda"].get("strict_load"):
                problems.append(f"inspection {checks}")
            print(f"  check: {len(report['files'])} files hash-checked; FIRE sha256 recorded "
                  f"{fire_sha[:16]}...; CO-SPY extracted {names}; DDA strict load; FIRE VAE "
                  f"{checks['fire']['vae']}")
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"the session did not leave the expected results: "
                            f"{type(exc).__name__}: {exc}")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1: fetch (local stand-ins), hash, extract, inspect")],
                   check=check)


P4_STAND_IN = '''"""DRY-RUN STAND-IN for DDA's model file -- NOT DDA's code.

DDA's own code runs only in the isolated Kaggle notebook (user decision 2026-11-01); the
local dry-run tests the runbook's plumbing with this stand-in instead.
"""
from dbp.baselines.dda import DDADetector


class DINOv2ModelWithLoRA(DDADetector):
    def __init__(self, name, num_classes=1, lora_rank=8, lora_alpha=1.0, lora_targets=None):
        super().__init__({"embed_dim": 32, "depth": 2, "num_heads": 2})
'''


def fixture_p4_dda_equivalence(work: Path, ref: str) -> Fixture:
    """The isolated check in miniature: the code-only bundle, two COCO-named images.

    The bundle is built by ``build_code_zip.extract(..., bundle="p4-dda-equiv")`` -- the
    upload's bytes -- and is the only input beside a public-COCO-shaped dataset, so the
    runbook's isolation assertion runs. DDA's archive is a stand-in (never DDA's code
    locally), so the two implementations agree exactly; the check proves the plumbing.
    """
    import zipfile

    import torch

    inputs = work / "input"
    bundle = inputs / "datasets/muhammadadeelshah/dbp-p4-dda-equiv/thesis_dbp"
    sys.path.insert(0, str(THESIS_DBP / "scripts"))
    import build_code_zip

    commit = build_code_zip.extract(ref, bundle, bundle="p4-dda-equiv")
    expect = json.loads((bundle / "env" / "p4_dda_equivalence_expect.json").read_text("utf-8"))
    coco = inputs / "datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
    for k, name in enumerate(expect["images"]):
        write_photo(coco / name, 960 + k, (640, 416) if k == 0 else (416, 640))
    sys.path.insert(0, str(bundle / "src"))
    os.environ["DBP_DRYRUN"] = "1"
    from dbp.baselines import dda

    torch.manual_seed(2026)
    ckpt = work / "p4_equiv_ckpt.pth"
    torch.save({"model": dda.DDADetector(P4_TINY_DDA).state_dict()}, ckpt)
    archive = work / "p4_equiv_archive.zip"
    root = f"Dual-Data-Alignment-{expect['dda_commit']}"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(f"{root}/Inference/models/dinov2_models_lora.py", P4_STAND_IN)
        handle.writestr(f"{root}/README.md", "dry-run stand-in\n")
    expect["dda_ckpt"]["sha256"] = sha256_file(ckpt)
    expect.update({"stage": "p4_dda_equivalence (dry-run fixture)", "local_archive": str(archive),
                   "local_ckpt": str(ckpt), "dryrun_dda_arch": P4_TINY_DDA,
                   "expected": {"wall": "dry-run fixture", "quota": "none"}})
    expect_path = work / "p4_dda_equivalence_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")

    def check(work_dir: Path) -> list[str]:
        problems: list[str] = []
        try:
            result = json.loads((work_dir / "working/p4_dda_equivalence.json")
                                .read_text("utf-8"))
            if not result["pass"] or result["max_abs_diff"] != 0.0 or \
                    len(result["images"]) != 2:
                problems.append(f"result {result}")
            if (bundle / "reports").exists():
                problems.append("the bundle carries reports/")
            files = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*")
                           if p.is_file())
            print(f"  check: bundle of {len(files)} files, no reports/; both logits per "
                  f"image agree (max |diff| {result['max_abs_diff']})")
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"the session did not leave the expected results: "
                            f"{type(exc).__name__}: {exc}")
        return problems

    return Fixture({"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
                   [Session("session 1: isolation asserted; both implementations scored")],
                   check=check)


P4_PANEL = (("e3_matched", "fake", "test", "GPT-Image2-fake", (420, 360)),
            ("e3_matched", "real", "test", "real-test", (300, 280)),
            ("e4_wildrf_matched", "fake", "test", "fake", (820, 800)),
            ("e4_wildrf_matched", "real", "calibration", "real-calibration", (400, 360)),
            ("e7_dda_coco", "fake", "test", "sdxl-vae", (500, 376)),
            ("e8_evalgen", "fake", "test", "Flux", (384, 384)))
P4_LUMA = (("e7_dda_coco_luma", "fake", "test", "FLUX.1", (400, 400)),
           ("e7_dda_coco_luma", "real", "test", "real-test", (400, 400)))


def fixture_p4_baselines(work: Path, ref: str) -> Fixture:
    """The P4 scoring in miniature: 7 rows on a tiny Panel B, two sessions.

    The weights are the weights fixture's stand-ins, laid out as the weights notebook's
    output (``p4_weights/`` plus its report). Planted: a 300 x 280 real (DDA pads it,
    CO-SPY upscales it), an 820 x 800 fake (the 768 cap cuts it), and long sides FIRE must
    crop to a multiple of 8. AEROBLADE uses the cached ``sd15`` and Kandinsky 2.2 MoVQ (the
    same ``vq`` path as 2.1). Session 2 attaches session 1: every row is carried forward
    and the runbook stops on "already scored"; the check proves a re-score is refused.
    """
    inputs = work / "input"
    code = inputs / "datasets/muhammadadeelshah/dbp-code-fixture/thesis_dbp"
    commit = copy_code(code, ref)
    p4_hub_cache()
    made = p4_tiny_weights(code, work)
    source = made["source"]
    output = inputs / "notebooks/muhammadadeelshah/dbp-p4-weights"
    layout = {"dda": "dda/DDA_ckpt.pth", "siglip": "siglip/open_clip_model.safetensors",
              "cospy_semantic": "cospy_zip/semantic_weights.pth",
              "cospy_artifact": "cospy_zip/artifact_weights.pth",
              "cospy_fusion": "cospy_zip/fusion_weights.pth",
              "fire": "fire/fire_imagenet_adm.pth"}
    names = {"dda": "DDA_ckpt.pth", "siglip": "open_clip_model.safetensors",
             "cospy_semantic": "semantic_weights.pth", "cospy_artifact": "artifact_weights.pth",
             "cospy_fusion": "fusion_weights.pth", "fire": "fire_imagenet_adm.pth"}
    for key, relative in layout.items():
        target = output / "p4_weights" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / names[key], target)
    report = output / "Implementation/thesis_dbp/reports/p4_weights.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"ok": True, "note": "dry-run fixture"}) + "\n", "utf-8")

    panel = inputs / "datasets/muhammadadeelshah/dbp-p3-panel-b/panel_b"
    columns = ["image_id", "relative_path", "label", "view", "role", "slot", "sha256"]

    def panel_rows(spec: tuple, offset: int) -> list[dict[str, str]]:
        rows = []
        for k, (view, label, role, slot, size) in enumerate(spec):
            image_id = f"p4_{offset + k:03d}"
            path = panel / view / f"{image_id}.png"
            write_image(path, 700 + offset + k, size, "PNG")
            rows.append({"image_id": image_id, "relative_path": f"panel/{view}/{image_id}.png",
                         "label": label, "view": view, "role": role, "slot": slot,
                         "sha256": sha256_file(path)})
        return rows

    lists = {"panel_b": panel_rows(P4_PANEL, 0), "luma": panel_rows(P4_LUMA, 50)}
    real = inputs / "datasets/muhammadadeelshah/report/p1_reals/t_real/r_p4_0.png"
    oi = inputs / "datasets/muhammadadeelshah/dbp-p3-open-images/oi/reals/t_oi/o_p4_0.png"
    write_image(real, 790, (384, 384), "PNG")
    write_image(oi, 791, (512, 384), "PNG")
    lists["threshold_reals"] = [
        {"image_id": "r_p4_0", "relative_path": "reals/t_real/r_p4_0.png", "label": "real",
         "view": "coco_val", "role": "threshold", "slot": "coco_val", "sha256": sha256_file(real)},
        {"image_id": "o_p4_0", "relative_path": "oi_reals/t_oi/o_p4_0.png", "label": "real",
         "view": "oi_held_out", "role": "threshold", "slot": "oi_held_out",
         "sha256": sha256_file(oi)}]
    list_dir = code / "reports" / "dryrun_fixture" / "p4_lists"
    for name, rows in lists.items():
        write_csv(list_dir / f"{name}.csv", columns, rows)
    test_rows = sum(1 for r in lists["panel_b"] if r["role"] == "test")

    # The real results committed in the code would mark 6 rows complete; the fixture scores
    # all seven on its own lists, so the code copy drops them (the copy only).
    for committed in ("p4_baselines", "p4_baseline_scores"):
        shutil.rmtree(code / "reports" / committed, ignore_errors=True)
    expect = json.loads((code / "env" / "p4_baselines_expect.json").read_text("utf-8"))
    rows = list(expect["rows"])
    unsharded = [r for r in rows if r != "aeroblade"]
    expect.update({
        "stage": "p4_baselines (dry-run fixture)", "gpus": 0, "hf_token": False,
        "lists": {"dir": list_dir.relative_to(code).as_posix(), "panel_b_test_rows": test_rows,
                  **{name: {"sha256": sha256_file(list_dir / f"{name}.csv"),
                            "rows": len(items)} for name, items in lists.items()}},
        "weights": {k: {"file": v, "sha256": sha256_file(output / "p4_weights" / v)}
                    for k, v in layout.items()},
        "aeroblade_aes": ["sd15", "movq_k22"],
        "shards": {"aeroblade": {"n": 2, "checkpoint_every": 2, "resume_rescore": 20}},
        "jobs": [f"cpu:{','.join(unsharded)},aeroblade#0", "cpu:aeroblade#1"],
        "batch": {"dda": 4, "cospy": 2}, "workers": 4, "margin_gb": 0.5,
        "smoke_deadline_hours": 1.0, "deadline_hours": 2.0, "poll_seconds": 2,
        "dryrun_arch": {"dda": P4_TINY_DDA,
                        "siglip": {"model": P4_SIGLIP, "config": str(made["config_dir"])}},
        "expected": {"wall": "dry-run fixture", "quota": "none"}})
    expect_path = work / "p4_baselines_expect.json"
    expect_path.write_text(json.dumps(expect, indent=2) + "\n", encoding="utf-8")
    carry_to = inputs / "notebooks/muhammadadeelshah/dbp-p4-baselines"
    flags = {"dda_native": ("padded", 1), "dda_crop308": ("padded", 1),
             "cospy_native": ("upscaled", 1), "aeroblade": ("cut_by_cap", 1)}

    def check(work_dir: Path) -> list[str]:
        try:
            return check_results(work_dir)
        except (OSError, KeyError, ValueError, IndexError) as exc:
            return [f"the sessions did not leave the expected results: "
                    f"{type(exc).__name__}: {exc}"]

    def check_results(work_dir: Path) -> list[str]:
        problems: list[str] = []
        dbp = work_dir / "working/Implementation/thesis_dbp"
        session1 = carry_to / "Implementation/thesis_dbp"
        plan1 = json.loads((session1 / "reports/p4_plan.json").read_text("utf-8"))
        plan2 = json.loads((dbp / "reports/p4_plan.json").read_text("utf-8"))
        units = [*unsharded, "aeroblade#0", "aeroblade#1"]
        if plan1["due"] != units or plan1["staged"]["verified"] != 10:
            problems.append(f"session 1 plan {plan1['due']} staged {plan1['staged']}")
        # Session 1 rehearsed a deadline: each shard stopped after one part of 2 images.
        shard1 = plan2["shards"]["aeroblade"]["units"]["aeroblade#1"]
        if shard1["saved_rows"] != 2 or plan2["shards"]["aeroblade"]["carried_files"] < 3:
            problems.append(f"session 2 did not resume shard 1: {shard1} carried "
                            f"{plan2['shards']['aeroblade']['carried_files']}")
        projection = json.loads((dbp / "reports/p4_projection.json").read_text("utf-8"))
        timed = [u for j in projection["jobs"] for u in j["units"]]
        if not projection["ok"] or {u["unit"] for u in timed} != set(plan2["due"]) or \
                any(u["rate"] <= 0 for u in timed):
            problems.append(f"fail-fast projection {projection}")
        resumes = [json.loads(line) for line in
                   (dbp / "reports/p4_baseline_scores/aeroblade/_shards/shard1/resumes.jsonl")
                   .read_text("utf-8").splitlines()]
        if not resumes or not all(r["ok"] for r in resumes) or resumes[0]["checked"] != 2:
            problems.append(f"resume checks {resumes}")
        merged = json.loads((dbp / "reports/p4_baselines/aeroblade.json").read_text("utf-8"))
        shards = merged["execution"]["shards"]
        if sum(s["items"] for s in shards.values()) != 10 or len(shards) != 2:
            problems.append(f"merged shards {shards}")
        for row in rows:
            summary = json.loads((dbp / "reports/p4_baselines" / f"{row}.json")
                                 .read_text("utf-8"))
            sizes = {k: v["rows"] for k, v in summary["lists"].items()}
            want = {"panel_b_test": test_rows} if expect["rows"][row]["condition"] != \
                "clean" else {"panel_b": 6, "luma": 2, "threshold_reals": 2}
            if sizes != want:
                problems.append(f"{row}: {sizes}")
            if row in flags:
                key, at_least = flags[row]
                if summary["lists"]["panel_b"][key] < at_least:
                    problems.append(f"{row}: {key} {summary['lists']['panel_b'][key]}")
        fire_summary = json.loads((dbp / "reports/p4_baselines/fire_native.json")
                                  .read_text("utf-8"))
        if fire_summary["vae_source"] != "pinned sd15" or \
                fire_summary["lists"]["panel_b"]["cropped"] < 1:
            problems.append(f"FIRE {fire_summary['vae_source']} {fire_summary['lists']}")
        # A complete row is never scored or merged again.
        again = subprocess.run(
            [sys.executable, str(dbp / "scripts/p4_merge.py"), "--plan",
             str(dbp / "reports/p4_plan.json")],
            capture_output=True, text=True, cwd=dbp, check=False,
            env=dict(os.environ, DBP_DRYRUN="1"))
        if again.returncode != 0 or "not merged again" not in again.stdout + again.stderr:
            problems.append(f"a second merge was not refused: {again.returncode}")
        rescore = subprocess.run(
            [sys.executable, str(dbp / "scripts/p4_score.py"), "--members", "aeroblade#1",
             "--device", "cpu", "--plan", str(dbp / "reports/p4_plan.json")],
            capture_output=True, text=True, cwd=dbp, check=False,
            env=dict(os.environ, DBP_DRYRUN="1"))
        if rescore.returncode == 0:
            problems.append("a finished shard was scored again")
        print(f"  check: session 1 stopped each shard after one part (rehearsed deadline); "
              f"session 2 resumed shard 1 from {shard1['saved_rows']} saved rows "
              f"(re-score check: {resumes[0]['checked']} rows, ok), fail-fast projected "
              f"{ {j['device']: j['hours'] for j in projection['jobs']} } h; merged "
              f"{ {u: s['items'] for u, s in shards.items()} } into aeroblade; flags "
              f"{[(r, flags[r][0]) for r in flags]} and FIRE crops recorded; second merge and "
              f"re-score refused")
        return problems

    return Fixture(
        {"DBP_INPUT_ROOT": str(inputs), "DBP_EXPECT": str(expect_path)}, commit,
        [Session("session 1: 7 rows; each AEROBLADE shard stops after one part (rehearsed "
                 "deadline)", env={"DBP_P4_STOP_AFTER_PARTS": "1"}, expect_exit="nonzero"),
         Session("session 2: session 1 attached; shards resume (re-score check); merged")],
        carry_to=carry_to, check=check)


FIXTURES: dict[str, Callable[[Path, str], Fixture]] = {
    "p2_s3_recovery": fixture_p2_s3_recovery,
    "p2_s2": fixture_p2_s2,
    "x1_format": fixture_x1_format,
    "p3_panel_b": fixture_p3_panel_b,
    "p3_open_images": fixture_p3_open_images,
    "p3_flickr_probe": fixture_p3_flickr_probe,
    "p3_oi_accept": fixture_p3_oi_accept,
    "p3_lolo": fixture_p3_lolo,
    "p3_train": fixture_p3_train,
    "p3_reconstruct": fixture_p3_reconstruct,
    "p3_tune": fixture_p3_tune,
    "p3_dev_panel": fixture_p3_dev_panel,
    "p3_truefake": fixture_p3_truefake,
    "p3_s2d": fixture_p3_s2d,
    "p3_final": fixture_p3_final,
    "p3_truefake_score": fixture_p3_truefake_score,
    "p4_weights": fixture_p4_weights,
    "p4_dda_equivalence": fixture_p4_dda_equivalence,
    "p4_baselines": fixture_p4_baselines,
}


# ------------------------------------------------------------------- driver ---


def remove_tree(path: Path) -> None:
    """Delete a fixture tree without following a junction into its target."""
    if not path.exists():
        return
    for item in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        is_junction = getattr(item, "is_junction", None)
        if item.is_symlink() or (is_junction and is_junction()):
            if item.is_dir() and not item.is_symlink():
                os.rmdir(item)
            else:
                item.unlink()
    shutil.rmtree(path)


def main() -> int:
    """Build the fixture and run the runbook. Returns the runbook's exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runbook", required=True, type=Path)
    parser.add_argument("--fixture", required=True, choices=sorted(FIXTURES))
    parser.add_argument("--work", type=Path, default=THESIS_DBP / ".cache" / "dryrun" / "work")
    parser.add_argument("--ref", default=":",
                        help="Code to test: ':' = the git index (what the next commit will "
                             "hold), or a commit such as HEAD.")
    parser.add_argument("--cache", type=Path, default=THESIS_DBP / ".cache" / "dryrun" / "tmp",
                        help="Persistent model caches (HF_HOME, TORCH_HOME, weights).")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # cell titles hold em dashes

    runbook = args.runbook if args.runbook.is_absolute() else THESIS_DBP / args.runbook
    cells = code_cells(parse_cells(runbook.read_text(encoding="utf-8")))
    print(f"runbook : {runbook.relative_to(THESIS_DBP)}  ({len(cells)} code cells)")
    for cell in cells:
        print(f"  cell {cell.index}  sha256 {cell.sha256[:16]}  {cell.title}")

    remove_tree(args.work)
    args.work.mkdir(parents=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    print(f"fixture : {args.fixture} -> {args.work}")
    built = time.perf_counter()
    fixture = FIXTURES[args.fixture](args.work, args.ref)
    print(f"          built in {time.perf_counter() - built:.1f} s; code at {fixture.commit}")

    script = args.work / "runbook_exact.py"
    script.write_text("\n\n".join(c.text for c in cells), encoding="utf-8")
    venv_scripts = Path(sys.executable).parent
    started = time.perf_counter()
    failures: list[str] = []
    working = args.work / "working"
    for number, session in enumerate(fixture.sessions, start=1):
        if number > 1 and fixture.carry_to is not None:
            # Kaggle: the previous version's output, attached as an input; a fresh
            # working directory for the new session.
            fixture.carry_to.parent.mkdir(parents=True, exist_ok=True)
            if fixture.carry_to.exists():         # only the latest version is attached
                shutil.rmtree(fixture.carry_to)
            os.replace(working, fixture.carry_to)
        env = dict(
            os.environ,
            **fixture.env,
            **session.env,
            DBP_WORKING_ROOT=str(working),
            DBP_TMP_ROOT=str(args.cache),
            DBP_DEVICE="cpu",
            DBP_DRYRUN="1",
            PYTHONUTF8="1",
            PATH=f"{venv_scripts}{os.pathsep}{os.environ.get('PATH', '')}",
        )
        print(f"\n=== {session.label} ===\nexecuting the {len(cells)} code cells verbatim "
              f"...\n" + "-" * 78, flush=True)
        result = subprocess.run([sys.executable, "-u", str(script)], cwd=args.work, env=env)
        print("-" * 78)
        ok = (result.returncode == 0) == (session.expect_exit == "zero")
        print(f"session {number}: exit {result.returncode}, expected {session.expect_exit} "
              f"-> {'as expected' if ok else 'WRONG'}")
        if not ok:
            failures.append(f"session {number} exited {result.returncode}")
            break
    if not failures and fixture.check is not None:
        print("\n=== checks on what the sessions left ===")
        os.environ["DBP_DRYRUN_TMP"] = str(args.cache)  # the sessions' /tmp, for the checks
        failures += fixture.check(args.work)
    elapsed = time.perf_counter() - started
    verdict = "PASSED" if not failures else f"FAILED ({'; '.join(failures)})"
    print(f"\nLOCAL DRY-RUN: {verdict}  in {elapsed:.0f} s  "
          f"[runbook {runbook.name}, code {fixture.commit}, fixture {args.fixture}, "
          f"{len(fixture.sessions)} session(s)]")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
