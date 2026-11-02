# %% [markdown]
# # P3 Phase B — draw and freeze Panel B (Kaggle CPU)
#
# Panel B is P3's once-only test panel (P3 pre-registration §2, FROZEN 2026-10-08). This
# session rebuilds the five evaluation views exactly as P1 did, proves every P2 panel row
# reproduces byte-identically, draws Panel B from the unused images
# (`dbp.eval.panel_b`), copies it into one verified directory and writes its digest.
#
# **Accelerator: None (CPU).** No GPU quota is used. **Internet:** on. **Secret:**
# `HF_TOKEN` (optional; the three HF datasets are not gated). **Mode:** Save & Run All
# (Commit). **Account:** `muhammadadeelshah` only (CLAUDE.md section 13).
#
# **Inputs to attach** (Cell 1 finds each by name):
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code, at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | generation and bank-pairs manifests (dedup references) |
# | dataset `chameleon-dataset` | E5 (Chameleon), extracted, private |
# | dataset `coco-2017-dataset` (awsaf49) | COCO `val2017`: E7's 5,000 reals (DDA-COCO has none) |
#
# E3, E7 and E8 are fetched from the Hugging Face Hub at P1's pinned revisions, and E4
# from Google Drive with `gdown` (installed by the bootstrap's opt-in, session only).
# **Publish the output as a PRIVATE dataset** (WildRF is research-only, Chameleon
# academic-only; section 6c).

# %% Cell 1 — find the inputs, and the helpers every later cell uses
import json
import os
import pathlib
import shutil
import subprocess
import sys

IN = pathlib.Path(os.environ.get("DBP_INPUT_ROOT", "/kaggle/input"))
WORK = pathlib.Path(os.environ.get("DBP_WORKING_ROOT", "/kaggle/working"))
TMP = pathlib.Path(os.environ.get("DBP_TMP_ROOT", "/tmp"))


def find(pattern):
    """The one attached input matching ``pattern``, whichever account owns it."""
    hits = sorted(p for p in IN.glob(pattern) if p.is_dir())
    assert len(hits) == 1, f"expected exactly one input matching {pattern!r}, found {hits}"
    return hits[0]


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "chameleon": find("datasets/*/chameleon-dataset/Chameleon"),
    "coco_val": find("datasets/*/coco-2017-dataset/coco2017/val2017"),
}
DBP = WORK / "Implementation" / "thesis_dbp"
PANEL_B = WORK / "panel_b"
ROOTS = WORK / "roots.json"
for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
    os.environ[key] = str(TMP / key.lower())
    os.makedirs(os.environ[key], exist_ok=True)


def sh(*command):
    """Run a command in the staged repo, streaming its output into this cell."""
    print("$", " ".join(str(c) for c in command), flush=True)
    with subprocess.Popen([str(c) for c in command], cwd=DBP, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
    return proc.returncode


def py(script, *args):
    """Run one committed script."""
    return sh(sys.executable, "-u", f"scripts/{script}", *args)


for name, path in INPUTS.items():
    print(f"{name:10s} {path}")

# %% Cell 2 — stage the code; set up the environment (CPU session, gdown opt-in)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_build_panel_b.py").is_file(), "the code dataset predates Panel B"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_panel_b_expect.json")).read_text())
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
os.environ["DBP_WITH_GDOWN"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: imports, input hashes and paths, before anything long starts
S2IN = INPUTS["s2_inputs"]
P2_SETS = pathlib.Path(EXPECT["p2_panel_sets"]) if EXPECT.get("p2_panel_sets") else \
    DBP / "reports" / "p2_panel_sets"
checks = ["--imports", ",".join(EXPECT["imports"]),
          "--require", f"chameleon={INPUTS['chameleon']}", "--require", f"p2_sets={P2_SETS}",
          "--require", f"coco_val={INPUTS['coco_val']}"]
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}",
               "--rows", f"{S2IN / name}={spec['rows']}"]
assert py("p2_preflight.py", "--stage", "p3_panel_b_data", *checks) == 0, \
    "PRE-FLIGHT FAILED -- nothing long has started"

# %% Cell 4 — fetch E3/E4/E7/E8 at P1's pinned sources (E5 attached; E7 reals = val2017)
local = [f"--local=e5={INPUTS['chameleon']}"]
local += [f"--local={name}={path}" for name, path in EXPECT.get("local", {}).items()]
coco = ["--coco-val", INPUTS["coco_val"]]
coco += ["--coco-val-count", EXPECT["coco_val_count"]] if EXPECT.get("coco_val_count") else []
assert py("p3_fetch_eval_sets.py", "--tmp", TMP, "--roots-out", ROOTS, *local, *coco) == 0, \
    "FETCH FAILED"

# %% Cell 5 — rebuild the five views with P1's recorded arguments; verify every P2 row
views = ["--roots", ROOTS, "--p2-panel-sets", P2_SETS]
views += ["--summaries", EXPECT["summaries"]] if EXPECT.get("summaries") else []
views += [] if EXPECT.get("label_check", True) else ["--no-label-check"]
assert py("p3_build_eval_views.py", *views) == 0, "VIEW REBUILD FAILED"

# %% Cell 6 — draw Panel B, then copy, verify and freeze it
draw = ["--p2-panel-sets", P2_SETS, "--gen-manifest", S2IN / "p1_gen_manifest.csv",
        "--pairs-manifest", S2IN / "p1_pairs_manifest.csv"]
draw += ["--slots", EXPECT["slots"]] if EXPECT.get("slots") else []
assert py("p3_build_panel_b.py", *draw) == 0, "PANEL B DRAW FAILED"
assert py("p3_materialise_panel_b.py", "--roots", ROOTS, "--panel-root", PANEL_B,
          "--cleanup-derived", DBP / "outputs" / "p1" / "eval") == 0, "MATERIALISE FAILED"

# %% Cell 7 — report (cannot fail the run)
for name in ("p3_eval_views_rebuild.json", "p3_panel_b_summary.json", "p3_panel_b_digest.json"):
    path = DBP / "reports" / name
    print(f"===== {name}\n" + (path.read_text() if path.is_file() else "(not written)"))

# %% Cell 8 — verdict: this cell fails the notebook if Panel B was not frozen
DIGEST = json.loads((DBP / "reports" / "p3_panel_b_digest.json").read_text())
assert DIGEST["rows"] > 0 and len(DIGEST["panel_digest"]) == 64, "Panel B is empty"
print("PANEL B FROZEN:", DIGEST["rows"], "rows, digest", DIGEST["panel_digest"])
