# %% [markdown]
# # P3 — rebuild the P2 development panel on `muhammadadeelshah` (Kaggle CPU)
#
# P2 read its panel from the `phase-2` notebook output of `adeelshah29`. Every thesis run
# uses `muhammadadeelshah` only (CLAUDE.md §13), and the panel holds WildRF and Chameleon
# (§6c), so P3 uses **only** this rebuilt copy; the `adeelshah29` copy is never attached
# again (user decision 2026-10-19).
#
# The session re-derives the evaluation views exactly as the Panel B session did, stages
# the corpus for the T-ctrl diagnostic view, and copies all 12,150 panel rows with P2's
# materialiser unchanged -- **every file re-hashed** against the committed
# `reports/p2_panel_sets/*.csv` -- then **requires both recorded digests**: 12,150 rows ->
# `2e1c1920…`, the 11,775 scoreable rows -> `bff608c3…`. A mismatch stops the session.
#
# **Accelerator: None (CPU)** -- no GPU quota. **Internet:** on (E3/E7/E8 from the Hub at
# P1's pinned revisions, E4 via `gdown`, session only). **Secret:** `HF_TOKEN` (optional).
# **Mode:** Save & Run All (Commit). **Account:** `muhammadadeelshah` only.
# **Publish the output's `dev_panel/` as the PRIVATE dataset `dbp-p2-panel`** (§6c).
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code, at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | the scoreable list `p2_panel_images.csv` and the manifests |
# | dataset `chameleon-dataset` | E5 (Chameleon), extracted, private |
# | dataset `coco-2017-dataset` (awsaf49) | COCO `val2017`: E7's reals |
# | dataset holding `p1_reals/` (slug `report`) | the T-ctrl reals |
# | dataset `thesis-dbp-1-dataset` | the T-ctrl reconstructions (pairs) |

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
    "reals": find("datasets/*/*/p1_reals"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
}
DBP = WORK / "Implementation" / "thesis_dbp"
PANEL = WORK / "dev_panel"
ROOTS = TMP / "roots.json"
CORPUS = TMP / "dev_panel_corpus"
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
assert (DBP / "scripts" / "p3_rebuild_dev_panel.py").is_file(), \
    "the code dataset predates this runbook"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_dev_panel_expect.json")).read_text())
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
assert py("p2_preflight.py", "--stage", "p3_dev_panel_data", *checks) == 0, \
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

# %% Cell 6 — stage the corpus (T-ctrl reals and their pairs), outside the output
assert py("p2_stage_corpus.py", "--root", CORPUS,
          "--link", f"reals={INPUTS['reals']}", "--expect", f"reals={EXPECT['corpus']['reals']}",
          "--link", f"pairs={INPUTS['pairs']}",
          "--expect", f"pairs={EXPECT['corpus']['pairs']}") == 0, "corpus staging FAILED"

# %% Cell 7 — copy every row, re-hash every file, require both digests
code = py("p3_rebuild_dev_panel.py", "--p2-panel-sets", P2_SETS, "--roots", ROOTS,
          "--corpus-root", CORPUS, "--panel-root", PANEL,
          "--scoreable-list", S2IN / "p2_panel_images.csv",
          "--expect-digest-all", EXPECT["digest_all"],
          "--expect-digest-scoreable", EXPECT["digest_scoreable"],
          "--cleanup-derived", DBP / "outputs" / "p1" / "eval")
print(f"rebuild exited {code} (0 = both digests match)")

# %% Cell 8 — report (cannot fail the run)
for name in ("p3_eval_views_rebuild.json", "p3_dev_panel_digest.json"):
    path = DBP / "reports" / name
    print(f"===== {name}\n" + (path.read_text() if path.is_file() else "(not written)"))

# %% Cell 9 — verdict: this cell fails the notebook unless the panel is byte-identical
RECORD = json.loads((DBP / "reports" / "p3_dev_panel_digest.json").read_text())
assert code == 0 and RECORD["ok"], f"DEV PANEL NOT VERIFIED: {RECORD['problems']}"
print("DEV PANEL REBUILT AND VERIFIED:", RECORD["rows_all"], "rows,",
      RECORD["rows_scoreable"], "scoreable,", round(RECORD["panel_bytes"] / 1e9, 2), "GB",
      "-- publish dev_panel/ as the PRIVATE dataset dbp-p2-panel")
