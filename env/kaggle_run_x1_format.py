# %% [markdown]
# # X1 format control — the X1 images re-extracted, as-is and re-encoded as JPEG q95 4:4:4
#
# EXPLORATORY. Re-extracts frozen DINOv3-B (S1) features for every image the X1 probe
# uses (`reports/p2_x1/x1_format_control_images.csv`, 14,205 rows), twice, in one session
# on one device: once as-is and once after a JPEG round-trip at quality 95, chroma
# 4:4:4, applied to every full image before the centre crop (`p2_extract_s1.py
# --reencode-jpeg 95`). The probe is re-fitted locally on both (`x1_format_control.py
# analyse`). No P2 model is touched.
#
# **Accelerator:** GPU T4 x2 recommended (one GPU is used; the original S1 caches were
# extracted on a T4 with fp16 autocast, so the as-is arm reproduces them). CPU also works
# and uses no GPU quota, but has never been timed. **Internet:** on. **Secret:**
# `HF_TOKEN`. **Mode:** Save & Run All (Commit). **Account:** your own only (section 13).
#
# **Inputs to attach** (Cell 1 finds each by name):
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code, at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | the generation manifest `gen_matched` is rebuilt from |
# | notebook `phase-2` | the frozen panel (E7 rows) |
# | dataset holding `p1_reals/` and `p1_generated/` (slug `report`) | corpus |
# | dataset `thesis-dbp-1-dataset` | corpus pair reconstructions |
#
# Every cell is a thin call to a committed, tested script (CLAUDE.md section 12).

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
DEVICE = os.environ.get("DBP_DEVICE") or ("cuda" if shutil.which("nvidia-smi") else "cpu")


def find(pattern):
    """The one attached input matching ``pattern``, whichever account owns it."""
    hits = sorted(p for p in IN.glob(pattern) if p.is_dir())
    assert len(hits) == 1, f"expected exactly one input matching {pattern!r}, found {hits}"
    return hits[0]


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "panel": find("notebooks/*/phase-2/panel"),
    "reals": find("datasets/*/*/p1_reals"),
    "generated": find("datasets/*/*/p1_generated"),
    "pairs": find("datasets/*/thesis-dbp-1-dataset/Implementation/thesis_dbp/outputs/p1/pairs"),
}
DBP = WORK / "Implementation" / "thesis_dbp"
P1 = WORK / "p1"
OUT = DBP / "outputs" / "x1_format"
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
print("device:", DEVICE)

# %% Cell 2 — stage the code; set up the environment (the bootstrap is the only way)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "x1_format_control.py").is_file(), "the code dataset predates X1 control"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "x1_format_expect.json")).read_text())
IMAGES = pathlib.Path(EXPECT["images"])
IMAGES = IMAGES if IMAGES.is_absolute() else DBP / IMAGES
print("expected:", json.dumps(EXPECT["expected"]))
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — corpus and gen_matched; PRE-FLIGHT 1: token, imports, hashes, every path
links = []
for name in ("reals", "generated", "pairs"):
    links += ["--link", f"{name}={INPUTS[name]}", "--expect", f"{name}={EXPECT['corpus'][name]}"]
assert py("p2_stage_corpus.py", "--root", P1, *links) == 0, "corpus staging FAILED"
GEN = INPUTS["s2_inputs"] / EXPECT["gen_manifest"]["name"]
assert py("p1_match_generations.py", "--gen-manifest", GEN,
          "--reals-manifest", "reports/p1_reals_manifest_t_real.csv",
          "--gen-root", P1 / "generated", "--out-root", P1, "--set-id", "gen_matched") == 0, \
    "gen_matched regeneration FAILED"
checks = ["--imports", ",".join(EXPECT["imports"]),
          "--sha256", f"{IMAGES}={EXPECT['images_sha256']}",
          "--rows", f"{IMAGES}={EXPECT['images_rows']}",
          "--sha256", f"{GEN}={EXPECT['gen_manifest']['sha256']}",
          "--resolve", IMAGES]
checks += ["--hf-token"] if EXPECT["hf_token"] else []
checks += ["--expect-gpus", "1"] if DEVICE == "cuda" else []
assert py("p2_preflight.py", "--stage", "x1_format_data", *checks,
          "--root", P1, "--panel-root", INPUTS["panel"],
          "--gen-matched-reference", INPUTS["code"] / "reports" / "p1_gen_matched_manifest.csv",
          "--gen-matched-dir", P1 / "gen_matched",
          "--expect-gen-matched", EXPECT["gen_matched"]) == 0, \
    "PRE-FLIGHT (data) FAILED -- nothing long has started"

EXTRACT = ["--images", IMAGES, "--root", P1, "--panel-root", INPUTS["panel"],
           "--device", DEVICE, "--batch", EXPECT["batch"]]
ARMS = {"x1_asis": [], "x1_jpeg95": ["--reencode-jpeg", EXPECT["quality"]]}

# %% Cell 4 — PRE-FLIGHT 2: both arms on the first images, end to end
for tag, extra in ARMS.items():
    assert py("p2_extract_s1.py", *EXTRACT, *extra, "--limit", EXPECT["smoke_images"],
              "--tag", f"smoke_{tag}", "--out-dir", TMP / "x1_smoke") == 0, \
        f"PRE-FLIGHT (model) FAILED for {tag} -- nothing long has started"

# %% Cell 5 — extract both arms (a failed arm does not stop the other)
codes = {tag: py("p2_extract_s1.py", *EXTRACT, *extra, "--tag", tag, "--out-dir", OUT)
         for tag, extra in ARMS.items()}
print("exit codes:", codes)

# %% Cell 6 — report (cannot fail the run)
py("x1_format_control.py", "summarise", "--cache-dir", OUT)

# %% Cell 7 — verdict: this cell fails the notebook if an arm did not complete
assert all(code == 0 for code in codes.values()), f"an extraction arm failed: {codes}"
