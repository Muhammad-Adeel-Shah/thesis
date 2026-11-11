# %% [markdown]
# # P3 Phase B step 3 — Open Images reals from the CVDF S3 copies (Kaggle CPU)
#
# Draws the 4,000 Open Images V7 real images P3 trains on (3,600 train + 400 held out).
# **Source (user decision 2026-10-15):** the CVDF copies in the public S3 bucket
# `open-images-dataset` (no credentials), "rescaled to have at most 1024 pixels on their
# longest side". Flickr refused every route to the originals (Kaggle IPs; a home IP at 4
# connections and at 1 request per 6 s), so the 2026-10-11 host choice is reversed; the
# rows, the seed-2026 walk, the attribution and every data rule are unchanged:
# shorter side >= 384, no resizing, native 384 crop to PNG (the COCO pipeline's own
# function), non-RGB modes and greyscale rejected, dedup (sha256 + pHash <= 6) against
# Panel B, the P2 panel, the training corpus, all of COCO and every E3/E4/E5 real, the
# first 4,000 in seed order, a seeded 3,600/400 split. `OriginalMD5` describes the Flickr
# original, so each file's sha256 is recorded instead. `df -h` at every stage.
#
# **Accelerator: None (CPU).** No GPU quota is used. **Internet:** on. **Mode:** Save & Run
# All (Commit). **Account:** `muhammadadeelshah` only (CLAUDE.md section 13).
#
# **Inputs to attach** (Cell 1 finds each by name):
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code, at the commit that holds this runbook |
# | dataset `dbp-p2-s2-inputs` | SD1.5 generation and bank-pair manifests (dedup references) |
# | dataset `dbp-p3-panel-b` | the Panel B session's output: the rebuilt E3/E4/E5 manifests |
# | dataset `coco-2017-dataset` (awsaf49) | COCO train2017 + val2017, hashed for the dedup |
#
# **Publish the output as a PRIVATE dataset** (`dbp-p3-open-images`): it holds the crops.

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


def find_file(base, name):
    """The one file called ``name`` anywhere under ``base``."""
    hits = sorted(p for p in base.rglob(name) if p.is_file())
    assert len(hits) == 1, f"expected exactly one {name} under {base}, found {hits}"
    return hits[0]


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "panel_b": find("datasets/*/dbp-p3-panel-b"),
    "coco": find("datasets/*/coco-2017-dataset/coco2017"),
}
DBP = WORK / "Implementation" / "thesis_dbp"
OUT = WORK / "oi"
COCO_INDEX = TMP / "p3_coco_index.csv"
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

# %% Cell 2 — stage the code; set up the environment (CPU session)
if DBP.exists():
    shutil.rmtree(DBP)
shutil.copytree(INPUTS["code"], DBP)
assert (DBP / "scripts" / "p3_fetch_open_images.py").is_file(), "the code predates step 3"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_open_images_expect.json")).read_text())
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: imports, input hashes, paths and disk, before anything long starts
S2IN = INPUTS["s2_inputs"]
REBUILD = json.loads(pathlib.Path(EXPECT.get("rebuild_record") or DBP /
                                  "reports/kaggle_20261011/p3_eval_views_rebuild.json").read_text())
MANIFESTS = {name: find_file(INPUTS["panel_b"], REBUILD[view]["manifest"])
             for name, view in EXPECT["eval_views"].items()}
METADATA = DBP / EXPECT["metadata"]["path"]
checks = ["--imports", ",".join(EXPECT["imports"]),
          "--require", f"coco_train={INPUTS['coco'] / 'train2017'}",
          "--require", f"coco_val={INPUTS['coco'] / 'val2017'}",
          "--sha256", f"{METADATA}={EXPECT['metadata']['sha256']}",
          "--rows", f"{METADATA}={EXPECT['metadata']['rows']}"]
for name, view in EXPECT["eval_views"].items():
    checks += ["--sha256", f"{MANIFESTS[name]}={REBUILD[view]['manifest_sha256']}",
               "--rows", f"{MANIFESTS[name]}={REBUILD[view]['rows']}"]
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}",
               "--rows", f"{S2IN / name}={spec['rows']}"]
assert py("p2_preflight.py", "--stage", "p3_open_images", *checks) == 0, \
    "PRE-FLIGHT FAILED -- nothing long has started"

# %% Cell 4 — hash all of COCO (sha256 + pHash), and count E3 reals found in COCO
coco = ["--coco-root", INPUTS["coco"], "--out", COCO_INDEX, "--e3-manifest", MANIFESTS["e3"]]
coco += [f"--expect={split}={n}" for split, n in EXPECT["coco_counts"].items()]
assert py("p3_coco_index.py", *coco) == 0, "COCO INDEX FAILED"

# %% Cell 5 — walk the seeded order: download, verify, crop, dedup; stop at the target
fetch = ["--metadata", METADATA, "--tmp", TMP / "oi_originals", "--out-root", OUT,
         "--coco-index", COCO_INDEX, "--target", EXPECT["target"],
         "--holdout", EXPECT["holdout"], "--workers", EXPECT["workers"],
         "--hosts", ",".join(EXPECT["hosts"]), "--source", EXPECT.get("source", "flickr"),
         "--min-free-tmp-gb", EXPECT["min_free_gb"]["tmp"],
         "--min-free-out-gb", EXPECT["min_free_gb"]["out"],
         "--gen-manifest", S2IN / "p1_gen_manifest.csv",
         "--pairs-manifest", S2IN / "p1_pairs_manifest.csv"]
fetch += [f"--eval-manifest={name}={path}" for name, path in MANIFESTS.items()]
fetch += ["--local-source", EXPECT["local_source"]] if EXPECT.get("local_source") else []
assert py("p3_fetch_open_images.py", *fetch) == 0, "OPEN IMAGES FETCH FAILED"

# %% Cell 6 — report (cannot fail the run)
for name in ("p3_coco_index_summary.json", "p3_open_images_summary.json"):
    path = DBP / "reports" / name
    print(f"===== {name}\n" + (path.read_text() if path.is_file() else "(not written)"))

# %% Cell 7 — verdict: this cell fails the notebook if the draw is incomplete
SUMMARY = json.loads((DBP / "reports" / "p3_open_images_summary.json").read_text())
assert SUMMARY["accepted"] == EXPECT["target"], f"accepted {SUMMARY['accepted']}"
print("OPEN IMAGES DRAWN:", SUMMARY["accepted"], "crops", SUMMARY["split"],
      "| downloaded", round(SUMMARY["downloaded_bytes"] / 1e9, 2), "GB",
      "| crops", round(SUMMARY["crop_bytes"] / 1e9, 2), "GB")
