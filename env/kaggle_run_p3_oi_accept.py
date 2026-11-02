# %% [markdown]
# # P3 Phase B step 3, route (ii) — dedup the local Open Images candidates (Kaggle CPU)
#
# Flickr refuses Kaggle's IPs, so the originals were fetched and cropped on the user's PC
# (`scripts/p3_oi_local_fetch.py`; user decision 2026-10-13, a §6a(a) deviation for this
# one source). This session does what needs the references: it re-hashes every uploaded
# crop against the committed candidate manifest, dedups each candidate (original and
# crop, sha256 and pHash ≤ 6) against all of COCO, Panel B, the P2 panel, the training
# corpus and every E3/E4/E5 real, accepts in the same seed order up to 4,000 and splits
# 3,600 / 400.
#
# **Accelerator: None (CPU).** **Internet:** on (bootstrap only). **Mode:** Save & Run All
# (Commit). **Account:** `muhammadadeelshah` only.
#
# | input | what |
# |---|---|
# | dataset `dbp-code*` | the code, with the committed candidate manifest |
# | dataset `dbp-p2-s2-inputs` | SD1.5 generation and bank-pair manifests |
# | dataset `dbp-p3-panel-b` | the rebuilt E3/E4/E5 manifests |
# | dataset `coco-2017-dataset` (awsaf49) | COCO train2017 + val2017 |
# | dataset `dbp-p3-oi-candidates` | the local stage's crops (`reals/t_oi/*.png`), PRIVATE |
#
# **Publish the output as a PRIVATE dataset** (`dbp-p3-open-images`).

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


def find_crops_root(base):
    """The one directory under ``base`` that holds ``reals/t_oi``."""
    hits = sorted(p.parent.parent for p in base.rglob("t_oi")
                  if p.is_dir() and p.parent.name == "reals")
    assert len(hits) == 1, f"expected exactly one reals/t_oi under {base}, found {hits}"
    return hits[0]


INPUTS = {
    "code": find("datasets/*/dbp-code*/thesis_dbp"),
    "s2_inputs": find("datasets/*/dbp-p2-s2-inputs"),
    "panel_b": find("datasets/*/dbp-p3-panel-b"),
    "coco": find("datasets/*/coco-2017-dataset/coco2017"),
    "candidates": find_crops_root(find("datasets/*/dbp-p3-oi-candidates")),
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
assert (DBP / "scripts" / "p3_oi_dedup_accept.py").is_file(), "the code predates route (ii)"
print("code commit:", (DBP / "COMMIT.txt").read_text().split()[0])
EXPECT = json.loads(pathlib.Path(
    os.environ.get("DBP_EXPECT", DBP / "env" / "p3_oi_accept_expect.json")).read_text())
print("expected:", json.dumps(EXPECT["expected"]))
os.environ["DBP_CPU_SESSION"] = "1"
assert sh("bash", DBP / "env" / "kaggle_bootstrap.sh") == 0, "bootstrap FAILED"

# %% Cell 3 — PRE-FLIGHT: imports, input hashes and paths, before anything long starts
S2IN = INPUTS["s2_inputs"]
REBUILD = json.loads(pathlib.Path(EXPECT.get("rebuild_record") or DBP /
                                  "reports/kaggle_20261011/p3_eval_views_rebuild.json").read_text())
MANIFESTS = {name: find_file(INPUTS["panel_b"], REBUILD[view]["manifest"])
             for name, view in EXPECT["eval_views"].items()}
CANDIDATES = DBP / EXPECT["candidates"]["path"]
checks = ["--imports", ",".join(EXPECT["imports"]),
          "--require", f"coco_train={INPUTS['coco'] / 'train2017'}",
          "--require", f"coco_val={INPUTS['coco'] / 'val2017'}",
          "--require", f"crops={INPUTS['candidates'] / 'reals' / 't_oi'}",
          "--sha256", f"{CANDIDATES}={EXPECT['candidates']['sha256']}",
          "--rows", f"{CANDIDATES}={EXPECT['candidates']['rows']}"]
for name, view in EXPECT["eval_views"].items():
    checks += ["--sha256", f"{MANIFESTS[name]}={REBUILD[view]['manifest_sha256']}",
               "--rows", f"{MANIFESTS[name]}={REBUILD[view]['rows']}"]
for name, spec in EXPECT["inputs"].items():
    checks += ["--sha256", f"{S2IN / name}={spec['sha256']}",
               "--rows", f"{S2IN / name}={spec['rows']}"]
assert py("p2_preflight.py", "--stage", "p3_oi_accept", *checks) == 0, \
    "PRE-FLIGHT FAILED -- nothing long has started"

# %% Cell 4 — hash all of COCO (sha256 + pHash)
coco = ["--coco-root", INPUTS["coco"], "--out", COCO_INDEX, "--e3-manifest", MANIFESTS["e3"]]
coco += [f"--expect={split}={n}" for split, n in EXPECT["coco_counts"].items()]
assert py("p3_coco_index.py", *coco) == 0, "COCO INDEX FAILED"

# %% Cell 5 — verify the crops, dedup in walk order, accept up to the target, split
accept = ["--candidates-manifest", CANDIDATES, "--candidates-root", INPUTS["candidates"],
          "--out-root", OUT, "--coco-index", COCO_INDEX, "--target", EXPECT["target"],
          "--holdout", EXPECT["holdout"],
          "--gen-manifest", S2IN / "p1_gen_manifest.csv",
          "--pairs-manifest", S2IN / "p1_pairs_manifest.csv"]
accept += [f"--eval-manifest={name}={path}" for name, path in MANIFESTS.items()]
assert py("p3_oi_dedup_accept.py", *accept) == 0, "ACCEPT FAILED (see the shortfall line)"

# %% Cell 6 — report (cannot fail the run)
for name in ("p3_coco_index_summary.json", "p3_oi_accept_summary.json"):
    path = DBP / "reports" / name
    print(f"===== {name}\n" + (path.read_text() if path.is_file() else "(not written)"))

# %% Cell 7 — verdict: this cell fails the notebook if fewer than the target were accepted
SUMMARY = json.loads((DBP / "reports" / "p3_oi_accept_summary.json").read_text())
assert SUMMARY["accepted"] == EXPECT["target"], f"accepted {SUMMARY['accepted']}"
print("OPEN IMAGES ACCEPTED:", SUMMARY["accepted"], SUMMARY["split"], "of",
      SUMMARY["candidates"], "candidates | dedup rejected", sum(SUMMARY["rejections"].values()),
      "| crops", round(SUMMARY["crop_bytes"] / 1e9, 2), "GB")
